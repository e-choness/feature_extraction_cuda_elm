#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/elm_ae.hpp"
#include "core/feature_map.hpp"
#include "core/random_additive_map.hpp"
#include "core/solver.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

namespace {

using namespace feature_elm;

template <typename FloatT>
std::vector<FloatT> randomMatrix(std::size_t rows, std::size_t cols, unsigned int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<FloatT> dist(FloatT(-1), FloatT(1));
  std::vector<FloatT> values(rows * cols);
  for (auto& v : values) {
    v = dist(gen);
  }
  return values;
}

template <typename FloatT>
FloatT maxAbsDiff(const std::vector<FloatT>& a, const std::vector<FloatT>& b) {
  EXPECT_EQ(a.size(), b.size());
  FloatT worst = FloatT(0);
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

#define SKIP_WITHOUT_GPU()                                 \
  if (!cuda_backend::isGpuAvailable()) {                   \
    GTEST_SKIP() << "No CUDA device; parity test skipped"; \
  }

// Non-square shapes on purpose: a transposed layout bug cannot hide behind symmetric dimensions.
class AdditiveParity : public ::testing::TestWithParam<ActivationKind> {};

TEST_P(AdditiveParity, GpuTransformMatchesCpuDouble) {
  SKIP_WITHOUT_GPU();
  constexpr std::size_t kSamples = 37;
  constexpr std::size_t kInputs = 11;
  constexpr std::size_t kHidden = 23;
  const auto input = randomMatrix<double>(kSamples, kInputs, 1u);

  RandomAdditiveMap<double> cpu(kInputs, kHidden, GetParam(), 7u, Backend::kCpu);
  RandomAdditiveMap<double> gpu(kInputs, kHidden, GetParam(), 7u, Backend::kGpu);
  std::vector<double> cpuOut;
  std::vector<double> gpuOut;
  ASSERT_TRUE(cpu.transform(input, kSamples, &cpuOut));
  ASSERT_TRUE(gpu.transform(input, kSamples, &gpuOut));
  EXPECT_LT(maxAbsDiff(cpuOut, gpuOut), 1e-10);
}

TEST_P(AdditiveParity, GpuTransformMatchesCpuFloat) {
  SKIP_WITHOUT_GPU();
  constexpr std::size_t kSamples = 300;
  constexpr std::size_t kInputs = 64;
  constexpr std::size_t kHidden = 129;
  const auto input = randomMatrix<float>(kSamples, kInputs, 2u);

  RandomAdditiveMap<float> cpu(kInputs, kHidden, GetParam(), 9u, Backend::kCpu);
  RandomAdditiveMap<float> gpu(kInputs, kHidden, GetParam(), 9u, Backend::kGpu);
  std::vector<float> cpuOut;
  std::vector<float> gpuOut;
  ASSERT_TRUE(cpu.transform(input, kSamples, &cpuOut));
  ASSERT_TRUE(gpu.transform(input, kSamples, &gpuOut));
  EXPECT_LT(maxAbsDiff(cpuOut, gpuOut), 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(Activations, AdditiveParity,
                         ::testing::Values(ActivationKind::kSigmoid, ActivationKind::kTanh,
                                           ActivationKind::kRelu));

TEST(GpuOpsTest, ElmAutoEncoderTransformMatchesCpu) {
  SKIP_WITHOUT_GPU();
  constexpr std::size_t kSamples = 50;
  constexpr std::size_t kInputs = 12;
  constexpr std::size_t kHidden = 7;
  const auto data = randomMatrix<double>(kSamples, kInputs, 3u);

  ElmAutoEncoderLayer<double> ae(kInputs, kHidden, ActivationKind::kTanh, 42u, 1e-3);
  ASSERT_TRUE(ae.fit(data, kSamples));

  std::vector<double> cpuOut;
  std::vector<double> gpuOut;
  ASSERT_TRUE(ae.transform(data, kSamples, &cpuOut));
  ASSERT_TRUE(cuda_backend::transformElmAutoEncoderGpu<double>(
      data, kSamples, kInputs, kHidden, ae.encoderWeights(), ae.encoderBiases(),
      ActivationKind::kTanh, &gpuOut));
  EXPECT_LT(maxAbsDiff(cpuOut, gpuOut), 1e-10);
}

struct RidgeShape {
  std::size_t samples;
  std::size_t features;
  std::size_t outputs;
};

class RidgeParity : public ::testing::TestWithParam<RidgeShape> {};

// Overdetermined, underdetermined and square systems, all with several outputs so the
// right-hand-side leading dimension is exercised.
TEST_P(RidgeParity, GpuQrMatchesCpuCholesky) {
  SKIP_WITHOUT_GPU();
  const auto [samples, features, outputs] = GetParam();
  const auto h = randomMatrix<double>(samples, features, 4u);
  const auto t = randomMatrix<double>(samples, outputs, 5u);
  const double alpha = 1e-2;

  BatchRidgeSolver<double> cpuSolver({alpha});
  std::vector<double> cpuWeights;
  std::vector<double> gpuWeights;
  ASSERT_TRUE(cpuSolver.solve(h, samples, t, outputs, &cpuWeights));
  ASSERT_TRUE(cuda_backend::solveRidgeGpu<double>(h, t, samples, outputs, {alpha}, &gpuWeights));
  EXPECT_LT(maxAbsDiff(cpuWeights, gpuWeights), 1e-8);
}

INSTANTIATE_TEST_SUITE_P(Shapes, RidgeParity,
                         ::testing::Values(RidgeShape{200, 17, 3}, RidgeShape{9, 30, 4},
                                           RidgeShape{16, 16, 1}, RidgeShape{1000, 128, 10}),
                         [](const auto& info) {
                           return std::to_string(info.param.samples) + "x" +
                                  std::to_string(info.param.features) + "x" +
                                  std::to_string(info.param.outputs);
                         });

TEST(GpuOpsTest, RidgeSolveRejectsInvalidInput) {
  std::vector<double> weights;
  const std::vector<double> h(50 * 16, 0.5);
  const std::vector<double> t(50 * 4, 1.0);
  // Mismatched targets, non-positive alpha, null output: rejected with or without a GPU.
  EXPECT_FALSE(
      cuda_backend::solveRidgeGpu<double>(h, std::vector<double>(7), 50, 4, {1e-3}, &weights));
  EXPECT_FALSE(cuda_backend::solveRidgeGpu<double>(h, t, 50, 4, {0.0}, &weights));
  EXPECT_FALSE(cuda_backend::solveRidgeGpu<double>(h, t, 50, 4, {1e-3}, nullptr));
}

TEST(GpuOpsTest, TransformRejectsShapeMismatch) {
  std::vector<double> output;
  // 100 values cannot be 12 samples x 8 inputs.
  EXPECT_FALSE(cuda_backend::transformRandomAdditiveGpu<double>(
      std::vector<double>(100), 12, 8, 16, std::vector<double>(8 * 16), std::vector<double>(16),
      ActivationKind::kTanh, &output));
}

TEST(GpuOpsTest, DeviceNameReflectsAvailability) {
  EXPECT_EQ(cuda_backend::gpuDeviceName().empty(), !cuda_backend::isGpuAvailable());
}

}  // namespace
