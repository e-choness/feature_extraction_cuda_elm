#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/elm_ae.hpp"
#include "core/feature_map.hpp"
#include "core/random_additive_map.hpp"
#include "core/rls_solver.hpp"
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

// Block RLS on the GPU vs sequential rank-1 RLS on the CPU: identical in exact arithmetic when the
// forgetting factor is 1. Chunks include one larger than the GPU's 512-sample sub-block.
template <typename FloatT>
void checkRlsParity(FloatT weightTolerance) {
  constexpr std::size_t kFeatures = 48;
  constexpr std::size_t kOutputs = 3;
  RlsOptions<FloatT> options;
  options.regularization = FloatT(1e-2);
  RlsSolver<FloatT> cpu(options, Backend::kCpu);
  RlsSolver<FloatT> gpu(options, Backend::kGpu);

  unsigned int seed = 10;
  bool first = true;
  for (std::size_t chunk : {std::size_t{100}, std::size_t{7}, std::size_t{640}, std::size_t{1}}) {
    const auto h = randomMatrix<FloatT>(chunk, kFeatures, seed++);
    const auto t = randomMatrix<FloatT>(chunk, kOutputs, seed++);
    if (first) {
      ASSERT_TRUE(cpu.initialize(h, chunk, t, kOutputs));
      ASSERT_TRUE(gpu.initialize(h, chunk, t, kOutputs));
      first = false;
    } else {
      ASSERT_TRUE(cpu.update(h, chunk, t));
      ASSERT_TRUE(gpu.update(h, chunk, t));
    }
  }
  EXPECT_FALSE(cpu.usesGpu());
  EXPECT_TRUE(gpu.usesGpu());
  EXPECT_LT(maxAbsDiff(cpu.weights(), gpu.weights()), weightTolerance);
  EXPECT_LT(maxAbsDiff(cpu.covariance(), gpu.covariance()), weightTolerance);
}

TEST(GpuOpsTest, BlockRlsMatchesSequentialRlsDouble) {
  SKIP_WITHOUT_GPU();
  checkRlsParity<double>(1e-9);
}

TEST(GpuOpsTest, BlockRlsMatchesSequentialRlsFloat) {
  SKIP_WITHOUT_GPU();
  checkRlsParity<float>(2e-3f);
}

// Nearly rank-64 sigmoid features with a small ridge (the OS-ELM digits demo at 2048 hidden units).
// Accumulating A = reg*I + sum H^T H in float32 left it indefinite, the solve failed and OS-ELM
// then read an empty weight vector and crashed.
TEST(GpuOpsTest, BlockRlsSurvivesRankDeficientFeatures) {
  SKIP_WITHOUT_GPU();
  constexpr std::size_t kSamples = 600;
  constexpr std::size_t kInputs = 16;
  constexpr std::size_t kFeatures = 512;
  const auto x = randomMatrix<float>(kSamples, kInputs, 3u);
  RandomAdditiveMap<float> map(kInputs, kFeatures, ActivationKind::kSigmoid, 4u, Backend::kCpu);
  std::vector<float> h;
  ASSERT_TRUE(map.transform(x, kSamples, &h));
  const auto t = randomMatrix<float>(kSamples, 2, 5u);
  RlsOptions<float> options;
  options.regularization = 1e-2f;
  RlsSolver<float> cpu(options, Backend::kCpu);
  RlsSolver<float> gpu(options, Backend::kGpu);
  ASSERT_TRUE(cpu.initialize(h, kSamples, t, 2));
  ASSERT_TRUE(gpu.initialize(h, kSamples, t, 2));
  ASSERT_EQ(gpu.weights().size(), kFeatures * 2);
  // Predictions, not raw weights: the problem is ill-conditioned, so many weight vectors fit alike.
  std::vector<float> pc(kSamples * 2, 0.0f);
  std::vector<float> pg(kSamples * 2, 0.0f);
  for (std::size_t s = 0; s < kSamples; ++s) {
    for (std::size_t o = 0; o < 2; ++o) {
      for (std::size_t f = 0; f < kFeatures; ++f) {
        pc[s * 2 + o] += h[s * kFeatures + f] * cpu.weights()[f * 2 + o];
        pg[s * 2 + o] += h[s * kFeatures + f] * gpu.weights()[f * 2 + o];
      }
    }
  }
  EXPECT_LT(maxAbsDiff(pc, pg), 5e-2f);
}

TEST(GpuOpsTest, ForgettingAndConstraintRlsStayOnCpu) {
  SKIP_WITHOUT_GPU();
  const auto h = randomMatrix<double>(50, 8, 1u);
  const auto t = randomMatrix<double>(50, 2, 2u);
  RlsOptions<double> forgetting;
  forgetting.forgettingFactor = 0.98;
  RlsSolver<double> fos(forgetting, Backend::kGpu);
  ASSERT_TRUE(fos.initialize(h, 50, t, 2));
  EXPECT_FALSE(fos.usesGpu());

  RlsOptions<double> constrained;
  constrained.constraint = RlsConstraint::kClassDistance;
  constrained.constraintStrength = 0.1;
  RlsSolver<double> celm(constrained, Backend::kGpu);
  ASSERT_TRUE(celm.initialize(h, 50, t, 2));
  EXPECT_FALSE(celm.usesGpu());
}

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
