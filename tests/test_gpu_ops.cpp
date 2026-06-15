#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/elm_ae.hpp"
#include "core/feature_map.hpp"
#include "core/random_additive_map.hpp"
#include "core/solver.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

namespace {

using namespace feature_elm;

bool isGpuAvailable() {
#ifdef __CUDACC__
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  return deviceCount > 0;
#else
  return false;
#endif
}

// MSE helper for comparing results
template <typename FloatT>
FloatT computeMSE(const std::vector<FloatT>& predicted, const std::vector<FloatT>& ground_truth) {
  if (predicted.size() != ground_truth.size()) {
    return std::numeric_limits<FloatT>::max();
  }
  FloatT mse = FloatT(0);
  for (std::size_t i = 0; i < predicted.size(); ++i) {
    FloatT diff = predicted[i] - ground_truth[i];
    mse += diff * diff;
  }
  return mse / predicted.size();
}

// Test GPU ops skip cleanly without GPU (sanity check on stub behavior)
TEST(GpuOpsTest, GpuOpsFailGracefullyWithoutGpu) {
  // On a system without GPU, these should return false
  // On a system with GPU, we just verify the API works
  std::vector<double> dummyInput(100);
  std::vector<double> dummyWeights(80);
  std::vector<double> dummyBiases(16);
  std::vector<double> output;

  bool result = cuda_backend::transformRandomAdditiveGpu<double>(dummyInput, dummyInput.size() / 8,
                                                                 8, 16, dummyWeights, dummyBiases,
                                                                 ActivationKind::kTanh, &output);

  EXPECT_FALSE(result);
}

// Test GPU ridge solve returns gracefully without GPU
TEST(GpuOpsTest, RidgeSolveGpuFailsGracefully) {
  const std::size_t numSamples = 50;
  const std::size_t numFeatures = 16;
  const std::size_t numOutputs = 4;

  std::vector<double> features(numSamples * numFeatures);
  std::vector<double> targets(numSamples * numOutputs);
  std::vector<double> weights;

  bool result = cuda_backend::solveRidgeGpu<double>(features, targets, numSamples, numOutputs,
                                                    {1e-6}, &weights);
  EXPECT_FALSE(result);
}

// Test GPU ELM-AE transform returns gracefully without GPU
TEST(GpuOpsTest, ElmAeTransformGpuFailsGracefully) {
  std::vector<double> input(100);
  std::vector<double> output;

  ElmAutoEncoderLayer<double> ae(8, 16, ActivationKind::kTanh, 42u, 1e-6);
  // Not calling fit since GPU will fail anyway

  bool result = cuda_backend::transformElmAutoEncoderGpu<double>(
      input, input.size() / 8, 8, 16, ae.encoderWeights(), ae.encoderBiases(),
      ActivationKind::kTanh, &output);

  EXPECT_FALSE(result);
}

}  // namespace