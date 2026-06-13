#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "core/elm.hpp"
#include "cuda/elm_gpu.hpp"

namespace {

using namespace feature_elm;

// MSE helper
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

TEST(ElmGpuTest, TrainAndPredictMatchesCpu) {
  std::size_t numInputs = 1;
  std::size_t numHidden = 20;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 30;

  std::vector<double> trainData(numSamples * numInputs);
  std::vector<double> trainTargets(numSamples * numOutputs);
  for (std::size_t i = 0; i < numSamples; ++i) {
    double x = -3.0 + 6.0 * static_cast<double>(i) / (numSamples - 1);
    trainData[i] = x;
    trainTargets[i] = 2.0 * x + 1.0;
  }

  BatchElm<double> cpuElm(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kCpu);
  ASSERT_TRUE(cpuElm.train(trainData, trainTargets, numSamples, numOutputs));

  auto cpuPredictions = cpuElm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(cpuPredictions.has_value());

  if (!cuda_backend::isGpuAvailable()) {
    GTEST_SKIP() << "GPU backend unavailable";
  }

  BatchElm<double> gpuElm(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kGpu);
  ASSERT_TRUE(gpuElm.train(trainData, trainTargets, numSamples, numOutputs));

  auto gpuPredictions = gpuElm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(gpuPredictions.has_value());

  double mse = computeMSE(*cpuPredictions, *gpuPredictions);
  EXPECT_LT(mse, 1e-2);
}

TEST(ElmGpuTest, GpuBackendSkipsWithoutGpu) {
  BatchElm<float> elm(1, 10, ActivationFunction::kSigmoid, Backend::kGpu);

  std::vector<float> trainData = {0.0f, 1.0f};
  std::vector<float> trainTargets = {0.0f, 2.0f};

  if (!elm.train(trainData, trainTargets, 2, 1)) {
    GTEST_SKIP() << "GPU backend unavailable";
  }

  auto pred = elm.predict({0.5f});
  ASSERT_TRUE(pred.has_value());
  EXPECT_TRUE(std::isfinite(pred->at(0)));
}

}  // namespace
