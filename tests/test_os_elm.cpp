#include <gtest/gtest.h>

#include "core/os_elm.hpp"
#include "cuda/elm_gpu.hpp"

namespace {

using namespace feature_elm;

TEST(OsElmTest, CpuOnlineUpdateMatchesBatchRegression) {
  std::size_t numInputs = 1;
  std::size_t numHidden = 20;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 20;

  OsElm<double> osElm(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kCpu);

  std::vector<double> trainData;
  std::vector<double> trainTargets;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double x = -5.0 + 10.0 * static_cast<double>(i) / (numSamples - 1);
    trainData.push_back(x);
    trainTargets.push_back(2.0 * x + 1.0);
  }

  ASSERT_TRUE(osElm.initialize(trainData, trainTargets, numSamples, numOutputs));
  EXPECT_TRUE(osElm.isInitialized());

  auto predictions = osElm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());
  EXPECT_EQ(predictions->size(), numSamples * numOutputs);

  double mse = 0.0;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double error = (*predictions)[i] - trainTargets[i];
    mse += error * error;
  }
  mse /= numSamples;
  EXPECT_LT(mse, 0.5);
}

TEST(OsElmTest, GpuOsElmSkipsWithoutGpu) {
  if (!cuda_backend::isGpuAvailable()) {
    GTEST_SKIP() << "GPU backend unavailable";
  }
  std::size_t numInputs = 1;
  std::size_t numHidden = 20;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 20;

  OsElm<double> osElm(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kGpu);

  std::vector<double> trainData;
  std::vector<double> trainTargets;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double x = -5.0 + 10.0 * static_cast<double>(i) / (numSamples - 1);
    trainData.push_back(x);
    trainTargets.push_back(2.0 * x + 1.0);
  }

  ASSERT_TRUE(osElm.initialize(trainData, trainTargets, numSamples, numOutputs));
  auto predictions = osElm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());

  double mse = 0.0;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double error = (*predictions)[i] - trainTargets[i];
    mse += error * error;
  }
  mse /= numSamples;
  EXPECT_LT(mse, 0.5);
}

}  // namespace
