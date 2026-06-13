#include <gtest/gtest.h>

#include "core/os_celm.hpp"

namespace {

using namespace feature_elm;

TEST(OsCelmTest, CpuConstrainedOnlineUpdateImprovesRegression) {
  std::size_t numInputs = 1;
  std::size_t numHidden = 20;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 20;

  OsCelm<double> osCelm(numInputs, numHidden, ActivationFunction::kSigmoid, 1e-2);

  std::vector<double> trainData;
  std::vector<double> trainTargets;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double x = -5.0 + 10.0 * static_cast<double>(i) / (numSamples - 1);
    trainData.push_back(x);
    trainTargets.push_back(3.0 * x - 2.0);
  }

  ASSERT_TRUE(osCelm.initialize(trainData, trainTargets, numSamples, numOutputs));

  auto predictions = osCelm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());
  EXPECT_EQ(predictions->size(), numSamples);

  double mse = 0.0;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double error = (*predictions)[i] - trainTargets[i];
    mse += error * error;
  }
  mse /= numSamples;
  EXPECT_LT(mse, 0.5);
}

}  // namespace
