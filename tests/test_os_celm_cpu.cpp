#include <gtest/gtest.h>

#include <vector>

#include "core/os_celm.hpp"
#include "core/os_elm.hpp"

namespace {

using namespace feature_elm;

std::vector<double> hiddenWeights() {
  return {0.75, -0.25, 0.5, 0.1, -0.4, 0.6, 0.2, -0.3};
}

std::vector<double> hiddenBiases() {
  return {0.1, -0.2, 0.3, -0.1};
}

TEST(OsCelmTest, ConstraintOffEqualsOsElm) {
  std::size_t numInputs = 2;
  std::size_t numHidden = 4;
  std::size_t numOutputs = 1;
  const RlsOptions<double> options{1e-3, 1.0, RlsConstraint::kNone, 0.0};
  const std::vector<double> data = {
      0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0,
  };
  const std::vector<double> targets = {0.0, 0.5, 1.0, 1.5, 2.0};

  OsElm<double> osElm(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kCpu,
                      hiddenWeights(), hiddenBiases(), options);
  OsCelm<double> osCelm(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kCpu,
                        hiddenWeights(), hiddenBiases(), 0.0, options);

  ASSERT_TRUE(osElm.initialize(data, targets, 5, numOutputs));
  ASSERT_TRUE(osCelm.initialize(data, targets, 5, numOutputs));
  ASSERT_TRUE(osElm.update(data, targets, 5));
  ASSERT_TRUE(osCelm.update(data, targets, 5));

  const auto elmPredictions = osElm.predictBatch(data, 5);
  const auto celmPredictions = osCelm.predictBatch(data, 5);
  ASSERT_TRUE(elmPredictions.has_value());
  ASSERT_TRUE(celmPredictions.has_value());
  ASSERT_EQ(elmPredictions->size(), celmPredictions->size());
  for (std::size_t i = 0; i < elmPredictions->size(); ++i) {
    EXPECT_NEAR((*elmPredictions)[i], (*celmPredictions)[i], 1e-12);
  }
}

TEST(OsCelmTest, CpuConstrainedOnlineUpdateUsesClassDistanceConstraint) {
  std::size_t numInputs = 1;
  std::size_t numHidden = 20;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 20;

  OsCelm<double> osCelm(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kCpu, 1e-2);

  std::vector<double> trainData;
  std::vector<double> trainTargets;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double x = -5.0 + 10.0 * static_cast<double>(i) / (numSamples - 1);
    trainData.push_back(x);
    trainTargets.push_back(3.0 * x - 2.0);
  }

  ASSERT_TRUE(osCelm.initialize(trainData, trainTargets, numSamples, numOutputs));
  EXPECT_EQ(osCelm.rlsOptions().constraint, RlsConstraint::kClassDistance);

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
