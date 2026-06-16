#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/os_elm.hpp"
#include "core/rls_solver.hpp"
#include "cuda/elm_gpu.hpp"

namespace {

using namespace feature_elm;

std::vector<double> hiddenWeights() {
  return {0.75, -0.25, 0.5, 0.1, -0.4, 0.6, 0.2, -0.3};
}

std::vector<double> hiddenBiases() {
  return {0.1, -0.2, 0.3, -0.1};
}

double maxAbs(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

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

TEST(OsElmTest, ForgettingFactorOneMatchesPlainOsElm) {
  const std::size_t numInputs = 2;
  const std::size_t numHidden = 4;
  const std::size_t numOutputs = 1;
  const std::vector<double> data = {
      0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0,
  };
  const std::vector<double> targets = {0.0, 0.5, 1.0, 1.5, 2.0};
  const RlsOptions<double> plainOptions{1e-3, 1.0, RlsConstraint::kNone, 0.0};

  OsElm<double> plain(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kCpu,
                      hiddenWeights(), hiddenBiases());
  OsElm<double> explicitPlain(numInputs, numHidden, ActivationFunction::kSigmoid, Backend::kCpu,
                              hiddenWeights(), hiddenBiases(), plainOptions);

  ASSERT_TRUE(plain.initialize(data, targets, 5, numOutputs));
  ASSERT_TRUE(explicitPlain.initialize(data, targets, 5, numOutputs));
  ASSERT_TRUE(plain.update(data, targets, 5));
  ASSERT_TRUE(explicitPlain.update(data, targets, 5));

  const auto plainPredictions = plain.predictBatch(data, 5);
  const auto explicitPredictions = explicitPlain.predictBatch(data, 5);
  ASSERT_TRUE(plainPredictions.has_value());
  ASSERT_TRUE(explicitPredictions.has_value());
  ASSERT_EQ(plainPredictions->size(), explicitPredictions->size());
  for (std::size_t i = 0; i < plainPredictions->size(); ++i) {
    EXPECT_NEAR((*plainPredictions)[i], (*explicitPredictions)[i], 1e-12);
  }
}

TEST(RlsSolverTest, RegularizationStabilizesIllConditionedInitialization) {
  const std::vector<double> features = {
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  };
  const std::vector<double> targets = {1.0, 1.0, 1.0};

  RlsSolver<double> weak({1e-6, 1.0, RlsConstraint::kNone, 0.0});
  RlsSolver<double> regularized({1e-1, 1.0, RlsConstraint::kNone, 0.0});

  ASSERT_TRUE(weak.initialize(features, 3, targets, 1));
  ASSERT_TRUE(regularized.initialize(features, 3, targets, 1));

  for (double weight : regularized.weights()) {
    EXPECT_TRUE(std::isfinite(weight));
  }
  EXPECT_LT(maxAbs(regularized.covariance()), maxAbs(weak.covariance()));
}

TEST(RlsSolverTest, ForgettingFactorAdaptsToDriftFasterThanPlainOsElm) {
  std::vector<double> stableFeatures(40, 1.0);
  std::vector<double> stableTargets(40, 1.0);
  std::vector<double> driftedFeatures(20, 1.0);
  std::vector<double> driftedTargets(20, -1.0);

  RlsSolver<double> plain({1e-6, 1.0, RlsConstraint::kNone, 0.0});
  RlsSolver<double> forgetting({1e-6, 0.2, RlsConstraint::kNone, 0.0});

  ASSERT_TRUE(plain.initialize(stableFeatures, stableFeatures.size(), stableTargets, 1));
  ASSERT_TRUE(forgetting.initialize(stableFeatures, stableFeatures.size(), stableTargets, 1));
  ASSERT_TRUE(plain.update(driftedFeatures, driftedFeatures.size(), driftedTargets));
  ASSERT_TRUE(forgetting.update(driftedFeatures, driftedFeatures.size(), driftedTargets));

  const double plainError = std::abs(plain.weights()[0] - (-1.0));
  const double forgettingError = std::abs(forgetting.weights()[0] - (-1.0));
  EXPECT_LT(forgettingError, plainError);
  EXPECT_LT(forgettingError, 0.1);
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
