#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "core/elm.hpp"

namespace {

using namespace feature_elm;

// Helper: Compute MSE between two vectors
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

// Test 1: Simple linear regression (y = 2x + 1)
TEST(ElmCpuTest, SimpleLinearRegression) {
  // Setup
  std::size_t numInputs = 1;
  std::size_t numHidden = 10;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 20;

  BatchElm<double> elm(numInputs, numHidden);

  // Generate training data: y = 2x + 1 + small noise
  std::vector<double> trainData;
  std::vector<double> trainTargets;

  for (int i = 0; i < static_cast<int>(numSamples); ++i) {
    double x = -5.0 + (10.0 / numSamples) * i;
    double y = 2.0 * x + 1.0;
    trainData.push_back(x);
    trainTargets.push_back(y);
  }

  // Train
  ASSERT_TRUE(elm.train(trainData, trainTargets, numSamples, numOutputs));
  EXPECT_TRUE(elm.isTrained());
  EXPECT_EQ(elm.numOutputs(), numOutputs);

  // Test on training data (should fit well)
  auto predictions = elm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());

  double mse = computeMSE(*predictions, trainTargets);
  EXPECT_LT(mse, 0.1) << "MSE on training data should be small; got " << mse;
}

// Test 2: Simple XOR-like classification
TEST(ElmCpuTest, XorClassification) {
  std::size_t numInputs = 2;
  std::size_t numHidden = 20;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 4;

  BatchElm<double> elm(numInputs, numHidden);

  // XOR problem: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
  std::vector<double> trainData = {
      0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
  };
  std::vector<double> trainTargets = {
      0.0,
      1.0,
      1.0,
      0.0,
  };

  ASSERT_TRUE(elm.train(trainData, trainTargets, numSamples, numOutputs));

  auto predictions = elm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());

  // Rough accuracy check: predictions should be close to targets
  double mse = computeMSE(*predictions, trainTargets);
  EXPECT_LT(mse, 0.3) << "XOR MSE should be reasonable; got " << mse;
}

// Test 3: Single sample prediction
TEST(ElmCpuTest, SingleSamplePrediction) {
  std::size_t numInputs = 2;
  std::size_t numHidden = 10;
  std::size_t numOutputs = 1;

  BatchElm<double> elm(numInputs, numHidden);

  // Dummy training
  std::vector<double> trainData = {0.0, 0.0, 1.0, 1.0};
  std::vector<double> trainTargets = {0.0, 2.0};

  ASSERT_TRUE(elm.train(trainData, trainTargets, 2, numOutputs));

  // Single sample prediction
  std::vector<double> testSample = {0.5, 0.5};
  auto pred = elm.predict(testSample);

  ASSERT_TRUE(pred.has_value());
  EXPECT_EQ(pred->size(), 1);
  EXPECT_TRUE(std::isfinite(pred->at(0)));
}

// Test 4: Not trained predictions should fail
TEST(ElmCpuTest, PredictBeforeTrainFails) {
  BatchElm<double> elm(2, 10);

  std::vector<double> input = {1.0, 2.0};
  auto pred = elm.predict(input);

  EXPECT_FALSE(pred.has_value());
}

// Test 5: Reset functionality
TEST(ElmCpuTest, ResetClearsModel) {
  std::size_t numInputs = 1;
  std::size_t numHidden = 5;
  std::size_t numOutputs = 1;

  BatchElm<double> elm(numInputs, numHidden);

  std::vector<double> trainData = {1.0, 2.0, 3.0};
  std::vector<double> trainTargets = {2.0, 4.0, 6.0};

  ASSERT_TRUE(elm.train(trainData, trainTargets, 3, numOutputs));
  EXPECT_TRUE(elm.isTrained());

  elm.reset();
  EXPECT_FALSE(elm.isTrained());
  EXPECT_EQ(elm.numOutputs(), 0);

  // Should fail to predict after reset
  std::vector<double> input = {1.0};
  auto pred = elm.predict(input);
  EXPECT_FALSE(pred.has_value());
}

// Test 6: Multi-output regression
TEST(ElmCpuTest, MultiOutputRegression) {
  std::size_t numInputs = 1;
  std::size_t numHidden = 15;
  std::size_t numOutputs = 3;
  std::size_t numSamples = 10;

  BatchElm<double> elm(numInputs, numHidden);

  // Generate data: y1 = x, y2 = 2x, y3 = 3x
  std::vector<double> trainData;
  std::vector<double> trainTargets;

  for (int i = 0; i < static_cast<int>(numSamples); ++i) {
    double x = -5.0 + (10.0 / numSamples) * i;
    trainData.push_back(x);
    trainTargets.push_back(x);
    trainTargets.push_back(2 * x);
    trainTargets.push_back(3 * x);
  }

  ASSERT_TRUE(elm.train(trainData, trainTargets, numSamples, numOutputs));

  auto predictions = elm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());
  EXPECT_EQ(predictions->size(), numSamples * numOutputs);

  double mse = computeMSE(*predictions, trainTargets);
  EXPECT_LT(mse, 0.1);
}

// Test 7: Float vs Double precision
TEST(ElmCpuTest, FloatPrecision) {
  std::size_t numInputs = 1;
  std::size_t numHidden = 10;
  std::size_t numOutputs = 1;
  std::size_t numSamples = 10;

  BatchElm<float> elm(numInputs, numHidden);

  std::vector<float> trainData;
  std::vector<float> trainTargets;

  for (int i = 0; i < static_cast<int>(numSamples); ++i) {
    float x = -1.0f + (2.0f / numSamples) * i;
    trainData.push_back(x);
    trainTargets.push_back(x * x);
  }

  ASSERT_TRUE(elm.train(trainData, trainTargets, numSamples, numOutputs));

  auto predictions = elm.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());

  float mse = computeMSE(*predictions, trainTargets);
  EXPECT_LT(mse, 1.0f);
}

// FloatPrecision above draws its hidden layer from std::random_device, and ~3% of draws made the
// float32 normal equations too ill-conditioned for Cholesky, so training returned false at random
// (seen in CI). Sweeping seeded hidden layers makes that case deterministic; the solver now falls
// back to Householder QR instead of failing.
TEST(ElmCpuTest, FloatTrainingSurvivesIllConditionedHiddenLayers) {
  constexpr std::size_t kInputs = 1;
  constexpr std::size_t kHidden = 10;
  constexpr std::size_t kSamples = 10;
  std::vector<float> x;
  std::vector<float> t;
  for (std::size_t i = 0; i < kSamples; ++i) {
    const float v = -1.0f + (2.0f / kSamples) * static_cast<float>(i);
    x.push_back(v);
    t.push_back(v * v);
  }
  int failures = 0;
  for (unsigned int seed = 0; seed < 300; ++seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> weights(kInputs * kHidden);
    std::vector<float> biases(kHidden);
    for (auto& w : weights) {
      w = dist(gen);
    }
    for (auto& b : biases) {
      b = dist(gen);
    }
    BatchElm<float> elm(kInputs, kHidden, ActivationFunction::kSigmoid, Backend::kCpu, weights,
                        biases);
    if (!elm.train(x, t, kSamples, 1)) {
      ++failures;
      continue;
    }
    const auto predictions = elm.predictBatch(x, kSamples);
    ASSERT_TRUE(predictions.has_value());
    EXPECT_LT(computeMSE(*predictions, t), 1.0f) << "seed " << seed;
  }
  EXPECT_EQ(failures, 0);
}

}  // namespace
