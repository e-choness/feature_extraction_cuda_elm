#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "core/elm.hpp"
#include "core/os_elm.hpp"

namespace {

using feature_elm::ActivationFunction;
using feature_elm::Backend;
using feature_elm::BatchElm;
using feature_elm::OsElm;

using FloatT = double;

std::vector<FloatT> fixedSeedInput(std::size_t numSamples, std::size_t numInputs,
                                   std::size_t seed) {
  std::mt19937 gen(static_cast<unsigned int>(seed));
  std::uniform_real_distribution<FloatT> dist(-1.0, 1.0);
  std::vector<FloatT> data(numSamples * numInputs);
  for (auto& v : data) {
    v = dist(gen);
  }
  return data;
}

std::vector<FloatT> fixedSeedTargets(std::size_t numSamples, std::size_t numOutputs,
                                     std::size_t seed) {
  std::mt19937 gen(static_cast<unsigned int>(seed + 1));
  std::uniform_real_distribution<FloatT> dist(0.0, 1.0);
  std::vector<FloatT> targets(numSamples * numOutputs);
  for (auto& v : targets) {
    v = dist(gen);
  }
  return targets;
}

std::vector<FloatT> fixedSeedWeights(std::size_t count, std::size_t seed) {
  std::mt19937 gen(static_cast<unsigned int>(seed));
  std::uniform_real_distribution<FloatT> dist(-0.5, 0.5);
  std::vector<FloatT> weights(count);
  for (auto& w : weights) {
    w = dist(gen);
  }
  return weights;
}

std::vector<FloatT> fixedSeedBiases(std::size_t count, std::size_t seed) {
  std::mt19937 gen(static_cast<unsigned int>(seed));
  std::uniform_real_distribution<FloatT> dist(-0.25, 0.25);
  std::vector<FloatT> biases(count);
  for (auto& b : biases) {
    b = dist(gen);
  }
  return biases;
}

[[nodiscard]] double computeMse(const std::vector<FloatT>& predicted,
                                const std::vector<FloatT>& groundTruth) {
  if (predicted.size() != groundTruth.size() || predicted.empty()) {
    return std::numeric_limits<double>::max();
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < predicted.size(); ++i) {
    double diff = static_cast<double>(predicted[i]) - static_cast<double>(groundTruth[i]);
    sum += diff * diff;
  }
  return sum / static_cast<double>(predicted.size());
}

class GoldenBaselineTest : public ::testing::Test {
 protected:
  static constexpr std::size_t kSeed = 42;
  static constexpr std::size_t kNumSamples = 25;
  static constexpr std::size_t kNumInputs = 4;
  static constexpr std::size_t kNumHidden = 16;
  static constexpr std::size_t kNumOutputs = 3;

  static constexpr double kBatchElmTrainMseThreshold = 0.25;
  static constexpr double kBatchElmPredictMseThreshold = 0.30;
  static constexpr double kOsElmTrainMseThreshold = 0.50;
  static constexpr double kOsElmPredictMseThreshold = 0.50;
};

TEST_F(GoldenBaselineTest, BatchElmTrainAndPredictMatchesReference) {
  const auto trainData = fixedSeedInput(kNumSamples, kNumInputs, kSeed);
  const auto trainTargets = fixedSeedTargets(kNumSamples, kNumOutputs, kSeed);
  const auto hiddenWeights = fixedSeedWeights(kNumInputs * kNumHidden, kSeed + 7);
  const auto hiddenBiases = fixedSeedBiases(kNumHidden, kSeed + 13);

  BatchElm<FloatT> model(kNumInputs, kNumHidden, ActivationFunction::kSigmoid, Backend::kCpu,
                         hiddenWeights, hiddenBiases);

  ASSERT_TRUE(model.train(trainData, trainTargets, kNumSamples, kNumOutputs));
  ASSERT_TRUE(model.isTrained());

  const auto predictions = model.predictBatch(trainData, kNumSamples);
  ASSERT_TRUE(predictions.has_value());
  ASSERT_EQ(predictions->size(), kNumSamples * kNumOutputs);

  const double mse = computeMse(*predictions, trainTargets);
  EXPECT_LT(mse, kBatchElmPredictMseThreshold)
      << "BatchElm prediction MSE regressed from baseline: " << mse;
}

TEST_F(GoldenBaselineTest, OsElmInitializeAndPredictMatchesReference) {
  const auto initialData = fixedSeedInput(kNumSamples, kNumInputs, kSeed);
  const auto initialTargets = fixedSeedTargets(kNumSamples, kNumOutputs, kSeed);

  OsElm<FloatT> model(kNumInputs, kNumHidden, ActivationFunction::kSigmoid, Backend::kCpu);

  ASSERT_TRUE(model.initialize(initialData, initialTargets, kNumSamples, kNumOutputs));
  ASSERT_TRUE(model.isInitialized());

  const auto predictions = model.predictBatch(initialData, kNumSamples);
  ASSERT_TRUE(predictions.has_value());
  ASSERT_EQ(predictions->size(), kNumSamples * kNumOutputs);

  const double mse = computeMse(*predictions, initialTargets);
  EXPECT_LT(mse, kOsElmPredictMseThreshold)
      << "OsElm prediction MSE regressed from baseline: " << mse;
}

TEST_F(GoldenBaselineTest, OsElmOnlineUpdateImprovesFit) {
  const auto initialData = fixedSeedInput(kNumSamples, kNumInputs, kSeed);
  const auto initialTargets = fixedSeedTargets(kNumSamples, kNumOutputs, kSeed);

  OsElm<FloatT> model(kNumInputs, kNumHidden, ActivationFunction::kSigmoid, Backend::kCpu);

  ASSERT_TRUE(model.initialize(initialData, initialTargets, kNumSamples, kNumOutputs));

  const auto updateData = fixedSeedInput(kNumSamples, kNumInputs, kSeed + 31);
  const auto updateTargets = fixedSeedTargets(kNumSamples, kNumOutputs, kSeed + 31);

  ASSERT_TRUE(model.update(updateData, updateTargets, kNumSamples));

  const auto predictions = model.predictBatch(updateData, kNumSamples);
  ASSERT_TRUE(predictions.has_value());

  const double mse = computeMse(*predictions, updateTargets);
  EXPECT_LT(mse, kOsElmPredictMseThreshold)
      << "OsElm update prediction MSE regressed from baseline: " << mse;
}

}  // namespace
