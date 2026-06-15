#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/elm.hpp"
#include "core/ml_elm.hpp"

namespace {

using feature_elm::ActivationFunction;
using feature_elm::Backend;
using feature_elm::BatchElm;
using feature_elm::MlElm;

double accuracyFromScores(const std::vector<double>& scores, const std::vector<double>& labels,
                          std::size_t numSamples, std::size_t numOutputs) {
  std::size_t correct = 0;
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    const std::size_t predicted = static_cast<std::size_t>(
        std::distance(scores.begin() + sample * numOutputs,
                      std::max_element(scores.begin() + sample * numOutputs,
                                       scores.begin() + (sample + 1) * numOutputs)));
    const std::size_t expected = static_cast<std::size_t>(
        std::distance(labels.begin() + sample * numOutputs,
                      std::max_element(labels.begin() + sample * numOutputs,
                                       labels.begin() + (sample + 1) * numOutputs)));
    if (predicted == expected) {
      ++correct;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(numSamples);
}

std::vector<double> xorData(std::size_t repeats, std::size_t* numSamples, std::size_t* numOutputs) {
  *numOutputs = 2;
  *numSamples = repeats * 4;
  std::vector<double> data(*numSamples * 2);
  std::vector<double> labels(*numSamples * 2);

  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::size_t i = 0; i < 4; ++i) {
      const std::size_t sample = repeat * 4 + i;
      const double x0 = (i & 1u) != 0 ? 1.0 : 0.0;
      const double x1 = (i & 2u) != 0 ? 1.0 : 0.0;
      data[sample * 2 + 0] = x0;
      data[sample * 2 + 1] = x1;
      labels[sample * 2 + (x0 == x1 ? 0 : 1)] = 1.0;
    }
  }

  return data;
}

TEST(MlElmCpuTest, StackedElmAeBeatsSingleLayerRandomElmOnXor) {
  std::size_t numSamples = 0;
  std::size_t numOutputs = 0;
  const std::vector<double> data = xorData(10, &numSamples, &numOutputs);
  std::vector<double> labels(numSamples * numOutputs);
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    const std::size_t caseIndex = sample % 4;
    const double label = caseIndex == 1 || caseIndex == 2 ? 1.0 : 0.0;
    labels[sample * numOutputs + 1] = label;
    labels[sample * numOutputs + 0] = 1.0 - label;
  }

  BatchElm<double> singleLayer(2, 2, ActivationFunction::kSigmoid, Backend::kCpu, 1e-6);
  ASSERT_TRUE(singleLayer.train(data, labels, numSamples, numOutputs));
  const auto singlePredictions = singleLayer.predictBatch(data, numSamples);
  ASSERT_TRUE(singlePredictions.has_value());
  const double singleAccuracy =
      accuracyFromScores(*singlePredictions, labels, numSamples, numOutputs);

  MlElm<double> mlElm(2, {6, 6}, ActivationFunction::kSigmoid, Backend::kCpu, 1e-6, 17u);
  ASSERT_TRUE(mlElm.train(data, labels, numSamples, numOutputs));
  EXPECT_TRUE(mlElm.isTrained());
  EXPECT_EQ(mlElm.finalFeatureDim(), 6u);
  const auto mlPredictions = mlElm.predictBatch(data, numSamples);
  ASSERT_TRUE(mlPredictions.has_value());
  const double mlAccuracy = accuracyFromScores(*mlPredictions, labels, numSamples, numOutputs);

  EXPECT_LE(singleAccuracy, 0.75);
  EXPECT_GT(mlAccuracy, 0.75);
  EXPECT_GT(mlAccuracy, singleAccuracy + 0.10);
}

TEST(MlElmCpuTest, PredictSingleSampleAfterTrain) {
  const std::size_t numSamples = 8;
  const std::size_t numOutputs = 1;
  std::vector<double> data(numSamples);
  std::vector<double> targets(numSamples);
  for (std::size_t i = 0; i < numSamples; ++i) {
    data[i] = static_cast<double>(i) / static_cast<double>(numSamples - 1);
    targets[i] = data[i] * data[i];
  }

  MlElm<double> model(1, {4, 4}, ActivationFunction::kRelu, Backend::kCpu, 1e-6, 31u);
  ASSERT_TRUE(model.train(data, targets, numSamples, numOutputs));

  const auto prediction = model.predict({0.5});
  ASSERT_TRUE(prediction.has_value());
  ASSERT_EQ(prediction->size(), 1u);
  EXPECT_TRUE(std::isfinite((*prediction)[0]));
}

TEST(MlElmCpuTest, OverflowedPredictionSampleCountReturnsNull) {
  const std::vector<double> data = {0.0, 0.5, 1.0};
  const std::vector<double> targets = {0.0, 0.25, 1.0};

  MlElm<double> model(1, {4}, ActivationFunction::kRelu, Backend::kCpu, 1e-6, 31u);
  ASSERT_TRUE(model.train(data, targets, 3, 1));

  EXPECT_FALSE(model.predictBatch(data, std::numeric_limits<std::size_t>::max()).has_value());
}

}  // namespace
