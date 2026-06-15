#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "core/elm_ae.hpp"

namespace {

using feature_elm::ActivationKind;
using feature_elm::ElmAutoEncoderLayer;

template <typename FloatT>
FloatT meanSquaredError(const std::vector<FloatT>& actual, const std::vector<FloatT>& expected) {
  FloatT mse = FloatT(0);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const FloatT diff = actual[i] - expected[i];
    mse += diff * diff;
  }
  return mse / static_cast<FloatT>(actual.size());
}

TEST(ElmAutoEncoderLayerTest, LearnsEncoderAndReconstructsInput) {
  const std::size_t numSamples = 24;
  const std::size_t inputDim = 3;
  const std::size_t outputDim = 8;
  std::vector<double> data(numSamples * inputDim);

  for (std::size_t i = 0; i < numSamples; ++i) {
    const double t = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(numSamples - 1);
    data[i * inputDim + 0] = t;
    data[i * inputDim + 1] = 2.0 * t + 0.2 * std::sin(3.0 * t);
    data[i * inputDim + 2] = -0.5 * t + 0.1 * std::cos(2.0 * t);
  }

  ElmAutoEncoderLayer<double> layer(inputDim, outputDim, ActivationKind::kSigmoid, 13u, 1e-8);
  ASSERT_TRUE(layer.fit(data, numSamples));
  EXPECT_TRUE(layer.isFitted());
  EXPECT_EQ(layer.inputDim(), inputDim);
  EXPECT_EQ(layer.outputDim(), outputDim);
  EXPECT_EQ(layer.encoderWeights().size(), inputDim * outputDim);
  EXPECT_EQ(layer.outputWeights().size(), outputDim * inputDim);

  std::vector<double> reconstruction;
  ASSERT_TRUE(layer.reconstruct(data, numSamples, &reconstruction));
  const double learnedMse = meanSquaredError(reconstruction, data);
  EXPECT_LT(learnedMse, 0.05);

  std::vector<double> zeroReconstruction(data.size(), 0.0);
  const double randomBaselineMse = meanSquaredError(zeroReconstruction, data);
  EXPECT_LT(learnedMse, randomBaselineMse * 0.1);
}

TEST(ElmAutoEncoderLayerTest, TransformUsesLearnedEncoderWeights) {
  const std::size_t numSamples = 3;
  const std::size_t inputDim = 2;
  const std::size_t outputDim = 2;
  const std::vector<double> data = {
      0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
  };

  ElmAutoEncoderLayer<double> layer(inputDim, outputDim, ActivationKind::kRelu, 5u, 1e-6);
  ASSERT_TRUE(layer.fit(data, numSamples));

  std::vector<double> output;
  ASSERT_TRUE(layer.transform(data, numSamples, &output));

  std::vector<double> expected(numSamples * outputDim, 0.0);
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t hidden = 0; hidden < outputDim; ++hidden) {
      double sum = 0.0;
      for (std::size_t input = 0; input < inputDim; ++input) {
        sum += data[sample * inputDim + input] * layer.encoderWeights()[input * outputDim + hidden];
      }
      expected[sample * outputDim + hidden] = std::max(0.0, sum + layer.encoderBiases()[hidden]);
    }
  }

  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-12);
  }
}

TEST(ElmAutoEncoderLayerTest, SameSeedIsReproducible) {
  const std::size_t numSamples = 6;
  const std::size_t inputDim = 2;
  const std::vector<double> data = {
      0.0, 0.0, 0.2, 0.1, 0.4, 0.2, 0.6, 0.3, 0.8, 0.4, 1.0, 0.5,
  };

  ElmAutoEncoderLayer<double> first(inputDim, 4, ActivationKind::kTanh, 123u, 1e-6);
  ElmAutoEncoderLayer<double> second(inputDim, 4, ActivationKind::kTanh, 123u, 1e-6);
  ASSERT_TRUE(first.fit(data, numSamples));
  ASSERT_TRUE(second.fit(data, numSamples));

  std::vector<double> firstOutput;
  std::vector<double> secondOutput;
  ASSERT_TRUE(first.transform(data, numSamples, &firstOutput));
  ASSERT_TRUE(second.transform(data, numSamples, &secondOutput));
  EXPECT_EQ(firstOutput, secondOutput);
}

TEST(ElmAutoEncoderLayerTest, OverflowedSampleCountFailsWithoutAllocation) {
  ElmAutoEncoderLayer<double> layer(1, 1, ActivationKind::kSigmoid, 1u, 1e-6);
  const std::vector<double> data = {0.5};
  std::vector<double> output;

  EXPECT_FALSE(layer.fit(data, std::numeric_limits<std::size_t>::max()));
  EXPECT_FALSE(layer.transform(data, std::numeric_limits<std::size_t>::max(), &output));
}

}  // namespace
