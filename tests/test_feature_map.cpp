#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/feature_map.hpp"
#include "core/identity_map.hpp"
#include "core/random_additive_map.hpp"

namespace {

using feature_elm::ActivationKind;
using feature_elm::FeatureMap;
using feature_elm::IdentityMap;
using feature_elm::RandomAdditiveMap;

static float sigmoid(float x) {
  return x > 0.0f ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
}

TEST(FeatureMapTest, AdditiveMapMatchesHandComputedSigmoid) {
  const std::vector<float> weights = {1.0f, 0.0f, -1.0f, 0.0f};
  const std::vector<float> biases = {0.0f, 0.5f};
  RandomAdditiveMap map(2, 2, ActivationKind::kSigmoid, 123u, weights, biases);
  const std::vector<float> input = {0.5f, -0.25f, 1.0f, 0.0f};
  std::vector<float> output;
  ASSERT_TRUE(map.transform(input, 2, &output));
  const std::vector<float> expected = {sigmoid(0.75f), sigmoid(0.5f), sigmoid(1.0f), sigmoid(0.5f)};
  ASSERT_EQ(output.size(), expected.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_FLOAT_EQ(output[i], expected[i]);
  }
}

TEST(FeatureMapTest, IdentityMapIsNoop) {
  IdentityMap map(3);
  const std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> output;
  ASSERT_TRUE(map.transform(input, 1, &output));
  EXPECT_EQ(output, input);
}

TEST(FeatureMapTest, SeedReproducibility) {
  RandomAdditiveMap first(4, 6, ActivationKind::kTanh, 77u);
  RandomAdditiveMap second(4, 6, ActivationKind::kTanh, 77u);
  std::vector<float> input(8, 0.25f);
  std::vector<float> outFirst;
  std::vector<float> outSecond;
  ASSERT_TRUE(first.transform(input, 2, &outFirst));
  ASSERT_TRUE(second.transform(input, 2, &outSecond));
  EXPECT_EQ(outFirst, outSecond);
}

}  // anonymous namespace