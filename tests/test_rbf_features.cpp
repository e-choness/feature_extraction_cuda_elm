#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/rbf_features.hpp"

namespace {

using namespace feature_elm;

TEST(RbfFeaturesTest, ComputesRbfFeaturesForSimpleInput) {
  std::size_t numSamples = 2;
  std::size_t inputDim = 1;
  std::size_t numCenters = 2;

  RbfParameters<double> params;
  ASSERT_TRUE(initializeRbfCentersRandom(numCenters, inputDim, &params, 123));
  params.width = 0.5;

  std::vector<double> input = {0.0, 1.0};
  std::vector<double> output;
  ASSERT_TRUE(computeRbfFeatures(input, numSamples, params, &output));
  EXPECT_EQ(output.size(), numSamples * numCenters);
  for (double value : output) {
    EXPECT_GE(value, 0.0);
    EXPECT_LE(value, 1.0);
  }
}

TEST(RbfFeaturesTest, ComputesCorrectRbfValues) {
  std::size_t numSamples = 1;
  std::size_t inputDim = 1;
  std::size_t numCenters = 2;
  double width = 1.0;

  RbfParameters<double> params;
  params.inputDim = inputDim;
  params.numCenters = numCenters;
  params.centers = {0.0, 1.0};
  params.width = width;

  std::vector<double> input = {2.0};
  std::vector<double> output;
  ASSERT_TRUE(computeRbfFeatures(input, numSamples, params, &output));

  EXPECT_EQ(output.size(), 2);
  EXPECT_NEAR(output[0], std::exp(-4.0 / 2.0), 1e-10);
  EXPECT_NEAR(output[1], std::exp(-1.0 / 2.0), 1e-10);
}

TEST(RbfFeaturesTest, KmeansCentersAreSelectedFromData) {
  std::size_t numCenters = 3;
  std::size_t inputDim = 2;
  std::size_t numSamples = 100;

  std::vector<double> data;
  for (std::size_t i = 0; i < numSamples; ++i) {
    data.push_back(static_cast<double>(i % 10));
    data.push_back(static_cast<double>(i / 10));
  }

  RbfParameters<double> params;
  ASSERT_TRUE(initializeRbfCentersKMeans(numCenters, inputDim, data, numSamples, &params, 42));

  EXPECT_EQ(params.centers.size(), numCenters * inputDim);

  for (std::size_t c = 0; c < numCenters; ++c) {
    for (std::size_t d = 0; d < inputDim; ++d) {
      EXPECT_GE(params.centers[c * inputDim + d], 0.0);
      EXPECT_LE(params.centers[c * inputDim + d], 9.0);
    }
  }
}

TEST(RbfFeaturesTest, WidthAffectsFeatureScale) {
  const double widthA = 0.5;
  const double widthB = 2.0;
  const double inputVal = 2.0;  // distance 2 from center at 0

  RbfParameters<double> paramsA;
  paramsA.inputDim = 1;
  paramsA.numCenters = 1;
  paramsA.centers = {0.0};
  paramsA.width = widthA;

  RbfParameters<double> paramsB;
  paramsB.inputDim = 1;
  paramsB.numCenters = 1;
  paramsB.centers = {0.0};
  paramsB.width = widthB;

  std::vector<double> input = {inputVal};
  std::vector<double> outputA, outputB;

  ASSERT_TRUE(computeRbfFeatures(input, 1, paramsA, &outputA));
  ASSERT_TRUE(computeRbfFeatures(input, 1, paramsB, &outputB));

  EXPECT_LT(outputA[0], outputB[0])
      << "Narrower width should produce smaller RBF values at distance";
}

}  // namespace
