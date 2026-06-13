#include <gtest/gtest.h>

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

}  // namespace
