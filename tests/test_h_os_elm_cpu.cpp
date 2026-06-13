#include <gtest/gtest.h>

#include "core/h_os_elm.hpp"

namespace {

using namespace feature_elm;

TEST(HierarchicalOsElmTest, CpuHierarchicalFeatureExtractionPredictsRegression) {
  std::size_t numInputs = 1;
  std::vector<std::size_t> hiddenNodesPerLayer = {10, 10};
  std::size_t numOutputs = 1;
  std::size_t numSamples = 30;

  HierarchicalOsElm<double> model(numInputs, hiddenNodesPerLayer, ActivationFunction::kSigmoid,
                                  Backend::kCpu);

  std::vector<double> trainData;
  std::vector<double> trainTargets;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double x = -3.0 + 6.0 * static_cast<double>(i) / (numSamples - 1);
    trainData.push_back(x);
    trainTargets.push_back(2.0 * x + 1.0);
  }

  ASSERT_TRUE(model.initialize(trainData, trainTargets, numSamples, numOutputs));
  auto predictions = model.predictBatch(trainData, numSamples);
  ASSERT_TRUE(predictions.has_value());

  double mse = 0.0;
  for (std::size_t i = 0; i < numSamples; ++i) {
    double error = (*predictions)[i] - trainTargets[i];
    mse += error * error;
  }
  mse /= numSamples;
  EXPECT_LT(mse, 1.0);
}

}  // namespace
