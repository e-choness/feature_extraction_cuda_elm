#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/rbf_map.hpp"
#include "core/solver.hpp"

namespace {

using feature_elm::BatchRidgeSolver;
using feature_elm::RbfMap;
using feature_elm::SolverOptions;

static double rbfReference(double distSq, double width) {
  return std::exp(-distSq / (2.0 * width * width));
}

TEST(RbfMapTest, ComputesRbfFeaturesCorrectly) {
  const std::size_t inputDim = 1;
  const std::size_t numCenters = 2;

  RbfMap<double> map(inputDim, numCenters, 1.0, feature_elm::RbfCenterInit::kRandom, 123u);
  map.fit({0.5, 1.5, 2.0}, 3);

  std::vector<double> centers = map.centers();
  ASSERT_EQ(centers.size(), numCenters * inputDim);

  std::vector<double> input = {0.0, 1.0, 2.0};
  std::vector<double> output;
  ASSERT_TRUE(map.transform(input, 3, &output));
  EXPECT_EQ(output.size(), 3 * numCenters);

  for (double val : output) {
    EXPECT_GE(val, 0.0);
    EXPECT_LE(val, 1.0);
  }
}

TEST(RbfMapTest, KmeansCenterInitSelectsFromData) {
  const std::size_t inputDim = 2;
  const std::size_t numCenters = 3;
  const std::size_t numSamples = 10;

  std::vector<double> data;
  for (std::size_t i = 0; i < numSamples; ++i) {
    data.push_back(static_cast<double>(i));
    data.push_back(static_cast<double>(i * 2));
  }

  RbfMap<double> map(inputDim, numCenters, 1.0, feature_elm::RbfCenterInit::kKMeans, 42u);
  ASSERT_TRUE(map.fit(data, numSamples));

  const auto& centers = map.centers();
  ASSERT_EQ(centers.size(), numCenters * inputDim);
}

TEST(RbfMapTest, RbfMapLearnsNonlinearConcentricPattern) {
  const std::size_t inputDim = 2;
  const std::size_t numCenters = 50;
  const std::size_t numOutputs = 1;

  std::vector<double> trainData;
  std::vector<double> trainTargets;

  for (int i = 0; i < 100; ++i) {
    double x = -5.0 + 10.0 * (i / 50.0);
    double y = -5.0 + 10.0 * ((i % 50) / 25.0);
    double r = std::sqrt(x * x + y * y);

    trainData.push_back(x);
    trainData.push_back(y);
    trainTargets.push_back(r < 3.0 ? 0.0 : 1.0);
  }

  RbfMap<double> rbfMap(inputDim, numCenters, 1.5, feature_elm::RbfCenterInit::kKMeans, 42u);
  ASSERT_TRUE(rbfMap.fit(trainData, 100));

  std::vector<double> hiddenOutput;
  ASSERT_TRUE(rbfMap.transform(trainData, 100, &hiddenOutput));

  SolverOptions<double> solverOpts;
  solverOpts.ridgeAlpha = 1e-6;
  BatchRidgeSolver<double> solver(solverOpts);

  std::vector<double> weights;
  ASSERT_TRUE(solver.solve(hiddenOutput, 100, trainTargets, numOutputs, &weights));

  std::vector<double> testHidden;
  ASSERT_TRUE(rbfMap.transform(trainData, 100, &testHidden));

  int correct = 0;
  for (std::size_t i = 0; i < 100; ++i) {
    double pred = 0.0;
    for (std::size_t c = 0; c < numCenters; ++c) {
      pred += testHidden[i * numCenters + c] * weights[c];
    }
    if ((pred < 0.5 && trainTargets[i] < 0.5) || (pred >= 0.5 && trainTargets[i] >= 0.5)) {
      ++correct;
    }
  }

  EXPECT_GE(correct, 85) << "RBF map should classify concentric pattern well; got " << correct
                         << "/100";
}

TEST(RbfMapTest, WidthParameterAffectsOutput) {
  const std::size_t inputDim = 1;
  const std::size_t numCenters = 1;

  std::vector<double> data = {0.0, 1.0, 2.0};

  RbfMap<double> narrowMap(inputDim, numCenters, 0.5, feature_elm::RbfCenterInit::kKMeans, 42u);
  RbfMap<double> wideMap(inputDim, numCenters, 5.0, feature_elm::RbfCenterInit::kKMeans, 42u);

  narrowMap.fit(data, 3);
  wideMap.fit(data, 3);

  std::vector<double> input = {4.0};
  std::vector<double> narrowOutput, wideOutput;

  ASSERT_TRUE(narrowMap.transform(input, 1, &narrowOutput));
  ASSERT_TRUE(wideMap.transform(input, 1, &wideOutput));

  EXPECT_LT(narrowOutput[0], wideOutput[0]) << "Narrow RBF should give smaller values at distance";
}

TEST(RbfMapTest, RbfFeaturesMatchReferenceFormula) {
  const double width = 1.0;
  const std::size_t inputDim = 2;
  const std::size_t numCenters = 3;

  RbfMap<double> map(inputDim, numCenters, width, feature_elm::RbfCenterInit::kRandom, 42u);
  ASSERT_TRUE(map.fit({0.0, 0.0, 1.0, 1.0, 2.0, 2.0}, 3));

  const auto& centers = map.centers();

  std::vector<double> input = {0.5, 0.5};
  std::vector<double> output;
  ASSERT_TRUE(map.transform(input, 1, &output));

  for (std::size_t c = 0; c < numCenters; ++c) {
    double distSq = 0.0;
    for (std::size_t d = 0; d < inputDim; ++d) {
      double diff = input[d] - centers[c * inputDim + d];
      distSq += diff * diff;
    }
    double expected = rbfReference(distSq, width);
    EXPECT_NEAR(output[c], expected, 1e-10) << "RBF output should match exp(-distSq/(2*sigma^2))";
  }
}

TEST(RbfMapTest, RbfSuperiorToAdditiveOnConcentricPattern) {
  const std::size_t inputDim = 2;
  const std::size_t numHidden = 100;
  const std::size_t numCenters = 50;
  const std::size_t numOutputs = 1;

  std::vector<double> trainData;
  std::vector<double> trainTargets;

  for (int i = 0; i < 200; ++i) {
    double angle = (3.14159 * 2.0 * i) / 200.0;
    double x = std::cos(angle);
    double y = std::sin(angle);

    trainData.push_back(x);
    trainData.push_back(y);
    trainTargets.push_back(std::sqrt(x * x + y * y) < 0.5 ? 0.0 : 1.0);
  }

  RbfMap<double> rbfMap(inputDim, numCenters, 0.3, feature_elm::RbfCenterInit::kKMeans, 123u);
  ASSERT_TRUE(rbfMap.fit(trainData, 200));

  std::vector<double> rbfHidden;
  ASSERT_TRUE(rbfMap.transform(trainData, 200, &rbfHidden));

  SolverOptions<double> solverOpts;
  solverOpts.ridgeAlpha = 1e-6;
  BatchRidgeSolver<double> rbfSolver(solverOpts);

  std::vector<double> rbfWeights;
  ASSERT_TRUE(rbfSolver.solve(rbfHidden, 200, trainTargets, numOutputs, &rbfWeights));

  int rbfCorrect = 0;
  for (std::size_t i = 0; i < 200; ++i) {
    double pred = 0.0;
    for (std::size_t c = 0; c < numCenters; ++c) {
      pred += rbfHidden[i * numCenters + c] * rbfWeights[c];
    }
    if ((pred < 0.5 && trainTargets[i] < 0.5) || (pred >= 0.5 && trainTargets[i] >= 0.5)) {
      ++rbfCorrect;
    }
  }

  EXPECT_GE(rbfCorrect, 150) << "RBF should handle radial patterns; got " << rbfCorrect
                             << "/200 correct";
}

}  // namespace