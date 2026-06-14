#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "core/solver.hpp"

namespace {

using namespace feature_elm;

template <typename FloatT>
FloatT vectorNorm(const std::vector<FloatT>& values) {
  FloatT sum = FloatT(0);
  for (FloatT value : values) {
    sum += value * value;
  }
  return std::sqrt(sum);
}

TEST(SolverTest, BatchRidgeSolverRecoversKnownWeights) {
  const std::vector<double> features = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
  };
  const std::vector<double> targets = {
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
  };
  const std::vector<double> expected = {
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
  };

  BatchRidgeSolver<double> solver(
      SolverOptions<double>{1e-12, RidgeSolvePath::kPrimal, RidgeSolveMethod::kCholesky});
  std::vector<double> weights;

  ASSERT_TRUE(solver.solve(features, 3, targets, 2, &weights));
  ASSERT_EQ(weights.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(weights[i], expected[i], 1e-9);
  }
}

TEST(SolverTest, LargerRidgeAlphaShrinksWeightNorm) {
  const std::vector<double> features = {
      1.0, 1.0, 1.0, -1.0, 1.0, 1.1, 1.0, -0.9, -1.0, 1.0, -1.0, -1.0,
  };
  const std::vector<double> targets = {
      2.05, -0.05, 2.15, 0.05, -0.95, 0.15,
  };

  BatchRidgeSolver<double> smallAlpha(
      SolverOptions<double>{1e-12, RidgeSolvePath::kPrimal, RidgeSolveMethod::kCholesky});
  BatchRidgeSolver<double> largeAlpha(
      SolverOptions<double>{1e-1, RidgeSolvePath::kPrimal, RidgeSolveMethod::kCholesky});
  std::vector<double> smallWeights;
  std::vector<double> largeWeights;

  ASSERT_TRUE(smallAlpha.solve(features, 6, targets, 1, &smallWeights));
  ASSERT_TRUE(largeAlpha.solve(features, 6, targets, 1, &largeWeights));

  EXPECT_LT(vectorNorm(largeWeights), vectorNorm(smallWeights));
}

TEST(SolverTest, PrimalAndDualSolutionsAgree) {
  const std::vector<double> features = {
      0.2, -0.1, 0.5, 0.7, 0.3, 0.1, -0.4, 0.6, 0.2, 0.9, -0.2, 0.4, 0.1, 0.8, -0.3,
  };
  const std::vector<double> targets = {
      1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
  };

  BatchRidgeSolver<double> primal(
      SolverOptions<double>{1e-5, RidgeSolvePath::kPrimal, RidgeSolveMethod::kCholesky});
  BatchRidgeSolver<double> dual(
      SolverOptions<double>{1e-5, RidgeSolvePath::kDual, RidgeSolveMethod::kCholesky});
  std::vector<double> primalWeights;
  std::vector<double> dualWeights;

  ASSERT_TRUE(primal.solve(features, 5, targets, 2, &primalWeights));
  ASSERT_TRUE(dual.solve(features, 5, targets, 2, &dualWeights));
  ASSERT_EQ(primalWeights.size(), dualWeights.size());

  for (std::size_t i = 0; i < primalWeights.size(); ++i) {
    EXPECT_NEAR(primalWeights[i], dualWeights[i], 1e-8);
  }
}

TEST(SolverTest, CholeskySucceedsOnIllConditionedRegularizedSystem) {
  std::vector<double> features(100);
  std::vector<double> targets(20);
  for (std::size_t sample = 0; sample < 20; ++sample) {
    for (std::size_t feature = 0; feature < 5; ++feature) {
      features[sample * 5 + feature] = 1.0;
    }
    targets[sample] = sample % 2 == 0 ? 1.0 : -1.0;
  }

  BatchRidgeSolver<double> solver(
      SolverOptions<double>{1e-3, RidgeSolvePath::kAuto, RidgeSolveMethod::kCholesky});
  std::vector<double> weights;

  ASSERT_TRUE(solver.solve(features, 20, targets, 1, &weights));
  ASSERT_EQ(weights.size(), 5u);
  for (double weight : weights) {
    EXPECT_TRUE(std::isfinite(weight));
  }
}

TEST(SolverTest, HouseholderQrPathSolvesRegularizedLeastSquares) {
  const std::vector<double> features = {
      1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
  };
  const std::vector<double> targets = {
      1.0,
      2.0,
      3.0,
      4.0,
  };

  BatchRidgeSolver<double> cholesky(
      SolverOptions<double>{1e-6, RidgeSolvePath::kPrimal, RidgeSolveMethod::kCholesky});
  BatchRidgeSolver<double> qr(
      SolverOptions<double>{1e-6, RidgeSolvePath::kPrimal, RidgeSolveMethod::kHouseholderQr});
  std::vector<double> choleskyWeights;
  std::vector<double> qrWeights;

  ASSERT_TRUE(cholesky.solve(features, 4, targets, 1, &choleskyWeights));
  ASSERT_TRUE(qr.solve(features, 4, targets, 1, &qrWeights));
  ASSERT_EQ(qrWeights.size(), choleskyWeights.size());

  for (std::size_t i = 0; i < qrWeights.size(); ++i) {
    EXPECT_NEAR(qrWeights[i], choleskyWeights[i], 1e-8);
  }
}

TEST(SolverTest, InvalidInputsFail) {
  BatchRidgeSolver<double> solver;
  std::vector<double> weights;

  EXPECT_FALSE(solver.solve({}, 0, {}, 0, &weights));
  EXPECT_FALSE(solver.solve({1.0, 2.0}, 2, {1.0}, 1, &weights));
}

}  // namespace
