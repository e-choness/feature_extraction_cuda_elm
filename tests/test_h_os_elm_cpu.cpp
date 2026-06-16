#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "core/elm_ae.hpp"
#include "core/h_os_elm.hpp"
#include "core/rls_solver.hpp"

namespace {

using feature_elm::ActivationFunction;
using feature_elm::Backend;
using feature_elm::ElmAutoEncoderLayer;
using feature_elm::HierarchicalOsElm;
using feature_elm::RlsConstraint;
using feature_elm::RlsOptions;

double mse(const std::vector<double>& predictions, const std::vector<double>& targets) {
  double result = 0.0;
  for (std::size_t i = 0; i < predictions.size(); ++i) {
    const double error = predictions[i] - targets[i];
    result += error * error;
  }
  return result / static_cast<double>(predictions.size());
}

TEST(HierarchicalOsElmTest, CpuOnlineChunksMatchFullInitializationForIdentityStack) {
  const std::size_t numSamples = 30;
  std::vector<double> data(numSamples);
  std::vector<double> targets(numSamples);
  for (std::size_t i = 0; i < numSamples; ++i) {
    data[i] = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(numSamples - 1);
    targets[i] = 2.0 * data[i] + 1.0;
  }

  std::vector<double> firstData(data.begin(), data.begin() + 10);
  std::vector<double> firstTargets(targets.begin(), targets.begin() + 10);
  std::vector<double> secondData(data.begin() + 10, data.begin() + 20);
  std::vector<double> secondTargets(targets.begin() + 10, targets.begin() + 20);

  HierarchicalOsElm<double> chunked(1, {}, ActivationFunction::kSigmoid, Backend::kCpu,
                                    RlsOptions<double>{1e-3, 1.0, RlsConstraint::kNone, 0.0}, 1e-6,
                                    9u);
  HierarchicalOsElm<double> full(1, {}, ActivationFunction::kSigmoid, Backend::kCpu,
                                 RlsOptions<double>{1e-3, 1.0, RlsConstraint::kNone, 0.0}, 1e-6,
                                 9u);

  std::vector<double> thirdData(data.begin() + 20, data.end());
  std::vector<double> thirdTargets(targets.begin() + 20, targets.end());

  ASSERT_TRUE(chunked.initialize(firstData, firstTargets, 10, 1));
  ASSERT_TRUE(chunked.update(secondData, secondTargets, 10));
  ASSERT_TRUE(chunked.update(thirdData, thirdTargets, 10));
  ASSERT_TRUE(full.initialize(data, targets, numSamples, 1));

  const auto chunkedPredictions = chunked.predictBatch(data, numSamples);
  const auto fullPredictions = full.predictBatch(data, numSamples);
  ASSERT_TRUE(chunkedPredictions.has_value());
  ASSERT_TRUE(fullPredictions.has_value());
  ASSERT_EQ(chunkedPredictions->size(), fullPredictions->size());

  for (std::size_t i = 0; i < chunkedPredictions->size(); ++i) {
    EXPECT_NEAR((*chunkedPredictions)[i], (*fullPredictions)[i], 1e-5);
  }
}

TEST(HierarchicalOsElmTest, CpuFeatureStackIsLearnedElmAe) {
  const std::size_t numSamples = 24;
  std::vector<double> data(numSamples);
  std::vector<double> targets(numSamples);
  for (std::size_t i = 0; i < numSamples; ++i) {
    const double x = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(numSamples - 1);
    data[i] = x;
    targets[i] = x * x;
  }

  HierarchicalOsElm<double> model(1, {4}, ActivationFunction::kSigmoid, Backend::kCpu,
                                  RlsOptions<double>{1e-3, 1.0, RlsConstraint::kNone, 0.0}, 1e-6,
                                  23u);
  ASSERT_TRUE(model.initialize(data, targets, numSamples, 1));
  EXPECT_TRUE(model.featureStack().isFitted());

  const auto& layers = model.featureStack().layers();
  ASSERT_EQ(layers.size(), 1u);
  const auto* layer = dynamic_cast<const ElmAutoEncoderLayer<double>*>(layers[0].get());
  ASSERT_NE(layer, nullptr);
  EXPECT_FALSE(layer->encoderWeights().empty());
  EXPECT_FALSE(layer->outputWeights().empty());

  const auto predictions = model.predictBatch(data, numSamples);
  ASSERT_TRUE(predictions.has_value());
  EXPECT_LT(mse(*predictions, targets), 10.0);
}

TEST(HierarchicalOsElmTest, CpuResetClearsLearnedFeatureStack) {
  const std::vector<double> data = {0.0, 1.0, 2.0};
  const std::vector<double> targets = {0.0, 1.0, 4.0};

  HierarchicalOsElm<double> model(1, {4}, ActivationFunction::kSigmoid, Backend::kCpu,
                                  RlsOptions<double>{1e-3, 1.0, RlsConstraint::kNone, 0.0}, 1e-6,
                                  23u);
  ASSERT_TRUE(model.initialize(data, targets, 3, 1));
  model.reset();

  EXPECT_FALSE(model.isInitialized());
  EXPECT_FALSE(model.featureStack().isFitted());
}

TEST(HierarchicalOsElmTest, CpuOverflowedPredictionSampleCountReturnsNull) {
  HierarchicalOsElm<double> model(1, {}, ActivationFunction::kSigmoid, Backend::kCpu,
                                  RlsOptions<double>{1e-3, 1.0, RlsConstraint::kNone, 0.0}, 1e-6,
                                  9u);
  const std::vector<double> data = {0.0, 1.0, 2.0};
  const std::vector<double> targets = {0.0, 1.0, 2.0};
  ASSERT_TRUE(model.initialize(data, targets, 3, 1));

  EXPECT_FALSE(model.predictBatch(data, std::numeric_limits<std::size_t>::max()).has_value());
}

}  // namespace
