#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>

#include "io/dataset.hpp"
#include "io/drift_stream.hpp"
#include "io/preprocess.hpp"

namespace {

using feature_elm::DatasetLoadResult;
using feature_elm::DriftStream;
using feature_elm::DriftStreamSample;
using feature_elm::loadCsv;
using feature_elm::oneHotEncode;
using feature_elm::preprocessDataset;

TEST(DatasetIoTest, LoaderLoadsDigitsShapesAndLabels) {
  const std::filesystem::path csvPath = "/workspace/data/datasets/digits_8x8.csv";

  ASSERT_TRUE(std::filesystem::exists(csvPath));

  const auto result = loadCsv(csvPath, 64, 0, true);
  ASSERT_TRUE(result.dataset.has_value()) << result.error;
  const auto& dataset = result.dataset.value();

  EXPECT_EQ(dataset.inputDim, 64u);
  EXPECT_EQ(dataset.numSamples, 1797u);
  EXPECT_EQ(dataset.labels.size(), dataset.numSamples);
  EXPECT_EQ(dataset.data.size(), dataset.numSamples * dataset.inputDim);

  std::set<int> labels(dataset.labels.begin(), dataset.labels.end());
  EXPECT_EQ(labels.size(), 10u);
  EXPECT_EQ(*labels.begin(), 0);
  EXPECT_EQ(*labels.rbegin(), 9);
}

TEST(DatasetIoTest, LoaderRejectsMalformedRows) {
  const std::filesystem::path csvPath = "/tmp/feature_elm_malformed_digits.csv";
  {
    std::ofstream file(csvPath);
    file << "label,pixel0,pixel1\n";
    file << "1,0.5\n";
  }

  const auto result = loadCsv(csvPath, 2, 0, true);
  EXPECT_FALSE(result.dataset.has_value());
  EXPECT_FALSE(result.error.empty());
}

TEST(DatasetIoTest, LoaderMissingFileReturnsError) {
  const auto result = loadCsv("/workspace/nonexistent.csv", 10);
  EXPECT_FALSE(result.dataset.has_value());
  EXPECT_FALSE(result.error.empty());
}

TEST(PreprocessTest, TrainTestSplitIsDeterministic) {
  std::vector<double> data = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
  std::vector<int> labels = {0, 1, 0, 1};

  const auto result1 = preprocessDataset(data, labels, 4, 2, 0.5, 42);
  const auto result2 = preprocessDataset(data, labels, 4, 2, 0.5, 42);

  EXPECT_EQ(result1.trainLabels, result2.trainLabels);
  EXPECT_EQ(result1.testLabels, result2.testLabels);
}

TEST(PreprocessTest, TrainTestSplitIsDisjoint) {
  std::vector<double> data = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
  std::vector<int> labels = {0, 1, 2, 3};

  const auto result = preprocessDataset(data, labels, 4, 2, 0.5, 42);

  std::set<std::vector<double>> trainRows;
  for (std::size_t i = 0; i < result.numTrainSamples; ++i) {
    trainRows.insert({result.trainData[i * 2], result.trainData[i * 2 + 1]});
  }

  for (std::size_t i = 0; i < result.numTestSamples; ++i) {
    const std::vector<double> testRow = {result.testData[i * 2], result.testData[i * 2 + 1]};
    EXPECT_EQ(trainRows.count(testRow), 0u);
  }
}

TEST(PreprocessTest, MinMaxNormalizeUsesProvidedScale) {
  const std::vector<double> data = {0.0, 10.0, 10.0, 15.0};
  const auto normalized = feature_elm::minMaxNormalize(data, {0.0, 10.0}, {10.0, 20.0});

  EXPECT_DOUBLE_EQ(normalized[0], 0.0);
  EXPECT_DOUBLE_EQ(normalized[1], 0.0);
  EXPECT_DOUBLE_EQ(normalized[2], 1.0);
  EXPECT_DOUBLE_EQ(normalized[3], 0.5);
}

TEST(PreprocessTest, OneHotEncodeCorrect) {
  std::vector<int> labels = {0, 1, 2, 0, 1};
  const auto oneHot = oneHotEncode(labels, 3);

  EXPECT_EQ(oneHot.size(), 15u);

  EXPECT_DOUBLE_EQ(oneHot[0], 1.0);
  EXPECT_DOUBLE_EQ(oneHot[1], 0.0);
  EXPECT_DOUBLE_EQ(oneHot[2], 0.0);

  EXPECT_DOUBLE_EQ(oneHot[3], 0.0);
  EXPECT_DOUBLE_EQ(oneHot[4], 1.0);
  EXPECT_DOUBLE_EQ(oneHot[5], 0.0);
}

TEST(DriftStreamTest, LabelsFollowRotatingBoundaryBeforeAndAfterDrift) {
  DriftStream::Config config;
  config.inputDim = 4;
  config.numClasses = 2;
  config.streamLength = 200;
  config.driftPoint = 100;
  config.seed = 123;

  DriftStream stream(config);

  EXPECT_FALSE(stream.hasDriftOccurred());

  for (std::size_t i = 0; i < config.streamLength; ++i) {
    const auto sample = stream.next();
    ASSERT_TRUE(sample.has_value());

    const bool postDrift = i >= config.driftPoint;
    const double angle = postDrift ? M_PI / 2.0 : 0.0;
    const double projection =
        sample->input[0] * std::cos(angle) + sample->input[1] * std::sin(angle);
    const int expectedLabel = projection >= 0.0 ? 1 : 0;

    EXPECT_EQ(sample->label, expectedLabel);
  }

  EXPECT_TRUE(stream.hasDriftOccurred());
}

TEST(DriftStreamTest, RequiresTwoDimensionsForRotatingBoundary) {
  DriftStream::Config config;
  config.inputDim = 1;
  config.streamLength = 1;
  config.seed = 42;

  DriftStream stream(config);

  const auto sample = stream.next();
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->input.size(), 2u);
}

TEST(DriftStreamTest, StreamIsDeterministicAndResettable) {
  DriftStream::Config config;
  config.inputDim = 2;
  config.streamLength = 5;
  config.seed = 42;

  DriftStream first(config);
  DriftStream second(config);

  std::vector<DriftStreamSample> samples;
  for (std::size_t i = 0; i < config.streamLength; ++i) {
    const auto sample = first.next();
    ASSERT_TRUE(sample.has_value());
    samples.push_back(*sample);
    const auto duplicate = second.next();
    ASSERT_TRUE(duplicate.has_value());
    EXPECT_EQ(duplicate->input, sample->input);
    EXPECT_EQ(duplicate->label, sample->label);
  }

  EXPECT_FALSE(first.next().has_value());

  first.reset();
  for (const auto& expected : samples) {
    const auto actual = first.next();
    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->input, expected.input);
    EXPECT_EQ(actual->label, expected.label);
  }
}

}  // namespace
