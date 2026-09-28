#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "io/dataset.hpp"
#include "io/drift_stream.hpp"
#include "io/idx_dataset.hpp"
#include "io/preprocess.hpp"

namespace {

using feature_elm::DatasetLoadResult;
using feature_elm::DriftStream;
using feature_elm::DriftStreamSample;
using feature_elm::loadCsv;
using feature_elm::oneHotEncode;
using feature_elm::preprocessDataset;

TEST(DatasetIoTest, LoaderLoadsDigitsShapesAndLabels) {
  const std::filesystem::path csvPath = FEATURE_ELM_SOURCE_DIR "/data/datasets/digits_8x8.csv";

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
  const auto result = loadCsv(FEATURE_ELM_SOURCE_DIR "/nonexistent.csv", 10);
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

namespace {

// Writes a tiny IDX pair: `count` images of rows x cols, pixel value = (item + index) % 256.
void writeIdx(const std::filesystem::path& images, const std::filesystem::path& labels,
              std::uint32_t count, std::uint32_t rows, std::uint32_t cols,
              std::uint32_t labelCount) {
  auto be = [](std::ofstream& out, std::uint32_t v) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(v >> 24), static_cast<unsigned char>(v >> 16),
        static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v)};
    out.write(reinterpret_cast<const char*>(bytes), 4);
  };
  std::ofstream img(images, std::ios::binary);
  be(img, 2051);
  be(img, count);
  be(img, rows);
  be(img, cols);
  for (std::uint32_t i = 0; i < count; ++i) {
    for (std::uint32_t p = 0; p < rows * cols; ++p) {
      img.put(static_cast<char>((i + p) % 256));
    }
  }
  std::ofstream lab(labels, std::ios::binary);
  be(lab, 2049);
  be(lab, labelCount);
  for (std::uint32_t i = 0; i < labelCount; ++i) {
    lab.put(static_cast<char>(i % 10));
  }
}

class IdxTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto dir = std::filesystem::temp_directory_path();
    const std::string tag = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    images_ = dir / ("felm_idx_images_" + tag);
    labels_ = dir / ("felm_idx_labels_" + tag);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove(images_, ec);
    std::filesystem::remove(labels_, ec);
  }
  std::filesystem::path images_;
  std::filesystem::path labels_;
};

}  // namespace

TEST_F(IdxTest, LoadsImagesScaledToUnitRange) {
  writeIdx(images_, labels_, 3, 2, 4, 3);
  const auto result = feature_elm::loadIdx(images_, labels_);
  ASSERT_TRUE(result.dataset.has_value()) << result.error;
  EXPECT_EQ(result.dataset->numSamples, 3u);
  EXPECT_EQ(result.dataset->inputDim, 8u);
  EXPECT_EQ(result.dataset->labels, (std::vector<int>{0, 1, 2}));
  EXPECT_FLOAT_EQ(result.dataset->data[0], 0.0f);
  EXPECT_FLOAT_EQ(result.dataset->data[8 + 3], 4.0f / 255.0f);  // item 1, pixel 3
}

TEST_F(IdxTest, RejectsCompressedMismatchedAndTruncatedFiles) {
  writeIdx(images_, labels_, 3, 2, 4, 2);  // label count differs
  EXPECT_FALSE(feature_elm::loadIdx(images_, labels_).dataset.has_value());

  {  // gzip magic instead of IDX: the common mistake of forgetting to decompress
    std::ofstream img(images_, std::ios::binary | std::ios::trunc);
    img.write("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 16);
  }
  const auto gz = feature_elm::loadIdx(images_, labels_);
  EXPECT_FALSE(gz.dataset.has_value());
  EXPECT_NE(gz.error.find("gzip"), std::string::npos);

  writeIdx(images_, labels_, 3, 2, 4, 3);
  std::filesystem::resize_file(images_, 16 + 8);  // header + one image only
  EXPECT_FALSE(feature_elm::loadIdx(images_, labels_).dataset.has_value());

  EXPECT_FALSE(feature_elm::loadIdx(images_.string() + ".missing", labels_).dataset.has_value());
}
