#include "io/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace feature_elm {

namespace {

void shuffleIndices(std::vector<std::size_t>* indices, unsigned int seed) {
  std::mt19937 rng(seed);
  std::shuffle(indices->begin(), indices->end(), rng);
}

std::size_t computeNumClasses(const std::vector<int>& labels) {
  const auto maxLabel = *std::max_element(labels.begin(), labels.end());
  return maxLabel >= 0 ? static_cast<std::size_t>(maxLabel) + 1u : 1u;
}

}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
PreprocessedData preprocessDataset(const std::vector<double>& data, const std::vector<int>& labels,
                                   std::size_t numSamples, std::size_t inputDim,
                                   double trainFraction, unsigned int seed) {
  PreprocessedData result;

  result.inputDim = inputDim;
  result.numClasses = computeNumClasses(labels);

  std::vector<std::size_t> indices(numSamples);
  std::iota(indices.begin(), indices.end(), 0);
  shuffleIndices(&indices, seed);

  const double trainCount = static_cast<double>(numSamples) * trainFraction;
  const std::size_t numTrain = static_cast<std::size_t>(trainCount);
  result.numTrainSamples = numTrain;
  result.numTestSamples = numSamples - numTrain;

  result.trainData.reserve(numTrain * inputDim);
  result.testData.reserve(result.numTestSamples * inputDim);
  result.trainLabels.reserve(numTrain);
  result.testLabels.reserve(result.numTestSamples);

  for (std::size_t i = 0; i < numTrain; ++i) {
    const std::size_t idx = indices[i];
    for (std::size_t d = 0; d < inputDim; ++d) {
      result.trainData.push_back(data[idx * inputDim + d]);
    }
    result.trainLabels.push_back(labels[idx]);
  }

  for (std::size_t i = numTrain; i < numSamples; ++i) {
    const std::size_t idx = indices[i];
    for (std::size_t d = 0; d < inputDim; ++d) {
      result.testData.push_back(data[idx * inputDim + d]);
    }
    result.testLabels.push_back(labels[idx]);
  }

  result.minValues.resize(inputDim);
  result.maxValues.resize(inputDim);
  for (std::size_t d = 0; d < inputDim; ++d) {
    double minVal = result.trainData[d];
    double maxVal = result.trainData[d];
    for (std::size_t s = 1; s < numTrain; ++s) {
      const double val = result.trainData[s * inputDim + d];
      minVal = std::min(minVal, val);
      maxVal = std::max(maxVal, val);
    }
    result.minValues[d] = minVal;
    result.maxValues[d] = maxVal;
  }

  result.trainData = minMaxNormalize(result.trainData, result.minValues, result.maxValues);
  result.testData = minMaxNormalize(result.testData, result.minValues, result.maxValues);

  result.trainOneHot = oneHotEncode(result.trainLabels, result.numClasses);
  result.testOneHot = oneHotEncode(result.testLabels, result.numClasses);

  return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

std::vector<double> minMaxNormalize(const std::vector<double>& data,
                                    const std::vector<double>& minValues,
                                    const std::vector<double>& maxValues) {
  const std::size_t inputDim = minValues.size();
  std::vector<double> result(data.size());

  for (std::size_t i = 0; i < data.size() / inputDim; ++i) {
    for (std::size_t d = 0; d < inputDim; ++d) {
      const double range = maxValues[d] - minValues[d];
      if (std::abs(range) > 1e-10) {
        result[i * inputDim + d] = (data[i * inputDim + d] - minValues[d]) / range;
      } else {
        result[i * inputDim + d] = 0.0;
      }
    }
  }
  return result;
}

std::vector<double> standardize(const std::vector<double>& data,
                                const std::vector<double>& meanValues,
                                const std::vector<double>& stdValues) {
  const std::size_t inputDim = meanValues.size();
  std::vector<double> result(data.size());

  for (std::size_t i = 0; i < data.size() / inputDim; ++i) {
    for (std::size_t d = 0; d < inputDim; ++d) {
      if (std::abs(stdValues[d]) > 1e-10) {
        result[i * inputDim + d] = (data[i * inputDim + d] - meanValues[d]) / stdValues[d];
      } else {
        result[i * inputDim + d] = 0.0;
      }
    }
  }
  return result;
}

std::vector<double> oneHotEncode(const std::vector<int>& labels, std::size_t numClasses) {
  std::vector<double> result(labels.size() * numClasses, 0.0);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int clampedLabel = std::clamp(labels[i], 0, static_cast<int>(numClasses) - 1);
    result[i * numClasses + clampedLabel] = 1.0;
  }
  return result;
}

}  // namespace feature_elm
