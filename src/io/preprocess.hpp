#ifndef FEATURE_ELM_IO_PREPROCESS_HPP_
#define FEATURE_ELM_IO_PREPROCESS_HPP_

#include <cstddef>
#include <vector>

namespace feature_elm {

struct PreprocessedData {
  std::vector<double> trainData;
  std::vector<double> trainOneHot;
  std::vector<double> testData;
  std::vector<double> testOneHot;
  std::vector<int> trainLabels;
  std::vector<int> testLabels;
  std::size_t numTrainSamples = 0;
  std::size_t numTestSamples = 0;
  std::size_t inputDim = 0;
  std::size_t numClasses = 0;
  std::vector<double> minValues;
  std::vector<double> maxValues;
};

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
PreprocessedData preprocessDataset(const std::vector<double>& data, const std::vector<int>& labels,
                                   std::size_t numSamples, std::size_t inputDim,
                                   double trainFraction = 0.8, unsigned int seed = 42);

std::vector<double> minMaxNormalize(const std::vector<double>& data,
                                    const std::vector<double>& minValues,
                                    const std::vector<double>& maxValues);

std::vector<double> standardize(const std::vector<double>& data,
                                const std::vector<double>& meanValues,
                                const std::vector<double>& stdValues);

std::vector<double> oneHotEncode(const std::vector<int>& labels, std::size_t numClasses);

}  // namespace feature_elm

#endif  // FEATURE_ELM_IO_PREPROCESS_HPP_