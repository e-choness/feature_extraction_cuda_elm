#ifndef FEATURE_ELM_IO_DATASET_HPP_
#define FEATURE_ELM_IO_DATASET_HPP_

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace feature_elm {

struct Dataset {
  std::size_t numSamples = 0;
  std::size_t inputDim = 0;
  std::vector<double> data;
  std::vector<int> labels;
};

struct DatasetLoadResult {
  std::optional<Dataset> dataset;
  std::string error;
};

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
DatasetLoadResult loadCsv(const std::filesystem::path& path, std::size_t inputDim,
                          std::size_t labelColumn = 0u, bool labelColumnFirst = false);

}  // namespace feature_elm

#endif  // FEATURE_ELM_IO_DATASET_HPP_