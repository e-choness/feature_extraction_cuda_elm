#ifndef FEATURE_ELM_IO_IDX_DATASET_HPP_
#define FEATURE_ELM_IO_IDX_DATASET_HPP_

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace feature_elm {

/// Images and labels from an IDX pair (the MNIST / Fashion-MNIST file format).
struct IdxDataset {
  std::size_t numSamples = 0;
  std::size_t inputDim = 0;  ///< rows x cols of each image
  std::vector<float> data;   ///< row-major numSamples x inputDim, pixel / 255 (0..1)
  std::vector<int> labels;
};

struct IdxLoadResult {
  std::optional<IdxDataset> dataset;
  std::string error;
};

/// Loads uncompressed IDX files: a uint8 image file (magic 2051) and a uint8 label file (magic
/// 2049) with the same number of items. `scripts/fetch_datasets.py` downloads and decompresses
/// MNIST and Fashion-MNIST into this form.
[[nodiscard]] IdxLoadResult loadIdx(const std::filesystem::path& imagesPath,
                                    const std::filesystem::path& labelsPath);

}  // namespace feature_elm

#endif  // FEATURE_ELM_IO_IDX_DATASET_HPP_
