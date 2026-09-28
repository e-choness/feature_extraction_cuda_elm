#include "io/idx_dataset.hpp"

#include <array>
#include <cstdint>
#include <fstream>

namespace feature_elm {

namespace {

constexpr std::uint32_t kImageMagic = 2051;
constexpr std::uint32_t kLabelMagic = 2049;
constexpr std::uint32_t kMaxItems = 10'000'000;  // sanity bound against corrupt headers

bool readBigEndian(std::ifstream& in, std::uint32_t* value) {
  std::array<unsigned char, 4> bytes{};
  in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!in) {
    return false;
  }
  *value = (std::uint32_t{bytes[0]} << 24) | (std::uint32_t{bytes[1]} << 16) |
           (std::uint32_t{bytes[2]} << 8) | std::uint32_t{bytes[3]};
  return true;
}

}  // namespace

IdxLoadResult loadIdx(const std::filesystem::path& imagesPath,
                      const std::filesystem::path& labelsPath) {
  IdxLoadResult result;
  std::ifstream images(imagesPath, std::ios::binary);
  std::ifstream labels(labelsPath, std::ios::binary);
  if (!images || !labels) {
    result.error = "cannot open " + (!images ? imagesPath.string() : labelsPath.string());
    return result;
  }

  std::uint32_t imageMagic = 0;
  std::uint32_t count = 0;
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  std::uint32_t labelMagic = 0;
  std::uint32_t labelCount = 0;
  if (!readBigEndian(images, &imageMagic) || !readBigEndian(images, &count) ||
      !readBigEndian(images, &rows) || !readBigEndian(images, &cols) ||
      !readBigEndian(labels, &labelMagic) || !readBigEndian(labels, &labelCount)) {
    result.error = "truncated IDX header (are the files still gzip-compressed?)";
    return result;
  }
  if (imageMagic != kImageMagic || labelMagic != kLabelMagic) {
    result.error = "not an IDX uint8 image/label pair (are the files still gzip-compressed?)";
    return result;
  }
  if (count != labelCount || count == 0 || count > kMaxItems || rows == 0 || cols == 0 ||
      rows > 4096 || cols > 4096) {
    result.error = "IDX sizes are inconsistent";
    return result;
  }

  IdxDataset dataset;
  dataset.numSamples = count;
  dataset.inputDim = std::size_t{rows} * cols;
  std::vector<unsigned char> pixels(dataset.numSamples * dataset.inputDim);
  std::vector<unsigned char> rawLabels(dataset.numSamples);
  images.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  labels.read(reinterpret_cast<char*>(rawLabels.data()),
              static_cast<std::streamsize>(rawLabels.size()));
  if (!images || !labels) {
    result.error = "IDX payload is truncated";
    return result;
  }
  dataset.data.resize(pixels.size());
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    dataset.data[i] = static_cast<float>(pixels[i]) / 255.0f;
  }
  dataset.labels.assign(rawLabels.begin(), rawLabels.end());
  result.dataset = std::move(dataset);
  return result;
}

}  // namespace feature_elm
