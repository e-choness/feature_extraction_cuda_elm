#include "io/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>

namespace feature_elm {

namespace {

std::vector<std::string> splitLine(const std::string& line, char delimiter = ',') {
  std::vector<std::string> tokens;
  std::istringstream iss(line);
  std::string token;
  while (std::getline(iss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

bool hasHeader(const std::vector<std::string>& tokens) {
  return std::all_of(tokens.begin(), tokens.end(), [](const std::string& token) {
    if (token.empty()) {
      return false;
    }

    const unsigned char first = static_cast<unsigned char>(token.front());
    if (std::isdigit(first) || token.front() == '-') {
      return std::none_of(token.begin(), token.end(),
                          [](unsigned char c) { return std::isdigit(c); });
    }

    return true;
  });
}

}  // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
DatasetLoadResult loadCsv(const std::filesystem::path& path, std::size_t inputDim,
                          std::size_t labelColumn, bool labelColumnFirst) {
  DatasetLoadResult result;

  if (!std::filesystem::exists(path)) {
    result.error = "File not found: " + path.string();
    return result;
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    result.error = "Failed to open file: " + path.string();
    return result;
  }

  Dataset dataset;
  std::string line;
  bool isHeader = true;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    auto tokens = splitLine(line);
    if (tokens.size() != inputDim + 1) {
      result.error = "Invalid row: expected " + std::to_string(inputDim + 1) + " columns, got " +
                     std::to_string(tokens.size());
      return result;
    }

    if (isHeader && hasHeader(tokens)) {
      isHeader = false;
      continue;
    }
    isHeader = false;

    if (dataset.numSamples == 0) {
      dataset.inputDim = inputDim;
    }

    (void)labelColumnFirst;
    const std::size_t labelIndex = labelColumn;
    for (std::size_t col = 0; col < tokens.size(); ++col) {
      if (col == labelIndex) {
        try {
          dataset.labels.push_back(std::stoi(tokens[col]));
        } catch (...) {
          result.error = "Failed to parse label at column " + std::to_string(col);
          return result;
        }
      } else {
        try {
          dataset.data.push_back(std::stod(tokens[col]));
        } catch (...) {
          result.error = "Failed to parse value at column " + std::to_string(col);
          return result;
        }
      }
    }
    ++dataset.numSamples;
  }

  if (dataset.numSamples == 0) {
    result.error = "Dataset contains no data rows";
    return result;
  }

  if (dataset.data.size() != dataset.numSamples * dataset.inputDim) {
    result.error = "Data size mismatch after loading";
    return result;
  }

  result.dataset = std::move(dataset);
  return result;
}

}  // namespace feature_elm
