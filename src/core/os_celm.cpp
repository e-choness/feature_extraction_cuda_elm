#include "core/os_celm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace feature_elm {

namespace {

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename FloatT>
[[nodiscard]] bool multiplyMatrixTranspose(const std::vector<FloatT>& A,
                                           const std::vector<FloatT>& B, std::size_t rowsA,
                                           std::size_t colsA, std::size_t colsB,
                                           std::vector<FloatT>* result) {
  if (A.empty() || B.empty() || result == nullptr) {
    return false;
  }
  result->assign(colsA * colsB, FloatT(0));
  for (std::size_t i = 0; i < colsA; ++i) {
    for (std::size_t j = 0; j < colsB; ++j) {
      FloatT sum = FloatT(0);
      for (std::size_t k = 0; k < rowsA; ++k) {
        sum += A[k * colsA + i] * B[k * colsB + j];
      }
      (*result)[i * colsB + j] = sum;
    }
  }
  return true;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace

template <typename FloatT>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
OsCelm<FloatT>::OsCelm(std::size_t numInputs, std::size_t numHiddenNodes,
                       ActivationFunction activation, FloatT constraintStrength)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      constraintStrength_(constraintStrength),
      isInitialized_(false) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FloatT> dis(static_cast<FloatT>(-1), static_cast<FloatT>(1));
  hiddenWeights_.resize(numInputs_ * numHiddenNodes_);
  for (auto& w : hiddenWeights_) {
    w = dis(gen);
  }
  hiddenBiases_.resize(numHiddenNodes_);
  for (auto& b : hiddenBiases_) {
    b = dis(gen);
  }
}

template <typename FloatT>
[[nodiscard]] bool OsCelm<FloatT>::computeHiddenOutput(const std::vector<FloatT>& input,
                                                       std::size_t numSamples,
                                                       std::vector<FloatT>* hiddenOutput) const {
  if (input.size() != numSamples * numInputs_ || hiddenOutput == nullptr) {
    return false;
  }
  hiddenOutput->assign(numSamples * numHiddenNodes_, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t hiddenIndex = 0; hiddenIndex < numHiddenNodes_; ++hiddenIndex) {
      FloatT sum = hiddenBiases_[hiddenIndex];
      for (std::size_t inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
        sum += hiddenWeights_[inputIndex * numHiddenNodes_ + hiddenIndex] *
               input[sample * numInputs_ + inputIndex];
      }
      if (activation_ == ActivationFunction::kSigmoid) {
        (*hiddenOutput)[sample * numHiddenNodes_ + hiddenIndex] =
            static_cast<FloatT>(1) / (static_cast<FloatT>(1) + std::exp(-sum));
      } else {
        (*hiddenOutput)[sample * numHiddenNodes_ + hiddenIndex] = std::exp(-sum * sum);
      }
    }
  }
  return true;
}

template <typename FloatT>
[[nodiscard]] bool OsCelm<FloatT>::initialize(const std::vector<FloatT>& initialData,
                                              const std::vector<FloatT>& initialTargets,
                                              std::size_t numSamples, std::size_t numOutputs) {
  if (isInitialized_) {
    return false;
  }
  if (initialData.size() != numSamples * numInputs_ ||
      initialTargets.size() != numSamples * numOutputs) {
    return false;
  }

  numOutputs_ = numOutputs;
  std::vector<FloatT> hiddenOutput;
  if (!computeHiddenOutput(initialData, numSamples, &hiddenOutput)) {
    return false;
  }

  outputWeights_.assign(numHiddenNodes_ * numOutputs_, FloatT(0));
  covariance_.assign(numHiddenNodes_ * numHiddenNodes_, FloatT(0));

  FloatT lambda = static_cast<FloatT>(1e-3);
  for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
    covariance_[i * numHiddenNodes_ + i] = static_cast<FloatT>(1) / lambda;
  }

  std::vector<FloatT> normalMatrix;
  if (!multiplyMatrixTranspose(hiddenOutput, hiddenOutput, numSamples, numHiddenNodes_,
                               numHiddenNodes_, &normalMatrix)) {
    return false;
  }
  for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
    normalMatrix[i * numHiddenNodes_ + i] += lambda;
  }
  std::vector<FloatT> targetProjection;
  if (!multiplyMatrixTranspose(hiddenOutput, initialTargets, numSamples, numHiddenNodes_,
                               numOutputs_, &targetProjection)) {
    return false;
  }

  // Solve normalMatrix * beta = targetProjection with Gaussian elimination.
  for (std::size_t col = 0; col < numHiddenNodes_; ++col) {
    std::size_t pivot = col;
    FloatT maxVal = std::abs(normalMatrix[pivot * numHiddenNodes_ + col]);
    for (std::size_t row = col + 1; row < numHiddenNodes_; ++row) {
      FloatT val = std::abs(normalMatrix[row * numHiddenNodes_ + col]);
      if (val > maxVal) {
        pivot = row;
        maxVal = val;
      }
    }
    if (pivot != col) {
      for (std::size_t j = col; j < numHiddenNodes_; ++j) {
        std::swap(normalMatrix[col * numHiddenNodes_ + j],
                  normalMatrix[pivot * numHiddenNodes_ + j]);
      }
      for (std::size_t j = 0; j < numOutputs_; ++j) {
        std::swap(targetProjection[col * numOutputs_ + j],
                  targetProjection[pivot * numOutputs_ + j]);
      }
    }
    if (std::abs(normalMatrix[col * numHiddenNodes_ + col]) <
        std::numeric_limits<FloatT>::epsilon()) {
      return false;
    }
    FloatT invPivot = static_cast<FloatT>(1) / normalMatrix[col * numHiddenNodes_ + col];
    for (std::size_t row = col + 1; row < numHiddenNodes_; ++row) {
      FloatT factor = normalMatrix[row * numHiddenNodes_ + col] * invPivot;
      for (std::size_t k = col; k < numHiddenNodes_; ++k) {
        normalMatrix[row * numHiddenNodes_ + k] -= factor * normalMatrix[col * numHiddenNodes_ + k];
      }
      for (std::size_t k = 0; k < numOutputs_; ++k) {
        targetProjection[row * numOutputs_ + k] -= factor * targetProjection[col * numOutputs_ + k];
      }
    }
  }

  for (std::size_t row = numHiddenNodes_; row-- > 0;) {
    for (std::size_t out = 0; out < numOutputs_; ++out) {
      FloatT sum = targetProjection[row * numOutputs_ + out];
      for (std::size_t col = row + 1; col < numHiddenNodes_; ++col) {
        sum -= normalMatrix[row * numHiddenNodes_ + col] * outputWeights_[col * numOutputs_ + out];
      }
      outputWeights_[row * numOutputs_ + out] = sum / normalMatrix[row * numHiddenNodes_ + row];
    }
  }

  isInitialized_ = true;
  return true;
}

template <typename FloatT>
[[nodiscard]] bool OsCelm<FloatT>::update(const std::vector<FloatT>& newData,
                                          const std::vector<FloatT>& newTargets,
                                          std::size_t numSamples) {
  if (!isInitialized_) {
    return false;
  }
  if (newData.size() != numSamples * numInputs_ || newTargets.size() != numSamples * numOutputs_) {
    return false;
  }

  std::vector<FloatT> hiddenOutput;
  if (!computeHiddenOutput(newData, numSamples, &hiddenOutput)) {
    return false;
  }
  return updateRecursiveLeastSquares(hiddenOutput, newTargets, numSamples);
}

template <typename FloatT>
[[nodiscard]] bool OsCelm<FloatT>::updateRecursiveLeastSquares(
    const std::vector<FloatT>& hiddenOutput, const std::vector<FloatT>& targets,
    std::size_t numSamples) {
  if (hiddenOutput.size() != numSamples * numHiddenNodes_ ||
      targets.size() != numSamples * numOutputs_) {
    return false;
  }

  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    std::vector<FloatT> hiddenRow(numHiddenNodes_);
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      hiddenRow[i] = hiddenOutput[sample * numHiddenNodes_ + i];
    }

    std::vector<FloatT> projectedCovariance(numHiddenNodes_);
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      FloatT sum = FloatT(0);
      for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
        sum += covariance_[i * numHiddenNodes_ + j] * hiddenRow[j];
      }
      projectedCovariance[i] = sum;
    }

    FloatT denominator = static_cast<FloatT>(1);
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      denominator += hiddenRow[i] * projectedCovariance[i];
    }
    if (std::abs(denominator) < std::numeric_limits<FloatT>::epsilon()) {
      return false;
    }

    FloatT kFactor = static_cast<FloatT>(1) / denominator;
    std::vector<FloatT> gain(numHiddenNodes_);
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      gain[i] = projectedCovariance[i] * kFactor;
    }

    std::vector<FloatT> nextCovariance(covariance_.size());
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
        nextCovariance[i * numHiddenNodes_ + j] =
            covariance_[i * numHiddenNodes_ + j] - gain[i] * projectedCovariance[j];
      }
    }

    std::vector<FloatT> error(numOutputs_);
    for (std::size_t out = 0; out < numOutputs_; ++out) {
      FloatT prediction = FloatT(0);
      for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
        prediction += outputWeights_[i * numOutputs_ + out] * hiddenRow[i];
      }
      error[out] = targets[sample * numOutputs_ + out] - prediction;
    }

    for (std::size_t out = 0; out < numOutputs_; ++out) {
      for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
        outputWeights_[i * numOutputs_ + out] += gain[i] * error[out];
      }
    }

    covariance_.swap(nextCovariance);

    FloatT regularizer =
        computeClassDistance(hiddenOutput, targets, numSamples) * constraintStrength_;
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      covariance_[i * numHiddenNodes_ + i] += regularizer;
    }
  }

  return true;
}

template <typename FloatT>
[[nodiscard]] FloatT OsCelm<FloatT>::computeClassDistance(const std::vector<FloatT>& hiddenOutput,
                                                          const std::vector<FloatT>& targets,
                                                          std::size_t numSamples) const {
  if (hiddenOutput.empty() || targets.empty()) {
    return static_cast<FloatT>(0);
  }
  FloatT totalDistance = static_cast<FloatT>(0);
  for (std::size_t sample = 1; sample < numSamples; ++sample) {
    FloatT labelDistance =
        std::abs(targets[sample * numOutputs_] - targets[(sample - 1) * numOutputs_]);
    FloatT featureDistance = static_cast<FloatT>(0);
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      FloatT diff = hiddenOutput[sample * numHiddenNodes_ + i] -
                    hiddenOutput[(sample - 1) * numHiddenNodes_ + i];
      featureDistance += diff * diff;
    }
    totalDistance +=
        labelDistance * std::sqrt(featureDistance + std::numeric_limits<FloatT>::epsilon());
  }
  return totalDistance / static_cast<FloatT>(numSamples);
}

template <typename FloatT>
[[nodiscard]] std::optional<std::vector<FloatT>> OsCelm<FloatT>::predict(
    const std::vector<FloatT>& input) const {
  if (!isInitialized_) {
    return std::nullopt;
  }
  if (input.size() != numInputs_) {
    return std::nullopt;
  }

  std::vector<FloatT> hiddenOutput(numHiddenNodes_);
  for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
    FloatT sum = hiddenBiases_[j];
    for (std::size_t i = 0; i < numInputs_; ++i) {
      sum += hiddenWeights_[i * numHiddenNodes_ + j] * input[i];
    }
    if (activation_ == ActivationFunction::kSigmoid) {
      hiddenOutput[j] = static_cast<FloatT>(1) / (static_cast<FloatT>(1) + std::exp(-sum));
    } else {
      hiddenOutput[j] = std::exp(-sum * sum);
    }
  }

  std::vector<FloatT> output(numOutputs_, FloatT(0));
  for (std::size_t out = 0; out < numOutputs_; ++out) {
    for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
      output[out] += outputWeights_[j * numOutputs_ + out] * hiddenOutput[j];
    }
  }
  return output;
}

template <typename FloatT>
[[nodiscard]] std::optional<std::vector<FloatT>> OsCelm<FloatT>::predictBatch(
    const std::vector<FloatT>& testData, std::size_t numSamples) const {
  if (!isInitialized_) {
    return std::nullopt;
  }
  if (testData.size() != numSamples * numInputs_) {
    return std::nullopt;
  }

  std::vector<FloatT> hiddenOutput;
  if (!computeHiddenOutput(testData, numSamples, &hiddenOutput)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(numSamples * numOutputs_, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t out = 0; out < numOutputs_; ++out) {
      for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
        output[sample * numOutputs_ + out] +=
            outputWeights_[j * numOutputs_ + out] * hiddenOutput[sample * numHiddenNodes_ + j];
      }
    }
  }
  return output;
}

template <typename FloatT>
void OsCelm<FloatT>::reset() noexcept {
  outputWeights_.clear();
  covariance_.clear();
  numOutputs_ = 0;
  isInitialized_ = false;
}

template class OsCelm<float>;
template class OsCelm<double>;

}  // namespace feature_elm
