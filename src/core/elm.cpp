#include "core/elm.hpp"
#include "cuda/elm_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace feature_elm {

template <typename FloatT>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
BatchElm<FloatT>::BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                           ActivationFunction activation, Backend backend)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      isTrained_(false),
      backend_(backend) {
  initializeHiddenLayer();
}

template <typename FloatT>
BatchElm<FloatT>::BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                           ActivationFunction activation, Backend backend,
                           const std::vector<FloatT>& hiddenWeights,
                           const std::vector<FloatT>& hiddenBiases)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      isTrained_(false),
      hiddenWeights_(hiddenWeights),
      hiddenBiases_(hiddenBiases),
      backend_(backend) {
  if (hiddenWeights_.size() != numInputs_ * numHiddenNodes_) {
    initializeHiddenLayer();
  }
  if (hiddenBiases_.size() != numHiddenNodes_) {
    initializeHiddenLayer();
  }
}

template <typename FloatT>
void BatchElm<FloatT>::initializeHiddenLayer() noexcept {
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
FloatT BatchElm<FloatT>::activate(FloatT x) const noexcept {
  if (activation_ == ActivationFunction::kSigmoid) {
    // Sigmoid: 1 / (1 + exp(-x))
    // Numerically stable version
    if (x > 0) {
      return static_cast<FloatT>(1.0) / (static_cast<FloatT>(1.0) + std::exp(-x));
    }
    FloatT expX = std::exp(x);
    return expX / (static_cast<FloatT>(1.0) + expX);
  }
  // RBF: exp(-x^2)
  return std::exp(-x * x);
}

template <typename FloatT>
std::vector<FloatT> BatchElm<FloatT>::computeHiddenOutput(const std::vector<FloatT>& input,
                                                          std::size_t numSamples) const {
  // NOLINTNEXTLINE(readability-identifier-naming)
  std::vector<FloatT> H(numSamples * numHiddenNodes_);

  for (std::size_t i = 0; i < numSamples; ++i) {
    for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
      // Compute: H[i,j] = activate(W[*,j]^T * x[i] + b[j])
      FloatT sum = hiddenBiases_[j];
      for (std::size_t k = 0; k < numInputs_; ++k) {
        sum += hiddenWeights_[k * numHiddenNodes_ + j] * input[i * numInputs_ + k];
      }
      // NOLINTNEXTLINE(readability-identifier-naming)
      H[i * numHiddenNodes_ + j] = activate(sum);
    }
  }
  return H;
}

template <typename FloatT>
std::vector<FloatT> BatchElm<FloatT>::solveLeastSquares(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const std::vector<FloatT>& H, const std::vector<FloatT>& T, std::size_t numSamples,
    std::size_t numOutputs) {
  // Solve: H·β = T for β using normal equations: β = (H^T·H)^(-1)·H^T·T

  // Compute H^T·H (numHiddenNodes × numHiddenNodes)
  // NOLINTNEXTLINE(readability-identifier-naming)
  std::vector<FloatT> HTH(numHiddenNodes_ * numHiddenNodes_, FloatT(0));
  for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
    for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
      FloatT sum = FloatT(0);
      for (std::size_t k = 0; k < numSamples; ++k) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        sum += H[k * numHiddenNodes_ + i] * H[k * numHiddenNodes_ + j];
      }
      // NOLINTNEXTLINE(readability-identifier-naming)
      HTH[i * numHiddenNodes_ + j] = sum;
    }
  }

  // Add regularization (small Tikhonov damping) for numerical stability
  FloatT lambda = static_cast<FloatT>(1e-8);
  for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
    // NOLINTNEXTLINE(readability-identifier-naming)
    HTH[i * numHiddenNodes_ + i] += lambda;
  }

  // Compute H^T·T (numHiddenNodes × numOutputs)
  // NOLINTNEXTLINE(readability-identifier-naming)
  std::vector<FloatT> HTT(numHiddenNodes_ * numOutputs, FloatT(0));
  for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
    for (std::size_t j = 0; j < numOutputs; ++j) {
      FloatT sum = FloatT(0);
      for (std::size_t k = 0; k < numSamples; ++k) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        sum += H[k * numHiddenNodes_ + i] * T[k * numOutputs + j];
      }
      // NOLINTNEXTLINE(readability-identifier-naming)
      HTT[i * numOutputs + j] = sum;
    }
  }

  // Gaussian elimination for solving HTH·β = HTT
  // NOLINTNEXTLINE(readability-identifier-naming)
  std::vector<FloatT> HTHcopy = HTH;
  // NOLINTNEXTLINE(readability-identifier-naming)
  std::vector<FloatT> HTTcopy = HTT;

  // Forward elimination
  for (std::size_t col = 0; col < numHiddenNodes_; ++col) {
    // Find pivot
    std::size_t pivotRow = col;
    FloatT maxVal = std::abs(HTHcopy[col * numHiddenNodes_ + col]);
    for (std::size_t row = col + 1; row < numHiddenNodes_; ++row) {
      FloatT val = std::abs(HTHcopy[row * numHiddenNodes_ + col]);
      if (val > maxVal) {
        maxVal = val;
        pivotRow = row;
      }
    }

    // Swap rows
    if (pivotRow != col) {
      for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
        std::swap(HTHcopy[col * numHiddenNodes_ + j], HTHcopy[pivotRow * numHiddenNodes_ + j]);
      }
      for (std::size_t j = 0; j < numOutputs; ++j) {
        std::swap(HTTcopy[col * numOutputs + j], HTTcopy[pivotRow * numOutputs + j]);
      }
    }

    // Check for singular matrix
    // NOLINTNEXTLINE(readability-identifier-naming)
    if (std::abs(HTHcopy[col * numHiddenNodes_ + col]) < static_cast<FloatT>(1e-15)) {
      return {};  // Singular matrix, return empty vector
    }

    // Eliminate column
    for (std::size_t row = col + 1; row < numHiddenNodes_; ++row) {
      // NOLINTNEXTLINE(readability-identifier-naming)
      FloatT factor = HTHcopy[row * numHiddenNodes_ + col] / HTHcopy[col * numHiddenNodes_ + col];
      for (std::size_t j = col; j < numHiddenNodes_; ++j) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        HTHcopy[row * numHiddenNodes_ + j] -= factor * HTHcopy[col * numHiddenNodes_ + j];
      }
      for (std::size_t j = 0; j < numOutputs; ++j) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        HTTcopy[row * numOutputs + j] -= factor * HTTcopy[col * numOutputs + j];
      }
    }
  }

  // Back substitution
  std::vector<FloatT> beta(numHiddenNodes_ * numOutputs);
  for (std::size_t row = numHiddenNodes_; row > 0; --row) {
    std::size_t i = row - 1;
    for (std::size_t j = 0; j < numOutputs; ++j) {
      // NOLINTNEXTLINE(readability-identifier-naming)
      FloatT sum = HTTcopy[i * numOutputs + j];
      for (std::size_t k = i + 1; k < numHiddenNodes_; ++k) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        sum -= HTHcopy[i * numHiddenNodes_ + k] * beta[k * numOutputs + j];
      }
      // NOLINTNEXTLINE(readability-identifier-naming)
      beta[i * numOutputs + j] = sum / HTHcopy[i * numHiddenNodes_ + i];
    }
  }

  return beta;
}

template <typename FloatT>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool BatchElm<FloatT>::train(const std::vector<FloatT>& trainData,
                             const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                             std::size_t numOutputs) {
  if (trainData.size() != numSamples * numInputs_) {
    return false;
  }
  if (trainTargets.size() != numSamples * numOutputs) {
    return false;
  }

  numOutputs_ = numOutputs;

  if (backend_ == Backend::kGpu) {
    std::vector<FloatT> gpuOutputWeights;
    if (!cuda_backend::trainBatchElmGpu(trainData, trainTargets, numSamples, numInputs_,
                                        numHiddenNodes_, numOutputs, hiddenWeights_,
                                        hiddenBiases_, activation_, &gpuOutputWeights)) {
      isTrained_ = false;
      return false;
    }
    outputWeights_ = std::move(gpuOutputWeights);
  } else {
    // Compute hidden layer output
    // NOLINTNEXTLINE(readability-identifier-naming)
    auto H = computeHiddenOutput(trainData, numSamples);

    // Solve least-squares for output weights
    // NOLINTNEXTLINE(readability-identifier-naming)
    outputWeights_ = solveLeastSquares(H, trainTargets, numSamples, numOutputs);
  }

  if (outputWeights_.empty()) {
    isTrained_ = false;
    return false;
  }

  isTrained_ = true;
  return true;
}

template <typename FloatT>
std::optional<std::vector<FloatT>> BatchElm<FloatT>::predict(
    const std::vector<FloatT>& input) const {
  if (!isTrained_) {
    return std::nullopt;
  }
  if (input.size() != numInputs_) {
    return std::nullopt;
  }

  if (backend_ == Backend::kGpu) {
    std::vector<FloatT> gpuPredictions;
    if (!cuda_backend::predictBatchElmGpu(input, 1, numInputs_, numHiddenNodes_, numOutputs_,
                                          hiddenWeights_, hiddenBiases_, outputWeights_,
                                          activation_, &gpuPredictions)) {
      return std::nullopt;
    }
    return gpuPredictions.empty() ? std::nullopt : std::optional<std::vector<FloatT>>(std::move(gpuPredictions));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto H = computeHiddenOutput(input, 1);

  std::vector<FloatT> output(numOutputs_);
  for (std::size_t i = 0; i < numOutputs_; ++i) {
    output[i] = FloatT(0);
    for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
      // NOLINTNEXTLINE(readability-identifier-naming)
      output[i] += outputWeights_[j * numOutputs_ + i] * H[j];
    }
  }

  return output;
}

template <typename FloatT>
std::optional<std::vector<FloatT>> BatchElm<FloatT>::predictBatch(
    const std::vector<FloatT>& testData, std::size_t numSamples) const {
  if (!isTrained_) {
    return std::nullopt;
  }
  if (testData.size() != numSamples * numInputs_) {
    return std::nullopt;
  }

  if (backend_ == Backend::kGpu) {
    std::vector<FloatT> gpuPredictions;
    if (!cuda_backend::predictBatchElmGpu(testData, numSamples, numInputs_, numHiddenNodes_,
                                          numOutputs_, hiddenWeights_, hiddenBiases_,
                                          outputWeights_, activation_, &gpuPredictions)) {
      return std::nullopt;
    }
    return gpuPredictions.empty() ? std::nullopt : std::optional<std::vector<FloatT>>(std::move(gpuPredictions));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto H = computeHiddenOutput(testData, numSamples);

  std::vector<FloatT> output(numSamples * numOutputs_);
  for (std::size_t i = 0; i < numSamples; ++i) {
    for (std::size_t j = 0; j < numOutputs_; ++j) {
      output[i * numOutputs_ + j] = FloatT(0);
      for (std::size_t k = 0; k < numHiddenNodes_; ++k) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        output[i * numOutputs_ + j] +=
            outputWeights_[k * numOutputs_ + j] * H[i * numHiddenNodes_ + k];
      }
    }
  }

  return output;
}

template <typename FloatT>
void BatchElm<FloatT>::reset() noexcept {
  outputWeights_.clear();
  numOutputs_ = 0;
  isTrained_ = false;
}

// Explicit template instantiations
template class BatchElm<float>;
template class BatchElm<double>;

}  // namespace feature_elm
