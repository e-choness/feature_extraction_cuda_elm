#include "core/elm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "cuda/elm_gpu.hpp"

namespace feature_elm {

template <typename FloatT>
BatchElm<FloatT>::BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                           ActivationFunction activation, Backend backend, FloatT ridgeAlpha)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      isTrained_(false),
      backend_(backend),
      ridgeAlpha_(ridgeAlpha),
      featureMap_(numInputs, numHiddenNodes, activationKind(activation)),
      solver_({ridgeAlpha}) {}

template <typename FloatT>
BatchElm<FloatT>::BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                           ActivationFunction activation, Backend backend,
                           const std::vector<FloatT>& hiddenWeights,
                           const std::vector<FloatT>& hiddenBiases, FloatT ridgeAlpha)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      isTrained_(false),
      backend_(backend),
      ridgeAlpha_(ridgeAlpha),
      featureMap_(numInputs, numHiddenNodes, activationKind(activation), std::nullopt,
                  hiddenWeights, hiddenBiases),
      solver_({ridgeAlpha}) {}

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

  std::vector<FloatT> hiddenOutput;
  if (!featureMap_.transform(trainData, numSamples, &hiddenOutput)) {
    isTrained_ = false;
    return false;
  }

  outputWeights_.clear();
  if (!solver_.solve(hiddenOutput, numSamples, trainTargets, numOutputs, &outputWeights_) ||
      outputWeights_.empty()) {
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

  std::vector<FloatT> hiddenOutput;
  if (!featureMap_.transform(input, 1, &hiddenOutput)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(numOutputs_);
  for (std::size_t i = 0; i < numOutputs_; ++i) {
    output[i] = FloatT(0);
    for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
      output[i] += hiddenOutput[j] * outputWeights_[j * numOutputs_ + i];
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

  std::vector<FloatT> hiddenOutput;
  if (!featureMap_.transform(testData, numSamples, &hiddenOutput)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(numSamples * numOutputs_);
  for (std::size_t i = 0; i < numSamples; ++i) {
    for (std::size_t j = 0; j < numOutputs_; ++j) {
      output[i * numOutputs_ + j] = 0;
      for (std::size_t k = 0; k < numHiddenNodes_; ++k) {
        output[i * numOutputs_ + j] +=
            hiddenOutput[i * numHiddenNodes_ + k] * outputWeights_[k * numOutputs_ + j];
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

template class BatchElm<float>;
template class BatchElm<double>;

}  // namespace feature_elm
