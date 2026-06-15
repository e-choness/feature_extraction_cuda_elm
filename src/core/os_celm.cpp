#include "core/os_celm.hpp"

#include <algorithm>
#include <random>

#include "cuda/elm_gpu.hpp"

namespace feature_elm {

namespace {

[[nodiscard]] ActivationKind activationKind(ActivationFunction activation) {
  switch (activation) {
    case ActivationFunction::kSigmoid:
      return ActivationKind::kSigmoid;
    case ActivationFunction::kTanh:
      return ActivationKind::kTanh;
    case ActivationFunction::kRelu:
      return ActivationKind::kRelu;
  }
  return ActivationKind::kSigmoid;
}

template <typename FloatT>
[[nodiscard]] std::vector<FloatT> normalizeHiddenWeights(std::size_t numInputs,
                                                         std::size_t numHiddenNodes,
                                                         const std::vector<FloatT>& weights) {
  if (weights.size() == numInputs * numHiddenNodes) {
    return weights;
  }
  std::vector<FloatT> normalized(numInputs * numHiddenNodes, FloatT(0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FloatT> dist(FloatT(-1), FloatT(1));
  for (auto& value : normalized) {
    value = dist(gen);
  }
  return normalized;
}

template <typename FloatT>
[[nodiscard]] std::vector<FloatT> normalizeHiddenBiases(std::size_t numHiddenNodes,
                                                        const std::vector<FloatT>& biases) {
  if (biases.size() == numHiddenNodes) {
    return biases;
  }
  std::vector<FloatT> normalized(numHiddenNodes, FloatT(0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FloatT> dist(FloatT(-1), FloatT(1));
  for (auto& value : normalized) {
    value = dist(gen);
  }
  return normalized;
}

template <typename FloatT>
[[nodiscard]] RlsOptions<FloatT> effectiveOptions(FloatT constraintStrength,
                                                  RlsOptions<FloatT> rlsOptions) {
  if (constraintStrength > FloatT(0)) {
    rlsOptions.constraint = RlsConstraint::kClassDistance;
    rlsOptions.constraintStrength = constraintStrength;
  }
  return rlsOptions;
}

}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename FloatT>
OsCelm<FloatT>::OsCelm(std::size_t numInputs, std::size_t numHiddenNodes,
                       ActivationFunction activation, FloatT constraintStrength, Backend backend,
                       RlsOptions<FloatT> rlsOptions)
    : OsCelm(numInputs, numHiddenNodes, activation, backend, {}, {}, constraintStrength,
             rlsOptions) {}

template <typename FloatT>
OsCelm<FloatT>::OsCelm(std::size_t numInputs, std::size_t numHiddenNodes,
                       ActivationFunction activation, Backend backend, FloatT constraintStrength,
                       RlsOptions<FloatT> rlsOptions)
    : OsCelm(numInputs, numHiddenNodes, activation, backend, {}, {}, constraintStrength,
             rlsOptions) {}

template <typename FloatT>
OsCelm<FloatT>::OsCelm(std::size_t numInputs, std::size_t numHiddenNodes,
                       ActivationFunction activation, Backend backend,
                       const std::vector<FloatT>& hiddenWeights,
                       const std::vector<FloatT>& hiddenBiases, FloatT constraintStrength,
                       RlsOptions<FloatT> rlsOptions)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      backend_(backend),
      isInitialized_(false),
      hiddenWeights_(normalizeHiddenWeights<FloatT>(numInputs, numHiddenNodes, hiddenWeights)),
      hiddenBiases_(normalizeHiddenBiases<FloatT>(numHiddenNodes, hiddenBiases)),
      featureMap_(numInputs_, numHiddenNodes_, activationKind(activation), std::nullopt,
                  hiddenWeights_, hiddenBiases_),
      rlsSolver_(effectiveOptions(constraintStrength, rlsOptions)) {}
// NOLINTEND(bugprone-easily-swappable-parameters)

template <typename FloatT>
void OsCelm<FloatT>::reset() noexcept {
  rlsSolver_.reset();
  numOutputs_ = 0;
  isInitialized_ = false;
}

template <typename FloatT>
[[nodiscard]] bool OsCelm<FloatT>::computeHiddenOutput(const std::vector<FloatT>& input,
                                                       std::size_t numSamples,
                                                       std::vector<FloatT>* hiddenOutput) const {
  if (backend_ == Backend::kGpu) {
    return cuda_backend::computeHiddenOutputDevice(input, numSamples, numInputs_, numHiddenNodes_,
                                                   hiddenWeights_, hiddenBiases_, activation_,
                                                   hiddenOutput);
  }
  return featureMap_.transform(input, numSamples, hiddenOutput);
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

  std::vector<FloatT> hiddenOutput;
  if (!computeHiddenOutput(initialData, numSamples, &hiddenOutput)) {
    return false;
  }
  if (!rlsSolver_.initialize(hiddenOutput, numSamples, initialTargets, numOutputs)) {
    return false;
  }

  numOutputs_ = numOutputs;
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
  return rlsSolver_.update(hiddenOutput, numSamples, newTargets);
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

  std::vector<FloatT> hiddenOutput;
  if (!computeHiddenOutput(input, 1, &hiddenOutput)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(numOutputs_, FloatT(0));
  const std::vector<FloatT>& weights = rlsSolver_.weights();
  for (std::size_t out = 0; out < numOutputs_; ++out) {
    for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
      output[out] += hiddenOutput[i] * weights[i * numOutputs_ + out];
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
  const std::vector<FloatT>& weights = rlsSolver_.weights();
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t out = 0; out < numOutputs_; ++out) {
      for (std::size_t i = 0; i < numHiddenNodes_; ++i) {
        output[sample * numOutputs_ + out] +=
            hiddenOutput[sample * numHiddenNodes_ + i] * weights[i * numOutputs_ + out];
      }
    }
  }
  return output;
}

template class OsCelm<float>;
template class OsCelm<double>;

}  // namespace feature_elm
