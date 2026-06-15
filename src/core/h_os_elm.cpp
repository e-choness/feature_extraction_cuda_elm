#include "core/h_os_elm.hpp"

#include <algorithm>
#include <utility>

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

}  // namespace

template <typename FloatT>
HierarchicalOsElm<FloatT>::HierarchicalOsElm(std::size_t numInputs,
                                             const std::vector<std::size_t>& hiddenNodesPerLayer,
                                             ActivationFunction activation, Backend backend,
                                             RlsOptions<FloatT> rlsOptions, FloatT ridgeAlpha,
                                             unsigned int seed)
    : numInputs_(numInputs),
      hiddenNodesPerLayer_(hiddenNodesPerLayer),
      activation_(activation),
      backend_(backend),
      ridgeAlpha_(ridgeAlpha),
      seed_(seed),
      isInitialized_(false),
      numOutputs_(0),
      featureStack_(numInputs, hiddenNodesPerLayer, activationKind(activation), seed, ridgeAlpha),
      rlsSolver_(rlsOptions) {}

template <typename FloatT>
[[nodiscard]] bool HierarchicalOsElm<FloatT>::computeHierarchicalFeatures(
    const std::vector<FloatT>& data, std::size_t numSamples, std::vector<FloatT>* features) const {
  if (data.size() != numSamples * numInputs_ || features == nullptr) {
    return false;
  }
  return featureStack_.transform(data, numSamples, features);
}

template <typename FloatT>
[[nodiscard]] bool HierarchicalOsElm<FloatT>::initialize(const std::vector<FloatT>& data,
                                                         const std::vector<FloatT>& targets,
                                                         std::size_t numSamples,
                                                         std::size_t numOutputs) {
  if (isInitialized_) {
    return false;
  }
  const auto expectedDataSize = checkedMatrixSize(numSamples, numInputs_);
  const auto expectedTargetsSize = checkedMatrixSize(numSamples, numOutputs);
  if (numSamples == 0 || !expectedDataSize.has_value() || !expectedTargetsSize.has_value() ||
      data.size() != *expectedDataSize || targets.size() != *expectedTargetsSize) {
    return false;
  }

  if (!featureStack_.fit(data, numSamples)) {
    return false;
  }

  std::vector<FloatT> features;
  if (!computeHierarchicalFeatures(data, numSamples, &features)) {
    return false;
  }
  if (!rlsSolver_.initialize(features, numSamples, targets, numOutputs)) {
    return false;
  }

  numOutputs_ = numOutputs;
  isInitialized_ = true;
  return true;
}

template <typename FloatT>
[[nodiscard]] bool HierarchicalOsElm<FloatT>::update(const std::vector<FloatT>& newData,
                                                     const std::vector<FloatT>& newTargets,
                                                     std::size_t numSamples) {
  if (!isInitialized_) {
    return false;
  }
  const auto expectedDataSize = checkedMatrixSize(numSamples, numInputs_);
  const auto expectedTargetsSize = checkedMatrixSize(numSamples, numOutputs_);
  if (numSamples == 0 || !expectedDataSize.has_value() || !expectedTargetsSize.has_value() ||
      newData.size() != *expectedDataSize || newTargets.size() != *expectedTargetsSize) {
    return false;
  }

  std::vector<FloatT> features;
  if (!computeHierarchicalFeatures(newData, numSamples, &features)) {
    return false;
  }

  return rlsSolver_.update(features, numSamples, newTargets);
}

template <typename FloatT>
[[nodiscard]] std::optional<std::vector<FloatT>> HierarchicalOsElm<FloatT>::predictBatch(
    const std::vector<FloatT>& testData, std::size_t numSamples) const {
  const auto expectedDataSize = checkedMatrixSize(numSamples, numInputs_);
  const auto outputSize = checkedMatrixSize(numSamples, numOutputs_);
  if (!isInitialized_ || numSamples == 0 || !expectedDataSize.has_value() ||
      !outputSize.has_value() || testData.size() != *expectedDataSize) {
    return std::nullopt;
  }

  std::vector<FloatT> features;
  if (!computeHierarchicalFeatures(testData, numSamples, &features)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(*outputSize, FloatT(0));
  const std::vector<FloatT>& weights = rlsSolver_.weights();
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t out = 0; out < numOutputs_; ++out) {
      for (std::size_t feature = 0; feature < featureStack_.outputDim(); ++feature) {
        output[sample * numOutputs_ + out] +=
            features[sample * featureStack_.outputDim() + feature] *
            weights[feature * numOutputs_ + out];
      }
    }
  }

  return output;
}

template <typename FloatT>
void HierarchicalOsElm<FloatT>::reset() noexcept {
  featureStack_.reset();
  rlsSolver_.reset();
  numOutputs_ = 0;
  isInitialized_ = false;
}

template class HierarchicalOsElm<float>;
template class HierarchicalOsElm<double>;

}  // namespace feature_elm
