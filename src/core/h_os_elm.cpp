#include "core/h_os_elm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace feature_elm {

namespace {

template <typename FloatT>
[[nodiscard]] bool computeLayerOutput(
    const std::vector<FloatT>& input,
    std::size_t numSamples,
    std::size_t numInputs,
    std::size_t numHiddenNodes,
    const std::vector<FloatT>& weights,
    const std::vector<FloatT>& biases,
    ActivationFunction activation,
    std::vector<FloatT>* output) {
  if (input.size() != numSamples * numInputs || output == nullptr) {
    return false;
  }

  output->assign(numSamples * numHiddenNodes, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t hiddenIndex = 0; hiddenIndex < numHiddenNodes; ++hiddenIndex) {
      FloatT sum = biases[hiddenIndex];
      for (std::size_t inputIndex = 0; inputIndex < numInputs; ++inputIndex) {
        sum += weights[inputIndex * numHiddenNodes + hiddenIndex] *
               input[sample * numInputs + inputIndex];
      }
      if (activation == ActivationFunction::kSigmoid) {
        (*output)[sample * numHiddenNodes + hiddenIndex] =
            static_cast<FloatT>(1) / (static_cast<FloatT>(1) + std::exp(-sum));
      } else {
        (*output)[sample * numHiddenNodes + hiddenIndex] = std::exp(-sum * sum);
      }
    }
  }
  return true;
}

}  // namespace

template <typename FloatT>
HierarchicalOsElm<FloatT>::HierarchicalOsElm(
    std::size_t numInputs,
    const std::vector<std::size_t>& hiddenNodesPerLayer,
    ActivationFunction activation,
    Backend backend)
    : numInputs_(numInputs),
      hiddenNodesPerLayer_(hiddenNodesPerLayer),
      activation_(activation),
      backend_(backend),
      isInitialized_(false),
      numOutputs_(0),
      hiddenWeights_(hiddenNodesPerLayer.size()),
      hiddenBiases_(hiddenNodesPerLayer.size()),
      topModel_(0, 0, activation, backend) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FloatT> dis(static_cast<FloatT>(-1), static_cast<FloatT>(1));

  std::size_t currentInputDim = numInputs_;
  for (std::size_t layer = 0; layer < hiddenNodesPerLayer_.size(); ++layer) {
    std::size_t layerNodes = hiddenNodesPerLayer_[layer];
    hiddenWeights_[layer].resize(currentInputDim * layerNodes);
    hiddenBiases_[layer].resize(layerNodes);
    for (auto& w : hiddenWeights_[layer]) {
      w = dis(gen);
    }
    for (auto& b : hiddenBiases_[layer]) {
      b = dis(gen);
    }
    currentInputDim = layerNodes;
  }
}

template <typename FloatT>
[[nodiscard]] bool HierarchicalOsElm<FloatT>::computeHierarchicalFeatures(
    const std::vector<FloatT>& data,
    std::size_t numSamples,
    std::vector<FloatT>* features) const {
  if (data.size() != numSamples * numInputs_ || features == nullptr) {
    return false;
  }

  std::vector<FloatT> currentInput = data;
  std::size_t currentDim = numInputs_;
  for (std::size_t layer = 0; layer < hiddenNodesPerLayer_.size(); ++layer) {
    std::vector<FloatT> layerOutput;
    if (!computeLayerOutput(currentInput, numSamples, currentDim,
                            hiddenNodesPerLayer_[layer], hiddenWeights_[layer],
                            hiddenBiases_[layer], activation_, &layerOutput)) {
      return false;
    }
    currentInput = std::move(layerOutput);
    currentDim = hiddenNodesPerLayer_[layer];
  }

  *features = std::move(currentInput);
  return true;
}

template <typename FloatT>
[[nodiscard]] bool HierarchicalOsElm<FloatT>::initialize(
    const std::vector<FloatT>& data,
    const std::vector<FloatT>& targets,
    std::size_t numSamples,
    std::size_t numOutputs) {
  if (isInitialized_) {
    return false;
  }
  if (data.size() != numSamples * numInputs_ ||
      targets.size() != numSamples * numOutputs) {
    return false;
  }

  std::vector<FloatT> features;
  if (!computeHierarchicalFeatures(data, numSamples, &features)) {
    return false;
  }

  std::size_t topInputDim = hiddenNodesPerLayer_.empty() ? numInputs_ : hiddenNodesPerLayer_.back();
  topModel_ = OsElm<FloatT>(topInputDim, topInputDim, activation_, backend_);
  if (!topModel_.initialize(features, targets, numSamples, numOutputs)) {
    return false;
  }

  numOutputs_ = numOutputs;
  isInitialized_ = true;
  return true;
}

template <typename FloatT>
[[nodiscard]] bool HierarchicalOsElm<FloatT>::update(
    const std::vector<FloatT>& newData,
    const std::vector<FloatT>& newTargets,
    std::size_t numSamples) {
  if (!isInitialized_) {
    return false;
  }
  if (newData.size() != numSamples * numInputs_ ||
      newTargets.size() != numSamples * numOutputs_) {
    return false;
  }

  std::vector<FloatT> features;
  if (!computeHierarchicalFeatures(newData, numSamples, &features)) {
    return false;
  }

  return topModel_.update(features, newTargets, numSamples);
}

template <typename FloatT>
[[nodiscard]] std::optional<std::vector<FloatT>> HierarchicalOsElm<FloatT>::predictBatch(
    const std::vector<FloatT>& testData,
    std::size_t numSamples) const {
  if (!isInitialized_) {
    return std::nullopt;
  }
  if (testData.size() != numSamples * numInputs_) {
    return std::nullopt;
  }

  std::vector<FloatT> features;
  if (!computeHierarchicalFeatures(testData, numSamples, &features)) {
    return std::nullopt;
  }

  return topModel_.predictBatch(features, numSamples);
}

template <typename FloatT>
void HierarchicalOsElm<FloatT>::reset() noexcept {
  hiddenWeights_.clear();
  hiddenBiases_.clear();
  topModel_.reset();
  numOutputs_ = 0;
  isInitialized_ = false;
}

template class HierarchicalOsElm<float>;
template class HierarchicalOsElm<double>;

}  // namespace feature_elm
