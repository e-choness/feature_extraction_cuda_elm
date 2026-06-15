#include "core/stacked_feature_map.hpp"

#include <utility>

namespace feature_elm {

template <typename FloatT>
StackedFeatureMap<FloatT>::StackedFeatureMap(std::size_t inputDim,
                                             std::vector<std::size_t> layerOutputDims,
                                             ActivationKind activation, unsigned int seed,
                                             FloatT ridgeAlpha)
    : inputDim_(inputDim),
      layerOutputDims_(std::move(layerOutputDims)),
      activation_(activation),
      seed_(seed),
      ridgeAlpha_(ridgeAlpha),
      isFitted_(false) {
  std::size_t currentDim = inputDim_;
  for (std::size_t layer = 0; layer < layerOutputDims_.size(); ++layer) {
    layers_.push_back(std::make_unique<ElmAutoEncoderLayer<FloatT>>(
        currentDim, layerOutputDims_[layer], activation_, seed_ + static_cast<unsigned int>(layer),
        ridgeAlpha_));
    currentDim = layerOutputDims_[layer];
  }
}

template <typename FloatT>
bool StackedFeatureMap<FloatT>::fit(const std::vector<FloatT>& data, std::size_t numSamples) {
  const auto expectedDataSize = checkedMatrixSize(numSamples, inputDim_);
  if (numSamples == 0 || !expectedDataSize.has_value() || data.size() != *expectedDataSize) {
    return false;
  }

  std::vector<FloatT> current = data;
  std::size_t currentDim = inputDim_;
  for (auto& layer : layers_) {
    if (layer == nullptr || layer->inputDim() != currentDim) {
      isFitted_ = false;
      return false;
    }
    if (!layer->fit(current, numSamples)) {
      isFitted_ = false;
      return false;
    }
    std::vector<FloatT> next;
    if (!layer->transform(current, numSamples, &next)) {
      isFitted_ = false;
      return false;
    }
    current = std::move(next);
    currentDim = layer->outputDim();
  }

  isFitted_ = true;
  return true;
}

template <typename FloatT>
bool StackedFeatureMap<FloatT>::transform(const std::vector<FloatT>& input, std::size_t numSamples,
                                          std::vector<FloatT>* output) const {
  const auto expectedInputSize = checkedMatrixSize(numSamples, inputDim_);
  if (input.empty() || output == nullptr || !expectedInputSize.has_value() ||
      input.size() != *expectedInputSize) {
    return false;
  }

  std::vector<FloatT> current = input;
  for (const auto& layer : layers_) {
    if (layer == nullptr) {
      return false;
    }
    std::vector<FloatT> next;
    if (!layer->transform(current, numSamples, &next)) {
      return false;
    }
    current = std::move(next);
  }

  *output = std::move(current);
  return true;
}

template <typename FloatT>
void StackedFeatureMap<FloatT>::reset() noexcept {
  layers_.clear();
  isFitted_ = false;
}

template class StackedFeatureMap<float>;
template class StackedFeatureMap<double>;

}  // namespace feature_elm
