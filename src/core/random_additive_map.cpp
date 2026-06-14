#include "core/random_additive_map.hpp"

#include <cmath>
#include <random>

namespace feature_elm {

namespace {

template <typename FloatT>
FloatT activate(FloatT x, ActivationKind kind) noexcept {
  switch (kind) {
    case ActivationKind::kSigmoid: {
      if (x > FloatT(0)) {
        return FloatT(1) / (FloatT(1) + std::exp(-x));
      }
      const FloatT ex = std::exp(x);
      return ex / (FloatT(1) + ex);
    }
    case ActivationKind::kTanh:
      return std::tanh(x);
    case ActivationKind::kRelu:
      return std::max(FloatT(0), x);
  }
  return x;
}

}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename FloatT>
RandomAdditiveMap<FloatT>::RandomAdditiveMap(std::size_t inputDim, std::size_t outputDim,
                                             ActivationKind activation,
                                             std::optional<unsigned int> seed,
                                             const std::vector<FloatT>& weights,
                                             const std::vector<FloatT>& biases)
    : inputDim_(inputDim),
      outputDim_(outputDim),
      activation_(activation),
      weights_(weights.empty() ? std::vector<FloatT>() : weights),
      biases_(biases.empty() ? std::vector<FloatT>() : biases) {
  if (weights_.empty() || weights_.size() != inputDim_ * outputDim_) {
    std::mt19937 gen(seed.value_or(std::random_device{}()));
    std::uniform_real_distribution<FloatT> dist(FloatT(-1), FloatT(1));
    weights_.resize(inputDim_ * outputDim_);
    for (auto& w : weights_) {
      w = dist(gen);
    }
  }
  if (biases_.empty() || biases_.size() != outputDim_) {
    std::mt19937 gen(seed.value_or(std::random_device{}()) + 1u);
    std::uniform_real_distribution<FloatT> dist(FloatT(-1), FloatT(1));
    biases_.resize(outputDim_);
    for (auto& b : biases_) {
      b = dist(gen);
    }
  }
}
// NOLINTEND(bugprone-easily-swappable-parameters)

template <typename FloatT>
bool RandomAdditiveMap<FloatT>::fit(const std::vector<FloatT>& /*data*/,
                                    std::size_t /*numSamples*/) {
  return true;
}

template <typename FloatT>
bool RandomAdditiveMap<FloatT>::transform(const std::vector<FloatT>& input, std::size_t numSamples,
                                          std::vector<FloatT>* output) const {
  if (input.empty() || output == nullptr || input.size() != numSamples * inputDim_) {
    return false;
  }
  output->resize(numSamples * outputDim_);
  for (std::size_t i = 0; i < numSamples; ++i) {
    for (std::size_t j = 0; j < outputDim_; ++j) {
      FloatT sum = biases_[j];
      for (std::size_t k = 0; k < inputDim_; ++k) {
        sum += input[i * inputDim_ + k] * weights_[k * outputDim_ + j];
      }
      (*output)[i * outputDim_ + j] = activate(sum, activation_);
    }
  }
  return true;
}

template class RandomAdditiveMap<float>;
template class RandomAdditiveMap<double>;

}  // namespace feature_elm
