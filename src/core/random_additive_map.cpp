#include "core/random_additive_map.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace feature_elm {

namespace {
FloatT activate(float x, ActivationKind kind) noexcept {
  switch (kind) {
    case ActivationKind::kSigmoid: {
      if (x > 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
      }
      const float ex = std::exp(x);
      return ex / (1.0f + ex);
    }
    case ActivationKind::kTanh:
      return std::tanh(x);
    case ActivationKind::kRelu:
      return std::max(0.0f, x);
  }
  return x;
}

FloatT activate(double x, ActivationKind kind) noexcept {
  switch (kind) {
    case ActivationKind::kSigmoid: {
      if (x > 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
      }
      const double ex = std::exp(x);
      return ex / (1.0 + ex);
    }
    case ActivationKind::kTanh:
      return std::tanh(x);
    case ActivationKind::kRelu:
      return std::max(0.0, x);
  }
  return x;
}
}  // namespace

template <typename FloatT>
RandomAdditiveMap<FloatT>::RandomAdditiveMap(std::size_t inputDim, std::size_t outputDim,
                                             ActivationKind activation, std::optional<unsigned int> seed,
                                             const std::vector<FloatT>& weights,
                                             const std::vector<FloatT>& biases)
    : inputDim_(inputDim),
      outputDim_(outputDim),
      activation_(activation),
      weights_(weights.empty() ? std::vector<FloatT>() : weights),
      biases_(biases.empty() ? std::vector<FloatT>() : biases) {
  if (this->weights_.empty() || this->weights_.size() != inputDim_ * outputDim_) {
    std::mt19937 gen(seed.value_or(std::random_device{}()));
    std::uniform_real_distribution<FloatT> dist(static_cast<FloatT>(-1), static_cast<FloatT>(1));
    this->weights_.resize(inputDim_ * outputDim_);
    for (auto& w : this->weights_) {
      w = dist(gen);
    }
  }
  if (this->biases_.empty() || this->biases_.size() != outputDim_) {
    std::mt19937 gen(seed.value_or(std::random_device{}()) + 1u);
    std::uniform_real_distribution<FloatT> dist(static_cast<FloatT>(-1), static_cast<FloatT>(1));
    this->biases_.resize(outputDim_);
    for (auto& b : this->biases_) {
      b = dist(gen);
    }
  }
}

template <typename FloatT>
bool RandomAdditiveMap<FloatT>::fit(const std::vector<FloatT>& /*data*/, std::size_t /*numSamples*/) {
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

}  // namespace feature_elm