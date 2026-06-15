#include "core/elm_ae.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace feature_elm {

namespace {

template <typename FloatT>
FloatT activate(FloatT value, ActivationKind activation) noexcept {
  switch (activation) {
    case ActivationKind::kSigmoid:
      if (value > FloatT(0)) {
        return FloatT(1) / (FloatT(1) + std::exp(-value));
      }
      {
        const FloatT expValue = std::exp(value);
        return expValue / (FloatT(1) + expValue);
      }
    case ActivationKind::kTanh:
      return std::tanh(value);
    case ActivationKind::kRelu:
      return std::max(FloatT(0), value);
  }
  return value;
}

template <typename FloatT>
void orthogonalizeColumns(std::vector<FloatT>* weights, std::size_t rows, std::size_t cols) {
  const std::size_t orthogonalColumns = std::min(rows, cols);
  for (std::size_t col = 0; col < orthogonalColumns; ++col) {
    for (std::size_t previous = 0; previous < col; ++previous) {
      FloatT dot = FloatT(0);
      for (std::size_t row = 0; row < rows; ++row) {
        dot += (*weights)[row * cols + col] * (*weights)[row * cols + previous];
      }
      for (std::size_t row = 0; row < rows; ++row) {
        (*weights)[row * cols + col] -= dot * (*weights)[row * cols + previous];
      }
    }

    FloatT norm = FloatT(0);
    for (std::size_t row = 0; row < rows; ++row) {
      norm = std::hypot(norm, (*weights)[row * cols + col]);
    }
    if (norm > std::numeric_limits<FloatT>::epsilon()) {
      for (std::size_t row = 0; row < rows; ++row) {
        (*weights)[row * cols + col] /= norm;
      }
    }
  }
}

}  // namespace

template <typename FloatT>
ElmAutoEncoderLayer<FloatT>::ElmAutoEncoderLayer(std::size_t inputDim, std::size_t outputDim,
                                                 ActivationKind activation, unsigned int seed,
                                                 FloatT ridgeAlpha)
    : inputDim_(inputDim),
      outputDim_(outputDim),
      activation_(activation),
      solver_({ridgeAlpha}),
      isFitted_(false),
      inputWeights_(checkedMatrixSize(inputDim, outputDim).value_or(0), FloatT(0)),
      biases_(outputDim, FloatT(0)),
      outputWeights_(checkedMatrixSize(outputDim, inputDim).value_or(0), FloatT(0)),
      encoderWeights_(checkedMatrixSize(inputDim, outputDim).value_or(0), FloatT(0)),
      encoderBiases_(outputDim, FloatT(0)) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<FloatT> distribution(static_cast<FloatT>(-1),
                                                      static_cast<FloatT>(1));

  for (FloatT& value : inputWeights_) {
    value = distribution(generator);
  }
  for (FloatT& value : biases_) {
    value = distribution(generator);
  }
  if (!inputWeights_.empty()) {
    orthogonalizeColumns(&inputWeights_, inputDim_, outputDim_);
  }
}

template <typename FloatT>
bool ElmAutoEncoderLayer<FloatT>::fit(const std::vector<FloatT>& data, std::size_t numSamples) {
  const auto expectedDataSize = checkedMatrixSize(numSamples, inputDim_);
  if (numSamples == 0 || inputDim_ == 0 || outputDim_ == 0 || !expectedDataSize.has_value() ||
      data.size() != *expectedDataSize) {
    return false;
  }

  const auto hiddenSize = checkedMatrixSize(numSamples, outputDim_);
  const auto encoderSize = checkedMatrixSize(inputDim_, outputDim_);
  if (!hiddenSize.has_value() || !encoderSize.has_value()) {
    return false;
  }

  std::vector<FloatT> hiddenOutput;
  if (!computeHiddenOutput(data, numSamples, &hiddenOutput)) {
    return false;
  }

  std::vector<FloatT> beta;
  if (!solver_.solve(hiddenOutput, numSamples, data, inputDim_, &beta) || beta.empty()) {
    isFitted_ = false;
    return false;
  }

  outputWeights_ = std::move(beta);
  encoderWeights_.assign(*encoderSize, FloatT(0));
  for (std::size_t hidden = 0; hidden < outputDim_; ++hidden) {
    for (std::size_t input = 0; input < inputDim_; ++input) {
      encoderWeights_[input * outputDim_ + hidden] = outputWeights_[hidden * inputDim_ + input];
    }
  }
  std::fill(encoderBiases_.begin(), encoderBiases_.end(), FloatT(0));
  isFitted_ = true;
  return true;
}

template <typename FloatT>
bool ElmAutoEncoderLayer<FloatT>::transform(const std::vector<FloatT>& input,
                                            std::size_t numSamples,
                                            std::vector<FloatT>* output) const {
  const auto expectedInputSize = checkedMatrixSize(numSamples, inputDim_);
  const auto outputSize = checkedMatrixSize(numSamples, outputDim_);
  if (!isFitted_ || input.empty() || output == nullptr || !expectedInputSize.has_value() ||
      !outputSize.has_value() || input.size() != *expectedInputSize) {
    return false;
  }

  output->assign(*outputSize, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t hidden = 0; hidden < outputDim_; ++hidden) {
      FloatT sum = encoderBiases_[hidden];
      for (std::size_t inputIndex = 0; inputIndex < inputDim_; ++inputIndex) {
        sum += input[sample * inputDim_ + inputIndex] *
               encoderWeights_[inputIndex * outputDim_ + hidden];
      }
      (*output)[sample * outputDim_ + hidden] = activate(sum, activation_);
    }
  }
  return true;
}

template <typename FloatT>
bool ElmAutoEncoderLayer<FloatT>::reconstruct(const std::vector<FloatT>& input,
                                              std::size_t numSamples,
                                              std::vector<FloatT>* reconstruction) const {
  const auto expectedInputSize = checkedMatrixSize(numSamples, inputDim_);
  const auto reconstructionSize = checkedMatrixSize(numSamples, inputDim_);
  if (!isFitted_ || input.empty() || reconstruction == nullptr || !expectedInputSize.has_value() ||
      !reconstructionSize.has_value() || input.size() != *expectedInputSize) {
    return false;
  }

  std::vector<FloatT> hiddenOutput;
  if (!computeHiddenOutput(input, numSamples, &hiddenOutput)) {
    return false;
  }

  reconstruction->assign(*reconstructionSize, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t output = 0; output < inputDim_; ++output) {
      FloatT sum = FloatT(0);
      for (std::size_t hidden = 0; hidden < outputDim_; ++hidden) {
        sum += hiddenOutput[sample * outputDim_ + hidden] *
               outputWeights_[hidden * inputDim_ + output];
      }
      (*reconstruction)[sample * inputDim_ + output] = sum;
    }
  }
  return true;
}

template <typename FloatT>
bool ElmAutoEncoderLayer<FloatT>::computeHiddenOutput(const std::vector<FloatT>& input,
                                                      std::size_t numSamples,
                                                      std::vector<FloatT>* hiddenOutput) const {
  const auto expectedInputSize = checkedMatrixSize(numSamples, inputDim_);
  const auto hiddenSize = checkedMatrixSize(numSamples, outputDim_);
  if (input.empty() || hiddenOutput == nullptr || !expectedInputSize.has_value() ||
      !hiddenSize.has_value() || input.size() != *expectedInputSize) {
    return false;
  }

  hiddenOutput->assign(*hiddenSize, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t hidden = 0; hidden < outputDim_; ++hidden) {
      FloatT sum = biases_[hidden];
      for (std::size_t inputIndex = 0; inputIndex < inputDim_; ++inputIndex) {
        sum += input[sample * inputDim_ + inputIndex] *
               inputWeights_[inputIndex * outputDim_ + hidden];
      }
      (*hiddenOutput)[sample * outputDim_ + hidden] = activate(sum, activation_);
    }
  }
  return true;
}

template class ElmAutoEncoderLayer<float>;
template class ElmAutoEncoderLayer<double>;

}  // namespace feature_elm
