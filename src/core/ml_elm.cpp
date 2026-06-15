#include "core/ml_elm.hpp"

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
MlElm<FloatT>::MlElm(std::size_t numInputs, const std::vector<std::size_t>& hiddenNodesPerLayer,
                     ActivationFunction activation, Backend backend, FloatT ridgeAlpha,
                     unsigned int seed)
    : numInputs_(numInputs),
      numOutputs_(0),
      hiddenNodesPerLayer_(hiddenNodesPerLayer),
      activation_(activation),
      backend_(backend),
      ridgeAlpha_(ridgeAlpha),
      seed_(seed),
      isTrained_(false),
      featureStack_(makeFeatureStack(numInputs, hiddenNodesPerLayer, activation, seed, ridgeAlpha)),
      solver_({ridgeAlpha}) {}

template <typename FloatT>
StackedFeatureMap<FloatT> MlElm<FloatT>::makeFeatureStack(
    std::size_t numInputs, const std::vector<std::size_t>& hiddenNodesPerLayer,
    ActivationFunction activation, unsigned int seed, FloatT ridgeAlpha) {
  return StackedFeatureMap<FloatT>(numInputs, hiddenNodesPerLayer, activationKind(activation), seed,
                                   ridgeAlpha);
}

template <typename FloatT>
bool MlElm<FloatT>::train(const std::vector<FloatT>& trainData,
                          const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                          std::size_t numOutputs) {
  const auto expectedDataSize = checkedMatrixSize(numSamples, numInputs_);
  const auto expectedTargetsSize = checkedMatrixSize(numSamples, numOutputs);
  if (numSamples == 0 || !expectedDataSize.has_value() || !expectedTargetsSize.has_value() ||
      trainData.size() != *expectedDataSize || trainTargets.size() != *expectedTargetsSize) {
    return false;
  }
  numOutputs_ = numOutputs;

  std::vector<FloatT> features;
  if (!featureStack_.fit(trainData, numSamples) ||
      !featureStack_.transform(trainData, numSamples, &features)) {
    isTrained_ = false;
    return false;
  }

  outputWeights_.clear();
  if (!solver_.solve(features, numSamples, trainTargets, numOutputs, &outputWeights_) ||
      outputWeights_.empty()) {
    isTrained_ = false;
    return false;
  }

  isTrained_ = true;
  return true;
}

template <typename FloatT>
std::optional<std::vector<FloatT>> MlElm<FloatT>::predict(const std::vector<FloatT>& input) const {
  if (!isTrained_ || input.size() != numInputs_) {
    return std::nullopt;
  }

  auto output = predictBatch(input, 1);
  if (!output.has_value() || output->empty()) {
    return std::nullopt;
  }
  return output;
}

template <typename FloatT>
std::optional<std::vector<FloatT>> MlElm<FloatT>::predictBatch(const std::vector<FloatT>& testData,
                                                               std::size_t numSamples) const {
  const auto expectedDataSize = checkedMatrixSize(numSamples, numInputs_);
  const auto outputSize = checkedMatrixSize(numSamples, numOutputs_);
  if (!isTrained_ || numSamples == 0 || !expectedDataSize.has_value() || !outputSize.has_value() ||
      testData.size() != *expectedDataSize) {
    return std::nullopt;
  }

  std::vector<FloatT> features;
  if (!featureStack_.transform(testData, numSamples, &features)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(*outputSize, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t out = 0; out < numOutputs_; ++out) {
      for (std::size_t feature = 0; feature < finalFeatureDim(); ++feature) {
        output[sample * numOutputs_ + out] += features[sample * finalFeatureDim() + feature] *
                                              outputWeights_[feature * numOutputs_ + out];
      }
    }
  }

  return output;
}

template <typename FloatT>
void MlElm<FloatT>::reset() noexcept {
  featureStack_.reset();
  solver_ = BatchRidgeSolver<FloatT>({ridgeAlpha_});
  outputWeights_.clear();
  numOutputs_ = 0;
  isTrained_ = false;
}

template class MlElm<float>;
template class MlElm<double>;

}  // namespace feature_elm
