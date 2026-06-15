#include "cuda/elm_gpu.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

namespace feature_elm::cuda_backend {

template <typename FloatT>
[[nodiscard]] bool trainBatchElmGpu(const std::vector<FloatT>& trainData,
                                    const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                                    std::size_t numInputs, std::size_t numHiddenNodes,
                                    std::size_t numOutputs,
                                    const std::vector<FloatT>& hiddenWeights,
                                    const std::vector<FloatT>& hiddenBiases,
                                    feature_elm::ActivationFunction activation,
                                    std::vector<FloatT>* outputWeights) {
  if (!isGpuAvailable()) {
    return false;
  }
  if (trainData.size() != numSamples * numInputs ||
      trainTargets.size() != numSamples * numOutputs ||
      hiddenWeights.size() != numInputs * numHiddenNodes || hiddenBiases.size() != numHiddenNodes ||
      outputWeights == nullptr) {
    return false;
  }

  // Convert ActivationFunction to ActivationKind
  ActivationKind kind = ActivationKind::kRelu;  // Default to kRelu
  switch (activation) {
    case feature_elm::ActivationFunction::kSigmoid:
      kind = ActivationKind::kSigmoid;
      break;
    case feature_elm::ActivationFunction::kTanh:
      kind = ActivationKind::kTanh;
      break;
    case feature_elm::ActivationFunction::kRelu:
      kind = ActivationKind::kRelu;
      break;
  }

  std::vector<FloatT> hiddenOutput(numSamples * numHiddenNodes);
  if (!transformRandomAdditiveGpu<FloatT>(trainData, numSamples, numInputs, numHiddenNodes,
                                          hiddenWeights, hiddenBiases, kind, &hiddenOutput)) {
    return false;
  }

  return solveRidgeGpu<FloatT>(hiddenOutput, trainTargets, numSamples, numOutputs,
                               {static_cast<FloatT>(1e-6)}, outputWeights);
}

template <typename FloatT>
[[nodiscard]] bool predictBatchElmGpu(
    const std::vector<FloatT>& testData, std::size_t numSamples, std::size_t numInputs,
    std::size_t numHiddenNodes, std::size_t numOutputs, const std::vector<FloatT>& hiddenWeights,
    const std::vector<FloatT>& hiddenBiases, const std::vector<FloatT>& outputWeights,
    feature_elm::ActivationFunction activation, std::vector<FloatT>* predictions) {
  if (!isGpuAvailable()) {
    return false;
  }
  if (testData.size() != numSamples * numInputs ||
      hiddenWeights.size() != numInputs * numHiddenNodes || hiddenBiases.size() != numHiddenNodes ||
      outputWeights.size() != numHiddenNodes * numOutputs || predictions == nullptr) {
    return false;
  }

  // Convert ActivationFunction to ActivationKind
  ActivationKind kind = ActivationKind::kRelu;  // Default to kRelu
  switch (activation) {
    case feature_elm::ActivationFunction::kSigmoid:
      kind = ActivationKind::kSigmoid;
      break;
    case feature_elm::ActivationFunction::kTanh:
      kind = ActivationKind::kTanh;
      break;
    case feature_elm::ActivationFunction::kRelu:
      kind = ActivationKind::kRelu;
      break;
  }

  std::vector<FloatT> hiddenOutput(numSamples * numHiddenNodes);
  if (!transformRandomAdditiveGpu<FloatT>(testData, numSamples, numInputs, numHiddenNodes,
                                          hiddenWeights, hiddenBiases, kind, &hiddenOutput)) {
    return false;
  }

  predictions->assign(numSamples * numOutputs, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t outputIndex = 0; outputIndex < numOutputs; ++outputIndex) {
      FloatT sum = static_cast<FloatT>(0);
      for (std::size_t hiddenIndex = 0; hiddenIndex < numHiddenNodes; ++hiddenIndex) {
        sum += hiddenOutput[sample * numHiddenNodes + hiddenIndex] *
               outputWeights[hiddenIndex * numOutputs + outputIndex];
      }
      (*predictions)[sample * numOutputs + outputIndex] = sum;
    }
  }

  return true;
}

template <typename FloatT>
[[nodiscard]] bool computeHiddenOutputDevice(const std::vector<FloatT>& input,
                                             std::size_t numSamples, std::size_t numInputs,
                                             std::size_t numHiddenNodes,
                                             const std::vector<FloatT>& hiddenWeights,
                                             const std::vector<FloatT>& hiddenBiases,
                                             feature_elm::ActivationFunction activation,
                                             std::vector<FloatT>* hiddenOutput) {
  if (!isGpuAvailable()) {
    return false;
  }
  // Convert ActivationFunction to ActivationKind
  ActivationKind kind = ActivationKind::kRelu;  // Default to kRelu
  switch (activation) {
    case feature_elm::ActivationFunction::kSigmoid:
      kind = ActivationKind::kSigmoid;
      break;
    case feature_elm::ActivationFunction::kTanh:
      kind = ActivationKind::kTanh;
      break;
    case feature_elm::ActivationFunction::kRelu:
      kind = ActivationKind::kRelu;
      break;
  }

  return transformRandomAdditiveGpu<FloatT>(input, numSamples, numInputs, numHiddenNodes,
                                            hiddenWeights, hiddenBiases, kind, hiddenOutput);
}

// Explicit template instantiations
template bool trainBatchElmGpu<float>(const std::vector<float>&, const std::vector<float>&,
                                      std::size_t, std::size_t, std::size_t, std::size_t,
                                      const std::vector<float>&, const std::vector<float>&,
                                      feature_elm::ActivationFunction, std::vector<float>*);

template bool trainBatchElmGpu<double>(const std::vector<double>&, const std::vector<double>&,
                                       std::size_t, std::size_t, std::size_t, std::size_t,
                                       const std::vector<double>&, const std::vector<double>&,
                                       feature_elm::ActivationFunction, std::vector<double>*);

template bool predictBatchElmGpu<float>(const std::vector<float>&, std::size_t, std::size_t,
                                        std::size_t, std::size_t, const std::vector<float>&,
                                        const std::vector<float>&, const std::vector<float>&,
                                        feature_elm::ActivationFunction, std::vector<float>*);

template bool predictBatchElmGpu<double>(const std::vector<double>&, std::size_t, std::size_t,
                                         std::size_t, std::size_t, const std::vector<double>&,
                                         const std::vector<double>&, const std::vector<double>&,
                                         feature_elm::ActivationFunction, std::vector<double>*);

template bool computeHiddenOutputDevice<float>(const std::vector<float>&, std::size_t, std::size_t,
                                               std::size_t, const std::vector<float>&,
                                               const std::vector<float>&,
                                               feature_elm::ActivationFunction,
                                               std::vector<float>*);

template bool computeHiddenOutputDevice<double>(
    const std::vector<double>&, std::size_t, std::size_t, std::size_t, const std::vector<double>&,
    const std::vector<double>&, feature_elm::ActivationFunction, std::vector<double>*);

}  // namespace feature_elm::cuda_backend