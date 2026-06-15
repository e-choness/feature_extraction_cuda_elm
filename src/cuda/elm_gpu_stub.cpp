#include "core/stacked_feature_map.hpp"
#include "cuda/elm_gpu.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

namespace feature_elm::cuda_backend {

[[nodiscard]] bool isGpuAvailable() noexcept {
  return false;
}

template <typename FloatT>
[[nodiscard]] bool trainBatchElmGpu(const std::vector<FloatT>& /*trainData*/,
                                    const std::vector<FloatT>& /*trainTargets*/,
                                    std::size_t /*numSamples*/, std::size_t /*numInputs*/,
                                    std::size_t /*numHiddenNodes*/, std::size_t /*numOutputs*/,
                                    const std::vector<FloatT>& /*hiddenWeights*/,
                                    const std::vector<FloatT>& /*hiddenBiases*/,
                                    feature_elm::ActivationFunction /*activation*/,
                                    std::vector<FloatT>* /*outputWeights*/) {
  return false;
}

template <typename FloatT>
[[nodiscard]] bool predictBatchElmGpu(const std::vector<FloatT>& /*testData*/,
                                      std::size_t /*numSamples*/, std::size_t /*numInputs*/,
                                      std::size_t /*numHiddenNodes*/, std::size_t /*numOutputs*/,
                                      const std::vector<FloatT>& /*hiddenWeights*/,
                                      const std::vector<FloatT>& /*hiddenBiases*/,
                                      const std::vector<FloatT>& /*outputWeights*/,
                                      feature_elm::ActivationFunction /*activation*/,
                                      std::vector<FloatT>* /*predictions*/) {
  return false;
}

template <typename FloatT>
[[nodiscard]] bool computeHiddenOutputDevice(const std::vector<FloatT>& /*input*/,
                                             std::size_t /*numSamples*/, std::size_t /*numInputs*/,
                                             std::size_t /*numHiddenNodes*/,
                                             const std::vector<FloatT>& /*hiddenWeights*/,
                                             const std::vector<FloatT>& /*hiddenBiases*/,
                                             feature_elm::ActivationFunction /*activation*/,
                                             std::vector<FloatT>* /*hiddenOutput*/) {
  return false;
}

template <typename FloatT>
[[nodiscard]] bool transformRandomAdditiveGpu(const std::vector<FloatT>& /*input*/,
                                              std::size_t /*numSamples*/, std::size_t /*numInputs*/,
                                              std::size_t /*numHiddenNodes*/,
                                              const std::vector<FloatT>& /*weights*/,
                                              const std::vector<FloatT>& /*biases*/,
                                              ActivationKind /*activation*/,
                                              std::vector<FloatT>* /*hiddenOutput*/) {
  return false;
}

template <typename FloatT>
[[nodiscard]] bool transformElmAutoEncoderGpu(const std::vector<FloatT>& /*input*/,
                                              std::size_t /*numSamples*/, std::size_t /*numInputs*/,
                                              std::size_t /*numHiddenNodes*/,
                                              const std::vector<FloatT>& /*encoderWeights*/,
                                              const std::vector<FloatT>& /*encoderBiases*/,
                                              ActivationKind /*activation*/,
                                              std::vector<FloatT>* /*hiddenOutput*/) {
  return false;
}

template <typename FloatT>
[[nodiscard]] bool solveRidgeGpu(const std::vector<FloatT>& /*features*/,
                                 const std::vector<FloatT>& /*targets*/, std::size_t /*numSamples*/,
                                 std::size_t /*numOutputs*/, SolverOptions<FloatT> /*options*/,
                                 std::vector<FloatT>* /*weights*/) {
  return false;
}

template <typename FloatT>
[[nodiscard]] bool transformStackedFeatureMapGpu(const StackedFeatureMap<FloatT>& /*featureStack*/,
                                                 const std::vector<FloatT>& /*input*/,
                                                 std::size_t /*numSamples*/,
                                                 std::vector<FloatT>* /*output*/) {
  return false;
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

template bool transformRandomAdditiveGpu<float>(const std::vector<float>&, std::size_t, std::size_t,
                                                std::size_t, const std::vector<float>&,
                                                const std::vector<float>&, ActivationKind,
                                                std::vector<float>*);

template bool transformRandomAdditiveGpu<double>(const std::vector<double>&, std::size_t,
                                                 std::size_t, std::size_t,
                                                 const std::vector<double>&,
                                                 const std::vector<double>&, ActivationKind,
                                                 std::vector<double>*);

template bool transformElmAutoEncoderGpu<float>(const std::vector<float>&, std::size_t, std::size_t,
                                                std::size_t, const std::vector<float>&,
                                                const std::vector<float>&, ActivationKind,
                                                std::vector<float>*);

template bool transformElmAutoEncoderGpu<double>(const std::vector<double>&, std::size_t,
                                                 std::size_t, std::size_t,
                                                 const std::vector<double>&,
                                                 const std::vector<double>&, ActivationKind,
                                                 std::vector<double>*);

template bool solveRidgeGpu<float>(const std::vector<float>&, const std::vector<float>&,
                                   std::size_t, std::size_t, SolverOptions<float>,
                                   std::vector<float>*);

template bool solveRidgeGpu<double>(const std::vector<double>&, const std::vector<double>&,
                                    std::size_t, std::size_t, SolverOptions<double>,
                                    std::vector<double>*);

template bool transformStackedFeatureMapGpu<float>(const StackedFeatureMap<float>&,
                                                   const std::vector<float>&, std::size_t,
                                                   std::vector<float>*);

template bool transformStackedFeatureMapGpu<double>(const StackedFeatureMap<double>&,
                                                    const std::vector<double>&, std::size_t,
                                                    std::vector<double>*);

}  // namespace feature_elm::cuda_backend