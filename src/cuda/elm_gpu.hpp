#ifndef FEATURE_ELM_CUDA_ELM_GPU_HPP_
#define FEATURE_ELM_CUDA_ELM_GPU_HPP_

#include <cstddef>
#include <vector>

#include "core/elm.hpp"
#include "cuda/gpu_ops.hpp"

namespace feature_elm::cuda_backend {

template <typename FloatT>
[[nodiscard]] bool trainBatchElmGpu(const std::vector<FloatT>& trainData,
                                    const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                                    std::size_t numInputs, std::size_t numHiddenNodes,
                                    std::size_t numOutputs,
                                    const std::vector<FloatT>& hiddenWeights,
                                    const std::vector<FloatT>& hiddenBiases,
                                    feature_elm::ActivationFunction activation,
                                    std::vector<FloatT>* outputWeights);

template <typename FloatT>
[[nodiscard]] bool predictBatchElmGpu(
    const std::vector<FloatT>& testData, std::size_t numSamples, std::size_t numInputs,
    std::size_t numHiddenNodes, std::size_t numOutputs, const std::vector<FloatT>& hiddenWeights,
    const std::vector<FloatT>& hiddenBiases, const std::vector<FloatT>& outputWeights,
    feature_elm::ActivationFunction activation, std::vector<FloatT>* predictions);

template <typename FloatT>
[[nodiscard]] bool computeHiddenOutputDevice(const std::vector<FloatT>& input,
                                             std::size_t numSamples, std::size_t numInputs,
                                             std::size_t numHiddenNodes,
                                             const std::vector<FloatT>& hiddenWeights,
                                             const std::vector<FloatT>& hiddenBiases,
                                             feature_elm::ActivationFunction activation,
                                             std::vector<FloatT>* hiddenOutput);

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_ELM_GPU_HPP_
