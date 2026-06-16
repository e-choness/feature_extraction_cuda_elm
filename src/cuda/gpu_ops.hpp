#ifndef FEATURE_ELM_CUDA_GPU_OPS_HPP_
#define FEATURE_ELM_CUDA_GPU_OPS_HPP_

#include <cstddef>
#include <vector>

#include "core/feature_map.hpp"

namespace feature_elm::cuda_backend {

[[nodiscard]] bool isGpuAvailable() noexcept;

template <typename FloatT>
[[nodiscard]] bool transformRandomAdditiveGpu(const std::vector<FloatT>& input,
                                              std::size_t numSamples, std::size_t numInputs,
                                              std::size_t numHiddenNodes,
                                              const std::vector<FloatT>& weights,
                                              const std::vector<FloatT>& biases,
                                              ActivationKind activation,
                                              std::vector<FloatT>* hiddenOutput);

template <typename FloatT>
[[nodiscard]] bool transformElmAutoEncoderGpu(const std::vector<FloatT>& input,
                                              std::size_t numSamples, std::size_t numInputs,
                                              std::size_t numHiddenNodes,
                                              const std::vector<FloatT>& encoderWeights,
                                              const std::vector<FloatT>& encoderBiases,
                                              ActivationKind activation,
                                              std::vector<FloatT>* hiddenOutput);

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_GPU_OPS_HPP_