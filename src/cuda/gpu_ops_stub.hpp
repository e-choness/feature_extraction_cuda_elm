#ifndef FEATURE_ELM_CUDA_GPU_OPS_STUB_HPP_
#define FEATURE_ELM_CUDA_GPU_OPS_STUB_HPP_

#include <cstddef>
#include <vector>

#include "core/feature_map.hpp"

namespace feature_elm::cuda_backend {

[[nodiscard]] inline bool isGpuAvailable() noexcept {
  return false;
}

template <typename FloatT>
[[nodiscard]] inline bool transformRandomAdditiveGpu(
    const std::vector<FloatT>& /*input*/, std::size_t /*numSamples*/, std::size_t /*numInputs*/,
    std::size_t /*numHiddenNodes*/, const std::vector<FloatT>& /*weights*/,
    const std::vector<FloatT>& /*biases*/, ActivationKind /*activation*/,
    std::vector<FloatT>* /*hiddenOutput*/) {
  return false;
}

template <typename FloatT>
[[nodiscard]] inline bool transformElmAutoEncoderGpu(
    const std::vector<FloatT>& /*input*/, std::size_t /*numSamples*/, std::size_t /*numInputs*/,
    std::size_t /*numHiddenNodes*/, const std::vector<FloatT>& /*encoderWeights*/,
    const std::vector<FloatT>& /*encoderBiases*/, ActivationKind /*activation*/,
    std::vector<FloatT>* /*hiddenOutput*/) {
  return false;
}

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_GPU_OPS_STUB_HPP_