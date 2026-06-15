#ifndef FEATURE_ELM_CUDA_FEATURE_MAP_GPU_HPP_
#define FEATURE_ELM_CUDA_FEATURE_MAP_GPU_HPP_

#include <cstddef>
#include <vector>

#include "core/feature_map.hpp"
#include "core/stacked_feature_map.hpp"
#include "cuda/gpu_ops.hpp"

namespace feature_elm::cuda_backend {

template <typename FloatT>
[[nodiscard]] bool transformStackedFeatureMapGpu(const StackedFeatureMap<FloatT>& featureStack,
                                                 const std::vector<FloatT>& input,
                                                 std::size_t numSamples,
                                                 std::vector<FloatT>* output) {
  if (input.empty() || output == nullptr) {
    return false;
  }

  std::vector<FloatT> current = input;
  for (const auto& layer : featureStack.layers()) {
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

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_FEATURE_MAP_GPU_HPP_