#ifndef FEATURE_ELM_CORE_DENSE_HPP_
#define FEATURE_ELM_CORE_DENSE_HPP_

// Shared CPU dense layer: out = act(input * weights + bias), row-major throughout.
//
// Used by the random/auto-encoder feature maps and by every model's batch prediction. Each output
// row is accumulated over contiguous weight rows (cache friendly), samples run in parallel, and
// every output element starts from its bias (or zero) and adds k = 0, 1, ... in order, so results
// are bit-identical to the textbook triple loop regardless of thread count.

#include <algorithm>
#include <cstddef>

#include "core/parallel.hpp"

namespace feature_elm {

struct IdentityActivation {
  template <typename FloatT>
  FloatT operator()(FloatT value) const noexcept {
    return value;
  }
};

/// input: samples x inDim, weights: inDim x outDim, biases: outDim or nullptr, output: samples x
/// outDim.
template <typename FloatT, typename Activation = IdentityActivation>
void denseForward(const FloatT* input, std::size_t numSamples, std::size_t inDim,
                  const FloatT* weights, const FloatT* biases, std::size_t outDim, FloatT* output,
                  Activation activation = {}) {
  FEATURE_ELM_OMP(omp parallel for schedule(static)
                      if (numSamples * inDim * outDim > kParallelWorkThreshold))
  for (std::size_t i = 0; i < numSamples; ++i) {
    FloatT* out = output + i * outDim;
    if (biases != nullptr) {
      std::copy(biases, biases + outDim, out);
    } else {
      std::fill(out, out + outDim, FloatT(0));
    }
    const FloatT* in = input + i * inDim;
    for (std::size_t k = 0; k < inDim; ++k) {
      const FloatT x = in[k];
      const FloatT* w = weights + k * outDim;
      for (std::size_t j = 0; j < outDim; ++j) {
        out[j] += x * w[j];
      }
    }
    for (std::size_t j = 0; j < outDim; ++j) {
      out[j] = activation(out[j]);
    }
  }
}

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_DENSE_HPP_
