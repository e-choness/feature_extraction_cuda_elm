#ifndef FEATURE_ELM_CUDA_RLS_GPU_HPP_
#define FEATURE_ELM_CUDA_RLS_GPU_HPP_

// Device-resident recursive least squares for OS-ELM and ReOS-ELM, in information form.
//
// Instead of propagating the covariance P, the GPU accumulates the information matrix and vector
//
//     A = reg * I + sum_k H_k^T H_k,        c = sum_k H_k^T T_k
//
// with GEMMs that only ever add (no cancellation), and solves A beta = c by Cholesky when the
// weights are read. With P0 = I / reg this equals sequential RLS in exact arithmetic (P = A^-1).
// The covariance form (P -= P H^T S^-1 H P) cancels catastrophically in float32, and even A must
// be accumulated in float64: with a small ridge and rank-deficient hidden features, float32
// rounding in sum H^T H exceeds the ridge and leaves A indefinite. Inputs and outputs stay FloatT.
//
// RlsSolver uses this only when the forgetting factor is 1 and no constraint is set; FOS-ELM and
// OS-CELM apply their terms per sample and stay on the CPU.

#include <cstddef>
#include <memory>
#include <vector>

namespace feature_elm::cuda_backend {

template <typename FloatT>
struct RlsGpuState;

template <typename FloatT>
struct RlsGpuStateDeleter {
  void operator()(RlsGpuState<FloatT>* state) const noexcept;
};

template <typename FloatT>
using RlsGpuStatePtr = std::unique_ptr<RlsGpuState<FloatT>, RlsGpuStateDeleter<FloatT>>;

/// A = reg * I, c = 0. Returns nullptr without a GPU or if allocation fails.
template <typename FloatT>
[[nodiscard]] RlsGpuStatePtr<FloatT> createRlsGpu(std::size_t numFeatures, std::size_t numOutputs,
                                                  FloatT regularization);

/// Absorbs numSamples rows: features row-major (numSamples x numFeatures), targets row-major
/// (numSamples x numOutputs). Large chunks are processed in sub-blocks.
template <typename FloatT>
[[nodiscard]] bool updateRlsGpu(RlsGpuState<FloatT>& state, const FloatT* features,
                                std::size_t numSamples, const FloatT* targets);

/// Solves A beta = c and copies beta to the host, row-major (numFeatures x numOutputs).
template <typename FloatT>
[[nodiscard]] bool downloadRlsWeights(RlsGpuState<FloatT>& state, std::vector<FloatT>* weights);

/// Computes P = A^-1 and copies it to the host (numFeatures x numFeatures). Diagnostic; O(F^3).
template <typename FloatT>
[[nodiscard]] bool downloadRlsCovariance(RlsGpuState<FloatT>& state,
                                         std::vector<FloatT>* covariance);

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_RLS_GPU_HPP_
