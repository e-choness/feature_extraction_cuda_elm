#ifndef FEATURE_ELM_CUDA_SOLVER_GPU_HPP_
#define FEATURE_ELM_CUDA_SOLVER_GPU_HPP_

#include <cstddef>
#include <vector>

#include "core/solver.hpp"

namespace feature_elm::cuda_backend {

template <typename FloatT>
[[nodiscard]] bool solveRidgeGpu(const std::vector<FloatT>& features,
                                 const std::vector<FloatT>& targets, std::size_t numSamples,
                                 std::size_t numOutputs, SolverOptions<FloatT> options,
                                 std::vector<FloatT>* weights);

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_SOLVER_GPU_HPP_
