#ifndef FEATURE_ELM_CORE_PARALLEL_HPP_
#define FEATURE_ELM_CORE_PARALLEL_HPP_

// OpenMP helpers for the CPU hot loops.
//
// FEATURE_ELM_OMP(directive) expands to `#pragma omp directive` when the library is built with
// OpenMP (-DFEATURE_ELM_OPENMP=ON, the default) and to nothing otherwise, so the code compiles
// warning-free either way.
//
// Every parallel loop in the library partitions *output elements* between threads and keeps each
// element's summation order unchanged, so results are bit-identical to a single-threaded run and do
// not depend on the thread count. Use OMP_NUM_THREADS to control the number of threads.

#include <cstddef>

#if defined(_OPENMP)
// Declared here instead of including <omp.h>: GCC's omp.h does not parse under clang-tidy, and
// this is the only runtime call the library needs (a stable C ABI in every OpenMP runtime).
extern "C" int omp_get_max_threads() noexcept;  // NOLINT(readability-identifier-naming)
#  define FEATURE_ELM_OMP(directive) _Pragma(#directive)
#else
#  define FEATURE_ELM_OMP(directive)
#endif

namespace feature_elm {

/// Loops with less work than this (multiply-adds, ~2-4 ms serially) stay serial. Waking a sleeping
/// OpenMP team costs up to a millisecond on some systems (hybrid CPUs, WSL2), so small loops lose.
inline constexpr std::size_t kParallelWorkThreshold = std::size_t{1} << 22;

/// Number of threads a parallel region will use (1 without OpenMP).
[[nodiscard]] inline int parallelThreads() noexcept {
#if defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_PARALLEL_HPP_
