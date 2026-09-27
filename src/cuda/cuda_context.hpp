#ifndef FEATURE_ELM_CUDA_CUDA_CONTEXT_HPP_
#define FEATURE_ELM_CUDA_CUDA_CONTEXT_HPP_

// Internal helpers shared by the .cu translation units. Not part of the public API and must only
// be included from files compiled by nvcc.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>
#include <limits>
#include <mutex>

#define FEATURE_ELM_CUDA_CHECK(expr) \
  do {                               \
    if ((expr) != cudaSuccess) {     \
      return false;                  \
    }                                \
  } while (0)

#define FEATURE_ELM_CUBLAS_CHECK(expr)     \
  do {                                     \
    if ((expr) != CUBLAS_STATUS_SUCCESS) { \
      return false;                        \
    }                                      \
  } while (0)

#define FEATURE_ELM_CUSOLVER_CHECK(expr)     \
  do {                                       \
    if ((expr) != CUSOLVER_STATUS_SUCCESS) { \
      return false;                          \
    }                                        \
  } while (0)

namespace feature_elm::cuda_backend::detail {

/// Library handles are expensive to create (they allocate device workspace), so the process keeps
/// one of each. cuBLAS/cuSOLVER handles must not be used from two threads at once, so callers hold
/// lock() for the duration of a GPU operation. The instance is deliberately never destroyed:
/// tearing handles down during static destruction can race the CUDA driver's own shutdown.
class Handles {
 public:
  Handles(const Handles&) = delete;
  Handles& operator=(const Handles&) = delete;

  [[nodiscard]] cublasHandle_t cublas() const noexcept {
    return cublas_;
  }
  [[nodiscard]] cusolverDnHandle_t cusolver() const noexcept {
    return cusolver_;
  }
  [[nodiscard]] std::unique_lock<std::mutex> lock() {
    return std::unique_lock<std::mutex>(mutex_);
  }

  static Handles& get() {
    static Handles* const instance = new Handles();
    return *instance;
  }

 private:
  Handles() {
    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
      cublas_ = nullptr;
    }
    if (cusolverDnCreate(&cusolver_) != CUSOLVER_STATUS_SUCCESS) {
      cusolver_ = nullptr;
    }
  }
  ~Handles() = default;

  cublasHandle_t cublas_ = nullptr;
  cusolverDnHandle_t cusolver_ = nullptr;
  std::mutex mutex_;
};

[[nodiscard]] inline bool fitsInt(std::size_t value) noexcept {
  return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

}  // namespace feature_elm::cuda_backend::detail

#endif  // FEATURE_ELM_CUDA_CUDA_CONTEXT_HPP_
