#include <cuda_runtime.h>

#include "cuda/simple_kernel.hpp"

namespace feature_elm::cuda_backend {

// Helper macro for CUDA error checking
#define CUDA_CHECK(expr)      \
  do {                        \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
      return false;           \
    }                         \
  } while (0)

/**
 * @brief CUDA kernel: element-wise vector addition.
 *
 * Each thread computes one element: c[i] = a[i] + b[i]
 */
template <typename T>
__global__ void vectorAddKernel(const T* a, const T* b, T* c, std::size_t numElements) {
  std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < numElements) {
    c[idx] = a[idx] + b[idx];
  }
}

template <typename T>
bool vectorAddGpu(const T* aDevice, const T* bDevice, T* cDevice, std::size_t numElements) {
  if (aDevice == nullptr || bDevice == nullptr || cDevice == nullptr || numElements == 0) {
    return false;
  }

  // Simple block/grid configuration: 256 threads per block
  std::size_t blockSize = 256;
  std::size_t gridSize = (numElements + blockSize - 1) / blockSize;

  vectorAddKernel<<<gridSize, blockSize>>>(aDevice, bDevice, cDevice, numElements);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());

  // Synchronize device
  CUDA_CHECK(cudaDeviceSynchronize());

  return true;
}

// Explicit template instantiations
template bool vectorAddGpu<float>(const float* aDevice, const float* bDevice, float* cDevice,
                                  std::size_t numElements);
template bool vectorAddGpu<double>(const double* aDevice, const double* bDevice, double* cDevice,
                                   std::size_t numElements);

#undef CUDA_CHECK

}  // namespace feature_elm::cuda_backend
