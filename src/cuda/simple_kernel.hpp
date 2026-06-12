#ifndef FEATURE_ELM_CUDA_SIMPLE_KERNEL_HPP_
#define FEATURE_ELM_CUDA_SIMPLE_KERNEL_HPP_

#include <cstddef>

namespace feature_elm::cuda_backend {

/**
 * @brief Host-callable function for vector addition on GPU.
 *
 * Computes: c = a + b for three vectors on device.
 * All inputs must be allocated on GPU using DeviceBuffer or similar.
 *
 * @param aDevice Device pointer to first vector (input)
 * @param bDevice Device pointer to second vector (input)
 * @param cDevice Device pointer to output vector (output)
 * @param numElements Number of elements in each vector
 * @return true on success, false on CUDA error
 */
template <typename T>
[[nodiscard]] bool vectorAddGpu(const T* aDevice, const T* bDevice, T* cDevice,
                                std::size_t numElements);

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_SIMPLE_KERNEL_HPP_
