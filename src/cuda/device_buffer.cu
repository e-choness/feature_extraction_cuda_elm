#include <utility>

#include "cuda/device_buffer.hpp"

namespace feature_elm::cuda_backend {

// Helper macro for CUDA error checking
#define CUDA_CHECK(expr)      \
  do {                        \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
      return false;           \
    }                         \
  } while (0)

#define CUDA_CHECK_VOID(expr) \
  do {                        \
    cudaError_t err = (expr); \
    (void)err;                \
  } while (0)

template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::size_t numElements) : ptr_(nullptr), size_(numElements) {
  if (numElements == 0) {
    return;
  }

  cudaError_t err = cudaMalloc(&ptr_, numElements * sizeof(T));
  if (err != cudaSuccess) {
    ptr_ = nullptr;
  }
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
  other.ptr_ = nullptr;
  other.size_ = 0;
}

template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& other) noexcept {
  deallocate();
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.ptr_ = nullptr;
  other.size_ = 0;
  return *this;
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer() {
  deallocate();
}

template <typename T>
void DeviceBuffer<T>::deallocate() noexcept {
  if (ptr_ != nullptr) {
    CUDA_CHECK_VOID(cudaFree(ptr_));
    ptr_ = nullptr;
  }
  size_ = 0;
}

template <typename T>
bool DeviceBuffer<T>::copyFromHost(const T* hostPtr, std::size_t count) {
  if (!isValid() || count > size_ || hostPtr == nullptr) {
    return false;
  }

  CUDA_CHECK(cudaMemcpy(ptr_, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
  return true;
}

template <typename T>
bool DeviceBuffer<T>::copyToHost(T* hostPtr, std::size_t count) const {
  if (!isValid() || count > size_ || hostPtr == nullptr) {
    return false;
  }

  CUDA_CHECK(cudaMemcpy(hostPtr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  return true;
}

// Explicit template instantiations
template class DeviceBuffer<float>;
template class DeviceBuffer<double>;
template class DeviceBuffer<int>;

#undef CUDA_CHECK
#undef CUDA_CHECK_VOID

}  // namespace feature_elm::cuda_backend
