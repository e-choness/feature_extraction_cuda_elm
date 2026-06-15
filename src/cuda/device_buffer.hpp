#ifndef FEATURE_ELM_CUDA_DEVICE_BUFFER_HPP_
#define FEATURE_ELM_CUDA_DEVICE_BUFFER_HPP_

#include <cstddef>

#ifdef __CUDACC__
#  include <cuda_runtime.h>
#endif

namespace feature_elm::cuda_backend {

#ifdef __CUDACC__

/**
 * @class DeviceBuffer
 * @brief RAII wrapper for GPU device memory allocation and management.
 *
 * Automatically allocates memory on construction and frees on destruction.
 * Provides safe copy and move semantics.
 *
 * Template Parameters:
 * - T: Element type (e.g., float, double)
 */
template <typename T>
class DeviceBuffer {
 public:
  /**
   * @brief Allocate device memory for numElements elements.
   *
   * @param numElements Number of elements to allocate
   */
  explicit DeviceBuffer(std::size_t numElements);

  // No copy semantics; device memory is not easily copyable
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Move semantics
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  ~DeviceBuffer();

  /**
   * @brief Get the device pointer.
   *
   * @return Pointer to device memory, or nullptr if allocation failed
   */
  [[nodiscard]] T* data() noexcept {
    return ptr_;
  }
  [[nodiscard]] const T* data() const noexcept {
    return ptr_;
  }

  /**
   * @brief Get the number of elements.
   *
   * @return Number of elements allocated
   */
  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }

  /**
   * @brief Check if buffer is allocated (non-null).
   *
   * @return true if allocation is valid, false otherwise
   */
  [[nodiscard]] bool isValid() const noexcept {
    return ptr_ != nullptr;
  }

  /**
   * @brief Copy data from host to device.
   *
   * @param hostPtr Source pointer on host
   * @param count Number of elements to copy (must be <= size())
   * @return true on success, false on error
   */
  [[nodiscard]] bool copyFromHost(const T* hostPtr, std::size_t count);

  /**
   * @brief Copy data from device to host.
   *
   * @param hostPtr Destination pointer on host
   * @param count Number of elements to copy (must be <= size())
   * @return true on success, false on error
   */
  [[nodiscard]] bool copyToHost(T* hostPtr, std::size_t count) const;

 private:
  T* ptr_;
  std::size_t size_;

  /**
   * @brief Free device memory.
   */
  void deallocate() noexcept;
};

#else  // !__CUDACC__

// Stub implementation for CPU-only builds
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t numElements) : size_(numElements) {}
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : size_(other.size_) {
    other.size_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    size_ = other.size_;
    other.size_ = 0;
    return *this;
  }
  ~DeviceBuffer() = default;

  [[nodiscard]] T* data() noexcept {
    return nullptr;
  }
  [[nodiscard]] const T* data() const noexcept {
    return nullptr;
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return size_;
  }
  [[nodiscard]] bool isValid() const noexcept {
    return false;
  }
  [[nodiscard]] bool copyFromHost(const T* /*hostPtr*/, std::size_t /*count*/) {
    return false;
  }
  [[nodiscard]] bool copyToHost(T* /*hostPtr*/, std::size_t /*count*/) const {
    return false;
  }

 private:
  std::size_t size_ = 0;
};

#endif  // __CUDACC__

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_DEVICE_BUFFER_HPP_