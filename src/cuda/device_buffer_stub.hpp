#ifndef FEATURE_ELM_CUDA_DEVICE_BUFFER_STUB_HPP_
#define FEATURE_ELM_CUDA_DEVICE_BUFFER_STUB_HPP_

#include <cstddef>

namespace feature_elm::cuda_backend {

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t /*numElements*/) {}
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept {
    size_ = other.size_;
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

}  // namespace feature_elm::cuda_backend

#endif  // FEATURE_ELM_CUDA_DEVICE_BUFFER_STUB_HPP_