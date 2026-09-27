#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "cuda/device_buffer.hpp"

namespace {

using namespace feature_elm::cuda_backend;

// Check if GPU is available
bool isGpuAvailable() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  return deviceCount > 0;
}

// Skip macro for GPU tests
#define SKIP_IF_NO_GPU                                      \
  if (!isGpuAvailable()) {                                  \
    GTEST_SKIP() << "GPU not available, skipping GPU test"; \
  }

// Test 1: Device buffer allocation
TEST(CudaInfraTest, DeviceBufferAllocation) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size = 1024;
  DeviceBuffer<float> buf(size);

  EXPECT_TRUE(buf.isValid());
  EXPECT_EQ(buf.size(), size);
  EXPECT_NE(buf.data(), nullptr);
}

// Test 2: Device buffer copy host to device
TEST(CudaInfraTest, DeviceBufferCopyHostToDevice) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size = 100;
  DeviceBuffer<float> devBuf(size);

  std::vector<float> hostData(size);
  for (std::size_t i = 0; i < size; ++i) {
    hostData[i] = static_cast<float>(i);
  }

  ASSERT_TRUE(devBuf.isValid());
  EXPECT_TRUE(devBuf.copyFromHost(hostData.data(), size));
}

// Test 3: Device buffer copy device to host
TEST(CudaInfraTest, DeviceBufferCopyDeviceToHost) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size = 100;
  DeviceBuffer<float> devBuf(size);

  std::vector<float> hostDataIn(size);
  std::vector<float> hostDataOut(size, 0.0f);

  for (std::size_t i = 0; i < size; ++i) {
    hostDataIn[i] = static_cast<float>(i);
  }

  ASSERT_TRUE(devBuf.isValid());
  EXPECT_TRUE(devBuf.copyFromHost(hostDataIn.data(), size));
  EXPECT_TRUE(devBuf.copyToHost(hostDataOut.data(), size));

  // Verify data integrity
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(hostDataOut[i], hostDataIn[i]);
  }
}

// Test 4: Device buffer move semantics
TEST(CudaInfraTest, DeviceBufferMoveSemantics) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size = 100;

  DeviceBuffer<float> buf1(size);
  ASSERT_TRUE(buf1.isValid());

  // Move construction
  DeviceBuffer<float> buf2(std::move(buf1));
  EXPECT_EQ(buf2.size(), size);
  EXPECT_TRUE(buf2.isValid());

  // Original should be invalidated
  EXPECT_EQ(buf1.size(), 0);
  EXPECT_FALSE(buf1.isValid());
}

// Test 5: Device buffer move assignment
TEST(CudaInfraTest, DeviceBufferMoveAssignment) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size1 = 100;
  constexpr std::size_t size2 = 200;

  DeviceBuffer<float> buf1(size1);
  DeviceBuffer<float> buf2(size2);

  EXPECT_TRUE(buf1.isValid());
  EXPECT_TRUE(buf2.isValid());
  EXPECT_EQ(buf1.size(), size1);
  EXPECT_EQ(buf2.size(), size2);

  buf1 = std::move(buf2);

  EXPECT_EQ(buf1.size(), size2);
  EXPECT_TRUE(buf1.isValid());
  EXPECT_EQ(buf2.size(), 0);
  EXPECT_FALSE(buf2.isValid());
}

// Test 6: copies larger than the allocation are rejected instead of overrunning device memory
TEST(CudaInfraTest, DeviceBufferRejectsOversizedCopy) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size = 16;
  DeviceBuffer<float> buf(size);
  std::vector<float> host(size * 2, 1.0f);

  ASSERT_TRUE(buf.isValid());
  EXPECT_FALSE(buf.copyFromHost(host.data(), host.size()));
  EXPECT_FALSE(buf.copyToHost(host.data(), host.size()));
  EXPECT_FALSE(buf.copyFromHost(nullptr, size));
}

}  // namespace
