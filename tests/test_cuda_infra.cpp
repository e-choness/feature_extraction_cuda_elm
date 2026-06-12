#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "cuda/device_buffer.hpp"
#include "cuda/simple_kernel.hpp"

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

// Test 6: Simple vector add kernel
TEST(CudaInfraTest, VectorAddKernel) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size = 1000;

  DeviceBuffer<float> devA(size);
  DeviceBuffer<float> devB(size);
  DeviceBuffer<float> devC(size);

  // Prepare input data
  std::vector<float> hostA(size);
  std::vector<float> hostB(size);
  std::vector<float> hostC(size, 0.0f);

  for (std::size_t i = 0; i < size; ++i) {
    hostA[i] = static_cast<float>(i);
    hostB[i] = static_cast<float>(i * 2);
  }

  // Copy to device
  ASSERT_TRUE(devA.copyFromHost(hostA.data(), size));
  ASSERT_TRUE(devB.copyFromHost(hostB.data(), size));

  // Run kernel
  ASSERT_TRUE(vectorAddGpu(devA.data(), devB.data(), devC.data(), size));

  // Copy result back to host
  ASSERT_TRUE(devC.copyToHost(hostC.data(), size));

  // Verify results
  for (std::size_t i = 0; i < size; ++i) {
    float expected = hostA[i] + hostB[i];
    EXPECT_FLOAT_EQ(hostC[i], expected);
  }
}

// Test 7: Vector add with double precision
TEST(CudaInfraTest, VectorAddKernelDouble) {
  SKIP_IF_NO_GPU;

  constexpr std::size_t size = 500;

  DeviceBuffer<double> devA(size);
  DeviceBuffer<double> devB(size);
  DeviceBuffer<double> devC(size);

  std::vector<double> hostA(size);
  std::vector<double> hostB(size);
  std::vector<double> hostC(size, 0.0);

  for (std::size_t i = 0; i < size; ++i) {
    hostA[i] = static_cast<double>(i) * 0.5;
    hostB[i] = static_cast<double>(i) * 1.5;
  }

  ASSERT_TRUE(devA.copyFromHost(hostA.data(), size));
  ASSERT_TRUE(devB.copyFromHost(hostB.data(), size));
  ASSERT_TRUE(vectorAddGpu(devA.data(), devB.data(), devC.data(), size));
  ASSERT_TRUE(devC.copyToHost(hostC.data(), size));

  for (std::size_t i = 0; i < size; ++i) {
    double expected = hostA[i] + hostB[i];
    EXPECT_DOUBLE_EQ(hostC[i], expected);
  }
}

// Test 8: CPU reference implementation for verification
TEST(CudaInfraTest, VectorAddCpuReference) {
  // This test runs on CPU only, verifying the logic we'll use for GPU

  constexpr std::size_t size = 100;

  std::vector<float> a(size);
  std::vector<float> b(size);
  std::vector<float> c(size, 0.0f);

  for (std::size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i + 1);
  }

  // Simple CPU addition
  for (std::size_t i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }

  // Verify
  for (std::size_t i = 0; i < size; ++i) {
    float expected = static_cast<float>(i) + static_cast<float>(i + 1);
    EXPECT_FLOAT_EQ(c[i], expected);
  }
}

}  // namespace
