#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "cuda/device_buffer.hpp"
#include "cuda/gpu_ops.hpp"

namespace feature_elm::cuda_backend {

namespace {

#define CUDA_CHECK(expr)      \
  do {                        \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
      return false;           \
    }                         \
  } while (0)

#define CUBLAS_CHECK(expr)                 \
  do {                                     \
    cublasStatus_t status = (expr);        \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      return false;                        \
    }                                      \
  } while (0)

class CublasHandle {
 public:
  CublasHandle() {
    (void)cublasCreate(&handle_);
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  ~CublasHandle() {
    if (handle_ != nullptr) {
      (void)cublasDestroy(handle_);
    }
  }

  [[nodiscard]] bool isValid() const noexcept {
    return handle_ != nullptr;
  }

  [[nodiscard]] cublasHandle_t get() const noexcept {
    return handle_;
  }

 private:
  cublasHandle_t handle_ = nullptr;
};

template <typename FloatT>
__host__ __device__ FloatT activateDevice(FloatT value, ActivationKind activation) noexcept {
  // CUDA math functions (exp, tanh) are overloaded intrinsics for both float and double
  switch (activation) {
    case ActivationKind::kSigmoid:
      if (value > FloatT(0)) {
        return FloatT(1) / (FloatT(1) + exp(-value));
      }
      return exp(value) / (FloatT(1) + exp(value));
    case ActivationKind::kTanh:
      return tanh(value);
    case ActivationKind::kRelu:
      return value > FloatT(0) ? value : FloatT(0);
  }
  return value;
}

template <typename FloatT>
__global__ void addBiasActivateKernel(const FloatT* matrix, const FloatT* biases, FloatT* output,
                                      std::size_t rows, std::size_t cols,
                                      ActivationKind activation) {
  // Matrix stored in column-major format for cuBLAS compatibility
  const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }

  for (std::size_t row = 0; row < rows; ++row) {
    // Access pattern: matrix[row + col * rows] reads column 'col', row 'row' in column-major
    output[col * rows + row] = activateDevice(matrix[row + col * rows] + biases[row], activation);
  }
}

template <typename FloatT>
std::vector<FloatT> rowMajorToColumnMajor(const std::vector<FloatT>& rowMajor, std::size_t rows,
                                          std::size_t cols) {
  // Convert row-major matrix to column-major for cuBLAS compatibility.
  // Input: rowMajor[row * cols + col], Output: columnMajor[col * rows + row]
  std::vector<FloatT> columnMajor(rows * cols);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      columnMajor[col * rows + row] = rowMajor[row * cols + col];
    }
  }
  return columnMajor;
}

template <typename FloatT>
struct CublasTraits;

template <>
struct CublasTraits<float> {
  static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k, const float* alpha,
                             const float* A, int lda, const float* B, int ldb, const float* beta,
                             float* C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

template <>
struct CublasTraits<double> {
  static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k, const double* alpha,
                             const double* A, int lda, const double* B, int ldb, const double* beta,
                             double* C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

bool checkedDimensions(std::size_t rows, std::size_t cols) noexcept {
  return rows <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         cols <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

}  // namespace

[[nodiscard]] bool isGpuAvailable() noexcept {
  int deviceCount = 0;
  const cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return err == cudaSuccess && deviceCount > 0;
}

template <typename FloatT>
bool transformRandomAdditiveGpu(const std::vector<FloatT>& input, std::size_t numSamples,
                                std::size_t numInputs, std::size_t numHiddenNodes,
                                const std::vector<FloatT>& weights,
                                const std::vector<FloatT>& biases, ActivationKind activation,
                                std::vector<FloatT>* hiddenOutput) {
  if (!isGpuAvailable() || input.empty() || weights.empty() || biases.empty() ||
      hiddenOutput == nullptr || numSamples == 0 || numInputs == 0 || numHiddenNodes == 0) {
    return false;
  }
  if (input.size() != numSamples * numInputs || weights.size() != numInputs * numHiddenNodes ||
      biases.size() != numHiddenNodes || !checkedDimensions(numInputs, numSamples) ||
      !checkedDimensions(numHiddenNodes, numInputs)) {
    return false;
  }

  const std::vector<FloatT> inputColumnMajor = rowMajorToColumnMajor(input, numSamples, numInputs);
  const std::vector<FloatT> weightsColumnMajor =
      rowMajorToColumnMajor(weights, numHiddenNodes, numInputs);

  DeviceBuffer<FloatT> devInput(inputColumnMajor.size());
  DeviceBuffer<FloatT> devWeights(weightsColumnMajor.size());
  DeviceBuffer<FloatT> devBiases(biases.size());
  DeviceBuffer<FloatT> devHiddenOutput(numSamples * numHiddenNodes);

  if (!devInput.isValid() || !devWeights.isValid() || !devBiases.isValid() ||
      !devHiddenOutput.isValid()) {
    return false;
  }

  if (!devInput.copyFromHost(inputColumnMajor.data(), inputColumnMajor.size()) ||
      !devWeights.copyFromHost(weightsColumnMajor.data(), weightsColumnMajor.size()) ||
      !devBiases.copyFromHost(biases.data(), biases.size())) {
    return false;
  }

  CublasHandle cublas;
  if (!cublas.isValid()) {
    return false;
  }

  const FloatT alpha = FloatT(1);
  const FloatT beta = FloatT(0);
  CUBLAS_CHECK(CublasTraits<FloatT>::gemm(
      cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(numHiddenNodes),
      static_cast<int>(numSamples), static_cast<int>(numInputs), &alpha, devWeights.data(),
      static_cast<int>(numHiddenNodes), devInput.data(), static_cast<int>(numInputs), &beta,
      devHiddenOutput.data(), static_cast<int>(numHiddenNodes)));

  // Block size of 128 threads is a CUDA occupancy sweet spot for most GPU architectures.
  const std::size_t blockSize = 128;
  // gridSize > 0 is guaranteed: numSamples > 0 is validated earlier (line 144).
  const std::size_t gridSize = (numSamples + blockSize - 1) / blockSize;
  addBiasActivateKernel<FloatT><<<gridSize, blockSize>>>(devHiddenOutput.data(), devBiases.data(),
                                                         devHiddenOutput.data(), numHiddenNodes,
                                                         numSamples, activation);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  hiddenOutput->assign(numSamples * numHiddenNodes, FloatT(0));
  return devHiddenOutput.copyToHost(hiddenOutput->data(), hiddenOutput->size());
}

template <typename FloatT>
bool transformElmAutoEncoderGpu(const std::vector<FloatT>& input, std::size_t numSamples,
                                std::size_t numInputs, std::size_t numHiddenNodes,
                                const std::vector<FloatT>& encoderWeights,
                                const std::vector<FloatT>& encoderBiases, ActivationKind activation,
                                std::vector<FloatT>* hiddenOutput) {
  if (!isGpuAvailable() || input.empty() || encoderWeights.empty() || encoderBiases.empty() ||
      hiddenOutput == nullptr || numSamples == 0 || numInputs == 0 || numHiddenNodes == 0) {
    return false;
  }
  if (input.size() != numSamples * numInputs ||
      encoderWeights.size() != numInputs * numHiddenNodes ||
      encoderBiases.size() != numHiddenNodes || !checkedDimensions(numInputs, numSamples) ||
      !checkedDimensions(numHiddenNodes, numInputs)) {
    return false;
  }

  const std::vector<FloatT> inputColumnMajor = rowMajorToColumnMajor(input, numSamples, numInputs);
  const std::vector<FloatT> weightsColumnMajor =
      rowMajorToColumnMajor(encoderWeights, numHiddenNodes, numInputs);

  DeviceBuffer<FloatT> devInput(inputColumnMajor.size());
  DeviceBuffer<FloatT> devWeights(weightsColumnMajor.size());
  DeviceBuffer<FloatT> devBiases(encoderBiases.size());
  DeviceBuffer<FloatT> devHiddenOutput(numSamples * numHiddenNodes);

  if (!devInput.isValid() || !devWeights.isValid() || !devBiases.isValid() ||
      !devHiddenOutput.isValid()) {
    return false;
  }

  if (!devInput.copyFromHost(inputColumnMajor.data(), inputColumnMajor.size()) ||
      !devWeights.copyFromHost(weightsColumnMajor.data(), weightsColumnMajor.size()) ||
      !devBiases.copyFromHost(encoderBiases.data(), encoderBiases.size())) {
    return false;
  }

  CublasHandle cublas;
  if (!cublas.isValid()) {
    return false;
  }

  const FloatT alpha = FloatT(1);
  const FloatT beta = FloatT(0);
  CUBLAS_CHECK(CublasTraits<FloatT>::gemm(
      cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(numHiddenNodes),
      static_cast<int>(numSamples), static_cast<int>(numInputs), &alpha, devWeights.data(),
      static_cast<int>(numHiddenNodes), devInput.data(), static_cast<int>(numInputs), &beta,
      devHiddenOutput.data(), static_cast<int>(numHiddenNodes)));

  // Block size of 128 threads is a CUDA occupancy sweet spot for most GPU architectures.
  const std::size_t blockSize = 128;
  // gridSize > 0 is guaranteed: numSamples > 0 is validated earlier (line 206).
  const std::size_t gridSize = (numSamples + blockSize - 1) / blockSize;
  addBiasActivateKernel<FloatT><<<gridSize, blockSize>>>(devHiddenOutput.data(), devBiases.data(),
                                                         devHiddenOutput.data(), numHiddenNodes,
                                                         numSamples, activation);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  hiddenOutput->assign(numSamples * numHiddenNodes, FloatT(0));
  return devHiddenOutput.copyToHost(hiddenOutput->data(), hiddenOutput->size());
}

template bool transformRandomAdditiveGpu<float>(const std::vector<float>&, std::size_t, std::size_t,
                                                std::size_t, const std::vector<float>&,
                                                const std::vector<float>&, ActivationKind,
                                                std::vector<float>*);

template bool transformRandomAdditiveGpu<double>(const std::vector<double>&, std::size_t,
                                                 std::size_t, std::size_t,
                                                 const std::vector<double>&,
                                                 const std::vector<double>&, ActivationKind,
                                                 std::vector<double>*);

template bool transformElmAutoEncoderGpu<float>(const std::vector<float>&, std::size_t, std::size_t,
                                                std::size_t, const std::vector<float>&,
                                                const std::vector<float>&, ActivationKind,
                                                std::vector<float>*);

template bool transformElmAutoEncoderGpu<double>(const std::vector<double>&, std::size_t,
                                                 std::size_t, std::size_t,
                                                 const std::vector<double>&,
                                                 const std::vector<double>&, ActivationKind,
                                                 std::vector<double>*);

#undef CUBLAS_CHECK
#undef CUDA_CHECK

}  // namespace feature_elm::cuda_backend