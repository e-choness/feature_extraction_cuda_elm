#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda/device_buffer.hpp"
#include "cuda/elm_gpu.hpp"

namespace feature_elm::cuda_backend {

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

#define CUSOLVER_CHECK(expr)                 \
  do {                                       \
    cusolverStatus_t status = (expr);        \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
      return false;                          \
    }                                        \
  } while (0)

__device__ float activateDevice(float x, feature_elm::ActivationFunction activation) {
  if (activation == feature_elm::ActivationFunction::kSigmoid) {
    if (x > 0.0f) {
      return 1.0f / (1.0f + expf(-x));
    }
    float expX = expf(x);
    return expX / (1.0f + expX);
  }
  return expf(-x * x);
}

__device__ double activateDevice(double x, feature_elm::ActivationFunction activation) {
  if (activation == feature_elm::ActivationFunction::kSigmoid) {
    if (x > 0.0) {
      return 1.0 / (1.0 + exp(-x));
    }
    double expX = exp(x);
    return expX / (1.0 + expX);
  }
  return exp(-x * x);
}

template <typename FloatT>
__global__ void hiddenOutputKernel(const FloatT* input, const FloatT* weights, const FloatT* biases,
                                   FloatT* hiddenOutput, std::size_t numSamples,
                                   std::size_t numInputs, std::size_t numHiddenNodes,
                                   feature_elm::ActivationFunction activation) {
  std::size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= numSamples) {
    return;
  }

  for (std::size_t hiddenIndex = 0; hiddenIndex < numHiddenNodes; ++hiddenIndex) {
    FloatT sum = biases[hiddenIndex];
    for (std::size_t inputIndex = 0; inputIndex < numInputs; ++inputIndex) {
      sum += input[sample * numInputs + inputIndex] *
             weights[inputIndex * numHiddenNodes + hiddenIndex];
    }
    hiddenOutput[hiddenIndex * numSamples + sample] = activateDevice(sum, activation);
  }
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

template <typename FloatT>
struct CusolverTraits;

template <>
struct CusolverTraits<float> {
  static cusolverStatus_t potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A,
                                int lda, float* work, int lwork, int* devInfo) {
    return cusolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, devInfo);
  }

  static cusolverStatus_t potrfBufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                          float* A, int lda, int* lwork) {
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
  }

  static cusolverStatus_t potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
                                const float* A, int lda, float* B, int ldb, int* devInfo) {
    return cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
  }
};

template <>
struct CusolverTraits<double> {
  static cusolverStatus_t potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A,
                                int lda, double* work, int lwork, int* devInfo) {
    return cusolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, devInfo);
  }

  static cusolverStatus_t potrfBufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                          double* A, int lda, int* lwork) {
    return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
  }

  static cusolverStatus_t potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
                                const double* A, int lda, double* B, int ldb, int* devInfo) {
    return cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
  }
};

template <typename FloatT>
[[nodiscard]] bool computeHiddenOutputDevice(const std::vector<FloatT>& input,
                                             std::size_t numSamples, std::size_t numInputs,
                                             std::size_t numHiddenNodes,
                                             const std::vector<FloatT>& hiddenWeights,
                                             const std::vector<FloatT>& hiddenBiases,
                                             feature_elm::ActivationFunction activation,
                                             std::vector<FloatT>* hiddenOutput) {
  if (input.empty() || hiddenWeights.empty() || hiddenBiases.empty() || hiddenOutput == nullptr) {
    return false;
  }

  DeviceBuffer<FloatT> devInput(input.size());
  DeviceBuffer<FloatT> devWeights(hiddenWeights.size());
  DeviceBuffer<FloatT> devBiases(hiddenBiases.size());
  DeviceBuffer<FloatT> devHiddenOutput(numSamples * numHiddenNodes);

  if (!devInput.isValid() || !devWeights.isValid() || !devBiases.isValid() ||
      !devHiddenOutput.isValid()) {
    return false;
  }

  if (!devInput.copyFromHost(input.data(), input.size()) ||
      !devWeights.copyFromHost(hiddenWeights.data(), hiddenWeights.size()) ||
      !devBiases.copyFromHost(hiddenBiases.data(), hiddenBiases.size())) {
    return false;
  }

  std::size_t threads = 128;
  std::size_t blocks = (numSamples + threads - 1) / threads;
  hiddenOutputKernel<<<blocks, threads>>>(devInput.data(), devWeights.data(), devBiases.data(),
                                          devHiddenOutput.data(), numSamples, numInputs,
                                          numHiddenNodes, activation);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  hiddenOutput->assign(numSamples * numHiddenNodes, FloatT(0));
  if (!devHiddenOutput.copyToHost(hiddenOutput->data(), hiddenOutput->size())) {
    return false;
  }

  return true;
}

template <typename FloatT>
[[nodiscard]] bool solveLeastSquaresDevice(const std::vector<FloatT>& hiddenOutput,
                                           const std::vector<FloatT>& trainTargets,
                                           std::size_t numSamples, std::size_t numHiddenNodes,
                                           std::size_t numOutputs,
                                           std::vector<FloatT>* outputWeights) {
  if (hiddenOutput.empty() || trainTargets.empty() || outputWeights == nullptr) {
    return false;
  }

  std::vector<FloatT> targetColumnMajor(numSamples * numOutputs);
  for (std::size_t row = 0; row < numSamples; ++row) {
    for (std::size_t col = 0; col < numOutputs; ++col) {
      targetColumnMajor[col * numSamples + row] = trainTargets[row * numOutputs + col];
    }
  }

  DeviceBuffer<FloatT> devH(hiddenOutput.size());
  DeviceBuffer<FloatT> devT(targetColumnMajor.size());
  DeviceBuffer<FloatT> devHTH(numHiddenNodes * numHiddenNodes);
  DeviceBuffer<FloatT> devHTT(numHiddenNodes * numOutputs);
  DeviceBuffer<FloatT> devBeta(numHiddenNodes * numOutputs);

  if (!devH.isValid() || !devT.isValid() || !devHTH.isValid() || !devHTT.isValid() ||
      !devBeta.isValid()) {
    return false;
  }

  if (!devH.copyFromHost(hiddenOutput.data(), hiddenOutput.size()) ||
      !devT.copyFromHost(targetColumnMajor.data(), targetColumnMajor.size())) {
    return false;
  }

  cublasHandle_t cublasHandle = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublasHandle));

  const FloatT alpha = static_cast<FloatT>(1);
  const FloatT beta = static_cast<FloatT>(0);

  // Compute H^T * H
  CUBLAS_CHECK(CublasTraits<FloatT>::gemm(
      cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(numHiddenNodes),
      static_cast<int>(numHiddenNodes), static_cast<int>(numSamples), &alpha, devH.data(),
      static_cast<int>(numSamples), devH.data(), static_cast<int>(numSamples), &beta, devHTH.data(),
      static_cast<int>(numHiddenNodes)));

  // Compute H^T * T
  CUBLAS_CHECK(CublasTraits<FloatT>::gemm(
      cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(numHiddenNodes),
      static_cast<int>(numOutputs), static_cast<int>(numSamples), &alpha, devH.data(),
      static_cast<int>(numSamples), devT.data(), static_cast<int>(numSamples), &beta, devHTT.data(),
      static_cast<int>(numHiddenNodes)));

  cublasDestroy(cublasHandle);

  // Regularize H^T * H in-place on host to make it positive definite
  std::vector<FloatT> hostHTH(numHiddenNodes * numHiddenNodes);
  if (!devHTH.copyToHost(hostHTH.data(), hostHTH.size())) {
    return false;
  }
  const FloatT lambda = static_cast<FloatT>(1e-8);
  for (std::size_t i = 0; i < numHiddenNodes; ++i) {
    hostHTH[i * numHiddenNodes + i] += lambda;
  }
  if (!devHTH.copyFromHost(hostHTH.data(), hostHTH.size())) {
    return false;
  }

  cusolverDnHandle_t cusolverHandle = nullptr;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));

  int info = 0;
  int lwork = 0;
  CUSOLVER_CHECK(CusolverTraits<FloatT>::potrfBufferSize(
      cusolverHandle, CUBLAS_FILL_MODE_UPPER, static_cast<int>(numHiddenNodes), devHTH.data(),
      static_cast<int>(numHiddenNodes), &lwork));
  DeviceBuffer<FloatT> devWork(static_cast<std::size_t>(lwork));
  DeviceBuffer<int> devInfo(1);
  if (!devWork.isValid() || !devInfo.isValid()) {
    cusolverDnDestroy(cusolverHandle);
    return false;
  }

  CUSOLVER_CHECK(CusolverTraits<FloatT>::potrf(
      cusolverHandle, CUBLAS_FILL_MODE_UPPER, static_cast<int>(numHiddenNodes), devHTH.data(),
      static_cast<int>(numHiddenNodes), devWork.data(), lwork, devInfo.data()));

  if (!devInfo.copyToHost(&info, 1) || info != 0) {
    cusolverDnDestroy(cusolverHandle);
    return false;
  }

  CUSOLVER_CHECK(CusolverTraits<FloatT>::potrs(
      cusolverHandle, CUBLAS_FILL_MODE_UPPER, static_cast<int>(numHiddenNodes),
      static_cast<int>(numOutputs), devHTH.data(), static_cast<int>(numHiddenNodes), devHTT.data(),
      static_cast<int>(numHiddenNodes), devInfo.data()));

  if (!devInfo.copyToHost(&info, 1) || info != 0) {
    cusolverDnDestroy(cusolverHandle);
    return false;
  }

  cusolverDnDestroy(cusolverHandle);

  std::vector<FloatT> hostBeta(numHiddenNodes * numOutputs);
  if (!devHTT.copyToHost(hostBeta.data(), hostBeta.size())) {
    return false;
  }

  outputWeights->assign(numHiddenNodes * numOutputs, static_cast<FloatT>(0));
  for (std::size_t i = 0; i < numHiddenNodes; ++i) {
    for (std::size_t j = 0; j < numOutputs; ++j) {
      outputWeights->at(i * numOutputs + j) = hostBeta[j * numHiddenNodes + i];
    }
  }

  return true;
}

[[nodiscard]] bool isGpuAvailable() noexcept {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return err == cudaSuccess && deviceCount > 0;
}

template <typename FloatT>
[[nodiscard]] bool trainBatchElmGpu(const std::vector<FloatT>& trainData,
                                    const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                                    std::size_t numInputs, std::size_t numHiddenNodes,
                                    std::size_t numOutputs,
                                    const std::vector<FloatT>& hiddenWeights,
                                    const std::vector<FloatT>& hiddenBiases,
                                    feature_elm::ActivationFunction activation,
                                    std::vector<FloatT>* outputWeights) {
  if (!isGpuAvailable()) {
    return false;
  }
  if (trainData.size() != numSamples * numInputs ||
      trainTargets.size() != numSamples * numOutputs ||
      hiddenWeights.size() != numInputs * numHiddenNodes || hiddenBiases.size() != numHiddenNodes ||
      outputWeights == nullptr) {
    return false;
  }

  std::vector<FloatT> hiddenOutput(numSamples * numHiddenNodes);
  if (!computeHiddenOutputDevice(trainData, numSamples, numInputs, numHiddenNodes, hiddenWeights,
                                 hiddenBiases, activation, &hiddenOutput)) {
    return false;
  }

  return solveLeastSquaresDevice(hiddenOutput, trainTargets, numSamples, numHiddenNodes, numOutputs,
                                 outputWeights);
}

template <typename FloatT>
[[nodiscard]] bool predictBatchElmGpu(
    const std::vector<FloatT>& testData, std::size_t numSamples, std::size_t numInputs,
    std::size_t numHiddenNodes, std::size_t numOutputs, const std::vector<FloatT>& hiddenWeights,
    const std::vector<FloatT>& hiddenBiases, const std::vector<FloatT>& outputWeights,
    feature_elm::ActivationFunction activation, std::vector<FloatT>* predictions) {
  if (!isGpuAvailable()) {
    return false;
  }
  if (testData.size() != numSamples * numInputs ||
      hiddenWeights.size() != numInputs * numHiddenNodes || hiddenBiases.size() != numHiddenNodes ||
      outputWeights.size() != numHiddenNodes * numOutputs || predictions == nullptr) {
    return false;
  }

  std::vector<FloatT> hiddenOutput(numSamples * numHiddenNodes);
  if (!computeHiddenOutputDevice(testData, numSamples, numInputs, numHiddenNodes, hiddenWeights,
                                 hiddenBiases, activation, &hiddenOutput)) {
    return false;
  }

  predictions->assign(numSamples * numOutputs, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t outputIndex = 0; outputIndex < numOutputs; ++outputIndex) {
      FloatT sum = static_cast<FloatT>(0);
      for (std::size_t hiddenIndex = 0; hiddenIndex < numHiddenNodes; ++hiddenIndex) {
        sum += hiddenOutput[sample * numHiddenNodes + hiddenIndex] *
               outputWeights[hiddenIndex * numOutputs + outputIndex];
      }
      (*predictions)[sample * numOutputs + outputIndex] = sum;
    }
  }

  return true;
}

// Explicit template instantiations
template bool trainBatchElmGpu<float>(const std::vector<float>&, const std::vector<float>&,
                                      std::size_t, std::size_t, std::size_t, std::size_t,
                                      const std::vector<float>&, const std::vector<float>&,
                                      feature_elm::ActivationFunction, std::vector<float>*);

template bool trainBatchElmGpu<double>(const std::vector<double>&, const std::vector<double>&,
                                       std::size_t, std::size_t, std::size_t, std::size_t,
                                       const std::vector<double>&, const std::vector<double>&,
                                       feature_elm::ActivationFunction, std::vector<double>*);

template bool predictBatchElmGpu<float>(const std::vector<float>&, std::size_t, std::size_t,
                                        std::size_t, std::size_t, const std::vector<float>&,
                                        const std::vector<float>&, const std::vector<float>&,
                                        feature_elm::ActivationFunction, std::vector<float>*);

template bool predictBatchElmGpu<double>(const std::vector<double>&, std::size_t, std::size_t,
                                         std::size_t, std::size_t, const std::vector<double>&,
                                         const std::vector<double>&, const std::vector<double>&,
                                         feature_elm::ActivationFunction, std::vector<double>*);

#undef CUSOLVER_CHECK
#undef CUBLAS_CHECK
#undef CUDA_CHECK

}  // namespace feature_elm::cuda_backend
