#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <limits>
#include <vector>

#include "cuda/device_buffer.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

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

#define CUSOLVER_CHECK(expr)                 \
  do {                                       \
    cusolverStatus_t status = (expr);        \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
      return false;                          \
    }                                        \
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

class CusolverHandle {
 public:
  CusolverHandle() {
    (void)cusolverDnCreate(&handle_);
  }

  CusolverHandle(const CusolverHandle&) = delete;
  CusolverHandle& operator=(const CusolverHandle&) = delete;

  ~CusolverHandle() {
    if (handle_ != nullptr) {
      (void)cusolverDnDestroy(handle_);
    }
  }

  [[nodiscard]] bool isValid() const noexcept {
    return handle_ != nullptr;
  }

  [[nodiscard]] cusolverDnHandle_t get() const noexcept {
    return handle_;
  }

 private:
  cusolverDnHandle_t handle_ = nullptr;
};

template <typename FloatT>
struct CublasTraits;

template <>
struct CublasTraits<float> {
  static cublasStatus_t trsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                             cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                             const float* alpha, const float* A, int lda, float* B, int ldb) {
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  }
};

template <>
struct CublasTraits<double> {
  static cublasStatus_t trsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                             cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                             const double* alpha, const double* A, int lda, double* B, int ldb) {
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  }
};

template <typename FloatT>
struct CusolverQrTraits;

template <>
struct CusolverQrTraits<float> {
  static cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda,
                                float* tau, float* work, int lwork, int* devInfo) {
    return cusolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo);
  }

  static cusolverStatus_t geqrfBufferSize(cusolverDnHandle_t handle, int m, int n, float* A,
                                          int lda, int* lwork) {
    return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
  }

  static cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A, int lda,
                                const float* tau, float* work, int lwork, int* devInfo) {
    return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
  }

  static cusolverStatus_t orgqrBufferSize(cusolverDnHandle_t handle, int m, int n, int k,
                                          const float* A, int lda, const float* tau, int* lwork) {
    return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
  }

  static cusolverStatus_t ormqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                cublasOperation_t trans, int m, int n, int k, const float* A,
                                int lda, const float* tau, float* C, int ldc, float* work,
                                int lwork, int* devInfo) {
    return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                            devInfo);
  }

  static cusolverStatus_t ormqrBufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                                          cublasOperation_t trans, int m, int n, int k,
                                          const float* A, int lda, const float* tau, const float* C,
                                          int ldc, int* lwork) {
    return cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
  }
};

template <>
struct CusolverQrTraits<double> {
  static cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda,
                                double* tau, double* work, int lwork, int* devInfo) {
    return cusolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo);
  }

  static cusolverStatus_t geqrfBufferSize(cusolverDnHandle_t handle, int m, int n, double* A,
                                          int lda, int* lwork) {
    return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
  }

  static cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A, int lda,
                                const double* tau, double* work, int lwork, int* devInfo) {
    return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
  }

  static cusolverStatus_t orgqrBufferSize(cusolverDnHandle_t handle, int m, int n, int k,
                                          const double* A, int lda, const double* tau, int* lwork) {
    return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
  }

  static cusolverStatus_t ormqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                cublasOperation_t trans, int m, int n, int k, const double* A,
                                int lda, const double* tau, double* C, int ldc, double* work,
                                int lwork, int* devInfo) {
    return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,
                            devInfo);
  }

  static cusolverStatus_t ormqrBufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                                          cublasOperation_t trans, int m, int n, int k,
                                          const double* A, int lda, const double* tau,
                                          const double* C, int ldc, int* lwork) {
    return cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
  }
};

bool checkedDimensions(std::size_t rows, std::size_t cols) noexcept {
  return rows <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         cols <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

}  // namespace

template <typename FloatT>
bool solveRidgeGpu(const std::vector<FloatT>& features, const std::vector<FloatT>& targets,
                   std::size_t numSamples, std::size_t numOutputs, SolverOptions<FloatT> options,
                   std::vector<FloatT>* weights) {
  if (!isGpuAvailable() || features.empty() || targets.empty() || weights == nullptr ||
      numSamples == 0 || numOutputs == 0 || features.size() % numSamples != 0 ||
      targets.size() != numSamples * numOutputs) {
    return false;
  }

  const std::size_t numFeatures = features.size() / numSamples;
  if (numFeatures == 0 || !checkedDimensions(numSamples + numFeatures, numFeatures) ||
      !checkedDimensions(numFeatures, numOutputs) || !(options.ridgeAlpha > FloatT(0)) ||
      !std::isfinite(options.ridgeAlpha)) {
    return false;
  }

  const std::size_t augmentedRows = numSamples + numFeatures;
  std::vector<FloatT> matrixColumnMajor(augmentedRows * numFeatures, FloatT(0));
  std::vector<FloatT> rhsColumnMajor(augmentedRows * numOutputs, FloatT(0));

  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t feature = 0; feature < numFeatures; ++feature) {
      matrixColumnMajor[feature + sample * augmentedRows] =
          features[sample * numFeatures + feature];
    }
    for (std::size_t output = 0; output < numOutputs; ++output) {
      rhsColumnMajor[output * augmentedRows + sample] = targets[sample * numOutputs + output];
    }
  }

  // ridgeAlpha bounds are validated: >0 and finite (lines 214-215)
  const FloatT sqrtAlpha = std::sqrt(options.ridgeAlpha);
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    // Add sqrt(alpha) to diagonal of augmented matrix (column-major: feature +
    // (numSamples+feature)*augmentedRows)
    matrixColumnMajor[feature + (numSamples + feature) * augmentedRows] = sqrtAlpha;
  }

  DeviceBuffer<FloatT> devMatrix(matrixColumnMajor.size());
  DeviceBuffer<FloatT> devRhs(rhsColumnMajor.size());
  DeviceBuffer<FloatT> devTau(numFeatures);
  DeviceBuffer<int> devInfo(1);

  if (!devMatrix.isValid() || !devRhs.isValid() || !devTau.isValid() || !devInfo.isValid()) {
    return false;
  }

  if (!devMatrix.copyFromHost(matrixColumnMajor.data(), matrixColumnMajor.size()) ||
      !devRhs.copyFromHost(rhsColumnMajor.data(), rhsColumnMajor.size())) {
    return false;
  }

  CublasHandle cublas;
  CusolverHandle cusolver;
  if (!cublas.isValid() || !cusolver.isValid()) {
    return false;
  }

  const int m = static_cast<int>(augmentedRows);
  const int n = static_cast<int>(numFeatures);
  const int nrhs = static_cast<int>(numOutputs);

  int geqrfWorkSize = 0;
  CUSOLVER_CHECK(CusolverQrTraits<FloatT>::geqrfBufferSize(cusolver.get(), m, n, devMatrix.data(),
                                                           m, &geqrfWorkSize));
  if (geqrfWorkSize <= 0) {
    return false;
  }
  DeviceBuffer<FloatT> geqrfWork(static_cast<std::size_t>(geqrfWorkSize));
  if (!geqrfWork.isValid()) {
    return false;
  }

  CUSOLVER_CHECK(CusolverQrTraits<FloatT>::geqrf(cusolver.get(), m, n, devMatrix.data(), m,
                                                 devTau.data(), geqrfWork.data(), geqrfWorkSize,
                                                 devInfo.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  int info = 0;
  if (!devInfo.copyToHost(&info, 1) || info != 0) {
    return false;
  }

  int orgqrWorkSize = 0;
  CUSOLVER_CHECK(CusolverQrTraits<FloatT>::orgqrBufferSize(
      cusolver.get(), m, n, n, devMatrix.data(), m, devTau.data(), &orgqrWorkSize));
  if (orgqrWorkSize <= 0) {
    return false;
  }
  DeviceBuffer<FloatT> orgqrWork(static_cast<std::size_t>(orgqrWorkSize));
  if (!orgqrWork.isValid()) {
    return false;
  }

  CUSOLVER_CHECK(CusolverQrTraits<FloatT>::orgqr(cusolver.get(), m, n, n, devMatrix.data(), m,
                                                 devTau.data(), orgqrWork.data(), orgqrWorkSize,
                                                 devInfo.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  if (!devInfo.copyToHost(&info, 1) || info != 0) {
    return false;
  }

  int ormqrWorkSize = 0;
  CUSOLVER_CHECK(CusolverQrTraits<FloatT>::ormqrBufferSize(
      cusolver.get(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, n, devMatrix.data(), m, devTau.data(),
      devRhs.data(), m, &ormqrWorkSize));
  if (ormqrWorkSize <= 0) {
    return false;
  }
  DeviceBuffer<FloatT> ormqrWork(static_cast<std::size_t>(ormqrWorkSize));
  if (!ormqrWork.isValid()) {
    return false;
  }

  CUSOLVER_CHECK(CusolverQrTraits<FloatT>::ormqr(
      cusolver.get(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, n, devMatrix.data(), m, devTau.data(),
      devRhs.data(), m, ormqrWork.data(), ormqrWorkSize, devInfo.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  if (!devInfo.copyToHost(&info, 1) || info != 0) {
    return false;
  }

  const FloatT alpha = FloatT(1);
  CUBLAS_CHECK(CublasTraits<FloatT>::trsm(cublas.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                          CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs, &alpha,
                                          devMatrix.data(), m, devRhs.data(), m));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<FloatT> solutionColumnMajor(numFeatures * numOutputs);
  if (!devRhs.copyFromHost(solutionColumnMajor.data(), solutionColumnMajor.size())) {
    return false;
  }

  // Convert column-major device output back to row-major weights for CPU use
  weights->assign(numFeatures * numOutputs, FloatT(0));
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    for (std::size_t output = 0; output < numOutputs; ++output) {
      // Column-major: [feature + output * numFeatures], transpose to row-major: [feature *
      // numOutputs + output]
      (*weights)[feature * numOutputs + output] =
          solutionColumnMajor[feature + output * numFeatures];
    }
  }

  return true;
}

template bool solveRidgeGpu<float>(const std::vector<float>&, const std::vector<float>&,
                                   std::size_t, std::size_t, SolverOptions<float>,
                                   std::vector<float>*);

template bool solveRidgeGpu<double>(const std::vector<double>&, const std::vector<double>&,
                                    std::size_t, std::size_t, SolverOptions<double>,
                                    std::vector<double>*);

#undef CUSOLVER_CHECK
#undef CUBLAS_CHECK
#undef CUDA_CHECK

}  // namespace feature_elm::cuda_backend