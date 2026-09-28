#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda/cuda_context.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

namespace feature_elm::cuda_backend {

namespace {

template <typename FloatT>
struct Lapack;

template <>
struct Lapack<float> {
  static cublasStatus_t syrk(cublasHandle_t h, int n, int k, const float* alpha, const float* A,
                             int lda, const float* beta, float* C, int ldc) {
    return cublasSsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha, A, lda, beta, C, ldc);
  }
  static cublasStatus_t gemm(cublasHandle_t h, int m, int n, int k, const float* alpha,
                             const float* A, int lda, const float* B, int ldb, const float* beta,
                             float* C, int ldc) {
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  static cusolverStatus_t potrfBufferSize(cusolverDnHandle_t h, int n, float* A, int lda, int* lw) {
    return cusolverDnSpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, lw);
  }
  static cusolverStatus_t potrf(cusolverDnHandle_t h, int n, float* A, int lda, float* work,
                                int lwork, int* info) {
    return cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, lwork, info);
  }
  static cusolverStatus_t potrs(cusolverDnHandle_t h, int n, int nrhs, const float* A, int lda,
                                float* B, int ldb, int* info) {
    return cusolverDnSpotrs(h, CUBLAS_FILL_MODE_LOWER, n, nrhs, A, lda, B, ldb, info);
  }
  static cusolverStatus_t geqrfBufferSize(cusolverDnHandle_t h, int m, int n, float* A, int lda,
                                          int* lwork) {
    return cusolverDnSgeqrf_bufferSize(h, m, n, A, lda, lwork);
  }
  static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, float* A, int lda, float* tau,
                                float* work, int lwork, int* info) {
    return cusolverDnSgeqrf(h, m, n, A, lda, tau, work, lwork, info);
  }
  static cusolverStatus_t ormqrBufferSize(cusolverDnHandle_t h, int m, int n, int k, const float* A,
                                          int lda, const float* tau, const float* C, int ldc,
                                          int* lwork) {
    return cusolverDnSormqr_bufferSize(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, n, k, A, lda, tau, C,
                                       ldc, lwork);
  }
  static cusolverStatus_t ormqr(cusolverDnHandle_t h, int m, int n, int k, const float* A, int lda,
                                const float* tau, float* C, int ldc, float* work, int lwork,
                                int* info) {
    return cusolverDnSormqr(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, n, k, A, lda, tau, C, ldc, work,
                            lwork, info);
  }
  static cublasStatus_t trsm(cublasHandle_t h, int m, int n, const float* alpha, const float* A,
                             int lda, float* B, int ldb) {
    return cublasStrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                       CUBLAS_DIAG_NON_UNIT, m, n, alpha, A, lda, B, ldb);
  }
};

template <>
struct Lapack<double> {
  static cublasStatus_t syrk(cublasHandle_t h, int n, int k, const double* alpha, const double* A,
                             int lda, const double* beta, double* C, int ldc) {
    return cublasDsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha, A, lda, beta, C, ldc);
  }
  static cublasStatus_t gemm(cublasHandle_t h, int m, int n, int k, const double* alpha,
                             const double* A, int lda, const double* B, int ldb, const double* beta,
                             double* C, int ldc) {
    return cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  static cusolverStatus_t potrfBufferSize(cusolverDnHandle_t h, int n, double* A, int lda,
                                          int* lw) {
    return cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, lw);
  }
  static cusolverStatus_t potrf(cusolverDnHandle_t h, int n, double* A, int lda, double* work,
                                int lwork, int* info) {
    return cusolverDnDpotrf(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, lwork, info);
  }
  static cusolverStatus_t potrs(cusolverDnHandle_t h, int n, int nrhs, const double* A, int lda,
                                double* B, int ldb, int* info) {
    return cusolverDnDpotrs(h, CUBLAS_FILL_MODE_LOWER, n, nrhs, A, lda, B, ldb, info);
  }
  static cusolverStatus_t geqrfBufferSize(cusolverDnHandle_t h, int m, int n, double* A, int lda,
                                          int* lwork) {
    return cusolverDnDgeqrf_bufferSize(h, m, n, A, lda, lwork);
  }
  static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, double* A, int lda, double* tau,
                                double* work, int lwork, int* info) {
    return cusolverDnDgeqrf(h, m, n, A, lda, tau, work, lwork, info);
  }
  static cusolverStatus_t ormqrBufferSize(cusolverDnHandle_t h, int m, int n, int k,
                                          const double* A, int lda, const double* tau,
                                          const double* C, int ldc, int* lwork) {
    return cusolverDnDormqr_bufferSize(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, n, k, A, lda, tau, C,
                                       ldc, lwork);
  }
  static cusolverStatus_t ormqr(cusolverDnHandle_t h, int m, int n, int k, const double* A, int lda,
                                const double* tau, double* C, int ldc, double* work, int lwork,
                                int* info) {
    return cusolverDnDormqr(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, n, k, A, lda, tau, C, ldc, work,
                            lwork, info);
  }
  static cublasStatus_t trsm(cublasHandle_t h, int m, int n, const double* alpha, const double* A,
                             int lda, double* B, int ldb) {
    return cublasDtrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                       CUBLAS_DIAG_NON_UNIT, m, n, alpha, A, lda, B, ldb);
  }
};

template <typename FloatT>
__global__ void addToDiagonal(FloatT* matrix, std::size_t n, FloatT value) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    matrix[i * n + i] += value;
  }
}

// Normal equations on the device: A = H^T H + alpha I (syrk, lower triangle), c = H^T T, then a
// Cholesky solve. The same strategy as the CPU BatchRidgeSolver, and the fast path: one syrk over
// the data instead of a QR of the (samples + features) x features augmented matrix. Returns false
// if A is not numerically positive definite, in which case the caller falls back to QR.
template <typename FloatT>
bool solveNormalEquations(const std::vector<FloatT>& features, const std::vector<FloatT>& targets,
                          std::size_t numSamples, std::size_t numFeatures, std::size_t numOutputs,
                          FloatT alpha, std::vector<FloatT>* weights, cublasHandle_t blas,
                          cusolverDnHandle_t solver) {
  if (!detail::fitsInt(numSamples) || !detail::fitsInt(numSamples * numFeatures) ||
      !detail::fitsInt(numFeatures * numFeatures)) {
    return false;
  }
  // H row-major (n x f) is H^T column-major (f x n): upload as-is. Targets go column-major.
  std::vector<FloatT> targetsColumnMajor(numSamples * numOutputs);
  for (std::size_t s = 0; s < numSamples; ++s) {
    for (std::size_t o = 0; o < numOutputs; ++o) {
      targetsColumnMajor[s + o * numSamples] = targets[s * numOutputs + o];
    }
  }
  DeviceBuffer<FloatT> ht(features.size());
  DeviceBuffer<FloatT> t(targetsColumnMajor.size());
  DeviceBuffer<FloatT> a(numFeatures * numFeatures);
  DeviceBuffer<FloatT> c(numFeatures * numOutputs);
  DeviceBuffer<int> info(1);
  if (!ht.copyFromHost(features.data(), features.size()) ||
      !t.copyFromHost(targetsColumnMajor.data(), targetsColumnMajor.size()) || !a.isValid() ||
      !c.isValid() || !info.isValid()) {
    return false;
  }
  const int f = static_cast<int>(numFeatures);
  const int n = static_cast<int>(numSamples);
  const int m = static_cast<int>(numOutputs);
  const FloatT one = FloatT(1);
  const FloatT zero = FloatT(0);
  FEATURE_ELM_CUBLAS_CHECK(
      Lapack<FloatT>::syrk(blas, f, n, &one, ht.data(), f, &zero, a.data(), f));
  constexpr unsigned int kThreads = 256;
  addToDiagonal<FloatT>
      <<<static_cast<unsigned int>((numFeatures + kThreads - 1) / kThreads), kThreads>>>(
          a.data(), numFeatures, alpha);
  FEATURE_ELM_CUDA_CHECK(cudaGetLastError());
  FEATURE_ELM_CUBLAS_CHECK(
      Lapack<FloatT>::gemm(blas, f, m, n, &one, ht.data(), f, t.data(), n, &zero, c.data(), f));

  int workSize = 0;
  FEATURE_ELM_CUSOLVER_CHECK(Lapack<FloatT>::potrfBufferSize(solver, f, a.data(), f, &workSize));
  DeviceBuffer<FloatT> work(static_cast<std::size_t>(workSize > 0 ? workSize : 1));
  if (!work.isValid()) {
    return false;
  }
  FEATURE_ELM_CUSOLVER_CHECK(
      Lapack<FloatT>::potrf(solver, f, a.data(), f, work.data(), workSize, info.data()));
  int status = 0;
  if (!info.copyToHost(&status, 1) || status != 0) {
    return false;
  }
  FEATURE_ELM_CUSOLVER_CHECK(
      Lapack<FloatT>::potrs(solver, f, m, a.data(), f, c.data(), f, info.data()));
  if (!info.copyToHost(&status, 1) || status != 0) {
    return false;
  }
  std::vector<FloatT> solution(numFeatures * numOutputs);
  if (!c.copyToHost(solution.data(), solution.size())) {
    return false;
  }
  weights->assign(numFeatures * numOutputs, FloatT(0));
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    for (std::size_t output = 0; output < numOutputs; ++output) {
      (*weights)[feature * numOutputs + output] = solution[feature + output * numFeatures];
    }
  }
  return std::all_of(weights->begin(), weights->end(), [](FloatT v) { return std::isfinite(v); });
}

}  // namespace

// Solves min ||H beta - T||^2 + alpha ||beta||^2 as the ordinary least-squares problem on the
// augmented system
//
//     [ H          ]          [ T ]
//     [ sqrt(a) I  ] beta  =  [ 0 ]
//
// via Householder QR (geqrf), Q^T applied to the right-hand side (ormqr) and a triangular solve
// with R (trsm). This avoids forming H^T H, so it stays accurate when H is ill-conditioned.
template <typename FloatT>
bool solveRidgeGpu(const std::vector<FloatT>& features, const std::vector<FloatT>& targets,
                   std::size_t numSamples, std::size_t numOutputs, SolverOptions<FloatT> options,
                   std::vector<FloatT>* weights) {
  if (weights == nullptr || features.empty() || numSamples == 0 || numOutputs == 0 ||
      features.size() % numSamples != 0 || targets.size() != numSamples * numOutputs ||
      !(options.ridgeAlpha > FloatT(0)) || !std::isfinite(options.ridgeAlpha)) {
    return false;
  }
  const std::size_t numFeatures = features.size() / numSamples;
  const std::size_t rows = numSamples + numFeatures;
  if (!detail::fitsInt(rows) || !detail::fitsInt(numOutputs) ||
      !detail::fitsInt(rows * numFeatures) || !isGpuAvailable()) {
    return false;
  }

  auto& handles = detail::Handles::get();
  const auto lock = handles.lock();
  if (handles.cublas() == nullptr || handles.cusolver() == nullptr) {
    return false;
  }

  if (solveNormalEquations(features, targets, numSamples, numFeatures, numOutputs,
                           options.ridgeAlpha, weights, handles.cublas(), handles.cusolver())) {
    return true;
  }
  // Fallback: QR on the augmented system, which never forms H^T H.

  // Pack column-major: A is rows x numFeatures, B is rows x numOutputs (leading dimension rows).
  std::vector<FloatT> a(rows * numFeatures, FloatT(0));
  std::vector<FloatT> b(rows * numOutputs, FloatT(0));
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t feature = 0; feature < numFeatures; ++feature) {
      a[sample + feature * rows] = features[sample * numFeatures + feature];
    }
    for (std::size_t output = 0; output < numOutputs; ++output) {
      b[sample + output * rows] = targets[sample * numOutputs + output];
    }
  }
  const FloatT sqrtAlpha = std::sqrt(options.ridgeAlpha);
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    a[(numSamples + feature) + feature * rows] = sqrtAlpha;
  }

  DeviceBuffer<FloatT> devA(a.size());
  DeviceBuffer<FloatT> devB(b.size());
  DeviceBuffer<FloatT> devTau(numFeatures);
  DeviceBuffer<int> devInfo(1);
  if (!devA.copyFromHost(a.data(), a.size()) || !devB.copyFromHost(b.data(), b.size()) ||
      !devTau.isValid() || !devInfo.isValid()) {
    return false;
  }

  const int m = static_cast<int>(rows);
  const int n = static_cast<int>(numFeatures);
  const int nrhs = static_cast<int>(numOutputs);

  int geqrfWork = 0;
  int ormqrWork = 0;
  FEATURE_ELM_CUSOLVER_CHECK(
      Lapack<FloatT>::geqrfBufferSize(handles.cusolver(), m, n, devA.data(), m, &geqrfWork));
  FEATURE_ELM_CUSOLVER_CHECK(Lapack<FloatT>::ormqrBufferSize(
      handles.cusolver(), m, nrhs, n, devA.data(), m, devTau.data(), devB.data(), m, &ormqrWork));
  const int workSize = geqrfWork > ormqrWork ? geqrfWork : ormqrWork;
  DeviceBuffer<FloatT> devWork(static_cast<std::size_t>(workSize > 0 ? workSize : 1));
  if (!devWork.isValid()) {
    return false;
  }

  int info = 0;
  FEATURE_ELM_CUSOLVER_CHECK(Lapack<FloatT>::geqrf(handles.cusolver(), m, n, devA.data(), m,
                                                   devTau.data(), devWork.data(), workSize,
                                                   devInfo.data()));
  if (!devInfo.copyToHost(&info, 1) || info != 0) {
    return false;
  }

  FEATURE_ELM_CUSOLVER_CHECK(Lapack<FloatT>::ormqr(handles.cusolver(), m, nrhs, n, devA.data(), m,
                                                   devTau.data(), devB.data(), m, devWork.data(),
                                                   workSize, devInfo.data()));
  if (!devInfo.copyToHost(&info, 1) || info != 0) {
    return false;
  }

  // R lives in the upper triangle of the leading n x n block of A.
  const FloatT one = FloatT(1);
  FEATURE_ELM_CUBLAS_CHECK(
      Lapack<FloatT>::trsm(handles.cublas(), n, nrhs, &one, devA.data(), m, devB.data(), m));

  if (!devB.copyToHost(b.data(), b.size())) {
    return false;
  }

  // The solution is the first numFeatures rows of B; return it row-major (features x outputs),
  // matching BatchRidgeSolver.
  weights->assign(numFeatures * numOutputs, FloatT(0));
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    for (std::size_t output = 0; output < numOutputs; ++output) {
      (*weights)[feature * numOutputs + output] = b[feature + output * rows];
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

}  // namespace feature_elm::cuda_backend
