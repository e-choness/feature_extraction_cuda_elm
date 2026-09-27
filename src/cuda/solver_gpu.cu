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
