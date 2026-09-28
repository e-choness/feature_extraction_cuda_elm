#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda/cuda_context.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/rls_gpu.hpp"

namespace feature_elm::cuda_backend {

namespace {

// Chunks larger than this are absorbed in sub-blocks (equivalent, and bounds the scratch space).
constexpr std::size_t kMaxBlock = 1024;
constexpr unsigned int kThreads = 256;

unsigned int gridFor(std::size_t n) {
  return static_cast<unsigned int>(std::min<std::size_t>((n + kThreads - 1) / kThreads, 65535));
}

__global__ void addToDiagonal(double* matrix, std::size_t n, double value) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    matrix[i * n + i] += value;
  }
}

}  // namespace

// The information matrix and vector are accumulated in float64 whatever FloatT is. With a small
// ridge and rank-deficient hidden features, float32 rounding in A = reg*I + sum H^T H is larger
// than reg itself and leaves A indefinite; no solve can recover from that, so precision has to be
// spent on the accumulation.
template <typename FloatT>
struct RlsGpuState {
  RlsGpuState(std::size_t f, std::size_t m)
      : features(f),
        outputs(m),
        a(f * f),
        c(f * m),
        ht(f * kMaxBlock),
        tcm(kMaxBlock * m),
        factor(f * f),
        solution(f * m),
        work(1),
        info(1) {}

  [[nodiscard]] bool valid() const noexcept {
    return a.isValid() && c.isValid() && ht.isValid() && tcm.isValid() && factor.isValid() &&
           solution.isValid() && work.isValid() && info.isValid();
  }

  std::size_t features;
  std::size_t outputs;
  DeviceBuffer<double> a;         // information matrix, features x features
  DeviceBuffer<double> c;         // information vector, column-major features x outputs
  DeviceBuffer<double> ht;        // H^T, column-major features x b (= H row-major)
  DeviceBuffer<double> tcm;       // T, column-major b x outputs
  DeviceBuffer<double> factor;    // Cholesky factor of A
  DeviceBuffer<double> solution;  // right-hand side / solution, column-major
  DeviceBuffer<double> work;
  DeviceBuffer<int> info;
  int workSize = 0;
};

namespace {

// Solves A X = B in place (B: column-major features x k, already on the device).
template <typename FloatT>
bool solveInPlace(RlsGpuState<FloatT>& s, double* b, std::size_t k, cusolverDnHandle_t solver) {
  const int f = static_cast<int>(s.features);
  if (cudaMemcpy(s.factor.data(), s.a.data(), s.features * s.features * sizeof(double),
                 cudaMemcpyDeviceToDevice) != cudaSuccess) {
    return false;
  }
  int info = 0;
  FEATURE_ELM_CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, f, s.factor.data(), f,
                                              s.work.data(), s.workSize, s.info.data()));
  if (!s.info.copyToHost(&info, 1) || info != 0) {
    return false;
  }
  FEATURE_ELM_CUSOLVER_CHECK(cusolverDnDpotrs(solver, CUBLAS_FILL_MODE_LOWER, f,
                                              static_cast<int>(k), s.factor.data(), f, b, f,
                                              s.info.data()));
  return s.info.copyToHost(&info, 1) && info == 0;
}

}  // namespace

template <typename FloatT>
void RlsGpuStateDeleter<FloatT>::operator()(RlsGpuState<FloatT>* state) const noexcept {
  delete state;
}

template <typename FloatT>
RlsGpuStatePtr<FloatT> createRlsGpu(std::size_t numFeatures, std::size_t numOutputs,
                                    FloatT regularization) {
  if (numFeatures == 0 || numOutputs == 0 || !(regularization > FloatT(0)) ||
      !detail::fitsInt(numFeatures * numFeatures) || !detail::fitsInt(numFeatures * kMaxBlock) ||
      !isGpuAvailable()) {
    return nullptr;
  }
  RlsGpuStatePtr<FloatT> state(new RlsGpuState<FloatT>(numFeatures, numOutputs));
  if (!state->valid()) {
    return nullptr;
  }
  auto& handles = detail::Handles::get();
  const auto lock = handles.lock();
  if (handles.cusolver() == nullptr ||
      cusolverDnDpotrf_bufferSize(handles.cusolver(), CUBLAS_FILL_MODE_LOWER,
                                  static_cast<int>(numFeatures), state->factor.data(),
                                  static_cast<int>(numFeatures),
                                  &state->workSize) != CUSOLVER_STATUS_SUCCESS) {
    return nullptr;
  }
  state->work = DeviceBuffer<double>(static_cast<std::size_t>(std::max(state->workSize, 1)));
  if (!state->work.isValid() ||
      cudaMemset(state->a.data(), 0, numFeatures * numFeatures * sizeof(double)) != cudaSuccess ||
      cudaMemset(state->c.data(), 0, numFeatures * numOutputs * sizeof(double)) != cudaSuccess) {
    return nullptr;
  }
  addToDiagonal<<<gridFor(numFeatures), kThreads>>>(state->a.data(), numFeatures,
                                                    static_cast<double>(regularization));
  if (cudaGetLastError() != cudaSuccess) {
    return nullptr;
  }
  return state;
}

template <typename FloatT>
bool updateRlsGpu(RlsGpuState<FloatT>& state, const FloatT* features, std::size_t numSamples,
                  const FloatT* targets) {
  if (features == nullptr || targets == nullptr || numSamples == 0) {
    return false;
  }
  auto& handles = detail::Handles::get();
  const auto lock = handles.lock();
  cublasHandle_t blas = handles.cublas();
  if (blas == nullptr) {
    return false;
  }

  const int f = static_cast<int>(state.features);
  const int m = static_cast<int>(state.outputs);
  std::vector<double> hostH;
  std::vector<double> hostT;
  for (std::size_t first = 0; first < numSamples; first += kMaxBlock) {
    const std::size_t count = std::min(kMaxBlock, numSamples - first);
    const int b = static_cast<int>(count);
    // H row-major (b x f) is H^T column-major (f x b). Targets go column-major (b x m).
    const FloatT* block = features + first * state.features;
    hostH.assign(block, block + count * state.features);
    if (!std::all_of(hostH.begin(), hostH.end(), [](double v) { return std::isfinite(v); })) {
      return false;  // mirrors the CPU path, which rejects a non-finite denominator
    }
    hostT.resize(count * state.outputs);
    for (std::size_t r = 0; r < count; ++r) {
      for (std::size_t o = 0; o < state.outputs; ++o) {
        hostT[r + o * count] = static_cast<double>(targets[(first + r) * state.outputs + o]);
      }
    }
    if (!state.ht.copyFromHost(hostH.data(), hostH.size()) ||
        !state.tcm.copyFromHost(hostT.data(), hostT.size())) {
      return false;
    }
    const double one = 1.0;
    // A += H^T H,  c += H^T T
    FEATURE_ELM_CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, f, f, b, &one,
                                         state.ht.data(), f, state.ht.data(), f, &one,
                                         state.a.data(), f));
    FEATURE_ELM_CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, f, m, b, &one,
                                         state.ht.data(), f, state.tcm.data(), b, &one,
                                         state.c.data(), f));
  }
  return true;
}

template <typename FloatT>
bool downloadRlsWeights(RlsGpuState<FloatT>& state, std::vector<FloatT>* weights) {
  if (weights == nullptr) {
    return false;
  }
  auto& handles = detail::Handles::get();
  const auto lock = handles.lock();
  if (handles.cusolver() == nullptr ||
      cudaMemcpy(state.solution.data(), state.c.data(),
                 state.features * state.outputs * sizeof(double),
                 cudaMemcpyDeviceToDevice) != cudaSuccess ||
      !solveInPlace(state, state.solution.data(), state.outputs, handles.cusolver())) {
    return false;
  }
  std::vector<double> columnMajor(state.features * state.outputs);
  if (!state.solution.copyToHost(columnMajor.data(), columnMajor.size())) {
    return false;
  }
  weights->resize(columnMajor.size());
  for (std::size_t i = 0; i < state.features; ++i) {
    for (std::size_t o = 0; o < state.outputs; ++o) {
      (*weights)[i * state.outputs + o] = static_cast<FloatT>(columnMajor[i + o * state.features]);
    }
  }
  return true;
}

template <typename FloatT>
bool downloadRlsCovariance(RlsGpuState<FloatT>& state, std::vector<FloatT>* covariance) {
  if (covariance == nullptr) {
    return false;
  }
  const std::size_t n = state.features;
  std::vector<double> matrix(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    matrix[i * n + i] = 1.0;
  }
  DeviceBuffer<double> inverse(n * n);
  auto& handles = detail::Handles::get();
  const auto lock = handles.lock();
  if (handles.cusolver() == nullptr || !inverse.copyFromHost(matrix.data(), matrix.size()) ||
      !solveInPlace(state, inverse.data(), n, handles.cusolver()) ||
      !inverse.copyToHost(matrix.data(), matrix.size())) {
    return false;
  }
  covariance->assign(matrix.begin(), matrix.end());
  return true;
}

#define FEATURE_ELM_INSTANTIATE_RLS_GPU(T)                                         \
  template struct RlsGpuStateDeleter<T>;                                           \
  template RlsGpuStatePtr<T> createRlsGpu<T>(std::size_t, std::size_t, T);         \
  template bool updateRlsGpu<T>(RlsGpuState<T>&, const T*, std::size_t, const T*); \
  template bool downloadRlsWeights<T>(RlsGpuState<T>&, std::vector<T>*);           \
  template bool downloadRlsCovariance<T>(RlsGpuState<T>&, std::vector<T>*);

FEATURE_ELM_INSTANTIATE_RLS_GPU(float)
FEATURE_ELM_INSTANTIATE_RLS_GPU(double)

#undef FEATURE_ELM_INSTANTIATE_RLS_GPU

}  // namespace feature_elm::cuda_backend
