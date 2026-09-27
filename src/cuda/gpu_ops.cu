#include <cmath>
#include <vector>

#include "cuda/cuda_context.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gpu_ops.hpp"

namespace feature_elm::cuda_backend {

namespace {

template <typename FloatT>
__device__ FloatT activateDevice(FloatT value, ActivationKind activation) noexcept {
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

// `matrix` is column-major (hidden x samples), so consecutive threads touch consecutive
// addresses; the bias index is the row, i.e. idx % hidden.
template <typename FloatT>
__global__ void addBiasActivateKernel(FloatT* matrix, const FloatT* biases, std::size_t hidden,
                                      std::size_t total, ActivationKind activation) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    matrix[idx] = activateDevice(matrix[idx] + biases[idx % hidden], activation);
  }
}

template <typename FloatT>
struct Gemm;

template <>
struct Gemm<float> {
  static cublasStatus_t call(cublasHandle_t handle, int m, int n, int k, const float* alpha,
                             const float* A, int lda, const float* B, int ldb, const float* beta,
                             float* C, int ldc) {
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C,
                       ldc);
  }
};

template <>
struct Gemm<double> {
  static cublasStatus_t call(cublasHandle_t handle, int m, int n, int k, const double* alpha,
                             const double* A, int lda, const double* B, int ldb, const double* beta,
                             double* C, int ldc) {
    return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C,
                       ldc);
  }
};

// Computes H = g(X * W + b) for row-major X (samples x inputs), row-major W (inputs x hidden) and
// writes row-major H (samples x hidden).
//
// A row-major R x C matrix is bit-identical to a column-major C x R matrix, so in cuBLAS terms
// this is H^T (hidden x samples) = W^T (hidden x inputs) * X^T (inputs x samples) with no host
// transposes: the host buffers are uploaded and downloaded as-is.
template <typename FloatT>
bool denseActivationGpu(const std::vector<FloatT>& input, std::size_t numSamples,
                        std::size_t numInputs, std::size_t numHiddenNodes,
                        const std::vector<FloatT>& weights, const std::vector<FloatT>& biases,
                        ActivationKind activation, std::vector<FloatT>* hiddenOutput) {
  if (hiddenOutput == nullptr || numSamples == 0 || numInputs == 0 || numHiddenNodes == 0 ||
      input.size() != numSamples * numInputs || weights.size() != numInputs * numHiddenNodes ||
      biases.size() != numHiddenNodes || !detail::fitsInt(numSamples) ||
      !detail::fitsInt(numInputs) || !detail::fitsInt(numHiddenNodes) || !isGpuAvailable()) {
    return false;
  }

  auto& handles = detail::Handles::get();
  const auto lock = handles.lock();
  if (handles.cublas() == nullptr) {
    return false;
  }

  const std::size_t total = numSamples * numHiddenNodes;
  DeviceBuffer<FloatT> devInput(input.size());
  DeviceBuffer<FloatT> devWeights(weights.size());
  DeviceBuffer<FloatT> devBiases(biases.size());
  DeviceBuffer<FloatT> devHidden(total);
  if (!devInput.copyFromHost(input.data(), input.size()) ||
      !devWeights.copyFromHost(weights.data(), weights.size()) ||
      !devBiases.copyFromHost(biases.data(), biases.size()) || !devHidden.isValid()) {
    return false;
  }

  const FloatT alpha = FloatT(1);
  const FloatT beta = FloatT(0);
  const int hidden = static_cast<int>(numHiddenNodes);
  const int inputs = static_cast<int>(numInputs);
  FEATURE_ELM_CUBLAS_CHECK(Gemm<FloatT>::call(
      handles.cublas(), hidden, static_cast<int>(numSamples), inputs, &alpha, devWeights.data(),
      hidden, devInput.data(), inputs, &beta, devHidden.data(), hidden));

  constexpr unsigned int kBlockSize = 256;
  const std::size_t wanted = (total + kBlockSize - 1) / kBlockSize;
  const unsigned int gridSize = static_cast<unsigned int>(wanted < 65535 ? wanted : 65535);
  addBiasActivateKernel<FloatT><<<gridSize, kBlockSize>>>(devHidden.data(), devBiases.data(),
                                                          numHiddenNodes, total, activation);
  FEATURE_ELM_CUDA_CHECK(cudaGetLastError());

  hiddenOutput->resize(total);
  // cudaMemcpy on the legacy default stream synchronises with the kernel above.
  return devHidden.copyToHost(hiddenOutput->data(), total);
}

}  // namespace

[[nodiscard]] bool isGpuAvailable() noexcept {
  static const bool available = [] {
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
  }();
  return available;
}

std::string gpuDeviceName() {
  if (!isGpuAvailable()) {
    return {};
  }
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    return {};
  }
  return props.name;
}

template <typename FloatT>
bool transformRandomAdditiveGpu(const std::vector<FloatT>& input, std::size_t numSamples,
                                std::size_t numInputs, std::size_t numHiddenNodes,
                                const std::vector<FloatT>& weights,
                                const std::vector<FloatT>& biases, ActivationKind activation,
                                std::vector<FloatT>* hiddenOutput) {
  return denseActivationGpu(input, numSamples, numInputs, numHiddenNodes, weights, biases,
                            activation, hiddenOutput);
}

template <typename FloatT>
bool transformElmAutoEncoderGpu(const std::vector<FloatT>& input, std::size_t numSamples,
                                std::size_t numInputs, std::size_t numHiddenNodes,
                                const std::vector<FloatT>& encoderWeights,
                                const std::vector<FloatT>& encoderBiases, ActivationKind activation,
                                std::vector<FloatT>* hiddenOutput) {
  return denseActivationGpu(input, numSamples, numInputs, numHiddenNodes, encoderWeights,
                            encoderBiases, activation, hiddenOutput);
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

}  // namespace feature_elm::cuda_backend
