// CPU-only builds (no CUDA toolkit) link this file instead of the .cu sources. Every GPU entry
// point reports failure, so callers that ask for Backend::kGpu fail cleanly rather than at link
// time.

#include "cuda/gpu_ops.hpp"
#include "cuda/rls_gpu.hpp"
#include "cuda/solver_gpu.hpp"

namespace feature_elm::cuda_backend {

std::string gpuDeviceName() {
  return {};
}

bool isGpuAvailable() noexcept {
  return false;
}

template <typename FloatT>
bool transformRandomAdditiveGpu(const std::vector<FloatT>& /*input*/, std::size_t /*numSamples*/,
                                std::size_t /*numInputs*/, std::size_t /*numHiddenNodes*/,
                                const std::vector<FloatT>& /*weights*/,
                                const std::vector<FloatT>& /*biases*/,
                                ActivationKind /*activation*/,
                                std::vector<FloatT>* /*hiddenOutput*/) {
  return false;
}

template <typename FloatT>
bool transformElmAutoEncoderGpu(const std::vector<FloatT>& /*input*/, std::size_t /*numSamples*/,
                                std::size_t /*numInputs*/, std::size_t /*numHiddenNodes*/,
                                const std::vector<FloatT>& /*encoderWeights*/,
                                const std::vector<FloatT>& /*encoderBiases*/,
                                ActivationKind /*activation*/,
                                std::vector<FloatT>* /*hiddenOutput*/) {
  return false;
}

template <typename FloatT>
bool solveRidgeGpu(const std::vector<FloatT>& /*features*/, const std::vector<FloatT>& /*targets*/,
                   std::size_t /*numSamples*/, std::size_t /*numOutputs*/,
                   SolverOptions<FloatT> /*options*/, std::vector<FloatT>* /*weights*/) {
  return false;
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
template bool solveRidgeGpu<float>(const std::vector<float>&, const std::vector<float>&,
                                   std::size_t, std::size_t, SolverOptions<float>,
                                   std::vector<float>*);
template bool solveRidgeGpu<double>(const std::vector<double>&, const std::vector<double>&,
                                    std::size_t, std::size_t, SolverOptions<double>,
                                    std::vector<double>*);

// RLS: never created without CUDA, so the deleter is never handed a live state.
template <typename FloatT>
struct RlsGpuState {};

template <typename FloatT>
void RlsGpuStateDeleter<FloatT>::operator()(RlsGpuState<FloatT>* state) const noexcept {
  delete state;
}

template <typename FloatT>
RlsGpuStatePtr<FloatT> createRlsGpu(std::size_t /*numFeatures*/, std::size_t /*numOutputs*/,
                                    FloatT /*regularization*/) {
  return nullptr;
}

template <typename FloatT>
bool updateRlsGpu(RlsGpuState<FloatT>& /*state*/, const FloatT* /*features*/,
                  std::size_t /*numSamples*/, const FloatT* /*targets*/) {
  return false;
}

template <typename FloatT>
bool downloadRlsWeights(RlsGpuState<FloatT>& /*state*/, std::vector<FloatT>* /*weights*/) {
  return false;
}

template <typename FloatT>
bool downloadRlsCovariance(RlsGpuState<FloatT>& /*state*/, std::vector<FloatT>* /*covariance*/) {
  return false;
}

template struct RlsGpuStateDeleter<float>;
template struct RlsGpuStateDeleter<double>;
template RlsGpuStatePtr<float> createRlsGpu<float>(std::size_t, std::size_t, float);
template RlsGpuStatePtr<double> createRlsGpu<double>(std::size_t, std::size_t, double);
template bool updateRlsGpu<float>(RlsGpuState<float>&, const float*, std::size_t, const float*);
template bool updateRlsGpu<double>(RlsGpuState<double>&, const double*, std::size_t, const double*);
template bool downloadRlsWeights<float>(RlsGpuState<float>&, std::vector<float>*);
template bool downloadRlsWeights<double>(RlsGpuState<double>&, std::vector<double>*);
template bool downloadRlsCovariance<float>(RlsGpuState<float>&, std::vector<float>*);
template bool downloadRlsCovariance<double>(RlsGpuState<double>&, std::vector<double>*);

}  // namespace feature_elm::cuda_backend
