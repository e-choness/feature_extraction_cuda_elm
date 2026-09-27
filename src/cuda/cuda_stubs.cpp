// CPU-only builds (no CUDA toolkit) link this file instead of the .cu sources. Every GPU entry
// point reports failure, so callers that ask for Backend::kGpu fail cleanly rather than at link
// time.

#include "cuda/gpu_ops.hpp"
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

}  // namespace feature_elm::cuda_backend
