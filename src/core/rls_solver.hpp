#ifndef FEATURE_ELM_CORE_RLS_SOLVER_HPP_
#define FEATURE_ELM_CORE_RLS_SOLVER_HPP_

#include <cstddef>
#include <vector>

#include "core/feature_map.hpp"
#include "cuda/rls_gpu.hpp"

namespace feature_elm {

enum class RlsConstraint { kNone, kClassDistance };

template <typename FloatT>
struct RlsOptions {
  FloatT regularization = static_cast<FloatT>(1e-3);
  FloatT forgettingFactor = static_cast<FloatT>(1);
  RlsConstraint constraint = RlsConstraint::kNone;
  FloatT constraintStrength = static_cast<FloatT>(1e-2);
};

template <typename FloatT>
class RlsSolver {
 public:
  /// With Backend::kGpu (and a CUDA device), plain RLS (forgetting factor 1, no constraint) runs
  /// as device-resident block updates; FOS-ELM and OS-CELM settings always use the CPU path.
  explicit RlsSolver(RlsOptions<FloatT> options = {}, Backend backend = Backend::kCpu);

  [[nodiscard]] bool initialize(const std::vector<FloatT>& features, std::size_t numSamples,
                                const std::vector<FloatT>& targets, std::size_t numOutputs);

  [[nodiscard]] bool update(const std::vector<FloatT>& features, std::size_t numSamples,
                            const std::vector<FloatT>& targets);

  [[nodiscard]] std::size_t numFeatures() const noexcept {
    return numFeatures_;
  }
  [[nodiscard]] std::size_t numOutputs() const noexcept {
    return numOutputs_;
  }
  [[nodiscard]] bool isInitialized() const noexcept {
    return isInitialized_;
  }
  /// Weights, row-major (features x outputs). Downloaded from the GPU on first access after an
  /// update.
  [[nodiscard]] const std::vector<FloatT>& weights() const;
  /// Covariance P (features x features). Downloaded from the GPU on first access after an update.
  [[nodiscard]] const std::vector<FloatT>& covariance() const;
  /// True when updates run on the GPU.
  [[nodiscard]] bool usesGpu() const noexcept {
    return gpu_ != nullptr;
  }
  [[nodiscard]] RlsOptions<FloatT> options() const noexcept {
    return options_;
  }

  void reset() noexcept;

 private:
  RlsOptions<FloatT> options_;
  std::size_t numFeatures_;
  std::size_t numOutputs_;
  bool isInitialized_;
  Backend backend_;
  mutable std::vector<FloatT> weights_;
  mutable std::vector<FloatT> covariance_;
  cuda_backend::RlsGpuStatePtr<FloatT> gpu_;
  mutable bool weightsStale_ = false;
  mutable bool covarianceStale_ = false;

  [[nodiscard]] bool updateRecursiveLeastSquares(const std::vector<FloatT>& features,
                                                 std::size_t numSamples,
                                                 const std::vector<FloatT>& targets);
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] FloatT computeClassDistance(const std::vector<FloatT>& features,
                                            const std::vector<FloatT>& targets,
                                            std::size_t numSamples) const;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_RLS_SOLVER_HPP_
