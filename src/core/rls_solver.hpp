#ifndef FEATURE_ELM_CORE_RLS_SOLVER_HPP_
#define FEATURE_ELM_CORE_RLS_SOLVER_HPP_

#include <cstddef>
#include <vector>

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
  explicit RlsSolver(RlsOptions<FloatT> options = {});

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
  [[nodiscard]] const std::vector<FloatT>& weights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const std::vector<FloatT>& covariance() const noexcept {
    return covariance_;
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
  std::vector<FloatT> weights_;
  std::vector<FloatT> covariance_;

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
