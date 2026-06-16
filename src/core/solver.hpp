#ifndef FEATURE_ELM_CORE_SOLVER_HPP_
#define FEATURE_ELM_CORE_SOLVER_HPP_

#include <cstddef>
#include <vector>

namespace feature_elm {

enum class RidgeSolvePath { kAuto, kPrimal, kDual };

enum class RidgeSolveMethod { kCholesky, kHouseholderQr };

template <typename FloatT>
struct SolverOptions {
  FloatT ridgeAlpha = static_cast<FloatT>(1e-6);
  RidgeSolvePath path = RidgeSolvePath::kAuto;
  RidgeSolveMethod method = RidgeSolveMethod::kCholesky;
};

template <typename FloatT>
class Solver {
 public:
  virtual ~Solver() = default;

  [[nodiscard]] virtual bool solve(const std::vector<FloatT>& features, std::size_t numSamples,
                                   const std::vector<FloatT>& targets, std::size_t numOutputs,
                                   std::vector<FloatT>* weights) const = 0;
};

template <typename FloatT>
class BatchRidgeSolver final : public Solver<FloatT> {
 public:
  explicit BatchRidgeSolver(SolverOptions<FloatT> options = {});

  [[nodiscard]] bool solve(const std::vector<FloatT>& features, std::size_t numSamples,
                           const std::vector<FloatT>& targets, std::size_t numOutputs,
                           std::vector<FloatT>* weights) const override;

  [[nodiscard]] FloatT ridgeAlpha() const noexcept {
    return options_.ridgeAlpha;
  }
  [[nodiscard]] RidgeSolvePath solvePath() const noexcept {
    return options_.path;
  }
  [[nodiscard]] RidgeSolveMethod solveMethod() const noexcept {
    return options_.method;
  }

 private:
  SolverOptions<FloatT> options_;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_SOLVER_HPP_
