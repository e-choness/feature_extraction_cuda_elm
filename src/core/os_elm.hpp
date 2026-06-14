#ifndef FEATURE_ELM_CORE_OS_ELM_HPP_
#define FEATURE_ELM_CORE_OS_ELM_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "core/elm.hpp"
#include "core/rls_solver.hpp"

namespace feature_elm {

/**
 * @class OsElm
 * @brief Online Sequential Extreme Learning Machine (OS-ELM) for additive hidden nodes.
 *
 * Supports incremental updates to output weights as new samples arrive.
 */
template <typename FloatT = double>
class OsElm {
 public:
  explicit OsElm(std::size_t numInputs, std::size_t numHiddenNodes,
                 ActivationFunction activation = ActivationFunction::kSigmoid,
                 Backend backend = Backend::kCpu,
                 RlsOptions<FloatT> rlsOptions = RlsOptions<FloatT>{});

  OsElm(std::size_t numInputs, std::size_t numHiddenNodes, ActivationFunction activation,
        Backend backend, const std::vector<FloatT>& hiddenWeights,
        const std::vector<FloatT>& hiddenBiases,
        RlsOptions<FloatT> rlsOptions = RlsOptions<FloatT>{});

  OsElm(const OsElm&) = delete;
  OsElm& operator=(const OsElm&) = delete;
  OsElm(OsElm&&) noexcept = default;
  OsElm& operator=(OsElm&&) noexcept = default;

  ~OsElm() = default;

  [[nodiscard]] bool initialize(const std::vector<FloatT>& initialData,
                                const std::vector<FloatT>& initialTargets, std::size_t numSamples,
                                std::size_t numOutputs);

  [[nodiscard]] bool update(const std::vector<FloatT>& newData,
                            const std::vector<FloatT>& newTargets, std::size_t numSamples);

  [[nodiscard]] std::optional<std::vector<FloatT>> predict(const std::vector<FloatT>& input) const;

  [[nodiscard]] std::optional<std::vector<FloatT>> predictBatch(const std::vector<FloatT>& testData,
                                                                std::size_t numSamples) const;

  [[nodiscard]] std::size_t numInputs() const noexcept {
    return numInputs_;
  }
  [[nodiscard]] std::size_t numHiddenNodes() const noexcept {
    return numHiddenNodes_;
  }
  [[nodiscard]] std::size_t numOutputs() const noexcept {
    return numOutputs_;
  }
  [[nodiscard]] bool isInitialized() const noexcept {
    return isInitialized_;
  }
  [[nodiscard]] Backend backend() const noexcept {
    return backend_;
  }
  [[nodiscard]] RlsOptions<FloatT> rlsOptions() const noexcept {
    return rlsSolver_.options();
  }

  void reset() noexcept;

 private:
  std::size_t numInputs_;
  std::size_t numHiddenNodes_;
  std::size_t numOutputs_;
  ActivationFunction activation_;
  Backend backend_;
  bool isInitialized_;

  std::vector<FloatT> hiddenWeights_;
  std::vector<FloatT> hiddenBiases_;
  RandomAdditiveMap<FloatT> featureMap_;
  RlsSolver<FloatT> rlsSolver_;

  [[nodiscard]] bool computeHiddenOutput(const std::vector<FloatT>& input, std::size_t numSamples,
                                         std::vector<FloatT>* hiddenOutput) const;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_OS_ELM_HPP_
