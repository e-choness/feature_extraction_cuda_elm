#ifndef FEATURE_ELM_CORE_OS_CELM_HPP_
#define FEATURE_ELM_CORE_OS_CELM_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "core/elm.hpp"

namespace feature_elm {

/**
 * @class OsCelm
 * @brief Constrained Online Sequential Extreme Learning Machine (OS-CELM).
 *
 * This variant of OS-ELM applies a simple class-distance-based constraint
 * to the covariance update to bias the model toward outputs that separate
 * class labels in feature space.
 */
template <typename FloatT = double>
class OsCelm {
 public:
  explicit OsCelm(std::size_t numInputs, std::size_t numHiddenNodes,
                  ActivationFunction activation = ActivationFunction::kSigmoid,
                  FloatT constraintStrength = static_cast<FloatT>(1e-2));

  OsCelm(const OsCelm&) = delete;
  OsCelm& operator=(const OsCelm&) = delete;
  OsCelm(OsCelm&&) noexcept = default;
  OsCelm& operator=(OsCelm&&) noexcept = default;

  ~OsCelm() = default;

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

  void reset() noexcept;

 private:
  std::size_t numInputs_;
  std::size_t numHiddenNodes_;
  std::size_t numOutputs_;
  ActivationFunction activation_;
  FloatT constraintStrength_;
  bool isInitialized_;

  std::vector<FloatT> hiddenWeights_;
  std::vector<FloatT> hiddenBiases_;
  std::vector<FloatT> outputWeights_;
  std::vector<FloatT> covariance_;

  [[nodiscard]] bool computeHiddenOutput(const std::vector<FloatT>& input, std::size_t numSamples,
                                         std::vector<FloatT>* hiddenOutput) const;

  [[nodiscard]] bool updateRecursiveLeastSquares(const std::vector<FloatT>& H,
                                                 const std::vector<FloatT>& T,
                                                 std::size_t numSamples);

  [[nodiscard]] FloatT computeClassDistance(const std::vector<FloatT>& H,
                                            const std::vector<FloatT>& T,
                                            std::size_t numSamples) const;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_OS_CELM_HPP_
