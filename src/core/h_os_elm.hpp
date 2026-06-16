#ifndef FEATURE_ELM_CORE_H_OS_ELM_HPP_
#define FEATURE_ELM_CORE_H_OS_ELM_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "core/elm.hpp"
#include "core/rls_solver.hpp"
#include "core/stacked_feature_map.hpp"

namespace feature_elm {

/**
 * @class HierarchicalOsElm
 * @brief Hierarchical OS-ELM with learned ELM-AE feature layers.
 *
 * Each hidden layer is an ELM-AE learned from the preceding layer, and the final
 * representation is trained online with recursive least squares.
 */
template <typename FloatT = double>
class HierarchicalOsElm {
 public:
  explicit HierarchicalOsElm(std::size_t numInputs,
                             const std::vector<std::size_t>& hiddenNodesPerLayer,
                             ActivationFunction activation = ActivationFunction::kSigmoid,
                             Backend backend = Backend::kCpu,
                             RlsOptions<FloatT> rlsOptions = RlsOptions<FloatT>{},
                             FloatT ridgeAlpha = static_cast<FloatT>(1e-6),
                             unsigned int seed = 42u);

  HierarchicalOsElm(const HierarchicalOsElm&) = delete;
  HierarchicalOsElm& operator=(const HierarchicalOsElm&) = delete;
  HierarchicalOsElm(HierarchicalOsElm&&) noexcept = default;
  HierarchicalOsElm& operator=(HierarchicalOsElm&&) noexcept = default;

  ~HierarchicalOsElm() = default;

  [[nodiscard]] bool initialize(const std::vector<FloatT>& data, const std::vector<FloatT>& targets,
                                std::size_t numSamples, std::size_t numOutputs);

  [[nodiscard]] bool update(const std::vector<FloatT>& newData,
                            const std::vector<FloatT>& newTargets, std::size_t numSamples);

  [[nodiscard]] std::optional<std::vector<FloatT>> predictBatch(const std::vector<FloatT>& testData,
                                                                std::size_t numSamples) const;

  [[nodiscard]] bool isInitialized() const noexcept {
    return isInitialized_;
  }
  [[nodiscard]] RlsOptions<FloatT> rlsOptions() const noexcept {
    return rlsSolver_.options();
  }
  [[nodiscard]] FloatT ridgeAlpha() const noexcept {
    return ridgeAlpha_;
  }
  [[nodiscard]] const StackedFeatureMap<FloatT>& featureStack() const noexcept {
    return featureStack_;
  }
  [[nodiscard]] StackedFeatureMap<FloatT>& featureStack() noexcept {
    return featureStack_;
  }
  [[nodiscard]] std::size_t numLayers() const noexcept {
    return hiddenNodesPerLayer_.size();
  }

  void reset() noexcept;

 private:
  std::size_t numInputs_;
  std::vector<std::size_t> hiddenNodesPerLayer_;
  ActivationFunction activation_;
  Backend backend_;
  FloatT ridgeAlpha_;
  unsigned int seed_;
  bool isInitialized_;
  std::size_t numOutputs_;

  StackedFeatureMap<FloatT> featureStack_;
  RlsSolver<FloatT> rlsSolver_;

  [[nodiscard]] bool computeHierarchicalFeatures(const std::vector<FloatT>& data,
                                                 std::size_t numSamples,
                                                 std::vector<FloatT>* features) const;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_H_OS_ELM_HPP_
