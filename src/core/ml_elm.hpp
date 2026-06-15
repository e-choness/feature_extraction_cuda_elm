#ifndef FEATURE_ELM_CORE_ML_ELM_HPP_
#define FEATURE_ELM_CORE_ML_ELM_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "core/elm.hpp"
#include "core/solver.hpp"
#include "core/stacked_feature_map.hpp"

namespace feature_elm {

template <typename FloatT = double>
class MlElm {
 public:
  explicit MlElm(std::size_t numInputs, const std::vector<std::size_t>& hiddenNodesPerLayer,
                 ActivationFunction activation = ActivationFunction::kSigmoid,
                 Backend backend = Backend::kCpu, FloatT ridgeAlpha = static_cast<FloatT>(1e-6),
                 unsigned int seed = 42u);

  MlElm(const MlElm&) = delete;
  MlElm& operator=(const MlElm&) = delete;
  MlElm(MlElm&&) noexcept = default;
  MlElm& operator=(MlElm&&) noexcept = default;

  ~MlElm() = default;

  [[nodiscard]] bool train(const std::vector<FloatT>& trainData,
                           const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                           std::size_t numOutputs);

  [[nodiscard]] std::optional<std::vector<FloatT>> predict(const std::vector<FloatT>& input) const;
  [[nodiscard]] std::optional<std::vector<FloatT>> predictBatch(const std::vector<FloatT>& testData,
                                                                std::size_t numSamples) const;

  [[nodiscard]] std::size_t numInputs() const noexcept {
    return numInputs_;
  }
  [[nodiscard]] std::size_t numOutputs() const noexcept {
    return numOutputs_;
  }
  [[nodiscard]] std::size_t finalFeatureDim() const noexcept {
    return featureStack_.outputDim();
  }
  [[nodiscard]] bool isTrained() const noexcept {
    return isTrained_;
  }
  [[nodiscard]] Backend backend() const noexcept {
    return backend_;
  }
  [[nodiscard]] ActivationFunction activation() const noexcept {
    return activation_;
  }
  [[nodiscard]] FloatT ridgeAlpha() const noexcept {
    return ridgeAlpha_;
  }
  [[nodiscard]] const std::vector<std::size_t>& hiddenNodesPerLayer() const noexcept {
    return hiddenNodesPerLayer_;
  }
  [[nodiscard]] const StackedFeatureMap<FloatT>& featureStack() const noexcept {
    return featureStack_;
  }
  [[nodiscard]] const BatchRidgeSolver<FloatT>& solver() const noexcept {
    return solver_;
  }
  [[nodiscard]] const std::vector<FloatT>& outputWeights() const noexcept {
    return outputWeights_;
  }

  void reset() noexcept;

 private:
  std::size_t numInputs_;
  std::size_t numOutputs_;
  std::vector<std::size_t> hiddenNodesPerLayer_;
  ActivationFunction activation_;
  Backend backend_;
  FloatT ridgeAlpha_;
  unsigned int seed_;
  bool isTrained_;

  StackedFeatureMap<FloatT> featureStack_;
  BatchRidgeSolver<FloatT> solver_;
  std::vector<FloatT> outputWeights_;

  [[nodiscard]] static StackedFeatureMap<FloatT> makeFeatureStack(
      std::size_t numInputs, const std::vector<std::size_t>& hiddenNodesPerLayer,
      ActivationFunction activation, unsigned int seed, FloatT ridgeAlpha);
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_ML_ELM_HPP_
