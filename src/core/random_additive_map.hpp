#ifndef FEATURE_ELM_CORE_RANDOM_ADDITIVE_MAP_HPP_
#define FEATURE_ELM_CORE_RANDOM_ADDITIVE_MAP_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "core/feature_map.hpp"

namespace feature_elm {

template <typename FloatT = float>
class RandomAdditiveMap final : public FeatureMap<FloatT> {
 public:
  explicit RandomAdditiveMap(std::size_t inputDim, std::size_t outputDim, ActivationKind activation,
                             std::optional<unsigned int> seed = std::nullopt,
                             Backend backend = Backend::kCpu,
                             const std::vector<FloatT>& weights = {},
                             const std::vector<FloatT>& biases = {});

  [[nodiscard]] std::size_t inputDim() const noexcept override {
    return inputDim_;
  }
  [[nodiscard]] std::size_t outputDim() const noexcept override {
    return outputDim_;
  }
  [[nodiscard]] const std::vector<FloatT>& inputWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const std::vector<FloatT>& biases() const noexcept {
    return biases_;
  }

  bool fit(const std::vector<FloatT>& data, std::size_t numSamples) override;
  bool transform(const std::vector<FloatT>& input, std::size_t numSamples,
                 std::vector<FloatT>* output) const override;

  [[nodiscard]] Backend backend() const noexcept {
    return backend_;
  }
  void setBackend(Backend backend) noexcept {
    backend_ = backend;
  }

 private:
  std::size_t inputDim_;
  std::size_t outputDim_;
  ActivationKind activation_;
  Backend backend_;
  std::vector<FloatT> weights_;
  std::vector<FloatT> biases_;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_RANDOM_ADDITIVE_MAP_HPP_