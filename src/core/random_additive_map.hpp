#ifndef FEATURE_ELM_CORE_RANDOM_ADDITIVE_MAP_HPP_
#define FEATURE_ELM_CORE_RANDOM_ADDITIVE_MAP_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "core/feature_map.hpp"

namespace feature_elm {

class RandomAdditiveMap final : public FeatureMap {
 public:
  explicit RandomAdditiveMap(std::size_t inputDim, std::size_t outputDim, ActivationKind activation,
                             std::optional<unsigned int> seed = std::nullopt,
                             const std::vector<float>& weights = {},
                             const std::vector<float>& biases = {});

  [[nodiscard]] std::size_t inputDim() const noexcept override { return inputDim_; }
  [[nodiscard]] std::size_t outputDim() const noexcept override { return outputDim_; }

  bool fit(const std::vector<float>& data, std::size_t numSamples) override;
  bool transform(const std::vector<float>& input, std::size_t numSamples,
                 std::vector<float>* output) const override;

 private:
  std::size_t inputDim_;
  std::size_t outputDim_;
  ActivationKind activation_;
  std::vector<float> weights_;
  std::vector<float> biases_;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_RANDOM_ADDITIVE_MAP_HPP_