#ifndef FEATURE_ELM_CORE_FEATURE_MAP_HPP_
#define FEATURE_ELM_CORE_FEATURE_MAP_HPP_

#include <cstddef>
#include <vector>

namespace feature_elm {

enum class ActivationKind { kSigmoid, kTanh, kRelu };

template <typename FloatT = float>
class FeatureMap {
 public:
  virtual ~FeatureMap() = default;

  [[nodiscard]] virtual std::size_t inputDim() const = 0;
  [[nodiscard]] virtual std::size_t outputDim() const = 0;
  virtual bool fit(const std::vector<FloatT>& data, std::size_t numSamples) = 0;
  virtual bool transform(const std::vector<FloatT>& input, std::size_t numSamples,
                         std::vector<FloatT>* output) const = 0;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_FEATURE_MAP_HPP_
