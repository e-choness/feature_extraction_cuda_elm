#ifndef FEATURE_ELM_CORE_IDENTITY_MAP_HPP_
#define FEATURE_ELM_CORE_IDENTITY_MAP_HPP_

#include <cstddef>
#include <vector>

#include "core/feature_map.hpp"

namespace feature_elm {

template <typename FloatT>
class IdentityMap final : public FeatureMap<FloatT> {
 public:
  explicit IdentityMap(std::size_t dim);

  [[nodiscard]] std::size_t inputDim() const noexcept { return dim_; }
  [[nodiscard]] std::size_t outputDim() const noexcept { return dim_; }

  bool fit(const std::vector<FloatT>& /*data*/, std::size_t /*numSamples*/) override {
    return true;
  }
  bool transform(const std::vector<FloatT>& input, std::size_t numSamples,
                 std::vector<FloatT>* output) const override;

 private:
  std::size_t dim_;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_IDENTITY_MAP_HPP_