#include "core/identity_map.hpp"

namespace feature_elm {

template <typename FloatT>
IdentityMap<FloatT>::IdentityMap(std::size_t dim) : dim_(dim) {}

template <typename FloatT>
bool IdentityMap<FloatT>::transform(const std::vector<FloatT>& input, std::size_t numSamples,
                                    std::vector<FloatT>* output) const {
  if (input.empty() || output == nullptr || input.size() != numSamples * dim_) {
    return false;
  }
  *output = input;
  return true;
}

}  // namespace feature_elm