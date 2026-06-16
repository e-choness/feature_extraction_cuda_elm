#include "core/rbf_map.hpp"

#include <cmath>
#include <limits>
#include <random>

#include "core/rbf_features.hpp"

namespace feature_elm {

template <typename FloatT>
RbfMap<FloatT>::RbfMap(std::size_t inputDim, std::size_t numCenters, FloatT width,
                       RbfCenterInit centerInit, unsigned int seed)
    : inputDim_(inputDim),
      numCenters_(numCenters),
      width_(width),
      centerInit_(centerInit),
      seed_(seed),
      centers_(numCenters * inputDim, FloatT(0)) {}

template <typename FloatT>
bool RbfMap<FloatT>::fit(const std::vector<FloatT>& data, std::size_t numSamples) {
  if (data.empty() || numSamples == 0 || numSamples < numCenters_) {
    return false;
  }

  if (centerInit_ == RbfCenterInit::kRandom) {
    std::mt19937 gen(seed_);
    std::uniform_real_distribution<FloatT> dis(std::numeric_limits<FloatT>::lowest(),
                                               std::numeric_limits<FloatT>::max());
    for (auto& center : centers_) {
      center = dis(gen);
    }
  } else {
    centers_.clear();
    centers_.resize(numCenters_ * inputDim_);

    for (std::size_t dim = 0; dim < inputDim_; ++dim) {
      FloatT minVal = std::numeric_limits<FloatT>::max();
      FloatT maxVal = std::numeric_limits<FloatT>::lowest();

      for (std::size_t i = 0; i < numSamples; ++i) {
        FloatT val = data[i * inputDim_ + dim];
        if (val < minVal)
          minVal = val;
        if (val > maxVal)
          maxVal = val;
      }

      std::mt19937 gen(static_cast<unsigned int>(seed_ + dim));
      std::uniform_real_distribution<FloatT> dis(minVal, maxVal);

      for (std::size_t center = 0; center < numCenters_; ++center) {
        centers_[center * inputDim_ + dim] = dis(gen);
      }
    }
  }

  return true;
}

template <typename FloatT>
bool RbfMap<FloatT>::transform(const std::vector<FloatT>& input, std::size_t numSamples,
                               std::vector<FloatT>* output) const {
  if (input.empty() || output == nullptr) {
    return false;
  }
  if (centers_.empty()) {
    return false;
  }

  RbfParameters<FloatT> params;
  params.inputDim = inputDim_;
  params.numCenters = numCenters_;
  params.centers = centers_;
  params.width = width_;

  return computeRbfFeatures(input, numSamples, params, output);
}

template class RbfMap<float>;
template class RbfMap<double>;

}  // namespace feature_elm