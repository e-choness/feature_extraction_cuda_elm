#include "core/rbf_features.hpp"

#include <cmath>
#include <numeric>
#include <random>

namespace feature_elm {

template <typename FloatT>
[[nodiscard]] bool computeRbfFeatures(const std::vector<FloatT>& input, std::size_t numSamples,
                                      const RbfParameters<FloatT>& params,
                                      std::vector<FloatT>* output) {
  if (input.empty() || params.centers.empty() || output == nullptr) {
    return false;
  }
  if (input.size() != numSamples * params.inputDim) {
    return false;
  }
  if (params.centers.size() != params.numCenters * params.inputDim) {
    return false;
  }

  output->assign(numSamples * params.numCenters, FloatT(0));
  FloatT widthSq = params.width * params.width;
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t center = 0; center < params.numCenters; ++center) {
      FloatT distSq = FloatT(0);
      for (std::size_t dim = 0; dim < params.inputDim; ++dim) {
        FloatT diff =
            input[sample * params.inputDim + dim] - params.centers[center * params.inputDim + dim];
        distSq += diff * diff;
      }
      (*output)[sample * params.numCenters + center] =
          std::exp(-distSq / (static_cast<FloatT>(2) * widthSq));
    }
  }
  return true;
}

template <typename FloatT>
[[nodiscard]] bool initializeRbfCentersRandom(std::size_t numCenters, std::size_t inputDim,
                                              RbfParameters<FloatT>* params, unsigned int seed) {
  if (params == nullptr) {
    return false;
  }
  params->numCenters = numCenters;
  params->inputDim = inputDim;
  params->centers.assign(numCenters * inputDim, FloatT(0));
  params->width = static_cast<FloatT>(1.0);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<FloatT> dis(static_cast<FloatT>(-1), static_cast<FloatT>(1));
  for (auto& center : params->centers) {
    center = dis(gen);
  }
  return true;
}

template class RbfParameters<float>;
template class RbfParameters<double>;
template bool computeRbfFeatures<float>(const std::vector<float>&, std::size_t,
                                        const RbfParameters<float>&, std::vector<float>*);
template bool computeRbfFeatures<double>(const std::vector<double>&, std::size_t,
                                         const RbfParameters<double>&, std::vector<double>*);
template bool initializeRbfCentersRandom<float>(std::size_t, std::size_t, RbfParameters<float>*,
                                                unsigned int);
template bool initializeRbfCentersRandom<double>(std::size_t, std::size_t, RbfParameters<double>*,
                                                 unsigned int);

}  // namespace feature_elm
