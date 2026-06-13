#ifndef FEATURE_ELM_CORE_RBF_FEATURES_HPP_
#define FEATURE_ELM_CORE_RBF_FEATURES_HPP_

#include <cstddef>
#include <vector>

namespace feature_elm {

template <typename FloatT = double>
struct RbfParameters {
  std::size_t inputDim = 0;
  std::size_t numCenters = 0;
  std::vector<FloatT> centers;  // Row-major: numCenters x inputDim
  FloatT width = static_cast<FloatT>(1.0);
};

template <typename FloatT = double>
[[nodiscard]] bool computeRbfFeatures(const std::vector<FloatT>& input,
                                      std::size_t numSamples,
                                      const RbfParameters<FloatT>& params,
                                      std::vector<FloatT>* output);

template <typename FloatT = double>
[[nodiscard]] bool initializeRbfCentersRandom(
    std::size_t numCenters,
    std::size_t inputDim,
    RbfParameters<FloatT>* params,
    unsigned int seed = 42);

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_RBF_FEATURES_HPP_
