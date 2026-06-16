#ifndef FEATURE_ELM_CORE_RBF_FEATURES_HPP_
#define FEATURE_ELM_CORE_RBF_FEATURES_HPP_

#include <cstddef>
#include <vector>

namespace feature_elm {

/**
 * @brief Parameters for RBF feature computation.
 * @tparam FloatT Floating point type
 */
template <typename FloatT = double>
struct RbfParameters {
  std::size_t inputDim = 0;                 ///< Input dimension
  std::size_t numCenters = 0;               ///< Number of RBF centers
  std::vector<FloatT> centers;              ///< Centers in row-major order (numCenters x inputDim)
  FloatT width = static_cast<FloatT>(1.0);  ///< Width parameter sigma
};

/**
 * @brief Computes RBF features for input data.
 * @param input Input matrix (numSamples x inputDim) in row-major order
 * @param numSamples Number of samples
 * @param params RBF parameters including centers and width
 * @param output Output matrix (numSamples x numCenters) in row-major order
 * @return true on success, false on invalid input
 */
template <typename FloatT = double>
[[nodiscard]] bool computeRbfFeatures(const std::vector<FloatT>& input, std::size_t numSamples,
                                      const RbfParameters<FloatT>& params,
                                      std::vector<FloatT>* output);

/**
 * @brief Initializes RBF centers randomly within data range.
 * @param numCenters Number of centers to initialize
 * @param inputDim Input dimension
 * @param params Output parameters (centers and width will be set)
 * @param seed Random seed (default 42)
 * @return true on success, false on invalid input
 */
template <typename FloatT = double>
[[nodiscard]] bool initializeRbfCentersRandom(std::size_t numCenters, std::size_t inputDim,
                                              RbfParameters<FloatT>* params,
                                              unsigned int seed = 42);

/**
 * @brief Initializes RBF centers using k-means++ style selection from data.
 * @param numCenters Number of centers to select
 * @param inputDim Input dimension
 * @param data Input data matrix (numSamples x inputDim) in row-major order
 * @param numSamples Number of samples in data
 * @param params Output parameters (centers and width will be set)
 * @param seed Random seed (default 42)
 * @return true on success, false on invalid input
 */
template <typename FloatT = double>
[[nodiscard]] bool initializeRbfCentersKMeans(std::size_t numCenters, std::size_t inputDim,
                                              const std::vector<FloatT>& data,
                                              std::size_t numSamples, RbfParameters<FloatT>* params,
                                              unsigned int seed = 42);

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_RBF_FEATURES_HPP_
