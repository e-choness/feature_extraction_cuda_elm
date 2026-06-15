#ifndef FEATURE_ELM_CORE_RBF_MAP_HPP_
#define FEATURE_ELM_CORE_RBF_MAP_HPP_

#include <cstddef>
#include <vector>

#include "core/feature_map.hpp"

namespace feature_elm {

enum class RbfCenterInit { kRandom, kKMeans };

template <typename FloatT = float>
class RbfMap final : public FeatureMap<FloatT> {
 public:
  /**
   * @brief Constructs an RBF feature map.
   * @param inputDim Input dimension
   * @param numCenters Number of RBF centers
   * @param width Width parameter sigma (default 1.0)
   * @param centerInit Center initialization method (random or k-means)
   * @param seed Random seed for reproducibility
   */
  explicit RbfMap(std::size_t inputDim, std::size_t numCenters,
                  FloatT width = static_cast<FloatT>(1.0),
                  RbfCenterInit centerInit = RbfCenterInit::kRandom, unsigned int seed = 42);

  [[nodiscard]] std::size_t inputDim() const noexcept override {
    return inputDim_;
  }
  [[nodiscard]] std::size_t outputDim() const noexcept override {
    return numCenters_;
  }

  /**
   * @brief Initializes centers from data using the specified method.
   * @param data Input data matrix (numSamples x inputDim) in row-major order
   * @param numSamples Number of samples
   * @return true on success, false on invalid input
   */
  bool fit(const std::vector<FloatT>& data, std::size_t numSamples) override;

  /**
   * @brief Transforms input to RBF features: phi_i(x) = exp(-||x - c_i||^2 / (2*sigma^2)).
   * @param input Input matrix (numSamples x inputDim) in row-major order
   * @param numSamples Number of samples
   * @param output Output matrix (numSamples x numCenters) in row-major order
   * @return true on success, false on invalid input
   */
  bool transform(const std::vector<FloatT>& input, std::size_t numSamples,
                 std::vector<FloatT>* output) const override;

  /** @return Centers in row-major order (numCenters x inputDim) */
  [[nodiscard]] const std::vector<FloatT>& centers() const noexcept {
    return centers_;
  }
  /** @return Width parameter sigma */
  [[nodiscard]] FloatT width() const noexcept {
    return width_;
  }

 private:
  std::size_t inputDim_;
  std::size_t numCenters_;
  FloatT width_;
  RbfCenterInit centerInit_;
  unsigned int seed_;
  std::vector<FloatT> centers_;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_RBF_MAP_HPP_