#ifndef FEATURE_ELM_CORE_FEATURE_MAP_HPP_
#define FEATURE_ELM_CORE_FEATURE_MAP_HPP_

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace feature_elm {

enum class Backend { kCpu, kGpu };

enum class ActivationKind { kSigmoid, kTanh, kRelu };

[[nodiscard]] inline std::optional<std::size_t> checkedMatrixSize(std::size_t rows,
                                                                  std::size_t cols) noexcept {
  if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
    return std::nullopt;
  }
  return rows * cols;
}

template <typename FloatT = float>
class FeatureMap {
 public:
  virtual ~FeatureMap() = default;

  [[nodiscard]] virtual std::size_t inputDim() const = 0;
  [[nodiscard]] virtual std::size_t outputDim() const = 0;
  virtual bool fit(const std::vector<FloatT>& data, std::size_t numSamples) = 0;
  virtual bool transform(const std::vector<FloatT>& input, std::size_t numSamples,
                         std::vector<FloatT>* output) const = 0;
  virtual void setBackend(Backend /*backend*/) noexcept {}
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_FEATURE_MAP_HPP_