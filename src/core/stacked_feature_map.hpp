#ifndef FEATURE_ELM_CORE_STACKED_FEATURE_MAP_HPP_
#define FEATURE_ELM_CORE_STACKED_FEATURE_MAP_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "core/elm_ae.hpp"
#include "core/feature_map.hpp"

namespace feature_elm {

template <typename FloatT = float>
class StackedFeatureMap final : public FeatureMap<FloatT> {
 public:
  explicit StackedFeatureMap(std::size_t inputDim, std::vector<std::size_t> layerOutputDims,
                             ActivationKind activation, unsigned int seed = 42u,
                             FloatT ridgeAlpha = static_cast<FloatT>(1e-6),
                             Backend backend = Backend::kCpu);

  [[nodiscard]] std::size_t inputDim() const noexcept override {
    return inputDim_;
  }
  [[nodiscard]] std::size_t outputDim() const noexcept override {
    return layerOutputDims_.empty() ? inputDim_ : layerOutputDims_.back();
  }

  bool fit(const std::vector<FloatT>& data, std::size_t numSamples) override;
  bool transform(const std::vector<FloatT>& input, std::size_t numSamples,
                 std::vector<FloatT>* output) const override;

  [[nodiscard]] bool isFitted() const noexcept {
    return isFitted_;
  }
  [[nodiscard]] const std::vector<std::size_t>& layerOutputDims() const noexcept {
    return layerOutputDims_;
  }
  [[nodiscard]] const std::vector<std::unique_ptr<FeatureMap<FloatT>>>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] std::vector<std::unique_ptr<FeatureMap<FloatT>>>& layers() noexcept {
    return layers_;
  }
  [[nodiscard]] Backend backend() const noexcept {
    return backend_;
  }
  void setBackend(Backend backend) noexcept {
    backend_ = backend;
  }

  void reset() noexcept;

 private:
  std::size_t inputDim_;
  std::vector<std::size_t> layerOutputDims_;
  ActivationKind activation_;
  unsigned int seed_;
  FloatT ridgeAlpha_;
  bool isFitted_;
  Backend backend_;

  std::vector<std::unique_ptr<FeatureMap<FloatT>>> layers_;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_STACKED_FEATURE_MAP_HPP_