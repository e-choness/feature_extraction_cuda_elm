#ifndef FEATURE_ELM_CORE_ELM_AE_HPP_
#define FEATURE_ELM_CORE_ELM_AE_HPP_

#include <cstddef>
#include <vector>

#include "core/feature_map.hpp"
#include "core/solver.hpp"

namespace feature_elm {

template <typename FloatT = float>
class ElmAutoEncoderLayer final : public FeatureMap<FloatT> {
 public:
  explicit ElmAutoEncoderLayer(std::size_t inputDim, std::size_t outputDim,
                               ActivationKind activation, unsigned int seed = 42u,
                               FloatT ridgeAlpha = static_cast<FloatT>(1e-6));

  [[nodiscard]] std::size_t inputDim() const noexcept override {
    return inputDim_;
  }
  [[nodiscard]] std::size_t outputDim() const noexcept override {
    return outputDim_;
  }

  bool fit(const std::vector<FloatT>& data, std::size_t numSamples) override;
  bool transform(const std::vector<FloatT>& input, std::size_t numSamples,
                 std::vector<FloatT>* output) const override;

  [[nodiscard]] bool reconstruct(const std::vector<FloatT>& input, std::size_t numSamples,
                                 std::vector<FloatT>* reconstruction) const;
  [[nodiscard]] bool isFitted() const noexcept {
    return isFitted_;
  }
  [[nodiscard]] ActivationKind activation() const noexcept {
    return activation_;
  }
  [[nodiscard]] FloatT ridgeAlpha() const noexcept {
    return solver_.ridgeAlpha();
  }
  [[nodiscard]] const std::vector<FloatT>& inputWeights() const noexcept {
    return inputWeights_;
  }
  [[nodiscard]] const std::vector<FloatT>& biases() const noexcept {
    return biases_;
  }
  [[nodiscard]] const std::vector<FloatT>& outputWeights() const noexcept {
    return outputWeights_;
  }
  [[nodiscard]] const std::vector<FloatT>& encoderWeights() const noexcept {
    return encoderWeights_;
  }
  [[nodiscard]] const std::vector<FloatT>& encoderBiases() const noexcept {
    return encoderBiases_;
  }
  [[nodiscard]] Backend backend() const noexcept {
    return backend_;
  }

  void setBackend(Backend backend) noexcept;

 private:
  std::size_t inputDim_;
  std::size_t outputDim_;
  ActivationKind activation_;
  BatchRidgeSolver<FloatT> solver_;
  bool isFitted_;
  Backend backend_;

  std::vector<FloatT> inputWeights_;
  std::vector<FloatT> biases_;
  std::vector<FloatT> outputWeights_;
  std::vector<FloatT> encoderWeights_;
  std::vector<FloatT> encoderBiases_;

  [[nodiscard]] bool computeHiddenOutput(const std::vector<FloatT>& input, std::size_t numSamples,
                                         std::vector<FloatT>* hiddenOutput) const;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_ELM_AE_HPP_