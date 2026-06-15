#ifndef FEATURE_ELM_CORE_ELM_HPP_
#define FEATURE_ELM_CORE_ELM_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "core/feature_map.hpp"
#include "core/random_additive_map.hpp"
#include "core/solver.hpp"

namespace feature_elm {

enum class ActivationFunction { kSigmoid, kTanh, kRelu };

/**
 * @class BatchElm
 * @brief Batch Extreme Learning Machine (ELM) with additive hidden nodes.
 *
 * Implements a single hidden layer feedforward network with:
 * - Random initialization of hidden layer weights and biases (additive nodes)
 * - Least-squares solution for output weights (using normal equations or QR/SVD)
 * - Optional GPU backend for training and prediction
 *
 * Template Parameters:
 * - FloatT: Floating point type (float, double, etc.)
 */
template <typename FloatT = double>
class BatchElm {
 public:
  /**
   * @param numInputs Input dimension
   * @param numHiddenNodes Number of hidden layer nodes
   * @param activation Activation function for hidden layer
   * @param backend Backend selection: CPU or GPU
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  explicit BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                    ActivationFunction activation = ActivationFunction::kSigmoid,
                    Backend backend = Backend::kCpu, FloatT ridgeAlpha = static_cast<FloatT>(1e-6));

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  BatchElm(std::size_t numInputs, std::size_t numHiddenNodes, ActivationFunction activation,
           Backend backend, const std::vector<FloatT>& hiddenWeights,
           const std::vector<FloatT>& hiddenBiases, FloatT ridgeAlpha = static_cast<FloatT>(1e-6));

  BatchElm(const BatchElm&) = delete;
  BatchElm& operator=(const BatchElm&) = delete;
  BatchElm(BatchElm&&) noexcept = default;
  BatchElm& operator=(BatchElm&&) noexcept = default;

  ~BatchElm() = default;

  /**
   * @brief Train the ELM on a batch of data using least-squares.
   *
   * @param trainData Matrix of shape (numSamples, numInputs) in row-major order
   * @param trainTargets Matrix of shape (numSamples, numOutputs) in row-major order
   * @return true if training succeeded, false otherwise
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] bool train(const std::vector<FloatT>& trainData,
                           const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                           std::size_t numOutputs);

  /**
   * @brief Predict on a single sample.
   *
   * @param input Input vector of size numInputs
   * @return Output vector (predictions) of size numOutputs, or std::nullopt on error
   */
  [[nodiscard]] std::optional<std::vector<FloatT>> predict(const std::vector<FloatT>& input) const;

  /**
   * @brief Batch predict on multiple samples.
   *
   * @param testData Matrix of shape (numSamples, numInputs) in row-major order
   * @param numSamples Number of test samples
   * @return Output matrix (numSamples * numOutputs) in row-major, or std::nullopt on error
   */
  [[nodiscard]] std::optional<std::vector<FloatT>> predictBatch(const std::vector<FloatT>& testData,
                                                                std::size_t numSamples) const;

  // Getters
  [[nodiscard]] std::size_t numInputs() const noexcept {
    return numInputs_;
  }
  [[nodiscard]] std::size_t numHiddenNodes() const noexcept {
    return numHiddenNodes_;
  }
  [[nodiscard]] std::size_t numOutputs() const noexcept {
    return numOutputs_;
  }
  [[nodiscard]] bool isTrained() const noexcept {
    return isTrained_;
  }
  [[nodiscard]] Backend backend() const noexcept {
    return backend_;
  }
  [[nodiscard]] FloatT ridgeAlpha() const noexcept {
    return ridgeAlpha_;
  }

  /**
   * @brief Reset the ELM (clear learned weights).
   */
  void reset() noexcept;

 private:
  std::size_t numInputs_;
  std::size_t numHiddenNodes_;
  std::size_t numOutputs_;
  ActivationFunction activation_;
  bool isTrained_;
  Backend backend_;
  FloatT ridgeAlpha_;

  RandomAdditiveMap<FloatT> featureMap_;
  BatchRidgeSolver<FloatT> solver_;

  // Output layer parameters (learned during training)
  std::vector<FloatT> outputWeights_;  // Shape: (numHiddenNodes, numOutputs)

  static ActivationKind activationKind(ActivationFunction activation) {
    switch (activation) {
      case ActivationFunction::kSigmoid:
        return ActivationKind::kSigmoid;
      case ActivationFunction::kTanh:
        return ActivationKind::kTanh;
      case ActivationFunction::kRelu:
        return ActivationKind::kRelu;
    }
    return ActivationKind::kSigmoid;
  }
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_ELM_HPP_