#ifndef FEATURE_ELM_CORE_ELM_HPP_
#define FEATURE_ELM_CORE_ELM_HPP_

#include <cstddef>
#include <optional>
#include <vector>

namespace feature_elm {

enum class ActivationFunction { kSigmoid, kRbf };

enum class Backend { kCpu, kGpu };

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
  explicit BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                    ActivationFunction activation = ActivationFunction::kSigmoid,
                    Backend backend = Backend::kCpu);

  BatchElm(std::size_t numInputs, std::size_t numHiddenNodes, ActivationFunction activation,
           Backend backend, const std::vector<FloatT>& hiddenWeights,
           const std::vector<FloatT>& hiddenBiases);

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

  // Hidden layer parameters (random, fixed after initialization)
  std::vector<FloatT> hiddenWeights_;  // Shape: (numInputs, numHiddenNodes)
  std::vector<FloatT> hiddenBiases_;   // Shape: (numHiddenNodes,)

  // Output layer parameters (learned during training)
  std::vector<FloatT> outputWeights_;  // Shape: (numHiddenNodes, numOutputs)
  Backend backend_;

  /**
   * @brief Initialize hidden layer weights and biases randomly.
   */
  void initializeHiddenLayer() noexcept;

  /**
   * @brief Compute hidden layer output (H matrix).
   *
   * @param input Input matrix of shape (numSamples, numInputs)
   * @param numSamples Number of samples
   * @return Hidden layer output H of shape (numSamples, numHiddenNodes)
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] std::vector<FloatT> computeHiddenOutput(const std::vector<FloatT>& input,
                                                        std::size_t numSamples) const;

  /**
   * @brief Apply activation function element-wise.
   *
   * @param x Input value
   * @return Activated value
   */
  [[nodiscard]] FloatT activate(FloatT x) const noexcept;

  /**
   * @brief Solve least-squares problem: H·β = T for β using normal equations.
   *
   * @param H Hidden layer output matrix (numSamples, numHiddenNodes)
   * @param T Target matrix (numSamples, numOutputs)
   * @param numSamples Number of samples
   * @param numOutputs Number of outputs
   * @return Output weights β of shape (numHiddenNodes, numOutputs), or empty on failure
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters,readability-identifier-naming)
  [[nodiscard]] std::vector<FloatT> solveLeastSquares(const std::vector<FloatT>& H,
                                                      const std::vector<FloatT>& T,
                                                      std::size_t numSamples,
                                                      std::size_t numOutputs);
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_ELM_HPP_
