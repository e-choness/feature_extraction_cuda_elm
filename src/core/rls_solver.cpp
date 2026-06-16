#include "core/rls_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace feature_elm {

namespace {

template <typename FloatT>
[[nodiscard]] bool isPositiveFinite(FloatT value) {
  return value > FloatT(0) && std::isfinite(value);
}

template <typename FloatT>
[[nodiscard]] FloatT targetDistance(const std::vector<FloatT>& targets, std::size_t indexA,
                                    std::size_t indexB, std::size_t numOutputs) {
  FloatT sum = FloatT(0);
  for (std::size_t output = 0; output < numOutputs; ++output) {
    const FloatT diff =
        targets[indexA * numOutputs + output] - targets[indexB * numOutputs + output];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

}  // namespace

template <typename FloatT>
RlsSolver<FloatT>::RlsSolver(RlsOptions<FloatT> options)
    : options_(options), numFeatures_(0), numOutputs_(0), isInitialized_(false) {
  if (!isPositiveFinite(options_.regularization) || !std::isfinite(options_.regularization)) {
    options_.regularization = static_cast<FloatT>(1e-3);
  }
  if (!(options_.forgettingFactor > FloatT(0)) || options_.forgettingFactor > FloatT(1) ||
      !std::isfinite(options_.forgettingFactor)) {
    options_.forgettingFactor = static_cast<FloatT>(1);
  }
  if (!(options_.constraintStrength >= FloatT(0)) || !std::isfinite(options_.constraintStrength)) {
    options_.constraintStrength = FloatT(0);
  }
}

template <typename FloatT>
bool RlsSolver<FloatT>::initialize(const std::vector<FloatT>& features, std::size_t numSamples,
                                   const std::vector<FloatT>& targets, std::size_t numOutputs) {
  if (isInitialized_ || features.empty() || targets.empty() || numSamples == 0 || numOutputs == 0 ||
      features.size() % numSamples != 0 || targets.size() != numSamples * numOutputs) {
    return false;
  }

  numFeatures_ = features.size() / numSamples;
  numOutputs_ = numOutputs;
  weights_.assign(numFeatures_ * numOutputs_, FloatT(0));
  covariance_.assign(numFeatures_ * numFeatures_, FloatT(0));

  const FloatT inverseRegularization = static_cast<FloatT>(1) / options_.regularization;
  for (std::size_t i = 0; i < numFeatures_; ++i) {
    covariance_[i * numFeatures_ + i] = inverseRegularization;
  }

  isInitialized_ = true;
  if (!update(features, numSamples, targets)) {
    reset();
    return false;
  }
  return true;
}

template <typename FloatT>
bool RlsSolver<FloatT>::update(const std::vector<FloatT>& features, std::size_t numSamples,
                               const std::vector<FloatT>& targets) {
  if (!isInitialized_ || features.empty() || targets.empty() || numSamples == 0 ||
      features.size() != numSamples * numFeatures_ || targets.size() != numSamples * numOutputs_) {
    return false;
  }
  return updateRecursiveLeastSquares(features, numSamples, targets);
}

template <typename FloatT>
bool RlsSolver<FloatT>::updateRecursiveLeastSquares(const std::vector<FloatT>& features,
                                                    std::size_t numSamples,
                                                    const std::vector<FloatT>& targets) {
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    std::vector<FloatT> projectedCovariance(numFeatures_, FloatT(0));
    for (std::size_t i = 0; i < numFeatures_; ++i) {
      FloatT sum = FloatT(0);
      for (std::size_t j = 0; j < numFeatures_; ++j) {
        sum += covariance_[i * numFeatures_ + j] * features[sample * numFeatures_ + j];
      }
      projectedCovariance[i] = sum;
    }

    FloatT denominator = options_.forgettingFactor;
    for (std::size_t i = 0; i < numFeatures_; ++i) {
      denominator += features[sample * numFeatures_ + i] * projectedCovariance[i];
    }
    if (!isPositiveFinite(denominator)) {
      return false;
    }

    std::vector<FloatT> gain(numFeatures_, FloatT(0));
    for (std::size_t i = 0; i < numFeatures_; ++i) {
      gain[i] = projectedCovariance[i] / denominator;
    }

    for (std::size_t output = 0; output < numOutputs_; ++output) {
      FloatT error = targets[sample * numOutputs_ + output];
      for (std::size_t i = 0; i < numFeatures_; ++i) {
        error -= features[sample * numFeatures_ + i] * weights_[i * numOutputs_ + output];
      }
      for (std::size_t i = 0; i < numFeatures_; ++i) {
        weights_[i * numOutputs_ + output] += gain[i] * error;
      }
    }

    std::vector<FloatT> nextCovariance(covariance_.size(), FloatT(0));
    for (std::size_t i = 0; i < numFeatures_; ++i) {
      for (std::size_t j = 0; j < numFeatures_; ++j) {
        nextCovariance[i * numFeatures_ + j] =
            covariance_[i * numFeatures_ + j] - gain[i] * projectedCovariance[j];
      }
    }
    const FloatT inverseForgetting = static_cast<FloatT>(1) / options_.forgettingFactor;
    for (FloatT& value : nextCovariance) {
      value *= inverseForgetting;
    }

    if (options_.constraint == RlsConstraint::kClassDistance &&
        options_.constraintStrength > FloatT(0)) {
      const FloatT regularizer =
          computeClassDistance(features, targets, numSamples) * options_.constraintStrength;
      for (std::size_t i = 0; i < numFeatures_; ++i) {
        nextCovariance[i * numFeatures_ + i] += regularizer;
      }
    }

    covariance_.swap(nextCovariance);
  }
  return true;
}

template <typename FloatT>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FloatT RlsSolver<FloatT>::computeClassDistance(const std::vector<FloatT>& features,
                                               const std::vector<FloatT>& targets,
                                               std::size_t numSamples) const {
  if (numSamples < 2) {
    return FloatT(0);
  }

  FloatT totalDistance = FloatT(0);
  for (std::size_t sample = 1; sample < numSamples; ++sample) {
    FloatT featureDistance = FloatT(0);
    for (std::size_t i = 0; i < numFeatures_; ++i) {
      const FloatT diff =
          features[sample * numFeatures_ + i] - features[(sample - 1) * numFeatures_ + i];
      featureDistance += diff * diff;
    }
    totalDistance += targetDistance(targets, sample, sample - 1, numOutputs_) *
                     std::sqrt(featureDistance + std::numeric_limits<FloatT>::epsilon());
  }
  return totalDistance / static_cast<FloatT>(numSamples - 1);
}

template <typename FloatT>
void RlsSolver<FloatT>::reset() noexcept {
  numFeatures_ = 0;
  numOutputs_ = 0;
  isInitialized_ = false;
  weights_.clear();
  covariance_.clear();
}

template class RlsSolver<float>;
template class RlsSolver<double>;

}  // namespace feature_elm
