#include "core/rls_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "core/parallel.hpp"

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
RlsSolver<FloatT>::RlsSolver(RlsOptions<FloatT> options, Backend backend)
    : options_(options), numFeatures_(0), numOutputs_(0), isInitialized_(false), backend_(backend) {
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

  // Block updates equal sequential rank-1 updates only without per-sample forgetting or constraint
  // terms, so FOS-ELM and OS-CELM settings stay on the CPU even with a GPU backend.
  const bool plainRls =
      options_.forgettingFactor == FloatT(1) &&
      (options_.constraint == RlsConstraint::kNone || options_.constraintStrength == FloatT(0));
  if (backend_ == Backend::kGpu && plainRls) {
    gpu_ = cuda_backend::createRlsGpu<FloatT>(numFeatures_, numOutputs_, options_.regularization);
  }

  if (gpu_ == nullptr) {
    weights_.assign(numFeatures_ * numOutputs_, FloatT(0));
    covariance_.assign(numFeatures_ * numFeatures_, FloatT(0));
    const FloatT inverseRegularization = static_cast<FloatT>(1) / options_.regularization;
    for (std::size_t i = 0; i < numFeatures_; ++i) {
      covariance_[i * numFeatures_ + i] = inverseRegularization;
    }
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
  if (gpu_ != nullptr) {
    weightsStale_ = true;
    covarianceStale_ = true;
    return cuda_backend::updateRlsGpu(*gpu_, features.data(), numSamples, targets.data());
  }
  return updateRecursiveLeastSquares(features, numSamples, targets);
}

template <typename FloatT>
const std::vector<FloatT>& RlsSolver<FloatT>::weights() const {
  if (weightsStale_ && gpu_ != nullptr) {
    if (!cuda_backend::downloadRlsWeights(*gpu_, &weights_)) {
      weights_.clear();
    }
    weightsStale_ = false;
  }
  return weights_;
}

template <typename FloatT>
const std::vector<FloatT>& RlsSolver<FloatT>::covariance() const {
  if (covarianceStale_ && gpu_ != nullptr) {
    if (!cuda_backend::downloadRlsCovariance(*gpu_, &covariance_)) {
      covariance_.clear();
    }
    covarianceStale_ = false;
  }
  return covariance_;
}

template <typename FloatT>
bool RlsSolver<FloatT>::updateRecursiveLeastSquares(const std::vector<FloatT>& features,
                                                    std::size_t numSamples,
                                                    const std::vector<FloatT>& targets) {
  // The class-distance term depends on the whole chunk, not on the sample, so compute it once.
  const bool applyConstraint = options_.constraint == RlsConstraint::kClassDistance &&
                               options_.constraintStrength > FloatT(0);
  const FloatT regularizer = applyConstraint ? computeClassDistance(features, targets, numSamples) *
                                                   options_.constraintStrength
                                             : FloatT(0);
  const FloatT inverseForgetting = static_cast<FloatT>(1) / options_.forgettingFactor;
  const std::size_t n = numFeatures_;
  // Two parallel regions per sample: only worth it for large covariances (barriers are expensive,
  // especially on hybrid/SMT CPUs).
  [[maybe_unused]] const bool parallel = n * n > (std::size_t{1} << 20);
  std::vector<FloatT> projectedCovariance(n, FloatT(0));
  std::vector<FloatT> gain(n, FloatT(0));

  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    const FloatT* x = features.data() + sample * n;
    FEATURE_ELM_OMP(omp parallel for schedule(static) if (parallel))
    for (std::size_t i = 0; i < n; ++i) {
      const FloatT* row = covariance_.data() + i * n;
      FloatT sum = FloatT(0);
      for (std::size_t j = 0; j < n; ++j) {
        sum += row[j] * x[j];
      }
      projectedCovariance[i] = sum;
    }

    FloatT denominator = options_.forgettingFactor;
    for (std::size_t i = 0; i < n; ++i) {
      denominator += x[i] * projectedCovariance[i];
    }
    if (!isPositiveFinite(denominator)) {
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      gain[i] = projectedCovariance[i] / denominator;
    }

    for (std::size_t output = 0; output < numOutputs_; ++output) {
      FloatT error = targets[sample * numOutputs_ + output];
      for (std::size_t i = 0; i < n; ++i) {
        error -= x[i] * weights_[i * numOutputs_ + output];
      }
      for (std::size_t i = 0; i < n; ++i) {
        weights_[i * numOutputs_ + output] += gain[i] * error;
      }
    }

    // In place: row i needs only its own old values, the gain and the projection, all computed
    // above. Each entry gets the same two operations as before (subtract, then scale).
    FEATURE_ELM_OMP(omp parallel for schedule(static) if (parallel))
    for (std::size_t i = 0; i < n; ++i) {
      FloatT* row = covariance_.data() + i * n;
      const FloatT gi = gain[i];
      for (std::size_t j = 0; j < n; ++j) {
        row[j] = (row[j] - gi * projectedCovariance[j]) * inverseForgetting;
      }
    }
    if (applyConstraint) {
      for (std::size_t i = 0; i < n; ++i) {
        covariance_[i * n + i] += regularizer;
      }
    }
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
  gpu_.reset();
  weightsStale_ = false;
  covarianceStale_ = false;
}

template class RlsSolver<float>;
template class RlsSolver<double>;

}  // namespace feature_elm
