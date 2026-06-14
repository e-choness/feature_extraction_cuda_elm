#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "core/solver.hpp"

namespace feature_elm {

namespace {

template <typename FloatT>
void addScaledIdentity(std::vector<FloatT>* matrix, std::size_t dim, FloatT scale) {
  for (std::size_t i = 0; i < dim; ++i) {
    (*matrix)[i * dim + i] += scale;
  }
}

template <typename FloatT>
[[nodiscard]] bool solveSpdCholesky(const std::vector<FloatT>& matrix,
                                    const std::vector<FloatT>& rhs, std::size_t dim,
                                    std::size_t numRhs, std::vector<FloatT>* solution) {
  if (matrix.size() != dim * dim || rhs.size() != dim * numRhs || solution == nullptr) {
    return false;
  }

  std::vector<FloatT> lower(dim * dim, FloatT(0));
  for (std::size_t i = 0; i < dim; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      FloatT sum = matrix[i * dim + j];
      for (std::size_t k = 0; k < j; ++k) {
        sum -= lower[i * dim + k] * lower[j * dim + k];
      }

      if (i == j) {
        if (!(sum > FloatT(0)) || !std::isfinite(sum)) {
          return false;
        }
        lower[i * dim + j] = std::sqrt(sum);
      } else {
        lower[i * dim + j] = sum / lower[j * dim + j];
      }
    }
  }

  solution->assign(dim * numRhs, FloatT(0));
  std::vector<FloatT> intermediate(dim * numRhs, FloatT(0));

  for (std::size_t rhsIndex = 0; rhsIndex < numRhs; ++rhsIndex) {
    for (std::size_t row = 0; row < dim; ++row) {
      FloatT sum = rhs[row * numRhs + rhsIndex];
      for (std::size_t col = 0; col < row; ++col) {
        sum -= lower[row * dim + col] * intermediate[col * numRhs + rhsIndex];
      }
      intermediate[row * numRhs + rhsIndex] = sum / lower[row * dim + row];
    }

    for (std::size_t row = dim; row > 0; --row) {
      const std::size_t i = row - 1;
      FloatT sum = intermediate[i * numRhs + rhsIndex];
      for (std::size_t col = i + 1; col < dim; ++col) {
        sum -= lower[col * dim + i] * (*solution)[col * numRhs + rhsIndex];
      }
      (*solution)[i * numRhs + rhsIndex] = sum / lower[i * dim + i];
    }
  }

  return true;
}

template <typename FloatT>
[[nodiscard]] bool solveRegularizedQr(const std::vector<FloatT>& features,
                                      const std::vector<FloatT>& targets, std::size_t numSamples,
                                      std::size_t numFeatures, std::size_t numOutputs,
                                      FloatT ridgeAlpha, std::vector<FloatT>* weights) {
  if (features.size() != numSamples * numFeatures || targets.size() != numSamples * numOutputs ||
      weights == nullptr) {
    return false;
  }
  if (!(ridgeAlpha >= FloatT(0)) || !std::isfinite(ridgeAlpha)) {
    return false;
  }

  const std::size_t augmentedRows = numSamples + numFeatures;
  std::vector<FloatT> augmented(augmentedRows * numFeatures, FloatT(0));
  std::vector<FloatT> rhs(augmentedRows * numOutputs, FloatT(0));

  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t feature = 0; feature < numFeatures; ++feature) {
      augmented[sample * numFeatures + feature] = features[sample * numFeatures + feature];
    }
    for (std::size_t output = 0; output < numOutputs; ++output) {
      rhs[sample * numOutputs + output] = targets[sample * numOutputs + output];
    }
  }

  const FloatT sqrtAlpha = std::sqrt(ridgeAlpha);
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    augmented[(numSamples + feature) * numFeatures + feature] = sqrtAlpha;
  }

  std::vector<FloatT> r = augmented;
  std::vector<FloatT> transformedRhs = rhs;

  for (std::size_t col = 0; col < numFeatures; ++col) {
    FloatT norm = FloatT(0);
    for (std::size_t row = col; row < augmentedRows; ++row) {
      norm = std::hypot(norm, r[row * numFeatures + col]);
    }
    if (!(norm > FloatT(0)) || !std::isfinite(norm)) {
      return false;
    }

    const FloatT sign = r[col * numFeatures + col] >= FloatT(0) ? FloatT(1) : FloatT(-1);
    std::vector<FloatT> householder(augmentedRows - col);
    householder[0] = r[col * numFeatures + col] + sign * norm;
    for (std::size_t row = col + 1; row < augmentedRows; ++row) {
      householder[row - col] = r[row * numFeatures + col];
    }

    FloatT householderNorm = FloatT(0);
    for (FloatT value : householder) {
      householderNorm = std::hypot(householderNorm, value);
    }
    if (!(householderNorm > FloatT(0)) || !std::isfinite(householderNorm)) {
      continue;
    }
    for (FloatT& value : householder) {
      value /= householderNorm;
    }

    for (std::size_t feature = col; feature < numFeatures; ++feature) {
      FloatT dot = FloatT(0);
      for (std::size_t row = col; row < augmentedRows; ++row) {
        dot += householder[row - col] * r[row * numFeatures + feature];
      }
      for (std::size_t row = col; row < augmentedRows; ++row) {
        r[row * numFeatures + feature] -= FloatT(2) * householder[row - col] * dot;
      }
    }

    for (std::size_t output = 0; output < numOutputs; ++output) {
      FloatT dot = FloatT(0);
      for (std::size_t row = col; row < augmentedRows; ++row) {
        dot += householder[row - col] * transformedRhs[row * numOutputs + output];
      }
      for (std::size_t row = col; row < augmentedRows; ++row) {
        transformedRhs[row * numOutputs + output] -= FloatT(2) * householder[row - col] * dot;
      }
    }
  }

  weights->assign(numFeatures * numOutputs, FloatT(0));
  for (std::size_t output = 0; output < numOutputs; ++output) {
    for (std::size_t row = numFeatures; row > 0; --row) {
      const std::size_t i = row - 1;
      FloatT sum = transformedRhs[i * numOutputs + output];
      for (std::size_t col = i + 1; col < numFeatures; ++col) {
        sum -= r[i * numFeatures + col] * (*weights)[col * numOutputs + output];
      }
      const FloatT diagonal = r[i * numFeatures + i];
      if (std::abs(diagonal) < std::numeric_limits<FloatT>::epsilon() || !std::isfinite(diagonal)) {
        return false;
      }
      (*weights)[i * numOutputs + output] = sum / diagonal;
    }
  }

  return true;
}

}  // namespace

template <typename FloatT>
BatchRidgeSolver<FloatT>::BatchRidgeSolver(SolverOptions<FloatT> options) : options_(options) {
  if (!(options_.ridgeAlpha > FloatT(0)) || !std::isfinite(options_.ridgeAlpha)) {
    options_.ridgeAlpha = static_cast<FloatT>(1e-6);
  }
}

template <typename FloatT>
bool BatchRidgeSolver<FloatT>::solve(const std::vector<FloatT>& features, std::size_t numSamples,
                                     const std::vector<FloatT>& targets, std::size_t numOutputs,
                                     std::vector<FloatT>* weights) const {
  if (features.empty() || targets.empty() || weights == nullptr || numSamples == 0 ||
      numOutputs == 0 || features.size() % numSamples != 0 ||
      targets.size() != numSamples * numOutputs) {
    return false;
  }

  const std::size_t numFeatures = features.size() / numSamples;
  if (numFeatures == 0) {
    return false;
  }

  if (options_.method == RidgeSolveMethod::kHouseholderQr) {
    return solveRegularizedQr(features, targets, numSamples, numFeatures, numOutputs,
                              options_.ridgeAlpha, weights);
  }

  const bool useDual = options_.path == RidgeSolvePath::kDual ||
                       (options_.path == RidgeSolvePath::kAuto && numSamples < numFeatures);
  if (useDual) {
    std::vector<FloatT> gram(numSamples * numSamples, FloatT(0));
    for (std::size_t sampleI = 0; sampleI < numSamples; ++sampleI) {
      for (std::size_t sampleJ = 0; sampleJ < numSamples; ++sampleJ) {
        FloatT sum = FloatT(0);
        for (std::size_t feature = 0; feature < numFeatures; ++feature) {
          sum +=
              features[sampleI * numFeatures + feature] * features[sampleJ * numFeatures + feature];
        }
        gram[sampleI * numSamples + sampleJ] = sum;
      }
    }
    addScaledIdentity(&gram, numSamples, options_.ridgeAlpha);

    std::vector<FloatT> gamma;
    if (!solveSpdCholesky(gram, targets, numSamples, numOutputs, &gamma)) {
      return false;
    }

    weights->assign(numFeatures * numOutputs, FloatT(0));
    for (std::size_t feature = 0; feature < numFeatures; ++feature) {
      for (std::size_t output = 0; output < numOutputs; ++output) {
        FloatT sum = FloatT(0);
        for (std::size_t sample = 0; sample < numSamples; ++sample) {
          sum += features[sample * numFeatures + feature] * gamma[sample * numOutputs + output];
        }
        (*weights)[feature * numOutputs + output] = sum;
      }
    }
    return true;
  }

  std::vector<FloatT> normal(numFeatures * numFeatures, FloatT(0));
  for (std::size_t featureI = 0; featureI < numFeatures; ++featureI) {
    for (std::size_t featureJ = 0; featureJ < numFeatures; ++featureJ) {
      FloatT sum = FloatT(0);
      for (std::size_t sample = 0; sample < numSamples; ++sample) {
        sum +=
            features[sample * numFeatures + featureI] * features[sample * numFeatures + featureJ];
      }
      normal[featureI * numFeatures + featureJ] = sum;
    }
  }
  addScaledIdentity(&normal, numFeatures, options_.ridgeAlpha);

  std::vector<FloatT> rhs(numFeatures * numOutputs, FloatT(0));
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    for (std::size_t output = 0; output < numOutputs; ++output) {
      FloatT sum = FloatT(0);
      for (std::size_t sample = 0; sample < numSamples; ++sample) {
        sum += features[sample * numFeatures + feature] * targets[sample * numOutputs + output];
      }
      rhs[feature * numOutputs + output] = sum;
    }
  }

  return solveSpdCholesky(normal, rhs, numFeatures, numOutputs, weights);
}

template class BatchRidgeSolver<float>;
template class BatchRidgeSolver<double>;

}  // namespace feature_elm
