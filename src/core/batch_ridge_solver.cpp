#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "core/parallel.hpp"
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

  // Right-looking Cholesky: after column k is final, subtract its contribution from the trailing
  // lower triangle in parallel (each thread owns whole rows). Every entry still receives the
  // subtractions for k = 0, 1, ... in order followed by the division, exactly as in the textbook
  // row-by-row algorithm, so the factor is bit-identical to a serial run.
  std::vector<FloatT> lower(dim * dim, FloatT(0));
  for (std::size_t i = 0; i < dim; ++i) {
    std::copy_n(matrix.begin() + static_cast<std::ptrdiff_t>(i * dim), i + 1,
                lower.begin() + static_cast<std::ptrdiff_t>(i * dim));
  }
  // Blocked by panels of kPanel columns so the parallel trailing update needs one barrier per panel
  // rather than one per column (barriers dominate on hybrid/SMT CPUs). Within a panel the columns
  // are factored serially; the trailing update then applies the panel's columns to each entry one
  // subtraction at a time, in k order.
  constexpr std::size_t kPanel = 64;
  for (std::size_t panel = 0; panel < dim; panel += kPanel) {
    const std::size_t panelEnd = std::min(dim, panel + kPanel);
    for (std::size_t k = panel; k < panelEnd; ++k) {
      const FloatT pivot = lower[k * dim + k];
      if (!(pivot > FloatT(0)) || !std::isfinite(pivot)) {
        return false;
      }
      const FloatT diagonal = std::sqrt(pivot);
      lower[k * dim + k] = diagonal;
      for (std::size_t i = k + 1; i < dim; ++i) {
        lower[i * dim + k] /= diagonal;
      }
      // Remaining columns of this panel only; columns beyond it are updated below.
      for (std::size_t i = k + 1; i < dim; ++i) {
        const FloatT lik = lower[i * dim + k];
        FloatT* row = lower.data() + i * dim;
        const std::size_t last = std::min(i + 1, panelEnd);
        for (std::size_t j = k + 1; j < last; ++j) {
          row[j] -= lik * lower[j * dim + k];
        }
      }
    }
    [[maybe_unused]] const std::size_t trailing = dim - panelEnd;
    FEATURE_ELM_OMP(omp parallel for schedule(dynamic, 8)
                        if (trailing * trailing / 2 * (panelEnd - panel) > kParallelWorkThreshold))
    for (std::size_t i = panelEnd; i < dim; ++i) {
      FloatT* row = lower.data() + i * dim;
      for (std::size_t j = panelEnd; j <= i; ++j) {
        const FloatT* rowJ = lower.data() + j * dim;
        FloatT value = row[j];
        for (std::size_t k = panel; k < panelEnd; ++k) {
          value -= row[k] * rowJ[k];
        }
        row[j] = value;
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

    FEATURE_ELM_OMP(omp parallel for schedule(static)
                        if ((augmentedRows - col) * (numFeatures - col) > kParallelWorkThreshold))
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

// NOLINTBEGIN(bugprone-easily-swappable-parameters): same shape as solveRegularizedQr.
// Ridge solve through the dual (samples x samples) Gram matrix. Accumulates in AccT, which is
// FloatT normally and double for the mixed-precision retry. With AccT == FloatT the arithmetic is
// the plain textbook loop, so results are unchanged by the templating.
template <typename AccT, typename FloatT>
[[nodiscard]] bool solveDualCholesky(const std::vector<FloatT>& features,
                                     const std::vector<FloatT>& targets, std::size_t numSamples,
                                     std::size_t numFeatures, std::size_t numOutputs,
                                     AccT ridgeAlpha, std::vector<AccT>* weights) {
  // Symmetric: compute the upper triangle and mirror it. Thread i writes row i (j >= i) and
  // column i (j > i), which no other thread touches.
  std::vector<AccT> gram(numSamples * numSamples, AccT(0));
  FEATURE_ELM_OMP(omp parallel for schedule(dynamic, 8)
                      if (numSamples * numSamples * numFeatures / 2 > kParallelWorkThreshold))
  for (std::size_t sampleI = 0; sampleI < numSamples; ++sampleI) {
    const FloatT* rowI = features.data() + sampleI * numFeatures;
    for (std::size_t sampleJ = sampleI; sampleJ < numSamples; ++sampleJ) {
      const FloatT* rowJ = features.data() + sampleJ * numFeatures;
      AccT sum = AccT(0);
      for (std::size_t feature = 0; feature < numFeatures; ++feature) {
        sum += static_cast<AccT>(rowI[feature]) * static_cast<AccT>(rowJ[feature]);
      }
      gram[sampleI * numSamples + sampleJ] = sum;
      gram[sampleJ * numSamples + sampleI] = sum;
    }
  }
  addScaledIdentity(&gram, numSamples, ridgeAlpha);

  const std::vector<AccT> rhs(targets.begin(), targets.end());
  std::vector<AccT> gamma;
  if (!solveSpdCholesky(gram, rhs, numSamples, numOutputs, &gamma)) {
    return false;
  }

  weights->assign(numFeatures * numOutputs, AccT(0));
  FEATURE_ELM_OMP(omp parallel for schedule(static)
                      if (numFeatures * numOutputs * numSamples > kParallelWorkThreshold))
  for (std::size_t feature = 0; feature < numFeatures; ++feature) {
    for (std::size_t output = 0; output < numOutputs; ++output) {
      AccT sum = AccT(0);
      for (std::size_t sample = 0; sample < numSamples; ++sample) {
        sum += static_cast<AccT>(features[sample * numFeatures + feature]) *
               gamma[sample * numOutputs + output];
      }
      (*weights)[feature * numOutputs + output] = sum;
    }
  }
  return true;
}

// Ridge solve through the primal normal equations (H^T H + alpha I) beta = H^T T, accumulated in
// AccT as above.
template <typename AccT, typename FloatT>
[[nodiscard]] bool solvePrimalCholesky(const std::vector<FloatT>& features,
                                       const std::vector<FloatT>& targets, std::size_t numSamples,
                                       std::size_t numFeatures, std::size_t numOutputs,
                                       AccT ridgeAlpha, std::vector<AccT>* weights) {
  // Upper triangle only, accumulated over blocks of samples that stay in cache. Within a block,
  // threads own whole rows of the normal matrix; each entry still sums over samples in order, so
  // results are bit-identical to the serial textbook loop.
  // Blocks of ~4M elements stay in cache; bigger blocks mean fewer barriers for small problems.
  const std::size_t kSampleBlock = std::clamp<std::size_t>(
      (std::size_t{1} << 22) / std::max<std::size_t>(numFeatures, 1), 256, 4096);
  std::vector<AccT> normal(numFeatures * numFeatures, AccT(0));
  std::vector<AccT> rhs(numFeatures * numOutputs, AccT(0));
  for (std::size_t first = 0; first < numSamples; first += kSampleBlock) {
    const std::size_t last = std::min(numSamples, first + kSampleBlock);
    [[maybe_unused]] const std::size_t blockWork =
        (last - first) * numFeatures * (numFeatures + numOutputs) / 2;
    FEATURE_ELM_OMP(omp parallel for schedule(dynamic, 8) if (blockWork > kParallelWorkThreshold))
    for (std::size_t featureI = 0; featureI < numFeatures; ++featureI) {
      AccT* normalRow = normal.data() + featureI * numFeatures;
      AccT* rhsRow = rhs.data() + featureI * numOutputs;
      for (std::size_t sample = first; sample < last; ++sample) {
        const FloatT* row = features.data() + sample * numFeatures;
        const FloatT* target = targets.data() + sample * numOutputs;
        const AccT value = static_cast<AccT>(row[featureI]);
        for (std::size_t featureJ = featureI; featureJ < numFeatures; ++featureJ) {
          normalRow[featureJ] += value * static_cast<AccT>(row[featureJ]);
        }
        for (std::size_t output = 0; output < numOutputs; ++output) {
          rhsRow[output] += value * static_cast<AccT>(target[output]);
        }
      }
    }
  }
  for (std::size_t featureI = 0; featureI < numFeatures; ++featureI) {
    for (std::size_t featureJ = 0; featureJ < featureI; ++featureJ) {
      normal[featureI * numFeatures + featureJ] = normal[featureJ * numFeatures + featureI];
    }
  }
  addScaledIdentity(&normal, numFeatures, ridgeAlpha);
  return solveSpdCholesky(normal, rhs, numFeatures, numOutputs, weights);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

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
  const auto cholesky = [&](auto ridge, auto* out) {
    return useDual ? solveDualCholesky(features, targets, numSamples, numFeatures, numOutputs,
                                       ridge, out)
                   : solvePrimalCholesky(features, targets, numSamples, numFeatures, numOutputs,
                                         ridge, out);
  };

  if (cholesky(options_.ridgeAlpha, weights)) {
    return true;
  }
  // Forming H^T H squares the condition number, so with float32 and a small ridge relative to the
  // data a pivot can go non-positive (random hidden layers make this a per-draw lottery, and large
  // sample counts make it likely). Retry in double precision, which is still a single cheap pass.
  if constexpr (!std::is_same_v<FloatT, double>) {
    std::vector<double> precise;
    if (cholesky(static_cast<double>(options_.ridgeAlpha), &precise)) {
      weights->assign(precise.begin(), precise.end());
      return true;
    }
  }
  // Last resort: Householder QR on the augmented system, which never forms H^T H.
  return solveRegularizedQr(features, targets, numSamples, numFeatures, numOutputs,
                            options_.ridgeAlpha, weights);
}

template class BatchRidgeSolver<float>;
template class BatchRidgeSolver<double>;

}  // namespace feature_elm
