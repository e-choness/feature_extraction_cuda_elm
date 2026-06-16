#include "core/rbf_features.hpp"

#include <cmath>
#include <numeric>
#include <random>

namespace feature_elm {

template <typename FloatT>
[[nodiscard]] bool computeRbfFeatures(const std::vector<FloatT>& input, std::size_t numSamples,
                                      const RbfParameters<FloatT>& params,
                                      std::vector<FloatT>* output) {
  if (input.empty() || params.centers.empty() || output == nullptr) {
    return false;
  }
  if (input.size() != numSamples * params.inputDim) {
    return false;
  }
  if (params.centers.size() != params.numCenters * params.inputDim) {
    return false;
  }

  output->assign(numSamples * params.numCenters, FloatT(0));
  FloatT widthSq = params.width * params.width;
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t center = 0; center < params.numCenters; ++center) {
      FloatT distSq = FloatT(0);
      for (std::size_t dim = 0; dim < params.inputDim; ++dim) {
        FloatT diff =
            input[sample * params.inputDim + dim] - params.centers[center * params.inputDim + dim];
        distSq += diff * diff;
      }
      (*output)[sample * params.numCenters + center] =
          std::exp(-distSq / (static_cast<FloatT>(2) * widthSq));
    }
  }
  return true;
}

template <typename FloatT>
[[nodiscard]] bool initializeRbfCentersRandom(std::size_t numCenters, std::size_t inputDim,
                                              RbfParameters<FloatT>* params, unsigned int seed) {
  if (params == nullptr) {
    return false;
  }
  params->numCenters = numCenters;
  params->inputDim = inputDim;
  params->centers.assign(numCenters * inputDim, FloatT(0));
  params->width = static_cast<FloatT>(1.0);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<FloatT> dis(static_cast<FloatT>(-1), static_cast<FloatT>(1));
  for (auto& center : params->centers) {
    center = dis(gen);
  }
  return true;
}

template <typename FloatT>
[[nodiscard]] bool initializeRbfCentersKMeans(std::size_t numCenters, std::size_t inputDim,
                                              const std::vector<FloatT>& data,
                                              std::size_t numSamples, RbfParameters<FloatT>* params,
                                              unsigned int seed) {
  if (params == nullptr || data.empty() || numSamples == 0 || numCenters == 0) {
    return false;
  }

  params->numCenters = numCenters;
  params->inputDim = inputDim;
  params->centers.resize(numCenters * inputDim);
  params->width = static_cast<FloatT>(1.0);

  std::mt19937 gen(seed);

  std::uniform_int_distribution<int> firstDist(0, static_cast<int>(numSamples - 1));
  int firstIdx = firstDist(gen);
  for (std::size_t dim = 0; dim < inputDim; ++dim) {
    params->centers[dim] = data[firstIdx * inputDim + dim];
  }

  for (std::size_t centerIdx = 1; centerIdx < numCenters; ++centerIdx) {
    std::vector<FloatT> dists(numSamples, FloatT(0));
    for (std::size_t i = 0; i < numSamples; ++i) {
      FloatT minDist = std::numeric_limits<FloatT>::max();
      for (std::size_t c = 0; c < centerIdx; ++c) {
        FloatT distSq = FloatT(0);
        for (std::size_t dim = 0; dim < inputDim; ++dim) {
          FloatT diff = data[i * inputDim + dim] - params->centers[c * inputDim + dim];
          distSq += diff * diff;
        }
        if (distSq < minDist) {
          minDist = distSq;
        }
      }
      dists[i] = minDist;
    }

    FloatT sum = FloatT(0);
    for (FloatT d : dists) {
      sum += d;
    }
    if (sum <= FloatT(0)) {
      sum = static_cast<FloatT>(1);
    }

    std::uniform_real_distribution<FloatT> probDist(0, sum);
    FloatT threshold = probDist(gen);
    FloatT running = FloatT(0);
    int selected = 0;
    for (std::size_t i = 0; i < numSamples; ++i) {
      running += dists[i];
      if (running >= threshold) {
        selected = static_cast<int>(i);
        break;
      }
    }

    for (std::size_t dim = 0; dim < inputDim; ++dim) {
      params->centers[centerIdx * inputDim + dim] = data[selected * inputDim + dim];
    }
  }

  return true;
}

template class RbfParameters<float>;
template class RbfParameters<double>;
template bool computeRbfFeatures<float>(const std::vector<float>&, std::size_t,
                                        const RbfParameters<float>&, std::vector<float>*);
template bool computeRbfFeatures<double>(const std::vector<double>&, std::size_t,
                                         const RbfParameters<double>&, std::vector<double>*);
template bool initializeRbfCentersRandom<float>(std::size_t, std::size_t, RbfParameters<float>*,
                                                unsigned int);
template bool initializeRbfCentersRandom<double>(std::size_t, std::size_t, RbfParameters<double>*,
                                                 unsigned int);
template bool initializeRbfCentersKMeans<float>(std::size_t, std::size_t, const std::vector<float>&,
                                                std::size_t, RbfParameters<float>*, unsigned int);
template bool initializeRbfCentersKMeans<double>(std::size_t, std::size_t,
                                                 const std::vector<double>&, std::size_t,
                                                 RbfParameters<double>*, unsigned int);

}  // namespace feature_elm
