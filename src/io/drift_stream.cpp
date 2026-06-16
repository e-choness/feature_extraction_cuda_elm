#include "io/drift_stream.hpp"

#include <random>

namespace feature_elm {

std::optional<DriftStreamSample> DriftStream::next() {
  if (currentSample_ >= config_.streamLength) {
    return std::nullopt;
  }

  std::mt19937 rng(static_cast<unsigned int>(config_.seed + currentSample_));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  DriftStreamSample sample;
  sample.input.resize(config_.inputDim);
  for (std::size_t d = 0; d < config_.inputDim; ++d) {
    sample.input[d] = dist(rng);
  }

  const double angle = hasDriftOccurred() ? angleB_ : angleA_;
  const double projection = sample.input[0] * std::cos(angle) + sample.input[1] * std::sin(angle);
  sample.label = projection >= 0.0 ? 1 : 0;

  ++currentSample_;
  return sample;
}

}  // namespace feature_elm
