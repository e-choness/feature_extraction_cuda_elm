#ifndef FEATURE_ELM_IO_DRIFT_STREAM_HPP_
#define FEATURE_ELM_IO_DRIFT_STREAM_HPP_

#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace feature_elm {

struct DriftStreamSample {
  std::vector<double> input;
  int label;
};

class DriftStream {
 public:
  struct Config {
    std::size_t inputDim = 2;
    std::size_t numClasses = 2;
    std::size_t streamLength = 1000;
    std::size_t driftPoint = 500;
    unsigned int seed = 42;
  };

  explicit DriftStream(Config config) : config_(std::move(config)), currentSample_(0) {
    if (config_.inputDim < 2u) {
      config_.inputDim = 2u;
    }
  }

  std::optional<DriftStreamSample> next();

  void reset() {
    currentSample_ = 0;
  }

  [[nodiscard]] std::size_t position() const noexcept {
    return currentSample_;
  }
  [[nodiscard]] bool hasDriftOccurred() const noexcept {
    return currentSample_ >= config_.driftPoint;
  }

 private:
  Config config_;
  std::size_t currentSample_;

  double angleA_ = 0.0;
  double angleB_ = M_PI / 2.0;
};

}  // namespace feature_elm

#endif  // FEATURE_ELM_IO_DRIFT_STREAM_HPP_