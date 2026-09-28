#include "core/elm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>

#include "core/dense.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

namespace feature_elm {

template <typename FloatT>
BatchElm<FloatT>::BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                           ActivationFunction activation, Backend backend, FloatT ridgeAlpha)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      isTrained_(false),
      backend_(backend),
      ridgeAlpha_(ridgeAlpha),
      featureMap_(numInputs, numHiddenNodes, activationKind(activation), std::nullopt, backend),
      solver_({ridgeAlpha}) {}

template <typename FloatT>
BatchElm<FloatT>::BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
                           ActivationFunction activation, Backend backend,
                           const std::vector<FloatT>& hiddenWeights,
                           const std::vector<FloatT>& hiddenBiases, FloatT ridgeAlpha)
    : numInputs_(numInputs),
      numHiddenNodes_(numHiddenNodes),
      numOutputs_(0),
      activation_(activation),
      isTrained_(false),
      backend_(backend),
      ridgeAlpha_(ridgeAlpha),
      featureMap_(numInputs, numHiddenNodes, activationKind(activation), std::nullopt, backend,
                  hiddenWeights, hiddenBiases),
      solver_({ridgeAlpha}) {}

template <typename FloatT>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool BatchElm<FloatT>::train(const std::vector<FloatT>& trainData,
                             const std::vector<FloatT>& trainTargets, std::size_t numSamples,
                             std::size_t numOutputs) {
  if (trainData.size() != numSamples * numInputs_) {
    return false;
  }
  if (trainTargets.size() != numSamples * numOutputs) {
    return false;
  }

  numOutputs_ = numOutputs;

  std::vector<FloatT> hiddenOutput;
  if (!featureMap_.transform(trainData, numSamples, &hiddenOutput)) {
    isTrained_ = false;
    return false;
  }

  outputWeights_.clear();
  if (backend_ == Backend::kGpu) {
    if (!cuda_backend::solveRidgeGpu<FloatT>(hiddenOutput, trainTargets, numSamples, numOutputs,
                                             {ridgeAlpha_}, &outputWeights_) ||
        outputWeights_.empty()) {
      isTrained_ = false;
      return false;
    }
  } else {
    if (!solver_.solve(hiddenOutput, numSamples, trainTargets, numOutputs, &outputWeights_) ||
        outputWeights_.empty()) {
      isTrained_ = false;
      return false;
    }
  }

  isTrained_ = true;
  return true;
}

template <typename FloatT>
std::optional<std::vector<FloatT>> BatchElm<FloatT>::predict(
    const std::vector<FloatT>& input) const {
  if (!isTrained_) {
    return std::nullopt;
  }
  if (input.size() != numInputs_) {
    return std::nullopt;
  }

  std::vector<FloatT> hiddenOutput;
  if (!featureMap_.transform(input, 1, &hiddenOutput)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(numOutputs_);
  for (std::size_t i = 0; i < numOutputs_; ++i) {
    output[i] = FloatT(0);
    for (std::size_t j = 0; j < numHiddenNodes_; ++j) {
      output[i] += hiddenOutput[j] * outputWeights_[j * numOutputs_ + i];
    }
  }

  return output;
}

template <typename FloatT>
std::optional<std::vector<FloatT>> BatchElm<FloatT>::predictBatch(
    const std::vector<FloatT>& testData, std::size_t numSamples) const {
  if (!isTrained_) {
    return std::nullopt;
  }
  if (testData.size() != numSamples * numInputs_) {
    return std::nullopt;
  }

  std::vector<FloatT> hiddenOutput;
  if (!featureMap_.transform(testData, numSamples, &hiddenOutput)) {
    return std::nullopt;
  }

  std::vector<FloatT> output(numSamples * numOutputs_);
  denseForward(hiddenOutput.data(), numSamples, numHiddenNodes_, outputWeights_.data(),
               static_cast<const FloatT*>(nullptr), numOutputs_, output.data());

  return output;
}

namespace {

// Model file layout (little-endian), version 1:
//   char[4]  magic "FELM"
//   uint32   version
//   uint32   scalar width in bytes (4 = float, 8 = double)
//   uint32   activation (ActivationFunction as integer)
//   uint64   inputs, hidden, outputs
//   float64  ridge alpha
//   scalar[inputs * hidden]   hidden weights (row-major, inputs x hidden)
//   scalar[hidden]            hidden biases
//   scalar[hidden * outputs]  output weights (row-major, hidden x outputs)
constexpr std::array<char, 4> kModelMagic = {'F', 'E', 'L', 'M'};
constexpr std::uint32_t kModelVersion = 1;
constexpr std::uint64_t kMaxModelDim = std::uint64_t{1}
                                       << 24;  // sanity bound against corrupt headers

template <typename T>
void writePod(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool readPod(std::ifstream& in, T* value) {
  in.read(reinterpret_cast<char*>(value), sizeof(T));
  return static_cast<bool>(in);
}

template <typename FloatT>
void writeArray(std::ofstream& out, const std::vector<FloatT>& values) {
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(FloatT)));
}

template <typename FloatT>
bool readArray(std::ifstream& in, std::size_t count, std::vector<FloatT>* values) {
  values->resize(count);
  in.read(reinterpret_cast<char*>(values->data()),
          static_cast<std::streamsize>(count * sizeof(FloatT)));
  return static_cast<bool>(in) &&
         std::all_of(values->begin(), values->end(), [](FloatT v) { return std::isfinite(v); });
}

}  // namespace

template <typename FloatT>
bool BatchElm<FloatT>::save(const std::filesystem::path& path) const {
  if (!isTrained_) {
    return false;
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out.write(kModelMagic.data(), kModelMagic.size());
  writePod(out, kModelVersion);
  writePod(out, static_cast<std::uint32_t>(sizeof(FloatT)));
  writePod(out, static_cast<std::uint32_t>(activation_));
  writePod(out, static_cast<std::uint64_t>(numInputs_));
  writePod(out, static_cast<std::uint64_t>(numHiddenNodes_));
  writePod(out, static_cast<std::uint64_t>(numOutputs_));
  writePod(out, static_cast<double>(ridgeAlpha_));
  writeArray(out, featureMap_.inputWeights());
  writeArray(out, featureMap_.biases());
  writeArray(out, outputWeights_);
  return static_cast<bool>(out);
}

template <typename FloatT>
std::optional<BatchElm<FloatT>> BatchElm<FloatT>::load(const std::filesystem::path& path,
                                                       Backend backend) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::nullopt;
  }
  std::array<char, 4> magic{};
  std::uint32_t version = 0;
  std::uint32_t scalarBytes = 0;
  std::uint32_t activation = 0;
  std::uint64_t inputs = 0;
  std::uint64_t hidden = 0;
  std::uint64_t outputs = 0;
  double ridge = 0.0;
  in.read(magic.data(), magic.size());
  if (!in || magic != kModelMagic || !readPod(in, &version) || version != kModelVersion ||
      !readPod(in, &scalarBytes) || scalarBytes != sizeof(FloatT) || !readPod(in, &activation) ||
      activation > static_cast<std::uint32_t>(ActivationFunction::kRelu) || !readPod(in, &inputs) ||
      !readPod(in, &hidden) || !readPod(in, &outputs) || !readPod(in, &ridge)) {
    return std::nullopt;
  }
  if (inputs == 0 || hidden == 0 || outputs == 0 || inputs > kMaxModelDim ||
      hidden > kMaxModelDim || outputs > kMaxModelDim || !std::isfinite(ridge)) {
    return std::nullopt;
  }
  std::vector<FloatT> weights;
  std::vector<FloatT> biases;
  std::vector<FloatT> outputWeights;
  if (!readArray(in, inputs * hidden, &weights) || !readArray(in, hidden, &biases) ||
      !readArray(in, hidden * outputs, &outputWeights)) {
    return std::nullopt;
  }
  BatchElm model(inputs, hidden, static_cast<ActivationFunction>(activation), backend, weights,
                 biases, static_cast<FloatT>(ridge));
  model.outputWeights_ = std::move(outputWeights);
  model.numOutputs_ = outputs;
  model.isTrained_ = true;
  return model;
}

template <typename FloatT>
void BatchElm<FloatT>::reset() noexcept {
  outputWeights_.clear();
  numOutputs_ = 0;
  isTrained_ = false;
}

template class BatchElm<float>;
template class BatchElm<double>;

}  // namespace feature_elm