// Full-size dataset benchmarks: train on all 60,000 MNIST / Fashion-MNIST training images (784
// inputs, 10 classes) on the CPU and the GPU, and report test accuracy on the 10,000 test images as
// a counter next to the time. Each configuration runs once (CPU runs take seconds).
//
// Data: run `python3 scripts/fetch_datasets.py` first; the directory can be overridden with
// FEATURE_ELM_DATA_DIR (default .cache/datasets, relative to the working directory). Benchmarks for
// missing datasets are reported as skipped.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/elm.hpp"
#include "core/os_elm.hpp"
#include "cuda/gpu_ops.hpp"
#include "io/idx_dataset.hpp"

namespace {

using namespace feature_elm;

constexpr std::size_t kClasses = 10;

struct Split {
  IdxDataset train;
  IdxDataset test;
  std::vector<float> trainTargets;  // one-hot
};

const Split* dataset(const std::string& name) {
  static std::map<std::string, std::unique_ptr<Split>> cache;
  if (auto it = cache.find(name); it != cache.end()) {
    return it->second.get();
  }
  const char* env = std::getenv("FEATURE_ELM_DATA_DIR");
  const std::filesystem::path dir =
      std::filesystem::path(env != nullptr ? env : ".cache/datasets") / name;
  auto train = loadIdx(dir / "train-images-idx3-ubyte", dir / "train-labels-idx1-ubyte");
  auto test = loadIdx(dir / "t10k-images-idx3-ubyte", dir / "t10k-labels-idx1-ubyte");
  std::unique_ptr<Split> split;
  if (train.dataset && test.dataset) {
    split = std::make_unique<Split>();
    split->train = std::move(*train.dataset);
    split->test = std::move(*test.dataset);
    split->trainTargets.assign(split->train.numSamples * kClasses, 0.0f);
    for (std::size_t i = 0; i < split->train.numSamples; ++i) {
      split->trainTargets[i * kClasses + static_cast<std::size_t>(split->train.labels[i])] = 1.0f;
    }
  }
  return (cache[name] = std::move(split)).get();
}

// Seeded ReLU hidden layer scaled for 784 inputs in 0..1, so CPU and GPU runs see identical
// weights.
std::pair<std::vector<float>, std::vector<float>> hiddenLayer(std::size_t inputs,
                                                              std::size_t hidden) {
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  // U(-a, a) with a = 5 / sqrt(inputs): pre-activations of roughly unit spread for ~150 inked
  // pixels.
  const float scale = 5.0f / std::sqrt(static_cast<float>(inputs));
  std::vector<float> weights(inputs * hidden);
  std::vector<float> biases(hidden);
  for (auto& w : weights) {
    w = dist(gen) * scale;
  }
  for (auto& b : biases) {
    b = dist(gen);
  }
  return {std::move(weights), std::move(biases)};
}

template <typename Model>
double testAccuracy(const Model& model, const IdxDataset& test) {
  const auto scores = model.predictBatch(test.data, test.numSamples);
  if (!scores) {
    return 0.0;
  }
  std::size_t correct = 0;
  for (std::size_t i = 0; i < test.numSamples; ++i) {
    const auto row = scores->begin() + static_cast<std::ptrdiff_t>(i * kClasses);
    const auto predicted = std::max_element(row, row + kClasses) - row;
    correct += predicted == test.labels[i] ? 1u : 0u;
  }
  return static_cast<double>(correct) / static_cast<double>(test.numSamples);
}

bool prepare(benchmark::State& state, const Split* split, Backend backend) {
  if (split == nullptr) {
    state.SkipWithError("dataset not found; run scripts/fetch_datasets.py");
    return false;
  }
  if (backend == Backend::kGpu && !cuda_backend::isGpuAvailable()) {
    state.SkipWithError("No GPU available");
    return false;
  }
  return true;
}

void BatchElmTrain(benchmark::State& state, const char* name, Backend backend) {
  const Split* split = dataset(name);
  if (!prepare(state, split, backend)) {
    return;
  }
  const auto hidden = static_cast<std::size_t>(state.range(0));
  const auto [weights, biases] = hiddenLayer(split->train.inputDim, hidden);
  BatchElm<float> model(split->train.inputDim, hidden, ActivationFunction::kRelu, backend, weights,
                        biases, 1.0f);
  for (auto _ : state) {
    if (!model.train(split->train.data, split->trainTargets, split->train.numSamples, kClasses)) {
      state.SkipWithError("training failed");
      return;
    }
  }
  state.counters["accuracy"] = testAccuracy(model, split->test);
  state.counters["samples"] = static_cast<double>(split->train.numSamples);
  state.SetLabel(backend == Backend::kGpu ? "GPU" : "CPU");
}

// OS-ELM streaming the training set: one initial block, then fixed-size chunks.
void OsElmStream(benchmark::State& state, const char* name, Backend backend) {
  const Split* split = dataset(name);
  if (!prepare(state, split, backend)) {
    return;
  }
  const auto hidden = static_cast<std::size_t>(state.range(0));
  const std::size_t dim = split->train.inputDim;
  const std::size_t n = split->train.numSamples;
  constexpr std::size_t kInitial = 4096;
  constexpr std::size_t kChunk = 1000;
  const auto [weights, biases] = hiddenLayer(dim, hidden);
  RlsOptions<float> rls;
  rls.regularization = 1.0f;
  std::optional<OsElm<float>> model;
  for (auto _ : state) {
    model.emplace(dim, hidden, ActivationFunction::kRelu, backend, weights, biases, rls);
    std::vector<float> x(split->train.data.begin(),
                         split->train.data.begin() + static_cast<std::ptrdiff_t>(kInitial * dim));
    std::vector<float> t(
        split->trainTargets.begin(),
        split->trainTargets.begin() + static_cast<std::ptrdiff_t>(kInitial * kClasses));
    if (!model->initialize(x, t, kInitial, kClasses)) {
      state.SkipWithError("initialize failed");
      return;
    }
    for (std::size_t first = kInitial; first < n; first += kChunk) {
      const std::size_t count = std::min(kChunk, n - first);
      x.assign(split->train.data.begin() + static_cast<std::ptrdiff_t>(first * dim),
               split->train.data.begin() + static_cast<std::ptrdiff_t>((first + count) * dim));
      t.assign(
          split->trainTargets.begin() + static_cast<std::ptrdiff_t>(first * kClasses),
          split->trainTargets.begin() + static_cast<std::ptrdiff_t>((first + count) * kClasses));
      if (!model->update(x, t, count)) {
        state.SkipWithError("update failed");
        return;
      }
    }
  }
  state.counters["accuracy"] = testAccuracy(*model, split->test);
  state.counters["samples"] = static_cast<double>(n);
  state.SetLabel(backend == Backend::kGpu ? "GPU" : "CPU");
}

}  // namespace

#define FEATURE_ELM_DATASET_BENCH(fn, tag, directory, backend, ...) \
  BENCHMARK_CAPTURE(fn, tag, directory, backend)                    \
      ->ArgsProduct({{__VA_ARGS__}})                                \
      ->Iterations(1)                                               \
      ->Unit(benchmark::kMillisecond)                               \
      ->UseRealTime()

FEATURE_ELM_DATASET_BENCH(BatchElmTrain, mnist_cpu, "mnist", Backend::kCpu, 1024, 2048, 4096);
FEATURE_ELM_DATASET_BENCH(BatchElmTrain, mnist_gpu, "mnist", Backend::kGpu, 1024, 2048, 4096);
FEATURE_ELM_DATASET_BENCH(BatchElmTrain, fashion_cpu, "fashion-mnist", Backend::kCpu, 1024, 2048,
                          4096);
FEATURE_ELM_DATASET_BENCH(BatchElmTrain, fashion_gpu, "fashion-mnist", Backend::kGpu, 1024, 2048,
                          4096);
FEATURE_ELM_DATASET_BENCH(OsElmStream, mnist_cpu, "mnist", Backend::kCpu, 1024, 2048);
FEATURE_ELM_DATASET_BENCH(OsElmStream, mnist_gpu, "mnist", Backend::kGpu, 1024, 2048);

BENCHMARK_MAIN();
