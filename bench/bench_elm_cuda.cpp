#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "cuda/elm_gpu.hpp"

namespace {

using namespace feature_elm;
using namespace feature_elm::cuda_backend;

void setGpuCounters(benchmark::State& state, std::size_t numSamples) {
  state.counters["device"] = benchmark::Counter(1.0);
  state.counters["dataset_size"] = benchmark::Counter(
      static_cast<double>(numSamples), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1000);
  state.SetLabel("device=GPU");
}

static void BenchmarkGpuHiddenOutput(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  const std::size_t numInputs = 64;
  const std::size_t numHiddenNodes = 128;
  const std::size_t numOutputs = 10;

  std::vector<float> trainData(numSamples * numInputs);
  std::vector<float> targets(numSamples * numOutputs);
  std::vector<float> hiddenWeights(numInputs * numHiddenNodes);
  std::vector<float> hiddenBiases(numHiddenNodes);

  std::mt19937 rng(125);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : trainData) {
    value = dist(rng);
  }
  for (auto& value : targets) {
    value = dist(rng);
  }
  for (auto& value : hiddenWeights) {
    value = dist(rng);
  }
  for (auto& value : hiddenBiases) {
    value = dist(rng);
  }

  setGpuCounters(state, numSamples);

  if (!isGpuAvailable()) {
    state.SkipWithError("No GPU available");
    return;
  }
  state.SetLabel("device=GPU");

  for (auto _ : state) {
    std::vector<float> hiddenOutput;
    benchmark::DoNotOptimize(
        computeHiddenOutputDevice(trainData, numSamples, numInputs, numHiddenNodes,
                                   hiddenWeights, hiddenBiases, ActivationFunction::kSigmoid,
                                   &hiddenOutput));
  }

  state.SetItemsProcessed(state.iterations() * numSamples);
}

static void BenchmarkGpuTrain(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  const std::size_t numInputs = 64;
  const std::size_t numHiddenNodes = 128;
  const std::size_t numOutputs = 10;

  std::vector<float> trainData(numSamples * numInputs);
  std::vector<float> targets(numSamples * numOutputs);
  std::vector<float> hiddenWeights(numInputs * numHiddenNodes);
  std::vector<float> hiddenBiases(numHiddenNodes);

  std::mt19937 rng(126);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : trainData) {
    value = dist(rng);
  }
  for (auto& value : targets) {
    value = dist(rng);
  }
  for (auto& value : hiddenWeights) {
    value = dist(rng);
  }
  for (auto& value : hiddenBiases) {
    value = dist(rng);
  }

  setGpuCounters(state, numSamples);

  if (!isGpuAvailable()) {
    state.SkipWithError("No GPU available");
    return;
  }
  state.SetLabel("device=GPU");

  for (auto _ : state) {
    std::vector<float> outputWeights;
    benchmark::DoNotOptimize(
        trainBatchElmGpu(trainData, targets, numSamples, numInputs, numHiddenNodes,
                         numOutputs, hiddenWeights, hiddenBiases,
                         ActivationFunction::kSigmoid, &outputWeights));
  }

  state.SetItemsProcessed(state.iterations() * numSamples);
}

BENCHMARK(BenchmarkGpuHiddenOutput)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BenchmarkGpuTrain)->Arg(256)->Arg(512);

}  // namespace

BENCHMARK_MAIN();
