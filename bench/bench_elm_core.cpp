#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/elm.hpp"

namespace {

using namespace feature_elm;

static void BenchmarkComputeHiddenOutput(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  const std::size_t numInputs = 64;
  const std::size_t numHiddenNodes = 128;
  const std::size_t numOutputs = 10;

  BatchElm<float> elm(numInputs, numHiddenNodes, ActivationFunction::kSigmoid, Backend::kCpu);

  std::vector<float> trainData(numSamples * numInputs);
  std::vector<float> targets(numSamples * numOutputs);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : trainData) {
    value = dist(rng);
  }
  for (auto& value : targets) {
    value = dist(rng);
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(elm.train(trainData, targets, numSamples, numOutputs));
  }

  state.SetItemsProcessed(state.iterations() * numSamples);
}

static void BenchmarkSolveLeastSquares(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  const std::size_t numHiddenNodes = 128;
  const std::size_t numOutputs = 10;

  std::vector<float> hiddenOutput(numSamples * numHiddenNodes);
  std::vector<float> targets(numSamples * numOutputs);
  std::mt19937 rng(124);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : hiddenOutput) {
    value = dist(rng);
  }
  for (auto& value : targets) {
    value = dist(rng);
  }

  for (auto _ : state) {
    BatchElm<float> elm(0, 0);
    benchmark::DoNotOptimize(elm);
    benchmark::DoNotOptimize(hiddenOutput);
    benchmark::DoNotOptimize(targets);
  }

  state.SetItemsProcessed(state.iterations() * numSamples);
}

BENCHMARK(BenchmarkComputeHiddenOutput)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BenchmarkSolveLeastSquares)->Arg(256)->Arg(512)->Arg(1024);

}  // namespace

BENCHMARK_MAIN();
