#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/elm_ae.hpp"
#include "core/random_additive_map.hpp"
#include "core/rbf_map.hpp"

namespace {

using feature_elm::ActivationKind;
using feature_elm::ElmAutoEncoderLayer;
using feature_elm::RandomAdditiveMap;
using feature_elm::RbfCenterInit;
using feature_elm::RbfMap;

std::vector<float> makeDataset(std::size_t numSamples, std::size_t numInputs, unsigned int seed) {
  std::vector<float> data(numSamples * numInputs);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t input = 0; input < numInputs; ++input) {
      const float phase = static_cast<float>(sample + 1) * static_cast<float>(input + 1);
      data[sample * numInputs + input] =
          0.25f * distribution(generator) + 0.75f * std::sin(phase * 0.01f);
    }
  }
  return data;
}

void setCommonCounters(benchmark::State& state, int deviceCode, std::size_t numSamples,
                       std::size_t outputDim) {
  state.counters["device"] = benchmark::Counter(static_cast<double>(deviceCode));
  state.counters["dataset_size"] = benchmark::Counter(
      static_cast<double>(numSamples), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1000);
  state.counters["feature_elements"] = benchmark::Counter(
      static_cast<double>(numSamples * outputDim), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1000);
}

void BenchmarkAdditiveMapTransform(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t kInputDim = 64;
  constexpr std::size_t kHiddenDim = 128;
  const std::vector<float> input = makeDataset(numSamples, kInputDim, 11u);
  RandomAdditiveMap<float> map(kInputDim, kHiddenDim, ActivationKind::kRelu, 17u);

  std::vector<float> output(numSamples * kHiddenDim);
  setCommonCounters(state, 0, numSamples, kHiddenDim);
  state.SetLabel("device=CPU");

  for (auto _ : state) {
    if (!map.transform(input, numSamples, &output)) {
      state.SkipWithError("additive map transform failed");
      return;
    }
    benchmark::DoNotOptimize(output.data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(numSamples * kHiddenDim));
}

void BenchmarkRbfMapTransform(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t kInputDim = 16;
  constexpr std::size_t kCenters = 64;
  const std::vector<float> input = makeDataset(numSamples, kInputDim, 23u);
  RbfMap<float> map(kInputDim, kCenters, 1.5f, RbfCenterInit::kRandom, 29u);
  if (!map.fit(input, numSamples)) {
    state.SkipWithError("RBF fit failed");
    return;
  }

  std::vector<float> output(numSamples * kCenters);
  setCommonCounters(state, 0, numSamples, kCenters);
  state.SetLabel("device=CPU");

  for (auto _ : state) {
    if (!map.transform(input, numSamples, &output)) {
      state.SkipWithError("RBF transform failed");
      return;
    }
    benchmark::DoNotOptimize(output.data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(numSamples * kCenters));
}

void BenchmarkElmAutoEncoderTransform(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t kInputDim = 32;
  constexpr std::size_t kHiddenDim = 64;
  const std::vector<float> input = makeDataset(numSamples, kInputDim, 31u);
  const std::vector<float> fitInput = makeDataset(numSamples, kInputDim, 31u);
  ElmAutoEncoderLayer<float> layer(kInputDim, kHiddenDim, ActivationKind::kRelu, 37u, 1e-6f);
  if (!layer.fit(fitInput, numSamples)) {
    state.SkipWithError("ELM-AE fit failed");
    return;
  }

  std::vector<float> output(numSamples * kHiddenDim);
  setCommonCounters(state, 0, numSamples, kHiddenDim);
  state.SetLabel("device=CPU");

  for (auto _ : state) {
    if (!layer.transform(input, numSamples, &output)) {
      state.SkipWithError("ELM-AE transform failed");
      return;
    }
    benchmark::DoNotOptimize(output.data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(numSamples * kHiddenDim));
}

}  // namespace

BENCHMARK(BenchmarkAdditiveMapTransform)->Arg(256)->Arg(1024)->Arg(4096);
BENCHMARK(BenchmarkRbfMapTransform)->Arg(128)->Arg(512)->Arg(2048);
BENCHMARK(BenchmarkElmAutoEncoderTransform)->Arg(128)->Arg(512)->Arg(2048);

BENCHMARK_MAIN();
