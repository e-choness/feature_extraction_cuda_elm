#include <benchmark/benchmark.h>

#include <algorithm>
#include <vector>

#include "core/ml_elm.hpp"

namespace {

using feature_elm::ActivationFunction;
using feature_elm::Backend;
using feature_elm::MlElm;

std::vector<double> makeXorData(std::size_t numSamples, std::size_t* numOutputs) {
  const std::size_t repeats = std::max<std::size_t>(1, numSamples / 4);
  const std::size_t actualSamples = repeats * 4;
  *numOutputs = 2;
  std::vector<double> data(actualSamples * 2);
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::size_t i = 0; i < 4; ++i) {
      const std::size_t sample = repeat * 4 + i;
      const double x0 = (i & 1u) != 0 ? 1.0 : 0.0;
      const double x1 = (i & 2u) != 0 ? 1.0 : 0.0;
      data[sample * 2 + 0] = x0;
      data[sample * 2 + 1] = x1;
    }
  }
  return data;
}

std::vector<double> makeOneHotLabels(std::size_t numSamples, std::size_t numOutputs) {
  std::vector<double> labels(numSamples * numOutputs);
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    const std::size_t label = sample % numOutputs;
    labels[sample * numOutputs + label] = 1.0;
  }
  return labels;
}

void setMlElmCounters(benchmark::State& state, std::size_t numSamples, std::size_t numOutputs) {
  state.counters["device"] = benchmark::Counter(0.0);
  state.counters["dataset_size"] = benchmark::Counter(
      static_cast<double>(numSamples), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1000);
  state.counters["num_outputs"] = benchmark::Counter(static_cast<double>(numOutputs));
  state.counters["feature_elements"] = benchmark::Counter(
      static_cast<double>(numSamples * 8), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1000);
}

void BenchmarkMlElmFit(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  std::size_t numOutputs = 0;
  const std::vector<double> data = makeXorData(numSamples, &numOutputs);
  const std::vector<double> labels = makeOneHotLabels(data.size() / 2, numOutputs);

  setMlElmCounters(state, data.size() / 2, numOutputs);
  state.SetLabel("device=CPU;operation=fit");

  for (auto _ : state) {
    MlElm<double> model(2, {8, 8}, ActivationFunction::kSigmoid, Backend::kCpu, 1e-6, 17u);
    if (!model.train(data, labels, data.size() / 2, numOutputs)) {
      state.SkipWithError("ML-ELM fit failed");
      return;
    }
    benchmark::DoNotOptimize(model.outputWeights().data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(data.size() / 2));
}

void BenchmarkMlElmForward(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  std::size_t numOutputs = 0;
  const std::vector<double> data = makeXorData(numSamples, &numOutputs);
  const std::vector<double> labels = makeOneHotLabels(data.size() / 2, numOutputs);
  MlElm<double> model(2, {8, 8}, ActivationFunction::kSigmoid, Backend::kCpu, 1e-6, 17u);
  if (!model.train(data, labels, data.size() / 2, numOutputs)) {
    state.SkipWithError("ML-ELM pretrain failed");
    return;
  }

  setMlElmCounters(state, data.size() / 2, numOutputs);
  state.SetLabel("device=CPU;operation=forward");

  for (auto _ : state) {
    const auto predictions = model.predictBatch(data, data.size() / 2);
    if (!predictions.has_value()) {
      state.SkipWithError("ML-ELM forward failed");
      return;
    }
    benchmark::DoNotOptimize(predictions->data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(data.size() / 2));
}

}  // namespace

BENCHMARK(BenchmarkMlElmFit)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BenchmarkMlElmForward)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK_MAIN();
