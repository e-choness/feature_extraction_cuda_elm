// End-to-end Batch ELM benchmarks with identical workloads on both backends, so CPU and GPU rows
// can be compared directly: 2048 samples x 64 inputs, 10 outputs, float32, seeded hidden layer.

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "core/elm.hpp"
#include "core/random_additive_map.hpp"
#include "cuda/gpu_ops.hpp"

namespace {

using namespace feature_elm;

constexpr std::size_t kSamples = 2048;
constexpr std::size_t kInputs = 64;
constexpr std::size_t kOutputs = 10;

std::vector<float> randomValues(std::size_t count, unsigned int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(count);
  for (auto& v : values) {
    v = dist(gen);
  }
  return values;
}

void runTrain(benchmark::State& state, Backend backend) {
  if (backend == Backend::kGpu && !cuda_backend::isGpuAvailable()) {
    state.SkipWithError("No GPU available");
    return;
  }
  const auto hidden = static_cast<std::size_t>(state.range(0));
  const auto data = randomValues(kSamples * kInputs, 1u);
  const auto targets = randomValues(kSamples * kOutputs, 2u);
  const auto weights = randomValues(kInputs * hidden, 3u);
  const auto biases = randomValues(hidden, 4u);
  BatchElm<float> model(kInputs, hidden, ActivationFunction::kSigmoid, backend, weights, biases,
                        1e-2f);
  for (auto _ : state) {
    if (!model.train(data, targets, kSamples, kOutputs)) {
      state.SkipWithError("training failed");
      return;
    }
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * kSamples));
  state.SetLabel(backend == Backend::kGpu ? "GPU" : "CPU");
}

void runTransform(benchmark::State& state, Backend backend) {
  if (backend == Backend::kGpu && !cuda_backend::isGpuAvailable()) {
    state.SkipWithError("No GPU available");
    return;
  }
  const auto hidden = static_cast<std::size_t>(state.range(0));
  const auto data = randomValues(kSamples * kInputs, 1u);
  RandomAdditiveMap<float> map(kInputs, hidden, ActivationKind::kSigmoid, 5u, backend);
  std::vector<float> out;
  for (auto _ : state) {
    if (!map.transform(data, kSamples, &out)) {
      state.SkipWithError("transform failed");
      return;
    }
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * kSamples));
  state.SetLabel(backend == Backend::kGpu ? "GPU" : "CPU");
}

void BenchmarkElmTrainCpu(benchmark::State& state) {
  runTrain(state, Backend::kCpu);
}
void BenchmarkElmTrainGpu(benchmark::State& state) {
  runTrain(state, Backend::kGpu);
}
void BenchmarkHiddenTransformCpu(benchmark::State& state) {
  runTransform(state, Backend::kCpu);
}
void BenchmarkHiddenTransformGpu(benchmark::State& state) {
  runTransform(state, Backend::kGpu);
}

}  // namespace

BENCHMARK(BenchmarkElmTrainCpu)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(BenchmarkElmTrainGpu)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(BenchmarkHiddenTransformCpu)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(BenchmarkHiddenTransformGpu)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
