#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "core/rls_solver.hpp"
#include "core/solver.hpp"
#include "cuda/gpu_ops.hpp"
#include "cuda/solver_gpu.hpp"

namespace {

using feature_elm::BatchRidgeSolver;
using feature_elm::RidgeSolveMethod;
using feature_elm::RidgeSolvePath;
using feature_elm::RlsConstraint;
using feature_elm::RlsOptions;
using feature_elm::RlsSolver;
using feature_elm::SolverOptions;

std::vector<double> makeFeatures(std::size_t numSamples, std::size_t numFeatures, unsigned int seed) {
  std::vector<double> features(numSamples * numFeatures);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(-0.5, 0.5);
  for (double& value : features) {
    value = distribution(generator);
  }
  return features;
}

std::vector<double> makeTargets(std::size_t numSamples, std::size_t numOutputs, unsigned int seed) {
  std::vector<double> targets(numSamples * numOutputs);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (double& value : targets) {
    value = distribution(generator);
  }
  return targets;
}

void setSolverCounters(benchmark::State& state, int deviceCode, std::size_t numSamples,
                       std::size_t numFeatures, std::size_t numOutputs) {
  state.counters["device"] = benchmark::Counter(static_cast<double>(deviceCode));
  state.counters["dataset_size"] = benchmark::Counter(
      static_cast<double>(numSamples), benchmark::Counter::kDefaults,
      benchmark::Counter::OneK::kIs1000);
  state.counters["num_features"] = benchmark::Counter(static_cast<double>(numFeatures));
  state.counters["num_outputs"] = benchmark::Counter(static_cast<double>(numOutputs));
}

void BenchmarkRidgeSolveCholeskyPrimal(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t kNumFeatures = 128;
  constexpr std::size_t kNumOutputs = 2;
  const std::vector<double> features = makeFeatures(numSamples, kNumFeatures, 41u);
  const std::vector<double> targets = makeTargets(numSamples, kNumOutputs, 43u);
  const BatchRidgeSolver<double> solver(SolverOptions<double>{
      1e-6, RidgeSolvePath::kPrimal, RidgeSolveMethod::kCholesky});
  std::vector<double> weights;

  setSolverCounters(state, 0, numSamples, kNumFeatures, kNumOutputs);
  state.SetLabel("device=CPU;path=primal;method=cholesky");

  for (auto _ : state) {
    if (!solver.solve(features, numSamples, targets, kNumOutputs, &weights)) {
      state.SkipWithError("primal Cholesky solve failed");
      return;
    }
    benchmark::DoNotOptimize(weights.data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(numSamples * kNumOutputs));
}

void BenchmarkRidgeSolveCholeskyDual(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t kNumFeatures = 256;
  constexpr std::size_t kNumOutputs = 2;
  const std::vector<double> features = makeFeatures(numSamples, kNumFeatures, 47u);
  const std::vector<double> targets = makeTargets(numSamples, kNumOutputs, 53u);
  const BatchRidgeSolver<double> solver(SolverOptions<double>{
      1e-6, RidgeSolvePath::kDual, RidgeSolveMethod::kCholesky});
  std::vector<double> weights;

  setSolverCounters(state, 0, numSamples, kNumFeatures, kNumOutputs);
  state.SetLabel("device=CPU;path=dual;method=cholesky");

  for (auto _ : state) {
    if (!solver.solve(features, numSamples, targets, kNumOutputs, &weights)) {
      state.SkipWithError("dual Cholesky solve failed");
      return;
    }
    benchmark::DoNotOptimize(weights.data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(numSamples * kNumOutputs));
}

void BenchmarkRidgeSolveGpuQr(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t kNumFeatures = 128;
  constexpr std::size_t kNumOutputs = 2;
  const std::vector<double> features = makeFeatures(numSamples, kNumFeatures, 59u);
  const std::vector<double> targets = makeTargets(numSamples, kNumOutputs, 61u);
  std::vector<double> weights;

  setSolverCounters(state, 1, numSamples, kNumFeatures, kNumOutputs);
  state.SetLabel("device=GPU;method=cusolver-qr");

  if (!feature_elm::cuda_backend::isGpuAvailable()) {
    state.SkipWithError("No GPU available");
    return;
  }

  for (auto _ : state) {
    if (!feature_elm::cuda_backend::solveRidgeGpu(
            features, targets, numSamples, kNumOutputs,
            SolverOptions<double>{1e-6, RidgeSolvePath::kPrimal, RidgeSolveMethod::kHouseholderQr},
            &weights)) {
      state.SkipWithError("GPU QR solve failed");
      return;
    }
    benchmark::DoNotOptimize(weights.data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(numSamples * kNumOutputs));
}

void BenchmarkRlsUpdate(benchmark::State& state) {
  const std::size_t numSamples = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t kNumFeatures = 64;
  constexpr std::size_t kNumOutputs = 1;
  const std::vector<double> initialFeatures = makeFeatures(16, kNumFeatures, 67u);
  const std::vector<double> initialTargets = makeTargets(16, kNumOutputs, 71u);
  const std::vector<double> features = makeFeatures(numSamples, kNumFeatures, 73u);
  const std::vector<double> targets = makeTargets(numSamples, kNumOutputs, 79u);
  RlsSolver<double> solver(RlsOptions<double>{1e-3, 1.0, RlsConstraint::kNone, 0.0});

  setSolverCounters(state, 0, numSamples, kNumFeatures, kNumOutputs);
  state.SetLabel("device=CPU;solver=rls");

  if (!solver.initialize(initialFeatures, initialFeatures.size() / kNumFeatures, initialTargets,
                         kNumOutputs)) {
    state.SkipWithError("RLS initialization failed");
    return;
  }

  for (auto _ : state) {
    if (!solver.update(features, numSamples, targets)) {
      state.SkipWithError("RLS update failed");
      return;
    }
    benchmark::DoNotOptimize(solver.weights().data());
  }

  state.SetItemsProcessed(state.iterations() * static_cast<long long>(numSamples * kNumOutputs));
}

}  // namespace

BENCHMARK(BenchmarkRidgeSolveCholeskyPrimal)->Arg(64)->Arg(128)->Arg(256);
BENCHMARK(BenchmarkRidgeSolveCholeskyDual)->Arg(32)->Arg(64)->Arg(128);
BENCHMARK(BenchmarkRidgeSolveGpuQr)->Arg(64)->Arg(128)->Arg(256);
BENCHMARK(BenchmarkRlsUpdate)->Arg(64)->Arg(128)->Arg(256);

BENCHMARK_MAIN();
