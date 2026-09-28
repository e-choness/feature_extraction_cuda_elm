# Benchmarks

Benchmarks measure the v2 primitives that matter: feature maps, solvers, online updates, and ML-ELM fit/forward paths.

## Running benchmarks

```bash
docker compose run --rm dev-gpu ./scripts/run_benchmarks.sh   # or `dev` on a machine without a GPU
docker compose run --rm dev ./scripts/gen_benchmark_badge.sh  # refresh the README table and badge
node scripts/gen_banner.mjs                                   # refresh images/banner.svg (CPU/GPU race numbers)
```

The script builds benchmark targets and writes JSON files to `data/benchmarks/latest/`.

## Output files

| File | Contents |
|---|---|
| `bench_feature_maps.json` | Additive, RBF, and ELM-AE transform benchmarks |
| `bench_solvers.json` | Ridge Cholesky (primal/dual), GPU ridge, and RLS update benchmarks |
| `bench_ml_elm.json` | ML-ELM fit and forward-pass benchmarks |
| `bench_elm.json` | Batch ELM training and hidden-layer transform, CPU and GPU on identical workloads |
| `bench_datasets.json` | Full MNIST / Fashion-MNIST (60k images): Batch ELM and streaming OS-ELM, CPU vs GPU, with test accuracy |

## Required fields

Successful benchmark entries include:

- `name`
- `real_time`
- `cpu_time`
- `iterations`
- `time_unit`
- custom `dataset_size` counter
- custom `device` counter such as `CPU` or `GPU:sm_89`

GPU benchmark entries may report `error_occurred: true` on CPU-only hosts. Treat those entries as skipped runtime data, not correctness failures.

## Interpreting results

- Feature-map benchmarks isolate transform cost for additive, RBF, and ELM-AE layers.
- Solver benchmarks compare CPU Cholesky paths and RLS updates.
- ML-ELM benchmarks measure fit and forward cost, not accuracy.
- Use Google Benchmark JSON for downstream badge and table generation.

## Example

```json
{
  "benchmarks": [
    {
      "name": "BM_AdditiveTransform/1024",
      "iterations": 100,
      "real_time": 12000,
      "cpu_time": 11980,
      "time_unit": "ns",
      "dataset_size": 1024,
      "device": "CPU"
    }
  ]
}
```

## Full-size datasets

`bench_datasets` trains on all 60,000 training images of MNIST and Fashion-MNIST (784 inputs, 10
classes, float32) and records test accuracy on the 10,000 test images as a counter next to the
time. This is where the GPU's advantage is realistic: the digits demo has only 1,437 training
samples.

```bash
python3 scripts/fetch_datasets.py                     # once; stored in .cache/datasets/
docker compose run --rm dev-gpu ./scripts/run_benchmarks.sh
```

It covers Batch ELM at 1,024, 2,048 and 4,096 hidden units, and OS-ELM streaming the training set
(a 4,096-image initial block, then chunks of 1,000). Each configuration runs once; the CPU cases take
several minutes, so set `FEATURE_ELM_BENCH_DATASETS=0` to skip the suite. Streaming OS-ELM reaches
the same accuracy as Batch ELM because both compute the same ridge solution; only the order in which
data arrives differs.

## Current snapshot

The committed snapshot (September 2026, RTX 4080) was recorded with a working GPU; earlier snapshots
had every GPU entry fail with "No GPU available". The README table is generated from it by
`scripts/gen_benchmark_badge.sh`.

`bench_elm` runs the same Batch ELM workload (2048 samples, 64 inputs, 10 outputs, float32) on both
backends, with the CPU using all 32 threads. Training is ~7× faster on the GPU at 256 hidden units
and ~42× faster at 2,048. On the full 60k-image datasets (see the README table), Batch ELM is 4–18×
faster and streaming OS-ELM 63–279× faster, because the GPU keeps the online solver's state
resident and absorbs each chunk with a few large GEMMs instead of per-sample rank-1 updates.

A tiny isolated ridge solve (256 features) is roughly a tie: fixed kernel-launch and transfer costs
dominate at that size. The demo's timings are noisier, because each request is a single cold call
on a desktop GPU that idles between requests; see [Demos](./demos.md#evaluation-results).
