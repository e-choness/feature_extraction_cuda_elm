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
| `bench_solvers.json` | Ridge Cholesky (primal/dual), GPU QR ridge, and RLS update benchmarks |
| `bench_ml_elm.json` | ML-ELM fit and forward-pass benchmarks |
| `bench_elm.json` | Batch ELM training and hidden-layer transform, CPU and GPU on identical workloads |

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

## Current snapshot

The committed snapshot (September 2026, RTX 4080) was recorded with a working GPU; earlier snapshots
had every GPU entry fail with "No GPU available". The README table is generated from it by
`scripts/gen_benchmark_badge.sh`.

`bench_elm` runs the same Batch ELM workload (2048 samples, 64 inputs, 10 outputs, float32) on both
backends. With the GPU warm, training is ~6× faster at 256 hidden nodes, ~10× at 512, ~15× at
1,024 and ~27× at 2,048. The hidden-layer transform alone is 65–87× faster, because the CPU
reference is a plain triple loop.

The one benchmark the CPU wins is the isolated ridge solve at 256 features (0.45 ms against
4.1 ms), where fixed kernel-launch and transfer costs dominate. The demo's timings are noisier and
less favourable to the GPU, because each request is a single cold call on a desktop GPU that idles
between requests; see [Demos](./demos.md#evaluation-results).
