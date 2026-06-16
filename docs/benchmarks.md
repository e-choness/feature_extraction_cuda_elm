# Benchmarks

Benchmarks measure the v2 primitives that matter: feature maps, solvers, online updates, and ML-ELM fit/forward paths.

## Running benchmarks

```bash
docker compose run --rm dev ./scripts/run_benchmarks.sh
```

The script builds benchmark targets and writes JSON files to `data/benchmarks/latest/`.

## Output files

| File | Contents |
|---|---|
| `bench_feature_maps.json` | Additive, RBF, and ELM-AE transform benchmarks |
| `bench_solvers.json` | Ridge Cholesky, dual/primal behavior, and RLS update benchmarks |
| `bench_ml_elm.json` | ML-ELM fit and forward-pass benchmarks |
| `bench_elm_cuda.json` | Legacy CUDA ELM primitive benchmarks retained for comparison |

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
