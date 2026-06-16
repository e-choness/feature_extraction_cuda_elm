# Benchmarks

Running and interpreting microbenchmarks.

## Running Benchmarks

```bash
# In Docker
docker compose run --rm dev ./scripts/run_benchmarks.sh
```

## Output

Benchmarks produce JSON files in `data/benchmarks/latest/`:
- `bench_feature_maps.json` - CPU feature-map transforms for additive, RBF, and ELM-AE layers
- `bench_solvers.json` - CPU ridge Cholesky, GPU cuSOLVER QR, and RLS update benchmarks
- `bench_ml_elm.json` - ML-ELM fit and forward-pass benchmarks
- `bench_elm_cuda.json` - legacy CUDA ELM primitive benchmarks

Successful benchmark entries include `device` and `dataset_size` counters. GPU entries are emitted with
`error_occurred: true` on CPU-only hosts.

## Measured Operations

- Additive, RBF, and ELM-AE feature-map transforms
- Ridge solve: CPU Cholesky primal/dual and GPU cuSOLVER QR
- Recursive least-squares updates for OS-ELM-style online training
- ML-ELM fit and forward pass

## Example Output

```json
{
  "benchmarks": [
    {
      "name": "BenchmarkComputeHiddenOutput/256",
      "iterations": 140,
      "real_time": 5.0277187071433868e+06,
      "cpu_time": 5.0274971428571427e+06,
      "time_unit": "ns"
    }
  ]
}
```