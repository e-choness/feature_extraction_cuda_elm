# Benchmarks

Running and interpreting microbenchmarks.

## Running Benchmarks

```bash
# In Docker
docker compose run --rm dev ./scripts/run_benchmarks.sh
```

## Output

Benchmarks produce JSON files in `data/benchmarks/latest/`:
- `bench_elm_core.json` - CPU benchmarks
- `bench_elm_cuda.json` - GPU benchmarks

## Measured Operations

- Hidden layer output computation (H matrix)
- Least-squares solving
- RBF feature mapping

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