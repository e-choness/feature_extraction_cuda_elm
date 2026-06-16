# Testing

Tests run through CTest inside Docker.

## Full suite

```bash
docker compose run --rm dev ctest --output-on-failure
```

## Step gates

Use the `-R` regex from `UPDATE_PLAN.md` for the current step, then run the full suite before closing the step.

```bash
docker compose run --rm dev ctest -R 'elm_ae|ml_elm|h_os_elm|golden_baseline' --output-on-failure
docker compose run --rm dev ctest --output-on-failure
```

## Test groups

| Prefix | Coverage |
|---|---|
| `sanity` | Basic build/runtime sanity |
| `feature_map` | Map interface, additive map, identity map |
| `elm_cpu` | Batch ELM CPU behavior |
| `solver` | Batch ridge solver stability |
| `os_elm` | Online sequential learning and forgetting |
| `os_celm` | Class-distance constraint behavior |
| `rbf_features`, `rbf_map` | RBF utilities and map |
| `elm_ae`, `ml_elm`, `h_os_elm` | Learned feature extraction |
| `cuda_infra`, `elm_gpu`, `gpu_ops` | CUDA primitives; skip without GPU |
| `dataset_io` | Dataset loader, preprocessing, drift stream |
| `demo_api` | Demo JSON helpers and benchmark snapshot loading |
| `golden_baseline` | Numeric regression anchor |

## GPU tests

GPU tests must skip cleanly when no device is visible. CPU is the correctness reference for GPU comparisons.

## Golden baseline

`tests/test_golden_baseline.cpp` records fixed-seed behavior. It is the anti-regression anchor for internal refactors.
