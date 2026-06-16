# Baseline lock and regression net

Captured on 2026-06-14 with the unmodified master tree.

- Test target: `ctest -R golden_baseline --output-on-failure`
- Full-suite target: `docker compose run --rm dev ctest --output-on-failure`
- Benchmark snapshot will be recorded here once a golden benchmark manifest exists.

## Captured test list (U0)

| CTest Name | Status |
|---|---|
| `sanity.*` | passed |
| `elm_cpu.*` | passed |
| `os_elm.*` | passed |
| `os_celm.*` | passed |
| `h_os_elm.*` | passed |
| `rbf_features.*` | passed |
| `demo_api.*` | passed |
| `cuda_infra.*` | skipped (CPU-only) |
| `elm_gpu.*` | skipped (CPU-only) |

## Golden baseline expectations

- `BatchElm` with fixed weights/biases and deterministic input targets reproduces a
  bounded prediction MSE on training data.
- `OsElm` after initialization reproduces bounded prediction MSE on the same data.
- Online `OsElm::update` improves fit on held-out data within the same bounds.
- The golden test file is the regression anchor for U1 onward. Any later step must keep
  these three cases green.

## Regression net

Use this document to record when a refactor changes numeric behavior. Updates should
include the step that triggered the change, a brief justification, and a new threshold
if the old one is no longer appropriate.
