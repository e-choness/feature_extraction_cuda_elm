# Configuration

The v2 code exposes configuration through model constructors and option structs rather than one monolithic `ElmConfig`. The effective configuration surface is the same: feature stack, solver options, backend, and seed.

## Common fields

| Field | Type | Default | Applies to | Notes |
|---|---:|---:|---|---|
| `numInputs` | `std::size_t` | required | all models | Input dimension |
| `numHiddenNodes` | `std::size_t` | required | `BatchElm`, `OsElm`, `OsCelm` | Additive hidden-node count |
| `hiddenNodesPerLayer` | `std::vector<std::size_t>` | required | `MlElm`, `HierarchicalOsElm` | One ELM-AE layer per entry |
| `ActivationFunction` / `ActivationKind` | enum | `kSigmoid` | additive maps and ELM-AE | `kSigmoid`, `kTanh`, `kRelu` |
| `Backend` | enum | `kCpu` | all models and feature maps | `kCpu` or `kGpu` |
| `seed` | `unsigned int` | `42u` | random maps and ELM-AE | Reproducible initialization |
| `ridgeAlpha` | `FloatT` | `1e-6` | batch ridge and ELM-AE | Positive regularization |

## Batch solver options

`SolverOptions<FloatT>` controls `BatchRidgeSolver`.

| Option | Default | Values | Meaning |
|---|---:|---|---|
| `ridgeAlpha` | `1e-6` | `> 0` | Regularizes `HᵀH + αI` or `HHᵀ + αI` |
| `path` | `kAuto` | `kAuto`, `kPrimal`, `kDual` | Selects primal when samples dominate hidden nodes and dual otherwise |
| `method` | `kCholesky` | `kCholesky`, `kHouseholderQr` | CPU factorization method; Cholesky is the stable default |

Toggle `path = kPrimal` for many samples and few hidden nodes. Toggle `path = kDual` for many hidden nodes and few samples. Use `kHouseholderQr` only for research comparisons because Cholesky is faster on the regularized SPD system.

## Online solver options

`RlsOptions<FloatT>` controls `RlsSolver`.

| Option | Default | Values | Meaning |
|---|---:|---|---|
| `regularization` | `1e-3` | `>= 0` | ReOS-ELM covariance regularization |
| `forgettingFactor` | `1` | `(0, 1]` | FOS-ELM forgetting; `1` is ordinary OS-ELM |
| `constraint` | `kNone` | `kNone`, `kClassDistance` | OS-CELM class-distance constraint |
| `constraintStrength` | `1e-2` | `>= 0` | Strength for class-distance constraint |

An omitted extension must reduce to the baseline: `forgettingFactor = 1`, `regularization = 0`, and `constraint = kNone` are the off-switches.

## RBF configuration

`RbfMap` is configured by:

| Parameter | Default | Notes |
|---|---:|---|
| `inputDim` | required | Input dimension |
| `numCenters` | required | Output dimension |
| `width` | `1.0` | Shared RBF width `σ` |
| `centerInit` | `kRandom` | `kRandom` or `kKMeans` |
| `seed` | `42` | Random center initialization |

## Demo configuration

`DemoConfig` controls the demo server.

| Field | Default | Notes |
|---|---:|---|
| `useGpu` | `false` | Enables GPU demo behavior when available |
| `staticPath` | `demo/ui` | Static web UI directory |
| `benchmarkPath` | `data/benchmarks/latest` | Benchmark JSON directory |
| `port` | `8888` | HTTP listen port |

## Drift stream configuration

`DriftStream::Config` controls synthetic non-stationary streams.

| Field | Default | Notes |
|---|---:|---|
| `inputDim` | `2` | Minimum is 2 |
| `numClasses` | `2` | Binary drift stream |
| `streamLength` | `1000` | Samples before exhaustion |
| `driftPoint` | `500` | Sample where the regime changes |
| `seed` | `42` | Reproducible stream |
