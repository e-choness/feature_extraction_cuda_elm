# API Reference

A map of the public API. Every type lives in namespace `feature_elm`, the CUDA entry points in
`feature_elm::cuda_backend`. Matrices are `std::vector<FloatT>` in row-major order, and fallible
calls return `bool` or `std::optional` instead of throwing. The headers are the source of truth;
each heading links to one.

## Common types — [`core/feature_map.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/feature_map.hpp)

| Symbol | Purpose |
|---|---|
| `enum class Backend { kCpu, kGpu }` | Where transforms and solves run. `kGpu` returns `false` when no CUDA device is present |
| `enum class ActivationKind { kSigmoid, kTanh, kRelu }` | Hidden-layer non-linearity used by feature maps |
| `enum class ActivationFunction { kSigmoid, kTanh, kRelu }` | The same choice at model level; `activationKind()` converts it |
| `template <class FloatT> class FeatureMap` | Interface: `inputDim()`, `outputDim()`, `fit(data, n)`, `transform(input, n, &out)`, `setBackend(b)` |

## Feature maps

| Class | Header | Notes |
|---|---|---|
| `RandomAdditiveMap<FloatT>(inputDim, outputDim, activation, seed?, backend, weights?, biases?)` | [`core/random_additive_map.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/random_additive_map.hpp) | `g(xW + b)` with random `W`, `b`; `fit` is a no-op. Weights are `inputDim × outputDim` |
| `RbfMap<FloatT>(inputDim, numCenters, width, RbfCenterInit, seed)` | [`core/rbf_map.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/rbf_map.hpp) | Gaussian RBF, `exp(-‖x - c‖² / 2σ²)`; `fit` picks centres (`kRandom` or `kKMeans`) |
| `ElmAutoEncoderLayer<FloatT>(inputDim, outputDim, activation, seed, ridgeAlpha)` | [`core/elm_ae.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/elm_ae.hpp) | Learned encoder: `fit` solves an auto-encoder and reuses its output weights; `reconstruct()` for diagnostics |
| `StackedFeatureMap<FloatT>(inputDim, layerDims, activation, seed, ridgeAlpha, backend)` | [`core/stacked_feature_map.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/stacked_feature_map.hpp) | Greedy layer-wise stack of auto-encoder layers; `setBackend` applies to every layer |
| `IdentityMap<FloatT>(dim)` | [`core/identity_map.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/identity_map.hpp) | Pass-through, for linear baselines |

Lower-level RBF helpers (`RbfParameters`, `computeRbfFeatures`, `initializeRbfCentersRandom`,
`initializeRbfCentersKMeans`) are in [`core/rbf_features.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/rbf_features.hpp).

## Solvers

| Symbol | Header | Notes |
|---|---|---|
| `SolverOptions<FloatT>{ridgeAlpha, path, method}` | [`core/solver.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/solver.hpp) | `path`: `kAuto` (dual when samples < features), `kPrimal` or `kDual`. `method`: `kCholesky` or `kHouseholderQr` |
| `BatchRidgeSolver<FloatT>(options).solve(H, n, T, outputs, &β)` | same | Solves `(HᵀH + αI)β = HᵀT`. `β` is `features × outputs` |
| `RlsOptions<FloatT>{regularization, forgettingFactor, constraint, constraintStrength}` | [`core/rls_solver.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/rls_solver.hpp) | Forgetting factor < 1 gives FOS-ELM; `RlsConstraint::kClassDistance` gives OS-CELM |
| `RlsSolver<FloatT>(options)`: `initialize(H, n, T, outputs)`, `update(H, n, T)` | same | Recursive least squares; `weights()` and `covariance()` expose the state |

## Models

All models share `predict(x)` and `predictBatch(X, n)`, which return `std::optional<std::vector<FloatT>>`.

| Class | Header | Training API |
|---|---|---|
| `BatchElm<FloatT>(inputs, hidden, activation, backend, ridgeAlpha)`; an overload also takes explicit `weights, biases` | [`core/elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/elm.hpp) | `train(X, T, n, outputs)` |
| `OsElm<FloatT>(inputs, hidden, activation, backend, rlsOptions)`; an overload also takes explicit `weights, biases` | [`core/os_elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/os_elm.hpp) | `initialize(X0, T0, n0, outputs)`, then `update(X, T, n)` per chunk |
| `OsCelm<FloatT>(…)` | [`core/os_celm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/os_celm.hpp) | Same as OS-ELM, with a class-distance constraint |
| `MlElm<FloatT>(inputs, hiddenPerLayer, activation, backend, ridgeAlpha, seed)` | [`core/ml_elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/ml_elm.hpp) | `train(X, T, n, outputs)`: auto-encoder stack plus ridge read-out |
| `HierarchicalOsElm<FloatT>(inputs, hiddenPerLayer, activation, backend, rlsOptions, ridgeAlpha, seed)` | [`core/h_os_elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/h_os_elm.hpp) | `initialize` / `update`: the online version of ML-ELM |

## Data

| Symbol | Header | Notes |
|---|---|---|
| `loadCsv(path, inputDim, labelColumn, labelColumnFirst)` | [`io/dataset.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/io/dataset.hpp) | Returns `DatasetLoadResult{optional<Dataset>, error}` |
| `preprocessDataset(data, labels, n, dim, trainFraction, seed)` | [`io/preprocess.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/io/preprocess.hpp) | Deterministic split, min-max scaling fitted on the training split, one-hot targets |
| `minMaxNormalize(data, min, max)`, `oneHotEncode(labels, classes)` | same | The building blocks used above |
| `DriftStream(Config{inputDim, numClasses, streamLength, driftPoint, seed})`: `next()`, `reset()` | [`io/drift_stream.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/io/drift_stream.hpp) | Synthetic rotating-boundary stream with one concept drift |

## CUDA backend — [`cuda/`](https://github.com/e-choness/feature_extraction_cuda_elm/tree/master/src/cuda)

You normally reach these through `Backend::kGpu`. They are public for benchmarks and advanced use.

| Symbol | Notes |
|---|---|
| `isGpuAvailable()`, `gpuDeviceName()` | Device detection; the result is cached per process |
| `transformRandomAdditiveGpu(...)`, `transformElmAutoEncoderGpu(...)` | cuBLAS GEMM plus a fused bias and activation kernel |
| `solveRidgeGpu(H, T, n, outputs, options, &β)` | cuSOLVER Householder QR on the augmented system; no `HᵀH` is formed |
| `DeviceBuffer<T>` | RAII device allocation with bounds-checked host copies |

CPU-only builds link stubs that return `false` from every GPU entry point (`src/cuda/cuda_stubs.cpp`).
