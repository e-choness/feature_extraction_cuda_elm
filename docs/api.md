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
| `RlsSolver<FloatT>(options, backend)`: `initialize(H, n, T, outputs)`, `update(H, n, T)` | same | Recursive least squares; `weights()` and `covariance()` expose the state. With `Backend::kGpu`, plain RLS (forgetting 1, no constraint) keeps its state on the GPU in information form (float64) and `usesGpu()` returns true; FOS/OS-CELM settings stay on the CPU |

## Models

All models share `predict(x)` and `predictBatch(X, n)`, which return `std::optional<std::vector<FloatT>>`.

| Class | Header | Training API |
|---|---|---|
| `BatchElm<FloatT>(inputs, hidden, activation, backend, ridgeAlpha)`; an overload also takes explicit `weights, biases` | [`core/elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/elm.hpp) | `train(X, T, n, outputs)` |
| `OsElm<FloatT>(inputs, hidden, activation, backend, rlsOptions)`; an overload also takes explicit `weights, biases` | [`core/os_elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/os_elm.hpp) | `initialize(X0, T0, n0, outputs)`, then `update(X, T, n)` per chunk |
| `OsCelm<FloatT>(…)` | [`core/os_celm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/os_celm.hpp) | Same as OS-ELM, with a class-distance constraint |
| `MlElm<FloatT>(inputs, hiddenPerLayer, activation, backend, ridgeAlpha, seed)` | [`core/ml_elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/ml_elm.hpp) | `train(X, T, n, outputs)`: auto-encoder stack plus ridge read-out |
| `HierarchicalOsElm<FloatT>(inputs, hiddenPerLayer, activation, backend, rlsOptions, ridgeAlpha, seed)` | [`core/h_os_elm.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/core/h_os_elm.hpp) | `initialize` / `update`: the online version of ML-ELM |

## Model files

A trained `BatchElm` can be saved and reloaded, so a model can be trained once (for example on a GPU)
and served anywhere:

```cpp
BatchElm<float> model(64, 4096, ActivationFunction::kRelu, Backend::kGpu);
model.train(X, T, n, 10);
model.save("digits.felm");                                   // false if untrained or unwritable

auto served = BatchElm<float>::load("digits.felm", Backend::kCpu);  // std::nullopt on bad files
```

The file is binary and little-endian, with a small header followed by the raw weights:

| Field | Type | Notes |
|---|---|---|
| magic | `char[4]` | `FELM` |
| version | `uint32` | currently 1 |
| scalar width | `uint32` | 4 (`float`) or 8 (`double`); must match the `BatchElm<FloatT>` that loads it |
| activation | `uint32` | `ActivationFunction` value |
| inputs, hidden, outputs | `uint64` × 3 | |
| ridge alpha | `float64` | |
| hidden weights | scalar × inputs·hidden | row-major, inputs × hidden |
| hidden biases | scalar × hidden | |
| output weights | scalar × hidden·outputs | row-major, hidden × outputs |

`load` rejects wrong magic, versions or widths, out-of-range sizes, non-finite values and truncated
files. A reloaded model predicts bit-identically to the original.

### `felm-train`

A command-line trainer built with the library (the `felm_train` target). It reads labelled CSV
(`label,f1,…,fN` with a header), trains on the GPU when one is available, reports per-class test
accuracy, and saves a model file:

```bash
felm-train --train train.csv --test test.csv --inputs 64 --classes 10 \
           --hidden 4096 --activation relu --ridge 1 --input-scale 64 --out model.felm
```

`--input-scale S` divides the random hidden weights by `S`. That is equivalent to scaling the inputs
down, but the saved model then works directly on raw features. `data/models/handwriting_8x8.felm`
is built this way; see [`data/models/README.md`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/data/models/README.md).

## Data

| Symbol | Header | Notes |
|---|---|---|
| `loadCsv(path, inputDim, labelColumn, labelColumnFirst)` | [`io/dataset.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/io/dataset.hpp) | Returns `DatasetLoadResult{optional<Dataset>, error}` |
| `preprocessDataset(data, labels, n, dim, trainFraction, seed)` | [`io/preprocess.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/io/preprocess.hpp) | Deterministic split, min-max scaling fitted on the training split, one-hot targets |
| `minMaxNormalize(data, min, max)`, `oneHotEncode(labels, classes)` | same | The building blocks used above |
| `DriftStream(Config{inputDim, numClasses, streamLength, driftPoint, seed})`: `next()`, `reset()` | [`io/drift_stream.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/io/drift_stream.hpp) | Synthetic rotating-boundary stream with one concept drift |
| `loadIdx(imagesPath, labelsPath)` | [`io/idx_dataset.hpp`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/src/io/idx_dataset.hpp) | Uncompressed IDX pairs (MNIST, Fashion-MNIST) → `IdxDataset` with float pixels in 0..1; `scripts/fetch_datasets.py` downloads them |

## CUDA backend — [`cuda/`](https://github.com/e-choness/feature_extraction_cuda_elm/tree/master/src/cuda)

You normally reach these through `Backend::kGpu`. They are public for benchmarks and advanced use.

| Symbol | Notes |
|---|---|
| `isGpuAvailable()`, `gpuDeviceName()` | Device detection; the result is cached per process |
| `transformRandomAdditiveGpu(...)`, `transformElmAutoEncoderGpu(...)` | cuBLAS GEMM plus a fused bias and activation kernel |
| `solveRidgeGpu(H, T, n, outputs, options, &β)` | Normal equations on the device (`syrk` + cuSOLVER Cholesky), falling back to Householder QR on the augmented system if the Cholesky fails |
| `DeviceBuffer<T>` | RAII device allocation with bounds-checked host copies |

CPU-only builds link stubs that return `false` from every GPU entry point (`src/cuda/cuda_stubs.cpp`).

## Threading

CPU hot loops (hidden-layer transforms, the normal equations, Cholesky, RLS updates, batch
prediction) are multithreaded with OpenMP when the library is built with `FEATURE_ELM_OPENMP=ON`
(the default). Threads split *output elements* and each element keeps its summation order, so
results are bit-identical to a serial run for any thread count. Use `OMP_NUM_THREADS` to limit
threads; `feature_elm::parallelThreads()` (in `core/parallel.hpp`) reports the count in use.
