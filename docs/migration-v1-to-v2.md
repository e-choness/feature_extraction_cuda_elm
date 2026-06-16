# Migration from v1 to v2

v2 preserves the project goal but changes internals to a composable pipeline.

## Architecture changes

| v1 | v2 | Migration |
|---|---|---|
| `computeHiddenOutput` duplicated across models | `FeatureMap::transform` | Use `RandomAdditiveMap`, `RbfMap`, `ElmAutoEncoderLayer`, or `StackedFeatureMap` |
| Some models accepted `Backend` | Every model accepts `Backend` | Pass `Backend::kCpu` or `Backend::kGpu` explicitly |
| RBF activation enum | `RbfMap` feature map | Replace `ActivationFunction::kRbf` usage with `RbfMap` |
| H-OS-ELM fixed random stack | ELM-AE learned stack | Use `HierarchicalOsElm` with `hiddenNodesPerLayer` |
| Hard-coded `1e-8` ridge term | `ridgeAlpha` | Pass a positive `ridgeAlpha` |
| Demo fabricated training data | Bundled dataset and real metrics | Use dataset IO and demo endpoints |

## API notes

- `BatchElm` keeps its constructor shape and adds `ridgeAlpha`.
- `OsElm` and `OsCelm` accept `RlsOptions`.
- `OsCelm` gains `Backend` for consistency.
- `HierarchicalOsElm` now learns ELM-AE layers during initialization.

## Code migration example

Before:

```cpp
feature_elm::BatchElm<float> model(numInputs, hiddenNodes, feature_elm::ActivationFunction::kSigmoid);
```

After, with explicit backend and ridge alpha:

```cpp
feature_elm::BatchElm<float> model(
    numInputs,
    hiddenNodes,
    feature_elm::ActivationFunction::kSigmoid,
    feature_elm::Backend::kCpu,
    1e-6f);
```

## Documentation migration

See [upgrade baseline](upgrade/baseline.md) and [H-OS-ELM migration](upgrade/h_os_elm.md) for regression notes.
