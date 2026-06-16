# H-OS-ELM migration to learned ELM-AE stacks

U5 replaces the v1 fixed random projection stack with a learned `ElmAutoEncoderLayer` stack and an online `RlsSolver` head.

## Behavior change

- `HierarchicalOsElm::initialize` fits the ELM-AE feature stack on the initialization chunk.
- `HierarchicalOsElm::update` updates only the online RLS output head.
- The old fixed `hiddenWeights_` / `hiddenBiases_` internals are removed.
- The final feature representation is exposed through `featureStack()` for inspection and tests.

## API notes

Existing four-argument construction still works:

```cpp
feature_elm::HierarchicalOsElm<double> model(
    numInputs,
    hiddenNodesPerLayer,
    feature_elm::ActivationFunction::kSigmoid,
    feature_elm::Backend::kCpu);
```

Additional controls are available through the extended constructor:

```cpp
feature_elm::HierarchicalOsElm<double> model(
    numInputs,
    hiddenNodesPerLayer,
    feature_elm::ActivationFunction::kSigmoid,
    feature_elm::Backend::kCpu,
    feature_elm::RlsOptions<double>{},
    1e-6,
    42u);
```

## Regression notes

- Chunked initialization should agree with full initialization where the online update assumptions hold.
- The feature stack weights must change after `initialize`; fixed random stacks are no longer valid H-OS-ELM behavior.
- See [testing.md](../testing.md) for the H-OS-ELM test group.
