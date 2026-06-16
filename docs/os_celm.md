# OS-CELM

OS-CELM is OS-ELM with the class-distance constraint attached to the RLS solver.

## Concept

The feature map remains the same additive or RBF map used by OS-ELM. The constraint changes how the online solver accounts for class separation during updates.

```cpp
feature_elm::RlsOptions<float> options;
options.constraint = feature_elm::RlsConstraint::kClassDistance;
options.constraintStrength = 1e-2f;
```

## API

```cpp
feature_elm::OsCelm<float> model(
    numInputs,
    numHiddenNodes,
    feature_elm::ActivationFunction::kSigmoid,
    feature_elm::Backend::kCpu,
    1e-2f,
    options);

model.initialize(initialData, initialTargets, initialSamples, numOutputs);
model.update(newData, newTargets, newSamples);
```

## Constraint off-switch

Set `constraint = RlsConstraint::kNone` to recover ordinary OS-ELM behavior. This keeps the extension testable and prevents hidden behavior changes.

## When to use

- Classification streams where class separation matters.
- Small initial chunks where constrained updates improve stability.
- Comparisons against unconstrained OS-ELM.

## When not to use

- Regression targets: class-distance constraints are not meaningful.
- Streams with many classes and sparse labels: verify the constraint does not overfit early class counts.
