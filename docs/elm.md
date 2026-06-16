# Batch ELM

Batch ELM is a single hidden-layer feedforward network with random additive features and a closed-form output layer.

## Math

For samples `X`, random additive features compute:

```text
H = g(XW + b)
```

The output weights solve the regularized least-squares problem:

```text
minβ ||Hβ - T||² + α||β||²
```

The CPU default solves the primal system when it is cheaper:

```text
(HᵀH + αI)β = HᵀT
```

When samples are fewer than hidden nodes, `BatchRidgeSolver` can use the dual form:

```text
(HHᵀ + αI)α_dual = T
β = Hᵀα_dual
```

## API

```cpp
feature_elm::BatchElm<float> model(
    numInputs,
    numHiddenNodes,
    feature_elm::ActivationFunction::kSigmoid,
    feature_elm::Backend::kCpu,
    1e-6f);

model.train(trainData, trainTargets, numSamples, numOutputs);
auto predictions = model.predictBatch(testData, testSamples);
```

`BatchElm` owns a `RandomAdditiveMap` and a `BatchRidgeSolver`. The constructor preserves the v1 shape while routing hidden computation through the feature-map layer.

## When to use

- Strong deterministic baseline.
- Small or medium tabular datasets.
- GPU batch transforms when `numSamples * numHiddenNodes` is large.
- Comparison baseline for RBF and ML-ELM.

## When not to use

- Data arrives continuously and full retraining is too expensive: use [OS-ELM](os_elm.md).
- The feature stack must be learned: use [ML-ELM](ml_elm.md).
- The problem is clearly center-based and local: compare against [RBF features](rbf.md).
