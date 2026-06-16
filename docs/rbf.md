# RBF features

RBF features are center-based radial basis functions. In v2, RBF is a `FeatureMap`, not an activation enum.

## Math

Each RBF node computes:

```text
φ_i(x) = exp(-||x - c_i||² / (2σ²))
```

where `c_i` is a center and `σ` is the shared width.

## API

```cpp
feature_elm::RbfMap<float> rbfMap(
    inputDim,
    numCenters,
    0.75f,
    feature_elm::RbfCenterInit::kKMeans,
    42u);

rbfMap.fit(trainData, numSamples);

std::vector<float> features;
rbfMap.transform(trainData, numSamples, &features);
```

## Center initialization

| Mode | Behavior | Use |
|---|---|---|
| `kRandom` | Centers are sampled from the data range | Fast, deterministic ablations |
| `kKMeans` | Centers are selected with k-means++ style sampling | Better coverage of data support |

## Width selection

- Small `σ` creates local, narrow basis functions.
- Large `σ` makes nodes overlap and behave more globally.
- Start near the median nearest-neighbor distance when the scale is unknown.

## When to use

- Inputs have meaningful Euclidean geometry.
- Classes form local clusters.
- A single additive ELM layer struggles with concentric or local boundaries.

## When not to use

- Inputs are high-dimensional and sparse without a meaningful distance metric.
- You need a cheap linear baseline: use `IdentityMap`.
- You need learned feature extraction: use [ELM-AE](elm_ae.md) and [ML-ELM](ml_elm.md).
