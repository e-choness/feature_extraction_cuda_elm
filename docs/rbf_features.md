# RBF Features

Radial Basis Function feature mapping for ELM variants.

## Overview

RBF hidden nodes compute activation based on distance from learned centers:

```
h_i(x) = exp(-||x - c_i||² / (2σ²))
```

Where:
- `c_i` are the RBF centers (initialized randomly or via k-means)
- `σ` controls the width of the radial basis functions

**Note (v2):** RBF is now a proper feature map (`RbfMap`), not an activation variant.
Use `RbfMap` directly or as part of a feature stack (e.g., in ML-ELM or Hierarchical OS-ELM).

## API

```cpp
// Create and train an RBF feature map
feature_elm::RbfMap<float> rbfMap(
    inputDim,      // Input dimension
    numCenters,    // Number of RBF centers
    width,         // Width parameter sigma
    feature_elm::RbfCenterInit::kKMeans,  // or kRandom
    seed
);

// Fit centers on training data
rbfMap.fit(trainData, numSamples);

// Transform data to RBF features
std::vector<float> hiddenOutput;
rbfMap.transform(input, numSamples, &hiddenOutput);

// Use with solver for classification
feature_elm::BatchRidgeSolver<float> solver;
solver.solve(hiddenOutput, numSamples, targets, numOutputs, &weights);
```

## Integration

`RbfMap` implements the `FeatureMap` interface and can be composed into stacked
feature maps for multilayer ELM variants. It provides true center-based RBF nodes
that enable learning nonlinear patterns like concentric decision boundaries.