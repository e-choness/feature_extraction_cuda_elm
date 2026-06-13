# RBF Features

Radial Basis Function feature mapping for ELM variants.

## Overview

RBF hidden nodes compute activation based on distance from learned centers:

```
h_i(x) = exp(-gamma * ||x - c_i||²)
```

Where:
- `c_i` are the RBF centers
- `gamma` controls the width of the radial basis functions

## API

```cpp
feature_elm::RbfParameters<float> params;
feature_elm::initializeRbfCentersRandom(trainData, numCenters, inputDim, &params);

std::vector<float> features;
feature_elm::computeRbfFeatures(input, numSamples, params, &features);

// Use with BatchElm
feature_elm::BatchElm<float> model(
    numInputs, numHiddenNodes, 
    feature_elm::ActivationFunction::kRbf,
    feature_elm::Backend::kCpu
);
```

## Integration

RBF nodes can be used with any ELM variant that supports custom activation functions.