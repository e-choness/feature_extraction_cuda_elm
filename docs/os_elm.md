# OS-ELM

Online Sequential ELM for incremental learning.

## Overview

OS-ELM updates the output weights sequentially as new data arrives, avoiding retraining on all historical data.

## Mathematical Foundation

Given initial hidden layer H₁ and targets T₁:
- Initial solution: β₁ = (H₁^T H₁)^{-1} H₁^T T₁

For new data H₂, T₂:
- Update matrices: P = (H₂^T H₂)^{-1}, T = H₂^T T₂  
- Increment: β_new = β_old + (P - P·H₁^T(H₁·P·H₁^T + I)^{-1}·P·H₁^T)·T

## API

```cpp
feature_elm::OsElm<float> model(
    numInputs, numHiddenNodes, feature_elm::ActivationFunction::kSigmoid
);

// Initial batch training
model.train(trainData, trainTargets, numSamples, numOutputs);

// Online updates
model.updateOnline(newData, newTargets, numNewSamples);
auto predictions = model.predict(input);
```

## Variants

- **OS-CELM**: Constrained OS-ELM with class centroid-based initialization
- **H-OS-ELM**: Hierarchical extension with stacked networks