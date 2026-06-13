# H-OS-ELM

Hierarchical Online Sequential ELM.

## Overview

H-OS-ELM stacks multiple OS-ELM subnetworks where each layer acts as a feature extractor for the next layer.

## Architecture

1. Lower layers extract features incrementally
2. Higher layers classify based on extracted features
3. Each layer can use different hidden node counts and activation functions

## API

```cpp
feature_elm::H_OsElm<float> model(
    inputDim,  // Input dimension
    {64, 32, 16},  // Hidden nodes per layer
    feature_elm::ActivationFunction::kSigmoid
);

model.train(trainData, trainTargets, numSamples, numOutputs);

// Online prediction
auto features = model.extractFeatures(input);
auto prediction = model.predict(input);
```