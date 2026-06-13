# Batch ELM

Batch Extreme Learning Machine with additive hidden nodes.

## Overview

Batch ELM is a single hidden layer feedforward neural network where:
- Hidden layer weights and biases are randomly initialized (typically from uniform distribution)
- Output weights are computed analytically via least-squares solution: H·β = T

## API

```cpp
feature_elm::BatchElm<float> model(
    numInputs,      // Input dimension
    numHiddenNodes, // Hidden layer size
    feature_elm::ActivationFunction::kSigmoid,
    feature_elm::Backend::kCpu  // or kGpu
);

model.train(trainData, trainTargets, numSamples, numOutputs);
auto predictions = model.predictBatch(testData, numSamples);
```

## Parameters

- `numInputs`: Number of input features
- `numHiddenNodes`: Number of hidden neurons (typically 10-1000)
- `ActivationFunction`: kSigmoid (default), kRbf
- `Backend`: kCpu or kGpu

## GPU Implementation

- Hidden layer output computed via CUDA kernel
- Least-squares solved via cuBLAS matrix multiplication and cuSOLVER Cholesky decomposition
- Uses float or double precision