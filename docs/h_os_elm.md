#H - OS - ELM

Hierarchical Online Sequential ELM with learned ELM-AE feature extraction.

## Overview

H-OS-ELM composes a `StackedFeatureMap` of `ElmAutoEncoderLayer` layers with an online `RlsSolver` head. The feature stack is fitted greedily during initialization, then the online head consumes the learned features during updates.

## Architecture

1. Each layer solves an ELM autoencoder with input as target during initialization.
2. The encoder for the next layer is derived from the transposed autoencoder output weights.
3. The learned feature stack is frozen after initialization;
online updates train only the RLS head.4. The final online head uses recursive least squares,
    including ReOS - ELM, FOS - ELM,
    and OS - CELM toggles through `RlsOptions`.

             ##API

```cpp feature_elm::HierarchicalOsElm<double>
                 model(inputDim, {64, 32}, feature_elm::ActivationFunction::kSigmoid,
                       feature_elm::Backend::kCpu);

model.initialize(trainData, trainTargets, numSamples, numOutputs);
model.update(newData, newTargets, newNumSamples);

auto prediction = model.predictBatch(testData, testNumSamples);
```

    The learned feature stack is available through `model.featureStack()`.
