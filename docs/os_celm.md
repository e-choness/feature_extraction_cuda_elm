# OS-CELM

Constrained Online Sequential ELM.

## Overview

OS-CELM applies class-distance-based constraints during hidden layer initialization to improve generalization.

## Key Differences from OS-ELM

- Hidden weights initialized to maximize separation between classes
- Centers placed near class centroids
- Better performance on classification tasks with limited data

## API

```cpp
feature_elm::OsCelm<float> model(
    numInputs, numHiddenNodes, feature_elm::ActivationFunction::kSigmoid
);

model.train(trainData, trainTargets, numSamples, numOutputs);
model.updateOnline(newData, newTargets, numNewSamples);
```