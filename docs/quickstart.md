# Quickstart

Use the CPU backend for correctness checks and the GPU backend for larger batch transforms and ridge solves.

## Train a batch ELM

```cpp
#include <vector>

#include "core/elm.hpp"

int main() {
  constexpr std::size_t numInputs = 2;
  constexpr std::size_t numHiddenNodes = 8;
  constexpr std::size_t numSamples = 4;
  constexpr std::size_t numOutputs = 1;

  std::vector<float> trainData = {
      0.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
      1.0f, 1.0f,
  };
  std::vector<float> targets = {0.0f, 1.0f, 1.0f, 2.0f};

  feature_elm::BatchElm<float> model(
      numInputs,
      numHiddenNodes,
      feature_elm::ActivationFunction::kSigmoid,
      feature_elm::Backend::kCpu,
      1e-6f);

  if (!model.train(trainData, targets, numSamples, numOutputs)) {
    return 1;
  }

  auto prediction = model.predict({0.5f, 0.5f});
  return prediction.has_value() ? 0 : 1;
}
```

## Train ML-ELM on stacked ELM-AE features

```cpp
#include <vector>

#include "core/ml_elm.hpp"

int main() {
  constexpr std::size_t numInputs = 64;
  constexpr std::size_t numSamples = 100;
  constexpr std::size_t numOutputs = 10;

  std::vector<float> trainData(numSamples * numInputs);
  std::vector<float> targets(numSamples * numOutputs);

  feature_elm::MlElm<float> model(
      numInputs,
      {64, 32},
      feature_elm::ActivationFunction::kSigmoid,
      feature_elm::Backend::kCpu,
      1e-6f,
      42u);

  return model.train(trainData, targets, numSamples, numOutputs) ? 0 : 1;
}
```

## Run an online stream

```cpp
#include <vector>

#include "core/os_elm.hpp"

int main() {
  feature_elm::RlsOptions<float> options;
  options.forgettingFactor = 0.98f;  // FOS-ELM behavior.

  feature_elm::OsElm<float> model(
      2,
      16,
      feature_elm::ActivationFunction::kSigmoid,
      feature_elm::Backend::kCpu,
      options);

  std::vector<float> initialData = {0.0f, 0.0f, 1.0f, 1.0f};
  std::vector<float> initialTargets = {0.0f, 1.0f};

  if (!model.initialize(initialData, initialTargets, 2, 1)) {
    return 1;
  }

  std::vector<float> newData = {2.0f, 2.0f};
  std::vector<float> newTargets = {1.0f};
  return model.update(newData, newTargets, 1) ? 0 : 1;
}
```

## Run the demo

```bash
docker build -f docker/Dockerfile.demo.cpu -t feature-elm-demo-cpu .
docker run --rm -p 8888:8888 feature-elm-demo-cpu
```

Open `http://localhost:8888` after the server starts.
