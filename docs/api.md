# API Reference

Library API for Feature Extraction CUDA ELM.

## BatchElm

```cpp
template <typename FloatT = double>
class BatchElm {
public:
    // Constructor
    BatchElm(std::size_t numInputs, std::size_t numHiddenNodes,
             ActivationFunction activation, Backend backend);
    
    // Train on batch data
    bool train(const std::vector<FloatT>& trainData,
               const std::vector<FloatT>& trainTargets,
               std::size_t numSamples, std::size_t numOutputs);
    
    // Predict single sample
    std::optional<std::vector<FloatT>> predict(
        const std::vector<FloatT>& input) const;
    
    // Predict batch
    std::optional<std::vector<FloatT>> predictBatch(
        const std::vector<FloatT>& testData,
        std::size_t numSamples) const;
};
```

## OsElm

```cpp
template <typename FloatT = double>
class OsElm {
public:
    bool train(const std::vector<FloatT>& trainData,
               const std::vector<FloatT>& trainTargets,
               std::size_t numSamples, std::size_t numOutputs);
    
    bool updateOnline(const std::vector<FloatT>& newData,
                      const std::vector<FloatT>& newTargets,
                      std::size_t numNewSamples);
    
    std::optional<std::vector<FloatT>> predict(
        const std::vector<FloatT>& input) const;
};
```

## OsCelm

```cpp
template <typename FloatT = double>
class OsCelm {
    // Same API as OsElm with constrained initialization
};
```

## H_OsElm

```cpp
template <typename FloatT = double>
class H_OsElm {
public:
    H_OsElm(std::size_t inputDim,
            std::initializer_list<std::size_t> hiddenNodesPerLayer,
            ActivationFunction activation);
    
    bool train(const std::vector<FloatT>& trainData,
               const std::vector<FloatT>& trainTargets,
               std::size_t numSamples, std::size_t numOutputs);
    
    std::optional<std::vector<FloatT>> predict(
        const std::vector<FloatT>& input) const;
    
    std::vector<FloatT> extractFeatures(
        const std::vector<FloatT>& input) const;
};
```

## Enums

```cpp
enum class ActivationFunction {
    kSigmoid  // 1/(1+exp(-x)) - additive activation
};

enum class Backend {
    kCpu,  // CPU implementation
    kGpu   // GPU implementation (requires CUDA)
};
```

**Note:** RBF (Radial Basis Function) is now implemented via `RbfMap` feature map,
not as an activation function. See `RbfMap` for center-based RBF nodes.