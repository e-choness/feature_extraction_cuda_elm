# Documentation

This directory contains documentation for the Feature Extraction CUDA ELM project.

## Algorithm Overviews

- [Batch ELM](batch_elm.md) - Single hidden layer feedforward network with random hidden weights
- [OS-ELM](os_elm.md) - Online Sequential ELM for incremental learning
- [OS-CELM](os_celm.md) - Constrained OS-ELM with class-distance constraints
- [H-OS-ELM](h_os_elm.md) - Hierarchical OS-ELM with stacked feature extraction
- [RBF Features](rbf_features.md) - Radial Basis Function feature mapping

## Usage Guides

- [Benchmarks](benchmarks.md) - Running and interpreting microbenchmarks
- [Demos](demos.md) - Using the GPU and CPU demo servers
- [API](api.md) - Library API reference

## Architecture

```mermaid
graph TD
    subgraph Core["src/core"]
        ELM[BatchElm]
        OS_ELM[OsElm]
        OS_CELM[OsCelm]
        H_OS_ELM[H_OsElm]
        RBF[RbfFeatures]
    end

    subgraph CUDA["src/cuda"]
        DeviceBuffer[DeviceBuffer]
        Stream[Stream]
        ElMGpu[ElmGpu]
    end

    subgraph App["src/app"]
        DemoBackend[DemoBackend]
        DemoApp[DemoApp]
    end

    ELM -->|uses| CUDA
    OS_ELM -->|uses| CUDA
    OS_CELM -->|uses| CUDA
    H_OS_ELM -->|uses| CUDA
    RBF -->|uses| CUDA
    DemoBackend -->|uses| Core
    DemoBackend -->|uses| CUDA
    DemoApp -->|uses| DemoBackend
```

## Dataflow

### OS-ELM

```mermaid
flowchart LR
    Input[Input Data] --> Hidden[Compute Hidden Output H]
    Hidden --> LS[Least Squares Solve]
    LS --> OutputWeights[Output Weights β]
    OutputWeights --> Predict[Prediction]

    subgraph OnlineUpdates["Online Updates"]
        H_old[Old H^T H] --> Inv[Invert]
        Inv --> Beta_old[Old Output Weights]
        New[New Data] --> H_new[Compute H_new]
        H_new --> Update[Update Formula]
        Update --> Beta_new[New Output Weights]
    end
```

### H-OS-ELM

```mermaid
flowchart LR
    Input --> Layer1[Layer 1 ELM]
    Layer1 --> Features1[Extracted Features]
    Features1 --> Layer2[Layer 2 ELM]
    Layer2 --> Features2[Extracted Features]
    Features2 --> Top[Top Level Classifier]
    Top --> Prediction
```

## Tech Stack Mindmap

```mermaid
mindmap
  root((Feature ELM))
    Core
      C++20
      CMake
      GoogleTest
    GPU
      CUDA 13.x
      cuBLAS
      cuSOLVER
      Thrust
    Benchmarks
      Google Benchmark
      JSON Output
    Demo
      HTTP Server
      Web UI
```