# Choosing a model

Choose the smallest model that matches the data and latency requirements.

| Need | Recommended model | Feature map | Solver | Backend |
|---|---|---|---|---|
| Fast baseline on tabular data | Batch ELM | `RandomAdditiveMap` | `BatchRidgeSolver` | CPU or GPU |
| Nonlinear toy with local structure | RBF ELM | `RbfMap` | `BatchRidgeSolver` | CPU |
| Incremental data chunks | OS-ELM | `RandomAdditiveMap` or `RbfMap` | `RlsSolver` | CPU |
| Concept drift | FOS-ELM | `RandomAdditiveMap` | `RlsSolver` with `forgettingFactor < 1` | CPU |
| Ill-conditioned online initialization | ReOS-ELM | `RandomAdditiveMap` | `RlsSolver` with `regularization > 0` | CPU |
| Class-distance constraint | OS-CELM | `RandomAdditiveMap` | `RlsSolver` with `constraint = kClassDistance` | CPU |
| Learned feature extraction | ML-ELM | `StackedFeatureMap<ElmAutoEncoderLayer>` | `BatchRidgeSolver` | CPU or GPU |
| Online learned feature stack | H-OS-ELM | `StackedFeatureMap<ElmAutoEncoderLayer>` | `RlsSolver` | CPU or GPU |

## Decision guide

1. Start with `BatchElm` when you need a deterministic baseline.
2. Switch to `MlElm` when a single random additive layer underfits nonlinear structure.
3. Use `OsElm` when data arrives in chunks and retraining the full batch is too expensive.
4. Enable forgetting only when the stream distribution changes over time.
5. Use `RbfMap` when centers and width are meaningful for the problem geometry.
6. Keep `IdentityMap` for ablations and linear baselines.

## Backend selection

- `Backend::kCpu` is the correctness reference and is required for all algorithms.
- `Backend::kGpu` is intended for batch feature transforms and ridge solves.
- GPU tests skip cleanly when no CUDA device is visible.
