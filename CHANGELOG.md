# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Composable `FeatureMap` interface with `RandomAdditiveMap`, `RbfMap`, `ElmAutoEncoderLayer`, and `StackedFeatureMap` implementations
- `BatchRidgeSolver` with tunable regularization alpha and Cholesky QR fallback
- `RlsSolver` for online learning with optional regularization and forgetting factor
- ML-ELM (multilayer ELM) with learned feature extraction via ELM-AE layers
- Hierarchical OS-ELM reimplemented as ELM-AE stack with online head
- Dataset I/O with CSV loader, preprocessing, and drift stream generator
- Real handwritten digits (8x8) dataset bundled
- GitHub Actions CI/CD with automatic documentation deployment
- Demo images published to GitHub Container Registry

### Changed
- RBF feature map now correctly implements radial basis function (not squashed additive)
- GPU backend unified at feature-map/solver layer for all batch models
- Demo now trains on real bundled dataset instead of fabricated data

### Removed
- Fake `ActivationFunction::kRbf` activation (replaced by proper `RbfMap`)
- Fixed random projection hierarchy (replaced by learned ELM-AE stack)