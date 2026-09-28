# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multithreaded CPU paths (OpenMP, `FEATURE_ELM_OPENMP`, on by default): hidden-layer transforms, the
  normal equations, a blocked Cholesky, RLS updates and batch prediction. Threads split output
  elements and keep each element's summation order, so results are bit-identical to a serial run
  for any thread count. Ridge solves at 2,048 hidden units run ~6× faster on 32 threads.
- GPU recursive least squares for OS-ELM, ReOS-ELM and H-OS-ELM: device-resident state in
  information form (A = reg·I + ΣHᵀH accumulated in float64, solved on read). OS-ELM on the digits
  demo trains ~25× faster on the GPU than on the 32-thread CPU, and 63–279× faster streaming MNIST,
  with identical accuracy.
- GPU Batch ELM solves the normal equations first (cuBLAS `syrk` + cuSOLVER Cholesky), falling back
  to QR on the augmented system only if the Cholesky fails. MNIST at 4,096 hidden units trains in
  ~0.5 s, ~16× faster than the 32-thread CPU.
- `feature_elm::loadIdx` for MNIST-format datasets, `scripts/fetch_datasets.py`, and
  `bench_datasets`: full 60k-image MNIST and Fashion-MNIST runs, CPU vs GPU, with test accuracy.
- Model files: `BatchElm::save` / `BatchElm::load` (versioned binary format, see docs/api.md#model-files)
  and `felm-train`, a CLI that trains a classifier from CSV on the GPU and saves it.
- `data/models/handwriting_8x8.felm`: a hand-drawn digit classifier trained on MNIST plus the UCI
  digits (96.9% on held-out MNIST handwriting), rebuilt by `scripts/build_handwriting_data.py`.
  The Space loads it through the new `felm_load_classifier` C API.
- Automatic Hugging Face deployment: after CI passes on `master`, the ZeroGPU Space is rebuilt, uploaded
  and checked until it runs the new commit.
- `scripts/release.sh X.Y.Z` cuts a release: version bump, dated CHANGELOG section, commit and annotated
  tag. The release workflow checks the tag against the sources and publishes only that version's
  CHANGELOG section as the release notes (it used to post the whole file).
- The docs home page "Try the demo" button opens the live Hugging Face Space.

### Fixed
- CPU ridge solves that hit an ill-conditioned float32 Cholesky now retry in float64 before the
  (slow) QR fallback; 2,048-unit training on 20k samples went from never finishing to ~7 s.
- OS-ELM, OS-CELM and H-OS-ELM prediction returned out-of-bounds reads (a crash) if the solver's
  weights were unavailable; it now returns `std::nullopt`.
- CPU RLS allocated a fresh covariance matrix per sample and recomputed the OS-CELM class-distance
  term per sample; both removed (bit-identical results).
- Hand-drawn digits in the Space: the UCI-only classifier scored 46.5% on real handwriting (digit 6:
  27%). The MNIST-trained model with UCI-faithful preprocessing (`digitprep.py`) scores 97.2%
  end to end.
- A stray `lib/provenance.json` build attestation was uploaded to the Space.

## [0.2.0] - 2026-09-27

First release verified end to end on a real GPU. Before this release the CUDA path had never run
successfully: 9 of 77 tests failed as soon as a GPU was present.

### Fixed
- **GPU ridge solver** (`solveRidgeGpu`): the augmented matrix was packed transposed and written past
  the end of a host buffer (heap corruption). `orgqr` overwrote the R factor that the triangular
  solve needed, and the result was copied host-to-device instead of back. Rewritten as
  geqrf → ormqr → trsm, and matches the CPU Cholesky solve to 1e-8.
- **GPU hidden-layer transform**: the input was transposed on the host and the weights were read
  with the wrong layout, so the GEMM computed the wrong product. It now uploads the row-major data
  unchanged, with no host transposes.
- **`DeviceBuffer` ODR violation**: the header defined two different classes depending on
  `__CUDACC__`, so host code always got a stub that could never allocate.
- **ML-ELM ignored `Backend::kGpu`**: the feature stack was built without the backend, and both the
  auto-encoder and read-out solves always ran on the CPU. ML-ELM now trains ~4× faster (float32, 512 hidden) on an
  RTX 4080.
- The demo server read arbitrary files (`GET //../../etc/passwd`), crashed on a malformed
  `Content-Length`, ignored the `DEMO_*` paths inside its own images, treated `DEMO_USE_GPU=0` as
  enabled, and failed every `/run-inference` call.
- Every CUDA Dockerfile referenced a base image tag that does not exist
  (`13.3.0-cudnn-devel-ubuntu22.04`).
- Tests hard-coded `/workspace` paths, and CRLF checkouts broke the container shell scripts.
- `BatchRidgeSolver` returned `false` whenever Cholesky hit a non-positive pivot. With float32 and a
  tiny ridge, about 3% of random hidden layers triggered this, which made `ElmCpuTest.FloatPrecision`
  flaky in CI. It now falls back to Householder QR on the augmented system, which the 0.1.0 notes
  already claimed; results are unchanged whenever Cholesky succeeds.

### Added
- Digits demo: a live 8×8 drawing pad, train-and-evaluate for Batch ELM, OS-ELM and ML-ELM
  (accuracy, timings, confusion matrix), a CPU-vs-GPU sweep, and a JSON API under `/api/*`.
- CPU/GPU parity tests (all activations, several matrix shapes) and demo accuracy floors,
  plus HTTP security tests.
- `gpuDeviceName()`, the `FEATURE_ELM_STATIC_CUDA` CMake option, and default CUDA architectures
  from sm_75 to sm_120.
- Hugging Face Space templates (`deploy/huggingface`) and a sync workflow, Dependabot, a
  `dev-gpu` compose service, and a CPU-only CI job.
- A free GPU route on Hugging Face: a Gradio Space for ZeroGPU (`deploy/huggingface/zerogpu`) that
  loads the library through a new C API (`src/capi`, `-DFEATURE_ELM_BUILD_CAPI=ON`). The C API is
  built by `docker/Dockerfile.capi` against CUDA 12.8 and glibc 2.35, and CI builds it on every push.
- A 16:9 animated banner generated by `scripts/gen_banner.mjs` from the committed benchmark
  snapshot, with MP4, poster JPEG and 2:1 social-preview renders.

### Changed
- CUDA 13.3 → 13.4.1, Ubuntu 22.04 → 24.04, CMake ≥ 3.24, and C++20 for device code.
- cuBLAS/cuSOLVER handles are created once per process instead of on every call.
- `BatchRidgeSolver` forms `HᵀH` row by row over the upper triangle instead of striding down
  columns: ~15× faster CPU training at 512 hidden nodes, with bit-identical results.
- The interactive demo classifier (1,024 ReLU units, ±1-pixel shift augmentation,
  centre-of-mass centring, global input scaling) now reads hand-drawn digits correctly. The old
  sigmoid model with per-pixel scaling called a drawn 7 a 2.
- The hand-written HTTP server is replaced by cpp-httplib 0.58.0, and JSON handling uses
  nlohmann/json 3.12.0 (both pinned by SHA-256 via FetchContent).
- Demo images are multi-stage and non-root (UID 1000), and listen on port 7860. The GPU image
  builds against CUDA 12.8 with static cuBLAS/cuSOLVER, so it runs on any R525+ driver instead of
  needing R580+. It is now 954 MB instead of 5.1 GB.
- Docs moved from MkDocs Material (in maintenance mode) to VitePress 1.6. Doxygen is gone: most
  headers had no doc comments, so a curated API page replaces it (and doxygen/graphviz leave the
  dev image and CI).
- GitHub Actions updated to current majors (checkout v7, build-push v7, deploy-pages v5, …).
  Release images carry SBOM and provenance attestations.

### Removed
- Dead code: the unused free-function GPU ELM (`cuda/elm_gpu.*`), the pass-through
  `feature_map_gpu.hpp`, the toy vector-add kernel (`simple_kernel.*`), four duplicated
  activation-mapping switches, the unused `standardize()`, the stub headers
  (`device_buffer_stub.hpp`, `gpu_ops_stub.hpp`) and the empty `feature_map_gpu.cu`.
- `bench_elm_core` and `bench_elm_cuda` are merged into `bench_elm`, which runs identical CPU and
  GPU workloads.
- The old JPEG banners (`images/feature-extraction-*.jpg`) and the 4:1 SVG banner.
- The `/run-inference` and `/run-benchmark` endpoints (replaced by `/api/evaluate` and
  `/api/benchmark`).

## [0.1.0]

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
- Demo endpoints for health, benchmark snapshots and inference

### Removed
- Fake `ActivationFunction::kRbf` activation (replaced by proper `RbfMap`)
- Fixed random projection hierarchy (replaced by learned ELM-AE stack)