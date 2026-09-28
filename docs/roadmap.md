# Roadmap

## Completed

- Composable `FeatureMap` and `Solver` architecture.
- Batch ridge solve with tunable regularization.
- Online RLS with ReOS-ELM, FOS-ELM, and OS-CELM toggles.
- Real RBF feature maps.
- Learned ELM-AE feature extraction, ML-ELM, and H-OS-ELM.
- CUDA primitive layer for shared GPU operations.
- Dataset IO, preprocessing, and drift streams.
- Narrative documentation suite.
- **v0.2.0:** a correct GPU backend, verified against the CPU with parity tests on a real device.
  ML-ELM now runs on the GPU as well.
- **v0.2.0:** a digits-classification demo on cpp-httplib, with CPU and GPU images that are
  ready for Hugging Face Spaces.
- **v0.2.0:** a VitePress documentation site (migrated from MkDocs Material, whose support ends
  in November 2026) with a hand-written API reference.
- **Unreleased:** multithreaded CPU paths (OpenMP, bit-identical to serial), device-resident GPU
  RLS for OS-ELM/ReOS-ELM/H-OS-ELM, and full-size MNIST / Fashion-MNIST benchmarks.
- **Unreleased:** model files (`BatchElm::save`/`load`), the `felm-train` CLI, and an MNIST-trained
  hand-drawn digit model for the live ZeroGPU Space, which deploys automatically.
- **v0.2.0:** CI covering CUDA-toolchain and CPU-only builds, style checks, and demo smoke tests,
  plus GHCR publishing with SBOM and provenance attestations, and Dependabot.

## Next

| Item | Why |
|---|---|
| Keep data resident on the device between transform and solve | Batch ELM still copies the hidden layer to the host between transform and solve (OS-ELM state is already device-resident) |
| GPU recursive least squares for FOS-ELM / OS-CELM | Their per-sample forgetting and constraint terms still run on the CPU |
| GPU prediction read-out (`H·β`) | `predictBatch` multiplies on the CPU after a GPU transform |
| Self-hosted GPU CI runner | Hosted runners skip every CUDA runtime test |

## Future ideas

- Optional Python bindings after a spec update.
- A WebAssembly build of the CPU core, so the demo can run as a free static Hugging Face Space.
- More detailed drift-stream scenarios beyond the bundled synthetic stream.
