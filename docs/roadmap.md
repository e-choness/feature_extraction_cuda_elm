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
- **v0.2.0:** CI covering CUDA-toolchain and CPU-only builds, style checks, and demo smoke tests,
  plus GHCR publishing with SBOM and provenance attestations, and Dependabot.

## Next

| Item | Why |
|---|---|
| GPU recursive least squares for OS-ELM / H-OS-ELM | Only the hidden transform runs on the GPU today, so online models see no speed-up |
| Keep data resident on the device between transform and solve | Each call currently copies to and from the host |
| GPU prediction read-out (`H·β`) | `predictBatch` multiplies on the CPU after a GPU transform |
| Multithreaded or BLAS-backed CPU solver | The CPU reference is single-threaded, which inflates GPU speed-ups |
| Self-hosted GPU CI runner | Hosted runners skip every CUDA runtime test |

## Future ideas

- Optional Python bindings after a spec update.
- More datasets and larger benchmark snapshots.
- A WebAssembly build of the CPU core, so the demo can run as a free static Hugging Face Space.
- Model export format for demo reproducibility.
- More detailed drift-stream scenarios beyond the bundled synthetic stream.
