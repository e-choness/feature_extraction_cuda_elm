# Getting started

This project is Docker-first. Build, test, benchmark, and run demos inside the development container so host CUDA toolkits, compilers, and Python packages do not affect results.

## Prerequisites

- Docker Engine or Docker Desktop with Compose v2.
- NVIDIA driver compatible with CUDA 13.x when using GPU tests or the GPU demo.
- `nvidia-container-toolkit` on Linux hosts that need GPU passthrough.
- No host compiler, CUDA toolkit, or CMake installation is required.

Verify Docker Compose from the repository root:

```bash
docker compose run --rm dev bash -lc 'cmake --version && ctest --version'
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/core/` | CPU feature maps, solvers, and model compositions |
| `src/cuda/` | CUDA primitive kernels and RAII wrappers |
| `src/io/` | Dataset loading, preprocessing, and drift streams |
| `src/app/` | Demo HTTP backend |
| `tests/` | GoogleTest unit and integration tests |
| `bench/` | Google Benchmark microbenchmarks |
| `docs/` | Narrative docs and API entry point |
| `docker/` | Development and demo images |

## First commands

```bash
# Configure and run the full CPU test suite
docker compose run --rm dev ctest --output-on-failure

# Run the docs link and Mermaid sanity gate
docker compose run --rm dev ./scripts/docs_check.sh

# Run benchmarks and write JSON to data/benchmarks/latest/
docker compose run --rm dev ./scripts/run_benchmarks.sh
```

## GPU runtime note

The CUDA image builds without a GPU. Runtime GPU tests and the GPU demo require a compatible NVIDIA driver and container toolkit. If no GPU is visible, CUDA tests skip cleanly instead of failing.
