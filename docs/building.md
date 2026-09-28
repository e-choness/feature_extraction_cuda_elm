# Building

All builds run in Docker.

## Configure

```bash
docker compose run --rm dev cmake -S . -B /tmp/feature_elm_build -G Ninja
```

## Build

```bash
docker compose run --rm dev cmake --build /tmp/feature_elm_build
```

## Common build options

| Option | Default | Notes |
|---|---:|---|
| `CMAKE_BUILD_TYPE` | `Debug` from Compose | Use `Release` for benchmarks |
| `ENABLE_CUDA` | `ON` | Turns CUDA backend support on when CUDA is found |
| `BUILD_TESTING` | `ON` via CTest | Enables test and benchmark targets |
| `CMAKE_CUDA_ARCHITECTURES` | sm_75 to sm_120 fat binary | The dev entrypoint uses `native` with a GPU and `75` without; set e.g. `89` for faster local builds |
| `FEATURE_ELM_OPENMP` | `ON` | Multithread the CPU hot loops with OpenMP. Results are bit-identical to a serial build; set `OMP_NUM_THREADS` to control the thread count |
| `FEATURE_ELM_STATIC_CUDA` | `OFF` | Link cuBLAS/cuSOLVER statically (used by the GPU demo image) |
| `FEATURE_ELM_BUILD_CAPI` | `OFF` | Build `libfeature_elm_capi.so`, the C API used by the ZeroGPU Space (turns on position-independent code) |

## Disable CUDA on CPU-only hosts

```bash
docker compose run --rm dev cmake -S . -B /tmp/feature_elm_build_cpu \
  -G Ninja \
  -DENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug
```

CUDA stubs keep the public API compileable when no toolkit is available.

## Build outputs

- Libraries: `/tmp/feature_elm_build/`
- Test executables: `/tmp/feature_elm_build/tests/`
- Benchmark executables: `/tmp/feature_elm_build/bench/`

Use `FEATURE_ELM_BUILD_DIR` when a script needs a non-default build directory.
