# Troubleshooting

## GPU not detected

Symptoms:

- CUDA tests fail to find a device.
- GPU demo reports `gpu_available:false`.

Checks:

```bash
docker compose run --rm dev nvidia-smi
docker compose run --rm dev bash -lc 'ls /dev/nvidia* 2>/dev/null || true'
```

Fixes:

- Install or enable `nvidia-container-toolkit`.
- Restart Docker after driver installation.
- Ensure the host driver supports CUDA 13.x.
- Confirm `NVIDIA_VISIBLE_DEVICES=all` is set for the container.

## Driver mismatch

Symptoms:

- `CUDA driver version is insufficient for CUDA runtime version`.
- `nvidia-smi` fails inside the container.

Fixes:

- Update the host NVIDIA driver.
- Rebuild the dev image only if the Dockerfile CUDA tag changes.
- Do not install a host CUDA toolkit to fix container runtime issues.

## Out of memory

Symptoms:

- cuBLAS or cuSOLVER allocation fails.
- GPU tests skip or fail on large matrices.

Fixes:

- Reduce `numSamples`, `numHiddenNodes`, or layer widths.
- Prefer the dual ridge path when hidden nodes greatly exceed samples.
- Run smaller benchmarks before scaling up.

## Port conflicts

Symptoms:

- Demo server fails to bind to `0.0.0.0:8888`.

Fixes:

```bash
docker run --rm -p 8080:8888 feature-elm-demo-cpu
```

Change the host port while keeping the container port at `8888`.

## Broken docs links

Run:

```bash
docker compose run --rm dev ./scripts/docs_check.sh
```

Fix Markdown links relative to the page that contains them. Use `../architecture.md` from nested pages such as `docs/upgrade/...`.

## Numerical instability

Symptoms:

- Predictions explode after online updates.
- Batch solve returns poor accuracy.

Fixes:

- Increase `ridgeAlpha`.
- For online streams, set `regularization > 0`.
- For drift, use `forgettingFactor < 1`.
- Compare CPU and GPU results on small matrices before scaling.
