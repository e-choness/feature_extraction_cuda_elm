# Deployment

Deployment is Docker-based. The development image is for tests and benchmarks; demo images are for serving.

## Local CPU demo

```bash
docker build -f docker/Dockerfile.demo.cpu -t feature-elm-demo-cpu .
docker run --rm -p 8888:8888 feature-elm-demo-cpu
```

## Local GPU demo

```bash
docker build -f docker/Dockerfile.demo.gpu -t feature-elm-demo-gpu .
docker run --rm --gpus all -p 8888:8888 feature-elm-demo-gpu
```

The GPU image uses the same HTTP surface and enables on-demand benchmark behavior when a CUDA device is visible.

## Ports and environment

| Setting | Default | Notes |
|---|---:|---|
| HTTP port | `8888` | Publish with `-p 8888:8888` |
| `DEMO_USE_GPU` | unset | Enables GPU demo path when set |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Compose dev service default |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Compose dev service default |

## Free-tier CPU hosting guide

For public repositories, the CPU demo can be hosted on free container tiers that support Docker images and environment variables.

1. Build and publish the CPU image to a registry supported by the host.
2. Set the HTTP port expected by the host, or keep the container port at `8888` and map it externally.
3. Do not enable GPU-only benchmark endpoints on CPU-only hosts.
4. Keep benchmark snapshots in the image or mounted volume under `data/benchmarks/latest`.

Plan-B registry mirrors are useful when a host cannot pull from the primary registry.

## Production notes

- Use CPU tests as the correctness gate on hosted CI because GPU runners are not free-tier standard.
- Build the CUDA image in CI to prove compilation, but do not run GPU tests there.
- Pin demo image tags to semantic versions for reproducible deployments.
