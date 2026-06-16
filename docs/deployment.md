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

For public repositories, the CPU demo can be hosted on free container tiers that support Docker images.

### Hugging Face Spaces (Docker)

1. Create a new Space with Docker runtime.
2. Set the Dockerfile path to `docker/Dockerfile.demo.cpu`.
3. The HTTP port `8888` is automatically mapped.
4. Benchmark snapshots are pre-bundled in the image.
5. Note: Spaces GPU support is available but requires a paid subscription.

### Render

1. Create a new Web Service.
2. Select Docker as the runtime.
3. Set the image source to `ghcr.io/<owner>/feature-elm-demo-cpu:latest` or use the Dockerfile.
4. Map port `8888` in the service configuration.
5. Render's free tier supports one web service with 750 hours/month.

### Fly.io

1. Install `flyctl` and run `fly launch`.
2. Select the CPU demo image: `ghcr.io/<owner>/feature-elm-demo-cpu:latest`.
3. The app listens on port `8888`.
4. Fly's free tier provides 3 shared-cpu apps with 160GB hours/month combined.

Plan-B registry mirrors are useful when a host cannot pull from GHCR. GitHub Actions may impose egress limits with a 30-day notice period; mirror to Docker Hub if needed.

### Image tags

Use semantic version tags for stable deployments:

```bash
# Pull a specific release
docker pull ghcr.io/<owner>/feature-elm-demo-cpu:v1.0.0
```

## Production notes

- Use CPU tests as the correctness gate on hosted CI because GPU runners are not free-tier standard.
- Build the CUDA image in CI to prove compilation, but do not run GPU tests there.
- Pin demo image tags to semantic versions for reproducible deployments.
