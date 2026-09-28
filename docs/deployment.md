# Deployment

Deployment is Docker-based. The development image (`docker/Dockerfile.dev`) is for building,
testing and benchmarking. The two demo images are for serving.

| Image | Base | Size | Runs on |
|---|---|---:|---|
| `Dockerfile.demo.cpu` | `ubuntu:24.04` | ~120 MB | Any x86-64 host |
| `Dockerfile.demo.gpu` | `nvidia/cuda:12.8.2-base-ubuntu24.04`, cuBLAS/cuSOLVER linked statically | ~950 MB | NVIDIA driver R525+ (any CUDA 12-capable driver), Turing (T4) or newer |
| `Dockerfile.capi` | builds `libfeature_elm_capi.so` (CUDA 12.8, glibc 2.35) | ~2 MB | The ZeroGPU Gradio Space |

Both images run as UID 1000, listen on port **7860**, and bundle the UI, the digits dataset and
the benchmark snapshots. The GPU binary is a fat binary for sm_75, 80, 86, 89, 90 and 120, with
PTX for newer GPUs. It falls back to the CPU backend when no GPU is visible.

## Local

```bash
docker build -f docker/Dockerfile.demo.cpu -t feature-elm-demo-cpu .
docker run --rm -p 7860:7860 feature-elm-demo-cpu

docker build -f docker/Dockerfile.demo.gpu -t feature-elm-demo-gpu .
docker run --rm --gpus all -p 7860:7860 feature-elm-demo-gpu
```

To build the GPU image faster, pass `--build-arg CUDA_ARCHITECTURES=89` (or your GPU's compute
capability).

## Published images

Every `v*` tag publishes both images to GitHub Container Registry, with SBOM and provenance
attestations:

```bash
docker pull ghcr.io/e-choness/feature_extraction_cuda_elm:cpu-latest
docker pull ghcr.io/e-choness/feature_extraction_cuda_elm:gpu-0.2.0
```

Tags are `<flavor>-<major>.<minor>.<patch>`, `<flavor>-<major>.<minor>` and `<flavor>-latest`.

## Environment

| Variable | Default in images | Notes |
|---|---|---|
| `DEMO_PORT` / `PORT` | `7860` | Listen port |
| `DEMO_USE_GPU` | CPU `0`, GPU `1` | `0` forces the CPU backend |
| `DEMO_MAX_HIDDEN` | CPU `1024`, GPU `2048` | Largest hidden layer `/api/evaluate` accepts |
| `DEMO_STATIC_PATH`, `DEMO_BENCHMARK_PATH`, `DEMO_DATASET_PATH` | under `/app` | See [configuration](./configuration.md#demo-configuration) |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | GPU image only |

## Where to host a GPU demo

A demo sits idle most of the time, so scale-to-zero matters more than the hourly price. Findings
as of September 2026:

| Option | GPU | Cost for a demo | When idle | Status in this repo |
|---|---|---|---|---|
| **Hugging Face ZeroGPU** (Gradio Space) | Half an RTX Pro 6000, shared | **Free**: 2 Spaces per free account (10 with PRO). Visitors get a daily GPU quota of a few minutes, and one training call uses well under a second | Nothing to pay | `deploy/huggingface/zerogpu`: **live** at [echoness/cuda-feature-extraction-elm](https://huggingface.co/spaces/echoness/cuda-feature-extraction-elm), deployed automatically |
| **Modal** | T4, L4, A10G, …; billed per second | T4 ≈ $0.59/h; the Starter plan includes $30/month of credit | Scales to zero | Not scripted; runs the GHCR GPU image unchanged |
| **Google Cloud Run** | L4 or RTX Pro 6000 | L4 ≈ $0.67/h, per second; no free GPU tier | Scales to zero; the GPU is ready in about 5 s | Not scripted; runs the GHCR GPU image |
| **Hugging Face Docker Space**, T4 small | T4 16 GB | $0.40/h while awake; creating a Docker Space needs PRO | Sleeps after the idle time you set | `deploy/huggingface/gpu` |
| Self-host (your own GPU) | Your card | Electricity | Always on | `docker run --gpus all …` behind a tunnel or reverse proxy |

Recommendation: start with **ZeroGPU**, the only free GPU route. If you need predictable latency,
move to **Modal**: its monthly credit covers a lightly used demo, and it runs the same image as
everywhere else.

All hosted options above run NVIDIA drivers that support CUDA 12.8 or newer, which is why the GPU
image and the ZeroGPU library target CUDA 12.8 rather than 13.x.

## Hugging Face Spaces

| Space | SDK and hardware | What runs | Cost |
|---|---|---|---|
| `deploy/huggingface/zerogpu` | Gradio on ZeroGPU | Gradio UI in Python; the C++/CUDA library through its C API | Free |
| `deploy/huggingface/cpu` | Docker on CPU basic | The CPU demo image | Free hardware; creating the Space needs PRO |
| `deploy/huggingface/gpu` | Docker on T4 small or better | The GPU demo image | $0.40/h and up |

Free CPU Spaces sleep after 48 hours without visitors and wake on the next visit. You can also
apply for a community GPU grant from the Space settings.

### ZeroGPU (free GPU)

ZeroGPU only supports Gradio Spaces, and it attaches a GPU only while a function decorated with
`@spaces.GPU` runs. The machine has no CUDA toolkit, and its driver supports CUDA 12.8. The Space
therefore works like this:

- `docker/Dockerfile.capi` builds `libfeature_elm_capi.so`, a small C API over the demo (see
  `src/capi/feature_elm_capi.h`). It uses CUDA 12.8 and glibc 2.35, and links cuBLAS/cuSOLVER
  dynamically, so the file is about 2 MB.
- At runtime those libraries come from NVIDIA's pip wheels, pinned as one matched CUDA 12.8 set in
  `requirements.txt`. PyTorch is not needed.
- `app.py` loads the library with `ctypes`. Drawing and CPU training run in the main process,
  which never initialises CUDA. GPU training and the CPU-vs-GPU sweep run inside `@spaces.GPU`
  functions.

It works on ZeroGPU: the live Space reports an *NVIDIA RTX PRO 6000 Blackwell Server Edition MIG
2g.48gb* and trains on it through cuBLAS/cuSOLVER. Native CUDA outside PyTorch isn't officially
documented for ZeroGPU, but this Space shows it works. Hand-drawn digits use a pre-trained model
(`data/models/handwriting_8x8.felm`, see [Demos → Hand-drawn digits](./demos.md#hand-drawn-digits))
that the app loads with `felm_load_classifier`.

**Automatic deployment:** after CI passes on a push to `master`, the *Deploy Hugging Face Space*
workflow builds the library, bundles the dataset and model, uploads them to the Space and waits
until the Space runs the new commit. The library build is reproducible and `hf upload` skips
unchanged files, so pushes that don't touch the demo leave the Space untouched. Set the
`HF_SPACE` repository variable to deploy to a different Space.

To try it locally:

```bash
docker build -f docker/Dockerfile.capi --output type=local,dest=deploy/huggingface/zerogpu/lib .
mkdir -p deploy/huggingface/zerogpu/data && cp data/datasets/digits_8x8.csv data/models/handwriting_8x8.felm deploy/huggingface/zerogpu/data/
docker run --rm --gpus all -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 \
  -v "$PWD/deploy/huggingface/zerogpu:/app" -w /app python:3.12-slim-bookworm \
  bash -c "pip install -q gradio==6.28.0 spaces -r requirements.txt && python app.py"
```

### Setup

1. Create the Space on huggingface.co: **Gradio** with ZeroGPU hardware for `zerogpu`, or
   **Docker** for `cpu`/`gpu`.
2. For `cpu`/`gpu`, make the GHCR package public: **Packages → feature_extraction_cuda_elm →
   Package settings → Change visibility**. Those Spaces only contain a `Dockerfile` that starts
   `FROM` the release image.
3. Add a write token as the `HF_TOKEN` repository secret.
4. Push to `master` (the ZeroGPU flavour deploys automatically once CI passes), or run the **Deploy
   Hugging Face Space** workflow by hand for any flavour. For `zerogpu` it
   builds the library and bundles the dataset first.

## Other container hosts

Any host that runs a Docker image and routes HTTP to one port can serve the demo images: set the
service port to 7860, or set `DEMO_PORT` to the port the host expects. For example, on Cloud Run
with an L4:

```bash
gcloud run deploy feature-elm --image <registry>/feature_extraction_cuda_elm:gpu-0.2.0 \
  --gpu 1 --gpu-type nvidia-l4 --cpu 4 --memory 16Gi --port 7860 --max-instances 1 \
  --no-gpu-zonal-redundancy --region us-central1
```

Cloud Run pulls from Artifact Registry or Docker Hub, so mirror the GHCR image there first. Check
each host's current terms before relying on them, because free tiers change often.

## Production notes

- Hosted CI has no GPUs. The CPU jobs are the correctness gate, and GPU tests skip there.
  Run `docker compose run --rm dev-gpu ctest --output-on-failure` on a GPU machine before releasing.
- Pin Space and deployment images to a version tag rather than `latest`.
- The server handles one training job at a time (`429` while busy). Run more replicas rather than
  raising that limit.
