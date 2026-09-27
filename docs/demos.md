# Demos

The demo is a small C++ web server (`demo_app`, built on cpp-httplib) plus a static UI. It trains
Extreme Learning Machines on the bundled UCI 8×8 digits dataset (1,437 training / 360 test images)
and lets you:

- **Draw a digit** on an 8×8 pad and watch a 1,024-node ELM classify it live.
- **Train & evaluate** Batch ELM, OS-ELM or ML-ELM with your choice of hidden size, activation,
  precision and backend, and see test accuracy, train/predict time and a confusion matrix.
- **Compare CPU and GPU** with a hidden-layer sweep plotted on a log scale.
- **Browse benchmark snapshots** from `data/benchmarks/latest`.

## Run it

```bash
# CPU image (~120 MB, runs anywhere)
docker build -f docker/Dockerfile.demo.cpu -t feature-elm-demo-cpu .
docker run --rm -p 7860:7860 feature-elm-demo-cpu

# GPU image (needs an NVIDIA driver with CUDA 13 support and the NVIDIA container toolkit)
docker build -f docker/Dockerfile.demo.gpu -t feature-elm-demo-gpu .
docker run --rm --gpus all -p 7860:7860 feature-elm-demo-gpu
```

Open <http://localhost:7860>. The status pill in the top-right corner shows whether a GPU was
found. Without one, the GPU image falls back to the CPU backend and the GPU option is disabled.

Prebuilt images are published to GHCR on every release as
`ghcr.io/e-choness/feature_extraction_cuda_elm:cpu-latest` and `:gpu-latest`.

## Evaluation results

Measured on an RTX 4080 with the GPU image (`/api/evaluate`, sigmoid, ridge α = 0.01, seed 42).
Times are medians of three runs; the CPU path is the single-threaded reference implementation.

**All models at 512 hidden nodes**

| Model | Precision | Test accuracy (CPU / GPU) | CPU train | GPU train | Speed-up |
|---|---|---:|---:|---:|---:|
| Batch ELM | float32 | 97.8% | 64 ms | 9.6 ms | ~7× |
| Batch ELM | float64 | 98.9% | 85 ms | 22 ms | ~4× |
| ML-ELM (512 → 256) | float32 | 97.5% / 98.3% | 500 ms | 136 ms | ~4× |
| ML-ELM (512 → 256) | float64 | 97.8% | 547 ms | 266 ms | ~2× |
| OS-ELM (chunks of 128) | float32 | 97.8% | 257 ms | 225 ms | ~1.1× |

**Batch ELM (float32) as the hidden layer grows**

| Hidden nodes | Test accuracy | CPU train | GPU train | Faster |
|---:|---:|---:|---:|---|
| 64 | 93.6% | 4 ms | 14–16 ms | CPU |
| 256 | 98.1% | 26 ms | 32–43 ms | CPU |
| 512 | 97.8% | 64 ms | 10–70 ms | GPU, up to ~7× |
| 1,024 | 98.3% | 272 ms | 102–129 ms | GPU, ~2.3× |
| 2,048 | n/a | 2,410 ms | 142 ms | **GPU, ~17×** |

GPU times are ranges because this desktop card also drives the display and drops to idle clocks
between requests; the first kernels after a pause run slowly.

What the numbers say:

- **Accuracy is backend-independent.** CPU and GPU use the same seeded hidden layer and agree to
  within float rounding.
- **The GPU pays off for large hidden layers.** The ridge solve scales with the cube of the
  hidden size. Below ~256 nodes, launch and transfer overhead make the CPU faster; by 2,048 nodes
  the GPU is ~17× faster.
- **Earlier, much larger speed-ups were a CPU artefact.** Before v0.2.0 the CPU solver formed
  `HᵀH` with cache-hostile column strides, which inflated GPU speed-ups at 512 nodes to ~100×.
- **float64 is slow on consumer GPUs**, which run FP64 at a small fraction of their FP32 rate.
  Data-centre GPUs (A100, H100) close most of that gap.
- **OS-ELM does not benefit yet.** Only its hidden-layer transform runs on the GPU. The recursive
  least-squares update stays on the CPU and dominates for small chunks. See the
  [roadmap](./roadmap.md).

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Version, `gpu_available`, `gpu_enabled` and device name (`/health` is an alias) |
| `GET` | `/api/dataset` | Dataset name and split sizes |
| `POST` | `/api/evaluate` | Train and evaluate one configuration |
| `POST` | `/api/benchmark` | Batch ELM sweep over 64–1024 hidden nodes on CPU (and GPU when enabled) |
| `POST` | `/api/classify` | Classify one image: `{"pixels": [64 values in 0..16]}` |
| `GET` | `/api/benchmarks` | List benchmark snapshot files |
| `GET` | `/api/benchmarks/{name}.json` | Fetch one snapshot. Only names present in the directory are served. |

`/api/evaluate` accepts (all optional):

```json
{
  "model": "elm",          // elm | os-elm | ml-elm
  "hidden": 256,           // 16 .. DEMO_MAX_HIDDEN
  "activation": "sigmoid", // sigmoid | tanh | relu
  "backend": "gpu",        // cpu | gpu (gpu falls back to cpu when unavailable)
  "precision": "float32",  // float32 | float64
  "ridge": 0.01,           // 1e-8 .. 1e3
  "seed": 42
}
```

```bash
curl -s localhost:7860/api/health
curl -s -X POST localhost:7860/api/evaluate -d '{"model":"ml-elm","hidden":512,"backend":"gpu"}'
```

The server runs one training job at a time and answers `429` while one is running. It limits
request bodies to 64 KB (`413`), validates every field (`400`), and serves static files with
path-traversal protection and `nosniff`/CSP headers.

## Configuration

See [Demo configuration](./configuration.md#demo-configuration) for the environment variables.
To deploy to Hugging Face Spaces, see [Deployment](./deployment.md#hugging-face-spaces).
