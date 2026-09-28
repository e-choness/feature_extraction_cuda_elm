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
Times are medians of three runs. The CPU path uses all 32 threads of an i9-14900K (OpenMP).

**All models at 512 hidden nodes**

| Model | Precision | Test accuracy | CPU train | GPU train | Speed-up |
|---|---|---:|---:|---:|---:|
| Batch ELM | float32 | 97.8% | 11 ms | 3.2 ms | ~3× |
| Batch ELM | float64 | 98.9% | 15 ms | 8.9 ms | ~1.7× |
| ML-ELM (512 → 256) | float32 | 97.5% | 50 ms | 13 ms | ~4× |
| ML-ELM (512 → 256) | float64 | 97.8% | 63 ms | 31 ms | ~2× |
| OS-ELM (chunks of 128) | float32 | 97.8% | 159 ms | 6.1 ms | **~26×** |
| OS-ELM (chunks of 128) | float64 | 98.9% | 197 ms | 7.9 ms | **~25×** |

**Batch ELM (float32) as the hidden layer grows**

| Hidden nodes | Test accuracy | CPU train | GPU train | Speed-up |
|---:|---:|---:|---:|---:|
| 64 | 93.6% | 1.1 ms | 0.7 ms | ~1.5× |
| 256 | 98.1% | 2.8 ms | 1.9 ms | ~1.5× |
| 512 | 97.8% | 8.0 ms | 2.6 ms | ~3× |
| 1,024 | 98.3% | 39 ms | 4.5 ms | ~9× |
| 2,048 | 98.3% | 92 ms | 40 ms | ~2× |

The digits set has only 1,437 training samples, so at 2,048 hidden nodes the solve switches to the
dual (sample-sized) system and both backends do less work. Desktop GPU times vary from run to run,
because the card also drives the display and drops to idle clocks between requests. For full-size
numbers (60,000 MNIST images), see the [benchmarks](./benchmarks.md#full-size-datasets).

What the numbers say:

- **Accuracy is backend-independent.** CPU and GPU use the same seeded hidden layer and agree to
  within float rounding.
- **The comparison is fair.** The CPU is multithreaded, and its results are bit-identical for any
  thread count. On this small dataset the GPU leads Batch ELM by 1.5–9×; on 60,000 MNIST images
  the lead grows to ~16× at 4,096 hidden nodes.
- **OS-ELM gains the most.** Its recursive least-squares state stays on the GPU between chunks (in
  float64, information form), so each chunk costs one GEMM instead of a CPU covariance update.
- **float64 is slow on consumer GPUs**, which run FP64 at a small fraction of their FP32 rate.
  Data-centre GPUs (A100, H100) close most of that gap.

## Hand-drawn digits

The Hugging Face Space classifies free-hand sketches with a model trained on MNIST handwriting
([`data/models/handwriting_8x8.felm`](https://github.com/e-choness/feature_extraction_cuda_elm/blob/master/data/models/README.md)).
The start-up classifier, trained only on the UCI digits, turned out to be poor at real handwriting;
visitors saw 8 read as 0 and curly 6s misread.

To measure this without hand-drawing thousands of digits, MNIST test digits (real handwriting from
hundreds of writers) were fed through the Space's sketch pipeline as if they were drawings:

| Classifier | Accuracy on handwriting | Digit 6 | Digit 8 |
|---|---:|---:|---:|
| UCI digits + shifts (the original Space model) | 46.5% | 27% | 80% |
| Same, UCI-faithful preprocessing | 51.7% | 37% | 69% |
| Same, plus rotation/scale/shear augmentation | 54.3% | 39% | 48% |
| **MNIST + UCI, 4,096 ReLU units (the current model)** | **97.2%** | **97%** | **97%** |

The last row was measured end to end through the Space's `classify()` on 2,000 MNIST test digits
rendered as sketchpad strokes. No preprocessing trick rescues the UCI-only model; with 1,797 digits
from 43 writers, it simply hasn't seen enough handwriting. The preprocessing
(`deploy/huggingface/zerogpu/digitprep.py`) mirrors how the UCI set was made: fit the digit into a
32×32 binary box, then count "on" pixels in 4×4 blocks. Training data and live drawings go through
the same code.

MNIST digits are written with a pen, so mouse drawings will score somewhat lower than 97%. They
should still be far better than before.

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
