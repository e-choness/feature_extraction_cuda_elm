# Demos

The demo is served by the C++ HTTP backend in `src/app/` and static UI assets in `demo/ui/`.

## CPU demo

```bash
docker build -f docker/Dockerfile.demo.cpu -t feature-elm-demo-cpu .
docker run --rm -p 8888:8888 feature-elm-demo-cpu
```

## GPU demo

```bash
docker build -f docker/Dockerfile.demo.gpu -t feature-elm-demo-gpu .
docker run --rm --gpus all -p 8888:8888 feature-elm-demo-gpu
```

## Current helper API

The current backend exposes these helper endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns JSON with `gpu_available` |
| `/benchmark-snapshots` | GET | Lists JSON files under `data/benchmarks/latest` |
| `/run-inference` | POST | Runs a small inference helper from an input array |
| `/run-benchmark` | POST | Returns benchmark JSON when GPU is enabled and available |

Example:

```bash
curl http://localhost:8888/health
curl http://localhost:8888/benchmark-snapshots
curl -X POST http://localhost:8888/run-inference \
  -H "Content-Type: application/json" \
  -d '{"input": [1.0, 2.0, 3.0, 4.0]}'
```

## v2 demo contract

The v2 demo surface specified by `PROJECT_SPEC.md` is:

| Endpoint | Method | Purpose |
|---|---|---|
| `/datasets` | GET | List bundled datasets |
| `/models` | GET | List model/backend choices |
| `/train` | POST | Train on bundled data and return accuracy/confusion/time |
| `/stream-step` | POST | Advance OS-ELM one chunk and return running accuracy |
| `/drift-run` | POST | Compare FOS-ELM and OS-ELM on a drift stream |
| `/benchmarks` | GET | Return benchmark snapshots |
| `/run-benchmark` | POST | GPU-only on-demand benchmark |

Until U8 is implemented, the current helper endpoints above remain the runtime surface.

## UI panels

The v2 UI should include:

- Batch panel for model/backend selection, accuracy, confusion matrix, and train time.
- Streaming panel for running accuracy as chunks arrive.
- Drift panel overlaying FOS-ELM and OS-ELM after the drift point.
- Benchmark overlay from snapshot JSON.
