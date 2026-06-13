# Demos

Using the GPU and CPU demo servers.

## GPU Demo

```bash
docker build -f docker/Dockerfile.demo.gpu -t feature-elm-demo-gpu .
docker run --rm --gpus all -p 8888:8888 feature-elm-demo-gpu
```

## CPU Demo

```bash
docker build -f docker/Dockerfile.demo.cpu -t feature-elm-demo-cpu .
docker run --rm -p 8888:8888 feature-elm-demo-cpu
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns GPU availability status |
| `/benchmark-snapshots` | GET | Lists available benchmark files |
| `/run-inference` | POST | Runs ELM prediction on input JSON |
| `/run-benchmark` | POST | Returns benchmark data (GPU only) |

## Request Examples

```bash
# Health check
curl http://localhost:8888/health

# Run inference
curl -X POST http://localhost:8888/run-inference \
  -H "Content-Type: application/json" \
  -d '{"input": [1.0, 2.0, 3.0, 4.0]}'

# Run benchmark (GPU only)
curl -X POST http://localhost:8888/run-benchmark
```