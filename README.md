# Feature Extraction CUDA ELM

GPU-accelerated Extreme Learning Machine (ELM) feature extraction library with online and hierarchical extensions.

## Quickstart (Docker-only)

```bash
# Build and run tests
docker compose run --rm dev ctest --output-on-failure

# Run benchmarks
docker compose run --rm dev ./scripts/run_benchmarks.sh

# Build GPU demo image
docker build -f docker/Dockerfile.demo.gpu -t feature-elm-demo-gpu .

# Run GPU demo
docker run --rm --gpus all -p 8888:8888 feature-elm-demo-gpu

# Build CPU-only demo image
docker build -f docker/Dockerfile.demo.cpu -t feature-elm-demo-cpu .

# Run CPU demo
docker run --rm -p 8888:8888 feature-elm-demo-cpu
```

## Algorithms

### Batch ELM
Single hidden layer feedforward network with random hidden weights and biases. Output weights computed via least-squares.

### OS-ELM (Online Sequential ELM)
Incremental learning algorithm that updates output weights as new data arrives, chunk by chunk or sample by sample.

### OS-CELM (Constrained OS-ELM)
Class-distance-based constraints on hidden parameters for improved generalization.

### H-OS-ELM (Hierarchical OS-ELM)
Multiple OS-ELM subnetworks acting as feature extractors with a top-level classifier.

### RBF Features
Radial Basis Function feature mapping for ELM variants.

## Repository Structure

```
src/core/       - CPU implementations (ELM, OS-ELM, OS-CELM, H-OS-ELM, RBF)
src/cuda/       - GPU implementations with cuBLAS/cuSOLVER integration
src/app/        - Demo HTTP server
bench/          - Google Benchmark microbenchmarks
tests/          - GoogleTest unit tests
docs/           - Algorithm documentation and diagrams
docker/         - Docker images for dev and demo environments
```

## Demo

The demo provides a web UI at http://localhost:8888 with:
- Health check endpoint showing GPU availability
- Benchmark snapshot listing
- Run inference endpoint for demo predictions
- On-demand benchmarking (GPU demo only)

## License

Public domain.