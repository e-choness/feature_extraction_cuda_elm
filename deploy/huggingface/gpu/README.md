---
title: Feature ELM (CUDA)
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: t4-small
pinned: false
license: mit
short_description: "Extreme Learning Machines on CUDA: cuBLAS + cuSOLVER vs CPU"
---

# Feature ELM (CUDA)

Live demo of [Feature Extraction CUDA ELM](https://github.com/e-choness/feature_extraction_cuda_elm):
draw a digit, train Batch ELM / OS-ELM / ML-ELM on the UCI 8×8 digits dataset, and compare CPU and
GPU training time.

This Space only contains a `Dockerfile` that starts from the release image published to GitHub
Container Registry. The source, docs and benchmarks live in the GitHub repository.
