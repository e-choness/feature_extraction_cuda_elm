---
title: Feature ELM (ZeroGPU)
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: "Extreme Learning Machines in C++/CUDA on ZeroGPU"
---

# Feature ELM on ZeroGPU

Draw a digit, train Batch ELM / OS-ELM / ML-ELM on the UCI 8×8 digits dataset, and compare CPU and GPU
training time, running the C++/CUDA library from
[Feature Extraction CUDA ELM](https://github.com/e-choness/feature_extraction_cuda_elm) on a ZeroGPU slot.

`lib/libfeature_elm_capi.so` is built by `docker/Dockerfile.capi` in the GitHub repository (CUDA 12.8,
glibc 2.35) and synced here by the *Sync Hugging Face Space* workflow.
