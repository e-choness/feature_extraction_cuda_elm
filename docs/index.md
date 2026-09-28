---
layout: home

hero:
  name: Feature ELM
  text: Extreme Learning Machines on CPU and CUDA
  tagline: Composable feature maps, closed-form ridge and recursive least-squares solvers, and online and hierarchical ELM variants in modern C++20, with cuBLAS/cuSOLVER acceleration.
  image:
    src: /logo.svg
    alt: Feature ELM
  actions:
    - theme: brand
      text: Get started
      link: /getting-started
    - theme: alt
      text: Choose a model
      link: /choosing-a-model
    - theme: alt
      text: Try the demo
      link: https://huggingface.co/spaces/echoness/cuda-feature-extraction-elm
    - theme: alt
      text: GitHub
      link: https://github.com/e-choness/feature_extraction_cuda_elm

features:
  - icon: ⚡
    title: Train in one solve
    details: Random hidden layers and a single regularised least-squares solve. No back-propagation, no epochs.
    link: /elm
  - icon: 🌊
    title: Online and drifting streams
    details: OS-ELM, ReOS-ELM, FOS-ELM and OS-CELM update chunk by chunk with recursive least squares and forgetting factors.
    link: /os_elm
  - icon: 🧱
    title: Hierarchical features
    details: ELM auto-encoder layers stack into ML-ELM and H-OS-ELM for learned multilayer representations.
    link: /ml_elm
  - icon: 🟩
    title: CUDA backend
    details: cuBLAS GEMM hidden-layer transforms, a cuSOLVER Cholesky ridge solve and device-resident online RLS behind a single Backend::kGpu switch.
    link: /architecture
  - icon: 📊
    title: Measured, not claimed
    details: Google Benchmark suites and a digits-classification demo report accuracy and CPU/GPU timings side by side.
    link: /benchmarks
  - icon: 🐳
    title: Docker-first
    details: One dev image with CUDA 13.4, GoogleTest and Google Benchmark. CPU and GPU demo images that are ready for Hugging Face Spaces.
    link: /deployment
---

## Tech stack

```mermaid
mindmap
  root((Feature ELM))
    Core
      C++20
      CMake
      GoogleTest
    Pipeline
      FeatureMap
      Solver
      Backend
    Algorithms
      Batch ELM
      OS-ELM
      ELM-AE
      ML-ELM
      RBF
    GPU
      CUDA 13.4
      cuBLAS
      cuSOLVER
    Docs
      VitePress
      Mermaid
```

## License and citation

- [License](./LICENSE.md): MIT License
- [Citation guide](./CITATION.md): how to cite this project and the underlying algorithms

## References

- Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. 2006. Extreme learning machine: theory and applications.
- Liang, Nan-Ying, Guang-Bin Huang, P. Saratchandran, and N. Sundararajan. 2006. A fast and accurate online sequential learning algorithm for feedforward networks.
- Kasun, L. L. C., Yang Yang, Guang-Bin Huang, and Zhiping Zhou. 2013. Extreme learning machine for multilayer perceptron and autoencoder feature learning.
- Broomhead, David S., and David Lowe. 1988. Multivariable functional interpolation and adaptive networks.
- Moody, John, and Christian J. Darken. 1989. Fast learning in networks of locally-tuned processing units.
