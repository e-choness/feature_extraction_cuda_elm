#ifndef FEATURE_ELM_CAPI_FEATURE_ELM_CAPI_H_
#define FEATURE_ELM_CAPI_FEATURE_ELM_CAPI_H_

// Minimal C ABI over the digits demo, for hosts that load the library dynamically (e.g. Python
// via ctypes in the Hugging Face ZeroGPU Space). Requests and responses are JSON strings with the
// same shapes as the demo's HTTP API (docs/demos.md).
//
// Every function writes a NUL-terminated JSON document into `out` (at most `capacity` bytes) and
// returns 0 on success, 1 if the output was truncated, or -1 on error (the JSON then describes it).
//
// CUDA is only touched by felm_health, felm_evaluate with "backend":"gpu" and felm_benchmark with
// use_gpu != 0. felm_init and felm_classify are CPU-only, so a host can call them in a process that
// must not initialise CUDA (ZeroGPU's main process).

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#  define FELM_API __attribute__((visibility("default")))
#else
#  define FELM_API
#endif

/// Loads the digits CSV and trains the interactive classifier on the CPU. Call once.
FELM_API int felm_init(const char* dataset_csv_path, char* out, size_t capacity);

/// Replaces the interactive classifier with a saved model (BatchElm<float>, 64 inputs, 10 classes),
/// e.g. data/models/handwriting_8x8.felm. felm_classify then takes 8x8 block-count features as-is.
FELM_API int felm_load_classifier(const char* model_path, char* out, size_t capacity);

/// {"status","version","gpu_available","gpu_enabled","device","max_hidden"}.
FELM_API int felm_health(char* out, size_t capacity);

/// Body of POST /api/evaluate.
FELM_API int felm_evaluate(const char* request_json, char* out, size_t capacity);

/// Body of POST /api/classify: {"pixels": [64 values in 0..16]}.
FELM_API int felm_classify(const char* request_json, char* out, size_t capacity);

/// Batch ELM hidden-layer sweep on the CPU and, when use_gpu != 0, the GPU.
FELM_API int felm_benchmark(int use_gpu, char* out, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif  // FEATURE_ELM_CAPI_FEATURE_ELM_CAPI_H_
