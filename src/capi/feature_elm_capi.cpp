#include "capi/feature_elm_capi.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

#include "app/demo_backend.hpp"
#include "cuda/gpu_ops.hpp"

namespace {

constexpr std::size_t kMaxHidden = 2048;
constexpr std::size_t kSweepMaxHidden = 1024;

std::unique_ptr<demo::DigitsLab> gLab;
std::mutex gMutex;  // the lab and the CUDA handles are shared; serialise calls

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): C-style buffer + status, by design.
int writeOut(const std::string& text, char* out, std::size_t capacity, int status = 0) {
  if (out == nullptr || capacity == 0) {
    return -1;
  }
  const std::size_t n = std::min(text.size(), capacity - 1);
  std::memcpy(out, text.data(), n);
  out[n] = '\0';
  if (status != 0) {
    return status;
  }
  return n < text.size() ? 1 : 0;
}

int writeError(const std::string& message, char* out, std::size_t capacity) {
  std::string escaped;
  for (char c : message) {
    if (c == '"' || c == '\\') {
      escaped.push_back('\\');
    }
    escaped.push_back(c == '\n' ? ' ' : c);
  }
  return writeOut(R"({"status":"error","message":")" + escaped + "\"}", out, capacity, -1);
}

template <typename Fn>
int guarded(char* out, std::size_t capacity, Fn&& fn) {
  try {
    const std::lock_guard lock(gMutex);
    return fn();
  } catch (const std::exception& e) {
    return writeError(e.what(), out, capacity);
  } catch (...) {
    return writeError("unknown error", out, capacity);
  }
}

}  // namespace

extern "C" {

int felm_init(const char* dataset_csv_path, char* out, size_t capacity) {
  return guarded(out, capacity, [&] {
    if (dataset_csv_path == nullptr) {
      return writeError("dataset path is null", out, capacity);
    }
    std::string error;
    auto lab = demo::DigitsLab::load(dataset_csv_path, &error, /*useGpu=*/false);
    if (lab == nullptr) {
      return writeError(error, out, capacity);
    }
    gLab = std::move(lab);
    return writeOut(R"({"status":"ok"})", out, capacity);
  });
}

int felm_health(char* out, size_t capacity) {
  return guarded(out, capacity, [&] {
    const bool gpu = feature_elm::cuda_backend::isGpuAvailable();
    return writeOut(
        demo::makeHealthResponse(gpu, gpu, feature_elm::cuda_backend::gpuDeviceName(), kMaxHidden),
        out, capacity);
  });
}

int felm_evaluate(const char* request_json, char* out, size_t capacity) {
  return guarded(out, capacity, [&] {
    if (gLab == nullptr) {
      return writeError("felm_init has not been called", out, capacity);
    }
    std::string error;
    const auto request = demo::parseEvaluationRequest(request_json != nullptr ? request_json : "",
                                                      kMaxHidden, &error);
    if (!request.has_value()) {
      return writeError(error, out, capacity);
    }
    const auto result = gLab->evaluate(*request);
    return writeOut(demo::evaluationToJson(result), out, capacity, result.ok ? 0 : -1);
  });
}

int felm_classify(const char* request_json, char* out, size_t capacity) {
  return guarded(out, capacity, [&] {
    if (gLab == nullptr) {
      return writeError("felm_init has not been called", out, capacity);
    }
    int status = 200;
    const std::string body =
        demo::classifyJson(*gLab, request_json != nullptr ? request_json : "", &status);
    return writeOut(body, out, capacity, status == 200 ? 0 : -1);
  });
}

int felm_benchmark(int use_gpu, char* out, size_t capacity) {
  return guarded(out, capacity, [&] {
    if (gLab == nullptr) {
      return writeError("felm_init has not been called", out, capacity);
    }
    const bool gpu = use_gpu != 0 && feature_elm::cuda_backend::isGpuAvailable();
    const std::string device = gpu ? feature_elm::cuda_backend::gpuDeviceName() : std::string();
    return writeOut(demo::runBenchmarkSweep(*gLab, gpu, kSweepMaxHidden, device), out, capacity);
  });
}

}  // extern "C"
