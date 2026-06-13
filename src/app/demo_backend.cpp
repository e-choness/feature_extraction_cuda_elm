#include "app/demo_backend.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <thread>

#include "app/third_party/httplib.h"
#include "core/elm.hpp"
#include "cuda/elm_gpu.hpp"

namespace fs = std::filesystem;

namespace demo {

namespace {

std::string escapeJsonString(const std::string& value) {
  std::ostringstream oss;
  for (char c : value) {
    switch (c) {
      case '"':
        oss << "\\\"";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\b':
        oss << "\\b";
        break;
      case '\f':
        oss << "\\f";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          oss << "\\u" << std::hex << std::uppercase << (static_cast<int>(c) & 0xFF);
        } else {
          oss << c;
        }
    }
  }
  return oss.str();
}

std::optional<std::vector<float>> parseInputArray(const std::string& body) {
  auto begin = body.find('[');
  auto end = body.rfind(']');
  if (begin == std::string::npos || end == std::string::npos || end <= begin) {
    return std::nullopt;
  }
  std::string inner = body.substr(begin + 1, end - begin - 1);
  std::istringstream stream(inner);
  std::vector<float> values;
  float value;
  while (stream >> value) {
    values.push_back(value);
    if (stream.peek() == ',') {
      stream.ignore();
    }
  }
  return values;
}

std::string inferFromInput(const std::vector<float>& input, bool useGpu, bool gpuAvailable) {
  const std::size_t numInputs = input.size();
  const std::size_t numHiddenNodes = 16;
  const std::size_t numOutputs = 2;
  const std::size_t numSamples = 8;
  std::vector<float> trainData(numSamples * numInputs);
  std::vector<float> trainTargets(numSamples * numOutputs);
  for (std::size_t sample = 0; sample < numSamples; ++sample) {
    for (std::size_t i = 0; i < numInputs; ++i) {
      trainData[sample * numInputs + i] = static_cast<float>(sample + 1) * input[i];
    }
    trainTargets[sample * numOutputs + 0] = static_cast<float>(sample);
    trainTargets[sample * numOutputs + 1] = static_cast<float>(sample) * 0.5f;
  }

  feature_elm::Backend backend = feature_elm::Backend::kCpu;
  if (useGpu && gpuAvailable) {
    backend = feature_elm::Backend::kGpu;
  }

  feature_elm::BatchElm<float> model(numInputs, numHiddenNodes,
                                     feature_elm::ActivationFunction::kSigmoid, backend);
  if (!model.train(trainData, trainTargets, numSamples, numOutputs)) {
    return R"({"status":"error","message":"Model training failed"})";
  }

  std::vector<float> predictions;
  auto result = model.predictBatch(input, 1);
  if (!result.has_value()) {
    return R"({"status":"error","message":"Prediction failed"})";
  }
  predictions = std::move(result.value());
  return makeInferenceResponse(input, predictions, backend == feature_elm::Backend::kGpu);
}

}  // namespace

struct DemoServer::Impl {
  DemoConfig config;
  httplib::Server server;
  bool running = false;
};

DemoServer::DemoServer(const DemoConfig& config) : impl_(new Impl{config, {}, false}) {}

DemoServer::~DemoServer() {
  stop();
  delete impl_;
}

bool DemoServer::start() {
  bool gpuAvailable = feature_elm::cuda_backend::isGpuAvailable();
  impl_->server.set_mount_point("/", impl_->config.staticPath);

  impl_->server.Get("/health", [gpuAvailable](const httplib::Request&) {
    return httplib::Response{
        200, makeHealthResponse(gpuAvailable), {{"Content-Type", "application/json"}}};
  });

  impl_->server.Get("/benchmark-snapshots", [this](const httplib::Request&) {
    std::vector<std::string> names;
    for (auto const& entry : fs::directory_iterator(impl_->config.benchmarkPath)) {
      if (entry.is_regular_file() && entry.path().extension() == ".json") {
        names.push_back(entry.path().filename().string());
      }
    }
    return httplib::Response{
        200, makeBenchmarkListResponse(names), {{"Content-Type", "application/json"}}};
  });

  impl_->server.Post("/run-inference", [this, gpuAvailable](const httplib::Request& req) {
    auto input = parseInputArray(req.body);
    if (!input.has_value() || input->empty()) {
      return httplib::Response{400,
                               R"({"status":"error","message":"Invalid input body"})",
                               {{"Content-Type", "application/json"}}};
    }
    std::string body = inferFromInput(*input, impl_->config.useGpu, gpuAvailable);
    return httplib::Response{200, body, {{"Content-Type", "application/json"}}};
  });

  impl_->server.Post("/run-benchmark", [this, gpuAvailable](const httplib::Request&) {
    if (!impl_->config.useGpu || !gpuAvailable) {
      std::ostringstream oss;
      oss << "{\"status\":\"ok\",\"message\":\"Benchmark skipped (GPU not available or not "
             "enabled)\",\"gpu_enabled\":false}";
      return httplib::Response{200, oss.str(), {{"Content-Type", "application/json"}}};
    }
    std::string benchmarkPath = fs::path(impl_->config.benchmarkPath) / "bench_elm_cuda.json";
    std::string content = loadBenchmarkSnapshot(benchmarkPath);
    if (content.empty()) {
      return httplib::Response{
          200,
          R"({"status":"ok","message":"No benchmark data available","data":null})",
          {{"Content-Type", "application/json"}}};
    }
    std::ostringstream oss;
    oss << "{\"status\":\"ok\",\"gpu_enabled\":true,\"data\":" << content << "}";
    return httplib::Response{200, oss.str(), {{"Content-Type", "application/json"}}};
  });

  impl_->running = impl_->server.listen("0.0.0.0", impl_->config.port);
  return impl_->running;
}

void DemoServer::stop() {
  if (impl_->running) {
    impl_->server.stop();
    impl_->running = false;
  }
}

bool DemoServer::isRunning() const noexcept {
  return impl_->running;
}

int DemoServer::port() const noexcept {
  return impl_->config.port;
}

std::string makeHealthResponse(bool gpuAvailable) noexcept {
  std::ostringstream oss;
  oss << "{\"status\":\"ok\",\"gpu_available\":" << (gpuAvailable ? "true" : "false") << "}";
  return oss.str();
}

std::string makeInferenceResponse(const std::vector<float>& input, const std::vector<float>& output,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  bool usedGpu) noexcept {
  std::ostringstream oss;
  oss << "{\"status\":\"ok\",\"used_gpu\":" << (usedGpu ? "true" : "false") << ",\"input\": [";
  for (std::size_t i = 0; i < input.size(); ++i) {
    if (i)
      oss << ", ";
    oss << input[i];
  }
  oss << "],\"output\": [";
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (i)
      oss << ", ";
    oss << output[i];
  }
  oss << "]}";
  return oss.str();
}

std::string makeBenchmarkListResponse(const std::vector<std::string>& filenames) noexcept {
  std::ostringstream oss;
  oss << "{\"snapshots\": [";
  for (std::size_t i = 0; i < filenames.size(); ++i) {
    if (i)
      oss << ", ";
    oss << "{\"name\": \"" << escapeJsonString(filenames[i]) << "\"}";
  }
  oss << "]}";
  return oss.str();
}

std::string loadBenchmarkSnapshot(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return "";
  }
  std::ostringstream oss;
  oss << file.rdbuf();
  return oss.str();
}

}  // namespace demo
