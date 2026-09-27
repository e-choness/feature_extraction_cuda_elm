#include "app/demo_backend.hpp"

#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <system_error>
#include <utility>

#include "core/elm.hpp"
#include "core/ml_elm.hpp"
#include "core/os_elm.hpp"
#include "core/version.hpp"
#include "cuda/gpu_ops.hpp"
#include "io/dataset.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace demo {

namespace {

constexpr std::size_t kDigitsInputDim = 64;
constexpr std::size_t kNumClasses = 10;
constexpr std::size_t kPayloadLimitBytes = std::size_t{64} * 1024;
constexpr std::size_t kClassifierHidden = 1024;
// Hand-drawn input scale: 64 inputs at up to 16/16 would saturate most hidden units, so pixels are
// divided by 64 (chosen by evaluating candidates on the test split and on brush-drawn digits).
constexpr double kClassifierInputScale = 64.0;
constexpr std::size_t kOsElmChunk = 128;

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string envOr(const char* name, std::string fallback) {
  const char* value = std::getenv(name);
  return (value != nullptr && *value != '\0') ? std::string(value) : std::move(fallback);
}

std::optional<feature_elm::ActivationFunction> parseActivation(std::string_view name) {
  if (name == "sigmoid") {
    return feature_elm::ActivationFunction::kSigmoid;
  }
  if (name == "tanh") {
    return feature_elm::ActivationFunction::kTanh;
  }
  if (name == "relu") {
    return feature_elm::ActivationFunction::kRelu;
  }
  return std::nullopt;
}

const char* modelName(ModelKind kind) {
  switch (kind) {
    case ModelKind::kBatchElm:
      return "elm";
    case ModelKind::kOsElm:
      return "os-elm";
    case ModelKind::kMlElm:
      return "ml-elm";
  }
  return "elm";
}

template <typename FloatT>
std::vector<FloatT> convert(const std::vector<double>& values) {
  return std::vector<FloatT>(values.begin(), values.end());
}

// BatchElm and OsElm draw their hidden layer from std::random_device when no weights are given;
// the demo passes seeded weights so that runs are reproducible and CPU/GPU runs are comparable.
template <typename FloatT>
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::pair<std::vector<FloatT>, std::vector<FloatT>> seededHiddenLayer(std::size_t inputs,
                                                                      std::size_t hidden,
                                                                      unsigned int seed) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  std::mt19937 gen(seed);
  std::uniform_real_distribution<FloatT> dist(FloatT(-1), FloatT(1));
  std::vector<FloatT> weights(inputs * hidden);
  std::vector<FloatT> biases(hidden);
  for (auto& w : weights) {
    w = dist(gen);
  }
  for (auto& b : biases) {
    b = dist(gen);
  }
  return {std::move(weights), std::move(biases)};
}

template <typename FloatT>
int argmax(const FloatT* scores, std::size_t count) {
  return static_cast<int>(std::max_element(scores, scores + count) - scores);
}

// Moves an 8x8 image by (dx, dy) pixels; pixels shifted in from outside are blank.
std::vector<double> shiftDigit(const std::vector<double>& image, int dx, int dy) {
  constexpr int kSide = 8;
  std::vector<double> out(kDigitsInputDim, 0.0);
  for (int row = 0; row < kSide; ++row) {
    for (int col = 0; col < kSide; ++col) {
      const int srcRow = row - dy;
      const int srcCol = col - dx;
      if (srcRow >= 0 && srcRow < kSide && srcCol >= 0 && srcCol < kSide) {
        const auto dst = static_cast<std::size_t>(row) * kSide + static_cast<std::size_t>(col);
        const auto src =
            static_cast<std::size_t>(srcRow) * kSide + static_cast<std::size_t>(srcCol);
        out[dst] = image[src];
      }
    }
  }
  return out;
}

// Shifts the image so its intensity-weighted centre lands on the grid centre (3.5, 3.5), limited
// to two pixels so that clipped strokes are not pushed off the grid.
std::vector<double> centreDigit(const std::vector<double>& image) {
  double total = 0.0;
  double rowSum = 0.0;
  double colSum = 0.0;
  for (std::size_t i = 0; i < kDigitsInputDim; ++i) {
    const double v = image[i];
    total += v;
    const std::size_t row = i / 8;
    const std::size_t col = i % 8;
    rowSum += v * static_cast<double>(row);
    colSum += v * static_cast<double>(col);
  }
  if (total <= 0.0) {
    return image;
  }
  const auto offset = [](double centre) {
    return std::clamp(static_cast<int>(std::lround(3.5 - centre)), -2, 2);
  };
  return shiftDigit(image, offset(colSum / total), offset(rowSum / total));
}

template <typename FloatT>
double scoreAccuracy(const std::vector<FloatT>& outputs, const std::vector<int>& labels,
                     std::array<std::array<int, 10>, 10>* confusion) {
  std::size_t correct = 0;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int predicted = argmax(outputs.data() + i * kNumClasses, kNumClasses);
    const int actual = labels[i];
    correct += predicted == actual ? 1u : 0u;
    if (confusion != nullptr && actual >= 0 && actual < 10) {
      (*confusion)[static_cast<std::size_t>(actual)][static_cast<std::size_t>(predicted)] += 1;
    }
  }
  return labels.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(labels.size());
}

// Trains `request.model` and returns train/test predictions through the common predictBatch API.
template <typename FloatT>
EvaluationResult runEvaluation(const feature_elm::PreprocessedData& data,
                               const EvaluationRequest& request) {
  EvaluationResult result;
  result.model = modelName(request.model);
  result.precision = std::is_same_v<FloatT, float> ? "float32" : "float64";
  result.hiddenNodes = request.hiddenNodes;
  result.trainSamples = data.numTrainSamples;
  result.testSamples = data.numTestSamples;

  const auto activation = parseActivation(request.activation);
  if (!activation.has_value()) {
    result.error = "unknown activation";
    return result;
  }
  const bool gpu = request.useGpu && feature_elm::cuda_backend::isGpuAvailable();
  const auto backend = gpu ? feature_elm::Backend::kGpu : feature_elm::Backend::kCpu;
  result.backend = gpu ? "gpu" : "cpu";

  const auto trainX = convert<FloatT>(data.trainData);
  const auto trainT = convert<FloatT>(data.trainOneHot);
  const auto testX = convert<FloatT>(data.testData);
  const auto alpha = static_cast<FloatT>(request.ridgeAlpha);
  const std::size_t n = data.numTrainSamples;
  const std::size_t dim = data.inputDim;
  const std::size_t hidden = request.hiddenNodes;

  std::optional<std::vector<FloatT>> trainOut;
  std::optional<std::vector<FloatT>> testOut;

  auto trainAndPredict = [&](auto& model, auto&& fit) {
    const auto start = Clock::now();
    if (!fit(model)) {
      return false;
    }
    result.trainMs = elapsedMs(start);
    const auto predictStart = Clock::now();
    testOut = model.predictBatch(testX, data.numTestSamples);
    result.predictMs = elapsedMs(predictStart);
    trainOut = model.predictBatch(trainX, n);
    return testOut.has_value() && trainOut.has_value();
  };

  bool ok = false;
  switch (request.model) {
    case ModelKind::kBatchElm: {
      auto [w, b] = seededHiddenLayer<FloatT>(dim, hidden, request.seed);
      feature_elm::BatchElm<FloatT> model(dim, hidden, *activation, backend, w, b, alpha);
      ok = trainAndPredict(model, [&](auto& m) { return m.train(trainX, trainT, n, kNumClasses); });
      break;
    }
    case ModelKind::kOsElm: {
      auto [w, b] = seededHiddenLayer<FloatT>(dim, hidden, request.seed);
      feature_elm::RlsOptions<FloatT> rls;
      rls.regularization = alpha;
      feature_elm::OsElm<FloatT> model(dim, hidden, *activation, backend, w, b, rls);
      ok = trainAndPredict(model, [&](auto& m) {
        // Initial block, then fixed-size chunks: the same data a batch ELM sees, streamed.
        const std::size_t initial = std::min(n, std::max<std::size_t>(200, hidden));
        std::vector<FloatT> x(trainX.begin(), trainX.begin() + initial * dim);
        std::vector<FloatT> t(trainT.begin(), trainT.begin() + initial * kNumClasses);
        if (!m.initialize(x, t, initial, kNumClasses)) {
          return false;
        }
        for (std::size_t offset = initial; offset < n; offset += kOsElmChunk) {
          const std::size_t count = std::min(kOsElmChunk, n - offset);
          x.assign(trainX.begin() + offset * dim, trainX.begin() + (offset + count) * dim);
          t.assign(trainT.begin() + offset * kNumClasses,
                   trainT.begin() + (offset + count) * kNumClasses);
          if (!m.update(x, t, count)) {
            return false;
          }
        }
        return true;
      });
      break;
    }
    case ModelKind::kMlElm: {
      // Two auto-encoder layers that compress, then the ridge read-out.
      const std::vector<std::size_t> layers = {hidden, std::max<std::size_t>(hidden / 2, 16)};
      feature_elm::MlElm<FloatT> model(dim, layers, *activation, backend, alpha, request.seed);
      ok = trainAndPredict(model, [&](auto& m) { return m.train(trainX, trainT, n, kNumClasses); });
      break;
    }
  }

  if (!ok) {
    result.error = "training or prediction failed";
    return result;
  }
  result.trainAccuracy = scoreAccuracy(*trainOut, data.trainLabels, nullptr);
  result.testAccuracy = scoreAccuracy(*testOut, data.testLabels, &result.confusion);
  result.ok = true;
  return result;
}

json errorJson(std::string_view message) {
  return json{{"status", "error"}, {"message", message}};
}

void sendJson(httplib::Response& res, const json& body, int status = 200) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

}  // namespace

bool parseBool(std::string_view value) noexcept {
  std::string lowered;
  lowered.reserve(value.size());
  for (char c : value) {
    lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

DemoConfig configFromEnvironment() {
  DemoConfig config;
  config.useGpu = parseBool(envOr("DEMO_USE_GPU", "0"));
  config.host = envOr("DEMO_HOST", config.host);
  config.staticPath = envOr("DEMO_STATIC_PATH", config.staticPath);
  config.benchmarkPath = envOr("DEMO_BENCHMARK_PATH", config.benchmarkPath);
  config.datasetPath = envOr("DEMO_DATASET_PATH", config.datasetPath);
  const std::string port = envOr("DEMO_PORT", envOr("PORT", ""));
  if (!port.empty()) {
    const int parsed = std::atoi(port.c_str());
    if (parsed > 0 && parsed < 65536) {
      config.port = parsed;
    }
  }
  const std::string maxHidden = envOr("DEMO_MAX_HIDDEN", "");
  if (!maxHidden.empty()) {
    const long parsed = std::atol(maxHidden.c_str());
    if (parsed >= 16) {
      config.maxHiddenNodes = static_cast<std::size_t>(parsed);
    }
  }
  return config;
}

// ---------------------------------------------------------------------------------------------
// DigitsLab

struct DigitsLab::Classifier {
  explicit Classifier(feature_elm::BatchElm<double> m) : model(std::move(m)) {}
  feature_elm::BatchElm<double> model;
};

DigitsLab::DigitsLab() = default;
DigitsLab::~DigitsLab() = default;

std::unique_ptr<DigitsLab> DigitsLab::load(const std::string& csvPath, std::string* error,
                                           bool useGpu) {
  auto loaded = feature_elm::loadCsv(csvPath, kDigitsInputDim, 0, true);
  if (!loaded.dataset.has_value()) {
    if (error != nullptr) {
      *error = "failed to load dataset '" + csvPath + "': " + loaded.error;
    }
    return nullptr;
  }
  const auto& ds = *loaded.dataset;
  std::unique_ptr<DigitsLab> lab(new DigitsLab());
  lab->data_ = feature_elm::preprocessDataset(ds.data, ds.labels, ds.numSamples, ds.inputDim);
  if (lab->data_.numClasses != kNumClasses) {
    if (error != nullptr) {
      *error = "expected 10 digit classes";
    }
    return nullptr;
  }

  // The interactive classifier sees hand-drawn input, which is off-centre and inked where the
  // scanned digits rarely are. Per-pixel min/max scaling would blow such pixels far outside the
  // training range, so this model uses global scaling on centre-of-mass-centred images and
  // is trained on +-1 pixel shifts (including diagonals) of every training image.
  const auto& d = lab->data_;
  std::vector<double> raw(d.trainData.size());
  for (std::size_t i = 0; i < d.numTrainSamples; ++i) {
    for (std::size_t p = 0; p < kDigitsInputDim; ++p) {
      const double range = d.maxValues[p] - d.minValues[p];
      const double value = d.trainData[i * kDigitsInputDim + p];
      raw[i * kDigitsInputDim + p] =
          range > 1e-10 ? value * range + d.minValues[p] : d.minValues[p];
    }
  }
  constexpr std::array<std::array<int, 2>, 9> kShifts = {
      {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}};
  std::vector<double> augmented;
  std::vector<double> augmentedTargets;
  augmented.reserve(raw.size() * kShifts.size());
  augmentedTargets.reserve(d.trainOneHot.size() * kShifts.size());
  for (std::size_t i = 0; i < d.numTrainSamples; ++i) {
    const std::vector<double> image(
        raw.begin() + static_cast<std::ptrdiff_t>(i * kDigitsInputDim),
        raw.begin() + static_cast<std::ptrdiff_t>((i + 1) * kDigitsInputDim));
    const auto centred = centreDigit(image);
    for (const auto& [dx, dy] : kShifts) {
      const auto shifted = shiftDigit(centred, dx, dy);
      for (double v : shifted) {
        augmented.push_back(v / kClassifierInputScale);
      }
      augmentedTargets.insert(
          augmentedTargets.end(),
          d.trainOneHot.begin() + static_cast<std::ptrdiff_t>(i * kNumClasses),
          d.trainOneHot.begin() + static_cast<std::ptrdiff_t>((i + 1) * kNumClasses));
    }
  }

  const auto backend = useGpu && feature_elm::cuda_backend::isGpuAvailable()
                           ? feature_elm::Backend::kGpu
                           : feature_elm::Backend::kCpu;
  auto [w, b] = seededHiddenLayer<double>(kDigitsInputDim, kClassifierHidden, 7u);
  feature_elm::BatchElm<double> model(kDigitsInputDim, kClassifierHidden,
                                      feature_elm::ActivationFunction::kRelu, backend, w, b, 1e-1);
  if (!model.train(augmented, augmentedTargets, d.numTrainSamples * kShifts.size(), kNumClasses)) {
    if (error != nullptr) {
      *error = "failed to train the interactive classifier";
    }
    return nullptr;
  }
  lab->classifier_ = std::make_unique<Classifier>(std::move(model));
  return lab;
}

EvaluationResult DigitsLab::evaluate(const EvaluationRequest& request) const {
  return request.precision == Precision::kFloat32 ? runEvaluation<float>(data_, request)
                                                  : runEvaluation<double>(data_, request);
}

std::optional<Classification> DigitsLab::classify(const std::vector<double>& pixels) const {
  if (pixels.size() != kDigitsInputDim) {
    return std::nullopt;
  }
  for (double p : pixels) {
    if (!std::isfinite(p) || p < 0.0 || p > 16.0) {
      return std::nullopt;
    }
  }
  std::vector<double> scaled = centreDigit(pixels);
  for (double& v : scaled) {
    v /= kClassifierInputScale;
  }
  auto scores = classifier_->model.predict(scaled);
  if (!scores.has_value()) {
    return std::nullopt;
  }
  Classification out;
  out.digit = argmax(scores->data(), scores->size());
  out.scores = std::move(*scores);
  return out;
}

// ---------------------------------------------------------------------------------------------
// JSON helpers

std::optional<EvaluationRequest> parseEvaluationRequest(const std::string& body,
                                                        std::size_t maxHidden, std::string* error) {
  auto fail = [&](std::string message) -> std::optional<EvaluationRequest> {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return std::nullopt;
  };
  const json parsed = json::parse(body.empty() ? std::string("{}") : body, nullptr, false);
  if (parsed.is_discarded() || !parsed.is_object()) {
    return fail("body must be a JSON object");
  }
  EvaluationRequest request;
  try {
    const std::string model = parsed.value("model", std::string("elm"));
    if (model == "elm") {
      request.model = ModelKind::kBatchElm;
    } else if (model == "os-elm") {
      request.model = ModelKind::kOsElm;
    } else if (model == "ml-elm") {
      request.model = ModelKind::kMlElm;
    } else {
      return fail("model must be one of: elm, os-elm, ml-elm");
    }
    const auto hidden = parsed.value("hidden", 256LL);
    if (hidden < 16 || static_cast<std::size_t>(hidden) > maxHidden) {
      return fail("hidden must be between 16 and " + std::to_string(maxHidden));
    }
    request.hiddenNodes = static_cast<std::size_t>(hidden);
    request.activation = parsed.value("activation", std::string("sigmoid"));
    if (!parseActivation(request.activation).has_value()) {
      return fail("activation must be one of: sigmoid, tanh, relu");
    }
    const std::string backend = parsed.value("backend", std::string("cpu"));
    if (backend != "cpu" && backend != "gpu") {
      return fail("backend must be cpu or gpu");
    }
    request.useGpu = backend == "gpu";
    const std::string precision = parsed.value("precision", std::string("float32"));
    if (precision != "float32" && precision != "float64") {
      return fail("precision must be float32 or float64");
    }
    request.precision = precision == "float32" ? Precision::kFloat32 : Precision::kFloat64;
    request.ridgeAlpha = parsed.value("ridge", 1e-2);
    if (!(request.ridgeAlpha >= 1e-8 && request.ridgeAlpha <= 1e3)) {
      return fail("ridge must be between 1e-8 and 1e3");
    }
    request.seed = parsed.value("seed", 42u);
  } catch (const json::exception&) {
    return fail("a field has the wrong type");
  }
  return request;
}

std::string evaluationToJson(const EvaluationResult& result) {
  if (!result.ok) {
    return errorJson(result.error).dump();
  }
  json confusion = json::array();
  for (const auto& row : result.confusion) {
    confusion.push_back(row);
  }
  return json{{"status", "ok"},
              {"model", result.model},
              {"backend", result.backend},
              {"precision", result.precision},
              {"hidden", result.hiddenNodes},
              {"train_samples", result.trainSamples},
              {"test_samples", result.testSamples},
              {"train_ms", result.trainMs},
              {"predict_ms", result.predictMs},
              {"train_accuracy", result.trainAccuracy},
              {"test_accuracy", result.testAccuracy},
              {"confusion", confusion}}
      .dump();
}

std::string makeHealthResponse(bool gpuAvailable, bool gpuEnabled, const std::string& deviceName,
                               std::size_t maxHidden) {
  return json{{"status", "ok"},
              {"version", feature_elm::versionString()},
              {"gpu_available", gpuAvailable},
              {"gpu_enabled", gpuEnabled},
              {"device", deviceName},
              {"max_hidden", maxHidden}}
      .dump();
}

std::string runBenchmarkSweep(const DigitsLab& lab, bool useGpu, std::size_t maxHidden,
                              const std::string& deviceName) {
  json rows = json::array();
  for (std::size_t hidden : {64u, 128u, 256u, 512u, 1024u}) {
    if (hidden > maxHidden) {
      break;
    }
    EvaluationRequest request;
    request.hiddenNodes = hidden;
    json row = {{"hidden", hidden}};
    const auto cpu = lab.evaluate(request);
    row["cpu_train_ms"] = cpu.trainMs;
    row["cpu_accuracy"] = cpu.testAccuracy;
    if (useGpu) {
      request.useGpu = true;
      const auto gpu = lab.evaluate(request);
      row["gpu_train_ms"] = gpu.ok ? json(gpu.trainMs) : json(nullptr);
      row["gpu_accuracy"] = gpu.ok ? json(gpu.testAccuracy) : json(nullptr);
    }
    rows.push_back(row);
  }
  return json{{"status", "ok"}, {"gpu_enabled", useGpu}, {"device", deviceName}, {"rows", rows}}
      .dump();
}

std::string classifyJson(const DigitsLab& lab, const std::string& body, int* status) {
  auto fail = [status](std::string_view message) {
    *status = 400;
    return errorJson(message).dump();
  };
  const json parsed = json::parse(body, nullptr, false);
  if (parsed.is_discarded() || !parsed.contains("pixels") || !parsed["pixels"].is_array()) {
    return fail("expected {\"pixels\": [64 numbers in 0..16]}");
  }
  std::vector<double> pixels;
  for (const auto& value : parsed["pixels"]) {
    if (!value.is_number()) {
      return fail("pixels must be numbers");
    }
    pixels.push_back(value.get<double>());
  }
  const auto result = lab.classify(pixels);
  if (!result.has_value()) {
    return fail("expected 64 pixel values in 0..16");
  }
  *status = 200;
  return json{{"status", "ok"}, {"digit", result->digit}, {"scores", result->scores}}.dump();
}

std::string makeBenchmarkListResponse(const std::vector<std::string>& filenames) {
  json snapshots = json::array();
  for (const auto& name : filenames) {
    snapshots.push_back({{"name", name}});
  }
  return json{{"snapshots", snapshots}}.dump();
}

std::vector<std::string> listBenchmarkSnapshots(const std::string& directory) {
  std::vector<std::string> names;
  std::error_code ec;
  for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
    if (it->is_regular_file(ec) && it->path().extension() == ".json") {
      names.push_back(it->path().filename().string());
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::string loadBenchmarkSnapshot(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return "";
  }
  std::ostringstream oss;
  oss << file.rdbuf();
  return oss.str();
}

// ---------------------------------------------------------------------------------------------
// DemoServer

struct DemoServer::Impl {
  DemoConfig config;
  httplib::Server server;
  std::unique_ptr<DigitsLab> lab;
  std::mutex jobMutex;  // one training job at a time keeps a small instance responsive
  bool gpuAvailable = false;
  std::string deviceName;
  std::atomic<bool> running{false};

  void registerRoutes();
};

DemoServer::DemoServer(DemoConfig config) : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

DemoServer::~DemoServer() {
  stop();
}

bool DemoServer::initialize(std::string* error) {
  impl_->lab = DigitsLab::load(impl_->config.datasetPath, error, impl_->config.useGpu);
  if (impl_->lab == nullptr) {
    return false;
  }
  impl_->gpuAvailable = feature_elm::cuda_backend::isGpuAvailable();
  impl_->deviceName = feature_elm::cuda_backend::gpuDeviceName();
  if (impl_->config.useGpu && impl_->gpuAvailable) {
    // Pay CUDA context and library-handle start-up once here, not inside the first timed run.
    EvaluationRequest warmup;
    warmup.hiddenNodes = 32;
    warmup.useGpu = true;
    (void)impl_->lab->evaluate(warmup);
  }
  impl_->registerRoutes();
  return true;
}

void DemoServer::Impl::registerRoutes() {
  server.set_payload_max_length(kPayloadLimitBytes);
  server.set_read_timeout(10, 0);
  server.set_write_timeout(10, 0);
  server.set_default_headers({{"X-Content-Type-Options", "nosniff"},
                              {"Referrer-Policy", "no-referrer"},
                              {"Content-Security-Policy",
                               "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                               "img-src 'self' data:; frame-ancestors *"}});
  server.set_exception_handler(
      [](const httplib::Request&, httplib::Response& res, const std::exception_ptr&) {
        sendJson(res, errorJson("internal error"), 500);
      });
  server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
    if (req.path.rfind("/api/", 0) == 0 && res.body.empty()) {
      sendJson(res, errorJson("not found"), res.status);
    }
  });

  const bool gpuUsable = config.useGpu && gpuAvailable;

  auto health = [this, gpuUsable](const httplib::Request&, httplib::Response& res) {
    res.set_content(makeHealthResponse(gpuAvailable, gpuUsable, deviceName, config.maxHiddenNodes),
                    "application/json");
  };
  server.Get("/health", health);
  server.Get("/api/health", health);

  server.Get("/api/dataset", [this](const httplib::Request&, httplib::Response& res) {
    const auto& d = lab->data();
    sendJson(res, {{"name", "UCI optical digits 8x8"},
                   {"input_dim", d.inputDim},
                   {"classes", d.numClasses},
                   {"train_samples", d.numTrainSamples},
                   {"test_samples", d.numTestSamples}});
  });

  server.Get("/api/benchmarks", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(makeBenchmarkListResponse(listBenchmarkSnapshots(config.benchmarkPath)),
                    "application/json");
  });

  // Only names that are actually present in the snapshot directory are served, so the path
  // parameter can never escape it.
  server.Get(R"(/api/benchmarks/([A-Za-z0-9_.-]+\.json))", [this](const httplib::Request& req,
                                                                  httplib::Response& res) {
    const std::string name = req.matches[1];
    const auto names = listBenchmarkSnapshots(config.benchmarkPath);
    if (std::find(names.begin(), names.end(), name) == names.end()) {
      sendJson(res, errorJson("unknown snapshot"), 404);
      return;
    }
    res.set_content(loadBenchmarkSnapshot((fs::path(config.benchmarkPath) / name).string()),
                    "application/json");
  });

  server.Post("/api/evaluate",
              [this, gpuUsable](const httplib::Request& req, httplib::Response& res) {
                std::string error;
                auto request = parseEvaluationRequest(req.body, config.maxHiddenNodes, &error);
                if (!request.has_value()) {
                  sendJson(res, errorJson(error), 400);
                  return;
                }
                request->useGpu = request->useGpu && gpuUsable;
                std::unique_lock lock(jobMutex, std::try_to_lock);
                if (!lock.owns_lock()) {
                  sendJson(res, errorJson("another job is running, retry shortly"), 429);
                  return;
                }
                const auto result = lab->evaluate(*request);
                res.status = result.ok ? 200 : 500;
                res.set_content(evaluationToJson(result), "application/json");
              });

  // Hidden-layer sweep of Batch ELM (float32) on CPU and, when enabled, GPU.
  server.Post("/api/benchmark", [this, gpuUsable](const httplib::Request&, httplib::Response& res) {
    std::unique_lock lock(jobMutex, std::try_to_lock);
    if (!lock.owns_lock()) {
      sendJson(res, errorJson("another job is running, retry shortly"), 429);
      return;
    }
    res.set_content(runBenchmarkSweep(*lab, gpuUsable, config.maxHiddenNodes, deviceName),
                    "application/json");
  });

  server.Post("/api/classify", [this](const httplib::Request& req, httplib::Response& res) {
    int status = 200;
    res.set_content(classifyJson(*lab, req.body, &status), "application/json");
    res.status = status;
  });

  if (!server.set_mount_point("/", config.staticPath)) {
    // Keep serving the API even if the UI directory is missing.
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
      res.set_content("Feature ELM demo API is running; UI assets were not found.", "text/plain");
    });
  }
}

bool DemoServer::listen() {
  impl_->running = true;
  const bool ok = impl_->server.listen(impl_->config.host, impl_->config.port);
  impl_->running = false;
  return ok;
}

int DemoServer::bindToAnyPort() {
  return impl_->server.bind_to_any_port(impl_->config.host);
}

bool DemoServer::listenAfterBind() {
  impl_->running = true;
  const bool ok = impl_->server.listen_after_bind();
  impl_->running = false;
  return ok;
}

void DemoServer::stop() {
  if (impl_ != nullptr) {
    impl_->server.stop();
  }
}

bool DemoServer::isRunning() const noexcept {
  return impl_->running;
}

}  // namespace demo
