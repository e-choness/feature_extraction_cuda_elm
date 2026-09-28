#ifndef FEATURE_ELM_APP_DEMO_BACKEND_HPP_
#define FEATURE_ELM_APP_DEMO_BACKEND_HPP_

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "io/preprocess.hpp"

namespace demo {

/// Runtime configuration. Every field can be set from the environment, see configFromEnvironment.
struct DemoConfig {
  bool useGpu = false;                                       ///< DEMO_USE_GPU
  std::string host = "0.0.0.0";                              ///< DEMO_HOST
  int port = 8888;                                           ///< DEMO_PORT, then PORT
  std::string staticPath = "demo/ui";                        ///< DEMO_STATIC_PATH
  std::string benchmarkPath = "data/benchmarks/latest";      ///< DEMO_BENCHMARK_PATH
  std::string datasetPath = "data/datasets/digits_8x8.csv";  ///< DEMO_DATASET_PATH
  std::size_t maxHiddenNodes = 2048;                         ///< DEMO_MAX_HIDDEN
};

/// Parses "1", "true", "yes", "on" (any case) as true; everything else, including "0", as false.
[[nodiscard]] bool parseBool(std::string_view value) noexcept;

/// Reads DemoConfig from DEMO_* environment variables, falling back to the defaults above.
[[nodiscard]] DemoConfig configFromEnvironment();

enum class ModelKind { kBatchElm, kOsElm, kMlElm };
enum class Precision { kFloat32, kFloat64 };

/// One train-and-evaluate run on the digits dataset.
struct EvaluationRequest {
  ModelKind model = ModelKind::kBatchElm;
  std::size_t hiddenNodes = 256;
  std::string activation = "sigmoid";  ///< sigmoid | tanh | relu
  bool useGpu = false;
  Precision precision = Precision::kFloat32;
  double ridgeAlpha = 1e-2;
  unsigned int seed = 42;
};

struct EvaluationResult {
  bool ok = false;
  std::string error;
  std::string model;
  std::string backend;  ///< "cpu" or "gpu": the backend that actually ran
  std::string precision;
  std::size_t hiddenNodes = 0;
  std::size_t trainSamples = 0;
  std::size_t testSamples = 0;
  double trainMs = 0.0;
  double predictMs = 0.0;
  double trainAccuracy = 0.0;
  double testAccuracy = 0.0;
  std::array<std::array<int, 10>, 10> confusion{};  ///< [true label][predicted label]
};

struct Classification {
  int digit = -1;
  std::vector<double> scores;  ///< raw ELM outputs, one per class
};

/// Owns the digits dataset and a small always-ready classifier for interactive predictions.
class DigitsLab {
 public:
  /// Loads and splits the dataset (80/20, fixed seed). Returns nullptr and sets *error on failure.
  /// With useGpu (and a CUDA device) the interactive classifier is trained and run on the GPU.
  [[nodiscard]] static std::unique_ptr<DigitsLab> load(const std::string& csvPath,
                                                       std::string* error, bool useGpu = false);

  [[nodiscard]] EvaluationResult evaluate(const EvaluationRequest& request) const;

  /// Classifies one 8x8 image given as 64 raw pixel intensities in [0, 16].
  [[nodiscard]] std::optional<Classification> classify(const std::vector<double>& pixels) const;

  /// Replaces the interactive classifier with a saved BatchElm<float> model (64 inputs, 10
  /// classes). Loaded models take 8x8 block-count features as they are. See scripts/
  /// build_handwriting_data.py for how data/models/handwriting_8x8.felm is produced.
  [[nodiscard]] bool loadClassifier(const std::string& path, std::string* error);

  [[nodiscard]] const feature_elm::PreprocessedData& data() const noexcept {
    return data_;
  }

  ~DigitsLab();
  DigitsLab(const DigitsLab&) = delete;
  DigitsLab& operator=(const DigitsLab&) = delete;

 private:
  DigitsLab();
  struct Classifier;
  feature_elm::PreprocessedData data_;
  std::unique_ptr<Classifier> classifier_;
};

/// Parses a JSON evaluation request, clamping nothing: invalid values are reported in *error.
[[nodiscard]] std::optional<EvaluationRequest> parseEvaluationRequest(const std::string& body,
                                                                      std::size_t maxHidden,
                                                                      std::string* error);

[[nodiscard]] std::string evaluationToJson(const EvaluationResult& result);
[[nodiscard]] std::string makeHealthResponse(bool gpuAvailable, bool gpuEnabled,
                                             const std::string& deviceName,
                                             std::size_t maxHidden = 2048);
/// Batch ELM (float32) hidden-layer sweep, 64..1024 nodes capped at maxHidden; GPU column when
/// useGpu.
[[nodiscard]] std::string runBenchmarkSweep(const DigitsLab& lab, bool useGpu,
                                            std::size_t maxHidden, const std::string& deviceName);
/// Handles a {"pixels": [...]} request body; sets *status to 200 or 400.
[[nodiscard]] std::string classifyJson(const DigitsLab& lab, const std::string& body, int* status);
[[nodiscard]] std::string makeBenchmarkListResponse(const std::vector<std::string>& filenames);

/// Lists *.json files directly inside `directory` (sorted); empty if it does not exist.
[[nodiscard]] std::vector<std::string> listBenchmarkSnapshots(const std::string& directory);
[[nodiscard]] std::string loadBenchmarkSnapshot(const std::string& path);

class DemoServer {
 public:
  explicit DemoServer(DemoConfig config);
  ~DemoServer();
  DemoServer(const DemoServer&) = delete;
  DemoServer& operator=(const DemoServer&) = delete;

  /// Loads the dataset and registers routes. Must succeed before listen().
  [[nodiscard]] bool initialize(std::string* error);

  /// Binds and serves until stop() is called. Returns false if the socket could not be bound.
  bool listen();

  /// Binds to an ephemeral port on the configured host and returns it (-1 on failure). For tests.
  int bindToAnyPort();
  /// Serves on a socket bound by bindToAnyPort().
  bool listenAfterBind();

  void stop();
  [[nodiscard]] bool isRunning() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace demo

#endif  // FEATURE_ELM_APP_DEMO_BACKEND_HPP_
