#include <gtest/gtest.h>
#include <httplib.h>

#include <array>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>

#include "app/demo_backend.hpp"
#include "cuda/gpu_ops.hpp"

namespace {

using json = nlohmann::json;

const std::string kRoot = FEATURE_ELM_SOURCE_DIR;
const std::string kDigits = kRoot + "/data/datasets/digits_8x8.csv";

// ---------------------------------------------------------------------------------------------
// Pure helpers

TEST(DemoApiTest, ParseBoolAcceptsCommonTruthyValues) {
  for (const char* value : {"1", "true", "TRUE", "yes", "on"}) {
    EXPECT_TRUE(demo::parseBool(value)) << value;
  }
  // "0" used to enable the GPU because only the variable's presence was checked.
  for (const char* value : {"0", "false", "no", "off", ""}) {
    EXPECT_FALSE(demo::parseBool(value)) << value;
  }
}

TEST(DemoApiTest, HealthResponseIsValidJson) {
  const json body = json::parse(demo::makeHealthResponse(true, false, "Test GPU"));
  EXPECT_EQ(body["status"], "ok");
  EXPECT_TRUE(body["gpu_available"].get<bool>());
  EXPECT_FALSE(body["gpu_enabled"].get<bool>());
  EXPECT_EQ(body["device"], "Test GPU");
  EXPECT_EQ(body["version"], "0.2.0");
  EXPECT_EQ(body["max_hidden"], 2048);
}

TEST(DemoApiTest, BenchmarkListEscapesNames) {
  const json body = json::parse(demo::makeBenchmarkListResponse({"a\"b.json", "c.json"}));
  ASSERT_EQ(body["snapshots"].size(), 2u);
  EXPECT_EQ(body["snapshots"][0]["name"], "a\"b.json");
}

TEST(DemoApiTest, ListsAndLoadsBenchmarkSnapshots) {
  const auto names = demo::listBenchmarkSnapshots(kRoot + "/data/benchmarks/latest");
  ASSERT_FALSE(names.empty());
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
  const std::string content =
      demo::loadBenchmarkSnapshot(kRoot + "/data/benchmarks/latest/" + names.front());
  EXPECT_NE(content.find("\"benchmarks\""), std::string::npos);
}

TEST(DemoApiTest, MissingSnapshotDirectoryIsEmptyNotFatal) {
  EXPECT_TRUE(demo::listBenchmarkSnapshots(kRoot + "/does/not/exist").empty());
  EXPECT_TRUE(demo::loadBenchmarkSnapshot("nonexistent/path.json").empty());
}

TEST(DemoApiTest, ParsesEvaluationRequestWithDefaults) {
  std::string error;
  const auto request = demo::parseEvaluationRequest("{}", 2048, &error);
  ASSERT_TRUE(request.has_value()) << error;
  EXPECT_EQ(request->model, demo::ModelKind::kBatchElm);
  EXPECT_EQ(request->hiddenNodes, 256u);
  EXPECT_FALSE(request->useGpu);

  const auto custom = demo::parseEvaluationRequest(
      R"({"model":"ml-elm","hidden":512,"activation":"tanh","backend":"gpu","precision":"float64"})",
      2048, &error);
  ASSERT_TRUE(custom.has_value()) << error;
  EXPECT_EQ(custom->model, demo::ModelKind::kMlElm);
  EXPECT_EQ(custom->hiddenNodes, 512u);
  EXPECT_TRUE(custom->useGpu);
  EXPECT_EQ(custom->precision, demo::Precision::kFloat64);
}

TEST(DemoApiTest, RejectsInvalidEvaluationRequests) {
  for (const char* body : {"not json", "[]", R"({"model":"svm"})", R"({"hidden":8})",
                           R"({"hidden":100000})", R"({"hidden":"many"})",
                           R"({"activation":"gelu"})", R"({"backend":"tpu"})", R"({"ridge":-1})"}) {
    std::string error;
    EXPECT_FALSE(demo::parseEvaluationRequest(body, 2048, &error).has_value()) << body;
    EXPECT_FALSE(error.empty()) << body;
  }
}

// ---------------------------------------------------------------------------------------------
// Model quality on the real dataset. The floors are deliberately loose (a well-tuned ELM gets
// ~97% on this split); they exist to catch broken solvers, not to track accuracy.

class DigitsLabTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string error;
    lab_ = demo::DigitsLab::load(kDigits, &error).release();
    ASSERT_NE(lab_, nullptr) << error;
  }
  static void TearDownTestSuite() {
    delete lab_;
    lab_ = nullptr;
  }
  static demo::DigitsLab* lab_;
};

demo::DigitsLab* DigitsLabTest::lab_ = nullptr;

TEST_F(DigitsLabTest, SplitsDatasetDeterministically) {
  EXPECT_EQ(lab_->data().numTrainSamples + lab_->data().numTestSamples, 1797u);
  EXPECT_EQ(lab_->data().numClasses, 10u);
}

struct ModelCase {
  demo::ModelKind model;
  double minAccuracy;
};

class DigitsModelTest : public DigitsLabTest, public ::testing::WithParamInterface<ModelCase> {};

TEST_P(DigitsModelTest, CpuReachesAccuracyFloor) {
  demo::EvaluationRequest request;
  request.model = GetParam().model;
  request.hiddenNodes = 256;
  const auto result = lab_->evaluate(request);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.backend, "cpu");
  EXPECT_GE(result.testAccuracy, GetParam().minAccuracy);

  int total = 0;
  for (const auto& row : result.confusion) {
    for (int count : row) {
      total += count;
    }
  }
  EXPECT_EQ(static_cast<std::size_t>(total), result.testSamples);
}

TEST_P(DigitsModelTest, GpuMatchesCpuAccuracy) {
  if (!feature_elm::cuda_backend::isGpuAvailable()) {
    GTEST_SKIP() << "No CUDA device";
  }
  demo::EvaluationRequest request;
  request.model = GetParam().model;
  request.hiddenNodes = 256;
  const auto cpu = lab_->evaluate(request);
  request.useGpu = true;
  const auto gpu = lab_->evaluate(request);
  ASSERT_TRUE(cpu.ok) << cpu.error;
  ASSERT_TRUE(gpu.ok) << gpu.error;
  EXPECT_EQ(gpu.backend, "gpu");
  // Same seeded hidden layer; float32 rounding may flip a handful of the 360 test predictions.
  EXPECT_NEAR(gpu.testAccuracy, cpu.testAccuracy, 0.02);
}

INSTANTIATE_TEST_SUITE_P(Models, DigitsModelTest,
                         ::testing::Values(ModelCase{demo::ModelKind::kBatchElm, 0.90},
                                           ModelCase{demo::ModelKind::kOsElm, 0.90},
                                           ModelCase{demo::ModelKind::kMlElm, 0.80}),
                         [](const auto& info) {
                           switch (info.param.model) {
                             case demo::ModelKind::kBatchElm:
                               return std::string("BatchElm");
                             case demo::ModelKind::kOsElm:
                               return std::string("OsElm");
                             case demo::ModelKind::kMlElm:
                               return std::string("MlElm");
                           }
                           return std::string("Unknown");
                         });

TEST_F(DigitsLabTest, ClassifiesRawDatasetImages) {
  std::ifstream csv(kDigits);
  std::string line;
  std::getline(csv, line);  // header
  int correct = 0;
  int seen = 0;
  while (seen < 50 && std::getline(csv, line)) {
    std::stringstream row(line);
    std::string cell;
    std::getline(row, cell, ',');
    const int label = std::stoi(cell);
    std::vector<double> pixels;
    while (std::getline(row, cell, ',')) {
      pixels.push_back(std::stod(cell));
    }
    const auto result = lab_->classify(pixels);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->scores.size(), 10u);
    correct += result->digit == label ? 1 : 0;
    ++seen;
  }
  EXPECT_GE(correct, 45);
}

// Replays the UI's soft brush (12 at the centre, 3 on each 4-neighbour) along a stroke path.
std::vector<double> brushStroke(const std::vector<std::array<int, 2>>& path) {
  std::vector<double> pixels(64, 0.0);
  int last = -1;
  for (const auto& [col, row] : path) {
    const int cell = row * 8 + col;
    if (cell == last) {
      continue;
    }
    last = cell;
    auto bump = [&](int r, int c, double amount) {
      if (r >= 0 && r < 8 && c >= 0 && c < 8) {
        double& p = pixels[static_cast<std::size_t>(r * 8 + c)];
        p = std::min(16.0, p + amount);
      }
    };
    bump(row, col, 12);
    bump(row - 1, col, 3);
    bump(row + 1, col, 3);
    bump(row, col - 1, 3);
    bump(row, col + 1, 3);
  }
  return pixels;
}

// Hand-drawn digits are off-centre and thinner or thicker than the scans; the interactive
// classifier used to call this 7 a 2.
TEST_F(DigitsLabTest, ClassifiesBrushDrawnDigits) {
  const std::vector<std::pair<int, std::vector<std::array<int, 2>>>> strokes = {
      {7,
       {{1, 1},
        {2, 1},
        {3, 1},
        {4, 1},
        {5, 1},
        {6, 1},
        {6, 2},
        {5, 3},
        {5, 4},
        {4, 5},
        {4, 6},
        {3, 7}}},
      {1, {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6}, {4, 7}}},
      {0,
       {{3, 0},
        {4, 0},
        {5, 1},
        {6, 2},
        {6, 3},
        {6, 4},
        {6, 5},
        {5, 6},
        {4, 7},
        {3, 7},
        {2, 6},
        {1, 5},
        {1, 4},
        {1, 3},
        {1, 2},
        {2, 1},
        {3, 0}}},
      {4,
       {{1, 0},
        {1, 1},
        {1, 2},
        {1, 3},
        {1, 4},
        {2, 4},
        {3, 4},
        {4, 4},
        {5, 4},
        {6, 4},
        {5, 0},
        {5, 1},
        {5, 2},
        {5, 3},
        {5, 5},
        {5, 6},
        {5, 7}}},
      {3,
       {{2, 1},
        {3, 1},
        {4, 1},
        {5, 1},
        {5, 2},
        {5, 3},
        {4, 3},
        {3, 3},
        {5, 4},
        {5, 5},
        {5, 6},
        {4, 7},
        {3, 7},
        {2, 7}}},
  };
  for (const auto& [digit, path] : strokes) {
    const auto result = lab_->classify(brushStroke(path));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->digit, digit);
  }
}

TEST_F(DigitsLabTest, ClassifyRejectsBadInput) {
  EXPECT_FALSE(lab_->classify(std::vector<double>(63, 0.0)).has_value());
  EXPECT_FALSE(lab_->classify(std::vector<double>(64, 17.0)).has_value());
  EXPECT_FALSE(lab_->classify(std::vector<double>(64, -1.0)).has_value());
}

// ---------------------------------------------------------------------------------------------
// HTTP integration: a real server on an ephemeral loopback port.

class DemoServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    demo::DemoConfig config;
    config.host = "127.0.0.1";
    config.staticPath = kRoot + "/demo/ui";
    config.benchmarkPath = kRoot + "/data/benchmarks/latest";
    config.datasetPath = kDigits;
    config.maxHiddenNodes = 512;
    server_ = std::make_unique<demo::DemoServer>(config);
    std::string error;
    ASSERT_TRUE(server_->initialize(&error)) << error;
    port_ = server_->bindToAnyPort();
    ASSERT_GT(port_, 0);
    thread_ = std::thread([this] { server_->listenAfterBind(); });
    client_ = std::make_unique<httplib::Client>("127.0.0.1", port_);
    client_->set_connection_timeout(5, 0);
    client_->set_read_timeout(60, 0);
  }

  void TearDown() override {
    server_->stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  std::unique_ptr<demo::DemoServer> server_;
  std::unique_ptr<httplib::Client> client_;
  std::thread thread_;
  int port_ = 0;
};

TEST_F(DemoServerTest, ServesHealthAndUi) {
  auto health = client_->Get("/api/health");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
  EXPECT_EQ(json::parse(health->body)["status"], "ok");

  auto index = client_->Get("/");
  ASSERT_TRUE(index);
  EXPECT_EQ(index->status, 200);
  EXPECT_NE(index->body.find("<html"), std::string::npos);
}

TEST_F(DemoServerTest, RejectsPathTraversal) {
  // The previous hand-written server returned /etc/passwd for this request.
  for (const char* path :
       {"//../../../etc/passwd", "/../../etc/passwd", "/%2e%2e/%2e%2e/etc/passwd",
        "/api/benchmarks/..%2F..%2Fetc%2Fpasswd.json"}) {
    auto res = client_->Get(path);
    if (res) {
      EXPECT_NE(res->status, 200) << path;
      EXPECT_EQ(res->body.find("root:"), std::string::npos) << path;
    }
  }
}

TEST_F(DemoServerTest, ServesOnlyListedSnapshots) {
  auto list = client_->Get("/api/benchmarks");
  ASSERT_TRUE(list);
  const auto snapshots = json::parse(list->body)["snapshots"];
  ASSERT_FALSE(snapshots.empty());
  auto first = client_->Get("/api/benchmarks/" + snapshots[0]["name"].get<std::string>());
  ASSERT_TRUE(first);
  EXPECT_EQ(first->status, 200);
  auto missing = client_->Get("/api/benchmarks/missing.json");
  ASSERT_TRUE(missing);
  EXPECT_EQ(missing->status, 404);
}

TEST_F(DemoServerTest, EvaluateReturnsMetrics) {
  auto res = client_->Post("/api/evaluate", R"({"model":"elm","hidden":128})", "application/json");
  ASSERT_TRUE(res);
  ASSERT_EQ(res->status, 200) << res->body;
  const json body = json::parse(res->body);
  EXPECT_GT(body["test_accuracy"].get<double>(), 0.85);
  EXPECT_EQ(body["confusion"].size(), 10u);
  EXPECT_EQ(body["backend"], "cpu");  // DemoConfig::useGpu defaults to false
}

TEST_F(DemoServerTest, EvaluateValidatesInput) {
  auto tooBig = client_->Post("/api/evaluate", R"({"hidden":4096})", "application/json");
  ASSERT_TRUE(tooBig);
  EXPECT_EQ(tooBig->status, 400);
  auto garbage = client_->Post("/api/evaluate", "{", "application/json");
  ASSERT_TRUE(garbage);
  EXPECT_EQ(garbage->status, 400);
}

TEST_F(DemoServerTest, RejectsOversizedPayload) {
  const std::string huge(200 * 1024, 'x');
  auto res = client_->Post("/api/classify", huge, "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 413);
}

TEST_F(DemoServerTest, SurvivesMalformedContentLength) {
  // The previous server called std::stoul on this header and terminated.
  httplib::Headers headers = {{"Content-Length", "abc"}};
  (void)client_->Post("/api/classify", headers, "", "application/json");
  auto health = client_->Get("/api/health");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
}

TEST_F(DemoServerTest, ClassifyEndpoint) {
  json zero = {
      {"pixels", {0, 0,  5, 13, 9,  1, 0,  0, 0,  0,  13, 15, 10, 15, 5, 0,  0,  3, 15, 2, 0,  11,
                  8, 0,  0, 4,  12, 0, 0,  8, 8,  0,  0,  5,  8,  0,  0, 9,  8,  0, 0,  4, 11, 0,
                  1, 12, 7, 0,  0,  2, 14, 5, 10, 12, 0,  0,  0,  0,  6, 13, 10, 0, 0,  0}}};
  auto res = client_->Post("/api/classify", zero.dump(), "application/json");
  ASSERT_TRUE(res);
  ASSERT_EQ(res->status, 200) << res->body;
  EXPECT_EQ(json::parse(res->body)["digit"], 0);

  auto bad = client_->Post("/api/classify", R"({"pixels":[1,2,3]})", "application/json");
  ASSERT_TRUE(bad);
  EXPECT_EQ(bad->status, 400);
}

}  // namespace
