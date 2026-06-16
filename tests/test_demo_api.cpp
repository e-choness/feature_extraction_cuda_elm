#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "app/demo_backend.hpp"
#include "app/third_party/httplib.h"

namespace {

TEST(DemoApiTest, MakeHealthResponseReturnsJson) {
  std::string response = demo::makeHealthResponse(true);
  EXPECT_NE(response.find("\"status\":\"ok\""), std::string::npos);
  EXPECT_NE(response.find("\"gpu_available\":true"), std::string::npos);
}

TEST(DemoApiTest, MakeHealthResponseNoGpu) {
  std::string response = demo::makeHealthResponse(false);
  EXPECT_NE(response.find("\"status\":\"ok\""), std::string::npos);
  EXPECT_NE(response.find("\"gpu_available\":false"), std::string::npos);
}

TEST(DemoApiTest, MakeInferenceResponseReturnsJson) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> output = {0.5f, 0.25f};
  std::string response = demo::makeInferenceResponse(input, output, false);
  EXPECT_NE(response.find("\"status\":\"ok\""), std::string::npos);
  EXPECT_NE(response.find("\"used_gpu\":false"), std::string::npos);
  EXPECT_NE(response.find("\"output\":"), std::string::npos);
}

TEST(DemoApiTest, LoadBenchmarkSnapshotReturnsContent) {
  std::string content =
      demo::loadBenchmarkSnapshot("/workspace/data/benchmarks/latest/bench_feature_maps.json");
  EXPECT_FALSE(content.empty());
  EXPECT_NE(content.find("\"benchmarks\""), std::string::npos);
}

TEST(DemoApiTest, LoadBenchmarkSnapshotMissingFile) {
  std::string content = demo::loadBenchmarkSnapshot("nonexistent/path.json");
  EXPECT_TRUE(content.empty());
}

TEST(DemoApiTest, MakeBenchmarkListResponse) {
  std::vector<std::string> names = {"bench_feature_maps.json", "bench_solvers.json",
                                    "bench_ml_elm.json", "bench_elm_cuda.json"};
  std::string response = demo::makeBenchmarkListResponse(names);
  EXPECT_NE(response.find("\"snapshots\""), std::string::npos);
  EXPECT_NE(response.find("\"bench_feature_maps.json\""), std::string::npos);
}

}  // namespace
