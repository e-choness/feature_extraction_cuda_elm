#ifndef FEATURE_ELM_APP_DEMO_BACKEND_HPP_
#define FEATURE_ELM_APP_DEMO_BACKEND_HPP_

#include <string>
#include <vector>

namespace demo {

struct DemoConfig {
  bool useGpu = false;
  std::string staticPath = "demo/ui";
  std::string benchmarkPath = "data/benchmarks/latest";
  int port = 8888;
};

class DemoServer {
 public:
  explicit DemoServer(const DemoConfig& config);
  ~DemoServer();

  bool start();
  void stop();
  bool isRunning() const noexcept;
  int port() const noexcept;

 private:
  struct Impl;
  Impl* impl_;
};

std::string makeHealthResponse(bool gpuAvailable) noexcept;
std::string makeInferenceResponse(const std::vector<float>& input, const std::vector<float>& output,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  bool usedGpu) noexcept;
std::string makeBenchmarkListResponse(const std::vector<std::string>& filenames) noexcept;
std::string loadBenchmarkSnapshot(const std::string& path);

}  // namespace demo

#endif  // FEATURE_ELM_APP_DEMO_BACKEND_HPP_
