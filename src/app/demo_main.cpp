#include <chrono>
#include <iostream>
#include <thread>

#include "app/demo_backend.hpp"

int main(int, char*[]) {
  demo::DemoConfig config;
  config.useGpu = (std::getenv("DEMO_USE_GPU") != nullptr);
  config.port = 8888;
  config.staticPath = "demo/ui";
  config.benchmarkPath = "data/benchmarks/latest";

  demo::DemoServer server(config);
  if (!server.start()) {
    std::cerr << "Failed to start demo server on port " << config.port << "\n";
    return 1;
  }

  std::cout << "Demo server running on http://0.0.0.0:" << config.port << "\n";
  while (server.isRunning()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}
