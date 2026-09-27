#include <csignal>
#include <iostream>

#include "app/demo_backend.hpp"

namespace {

demo::DemoServer* gServer = nullptr;

extern "C" void handleSignal(int /*signal*/) {
  if (gServer != nullptr) {
    gServer->stop();
  }
}

}  // namespace

int main() {
  const demo::DemoConfig config = demo::configFromEnvironment();
  demo::DemoServer server(config);

  std::string error;
  if (!server.initialize(&error)) {
    std::cerr << "demo: " << error << "\n";
    return 1;
  }

  gServer = &server;
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  std::cout << "Feature ELM demo listening on http://" << config.host << ":" << config.port
            << " (GPU " << (config.useGpu ? "requested" : "disabled") << ")\n"
            << std::flush;
  if (!server.listen()) {
    std::cerr << "demo: failed to listen on " << config.host << ":" << config.port << "\n";
    return 1;
  }
  return 0;
}
