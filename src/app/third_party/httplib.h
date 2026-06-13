// Lightweight single-header HTTP server/client.
// Public domain version of cpp-httplib available at https://github.com/yhirose/cpp-httplib

#ifndef FEATURE_ELM_APP_HTTP_HPP_
#define FEATURE_ELM_APP_HTTP_HPP_

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32")
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace httplib {

struct Request {
  std::string method;
  std::string path;
  std::string body;
  std::map<std::string, std::string> headers;
};

struct Response {
  int status = 200;
  std::string body;
  std::map<std::string, std::string> headers;
};

using Handler = std::function<Response(const Request&)>;

class Server {
 public:
  Server() = default;
  ~Server() {
    stop();
  }

  void set_mount_point(const std::string& mount_path, const std::string& base_dir) {
    mount_path_ = mount_path;
    base_dir_ = base_dir;
  }

  void Get(const std::string& path, Handler handler) {
    handlers_[path + ":GET"] = std::move(handler);
  }

  void Post(const std::string& path, Handler handler) {
    handlers_[path + ":POST"] = std::move(handler);
  }

  bool listen(const std::string& host, int port) {
    if (running_) {
      return false;
    }

    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      return false;
    }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&opt), sizeof(opt));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(host.c_str());
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      close_socket(sock);
      return false;
    }

    if (listen_socket(sock) < 0) {
      close_socket(sock);
      return false;
    }

    running_ = true;
    thread_ = std::thread([this, sock]() {
      while (running_) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(sock, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_sock < 0) {
          continue;
        }
        handleClient(client_sock);
        close_socket(client_sock);
      }
      close_socket(sock);
    });
    return true;
  }

  void stop() {
    if (!running_) {
      return;
    }
    running_ = false;
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  static void close_socket(int sock) {
#if defined(_WIN32)
    closesocket(sock);
#else
    close(sock);
#endif
  }

  static int listen_socket(int sock) {
    return ::listen(sock, SOMAXCONN);
  }

  void handleClient(int client_sock) {
    char buffer[4096];
    int len = recv(client_sock, buffer, sizeof(buffer), 0);
    if (len <= 0) {
      return;
    }
    std::string request_text(buffer, len);
    Request request = parseRequest(request_text);
    Response response;
    std::string handlerKey = request.path + ":" + request.method;
    auto it = handlers_.find(handlerKey);
    if (it != handlers_.end()) {
      response = it->second(request);
    } else if (!mount_path_.empty() && request.method == "GET" &&
               request.path.rfind(mount_path_, 0) == 0) {
      response = serveStaticFile(request.path.substr(mount_path_.size()));
    } else {
      response.status = 404;
      response.body = "Not Found";
    }

    std::ostringstream out;
    out << "HTTP/1.1 " << response.status << " OK\r\n";
    out << "Content-Length: " << response.body.size() << "\r\n";
    for (auto& header : response.headers) {
      out << header.first << ": " << header.second << "\r\n";
    }
    out << "\r\n";
    out << response.body;
    std::string resp_str = out.str();
    send(client_sock, resp_str.c_str(), static_cast<int>(resp_str.size()), 0);
  }

  Request parseRequest(const std::string& raw) {
    Request req;
    std::istringstream stream(raw);
    std::string line;
    if (!std::getline(stream, line)) {
      return req;
    }
    std::istringstream request_line(line);
    request_line >> req.method >> req.path;

    while (std::getline(stream, line) && !line.empty() && line != "\r") {
      auto colon = line.find(':');
      if (colon != std::string::npos) {
        std::string key = line.substr(0, colon);
        std::string value = line.substr(colon + 1);
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
          value.erase(value.begin());
        }
        if (!value.empty() && value.back() == '\r') {
          value.pop_back();
        }
        req.headers[key] = value;
      }
    }

    if (req.headers.count("Content-Length") > 0) {
      size_t content_length = std::stoul(req.headers.at("Content-Length"));
      req.body.resize(content_length);
      stream.read(&req.body[0], content_length);
    }
    return req;
  }

  Response serveStaticFile(const std::string& path) {
    Response res;
    std::string file_path = base_dir_ + path;
    if (file_path.empty()) {
      res.status = 404;
      res.body = "Not Found";
      return res;
    }

    FILE* file = fopen(file_path.c_str(), "rb");
    if (!file) {
      res.status = 404;
      res.body = "Not Found";
      return res;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    std::string body;
    body.resize(size);
    std::ignore = fread(&body[0], 1, size, file);
    fclose(file);
    res.body = std::move(body);
    res.headers["Content-Type"] = mimeType(file_path);
    return res;
  }

  std::string mimeType(const std::string& path) const {
    if (path.ends_with(".html")) {
      return "text/html";
    }
    if (path.ends_with(".js")) {
      return "application/javascript";
    }
    if (path.ends_with(".css")) {
      return "text/css";
    }
    if (path.ends_with(".json")) {
      return "application/json";
    }
    return "text/plain";
  }

  std::map<std::string, Handler> handlers_;
  std::string mount_path_;
  std::string base_dir_;
  bool running_ = false;
  std::thread thread_;
};

class Client {
 public:
  Client(const std::string& host, int port) : host_(host), port_(port) {}

  std::string Get(const std::string& path) {
    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n";
    req << "Host: " << host_ << ":" << port_ << "\r\n";
    req << "Connection: close\r\n";
    req << "\r\n";
    return request(req.str());
  }

  std::string Post(const std::string& path, const std::string& body) {
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n";
    req << "Host: " << host_ << ":" << port_ << "\r\n";
    req << "Content-Type: application/json\r\n";
    req << "Content-Length: " << body.size() << "\r\n";
    req << "Connection: close\r\n";
    req << "\r\n";
    req << body;
    return request(req.str());
  }

 private:
  std::string request(const std::string& req) {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      return {};
    }
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      close_socket(sock);
      return {};
    }
    send(sock, req.c_str(), static_cast<int>(req.size()), 0);
    std::string resp;
    char buffer[4096];
    int len = 0;
    while ((len = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
      resp.append(buffer, len);
    }
    close_socket(sock);
    auto pos = resp.find("\r\n\r\n");
    if (pos == std::string::npos) {
      return resp;
    }
    return resp.substr(pos + 4);
  }

  static void close_socket(int sock) {
#if defined(_WIN32)
    closesocket(sock);
#else
    close(sock);
#endif
  }

  std::string host_;
  int port_;
};

}  // namespace httplib

#endif  // FEATURE_ELM_APP_HTTP_HPP_
