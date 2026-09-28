// felm-train: train a Batch ELM classifier from CSV and save it as a model file.
//
//   felm-train --train train.csv --inputs 64 --classes 10 --out model.felm
//              [--test test.csv] [--hidden 4096] [--activation relu] [--ridge 1]
//              [--input-scale 64] [--seed 7] [--cpu]
//
// CSV rows are "label,feature1,...,featureN" with a header line. Training runs on the GPU when one
// is available (pass --cpu to force the CPU). --input-scale divides the random hidden weights,
// which is equivalent to scaling the inputs down but keeps the saved model usable on raw features.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/elm.hpp"
#include "cuda/gpu_ops.hpp"
#include "io/dataset.hpp"

namespace {

using feature_elm::ActivationFunction;
using feature_elm::Backend;
using feature_elm::BatchElm;

struct Options {
  std::string trainPath;
  std::string testPath;
  std::string outPath;
  std::size_t inputs = 0;
  std::size_t classes = 0;
  std::size_t hidden = 4096;
  std::string activation = "relu";
  float ridge = 1.0f;
  float inputScale = 1.0f;
  unsigned int seed = 7;
  bool forceCpu = false;
};

int usage(const char* message) {
  std::cerr << "felm-train: " << message << "\n"
            << "usage: felm-train --train FILE --inputs N --classes K --out FILE [--test FILE]\n"
            << "                  [--hidden H] [--activation sigmoid|tanh|relu] [--ridge A]\n"
            << "                  [--input-scale S] [--seed N] [--cpu]\n";
  return 2;
}

std::optional<Options> parse(int argc, char** argv) {
  Options o;
  std::map<std::string, std::string*> strings = {{"--train", &o.trainPath},
                                                 {"--test", &o.testPath},
                                                 {"--out", &o.outPath},
                                                 {"--activation", &o.activation}};
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--cpu") {
      o.forceCpu = true;
      continue;
    }
    if (i + 1 >= argc) {
      return std::nullopt;
    }
    const std::string value = argv[++i];
    if (auto it = strings.find(arg); it != strings.end()) {
      *it->second = value;
    } else if (arg == "--inputs") {
      o.inputs = std::stoul(value);
    } else if (arg == "--classes") {
      o.classes = std::stoul(value);
    } else if (arg == "--hidden") {
      o.hidden = std::stoul(value);
    } else if (arg == "--ridge") {
      o.ridge = std::stof(value);
    } else if (arg == "--input-scale") {
      o.inputScale = std::stof(value);
    } else if (arg == "--seed") {
      o.seed = static_cast<unsigned int>(std::stoul(value));
    } else {
      return std::nullopt;
    }
  }
  return o;
}

std::optional<ActivationFunction> activationFrom(const std::string& name) {
  if (name == "sigmoid") {
    return ActivationFunction::kSigmoid;
  }
  if (name == "tanh") {
    return ActivationFunction::kTanh;
  }
  if (name == "relu") {
    return ActivationFunction::kRelu;
  }
  return std::nullopt;
}

std::optional<feature_elm::Dataset> loadSet(const std::string& path, std::size_t inputs) {
  auto result = feature_elm::loadCsv(path, inputs, 0, true);
  if (!result.dataset.has_value()) {
    std::cerr << "felm-train: cannot load " << path << ": " << result.error << "\n";
  }
  return result.dataset;
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<Options> parsed;
  try {
    parsed = parse(argc, argv);
  } catch (const std::exception&) {
    return usage("invalid numeric argument");
  }
  if (!parsed || parsed->trainPath.empty() || parsed->outPath.empty() || parsed->inputs == 0 ||
      parsed->classes < 2 || parsed->hidden == 0 || !(parsed->inputScale > 0.0f)) {
    return usage("missing or invalid arguments");
  }
  const Options& o = *parsed;
  const auto activation = activationFrom(o.activation);
  if (!activation) {
    return usage("unknown activation");
  }
  const auto train = loadSet(o.trainPath, o.inputs);
  if (!train) {
    return 1;
  }
  const feature_elm::Dataset& trainSet = *train;

  std::vector<float> x(trainSet.data.begin(), trainSet.data.end());
  std::vector<float> t(trainSet.numSamples * o.classes, 0.0f);
  for (std::size_t i = 0; i < trainSet.numSamples; ++i) {
    const int label = trainSet.labels[i];
    if (label < 0 || static_cast<std::size_t>(label) >= o.classes) {
      std::cerr << "felm-train: label " << label << " out of range on row " << i + 1 << "\n";
      return 1;
    }
    t[i * o.classes + static_cast<std::size_t>(label)] = 1.0f;
  }

  std::mt19937 gen(o.seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> weights(o.inputs * o.hidden);
  std::vector<float> biases(o.hidden);
  for (auto& w : weights) {
    w = dist(gen) / o.inputScale;
  }
  for (auto& b : biases) {
    b = dist(gen);
  }

  const bool gpu = !o.forceCpu && feature_elm::cuda_backend::isGpuAvailable();
  const Backend backend = gpu ? Backend::kGpu : Backend::kCpu;
  std::cout << "training " << o.hidden << " hidden units on " << trainSet.numSamples << " samples ("
            << (gpu ? feature_elm::cuda_backend::gpuDeviceName() : std::string("CPU")) << ")\n";

  BatchElm<float> model(o.inputs, o.hidden, *activation, backend, weights, biases, o.ridge);
  const auto start = std::chrono::steady_clock::now();
  if (!model.train(x, t, trainSet.numSamples, o.classes)) {
    std::cerr << "felm-train: training failed\n";
    return 1;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "trained in " << seconds << " s\n";

  if (!o.testPath.empty()) {
    const auto test = loadSet(o.testPath, o.inputs);
    if (!test) {
      return 1;
    }
    const feature_elm::Dataset& testSet = *test;
    const std::vector<float> tx(testSet.data.begin(), testSet.data.end());
    const auto scores = model.predictBatch(tx, testSet.numSamples);
    if (!scores) {
      std::cerr << "felm-train: prediction failed\n";
      return 1;
    }
    const std::vector<float>& scoreValues = *scores;
    std::vector<std::size_t> correct(o.classes, 0);
    std::vector<std::size_t> total(o.classes, 0);
    for (std::size_t i = 0; i < testSet.numSamples; ++i) {
      const auto row = scoreValues.begin() + static_cast<std::ptrdiff_t>(i * o.classes);
      const auto predicted = static_cast<std::size_t>(
          std::max_element(row, row + static_cast<std::ptrdiff_t>(o.classes)) - row);
      const auto label = static_cast<std::size_t>(testSet.labels[i]);
      if (label < o.classes) {
        total[label] += 1;
        correct[label] += predicted == label ? 1u : 0u;
      }
    }
    std::size_t allCorrect = 0;
    std::size_t all = 0;
    std::cout << "test accuracy per class:";
    for (std::size_t c = 0; c < o.classes; ++c) {
      allCorrect += correct[c];
      all += total[c];
      std::cout << " " << c << ":"
                << (total[c]
                        ? 100.0 * static_cast<double>(correct[c]) / static_cast<double>(total[c])
                        : 0.0)
                << "%";
    }
    std::cout << "\ntest accuracy: "
              << 100.0 * static_cast<double>(allCorrect) / static_cast<double>(all) << "% on "
              << all << " samples\n";
  }

  if (!model.save(o.outPath)) {
    std::cerr << "felm-train: cannot write " << o.outPath << "\n";
    return 1;
  }
  std::cout << "saved " << o.outPath << "\n";
  return 0;
}
