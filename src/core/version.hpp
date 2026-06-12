#ifndef FEATURE_ELM_CORE_VERSION_HPP_
#define FEATURE_ELM_CORE_VERSION_HPP_

#include <string>
#include <string_view>

namespace feature_elm {

struct Version {
  int major;
  int minor;
  int patch;
};

inline constexpr Version kVersion{0, 1, 0};
inline constexpr std::string_view kVersionText{"0.1.0"};

[[nodiscard]] constexpr std::string_view projectName() noexcept {
  return "feature_extraction_cuda_elm";
}

[[nodiscard]] std::string versionString();

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_VERSION_HPP_
