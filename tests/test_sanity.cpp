#include <gtest/gtest.h>

#include <string>

#include "core/version.hpp"

namespace {

TEST(SanityTest, ReportsProjectMetadata) {
  EXPECT_EQ(feature_elm::projectName(), "feature_extraction_cuda_elm");
  // scripts/release.sh bumps both fields; the release workflow checks them against the tag.
  const auto& v = feature_elm::kVersion;
  const std::string expected =
      std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.patch);
  EXPECT_EQ(feature_elm::versionString(), expected);
}

}  // namespace
