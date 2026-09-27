#include <gtest/gtest.h>

#include "core/version.hpp"

namespace {

TEST(SanityTest, ReportsProjectMetadata) {
  EXPECT_EQ(feature_elm::projectName(), "feature_extraction_cuda_elm");
  EXPECT_EQ(feature_elm::versionString(), "0.2.0");
  EXPECT_EQ(feature_elm::kVersion.major, 0);
  EXPECT_EQ(feature_elm::kVersion.minor, 2);
  EXPECT_EQ(feature_elm::kVersion.patch, 0);
}

}  // namespace
