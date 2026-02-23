#include <gtest/gtest.h>
#include "runtime/speculative.h"

namespace imp {
namespace {

TEST(SpeculativeConfigTest, DefaultValues) {
    SpeculativeConfig config;
    EXPECT_EQ(config.spec_k, 4);
    EXPECT_FLOAT_EQ(config.acceptance_threshold, 0.0f);
}

TEST(SpeculativeDecoderTest, UninitializedState) {
    SpeculativeDecoder decoder;
    EXPECT_EQ(decoder.spec_k(), 4);
}

} // namespace
} // namespace imp
