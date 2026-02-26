#include <gtest/gtest.h>
#include "runtime/speculative.h"

namespace imp {
namespace {

TEST(SpeculativeConfigTest, DefaultValues) {
    SpeculativeConfig config;
    EXPECT_EQ(config.spec_k, 4);
    EXPECT_FLOAT_EQ(config.acceptance_threshold, 0.0f);
}

TEST(SpeculativeConfigTest, CustomValues) {
    SpeculativeConfig config;
    config.spec_k = 8;
    config.acceptance_threshold = 0.1f;
    EXPECT_EQ(config.spec_k, 8);
    EXPECT_FLOAT_EQ(config.acceptance_threshold, 0.1f);
}

TEST(SpeculativeDecoderTest, UninitializedState) {
    SpeculativeDecoder decoder;
    EXPECT_EQ(decoder.spec_k(), 4);
}

TEST(SpeculativeDecoderTest, InitWithNullTarget) {
    SpeculativeDecoder decoder;
    SpeculativeConfig config;
    // init with null target executor should fail
    bool ok = decoder.init(nullptr, nullptr, nullptr, nullptr, nullptr, config);
    EXPECT_FALSE(ok);
}

TEST(SpeculativeDecoderTest, DraftWithoutInit) {
    SpeculativeDecoder decoder;
    // draft_tokens on uninitialized decoder should return empty
    auto tokens = decoder.draft_tokens(0, 0, 0, nullptr);
    EXPECT_TRUE(tokens.empty());
}

TEST(SpeculativeDecoderTest, StepWithoutInit) {
    SpeculativeDecoder decoder;
    // step on uninitialized decoder should return empty
    auto tokens = decoder.step(0, 0, 0, 0.0f, 1.0f, 0, -1, nullptr);
    EXPECT_TRUE(tokens.empty());
}

TEST(SpeculativeDecoderTest, VerifyWithoutInit) {
    SpeculativeDecoder decoder;
    std::vector<int32_t> draft = {1, 2, 3};
    auto result = decoder.verify(draft, 0, 0, 0, 0.0f, 1.0f, 0, -1, nullptr);
    EXPECT_EQ(result.n_accepted, 0);
    EXPECT_TRUE(result.accepted.empty());
    EXPECT_EQ(result.next_token, -1);
}

} // namespace
} // namespace imp
