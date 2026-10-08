// GGUF MoE NVFP4 decode cache rule: whole layer or nothing (src/exec/moe_nvfp4_cache_rule.h).

#include <gtest/gtest.h>

#include "exec/moe_nvfp4_cache_rule.h"

using imp::moe_layer_cache_whole;
using imp::MoeProjCacheState;

namespace {
constexpr MoeProjCacheState kAbsent{};
constexpr MoeProjCacheState kEligible{true, false, true};
constexpr MoeProjCacheState kIneligible{true, false, false};
constexpr MoeProjCacheState kCached{true, true, false};
}  // namespace

// UD-Q4_K_M: gate/up Q4_K (not eligible), down Q5_K/Q6_K (eligible). Caching down alone wasted
// 5760 MiB on Qwen3.6-35B-A3B.
TEST(MoeNvfp4CacheRule, LoneEligibleDownIsNotCached) {
    EXPECT_FALSE(moe_layer_cache_whole(kIneligible, kIneligible, kEligible));
    EXPECT_FALSE(moe_layer_cache_whole(kEligible, kIneligible, kEligible));
    EXPECT_FALSE(moe_layer_cache_whole(kEligible, kEligible, kIneligible));
}

TEST(MoeNvfp4CacheRule, AllEligibleIsCached) {
    EXPECT_TRUE(moe_layer_cache_whole(kEligible, kEligible, kEligible));
    EXPECT_TRUE(moe_layer_cache_whole(kCached, kEligible, kEligible));
    EXPECT_TRUE(moe_layer_cache_whole(kCached, kCached, kCached));
}

// Non-gated MoE (Nemotron-H): no gate projection, up + down decide.
TEST(MoeNvfp4CacheRule, NonGatedNeedsUpAndDown) {
    EXPECT_TRUE(moe_layer_cache_whole(kAbsent, kEligible, kEligible));
    EXPECT_TRUE(moe_layer_cache_whole(kAbsent, kCached, kCached));
    EXPECT_FALSE(moe_layer_cache_whole(kAbsent, kAbsent, kEligible));
    EXPECT_FALSE(moe_layer_cache_whole(kAbsent, kIneligible, kEligible));
}

TEST(MoeNvfp4CacheRule, DenseLayerIsNotCached) {
    EXPECT_FALSE(moe_layer_cache_whole(kAbsent, kAbsent, kAbsent));
}
