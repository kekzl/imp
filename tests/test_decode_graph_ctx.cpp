// Decode graph re-capture points (#2182): runtime/decode_graph_ctx.h is header-only.
// Two identical requests in one engine must re-capture at the same contexts under
// runtime.deterministic; the default keeps the growth-only high-water mark (#948).
#include "runtime/decode_graph_ctx.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace imp;

namespace {

// Contexts at which one request of `n_prompt` + `n_gen` tokens re-captures.
std::vector<int> captures(int& hw, uint64_t& key, int req_id, int n_prompt, int n_gen, bool det) {
    std::vector<int> at;
    const std::array<int, 1> ids{req_id};
    for (int ctx = n_prompt + 1; ctx <= n_prompt + n_gen; ++ctx)
        if (decode_graph_ctx_recapture(hw, key, ctx, decode_graph_batch_key(ids), det))
            at.push_back(ctx);
    return at;
}

}  // namespace

TEST(DecodeGraphCtx, BucketPow2) {
    EXPECT_EQ(decode_graph_bucket_pow2(0), 1);
    EXPECT_EQ(decode_graph_bucket_pow2(1), 1);
    EXPECT_EQ(decode_graph_bucket_pow2(21), 32);
    EXPECT_EQ(decode_graph_bucket_pow2(128), 128);
    EXPECT_EQ(decode_graph_bucket_pow2(129), 256);
}

TEST(DecodeGraphCtx, DeterministicSecondRequestCapturesAtSameContexts) {
    // #2182 shape: 20-token prompt, 127 decode steps, run twice in one engine.
    int hw = 0;
    uint64_t key = 0;
    const std::vector<int> run1 = captures(hw, key, /*req_id=*/1, 20, 127, true);
    const std::vector<int> run2 = captures(hw, key, /*req_id=*/2, 20, 127, true);
    EXPECT_EQ(run1, (std::vector<int>{21, 33, 65, 129}));
    EXPECT_EQ(run1, run2);
}

TEST(DecodeGraphCtx, DefaultKeepsGrowthOnlyHighWater) {
    // Default mode: run 2 never re-captures, it replays run 1's ctx-129 capture (#948 contract).
    int hw = 0;
    uint64_t key = 0;
    EXPECT_EQ(captures(hw, key, 1, 20, 127, false), (std::vector<int>{21, 33, 65, 129}));
    EXPECT_TRUE(captures(hw, key, 2, 20, 127, false).empty());
    EXPECT_EQ(hw, 256);
}

TEST(DecodeGraphCtx, DeterministicSameBatchStaysGrowthOnly) {
    int hw = 0;
    uint64_t key = 0;
    const std::array<int, 2> ids{3, 4};
    const uint64_t k = decode_graph_batch_key(ids);
    EXPECT_TRUE(decode_graph_ctx_recapture(hw, key, 100, k, true));
    EXPECT_FALSE(decode_graph_ctx_recapture(hw, key, 101, k, true));
    EXPECT_FALSE(decode_graph_ctx_recapture(hw, key, 60, k, true));  // shrink: no re-capture
    EXPECT_TRUE(decode_graph_ctx_recapture(hw, key, 129, k, true));
}

TEST(DecodeGraphCtx, BatchKeyDependsOnIdsAndOrder) {
    const std::array<int, 2> a{3, 4}, b{4, 3}, c{3, 5};
    EXPECT_NE(decode_graph_batch_key(a), decode_graph_batch_key(b));
    EXPECT_NE(decode_graph_batch_key(a), decode_graph_batch_key(c));
    EXPECT_EQ(decode_graph_batch_key(a), decode_graph_batch_key(std::array<int, 2>{3, 4}));
}

TEST(DecodeGraphCtx, BatchKeyOfRequestsMatchesIds) {
    struct R {
        int id;
    };
    R r3{3}, r4{4};
    const std::vector<const R*> reqs{&r3, &r4};
    const std::array<int, 2> ids{3, 4};
    EXPECT_EQ(decode_graph_batch_key_of(reqs), decode_graph_batch_key(ids));
}
