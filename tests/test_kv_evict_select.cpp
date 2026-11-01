// KV eviction block selection (kv_cache.evict_keep_ratio, src/memory/kv_evict_select.h): budget
// arithmetic, forced sink / recent blocks, top-score pick, deterministic ties, NaN handling.

#include <gtest/gtest.h>
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/kv_evict_select.h"
#include <cmath>
#include <memory>
#include <vector>

namespace imp {
namespace {

KvEvictParams params(float ratio, int sink = 16, int recent = 16) {
    KvEvictParams p;
    p.keep_ratio = ratio;
    p.sink_tokens = sink;
    p.recent_tokens = recent;
    p.block_size = 16;
    return p;
}

TEST(KvEvictSelect, OffAndFullRatioEvictNothing) {
    const std::vector<float> s(160, 1.0f);
    EXPECT_TRUE(kv_evict_select(s, 160, params(0.0f)).empty());
    EXPECT_TRUE(kv_evict_select(s, 160, params(1.0f)).empty());
}

TEST(KvEvictSelect, CandidateRangeExcludesSinkAndRecentBlocks) {
    // 170 tokens = 10 full blocks + partial; recent 20 tokens starts in block 9 (token 150).
    const KvEvictRange r = kv_evict_candidates(170, params(0.5f, 16, 20));
    EXPECT_EQ(r.begin, 16);
    EXPECT_EQ(r.end, 144);
    // Too short: sinks and window overlap.
    const KvEvictRange none = kv_evict_candidates(30, params(0.5f, 16, 16));
    EXPECT_EQ(none.begin, none.end);
}

TEST(KvEvictSelect, KeepsTopScoredBlocksWithinBudget) {
    // 160 tokens = 10 blocks; sink block 0, recent block 9; budget ceil(0.5 * 10) = 5 -> 3 slots.
    std::vector<float> s(160, 0.0f);
    s[3 * 16 + 5] = 9.0f;   // block 3
    s[6 * 16 + 0] = 8.0f;   // block 6
    s[7 * 16 + 15] = 7.0f;  // block 7
    s[2 * 16 + 1] = 1.0f;   // block 2, below the cut
    const std::vector<int> ev = kv_evict_select(s, 160, params(0.5f));
    const std::vector<int> want = {1, 2, 4, 5, 8};
    EXPECT_EQ(ev, want);
}

TEST(KvEvictSelect, TiesKeepEarlierBlocks) {
    const std::vector<float> s(160, 0.5f);
    const std::vector<int> ev = kv_evict_select(s, 160, params(0.5f));
    const std::vector<int> want = {4, 5, 6, 7, 8};
    EXPECT_EQ(ev, want);
}

TEST(KvEvictSelect, NanScoresRankLast) {
    std::vector<float> s(160, NAN);
    s[8 * 16] = -3.0f;  // only finite candidate
    const std::vector<int> ev = kv_evict_select(s, 160, params(0.25f));
    // budget ceil(2.5) = 3, forced 2 -> 1 slot, block 8 stays.
    const std::vector<int> want = {1, 2, 3, 4, 5, 6, 7};
    EXPECT_EQ(ev, want);
}

TEST(KvEvictSelect, ForcedBlocksAboveBudgetEvictAllCandidates) {
    const std::vector<float> s(160, 1.0f);
    // budget ceil(0.1f * 10) = 2 = forced blocks: every candidate goes.
    const std::vector<int> ev = kv_evict_select(s, 160, params(0.1f));
    EXPECT_EQ(ev.size(), 8u);
    EXPECT_EQ(ev.front(), 1);
    EXPECT_EQ(ev.back(), 8);
}

// KVCacheManager::evict_blocks: listed blocks become -1 holes and return to the pool; out-of-range,
// already-freed and pinned blocks stay; the table length never changes.
TEST(KvEvictBlocks, FreesListedBlocksAsHoles) {
    auto mgr = std::make_unique<KVCacheManager>(KVCache::for_accounting(2, 4, 64, QType::F16, 16));
    ASSERT_TRUE(mgr->allocate_blocks(0, 10));
    const std::vector<int> before = mgr->block_table(0);
    const int free_before = mgr->num_free_blocks();
    mgr->pin_prefix(0, 2);  // blocks 0, 1 pinned
    const std::vector<int> ev = {1, 3, 5, 99, -1};
    EXPECT_EQ(mgr->evict_blocks(0, ev), 2);
    const std::vector<int>& after = mgr->block_table(0);
    ASSERT_EQ(after.size(), before.size());
    EXPECT_EQ(after[1], before[1]);  // pinned
    EXPECT_EQ(after[3], -1);
    EXPECT_EQ(after[5], -1);
    EXPECT_EQ(after[4], before[4]);
    EXPECT_EQ(mgr->num_free_blocks(), free_before + 2);
    EXPECT_EQ(mgr->evict_blocks(0, ev), 0);  // idempotent
    EXPECT_EQ(mgr->evict_blocks(7, ev), 0);  // unknown sequence
    mgr->unpin_prefix(0);
    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_free_blocks(), 16);
}

}  // namespace
}  // namespace imp
