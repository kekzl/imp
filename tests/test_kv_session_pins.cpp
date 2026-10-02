// Session pins (#2407): session_id owner, turn replace, budget, TTL, close. Accounting-only
// KVCache (no VRAM), CPU unit lane.
#include <gtest/gtest.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace imp {
namespace {

std::unique_ptr<KVCacheManager> MakeManager(int max_blocks) {
    return std::make_unique<KVCacheManager>(KVCache::for_accounting(2, 4, 64, QType::F16, max_blocks));
}

std::vector<int32_t> Tokens(int n_full_blocks, int token_base) {
    std::vector<int32_t> tokens(static_cast<size_t>(n_full_blocks) * 16);
    std::iota(tokens.begin(), tokens.end(), token_base);
    return tokens;
}

// One agent turn: prompt of n_full_blocks blocks from token_base, finish = hash + session pin + free
// (Engine::finish_request order). Returns the prefix-cache hit in blocks.
int SessionTurn(KVCacheManager* mgr, int seq_id, const char* session, int n_full_blocks, int token_base,
                int64_t now_ms) {
    const auto tokens = Tokens(n_full_blocks, token_base);
    const int hit = mgr->allocate_blocks_with_prefix(seq_id, tokens);
    mgr->register_block_hashes(seq_id, tokens);
    if (session != nullptr)
        mgr->pin_session(session, seq_id, n_full_blocks, now_ms);
    mgr->free_sequence(seq_id);
    return hit;
}

// Issue acceptance shape on the manager: A turn 1 (2 blocks), filler that needs 6 of 8 blocks
// twice over, A turn 2 (same 2 blocks + 1). Returns turn 2's hit.
int TurnTwoHitAfterFiller(bool pin) {
    auto mgr = MakeManager(8);
    mgr->set_prefix_caching_enabled(true);
    mgr->set_pin_budget_blocks(4);
    EXPECT_EQ(SessionTurn(mgr.get(), 0, pin ? "A" : nullptr, 2, 100, 0), 0);
    EXPECT_EQ(SessionTurn(mgr.get(), 1, nullptr, 6, 1000, 1), 0);
    EXPECT_EQ(SessionTurn(mgr.get(), 2, nullptr, 6, 2000, 2), 0);
    const auto turn2 = Tokens(3, 100);
    const int hit = mgr->allocate_blocks_with_prefix(3, turn2);
    mgr->free_sequence(3);
    return hit;
}

TEST(KVSessionPins, TurnTwoKeepsItsPrefixHitUnderPressure) {
    EXPECT_EQ(TurnTwoHitAfterFiller(/*pin=*/true), 2);   // pinned arm: the whole turn-1 prompt
    EXPECT_EQ(TurnTwoHitAfterFiller(/*pin=*/false), 0);  // unpinned arm: LRU evicted it
}

TEST(KVSessionPins, EvictionTakesUnpinnedLruFirst) {
    auto mgr = MakeManager(8);
    mgr->set_prefix_caching_enabled(true);
    SessionTurn(mgr.get(), 0, "A", 2, 100, 0);
    SessionTurn(mgr.get(), 1, nullptr, 2, 900, 1);
    // 4 cached blocks: A's 2 (older) pinned, the filler's 2 reclaimable.
    EXPECT_EQ(mgr->num_reclaimable_cached_blocks(), 2);
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_FALSE(mgr->evict_cached_block());
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);
}

TEST(KVSessionPins, NextTurnReplacesThePin) {
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    EXPECT_EQ(SessionTurn(mgr.get(), 0, "A", 2, 100, 0), 0);
    EXPECT_EQ(mgr->num_sessions(), 1);
    EXPECT_EQ(mgr->num_session_pinned_blocks(), 2);
    EXPECT_EQ(SessionTurn(mgr.get(), 1, "A", 3, 100, 1), 2);  // turn 2 reuses turn 1's blocks
    EXPECT_EQ(mgr->num_sessions(), 1);
    EXPECT_EQ(mgr->num_session_pinned_blocks(), 3);
    EXPECT_EQ(mgr->num_pinned_blocks(), 3);
}

TEST(KVSessionPins, CloseReleasesItsBlocks) {
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    SessionTurn(mgr.get(), 0, "A", 2, 100, 0);
    SessionTurn(mgr.get(), 1, "B", 1, 900, 0);
    EXPECT_EQ(mgr->close_session("A"), 2);
    EXPECT_EQ(mgr->close_session("A"), 0);  // idempotent
    EXPECT_EQ(mgr->close_session("nope"), 0);
    EXPECT_EQ(mgr->num_sessions(), 1);
    EXPECT_EQ(mgr->num_session_pinned_blocks(), 1);
    EXPECT_EQ(mgr->num_pinned_blocks(), 1);
    EXPECT_EQ(mgr->num_reclaimable_cached_blocks(), 2);
}

TEST(KVSessionPins, TtlExpiresIdleSessionsOnly) {
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    SessionTurn(mgr.get(), 0, "A", 2, 100, 0);
    SessionTurn(mgr.get(), 1, "B", 2, 900, 5000);
    EXPECT_EQ(mgr->expire_sessions(7000, 0), 0);     // ttl 0: no TTL
    EXPECT_EQ(mgr->expire_sessions(6000, 6000), 0);  // A idle exactly ttl: kept
    EXPECT_EQ(mgr->expire_sessions(7000, 6000), 1);  // A idle 7000 > 6000
    EXPECT_EQ(mgr->num_sessions(), 1);
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);
    SessionTurn(mgr.get(), 2, "B", 2, 900, 10000);  // a new turn refreshes B
    EXPECT_EQ(mgr->expire_sessions(15000, 6000), 0);
    EXPECT_EQ(mgr->expire_sessions(16001, 6000), 1);
    EXPECT_EQ(mgr->num_sessions(), 0);
    EXPECT_EQ(mgr->num_pinned_blocks(), 0);
    EXPECT_EQ(mgr->num_session_pinned_blocks(), 0);
}

TEST(KVSessionPins, BudgetUnpinsLeastRecentlyPinnedSession) {
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    mgr->set_pin_budget_blocks(4);
    SessionTurn(mgr.get(), 0, "A", 2, 100, 0);
    SessionTurn(mgr.get(), 1, "B", 2, 900, 1);
    SessionTurn(mgr.get(), 2, "A", 2, 100, 2);  // A pinned again: B is now the oldest pin
    SessionTurn(mgr.get(), 3, "C", 2, 1700, 3);
    EXPECT_EQ(mgr->num_pinned_blocks(), 4);
    EXPECT_EQ(mgr->num_sessions(), 2);
    EXPECT_EQ(mgr->close_session("B"), 0);  // B was released by the budget
    EXPECT_EQ(mgr->close_session("A"), 2);
    EXPECT_EQ(mgr->close_session("C"), 2);
}

TEST(KVSessionPins, ShareTheCacheControlBudget) {
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    mgr->set_pin_budget_blocks(3);
    const auto cc = Tokens(2, 100);  // cache_control pin (pin_prefix), oldest owner
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, cc), 0);
    mgr->register_block_hashes(0, cc);
    mgr->pin_prefix(0, 2);
    mgr->free_sequence(0);
    SessionTurn(mgr.get(), 1, "A", 2, 900, 0);
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);  // budget 3: the cache_control pin went first
    EXPECT_EQ(mgr->num_session_pinned_blocks(), 2);
    SessionTurn(mgr.get(), 2, "B", 5, 1700, 1);  // larger than the budget: capped, A released
    EXPECT_EQ(mgr->num_pinned_blocks(), 3);
    EXPECT_EQ(mgr->num_sessions(), 1);
    EXPECT_EQ(mgr->num_session_pinned_blocks(), 3);
}

// #2503: budget 0 (server.prefix_pin_budget_pct = 0) = no cache_control pin and no session pin.
// budget > 0: both pin. StreamingLLM sinks pin at 0 (correctness pin, not a cache pin).
int PinnedAfterCacheControlAndSession(int budget_blocks) {
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    mgr->set_pin_budget_blocks(budget_blocks);
    const auto cc = Tokens(2, 100);
    EXPECT_EQ(mgr->allocate_blocks_with_prefix(0, cc), 0);
    mgr->register_block_hashes(0, cc);
    mgr->pin_prefix(0, 2);
    mgr->free_sequence(0);
    SessionTurn(mgr.get(), 1, "A", 2, 900, 0);
    EXPECT_EQ(mgr->num_sessions(), budget_blocks == 0 ? 0 : 1);
    EXPECT_EQ(mgr->num_session_pinned_blocks(), budget_blocks == 0 ? 0 : 2);
    return mgr->num_pinned_blocks();
}

TEST(KVSessionPins, BudgetZeroMeansNoPins) {
    EXPECT_EQ(PinnedAfterCacheControlAndSession(0), 0);
    EXPECT_EQ(PinnedAfterCacheControlAndSession(4), 4);
    EXPECT_EQ(PinnedAfterCacheControlAndSession(3), 2);  // FIFO: cache_control pin released first
}

TEST(KVSessionPins, BudgetZeroKeepsStreamingSinkPins) {
    auto mgr = MakeManager(32);
    mgr->set_pin_budget_blocks(0);
    ASSERT_TRUE(mgr->allocate_blocks(0, 20));
    EXPECT_EQ(mgr->evict_middle_blocks(0, /*n_sink_tokens=*/4, /*n_window_tokens=*/64), 14);
    EXPECT_EQ(mgr->num_pinned_blocks(), 1);
}

}  // namespace
}  // namespace imp
