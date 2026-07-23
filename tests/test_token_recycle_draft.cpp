// Host-side tests for the Token-Recycling adjacency drafter
// (src/runtime/token_recycle_draft.cpp) used by speculative decoding.
// Design: docs/plans/2026-07-22-token-recycling-spec-tree.md (Token
// Recycling, ACL 2025, arXiv 2408.08696).

#include "runtime/token_recycle_draft.h"

#include <gtest/gtest.h>
#include <vector>

using imp::TokenRecycleTable;

namespace {

TEST(TokenRecycle, EmptyOnUnseenToken) {
    TokenRecycleTable t(100, 4);
    EXPECT_FALSE(t.has(7));
    EXPECT_TRUE(t.draft_linear(7, 8).empty());
}

TEST(TokenRecycle, PairChainDraft) {
    TokenRecycleTable t(100, 4);
    // Observed stream: 1 -> 2 -> 3 -> 4
    t.observe_pair(1, 2);
    t.observe_pair(2, 3);
    t.observe_pair(3, 4);
    auto d = t.draft_linear(1, 8);
    ASSERT_EQ(d.size(), 3u);  // chain ends at 4 (no successor)
    EXPECT_EQ(d[0], 2);
    EXPECT_EQ(d[1], 3);
    EXPECT_EQ(d[2], 4);
}

TEST(TokenRecycle, TruncatesAtK) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);
    t.observe_pair(2, 3);
    t.observe_pair(3, 4);
    auto d = t.draft_linear(1, 2);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0], 2);
    EXPECT_EQ(d[1], 3);
}

TEST(TokenRecycle, MostRecentPairWins) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);
    t.observe_pair(1, 3);  // newer observation takes the front slot
    auto d = t.draft_linear(1, 1);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 3);
}

TEST(TokenRecycle, RepeatedPairPromotesNotDuplicates) {
    TokenRecycleTable t(100, 2);
    t.observe_pair(1, 2);
    t.observe_pair(1, 3);
    t.observe_pair(1, 2);  // promote 2 back to front; 3 must survive in slot 1
    auto d = t.draft_linear(1, 1);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 2);
    EXPECT_EQ(t.successor(1, 1), 3);
}

TEST(TokenRecycle, SlotsEvictOldest) {
    TokenRecycleTable t(100, 2);
    t.observe_pair(1, 10);
    t.observe_pair(1, 11);
    t.observe_pair(1, 12);  // evicts 10 (LRU with 2 slots)
    EXPECT_EQ(t.successor(1, 0), 12);
    EXPECT_EQ(t.successor(1, 1), 11);
    EXPECT_EQ(t.successor(1, 2), -1);  // out of slots
}

TEST(TokenRecycle, TopKObservationSetsRankOrder) {
    TokenRecycleTable t(100, 4);
    const int32_t ids[3] = {5, 6, 7};  // model's top-3 for token 1, best first
    t.observe_topk(1, ids, 3);
    EXPECT_EQ(t.successor(1, 0), 5);
    EXPECT_EQ(t.successor(1, 1), 6);
    EXPECT_EQ(t.successor(1, 2), 7);
    auto d = t.draft_linear(1, 1);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 5);
}

TEST(TokenRecycle, SelfLoopChainStillBounded) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 1);  // degenerate self-successor
    auto d = t.draft_linear(1, 5);
    ASSERT_EQ(d.size(), 5u);  // bounded by k, verify is the safety net
    for (int32_t v : d) EXPECT_EQ(v, 1);
}

TEST(TokenRecycle, IgnoresOutOfRangeIds) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 200);   // next out of vocab — dropped
    t.observe_pair(-1, 2);    // prev invalid — dropped
    t.observe_pair(200, 2);   // prev out of vocab — dropped
    EXPECT_FALSE(t.has(1));
    EXPECT_TRUE(t.draft_linear(1, 4).empty());
    const int32_t ids[2] = {150, 3};  // out-of-range entry skipped, valid kept
    t.observe_topk(1, ids, 2);
    EXPECT_EQ(t.successor(1, 0), 3);
    EXPECT_EQ(t.successor(1, 1), -1);
}

TEST(TokenRecycle, CandidatesBranchAtRoot) {
    TokenRecycleTable t(100, 4);
    const int32_t ids[3] = {5, 6, 7};  // top-3 successors of 1
    t.observe_topk(1, ids, 3);
    t.observe_pair(5, 50);  // chain continuations
    t.observe_pair(6, 60);
    // 3 candidates, depth 2: [5,50], [6,60], [7] (7 has no successor).
    auto c = t.draft_candidates(1, 3, 2);
    ASSERT_EQ(c.size(), 3u);
    ASSERT_EQ(c[0].size(), 2u);
    EXPECT_EQ(c[0][0], 5);
    EXPECT_EQ(c[0][1], 50);
    ASSERT_EQ(c[1].size(), 2u);
    EXPECT_EQ(c[1][0], 6);
    EXPECT_EQ(c[1][1], 60);
    ASSERT_EQ(c[2].size(), 1u);
    EXPECT_EQ(c[2][0], 7);
}

TEST(TokenRecycle, CandidatesLimitedBySlots) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);  // only one successor known
    auto c = t.draft_candidates(1, 4, 3);
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0][0], 2);
}

TEST(TokenRecycle, CandidatesEmptyOnUnseenRoot) {
    TokenRecycleTable t(100, 4);
    EXPECT_TRUE(t.draft_candidates(9, 4, 3).empty());
}

}  // namespace
