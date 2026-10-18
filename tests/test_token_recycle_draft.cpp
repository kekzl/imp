// Host-side Token-Recycling adjacency drafter (token_recycle_draft.cpp), speculative
// decoding. Design: docs/plans/2026-07-22-token-recycling-spec-tree.md (ACL 2025, arXiv
// 2408.08696).

#include "runtime/token_recycle_draft.h"

#include <gtest/gtest.h>
#include <span>
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
    t.observe_topk(1, std::span(ids, 3));
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
    t.observe_topk(1, std::span(ids, 2));
    EXPECT_EQ(t.successor(1, 0), 3);
    EXPECT_EQ(t.successor(1, 1), -1);
}

TEST(TokenRecycle, CandidatesBranchAtRoot) {
    TokenRecycleTable t(100, 4);
    const int32_t ids[3] = {5, 6, 7};  // top-3 successors of 1
    t.observe_topk(1, std::span(ids, 3));
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

// 1 -> {5, 6, 7}, 5 -> {50, 51}, 50 -> {500}. Rank sums: [5,50,500] 0, [6] 1, [5,51] 1 (deeper,
// so after [6]), [7] 2. draft_candidates never reaches [5,51]: it branches at the root only.
static TokenRecycleTable tree_table() {
    TokenRecycleTable t(1000, 4);
    const int32_t root[3] = {5, 6, 7};
    t.observe_topk(1, std::span(root, 3));
    const int32_t five[2] = {50, 51};
    t.observe_topk(5, std::span(five, 2));
    t.observe_pair(50, 500);
    return t;
}

TEST(TokenRecycle, TreeBranchesBelowTheRootByRankSum) {
    const TokenRecycleTable t = tree_table();
    const auto c = t.draft_tree(1, 4, 3);
    const std::vector<std::vector<int32_t>> want = {{5, 50, 500}, {6}, {5, 51}, {7}};
    EXPECT_EQ(c, want);
    for (const auto& cand : t.draft_candidates(1, 4, 3))
        EXPECT_NE(cand, (std::vector<int32_t>{5, 51})) << "the root-only drafter has no second-hop branch";
}

TEST(TokenRecycle, TreeLearnsSlotHitRates) {
    TokenRecycleTable t = tree_table();
    EXPECT_DOUBLE_EQ(t.rank_hit_rate(0), 0.5);  // prior 2^-(r+1)
    for (int i = 0; i < 100; ++i)
        t.observe_rank(1, 7);  // the model kept choosing slot 2 of token 1
    t.observe_rank(1, 999);    // a miss counts as a sample
    EXPECT_GT(t.rank_hit_rate(2), t.rank_hit_rate(0));
    const auto c = t.draft_tree(1, 2, 3);
    const std::vector<std::vector<int32_t>> want = {{7}, {5, 50, 500}};
    EXPECT_EQ(c, want) << "slot 2 is now free, slot 1 ([6]) no longer makes the cut";
}

TEST(TokenRecycle, CandidateRanksCountSharedPrefixesOnce) {
    TokenRecycleTable t = tree_table();
    const std::vector<std::vector<int32_t>> mc = {{5, 50}, {5, 51}};
    // rows_per_cand 3: [t0], [t0 5], [t0 5 50] | [t0] dup, [t0 5] dup, [t0 5 51]
    const int32_t argmax[6] = {5, 50, 500, 5, 50, 7};
    observe_candidate_ranks(t, false, 1, mc, argmax);
    EXPECT_EQ(t.rank_samples(), 0u) << "speculative.recycle_tree off";
    observe_candidate_ranks(t, true, 1, mc, argmax);
    EXPECT_EQ(t.rank_samples(), 3u) << "2 rows are shared prefixes; [t0 5 51] follows 51, an empty row";
    EXPECT_GT(t.rank_hit_rate(0), 0.5) << "all 3 argmaxes sat in slot 0";
}

TEST(TokenRecycle, TreeKeepsWidthAndDepth) {
    const TokenRecycleTable t = tree_table();
    const auto c = t.draft_tree(1, 2, 2);
    const std::vector<std::vector<int32_t>> want = {{5, 50}, {6}};
    EXPECT_EQ(c, want);
    EXPECT_TRUE(t.draft_tree(9, 4, 3).empty()) << "unseen root";
    EXPECT_TRUE(t.draft_tree(1, 0, 3).empty());
}

// Streak gating (#1055): a verify step costs ~1.4x a decode step, so drafts
// should only fire on CONFIRMED bigrams — the front slot must have repeated
// at least min_streak times. Fresh (streak-0) successors extend nothing.
TEST(TokenRecycle, StreakGateBlocksUnconfirmedBigram) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);  // first sighting: streak 0
    EXPECT_TRUE(t.draft_linear(1, 4, /*min_streak=*/1).empty());
    t.observe_pair(1, 2);  // repeat confirms: streak 1
    auto d = t.draft_linear(1, 4, 1);
    ASSERT_EQ(d.size(), 1u);  // 2 has no confirmed successor -> chain ends
    EXPECT_EQ(d[0], 2);
}

TEST(TokenRecycle, StreakResetsOnBrandNewSuccessor) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);
    t.observe_pair(1, 2);  // streak 1
    t.observe_pair(1, 3);  // brand-new successor -> streak resets
    EXPECT_TRUE(t.draft_linear(1, 4, 1).empty());
}

TEST(TokenRecycle, StreakCountsNonConsecutiveRepeats) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);
    t.observe_pair(1, 3);  // interleaved successor
    t.observe_pair(1, 2);  // 2 re-observed -> confirmed despite the gap
    auto d = t.draft_linear(1, 4, 1);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 2);
}

TEST(TokenRecycle, MinStreakZeroKeepsOldBehavior) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);
    auto d = t.draft_linear(1, 4, 0);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 2);
}

TEST(TokenRecycle, StreakGateAppliesPerHop) {
    TokenRecycleTable t(100, 4);
    t.observe_pair(1, 2);
    t.observe_pair(1, 2);  // 1->2 confirmed
    t.observe_pair(2, 3);
    t.observe_pair(2, 3);  // 2->3 confirmed
    t.observe_pair(3, 4);  // 3->4 unconfirmed
    auto d = t.draft_linear(1, 8, 1);
    ASSERT_EQ(d.size(), 2u);  // stops before the unconfirmed hop
    EXPECT_EQ(d[0], 2);
    EXPECT_EQ(d[1], 3);
}

}  // namespace
