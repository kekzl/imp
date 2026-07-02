// Host-side tests for the suffix-indexed draft matcher
// (src/runtime/suffix_draft.cpp) used by speculative decoding
// (speculative.suffix). Mirrors the ngram_draft battery where semantics
// coincide, plus the vote/adaptive-length/incremental behaviors that differ.

#include "runtime/suffix_draft.h"

#include <gtest/gtest.h>
#include <vector>

using imp::SuffixDraftIndex;

namespace {

SuffixDraftIndex make_index(const std::vector<int32_t>& hist, int min_match = 3, int max_match = 8) {
    SuffixDraftIndex idx(min_match, max_match);
    idx.append(hist.data(), static_cast<int>(hist.size()));
    return idx;
}

std::vector<int32_t> draft(const std::vector<int32_t>& hist, int k, int min_match = 3, int max_match = 8,
                           int k_max = 0, int* start = nullptr) {
    auto idx = make_index(hist, min_match, max_match);
    return idx.draft(k, k_max > 0 ? k_max : k, start);
}

TEST(SuffixDraft, EmptyWhenNoRepeat) { EXPECT_TRUE(draft({1, 2, 3, 4, 5, 6, 7, 8}, 4).empty()); }

TEST(SuffixDraft, EmptyWhenTooShort) {
    EXPECT_TRUE(draft({1, 2, 3}, 4).empty());
    EXPECT_TRUE(draft({}, 4).empty());
}

TEST(SuffixDraft, FindsSimpleRepeat) {
    std::vector<int32_t> h = {10, 11, 12, 13, 14, 99, 10, 11, 12};
    auto d = draft(h, 4);
    ASSERT_EQ(d.size(), 4u);
    EXPECT_EQ(d[0], 13);
    EXPECT_EQ(d[1], 14);
    EXPECT_EQ(d[2], 99);
    EXPECT_EQ(d[3], 10);
}

TEST(SuffixDraft, TruncatesAtK) {
    std::vector<int32_t> h = {1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 1, 2, 3};
    auto d = draft(h, 2);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0], 4);
    EXPECT_EQ(d[1], 5);
}

TEST(SuffixDraft, PrefersLongerMatchOnVoteTie) {
    // Two occurrences of suffix [7,8,9] with different continuations
    // (1 vote each): the one with the longer backward context match wins.
    std::vector<int32_t> h = {6, 7, 8, 9, 100, 101, 5, 7, 8, 9, 200, 201, 6, 7, 8, 9};
    auto d = draft(h, 2, 3, 8);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0], 100);
    EXPECT_EQ(d[1], 101);
}

TEST(SuffixDraft, PrefersRecentOnEqualLength) {
    std::vector<int32_t> h = {1, 7, 8, 9, 100, 2, 7, 8, 9, 200, 3, 7, 8, 9};
    auto d = draft(h, 1, 3, 3);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 200);
}

TEST(SuffixDraft, MajorityVoteBeatsRecency) {
    // Suffix [7,8,9] occurs three times: twice continuing with 100, once
    // (most recently) with 200. Frequency wins over recency — this is the
    // deliberate semantic upgrade over ngram_draft.
    std::vector<int32_t> h = {1, 7, 8, 9, 100, 2, 7, 8, 9, 100, 3, 7, 8, 9, 200, 4, 7, 8, 9};
    auto d = draft(h, 1, 3, 3);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 100);
}

TEST(SuffixDraft, VoteNarrowsToSurvivors) {
    // Both occurrences agree on the first continuation token (20) then
    // diverge (100 vs 200); after the divergence only the surviving branch
    // keeps voting. Majority path: 20, then tie broken by recency → 200.
    std::vector<int32_t> h = {7, 8, 9, 20, 100, 5, 7, 8, 9, 20, 200, 6, 7, 8, 9};
    auto d = draft(h, 3, 3, 3);
    ASSERT_EQ(d.size(), 3u);
    EXPECT_EQ(d[0], 20);
    EXPECT_EQ(d[1], 200);
    EXPECT_EQ(d[2], 6);
}

TEST(SuffixDraft, OverlappingPeriodicPattern) {
    std::vector<int32_t> h = {7, 8, 7, 8, 7, 8};
    auto d = draft(h, 2, 2, 4);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0], 7);
    EXPECT_EQ(d[1], 8);
}

TEST(SuffixDraft, DegenerateParamsRejected) {
    std::vector<int32_t> h = {1, 2, 3, 1, 2, 3};
    auto idx = make_index(h);
    EXPECT_TRUE(idx.draft(0, 0).empty());
    SuffixDraftIndex empty_idx(3, 8);
    empty_idx.append(nullptr, 5);
    EXPECT_EQ(empty_idx.size(), 0);
    EXPECT_TRUE(empty_idx.draft(4, 4).empty());
}

TEST(SuffixDraft, IncrementalAppendMatchesBulkBuild) {
    // Building the index token-by-token (the engine's steady state) must
    // produce the same drafts as one bulk append — boundary grams included.
    std::vector<int32_t> h = {10, 11, 12, 13, 14, 99, 42, 10, 11, 12, 13, 14, 99, 10, 11, 12};
    auto bulk = make_index(h, 3, 8);
    SuffixDraftIndex inc(3, 8);
    for (const int32_t t : h)
        inc.append(&t, 1);
    ASSERT_EQ(bulk.size(), inc.size());
    int s1 = -1, s2 = -1;
    auto d1 = bulk.draft(8, 8, &s1);
    auto d2 = inc.draft(8, 8, &s2);
    EXPECT_EQ(d1, d2);
    EXPECT_EQ(s1, s2);
    ASSERT_FALSE(d1.empty());
    EXPECT_EQ(d1[0], 13);
}

TEST(SuffixDraft, AdaptiveLengthExtendsOnMultiOccurrenceAgreement) {
    // The continuation [20..29] appears after [7,8,9] twice and both
    // occurrences agree: the draft may extend past base k (4) up to k_max.
    std::vector<int32_t> h;
    for (int rep = 0; rep < 2; ++rep) {
        h.insert(h.end(), {7, 8, 9});
        for (int i = 0; i < 10; ++i)
            h.push_back(20 + i);
        h.push_back(90 + rep);  // diverge after the shared run
    }
    h.insert(h.end(), {7, 8, 9});
    auto d = draft(h, 4, 3, 3, /*k_max=*/16);
    ASSERT_EQ(d.size(), 10u);  // unanimous shared run, stops at divergence
    for (int i = 0; i < 10; ++i)
        EXPECT_EQ(d[i], 20 + i);
}

TEST(SuffixDraft, AdaptiveLengthStopsAtKForSingleWeakMatch) {
    // A single occurrence with a minimal (min_match-length) context match
    // is weak evidence: the draft must stop at base k.
    std::vector<int32_t> h = {5, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 6, 7, 8, 9};
    auto d = draft(h, 4, 3, 8, /*k_max=*/16);
    ASSERT_EQ(d.size(), 4u);
    EXPECT_EQ(d[0], 20);
    EXPECT_EQ(d[3], 23);
}

TEST(SuffixDraft, AdaptiveLengthExtendsOnMaximalContextMatch) {
    // A single occurrence whose backward match saturates max_match (the
    // prediction-region case) is strong evidence: extend to k_max.
    std::vector<int32_t> h;
    for (int i = 0; i < 20; ++i)
        h.push_back(100 + i);  // "prediction"
    h.push_back(999);
    for (int i = 0; i < 8; ++i)
        h.push_back(100 + i);  // completion tracks it
    auto d = draft(h, 4, 3, /*max_match=*/6, /*k_max=*/10);
    ASSERT_EQ(d.size(), 10u);
    for (int i = 0; i < 10; ++i)
        EXPECT_EQ(d[i], 108 + i);
}

TEST(SuffixDraft, DraftStartClassifiesSourceRegion) {
    std::vector<int32_t> h = {10, 11, 12, 13, 14, 99, 10, 11, 12};
    int start = -1;
    auto idx = make_index(h);
    auto d = idx.draft(4, 4, &start);
    ASSERT_FALSE(d.empty());
    EXPECT_EQ(start, 3);  // one past the matched [10,11,12] occurrence
}

}  // namespace
