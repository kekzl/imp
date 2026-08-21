// Host-side tests for the prompt-lookup n-gram draft matcher
// (src/runtime/ngram_draft.cpp) used by speculative decoding.

#include "runtime/ngram_draft.h"

#include <gtest/gtest.h>
#include <vector>

using imp::ngram_draft;

namespace {

std::vector<int32_t> draft(const std::vector<int32_t>& hist, int k, int min_match = 3,
                           int max_match = 8) {
    return ngram_draft(hist, k, min_match, max_match).tokens;
}

TEST(NgramDraft, EmptyWhenNoRepeat) {
    EXPECT_TRUE(draft({1, 2, 3, 4, 5, 6, 7, 8}, 4).empty());
}

TEST(NgramDraft, EmptyWhenTooShort) {
    EXPECT_TRUE(draft({1, 2, 3}, 4).empty());
    EXPECT_TRUE(draft({}, 4).empty());
}

TEST(NgramDraft, FindsSimpleRepeat) {
    // Suffix [10,11,12] occurred earlier; the draft is what followed it.
    std::vector<int32_t> h = {10, 11, 12, 13, 14, 99, 10, 11, 12};
    auto d = draft(h, 4);
    ASSERT_EQ(d.size(), 4u);
    EXPECT_EQ(d[0], 13);
    EXPECT_EQ(d[1], 14);
    EXPECT_EQ(d[2], 99);
    EXPECT_EQ(d[3], 10);
}

TEST(NgramDraft, TruncatesAtK) {
    std::vector<int32_t> h = {1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 1, 2, 3};
    auto d = draft(h, 2);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0], 4);
    EXPECT_EQ(d[1], 5);
}

TEST(NgramDraft, PrefersLongerMatch) {
    // Two occurrences of suffix [7,8,9]: the earlier one extends to a
    // 4-gram match [6,7,8,9]; the later only matches 3. Longer must win
    // even though it is older.
    std::vector<int32_t> h = {6, 7, 8, 9, 100, 101, 5, 7, 8, 9, 200, 201, 6, 7, 8, 9};
    auto d = draft(h, 2, 3, 8);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0], 100);
    EXPECT_EQ(d[1], 101);
}

TEST(NgramDraft, PrefersRecentOnEqualLength) {
    // Suffix [7,8,9] twice with different continuations and equal match
    // length (no 4-gram extension anywhere): the more recent wins.
    std::vector<int32_t> h = {1, 7, 8, 9, 100, 2, 7, 8, 9, 200, 3, 7, 8, 9};
    auto d = draft(h, 1, 3, 3);
    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0], 200);
}

TEST(NgramDraft, OverlappingPeriodicPattern) {
    // "A B A B A B" → suffix [A,B] (min_match 2) matches the overlapping
    // earlier occurrence; the continuation is the periodic extension.
    std::vector<int32_t> h = {7, 8, 7, 8, 7, 8};
    auto d = draft(h, 2, 2, 4);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0], 7);
    EXPECT_EQ(d[1], 8);
}

TEST(NgramDraft, DegenerateParamsRejected) {
    std::vector<int32_t> h = {1, 2, 3, 1, 2, 3};
    // The old signature took (pointer, length) and had to defend against a
    // null pointer carrying a length of 6. A span cannot be in that state, so
    // the empty case is the only one left to check.
    EXPECT_TRUE(ngram_draft({}, 4, 3, 8).empty());
    EXPECT_TRUE(draft(h, 0).empty());
    EXPECT_TRUE(draft(h, 4, 0, 8).empty());
}

}  // namespace
