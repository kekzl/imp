// Qwen4Exp PLE n-gram context from a sequence's token history (model/ngram_table.h
// ngram_context_at): the stateless replacement for the executor-wide context that made
// batched decode share one sequence's context.

#include <gtest/gtest.h>

#include "model/ngram_table.h"

#include <vector>

namespace imp {
namespace {

constexpr int32_t kEos = 7;

TEST(NGramContext, SequenceStartIsEos) {
    const std::vector<int32_t> in = {10, 11, 12};
    int32_t ctx[2] = {-1, -1};
    EXPECT_TRUE(ngram_context_at(in.data(), 3, nullptr, 0, 0, 2, kEos, ctx));
    EXPECT_EQ(ctx[0], kEos);
    EXPECT_EQ(ctx[1], kEos);
    EXPECT_TRUE(ngram_context_at(in.data(), 3, nullptr, 0, 1, 2, kEos, ctx));
    EXPECT_EQ(ctx[0], kEos);
    EXPECT_EQ(ctx[1], 10);
}

TEST(NGramContext, SpansPromptAndOutput) {
    const std::vector<int32_t> in = {10, 11, 12};
    const std::vector<int32_t> out = {20, 21};
    int32_t ctx[2] = {};
    // Chunk continuation inside the prompt.
    EXPECT_TRUE(ngram_context_at(in.data(), 3, out.data(), 2, 2, 2, kEos, ctx));
    EXPECT_EQ(ctx[0], 10);
    EXPECT_EQ(ctx[1], 11);
    // First decode row (feeds out[0] at position 3): context = last two prompt tokens.
    EXPECT_TRUE(ngram_context_at(in.data(), 3, out.data(), 2, 3, 2, kEos, ctx));
    EXPECT_EQ(ctx[0], 11);
    EXPECT_EQ(ctx[1], 12);
    // Across the boundary.
    EXPECT_TRUE(ngram_context_at(in.data(), 3, out.data(), 2, 4, 2, kEos, ctx));
    EXPECT_EQ(ctx[0], 12);
    EXPECT_EQ(ctx[1], 20);
    // Inside the output.
    EXPECT_TRUE(ngram_context_at(in.data(), 3, out.data(), 2, 5, 2, kEos, ctx));
    EXPECT_EQ(ctx[0], 20);
    EXPECT_EQ(ctx[1], 21);
}

TEST(NGramContext, PastTheHistoryIsReportedAndEosFilled) {
    const std::vector<int32_t> in = {10, 11, 12};
    int32_t ctx[2] = {};
    EXPECT_FALSE(ngram_context_at(in.data(), 3, nullptr, 0, 4, 2, kEos, ctx));
    EXPECT_EQ(ctx[0], 12);
    EXPECT_EQ(ctx[1], kEos);
    // No history at all: only a sequence start is answerable.
    EXPECT_FALSE(ngram_context_at(nullptr, 0, nullptr, 0, 3, 2, kEos, ctx));
    EXPECT_TRUE(ngram_context_at(nullptr, 0, nullptr, 0, 0, 2, kEos, ctx));
}

TEST(NGramContext, TwoSequencesDoNotShareContext) {
    // The batched decode step builds one context per row from that row's own history.
    const std::vector<int32_t> a_in = {1, 2, 3}, a_out = {4};
    const std::vector<int32_t> b_in = {9, 8}, b_out = {6, 5, 4};
    int32_t ctx[2][2] = {};
    ASSERT_TRUE(ngram_context_at(a_in.data(), 3, a_out.data(), 1, 3, 2, kEos, ctx[0]));
    ASSERT_TRUE(ngram_context_at(b_in.data(), 2, b_out.data(), 3, 4, 2, kEos, ctx[1]));
    EXPECT_EQ(ctx[0][0], 2);
    EXPECT_EQ(ctx[0][1], 3);
    EXPECT_EQ(ctx[1][0], 6);
    EXPECT_EQ(ctx[1][1], 5);
}

}  // namespace
}  // namespace imp
