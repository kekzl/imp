// Streaming logprobs read the worker's per-token copy (TokenExtras), never the Request vectors the
// worker keeps appending to: a growing std::vector reallocates under a reader on another thread.

#include "batching_engine.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace {

imp::TokenLogprobInfo info(const std::string& text, float lp) {
    imp::TokenLogprobInfo i;
    i.text = text;
    i.logprob = lp;
    i.top.push_back({7, lp, text});
    return i;
}

TEST(TokenExtras, HandlerKeepsTheWorkersCopyWhileTheRequestGrows) {
    imp::Request r;
    r.logprobs = true;
    r.output_logprobs.push_back(info("Paris", -0.25f));
    r.output_lens.push_back({imp::LogitLensLayer{-3.0f, 9, 4.5f, {{1, -1.0f, "x"}}}});
    ServerRequest sr;
    sr.push_token(42, false, nullptr, 0, token_extras(r, 0));

    // The worker's next steps: the vectors grow far past their capacity and get rewritten.
    const auto* before = r.output_logprobs.data();
    for (int k = 0; k < 4096; ++k)
        r.output_logprobs.push_back(info("pad", -9.0f));
    ASSERT_NE(r.output_logprobs.data(), before);  // reallocated: an old reference would dangle
    r.output_logprobs[0] = info("CHANGED", 0.0f);
    r.output_lens.clear();

    const auto ex = sr.extras_at(0);
    ASSERT_NE(ex, nullptr);
    EXPECT_EQ(ex->lp.text, "Paris");
    EXPECT_FLOAT_EQ(ex->lp.logprob, -0.25f);
    ASSERT_EQ(ex->lp.top.size(), 1u);
    ASSERT_EQ(ex->lens.size(), 1u);
    EXPECT_EQ(ex->lens[0].rank, 9);
}

TEST(TokenExtras, NoCopyWithoutLogprobsOrEntry) {
    imp::Request r;
    r.output_logprobs.push_back(info("a", -1.0f));
    EXPECT_EQ(token_extras(r, 0), nullptr);  // logprobs off
    r.logprobs = true;
    EXPECT_NE(token_extras(r, 0), nullptr);
    EXPECT_EQ(token_extras(r, 1), nullptr);  // no entry yet
    ServerRequest sr;
    sr.push_token(5, false, nullptr, 3, nullptr);
    EXPECT_EQ(sr.extras_at(3), nullptr);
    EXPECT_EQ(sr.extras_at(100), nullptr);
    sr.push_token(6, false, nullptr, 4, token_extras(r, 0));  // positions may skip (stop-token gaps)
    EXPECT_NE(sr.extras_at(4), nullptr);
    EXPECT_EQ(sr.extras_at(3), nullptr);
}

}  // namespace
