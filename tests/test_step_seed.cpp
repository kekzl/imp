#include <gtest/gtest.h>
#include "runtime/engine_internal.h"

#include <climits>

namespace imp {
namespace {

// compute_step_seed feeds the host samplers; a negative seed reads as unset and draws with the
// fixed 42u. Unmasked, hash(id) ^ clock went negative whenever clock bit 31 was set (2.1 s of every
// 4.3 s), which a test cannot reach on demand; seed INT_MAX + step takes the same overflow.
TEST(StepSeed, SeededRequestAddsTheOutputCount) {
    Request req;
    req.seed = 7;
    req.output_tokens = {11, 12, 13};
    EXPECT_EQ(engine_internal::compute_step_seed(req), 10);
    req.seed = INT_MAX;
    EXPECT_GE(engine_internal::compute_step_seed(req), 0);
}

// An unseeded request keeps one base: a draft drawn for output index i + lv before the verify step must see
// the seed the verify row gets (speculative.mtp_draft_coupled). hash(id) ^ clock per call gave two bases.
TEST(StepSeed, UnseededRequestKeepsOneBase) {
    Request req;
    req.id = 5;
    const int s0 = engine_internal::compute_step_seed(req);
    for (int i = 0; i < 1000; ++i)
        ASSERT_EQ(engine_internal::compute_step_seed(req), s0);
    req.output_tokens = {1, 2, 3};
    EXPECT_EQ(engine_internal::compute_step_seed(req),
              static_cast<int>((static_cast<unsigned>(s0) + 3u) & 0x7fffffffu));
}

std::shared_ptr<Request> sampled_request(int seed, float temperature, int top_k, size_t n_out) {
    auto r = std::make_shared<Request>();
    r->seed = seed;
    r->temperature = temperature;
    r->top_k = top_k;
    r->top_p = 0.9f;
    r->output_tokens.assign(n_out, 1);
    return r;
}

TEST(DraftRowDraws, LevelRowsTakeTheVerifyRowsSeedAndGreedyRowsTopK1) {
    // reqs[1] is unbound. Bound order {2, 0}: i = 0 is the greedy request, i = 1 the sampled one.
    const std::vector<std::shared_ptr<Request>> reqs = {sampled_request(100, 0.7f, 20, 4),
                                                        sampled_request(900, 1.0f, 40, 9),
                                                        sampled_request(50, 0.0f, 20, 2)};
    const int levels = 4, vocab = 1000;
    const auto rows = engine_internal::draft_row_draws(reqs, {2, 0}, levels, vocab);
    ASSERT_EQ(rows.size(), 8u);
    const auto verify = engine_internal::verify_row_draws({reqs[0]}, levels, vocab);
    ASSERT_EQ(verify.size(), 4u);
    for (int lv = 0; lv < levels; ++lv) {
        EXPECT_EQ(rows[(lv * 2) + 0].top_k, 1) << lv;
        const VerifySampling& r = rows[(lv * 2) + 1];
        EXPECT_EQ(r.top_k, 20) << lv;
        EXPECT_EQ(r.seed0, 104 + lv);
        EXPECT_EQ(r.seed0, verify[lv].seed0);
        EXPECT_FLOAT_EQ(r.temperature, 0.7f);
        EXPECT_FLOAT_EQ(r.top_p, 0.9f);
    }
}

TEST(DraftRowDraws, EmptyWithoutASampledRowTheSamplerHolds) {
    const int vocab = 1000;
    EXPECT_TRUE(engine_internal::draft_row_draws({sampled_request(1, 0.0f, 20, 0)}, {0}, 4, vocab).empty());
    // top_k past SAMPLE_MAX_TOP_K: the row sampler cannot draw it, the request keeps argmax drafts.
    EXPECT_TRUE(
        engine_internal::draft_row_draws({sampled_request(1, 1.0f, SAMPLE_MAX_TOP_K + 1, 0)}, {0}, 4, vocab)
            .empty());
    const auto wrap = engine_internal::draft_row_draws({sampled_request(INT_MAX, 1.0f, 20, 0)}, {0}, 2,
                                                       vocab);
    ASSERT_EQ(wrap.size(), 2u);
    EXPECT_EQ(wrap[0].seed0, INT_MAX);
    EXPECT_EQ(wrap[1].seed0, 0);
}

}  // namespace
}  // namespace imp
