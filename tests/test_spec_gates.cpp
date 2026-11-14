// Regression: speculative.ngram=false used to disable MTP outright, because the shared
// verify-step entry gate asked the n-gram question; with mtp_k=2 the head drafted nothing and
// nothing logged it (found via nsys kernel-instance-count comparison). CPU tests on purpose:
// the rule was an inline Engine method unreachable from any CI lane.

#include "runtime/spec_accept.h"
#include "runtime/spec_gates.h"
#include "runtime/think_stop_logic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using imp::spec_any_drafter;
using imp::spec_ngram_source;
using imp::SpecDrafterState;

namespace {

SpecDrafterState state(bool ngram, bool mtp, bool recycling, bool capable = true) {
    SpecDrafterState s;
    s.ngram_on = ngram;
    s.mtp_on = mtp;
    s.recycling_on = recycling;
    s.model_capable = capable;
    return s;
}

// The regression itself: MTP must reach the verify step with n-gram off.
TEST(SpecGates, MtpDraftsWithNgramOff) {
    EXPECT_TRUE(spec_any_drafter(state(false, true, false)));
}

// ...and the n-gram flag must still mean something while it does.
TEST(SpecGates, NgramSourceStaysOffWhenOnlyMtpIsOn) {
    EXPECT_FALSE(spec_ngram_source(state(false, true, false)));
}

TEST(SpecGates, TokenRecyclingAloneEntersTheVerify) {
    EXPECT_TRUE(spec_any_drafter(state(false, false, true)));
    EXPECT_FALSE(spec_ngram_source(state(false, false, true)));
}

TEST(SpecGates, DSparkAloneEntersTheVerify) {
    SpecDrafterState s = state(false, false, false);
    s.dspark_on = true;
    EXPECT_TRUE(spec_any_drafter(s));
    EXPECT_FALSE(spec_ngram_source(s));
    s.model_capable = false;
    EXPECT_FALSE(spec_any_drafter(s)) << "an incapable model never speculates";
}

TEST(SpecGates, NgramAloneIsUnaffected) {
    EXPECT_TRUE(spec_any_drafter(state(true, false, false)));
    EXPECT_TRUE(spec_ngram_source(state(true, false, false)));
}

// The reverse control: with nothing enabled the step must not be entered, or
// "any drafter" would be true for everyone and the predicate would say nothing.
TEST(SpecGates, NothingEnabledDraftsNothing) {
    EXPECT_FALSE(spec_any_drafter(state(false, false, false)));
    EXPECT_FALSE(spec_ngram_source(state(false, false, false)));
}

// Model capability overrules every flag: a model that cannot speculate must
// not look enabled to anyone, or the decode loop chops itself into bursts for
// drafts that can never happen (#1299).
TEST(SpecGates, IncapableModelOverridesEveryFlag) {
    for (int bits = 0; bits < 8; ++bits) {
        const SpecDrafterState s =
            state(bits & 1, bits & 2, bits & 4, /*capable=*/false);
        EXPECT_FALSE(spec_any_drafter(s)) << "bits=" << bits;
        EXPECT_FALSE(spec_ngram_source(s)) << "bits=" << bits;
    }
}

// spec_ngram_source must never be true where spec_any_drafter is false: the
// matcher runs strictly inside the step it is allowed to enter.
TEST(SpecGates, NgramSourceImpliesTheVerifyIsEntered) {
    for (int bits = 0; bits < 16; ++bits) {
        const SpecDrafterState s = state(bits & 1, bits & 2, bits & 4, bits & 8);
        if (spec_ngram_source(s)) {
            EXPECT_TRUE(spec_any_drafter(s)) << "bits=" << bits;
        }
    }
}

// Capture-mode FA2 derives q_offset = kv_len - rows (attention_fmha_sm120.cu). Both replay
// routes must land on q_offset = p0. The graph used to get p0 + 1 + matched: q_offset
// p0 - (chunk_pad - 1 - matched), every replayed row missing its last keys (#2275).
TEST(SpecGates, HybridReplayKeepsQOffsetAtP0) {
    for (int chunk_pad : {3, 9, 17, 33}) {
        for (int matched = 1; matched < chunk_pad - 1; ++matched) {
            const int p0 = 1000;
            const int kv_graph = imp::spec_replay_kv_len(true, p0, chunk_pad, matched);
            const int kv_eager = imp::spec_replay_kv_len(false, p0, chunk_pad, matched);
            EXPECT_EQ(kv_graph - chunk_pad, p0) << "graph pad=" << chunk_pad << " m=" << matched;
            EXPECT_EQ(kv_eager - (matched + 1), p0) << "eager pad=" << chunk_pad << " m=" << matched;
            EXPECT_NE((p0 + 1 + matched) - chunk_pad, p0) << "old graph kv_len is the defect";
        }
    }
}

// A JSON / schema / grammar / tool-constrained request never verifies: binding it put Qwen3.8-27B
// json_schema turns on the eager MTP path, 92.6 -> 69.0 tok/s.
TEST(SpecGates, MtpBindsOnlyARequestThatCanVerify) {
    EXPECT_TRUE(imp::mtp_binds_request({.mtp_armed = true}));
    EXPECT_FALSE(imp::mtp_binds_request({.mtp_armed = true, .constrained = true}));
    EXPECT_FALSE(imp::mtp_binds_request({.mtp_armed = true, .sampling_refused = true}));
    EXPECT_FALSE(imp::mtp_binds_request({.mtp_armed = false}));
}

// Linear chunk [t0, d1..dK]: the slab ends at row K. A stop at row j < K leaves it ahead.
TEST(SpecGates, HybridSlabMatchesOnlyAtTheCommittedRow) {
    const int K = 16;
    for (int j = 0; j <= K; ++j)
        EXPECT_EQ(imp::spec_hybrid_slab_at_row(j, K), j == K) << "j=" << j;
}

// Block drafter economics. Measured Qwen3-8B-NVFP4 prose: the n-gram rule stopped DSpark after 12
// verifies (12 of 84 drafts accepted = 14 %, 2.00 tok/verify); priced per verify it ran on to 109 verifies
// at 2.34 tok/verify, 348 vs 282 tok/s nospec.
TEST(SpecGates, BlockDrafterIsPricedPerVerify) {
    imp::SpecEconState poor{12, 84, 12, 24, true, 1.8f};
    EXPECT_FALSE(imp::spec_economics(poor).stop);
    EXPECT_FALSE(imp::spec_economics({109, 763, 146, 255, true, 1.8f}).stop);
    poor.block_drafter = false;
    EXPECT_TRUE(imp::spec_economics(poor).stop);
    EXPECT_TRUE(imp::spec_economics(poor).final);
    poor.block_drafter = true;
    poor.min_emit = 0.0f;
    EXPECT_TRUE(imp::spec_economics(poor).stop) << "min_emit 0 = the n-gram rule";
    const imp::SpecEconState low{12, 84, 9, 21, true, 1.8f};  // 1.75 tok/verify
    EXPECT_TRUE(imp::spec_economics(low).stop);
    EXPECT_FALSE(imp::spec_economics(low).final) << "a block drafter re-arms after the next burst";
    const imp::SpecEconState early{7, 49, 0, 7, true, 1.8f};
    EXPECT_FALSE(imp::spec_economics(early).stop) << "no verdict before 8 verifies";
}

// A model of the engine side of spec_accept_rows: think state as track_think_state keeps it,
// the KV rows the rollback keeps, and what the next step forwards.
constexpr int32_t kOpen = 900, kClose = 901, kEos = 902;

struct AcceptRun {
    imp::SpecAccept r;
    bool finished = false;
    bool in_think = false;
    std::vector<int32_t> out;
};

AcceptRun accept(const std::vector<int32_t>& argmax, const std::vector<int32_t>& draft, bool in_think,
                 bool budgeted, int masked_row = -1, int think_left = imp::think_logic::kThinkUnbounded) {
    AcceptRun a;
    a.in_think = in_think;
    const auto emit = [&](int, int32_t tok) {
        a.out.push_back(tok);
        if (tok == kOpen)
            a.in_think = true;
        if (tok == kClose)
            a.in_think = false;
        a.finished = tok == kEos;
        return a.finished;
    };
    const int nd = static_cast<int>(draft.size());
    a.r = imp::spec_accept_rows(argmax.data(), draft.data(), nd, nd + 1, budgeted, a.in_think, think_left,
                                emit, [&](int j) { return j + 1 != masked_row; });
    return a;
}

// Every argmax/draft pattern over {plain, <think>, </think>, eos} at 1 .. 3 drafts: the kept KV rows are
// t0 + matched drafts, so an unfinished request must hold emitted == matched + 1 (B-68: the think break
// sat behind matched++, kept </think> in KV and forwarded it again as the next t0).
TEST(SpecAcceptRows, UnfinishedChunkLeavesExactlyTheBonusOutsideTheKeptRows) {
    const int32_t alphabet[] = {7, kOpen, kClose, kEos};
    int crossings = 0;
    for (int nd = 1; nd <= 3; ++nd) {
        const int rows = nd + 1;
        int n_am = 1, n_dr = 1;
        for (int i = 0; i < rows; ++i)
            n_am *= 4;
        for (int i = 0; i < nd; ++i)
            n_dr *= 5;
        for (int a = 0; a < n_am; ++a) {
            std::vector<int32_t> am(rows);
            for (int i = 0, v = a; i < rows; ++i, v /= 4)
                am[i] = alphabet[v % 4];
            for (int d = 0; d < n_dr; ++d) {
                std::vector<int32_t> dr(nd);
                for (int i = 0, v = d; i < nd; ++i, v /= 5)
                    dr[i] = (v % 5 == 4) ? 8 : alphabet[v % 5];  // 8 never matches
                for (int flags = 0; flags < 4; ++flags) {
                    const bool in_think = (flags & 1) != 0, budgeted = (flags & 2) != 0;
                    const AcceptRun run = accept(am, dr, in_think, budgeted);
                    const auto& r = run.r;
                    ASSERT_EQ(static_cast<int>(run.out.size()), r.emitted);
                    ASSERT_LE(r.matched, nd);
                    for (int i = 0; i < r.matched; ++i)
                        ASSERT_EQ(am[i], dr[i]) << "an accepted row holds its draft token";
                    if (run.finished) {
                        ASSERT_EQ(run.out.back(), kEos);
                        continue;
                    }
                    ASSERT_EQ(r.emitted, r.matched + 1) << "nd " << nd << " am " << a << " dr " << d;
                    if (budgeted && run.in_think != in_think) {
                        ++crossings;
                        const int32_t edge = in_think ? kClose : kOpen;
                        ASSERT_EQ(run.out.back(), edge) << "the crossing token is the bonus";
                        for (int i = 0; i < r.matched; ++i)
                            ASSERT_NE(am[i], edge) << "no kept KV row holds the crossing token";
                    }
                }
            }
        }
    }
    EXPECT_GT(crossings, 10000) << "the sweep must reach the think edge";
}

TEST(SpecAcceptRows, ThinkEdgeAcceptedDraftStaysTheBonus) {
    // The head drafted </think> and the model agreed: 1 token, 0 kept draft rows.
    AcceptRun run = accept({kClose, 7, 7}, {kClose, 7}, /*in_think=*/true, /*budgeted=*/true);
    EXPECT_EQ(run.r.matched, 0);
    EXPECT_EQ(run.r.emitted, 1);
    // One draft in, </think> second: the draft before it is kept.
    run = accept({7, kClose, 7}, {7, kClose}, true, true);
    EXPECT_EQ(run.r.matched, 1);
    EXPECT_EQ(run.r.emitted, 2);
    // No budget: the edge does not end the chunk.
    run = accept({kClose, 7, 7}, {kClose, 7}, true, /*budgeted=*/false);
    EXPECT_EQ(run.r.matched, 2);
    EXPECT_EQ(run.r.emitted, 3);
    // Started outside, stays outside: full accept plus the bonus.
    run = accept({7, 7, 7}, {7, 7}, false, true);
    EXPECT_EQ(run.r.matched, 2);
    EXPECT_EQ(run.r.emitted, 3);
}

TEST(SpecAcceptRows, MaskedNextRowAndStopEndTheChunk) {
    AcceptRun run = accept({7, 7, 7}, {7, 7}, false, false, /*masked_row=*/2);
    EXPECT_EQ(run.r.matched, 1) << "row 2's token is masked out: row 1's token is the bonus";
    EXPECT_EQ(run.r.emitted, 2);
    run = accept({7, kEos, 7}, {7, kEos}, false, false);
    EXPECT_TRUE(run.finished);
    EXPECT_EQ(run.r.matched, 1);
    EXPECT_EQ(run.r.emitted, 2);
    run = accept({7, 8, 7}, {7, 7}, false, false);
    EXPECT_EQ(run.r.matched, 1) << "the diverging token is emitted as the bonus";
    EXPECT_EQ(run.r.emitted, 2);
}

// B-69.1: a fully accepted chunk of 1 .. 4 drafts that starts gap tokens short of the think budget
// emits min(gap, drafts + 1) tokens, so the closer is due at the eager step's count (was: up to
// drafts tokens past it).
TEST(SpecAcceptRows, ChunkEndsAtTheThinkBudget) {
    namespace tl = imp::think_logic;
    const int max_tokens = 1024;
    const int limit = tl::think_limit(max_tokens, 0.5f);
    ASSERT_GT(limit, 16);
    for (int nd = 1; nd <= 4; ++nd) {
        for (int gap = 1; gap <= nd + 3; ++gap) {
            std::vector<int32_t> out(static_cast<size_t>(limit - gap), 7);  // started in think: all reasoning
            const int left = tl::think_tokens_left(0.5f, kClose, max_tokens, out, kOpen,
                                                   /*started_in_think=*/true);
            ASSERT_EQ(left, gap);
            const std::vector<int32_t> am(static_cast<size_t>(nd) + 1, 7), dr(static_cast<size_t>(nd), 7);
            const AcceptRun run = accept(am, dr, /*in_think=*/true, /*budgeted=*/true, -1, left);
            const int want = gap < nd + 1 ? gap : nd + 1;
            EXPECT_EQ(run.r.emitted, want) << "nd " << nd << " gap " << gap;
            EXPECT_EQ(run.r.matched, want - 1) << "the last token is the bonus";
            out.insert(out.end(), run.out.begin(), run.out.end());
            EXPECT_LE(static_cast<int>(out.size()), limit) << "nd " << nd << " gap " << gap;
            EXPECT_EQ(tl::should_force_think_end(0.5f, kClose, max_tokens, out, kOpen, true), gap <= nd + 1);
        }
    }
    // No budget, no closer id, or outside the block: unbounded.
    const std::vector<int32_t> out(8, 7);
    EXPECT_EQ(tl::think_tokens_left(0.0f, kClose, max_tokens, out, kOpen, true), tl::kThinkUnbounded);
    EXPECT_EQ(tl::think_tokens_left(0.5f, -1, max_tokens, out, kOpen, true), tl::kThinkUnbounded);
    EXPECT_EQ(tl::think_tokens_left(0.5f, kClose, max_tokens, out, kOpen, false), tl::kThinkUnbounded);
    const AcceptRun free_run = accept({7, 7, 7}, {7, 7}, true, true);
    EXPECT_EQ(free_run.r.emitted, 3);
}

}  // namespace
