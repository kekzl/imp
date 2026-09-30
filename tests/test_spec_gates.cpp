// Regression: speculative.ngram=false used to disable MTP outright, because the shared
// verify-step entry gate asked the n-gram question; with mtp_k=2 the head drafted nothing and
// nothing logged it (found via nsys kernel-instance-count comparison). CPU tests on purpose:
// the rule was an inline Engine method unreachable from any CI lane.

#include "runtime/spec_gates.h"

#include <gtest/gtest.h>

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

// Linear chunk [t0, d1..dK]: the slab ends at row K. A stop at row j < K leaves it ahead.
TEST(SpecGates, HybridSlabMatchesOnlyAtTheCommittedRow) {
    const int K = 16;
    for (int j = 0; j <= K; ++j)
        EXPECT_EQ(imp::spec_hybrid_slab_at_row(false, 0, j, K, 0), j == K) << "j=" << j;
}

// Grouped chunk: candidate 0 runs on the live slot and commits at its group's last row;
// any other winner leaves candidate 0's state there.
TEST(SpecGates, HybridGroupedSlabIsCandidateZerosLastRow) {
    const int rows = 4;  // 1 + mc_depth
    EXPECT_TRUE(imp::spec_hybrid_slab_at_row(true, 0, rows - 1, rows - 1, rows));
    EXPECT_FALSE(imp::spec_hybrid_slab_at_row(true, 0, rows - 2, rows - 1, rows));
    EXPECT_FALSE(imp::spec_hybrid_slab_at_row(true, 1, rows - 1, rows - 1, rows));
}

}  // namespace
