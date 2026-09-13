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
        if (spec_ngram_source(s))
            EXPECT_TRUE(spec_any_drafter(s)) << "bits=" << bits;
    }
}

}  // namespace
