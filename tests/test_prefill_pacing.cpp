// Prefill pacing under decode (runtime.prefill_cap_fairness).
//
// runtime/prefill_pacing.h is free of CUDA, RuntimeConfig and Engine so the
// cap-by-ratio table runs in the unit lane. The row that matters most is the
// protection scenario: many decoders, one ingest, the #1643 schedule must come
// out unchanged at every shipped weight.
#include "runtime/prefill_pacing.h"

#include <gtest/gtest.h>

using namespace imp;

TEST(PrefillPacing, NobodyDecodingIsUncapped) {
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 32, 0, 4), 0);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 32, 0, 0), 0);
}

TEST(PrefillPacing, CapOffStaysOff) {
    EXPECT_EQ(paced_prefill_cap(0, 2048, 32, 1, 4), 0);
    EXPECT_EQ(paced_prefill_cap(-1, 2048, 32, 1, 4), 0);
}

TEST(PrefillPacing, WeightZeroKeepsTheCap) {
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 31, 1, 0), 1024);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 1, 31, 0), 1024);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 31, 1, -3), 1024);
}

TEST(PrefillPacing, SymmetricWeightOne) {
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 31, 1, 1), 2048);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 20, 12, 1), 1706);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 16, 16, 1), 1024);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 1, 31, 1), 1024);
}

TEST(PrefillPacing, DefaultWeightFour) {
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 31, 1, 4), 2048);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 8, 16, 4), 2048);  // 32 / 16 = 2x
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 5, 16, 4), 1280);  // 20 / 16
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 4, 16, 4), 1024);  // 16 is not above 16
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 8, 31, 4), 1057);  // 32 / 31, the caller block-rounds
    EXPECT_EQ(paced_prefill_cap(1024, 4096, 31, 1, 4), 4096);
}

TEST(PrefillPacing, ProtectionScenarioIsUnchangedAtEveryWeightBelowTheDecoderCount) {
    // 31 decoders, one ingest (#1643).
    for (int w = 0; w <= 30; w++)
        EXPECT_EQ(paced_prefill_cap(1024, 2048, 1, 31, w), 1024) << "weight " << w;
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 1, 31, 32), 1057);
}

TEST(PrefillPacing, FullChunkBelowTheCapReturnsTheCap) {
    // A hybrid's executor max (2048) under a 4096 cap: the cap never bites,
    // the caller compares against the chunk.
    EXPECT_EQ(paced_prefill_cap(4096, 2048, 31, 1, 4), 4096);
}
