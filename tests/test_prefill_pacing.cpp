// Prefill pacing under decode (runtime.prefill_cap_fairness).
//
// runtime/prefill_pacing.h is free of CUDA, RuntimeConfig and Engine so the
// cap-by-ratio table runs in the unit lane. The row that matters most is the
// protection scenario: many decoders, one ingest, the #1643 schedule must come
// out unchanged.
#include "runtime/prefill_pacing.h"

#include <gtest/gtest.h>

using namespace imp;

TEST(PrefillPacing, NobodyDecodingIsUncapped) {
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 32, 0, true), 0);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 32, 0, false), 0);
}

TEST(PrefillPacing, CapOffStaysOff) {
    EXPECT_EQ(paced_prefill_cap(0, 2048, 32, 1, true), 0);
    EXPECT_EQ(paced_prefill_cap(-1, 2048, 32, 1, true), 0);
}

TEST(PrefillPacing, FairnessOffKeepsTheCap) {
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 31, 1, false), 1024);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 1, 31, false), 1024);
}

TEST(PrefillPacing, ProtectionScenarioIsUnchanged) {
    // 31 decoders, one ingest (#1643), and the even split.
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 1, 31, true), 1024);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 16, 16, true), 1024);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 15, 16, true), 1024);
}

TEST(PrefillPacing, BurstScalesByRatioUpToTheFullChunk) {
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 31, 1, true), 2048);
    EXPECT_EQ(paced_prefill_cap(1024, 2048, 20, 12, true), 1706);
    EXPECT_EQ(paced_prefill_cap(1024, 4096, 31, 1, true), 4096);
    EXPECT_EQ(paced_prefill_cap(1024, 4096, 3, 1, true), 3072);
}

TEST(PrefillPacing, FullChunkBelowTheCapReturnsTheCap) {
    // A hybrid's executor max (2048) under a 4096 cap: the cap never bites,
    // the caller compares against the chunk.
    EXPECT_EQ(paced_prefill_cap(4096, 2048, 31, 1, true), 4096);
}
