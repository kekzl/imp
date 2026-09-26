// Graph-demotion reasons (audit F-14): eight sites across four TUs route through
// Engine::demote_graphs_(), which records the reason so resolved-dispatch can print
// graphs=0(mamba2_ssm_layers) instead of a bare graphs=0. A new enumerator without a switch
// arm prints "?" and turns that informative line back into noise - the failure this file
// catches. CPU-only: runtime/graph_eligibility.h is deliberately free of CUDA/RuntimeConfig/Engine.

#include "runtime/graph_eligibility.h"
#include "runtime/snapshot_boundary.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

using namespace imp;

namespace {

// Keep in lock-step with runtime/graph_eligibility.h.
const GraphDemotionReason kAll[] = {
    GraphDemotionReason::None,
    GraphDemotionReason::ConfigNever,
    GraphDemotionReason::CalibrationActive,
    GraphDemotionReason::StreamingKvConfigured,
    GraphDemotionReason::ExpertsOnHost,
    GraphDemotionReason::PinnedSampleBufUnavailable,
    GraphDemotionReason::StreamingKvKvPressure,
};

}  // namespace

// A reason added without a switch arm would print "?" in the dispatch summary.
TEST(GraphEligibility, EveryReasonHasAName) {
    for (GraphDemotionReason r : kAll) {
        const char* name = graph_demotion_reason_name(r);
        ASSERT_NE(name, nullptr);
        EXPECT_STRNE(name, "?") << "GraphDemotionReason value " << static_cast<int>(r)
                                << " has no name — it would print '?' in 'Resolved dispatch: graphs=0(?)'";
        EXPECT_GT(std::string(name).size(), 0u);
    }
}

// Two reasons sharing a name make the summary ambiguous exactly when it matters
// (telling "never eligible" apart from "demoted under pressure").
TEST(GraphEligibility, NamesAreDistinct) {
    std::set<std::string> seen;
    for (GraphDemotionReason r : kAll) {
        std::string name = graph_demotion_reason_name(r);
        EXPECT_TRUE(seen.insert(name).second) << "duplicate reason name: " << name;
    }
}

// The mid-run demotion is a one-way state change taken while requests are in
// flight; the audit asked for it to stay distinguishable from the init-time
// reasons, which describe the model itself.
TEST(GraphEligibility, OnlyKvPressureIsMidRun) {
    for (GraphDemotionReason r : kAll) {
        const bool expected = (r == GraphDemotionReason::StreamingKvKvPressure);
        EXPECT_EQ(graph_demotion_is_mid_run(r), expected)
            << "mid-run classification wrong for " << graph_demotion_reason_name(r);
    }
}

// `None` is the "graphs still on" sentinel — if it ever read as a demotion the
// summary would claim a reason for an engine that never lost graphs.
TEST(GraphEligibility, NoneIsNotMidRunAndNamesItself) {
    EXPECT_FALSE(graph_demotion_is_mid_run(GraphDemotionReason::None));
    EXPECT_STREQ(graph_demotion_reason_name(GraphDemotionReason::None), "none");
}

// The decision itself, not only the enum (AUDIT_arch_2026 C-4). The #1879
// shape: a 3000-block pool with 0 free and 984 reclaimable is NOT exhausted.
// A need of 1000 blocks isolates the 10 % line.
TEST(GraphEligibility, KvPressureCountsReclaimableBlocksAsFree) {
    EXPECT_FALSE(kv_pressure_demotes_graphs(/*free=*/0, /*reclaimable=*/984, /*total=*/3000, /*unmet=*/1000));
    EXPECT_TRUE(kv_pressure_demotes_graphs(0, 0, 3000, 1000));
    EXPECT_TRUE(kv_pressure_demotes_graphs(299, 0, 3000, 1000));
    EXPECT_FALSE(kv_pressure_demotes_graphs(300, 0, 3000, 1000)) << "a tenth exactly is not under a tenth";
    EXPECT_FALSE(kv_pressure_demotes_graphs(0, 0, 0, 1000)) << "no pool, no pressure";
    EXPECT_FALSE(kv_pressure_demotes_graphs(0, 0, 9, 1000)) << "total/10 == 0 can never be undercut";
}

// A nearly full pool is not under pressure while its supply covers what the live
// sequences still need. The 112k-prompt shape on a 7703-block pool: 674 free, 126 needed.
TEST(GraphEligibility, KvPressureIgnoresAPoolThatCoversTheLiveNeed) {
    EXPECT_FALSE(kv_pressure_demotes_graphs(674, 0, 7703, 126)) << "the prompt's own tail fits";
    EXPECT_TRUE(kv_pressure_demotes_graphs(674, 0, 7703, 674)) << "need equal to supply leaves no margin";
    EXPECT_TRUE(kv_pressure_demotes_graphs(100, 0, 7703, 50)) << "50 spare is under 1 % (77)";
    EXPECT_FALSE(kv_pressure_demotes_graphs(87, 0, 7703, 10)) << "exactly 1 % spare is not under it";
    EXPECT_FALSE(kv_pressure_demotes_graphs(0, 0, 35, 0)) << "a full pool that needs nothing more";
    EXPECT_FALSE(kv_pressure_demotes_graphs(2, 0, 35, 2)) << "a small pool covering the need exactly";
    EXPECT_TRUE(kv_pressure_demotes_graphs(1, 0, 24, 38)) << "a generation longer than the pool";
    EXPECT_FALSE(kv_pressure_demotes_graphs(800, 0, 7703, 5000)) << "the 10 % line still gates first";
}

// The valve's need looks kKvValveHorizonTokens ahead, not to max_tokens: a 118676-token
// prompt with max_tokens 12000 on a 7904-block pool needs 32 blocks, not 750.
TEST(GraphEligibility, UnmetBlocksLookOnlyAHorizonAhead) {
    EXPECT_EQ(kv_unmet_blocks(118676, 12000, 7418, 16), (118676 + 511 + 15) / 16 - 7418) << "capped at 512 tokens";
    EXPECT_EQ(kv_unmet_blocks(118676, 12000, 7418, 16), 32);
    EXPECT_EQ(kv_unmet_blocks(160, 400, 10, 16), 35 - 10) << "under the horizon: to max_tokens, minus the last token";
    EXPECT_EQ(kv_unmet_blocks(160, 1, 10, 16), 0) << "the last sampled token needs no slot";
    EXPECT_EQ(kv_unmet_blocks(160, 0, 10, 16), 0) << "nothing left to generate";
    EXPECT_EQ(kv_unmet_blocks(160, 400, 40, 16), 0) << "a table already past the need";
    EXPECT_EQ(kv_unmet_blocks(160, 400, 10, 0), 0) << "no block size, no need";
}

// The way back (C-3): a fifth free lifts it, a live evicted sequence pins it.
TEST(GraphEligibility, KvPressureLiftsAtAFifthUnlessALiveSequenceWasEvicted) {
    EXPECT_FALSE(kv_pressure_repromotes_graphs(299, 0, 3000, 0)) << "still under the trigger";
    EXPECT_FALSE(kv_pressure_repromotes_graphs(599, 0, 3000, 0)) << "between the two lines: hysteresis";
    EXPECT_TRUE(kv_pressure_repromotes_graphs(600, 0, 3000, 0));
    EXPECT_TRUE(kv_pressure_repromotes_graphs(0, 600, 3000, 0)) << "reclaimable counts on the way back too";
    EXPECT_FALSE(kv_pressure_repromotes_graphs(3000, 0, 3000, 1)) << "a live evicted sequence pins the demotion";
    EXPECT_FALSE(kv_pressure_repromotes_graphs(0, 0, 0, 0));
}


// The last burst of a request: 7 tokens left under a miss_burst of 8 must
// launch 7 steps, or the parked runner refuses the rearm (76 + 8 > 83) and
// recaptures. The KV-starved shape (3 reserved under 8) is the same clamp.
TEST(GraphEligibility, BurstStepLimitNeverExceedsTheTokensTheBurstHas) {
    EXPECT_EQ(burst_launch_step_limit(8, 7), 7);
    EXPECT_EQ(burst_launch_step_limit(8, 3), 3) << "KV reserved for 3 tokens bounds the launch";
    EXPECT_EQ(burst_launch_step_limit(8, 12), 8) << "a wider reservation keeps the burst";
    EXPECT_EQ(burst_launch_step_limit(8, 8), 8);
    EXPECT_EQ(burst_launch_step_limit(0, 5), 0) << "0 stays unbounded (the loop runs to max_steps)";
    EXPECT_EQ(burst_launch_step_limit(8, 0), 8) << "nothing to clamp against";
}

// Prefix-cache snapshot boundary: the largest block-aligned prompt position,
// and none at all under the minimum (a 35-token prompt is then one prefill
// chunk instead of 32 + 3 with a sync between, 2026-09-08).
TEST(GraphEligibility, SnapshotBoundaryIsBlockAlignedAndSkipsShortPrompts) {
    EXPECT_EQ(snapshot_boundary(35, 16, 256), 0) << "32 < 256: no snapshot, no split";
    EXPECT_EQ(snapshot_boundary(35, 16, 0), 32) << "0 keeps every block-aligned prompt";
    EXPECT_EQ(snapshot_boundary(257, 16, 256), 256) << "the minimum itself qualifies";
    EXPECT_EQ(snapshot_boundary(256, 16, 256), 0)
        << "an aligned 256-token prompt snapshots at 240 (< min): a restore must leave one token";
    EXPECT_EQ(snapshot_boundary(512, 16, 256), 496)
        << "an aligned prompt snapshots one block short, or the (n-1)/bs admission cap never matches it";
    EXPECT_EQ(snapshot_boundary(513, 16, 256), 512);
    EXPECT_EQ(snapshot_boundary(1082, 16, 256), 1072);
    EXPECT_EQ(snapshot_boundary(1082, 32, 256), 1056) << "block size decides the alignment";
    EXPECT_EQ(snapshot_boundary(15, 16, 0), 0) << "under one block there is nothing to save";
    EXPECT_EQ(snapshot_boundary(35, 0, 0), 0);
}

// Invariant both snapshot savers must hold: what a save stores must be admissible by a
// restore. Restore caps at (n-1)/bs blocks and requires an exact length match, so a save one
// block too long is never found. The SWA saver floored the full length and lost every
// block-aligned prompt's snapshot until 2026-09-12; the hybrid saver had the same bug, fixed
// in #1960.
TEST(GraphEligibility, EverySnapshotBoundaryFitsUnderTheRestoreCap) {
    for (int bs : {16, 32}) {
        for (int n = 1; n <= 2100; ++n) {
            const int saved = snapshot_boundary(n, bs, 0);
            ASSERT_EQ(saved % bs, 0) << "n=" << n << " bs=" << bs;
            ASSERT_LE(saved / bs, (n - 1) / bs)
                << "n=" << n << " bs=" << bs << ": saved " << saved
                << " tokens, which the (n-1)/bs restore cap can never admit";
        }
    }
}
