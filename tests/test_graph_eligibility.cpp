// Graph-demotion reasons (audit finding F-14).
//
// Eight sites across four TUs can turn CUDA graphs off. They now route through
// Engine::demote_graphs_(), which records the reason so the resolved-dispatch
// summary can print `graphs=0(mamba2_ssm_layers)` instead of a bare `graphs=0`.
//
// That readback is only worth anything if every reason has a name: a new
// enumerator without a switch arm prints "?" and turns the informative half of
// the line back into noise. That is the failure this file catches.
//
// CPU-only by design — runtime/graph_eligibility.h is deliberately free of CUDA,
// RuntimeConfig and Engine so this can run in the unit lane.

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
TEST(GraphEligibility, KvPressureCountsReclaimableBlocksAsFree) {
    EXPECT_FALSE(kv_pressure_demotes_graphs(/*free=*/0, /*reclaimable=*/984, /*total=*/3000));
    EXPECT_TRUE(kv_pressure_demotes_graphs(0, 0, 3000));
    EXPECT_TRUE(kv_pressure_demotes_graphs(299, 0, 3000));
    EXPECT_FALSE(kv_pressure_demotes_graphs(300, 0, 3000)) << "a tenth exactly is not under a tenth";
    EXPECT_FALSE(kv_pressure_demotes_graphs(0, 0, 0)) << "no pool, no pressure";
    EXPECT_FALSE(kv_pressure_demotes_graphs(0, 0, 9)) << "total/10 == 0 can never be undercut";
}

// The way back (C-3): a fifth free lifts it, an eviction pins it.
TEST(GraphEligibility, KvPressureLiftsAtAFifthUnlessSomethingWasEvicted) {
    EXPECT_FALSE(kv_pressure_repromotes_graphs(299, 0, 3000, 0)) << "still under the trigger";
    EXPECT_FALSE(kv_pressure_repromotes_graphs(599, 0, 3000, 0)) << "between the two lines: hysteresis";
    EXPECT_TRUE(kv_pressure_repromotes_graphs(600, 0, 3000, 0));
    EXPECT_TRUE(kv_pressure_repromotes_graphs(0, 600, 3000, 0)) << "reclaimable counts on the way back too";
    EXPECT_FALSE(kv_pressure_repromotes_graphs(3000, 0, 3000, 1)) << "one evicted block pins the demotion";
    EXPECT_FALSE(kv_pressure_repromotes_graphs(0, 0, 0, 0));
}

// The valve is F16-only, and on every other KV dtype the same 90 % pressure
// used to fall through an `if` with no log at all. The next thing an operator
// saw was the hard cancel in decode_prepare_kv_ once the pool ran dry.
TEST(GraphEligibility, QuantizedKvPressureWarnsBecauseThereIsNoValve) {
    // Qwen3.8-27B-NVFP4 resolves FP8 KV by default: same pool state, no valve.
    EXPECT_TRUE(kv_pressure_warns_no_streaming_valve(0, 0, 3000, /*kv_is_f16=*/false));
    EXPECT_TRUE(kv_pressure_warns_no_streaming_valve(299, 0, 3000, false));

    // F16 has the valve; it arms StreamingLLM and logs that. Two lines for one
    // event would be worse than none.
    EXPECT_FALSE(kv_pressure_warns_no_streaming_valve(0, 0, 3000, /*kv_is_f16=*/true));

    // Below the trigger there is nothing to say, on either dtype.
    EXPECT_FALSE(kv_pressure_warns_no_streaming_valve(300, 0, 3000, false))
        << "a tenth exactly is not under a tenth";
    EXPECT_FALSE(kv_pressure_warns_no_streaming_valve(0, 984, 3000, false))
        << "reclaimable prefix-cache blocks are free for this purpose (#1879)";
    EXPECT_FALSE(kv_pressure_warns_no_streaming_valve(0, 0, 0, false)) << "no pool, no pressure";
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

// The invariant both snapshot savers have to hold: what a save stores must be
// something a restore can still admit. The restore caps at (n - 1) / bs blocks
// and matches the stored length exactly, so a save one block too long is a
// snapshot that is never found. The SWA saver floored the full length and lost
// every block-aligned prompt that way until 2026-09-12; the hybrid one had the
// same bug fixed in #1960.
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
