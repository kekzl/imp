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
