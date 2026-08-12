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
    GraphDemotionReason::Gemma4NoGraphs,
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
