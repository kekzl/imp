// Invariant I3: one process graph epoch. A captured graph stores it and refuses to replay under
// another one; after one recapture it replays again. No CUDA call: the runner here is a model of
// the stored-epoch compare every capture site does (cuda_graph.cu, the MTP head, DSpark, Inkling).

#include <gtest/gtest.h>

#include "core/graph_epoch.h"
#include "exec/workspace.h"
#include "runtime/cuda_graph.h"

namespace imp {
namespace {

// The compare of CudaGraphRunner::execute, without a device.
struct ModelRunner {
    uint64_t captured_epoch = 0;
    bool captured = false;
    int captures = 0, replays = 0;

    void step() {
        if (captured && captured_epoch != graph_epoch())
            captured = false;
        if (!captured) {
            captured_epoch = graph_epoch();
            captured = true;
            ++captures;
        }
        ++replays;
    }
};

TEST(GraphEpoch, BumpIsMonotonicAndNeverZero) {
    const uint64_t e0 = graph_epoch();
    EXPECT_NE(e0, 0u) << "0 is the stamp of a runner that captured nothing";
    EXPECT_EQ(graph_epoch(), e0) << "a read does not move it";
    EXPECT_EQ(bump_graph_epoch(), e0 + 1);
    EXPECT_EQ(graph_epoch(), e0 + 1);
    EXPECT_EQ(bump_graph_epoch(), e0 + 2);
}

TEST(GraphEpoch, RunnerRefusesAStaleGraphAndRecapturesOnce) {
    ModelRunner r;
    for (int i = 0; i < 4; ++i)
        r.step();
    EXPECT_EQ(r.captures, 1);
    (void)bump_graph_epoch();
    for (int i = 0; i < 4; ++i)
        r.step();
    EXPECT_EQ(r.captures, 2) << "one recapture per bump, then replays";
    EXPECT_EQ(r.replays, 8);
}

TEST(GraphEpoch, EveryRunnerSeesOneBump) {
    ModelRunner a, b;
    a.step();
    b.step();
    (void)bump_graph_epoch();
    a.step();
    b.step();
    EXPECT_EQ(a.captures, 2);
    EXPECT_EQ(b.captures, 2);
}

TEST(GraphEpoch, WorkspaceGenerationIsTheGraphEpoch) {
    Workspace ws;
    const uint64_t e0 = graph_epoch();
    EXPECT_EQ(ws.generation(), e0);
    ws.bump_generation();  // small-M scratch regrow
    EXPECT_EQ(graph_epoch(), e0 + 1);
    EXPECT_EQ(ws.generation(), e0 + 1);
}

// The regrow rule of the 4 DeviceOwned owners (LoRA, 2 verify scratches, released-source scratch).
TEST(GraphEpoch, RegrowBumpsOnlyWhenALiveAddressWentAway) {
    int a = 0, b = 0;
    const uint64_t e0 = graph_epoch();
    EXPECT_FALSE(bump_graph_epoch_if_moved(nullptr, &a)) << "first allocation: no graph holds an address yet";
    EXPECT_FALSE(bump_graph_epoch_if_moved(nullptr, nullptr)) << "first allocation failed";
    EXPECT_FALSE(bump_graph_epoch_if_moved(&a, &a)) << "the allocator handed the same address back";
    EXPECT_EQ(graph_epoch(), e0) << "3 calls without a move: 0 recaptures";
    EXPECT_TRUE(bump_graph_epoch_if_moved(&a, &b)) << "regrown at another address";
    EXPECT_EQ(graph_epoch(), e0 + 1);
    EXPECT_TRUE(bump_graph_epoch_if_moved(&a, nullptr))
        << "released, the regrow failed: the old address is dead";
    EXPECT_EQ(graph_epoch(), e0 + 2);
}

TEST(GraphEpoch, RegrowWithoutAMoveLeavesEveryRunnerReplaying) {
    int a = 0, b = 0;
    ModelRunner r;
    r.step();
    (void)bump_graph_epoch_if_moved(nullptr, &a);
    (void)bump_graph_epoch_if_moved(&a, &a);
    r.step();
    EXPECT_EQ(r.captures, 1);
    (void)bump_graph_epoch_if_moved(&a, &b);
    r.step();
    r.step();
    EXPECT_EQ(r.captures, 2) << "one recapture for the one move";
}

TEST(GraphEpoch, RunnerWithoutAGraphIsNotReadyAndDoesNotReplay) {
    CudaGraphRunner r;
    EXPECT_FALSE(r.is_ready());
    EXPECT_FALSE(r.replay_only(nullptr)) << "no captured graph: no CUDA call";
    (void)bump_graph_epoch();
    EXPECT_FALSE(r.is_ready());
    r.mark_process_warm();
    EXPECT_TRUE(r.graph_path_available()) << "a bump leaves a warm runner on the capture path";
}

}  // namespace
}  // namespace imp
