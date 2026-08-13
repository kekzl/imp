// Which MoE expert placements can actually be served (2026-08-13).
//
// An NVFP4-prequant expert that stays on host reaches the generic cuBLAS path
// with scales == nullptr, where gemm() logs an ERROR and returns WITHOUT
// multiplying. The forward continues, the missing experts contribute zero, and
// the process exits 0 with a wrong answer. Reproduced on
// Qwen3-30B-A3B-NVFP4-Modelopt: 8 of 48 layers on host answers "the capital of
// France is the city of the same name, France itself"; all 48 repeats "ftp".
//
// The load must therefore be refused where the placement is decided. These
// tests pin the predicate that decides it. CPU-only by construction: it is
// pure, and takes the placement as data rather than reading a checkpoint.

#include <gtest/gtest.h>

#include "model/expert_placement.h"

using namespace imp;

namespace {

// A 4-layer model whose layers 1 and 3 carry experts (0 and 2 are dense).
// Interleaving matters: a predicate that scans `experts_upload_layer` alone,
// without gating on "is this an MoE layer at all", reads the dense layers'
// false and refuses every model. That mutant passes a uniform-layer fixture.
constexpr size_t kExpertBytes = 512ull * 1024 * 1024;

std::vector<size_t> interleaved_costs() { return {0, kExpertBytes, 0, kExpertBytes}; }

}  // namespace

TEST(ExpertPlacement, Nvfp4WithEveryExpertLayerResidentIsServeable) {
    // Dense layers are false here and must not be read as "host-resident".
    const std::vector<bool> upload = {false, true, false, true};
    EXPECT_TRUE(expert_placement_is_serveable(true, interleaved_costs(), upload));
    EXPECT_EQ(expert_placement_host_layers(interleaved_costs(), upload), 0);
}

TEST(ExpertPlacement, Nvfp4WithOneExpertLayerOnHostIsRefused) {
    // The 8-of-48 arm above, reduced: one layer short is already wrong output,
    // so the threshold is one, not "most of them".
    const std::vector<bool> upload = {false, true, false, false};
    EXPECT_FALSE(expert_placement_is_serveable(true, interleaved_costs(), upload));
    EXPECT_EQ(expert_placement_host_layers(interleaved_costs(), upload), 1);
}

TEST(ExpertPlacement, Nvfp4WithEveryExpertLayerOnHostIsRefused) {
    const std::vector<bool> upload = {false, false, false, false};
    EXPECT_FALSE(expert_placement_is_serveable(true, interleaved_costs(), upload));
    EXPECT_EQ(expert_placement_host_layers(interleaved_costs(), upload), 2);
}

// The refusal must be scoped to the weight format that lacks a host path.
// GGUF-class experts got one in #1370 and are measured at 48.3 tok/s with all
// 48 layers host-resident, so refusing them would delete a working feature.
TEST(ExpertPlacement, GgufClassExpertsOnHostStayServeable) {
    const std::vector<bool> upload = {false, false, false, false};
    EXPECT_TRUE(expert_placement_is_serveable(false, interleaved_costs(), upload));
}

// A dense model has no expert layers at all: every cost is zero, so nothing
// can be host-resident and the NVFP4 flag must not refuse it.
TEST(ExpertPlacement, DenseNvfp4ModelIsServeable) {
    const std::vector<size_t> costs = {0, 0, 0, 0};
    const std::vector<bool> upload = {false, false, false, false};
    EXPECT_TRUE(expert_placement_is_serveable(true, costs, upload));
    EXPECT_EQ(expert_placement_host_layers(costs, upload), 0);
}

// Callers build the two vectors separately (`compute_expert_layer_costs_`
// sizes one, the caller sizes the other from n_layers). A predicate that
// indexed by the longer of the two would read out of bounds; pin that it
// stops at the shorter.
TEST(ExpertPlacement, MismatchedLengthsStopAtTheShorter) {
    const std::vector<size_t> costs = {0, kExpertBytes, kExpertBytes};
    const std::vector<bool> upload = {false, true};
    EXPECT_TRUE(expert_placement_is_serveable(true, costs, upload));
    EXPECT_EQ(expert_placement_host_layers(costs, upload), 0);
}
