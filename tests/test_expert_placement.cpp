// Which MoE expert placements depend on the NVFP4 host-offload path (2026-08-13).
//
// History: an NVFP4-prequant expert that stayed on host used to reach the
// generic cuBLAS path with scales == nullptr, where gemm() logged an ERROR and
// returned WITHOUT multiplying. The forward continued, the missing experts
// contributed zero, and the process exited 0 with a wrong answer. Reproduced on
// Qwen3-30B-A3B-NVFP4-Modelopt: 8 of 48 layers on host answered "the capital of
// France is the city of the same name, France itself"; all 48 repeated "ftp".
// #1403 refused that placement outright.
//
// The path exists now (see exec/nvfp4_expert_offload.h), so this predicate no
// longer decides servability — it decides whether the placement DEPENDS on that
// path. The refusal moved to verify_host_expert_placement(), which runs once
// the expert cache is sized, because that is the fact this cannot see.
//
// These tests pin the predicate. CPU-only by construction: it is pure, and
// takes the placement as data rather than reading a checkpoint.

#include <gtest/gtest.h>

#include "model/expert_placement.h"

using namespace imp;

namespace {

// A 4-layer model whose layers 1 and 3 carry experts (0 and 2 are dense).
// Interleaving matters: a predicate that scans `experts_upload_layer` alone,
// without gating on "is this an MoE layer at all", reads the dense layers'
// false and claims every model needs the path. That mutant passes a
// uniform-layer fixture.
constexpr size_t kExpertBytes = 512ull * 1024 * 1024;

std::vector<size_t> interleaved_costs() { return {0, kExpertBytes, 0, kExpertBytes}; }

}  // namespace

TEST(ExpertPlacement, Nvfp4WithEveryExpertLayerResidentNeedsNoHostPath) {
    // Dense layers are false here and must not be read as "host-resident".
    const std::vector<bool> upload = {false, true, false, true};
    EXPECT_FALSE(expert_placement_needs_host_path(true, interleaved_costs(), upload));
    EXPECT_EQ(expert_placement_host_layers(interleaved_costs(), upload), 0);
}

TEST(ExpertPlacement, Nvfp4WithOneExpertLayerOnHostNeedsTheHostPath) {
    // The 8-of-48 arm above, reduced: one layer short already changed the
    // output, so the threshold is one, not "most of them".
    const std::vector<bool> upload = {false, true, false, false};
    EXPECT_TRUE(expert_placement_needs_host_path(true, interleaved_costs(), upload));
    EXPECT_EQ(expert_placement_host_layers(interleaved_costs(), upload), 1);
}

TEST(ExpertPlacement, Nvfp4WithEveryExpertLayerOnHostNeedsTheHostPath) {
    const std::vector<bool> upload = {false, false, false, false};
    EXPECT_TRUE(expert_placement_needs_host_path(true, interleaved_costs(), upload));
    EXPECT_EQ(expert_placement_host_layers(interleaved_costs(), upload), 2);
}

// GGUF-class experts have their own host path (#1370), measured at 48.3 tok/s
// with all 48 layers host-resident. They must not be routed through the NVFP4
// one, which addresses a different slot layout.
TEST(ExpertPlacement, GgufClassExpertsDoNotUseTheNvfp4HostPath) {
    const std::vector<bool> upload = {false, false, false, false};
    EXPECT_FALSE(expert_placement_needs_host_path(false, interleaved_costs(), upload));
}

// A dense model has no expert layers at all: every cost is zero, so nothing
// can be host-resident and the NVFP4 flag must not claim it needs the path.
TEST(ExpertPlacement, DenseNvfp4ModelNeedsNoHostPath) {
    const std::vector<size_t> costs = {0, 0, 0, 0};
    const std::vector<bool> upload = {false, false, false, false};
    EXPECT_FALSE(expert_placement_needs_host_path(true, costs, upload));
    EXPECT_EQ(expert_placement_host_layers(costs, upload), 0);
}

// Callers build the two vectors separately (`compute_expert_layer_costs_`
// sizes one, the caller sizes the other from n_layers). A predicate that
// indexed by the longer of the two would read out of bounds; pin that it
// stops at the shorter.
TEST(ExpertPlacement, MismatchedLengthsStopAtTheShorter) {
    const std::vector<size_t> costs = {0, kExpertBytes, kExpertBytes};
    const std::vector<bool> upload = {false, true};
    EXPECT_FALSE(expert_placement_needs_host_path(true, costs, upload));
    EXPECT_EQ(expert_placement_host_layers(costs, upload), 0);
}
