// MoE expert placements depending on the NVFP4 host-offload path (2026-08-13): an NVFP4-
// prequant expert left on host used to reach the generic cuBLAS path with scales==nullptr,
// where gemm() logged an ERROR and returned WITHOUT multiplying - missing experts
// contributed zero and the process exited 0 with a wrong answer (Qwen3-30B-A3B-NVFP4-
// Modelopt: 8/48 host layers answered garbage, all 48 repeated "ftp"). #1403 refused that
// placement outright.
// The path now exists (exec/nvfp4_expert_offload.h), so this predicate decides only whether
// a placement DEPENDS on it; the refusal moved to verify_host_expert_placement(). CPU-only.

#include <gtest/gtest.h>

#include "model/expert_placement.h"

using namespace imp;

namespace {

// 4-layer fixture with experts on layers 1 and 3 (0,2 dense): a predicate that scans
// experts_upload_layer without gating on "is this an MoE layer" would read dense layers
// false and wrongly claim every model needs the path - a mutant a uniform-layer fixture
// would miss.
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

// Callers build the two cost vectors separately (compute_expert_layer_costs_ sizes one, the
// caller sizes the other from n_layers): a predicate indexing by the longer would read out
// of bounds; pins that it stops at the shorter.
TEST(ExpertPlacement, MismatchedLengthsStopAtTheShorter) {
    const std::vector<size_t> costs = {0, kExpertBytes, kExpertBytes};
    const std::vector<bool> upload = {false, true};
    EXPECT_FALSE(expert_placement_needs_host_path(true, costs, upload));
    EXPECT_EQ(expert_placement_host_layers(costs, upload), 0);
}
