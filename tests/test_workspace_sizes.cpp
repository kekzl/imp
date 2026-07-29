// exec_t2_demand / exec_max_tokens — the arithmetic the T2 arena is sized from
// (docs/MEMORY_ARCHITECTURE.md A7 step 4b).
//
// This is the #1103 failure class in miniature: under-reserve here and the
// pre-dequant cache build expands into the space the arena should have held,
// the card reaches 0.0 MiB free, and WSL2/WDDM starts spilling to host memory
// at a 6.5x bandwidth penalty. The header has claimed since it was written that
// this code is "pure and CUDA-free so it can be unit-tested on the CPU lane" —
// which was true and untested. These pin the three behaviours that were each
// discovered the expensive way.

#include <gtest/gtest.h>

#include "exec/workspace_sizes.h"

using namespace imp;

namespace {

constexpr size_t kMiB = 1024ull * 1024;

// A plain dense model: one weight, nothing exotic.
ExecShape dense_shape() {
    ExecShape s;
    s.max_seq_len_cfg = 4096;
    s.d_model = 4096;
    s.d_ff = 12288;
    s.weights = {{4096, 4096}};
    return s;
}

}  // namespace

TEST(ExecMaxTokens, ClampsToFourThousandNinetySix) {
    ExecShape s = dense_shape();
    EXPECT_EQ(exec_max_tokens(s, 131072), 4096);
    EXPECT_EQ(exec_max_tokens(s, 1024), 1024);
}

TEST(ExecMaxTokens, FallsBackToTheConfigWhenNoOverrideIsGiven) {
    ExecShape s = dense_shape();
    s.max_seq_len_cfg = 2048;
    EXPECT_EQ(exec_max_tokens(s, 0), 2048);
    s.max_seq_len_cfg = 0;
    EXPECT_EQ(exec_max_tokens(s, 0), 4096) << "a zero on both sides must not size the arena at 0";
}

// AUDIT B18. executor_workspace.cu reads has_gdn_ before it is assigned, so the
// 2048 cap fires for SSM+MoE and never for pure GDN. This test pins the
// AS-BUILT behaviour deliberately: "fixing" it to the intended condition would
// reserve at T=2048 while the executor still allocates at T=4096 — a 2x
// under-reservation, i.e. exactly the bug this file exists to prevent.
TEST(ExecMaxTokens, ReplicatesTheAsBuiltGdnCapAndNotTheIntendedOne) {
    ExecShape ssm_moe = dense_shape();
    ssm_moe.is_ssm = true;
    ssm_moe.is_moe = true;
    EXPECT_EQ(exec_max_tokens(ssm_moe, 4096), 2048) << "SSM+MoE is the case that fires";

    ExecShape gdn_only = dense_shape();
    gdn_only.is_ssm = true;
    gdn_only.is_moe = false;
    EXPECT_EQ(exec_max_tokens(gdn_only, 4096), 4096)
        << "pure GDN must NOT be capped — the executor allocates at 4096 there, and reserving "
           "2048 would under-provision by 2x (AUDIT B18)";
}

TEST(ExecMaxWeightK, TakesTheLargestLogicalK) {
    ExecShape s = dense_shape();
    s.weights = {{4096, 4096}, {1024, 12288}, {4096, 512}};
    EXPECT_EQ(exec_max_weight_k(s), 12288);
}

TEST(ExecT2Demand, MmvqFollowsTheGemmScratchFormula) {
    ExecShape s = dense_shape();
    s.weights = {{4096, 12288}};
    const ExecT2Demand d = exec_t2_demand(s, 1024);
    // max_tokens * ceil(K/32) * 36 * 2
    const size_t expect = 1024ull * ((12288 + 31) / 32) * 36 * 2;
    EXPECT_EQ(d.mmvq_scratch, expect);
}

TEST(ExecT2Demand, EmptyShapeAsksForNothingRatherThanGuessing) {
    ExecShape s;
    const ExecT2Demand d = exec_t2_demand(s, 4096);
    EXPECT_EQ(d.mmvq_scratch, 0u);
    EXPECT_EQ(d.nvfp4_dequant, 0u);
    EXPECT_EQ(d.total(), 0u);
}

// A dequant target above the 512 MiB cap is served by the uncapturable path, so
// it must not raise the reservation — otherwise every large-weight model would
// reserve half a gigabyte of arena it never takes from.
TEST(ExecT2Demand, TargetsAboveTheCapDoNotRaiseTheReservation) {
    ExecShape s = dense_shape();
    // 4096 x 4096 x 2 = 32 MiB (under the cap), 32768 x 32768 x 2 = 2 GiB (over).
    s.weights = {{4096, 4096}, {32768, 32768}};
    s.d_ff = 0;  // isolate the tensor scan from the config-derived terms
    const ExecT2Demand d = exec_t2_demand(s, 1024);
    EXPECT_EQ(d.nvfp4_dequant, 32 * kMiB) << "the 2 GiB target must be ignored, not clamped to 512";
}

TEST(ExecT2Demand, PicksTheLargestTargetThatStillFitsTheCap) {
    ExecShape s = dense_shape();
    s.weights = {{1024, 1024}, {4096, 4096}, {2048, 2048}};
    s.d_ff = 0;
    const ExecT2Demand d = exec_t2_demand(s, 1024);
    EXPECT_EQ(d.nvfp4_dequant, 32 * kMiB);
}

// AUDIT B23, the gpt-oss case. Its experts arrive pre-upload as a 4D U8
// expert_gate_up_packed_blocks slot, so a tensor scan sees nothing resembling
// the dequant target and the reservation came out 22.5 MiB where 31.64 was
// needed. The real target is one expert's FUSED gate_up (2*expert_d_ff x
// d_model), which only the config knows before the upload.
TEST(ExecT2Demand, ConfigDerivedFusedExpertShapeIsChargedWhenTensorsCannotShowIt) {
    ExecShape s;
    s.max_seq_len_cfg = 4096;
    s.d_model = 2880;
    s.n_experts = 32;
    // expert_d_ff MUST differ from d_ff here, or the dense `2*d_ff x d_model`
    // term produces the same number and the assertion below passes without the
    // expert term contributing anything. Mutation-checked: with d_ff == 2880
    // this test survived deleting the fused-expert charge outright.
    s.expert_d_ff = 2880;
    s.d_ff = 512;
    // Only a small attention weight is visible pre-upload — the experts are not.
    s.weights = {{2880, 2880}};

    const ExecT2Demand d = exec_t2_demand(s, 4096);
    const size_t fused_gate_up = 2ull * 2880 * 2880 * 2;  // 2*expert_d_ff x d_model, fp16
    EXPECT_EQ(d.nvfp4_dequant, fused_gate_up)
        << "the fused gate_up must be charged from the config; a tensor scan alone under-reserves "
           "and the workspace is then refused mid-build (AUDIT B23)";
    EXPECT_GT(d.nvfp4_dequant, 2880ull * 2880 * 2)
        << "and it must exceed what the visible tensors alone would have asked for";
    EXPECT_GT(d.nvfp4_dequant, 2ull * 512 * 2880 * 2)
        << "and it must exceed the dense d_ff term, or this test is not measuring the expert path";
}

TEST(ExecT2Demand, ExpertDFfFallsBackToDFfWhenUnset) {
    ExecShape a;
    a.d_model = 2048;
    a.d_ff = 8192;
    a.n_experts = 8;
    a.expert_d_ff = 0;  // unset
    a.weights = {{2048, 2048}};

    ExecShape b = a;
    b.expert_d_ff = 8192;

    EXPECT_EQ(exec_t2_demand(a, 1024).nvfp4_dequant, exec_t2_demand(b, 1024).nvfp4_dequant);
}

TEST(ExecT2Demand, TotalIsTheSumOfBothTenants) {
    ExecShape s = dense_shape();
    const ExecT2Demand d = exec_t2_demand(s, 1024);
    EXPECT_EQ(d.total(), d.mmvq_scratch + d.nvfp4_dequant);
    EXPECT_GT(d.total(), 0u);
}
