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

// The batched-MoE pointer arrays are charged only for a MoE model, and they scale
// with n_experts. Small, but the point of charging them is that the arena is
// SIZED for its tenants rather than absorbing them into slack (A7 step 4b.2).
TEST(ExecT2Demand, MoeArraysAreChargedOnlyForMoeAndScaleWithExperts) {
    ExecShape dense = dense_shape();
    EXPECT_EQ(exec_t2_demand(dense, 1024).moe_arrays, 0u) << "a dense model has none of these";

    ExecShape moe = dense_shape();
    moe.n_experts = 32;
    moe.expert_d_ff = 2880;
    const size_t at32 = exec_t2_demand(moe, 1024).moe_arrays;
    EXPECT_GT(at32, 0u);

    moe.n_experts = 128;
    const size_t at128 = exec_t2_demand(moe, 1024).moe_arrays;
    EXPECT_GT(at128, at32) << "it must follow n_experts";
    // Ten arrays of pointers/floats over 128 experts is kilobytes, not megabytes;
    // a term that came out large would mean the arithmetic drifted from the site.
    EXPECT_LT(at128, 64u * 1024) << "these are pointer and scale arrays, a few KiB at most";
}

// The FP8 reduction scratch is charged only when FP8 prefill is on — which on
// sm_120 it never is ("FP8 prefill: auto -> DISABLED on sm_120"). So the RUNTIME
// path is unreachable on the target and only this arithmetic is verifiable; the
// test exists so the term cannot drift unnoticed against the site it mirrors.
TEST(ExecT2Demand, Fp8ReductionIsChargedOnlyWhenFp8PrefillIsOn) {
    ExecShape s = dense_shape();
    s.n_heads = 32;
    s.head_dim = 128;
    EXPECT_EQ(exec_t2_demand(s, 1024).fp8_reduction, 0u) << "off by default, as on sm_120";

    s.use_fp8_prefill = true;
    const size_t on = exec_t2_demand(s, 1024).fp8_reduction;
    EXPECT_GT(on, 0u);

    // It follows max_tokens x max_dim / 1024, so a longer context grows it.
    ExecShape longer = s;
    EXPECT_GT(exec_t2_demand(longer, 4096).fp8_reduction, exec_t2_demand(longer, 512).fp8_reduction)
        << "the reduction grid is derived from the activation size, which scales with tokens";

    // A per-block absmax array over ~11M elements is tens of KiB, not MiB.
    EXPECT_LT(exec_t2_demand(s, 4096).fp8_reduction, 1024u * 1024);
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

TEST(ExecT2Demand, TotalIsTheSumOfEveryTenant) {
    ExecShape s = dense_shape();
    const ExecT2Demand d = exec_t2_demand(s, 1024);
    EXPECT_EQ(d.total(), d.mmvq_scratch + d.nvfp4_dequant + d.sample_scratch + d.moe_arrays +
                             d.fp8_reduction + d.quant_scratch + d.splitk_scratch);
    EXPECT_GT(d.total(), 0u);
}

// The dp4a staging family. Its max-K scan is NOT exec_max_weight_k's: the site
// reads the raw shape[1]/shape[2] of a narrower tensor list, so ExecShape carries
// it separately and this pins that the term follows THAT number.
TEST(ExecT2Demand, QuantScratchFollowsTheSitesOwnMaxKAndNotTheLogicalOne) {
    ExecShape s = dense_shape();
    s.weights = {{4096, 32768}};  // logical K = 32768, which must NOT be used here
    s.mmvq_max_k = 4096;
    const size_t at_4096 = exec_t2_demand(s, 1024).quant_scratch;
    EXPECT_GT(at_4096, 0u);

    s.mmvq_max_k = 8192;
    EXPECT_GT(exec_t2_demand(s, 1024).quant_scratch, at_4096) << "it must follow the site's max_k";

    // 128 blocks x 8 rows x (48 + 4) bytes + the 4-word mask + alignment.
    ExecShape exact = dense_shape();
    exact.mmvq_max_k = 4096;  // 128 blocks
    exact.max_batch_size = 1;  // rows = max(1, 8) = 8
    const size_t expect = 128ull * 8 * (48 + sizeof(float)) + 4 * sizeof(uint32_t) + 5 * 256;
    EXPECT_EQ(exec_t2_demand(exact, 1024).quant_scratch, expect);

    // Rows follow the batch but cap at 16, so a 64-way server does not inflate a
    // K-sized buffer 8x. The site caps it; a term that did not would over-reserve
    // exactly where VRAM is tightest.
    exact.max_batch_size = 64;
    EXPECT_EQ(exec_t2_demand(exact, 1024).quant_scratch,
              128ull * 16 * (48 + sizeof(float)) + 4 * sizeof(uint32_t) + 5 * 256);
}

// The MoE down projection quantizes top_k expert activations contiguously, so on
// an MoE model that term can exceed max_k/32 — and did on every model the dp4a
// MoE path serves. Sizing the buffer from max_k alone would overrun it.
TEST(ExecT2Demand, QuantScratchTakesTheMoeDownProjectionWhenItIsLarger) {
    ExecShape s = dense_shape();
    s.mmvq_max_k = 2048;  // 64 blocks
    const size_t dense_only = exec_t2_demand(s, 1024).quant_scratch;

    s.n_experts = 128;
    s.n_experts_active = 8;
    s.mmvq_max_expert_down_k = 2048;  // 8 * 64 = 512 blocks, 8x the dense term
    const size_t with_moe = exec_t2_demand(s, 1024).quant_scratch;
    EXPECT_GT(with_moe, dense_only * 4)
        << "top_k * down_k/32 must win over max_k/32 — the site takes the max of the two";

    // And top_k is the multiplier, not a constant.
    s.n_experts_active = 4;
    EXPECT_LT(exec_t2_demand(s, 1024).quant_scratch, with_moe);
}

// The prefill pair only exists for a model with Q4_K/Q5_K dense weights (the
// only quants the dp4a dense-prefill GEMM reads directly), and it is sized from
// kDp4aDenseMaxM=64, NOT from max_tokens — the kernel is not taken above M=64,
// so sizing it from the context would reserve 64x too much at ctx 4096.
TEST(ExecT2Demand, PrefillPairIsChargedOnlyForSub5BitDenseAndCapsAtM64) {
    ExecShape s = dense_shape();
    s.mmvq_max_k = 4096;
    const size_t without = exec_t2_demand(s, 4096).quant_scratch;

    s.has_sub5bit_dense = true;
    const size_t with = exec_t2_demand(s, 4096).quant_scratch;
    EXPECT_GT(with, without);
    // 64 * 128 blocks * 52 bytes.
    EXPECT_EQ(with - without, 64ull * 128 * (48 + sizeof(float)));

    // Capped at M=64: a longer context must not grow it.
    EXPECT_EQ(exec_t2_demand(s, 131072).quant_scratch, with)
        << "the dp4a dense prefill kernel is not taken above M=64, so neither is the reservation";
    // And at max_tokens == 1 (decode-only sizing) the pair is not charged at all.
    EXPECT_EQ(exec_t2_demand(s, 1).quant_scratch, without);
}

TEST(ExecT2Demand, SplitkFollowsHeadsBatchAndContextAndCapsAt128Splits) {
    ExecShape s = dense_shape();
    s.n_heads = 32;
    s.head_dim = 128;
    s.max_batch_size = 1;  // max_logit_tokens floor of 8

    // ctx 1024 -> 64 KV blocks -> 64 splits (below the cap).
    const size_t expect = 8ull * 32 * 64 * (2 + 128) * sizeof(float) + 256;
    EXPECT_EQ(exec_t2_demand(s, 1024).splitk_scratch, expect);

    // The split count caps at 128, so beyond ctx 2048 the term stops growing.
    const size_t at_2048 = exec_t2_demand(s, 2048).splitk_scratch;
    EXPECT_EQ(exec_t2_demand(s, 4096).splitk_scratch, at_2048)
        << "max_splits = min(128, ctx_blocks); reserving past the cap would be dead VRAM";

    // It follows the batch, with the floor of 8 shared with the sampling scratch.
    s.max_batch_size = 16;
    EXPECT_EQ(exec_t2_demand(s, 1024).splitk_scratch, expect * 2 - 256);

    // A shape with no attention heads asks for nothing rather than dividing by zero.
    ExecShape headless = dense_shape();
    EXPECT_EQ(exec_t2_demand(headless, 1024).splitk_scratch, 0u);
}

// The sampling scratch is sized from max_logit_tokens = max(max_batch, 8), which
// is the BATCH and not the context. I mistook it for the context once and wrongly
// ruled the tenant out as ~115 MiB when it is ~1 MiB (AUDIT B52 corrected by
// B53), so this pins which quantity it follows.
TEST(ExecT2Demand, SampleScratchFollowsTheBATCHNotTheContext) {
    constexpr size_t kSample =
        sizeof(int32_t) + 64 * (2 * sizeof(float) + 128 * (sizeof(float) + sizeof(int32_t)));

    ExecShape s = dense_shape();
    s.max_batch_size = 1;
    // max(1, 8) = 8 slots, two parities.
    EXPECT_EQ(exec_t2_demand(s, 1024).sample_scratch, 2ull * 8 * kSample);

    // Below the floor of 8 the size must not shrink.
    s.max_batch_size = 4;
    EXPECT_EQ(exec_t2_demand(s, 1024).sample_scratch, 2ull * 8 * kSample);

    // Above it, it follows the batch.
    s.max_batch_size = 16;
    EXPECT_EQ(exec_t2_demand(s, 1024).sample_scratch, 2ull * 16 * kSample);

    // And it is INDEPENDENT of the context, which is the mistake this guards.
    ExecShape long_ctx = dense_shape();
    long_ctx.max_batch_size = 16;
    EXPECT_EQ(exec_t2_demand(long_ctx, 131072).sample_scratch,
              exec_t2_demand(long_ctx, 128).sample_scratch)
        << "sizing it from the context would ask for ~115 MiB instead of ~1 MiB";

    // Sanity on the order of magnitude that made the wrong call wrong.
    EXPECT_LT(exec_t2_demand(long_ctx, 131072).sample_scratch, 4ull * 1024 * 1024);
}
