// Launcher argument structs (audit D-11, B-133): every field is named at the call site, so a
// declaration-order swap fails the build there (nvcc/g++ out-of-order designated initializers).
// Pinned here: the omitted-field defaults the old trailing parameters had, and the state types.

#include <gtest/gtest.h>

#include "compute/attention_apa.h"
#include "compute/dspark_attention.h"
#include "compute/gdn.h"
#include "compute/gemma_pli.h"
#include "compute/mtp_forward.h"
#include "compute/mtp_forward_internal.h"
#include "compute/nvfp4_quant_hw.h"
#include "compute/ple.h"
#include "compute/qsa_indexer.h"
#include "compute/rope.h"
#include "compute/ssm.h"
#include "exec/executor_kernels.h"
#include "exec/sparse_attn_select.h"

#include <type_traits>

namespace {

using imp::AttnApaPagedArgs;
using imp::DsparkAttentionDevArgs;
using imp::GdnChunkparArgs;
using imp::GdnScanArgs;
using imp::GdnScanBatchedArgs;
using imp::GdnScanDecodeArgs;
using imp::KvWriteRopeArgs;
using imp::MtpAttnScanSlotsArgs;
using imp::MtpFeedRowsArgs;
using imp::MtpFeedRowsDevArgs;
using imp::MtpMropeArgs;
using imp::Nvfp4DequantHwArgs;
using imp::Nvfp4QuantHwArgs;
using imp::PleConvAddArgs;
using imp::PliProjectCombineArgs;
using imp::QknormRopeArgs;
using imp::QsaGatherKvArgs;
using imp::SparseKeyMinmaxArgs;
using imp::SparsePrefillSelectArgs;
using imp::SsmConv1dGroupedArgs;
using imp::SsmScanDecodeArgs;
using imp::SsmScanDecodeBatchedArgs;
using imp::SsmScanPrefillArgs;

static_assert(std::is_aggregate_v<Nvfp4QuantHwArgs>);
static_assert(std::is_aggregate_v<Nvfp4DequantHwArgs>);
static_assert(std::is_aggregate_v<GdnScanDecodeArgs>);
static_assert(std::is_aggregate_v<MtpAttnScanSlotsArgs>);
static_assert(std::is_aggregate_v<MtpFeedRowsArgs>);
static_assert(std::is_aggregate_v<MtpFeedRowsDevArgs>);
static_assert(std::is_aggregate_v<QsaGatherKvArgs>);
static_assert(std::is_aggregate_v<PleConvAddArgs>);
static_assert(std::is_aggregate_v<DsparkAttentionDevArgs>);
static_assert(std::is_aggregate_v<PliProjectCombineArgs>);

static_assert(std::is_aggregate_v<GdnScanBatchedArgs<float>>);
static_assert(std::is_aggregate_v<SparsePrefillSelectArgs>);
static_assert(std::is_aggregate_v<SparseKeyMinmaxArgs>);
static_assert(std::is_aggregate_v<AttnApaPagedArgs>);
static_assert(std::is_aggregate_v<SsmScanDecodeArgs>);
static_assert(std::is_aggregate_v<SsmScanDecodeBatchedArgs>);
static_assert(std::is_aggregate_v<SsmConv1dGroupedArgs>);
static_assert(std::is_aggregate_v<MtpMropeArgs>);
static_assert(std::is_aggregate_v<KvWriteRopeArgs>);
static_assert(std::is_aggregate_v<QknormRopeArgs>);
static_assert(std::is_aggregate_v<GdnScanArgs<float>>);
static_assert(std::is_aggregate_v<GdnChunkparArgs<float>>);
static_assert(std::is_aggregate_v<SsmScanPrefillArgs>);

// The BF16 instantiations carry BF16 state and snapshot pointers.
static_assert(std::is_same_v<decltype(GdnScanBatchedArgs<__nv_bfloat16>::h_state_pool), __nv_bfloat16*>);
static_assert(std::is_same_v<decltype(GdnScanBatchedArgs<__nv_bfloat16>::h_snap), __nv_bfloat16*>);
static_assert(std::is_same_v<decltype(GdnScanArgs<__nv_bfloat16>::h_state), __nv_bfloat16*>);
static_assert(std::is_same_v<decltype(GdnScanArgs<__nv_bfloat16>::h_snap), __nv_bfloat16*>);

// Defaults = the old trailing parameter defaults: tiled layout, no padding bound, no ragged
// offsets, no snapshot, in-place commit, no factor rows, the legacy stream.
TEST(GdnScanBatchedArgs, OmittedFieldsTakeTheOldDefaults) {
    const GdnScanBatchedArgs<float> a{.n_seq = 1};
    EXPECT_EQ(a.grouped_layout, 0);
    EXPECT_EQ(a.d_real_n, nullptr);
    EXPECT_EQ(a.seq_row_offsets, nullptr);
    EXPECT_EQ(a.h_snap, nullptr);
    EXPECT_EQ(a.d_snap_n, nullptr);
    EXPECT_EQ(a.out_slots, nullptr);
    EXPECT_EQ(a.snap_slots, nullptr);
    EXPECT_EQ(a.fac_out, nullptr);
    EXPECT_EQ(a.fac_stride, 0);
    EXPECT_EQ(a.stream, nullptr);
}

TEST(KvWriteRopeArgs, OmittedFieldsTakeTheOldDefaults) {
    const KvWriteRopeArgs a{.threads = 1};
    EXPECT_EQ(a.inv_scaling, 1.0f);
    EXPECT_FALSE(a.neox);
    EXPECT_EQ(a.longrope_inv_freqs, nullptr);
    EXPECT_EQ(a.stream, nullptr);
}

// Defaults = the old qknorm_rope_fused defaults: theta 1e4, unscaled, full head_dim, interleaved
// pairs, no YaRN (attn_factor 1), no LongRoPE, no M-RoPE, one token.
TEST(QknormRopeArgs, OmittedFieldsTakeTheOldDefaults) {
    const QknormRopeArgs a{.n_heads = 1};
    EXPECT_EQ(a.theta, 10000.0f);
    EXPECT_EQ(a.scaling, 1.0f);
    EXPECT_EQ(a.rope_dim, 0);
    EXPECT_FALSE(a.neox);
    EXPECT_EQ(a.stream, nullptr);
    EXPECT_EQ(a.weight_offset, 0.0f);
    EXPECT_EQ(a.ext_factor, 0.0f);
    EXPECT_EQ(a.attn_factor, 1.0f);
    EXPECT_EQ(a.corr_dims, nullptr);
    EXPECT_EQ(a.longrope_inv_freqs, nullptr);
    EXPECT_EQ(a.mrope.positions, nullptr);
    EXPECT_EQ(a.n_tokens, 1);
}

// Defaults = the old trailing defaults: tiled layout, no padding bound, no snapshot, no second slab.
TEST(GdnScanArgs, OmittedFieldsTakeTheOldDefaults) {
    const GdnScanArgs<float> a{.n_tokens = 1};
    EXPECT_EQ(a.grouped_layout, 0);
    EXPECT_EQ(a.d_real_n, nullptr);
    EXPECT_EQ(a.h_snap, nullptr);
    EXPECT_EQ(a.d_snap_n, nullptr);
    EXPECT_EQ(a.snap2_off, 0);
}

// Defaults the two deleted field-order cases asserted: strip_chunks 0 = auto; FP32 state, no SSD,
// no padding bound.
static_assert(GdnChunkparArgs<__nv_bfloat16>{}.strip_chunks == 0);
static_assert(SsmScanPrefillArgs{}.h_dtype == imp::QType::F32 && !SsmScanPrefillArgs{}.ssd &&
              SsmScanPrefillArgs{}.d_real_n == nullptr);

// Defaults the seven deleted field-order cases asserted (one field swap per struct fails the build
// at the designated-initializer call sites): no output table / sequence offsets by default,
// APA owner -1 = uncached, FP32 SSM state on the default stream, no grouped-conv side tables.
static_assert(SparsePrefillSelectArgs{}.out_block_table == nullptr);
static_assert(SparseKeyMinmaxArgs{}.seq_offsets == nullptr);
static_assert(AttnApaPagedArgs{}.owner == -1);
static_assert(SsmScanDecodeArgs{}.h_dtype == imp::QType::F32 && SsmScanDecodeArgs{}.stream == nullptr);
static_assert(SsmConv1dGroupedArgs{}.d_real_n == nullptr && SsmConv1dGroupedArgs{}.conv_snap == nullptr &&
              SsmConv1dGroupedArgs{}.d_snap_n == nullptr && SsmConv1dGroupedArgs{}.conv_prev == nullptr &&
              SsmConv1dGroupedArgs{}.out_slots == nullptr && SsmConv1dGroupedArgs{}.snap_slots == nullptr);

// Defaults = the old trailing defaults: one row, positions pos + r (no per-row table).
TEST(MtpMropeArgs, OmittedFieldsTakeTheOldDefaults) {
    const MtpMropeArgs a{.pos = 1};
    EXPECT_EQ(a.n_rows, 1);
    EXPECT_EQ(a.d_row_pos, nullptr);
}

// Defaults the nine deleted slice-7 field-order cases asserted (one field swap per struct fails
// the build at the designated-initializer call sites): tiled GDN decode without z, no QSA ctx
// table, no PLE snapshot / real-row tables, no DSpark first-row / block tables.
static_assert(GdnScanDecodeArgs{}.grouped_layout == 0 && GdnScanDecodeArgs{}.z == nullptr);
static_assert(QsaGatherKvArgs{}.scratch_ctx == nullptr);
static_assert(PleConvAddArgs{}.snap_state == nullptr && PleConvAddArgs{}.d_snap_n == nullptr &&
              PleConvAddArgs{}.d_real_n == nullptr);
static_assert(DsparkAttentionDevArgs{}.d_first == nullptr && DsparkAttentionDevArgs{}.blocks == nullptr);

// Defaults = the old trailing defaults: no ring, no slot bound, no split-K.
TEST(MtpAttnScanSlotsArgs, OmittedFieldsTakeTheOldDefaults) {
    const MtpAttnScanSlotsArgs a{.n_rows = 1};
    EXPECT_EQ(a.ring, 0);
    EXPECT_EQ(a.d_hi, nullptr);
    EXPECT_EQ(a.split_part, nullptr);
}

// Defaults = the old trailing defaults: no post norm, eps 1e-6, offset 0.
TEST(MtpFeedRowsArgs, OmittedFieldsTakeTheOldDefaults) {
    const MtpFeedRowsArgs a{.n_rows = 1};
    const MtpFeedRowsDevArgs d{.n_rows = 1};
    EXPECT_EQ((std::array{a.post_norm, d.post_norm}), (std::array<const imp::Tensor*, 2>{}));
    EXPECT_EQ((std::array{a.post_norm_eps, d.post_norm_eps}), (std::array{1e-6f, 1e-6f}));
    EXPECT_EQ((std::array{a.post_norm_offset, d.post_norm_offset}), (std::array{0.0f, 0.0f}));
}

}  // namespace

// B-133 slice 8: a field swap in each struct fails the build at its src call site (designated
// initializers), so only the omitted-field defaults are pinned, at compile time.
#include "compute/gemm.h"
#include "exec/attention_prefill_args.h"
#include "exec/greedy_argmax_args.h"

static_assert(std::is_aggregate_v<imp::MoeDp4aGateUpArgs> && imp::MoeDp4aGateUpArgs{}.stream == nullptr);
static_assert(std::is_aggregate_v<imp::SparseSelectBlocksArgs>);
// try_fa2_fp16qk_prefill's old trailing defaults: no device kv length, no sinks.
static_assert(imp::Fa2PrefillArgs{}.d_kv_len == nullptr && imp::Fa2PrefillArgs{}.sinks == nullptr);
static_assert(std::is_aggregate_v<imp::AttnPrefillTierArgs>);
// greedy_argmax_all's old trailing defaults: no penalties, no top-m, no bans, GEMV LM head.
static_assert(imp::GreedyArgmaxArgs{}.d_hist == nullptr && imp::GreedyArgmaxArgs{}.n_hist == 0 &&
              imp::GreedyArgmaxArgs{}.d_draft == nullptr);
static_assert(imp::GreedyArgmaxArgs{}.rep_pen == 1.0f && imp::GreedyArgmaxArgs{}.freq_pen == 0.0f &&
              imp::GreedyArgmaxArgs{}.pres_pen == 0.0f);
static_assert(imp::GreedyArgmaxArgs{}.d_topm == nullptr && imp::GreedyArgmaxArgs{}.topm == 0 &&
              imp::GreedyArgmaxArgs{}.d_banned == nullptr && imp::GreedyArgmaxArgs{}.n_banned == 0);
static_assert(!imp::GreedyArgmaxArgs{}.allow_w4a4_head && imp::GreedyArgmaxArgs{}.d_banned_alt == nullptr &&
              imp::GreedyArgmaxArgs{}.n_banned_alt == 0 && imp::GreedyArgmaxArgs{}.h_row_alt == nullptr);
static_assert(imp::GreedyArgmaxArgs{}.h_row_hist == nullptr &&
              imp::GreedyArgmaxArgs{}.h_row_hist_n == nullptr &&
              imp::GreedyArgmaxArgs{}.h_row_pens == nullptr);

// B-133 slice 9: same rule; the structs without old defaults are pinned as aggregates only.
#include "compute/attention_cublas.h"
#include "compute/mla_absorb.h"
#include "exec/decode_pipeline_advance.h"
#include "exec/moe_decode_args.h"
#include "exec/moe_router_route.h"
#include "exec/workspace.h"
#include "quant/nvfp4_gemm_tile.h"

static_assert(std::is_aggregate_v<imp::Workspace::InitArgs> &&
              std::is_aggregate_v<imp::RouterFusedRouteArgs> &&
              std::is_aggregate_v<imp::DecodePipelineAdvanceArgs> &&
              std::is_aggregate_v<imp::MoeDecodeFastArgs>);
// gdn_scan_reference_f32: tiled layout, no device row count.
static_assert(imp::GdnScanReferenceArgs{}.grouped_layout == 0 &&
              imp::GdnScanReferenceArgs{}.d_real_n == nullptr);
// The two tile GEMMs: no stream-K workspace; gemm_fp8_tile auto variant, per-tensor activation scale.
static_assert(imp::GemmNvfp4TileSwigluQuantArgs{}.workspace == nullptr &&
              imp::GemmNvfp4TileSwigluQuantArgs{}.workspace_bytes == 0);
static_assert(imp::GemmFp8TileArgs{}.variant == -1 && imp::GemmFp8TileArgs{}.workspace == nullptr &&
              imp::GemmFp8TileArgs{}.workspace_bytes == 0 && !imp::GemmFp8TileArgs{}.a_per_row);
// mtp_draft_step: host token path, no NVFP4 / FP8 head view.
static_assert(imp::MtpDraftStepArgs{}.lm_head_nvfp4 == nullptr &&
              imp::MtpDraftStepArgs{}.d_prev_token == nullptr &&
              imp::MtpDraftStepArgs{}.d_out_token == nullptr &&
              imp::MtpDraftStepArgs{}.lm_head_fp8 == nullptr &&
              imp::MtpDraftStepArgs{}.lm_head_fp8_scales == nullptr);
// mtp_workspace_allocate's 2-argument form: no MoE, no attention, one KV slot, no hc streams.
static_assert(imp::MtpWorkspaceAllocateArgs{}.n_experts == 0 && imp::MtpWorkspaceAllocateArgs{}.top_k == 0 &&
              imp::MtpWorkspaceAllocateArgs{}.expert_d_ff == 0 &&
              imp::MtpWorkspaceAllocateArgs{}.shared_d_ff == 0);
static_assert(imp::MtpWorkspaceAllocateArgs{}.num_heads == 0 &&
              imp::MtpWorkspaceAllocateArgs{}.num_kv_heads == 0 &&
              imp::MtpWorkspaceAllocateArgs{}.head_dim == 0 &&
              imp::MtpWorkspaceAllocateArgs{}.max_seq_len == 0);
static_assert(imp::MtpWorkspaceAllocateArgs{}.n_kv_slots == 1 &&
              imp::MtpWorkspaceAllocateArgs{}.hc_count == 0 &&
              imp::MtpWorkspaceAllocateArgs{}.hc_lowrank == 0);
static_assert(imp::MlaAbsorbedDecodeArgs{}.stream == nullptr);
// attention_cublas_prefill{,_sliced}: no softcap, square path, default stream, no window, no sinks.
static_assert(imp::AttnCublasPrefillArgs{}.softcap == 0.0f && imp::AttnCublasPrefillArgs{}.q_offset == 0 &&
              imp::AttnCublasPrefillArgs{}.stream == nullptr &&
              imp::AttnCublasPrefillArgs{}.sliding_window == 0 &&
              imp::AttnCublasPrefillArgs{}.sinks == nullptr);
// rope_forward: theta 10000, no scaling, full head_dim, interleaved pairs, no YaRN / LongRoPE / M-RoPE.
static_assert(imp::RopeForwardArgs{}.theta == 10000.0f && imp::RopeForwardArgs{}.scaling == 1.0f &&
              imp::RopeForwardArgs{}.rope_dim == 0 && !imp::RopeForwardArgs{}.neox);
static_assert(imp::RopeForwardArgs{}.ext_factor == 0.0f && imp::RopeForwardArgs{}.attn_factor == 1.0f &&
              imp::RopeForwardArgs{}.corr_dims == nullptr && imp::RopeForwardArgs{}.stream == nullptr);
static_assert(imp::RopeForwardArgs{}.longrope_inv_freqs == nullptr &&
              imp::RopeForwardArgs{}.mrope.positions == nullptr &&
              imp::RopeForwardArgs{}.mrope.pos_delta == nullptr);
// run_moe_decode_nvfp4_host: one row.
static_assert(imp::MoeDecodeNvfp4HostArgs{}.n_rows == 1);
