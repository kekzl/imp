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
