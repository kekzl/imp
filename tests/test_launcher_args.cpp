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

// 17 fields each, the old nvfp4_quant_hw_fp16 / nvfp4_dequant_hw_fp16 parameter lists in order.
TEST(Nvfp4QuantHwArgs, CarriesEveryFieldInDeclarationOrder) {
    const Nvfp4QuantHwArgs a{.batch_size = 1,
                             .n_heads = 2,
                             .n_tokens = 3,
                             .head_dim = 4,
                             .stride_bz_input = 5,
                             .stride_h_input = 6,
                             .stride_seq_input = 7,
                             .stride_bz_output = 8,
                             .stride_h_output = 9,
                             .stride_seq_output = 10,
                             .stride_bz_output_sf = 11,
                             .stride_h_output_sf = 12,
                             .stride_seq_output_sf = 13};
    const auto& [in, fp4, sf, b, nh, nt, hd, s0, s1, s2, s3, s4, s5, s6, s7, s8, st] = a;
    EXPECT_EQ((std::array{b, nh, nt, hd}), (std::array{1, 2, 3, 4}));
    EXPECT_EQ((std::array<int64_t, 9>{s0, s1, s2, s3, s4, s5, s6, s7, s8}),
              (std::array<int64_t, 9>{5, 6, 7, 8, 9, 10, 11, 12, 13}));
}

TEST(Nvfp4DequantHwArgs, CarriesEveryFieldInDeclarationOrder) {
    const Nvfp4DequantHwArgs a{.batch_size = 1,
                               .n_heads = 2,
                               .n_tokens = 3,
                               .head_dim = 4,
                               .stride_bz_input = 5,
                               .stride_h_input = 6,
                               .stride_seq_input = 7,
                               .stride_bz_output = 8,
                               .stride_h_output = 9,
                               .stride_seq_output = 10,
                               .stride_bz_input_sf = 11,
                               .stride_h_input_sf = 12,
                               .stride_seq_input_sf = 13};
    const auto& [fp4, sf, out, b, nh, nt, hd, s0, s1, s2, s3, s4, s5, s6, s7, s8, st] = a;
    EXPECT_EQ((std::array{b, nh, nt, hd}), (std::array{1, 2, 3, 4}));
    EXPECT_EQ((std::array<int64_t, 9>{s0, s1, s2, s3, s4, s5, s6, s7, s8}),
              (std::array<int64_t, 9>{5, 6, 7, 8, 9, 10, 11, 12, 13}));
}

// 16 fields, the old gdn_scan_decode_f32 parameter list in order; grouped_layout defaults to 0 (tiled).
TEST(GdnScanDecodeArgs, CarriesEveryFieldInDeclarationOrder) {
    const GdnScanDecodeArgs a{.n_heads = 1, .head_dim_ssm = 2, .state_size = 3, .n_groups = 4};
    const auto& [x, B, C, al, be, alog, dtb, h, y, z, nh, hd, ss, ng, st, gl] = a;
    EXPECT_EQ((std::array{nh, hd, ss, ng, gl}), (std::array{1, 2, 3, 4, 0}));
    EXPECT_EQ(z, nullptr);
}

// 16 fields, the old mtp_attn_scan_slots parameter list in order.
TEST(MtpAttnScanSlotsArgs, CarriesEveryFieldInDeclarationOrder) {
    const MtpAttnScanSlotsArgs a{.slot_elems = 7,
                                 .max_pos = 1,
                                 .n_rows = 2,
                                 .num_heads = 3,
                                 .num_kv_heads = 4,
                                 .head_dim = 5,
                                 .ring = 6};
    const auto& [q, k, v, out, slots, pos, se, mp, nr, nh, nkv, hd, st, ring, hi, part] = a;
    EXPECT_EQ((std::array{mp, nr, nh, nkv, hd, ring}), (std::array{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(se, 7);
}

// Defaults = the old trailing defaults: no ring, no slot bound, no split-K.
TEST(MtpAttnScanSlotsArgs, OmittedFieldsTakeTheOldDefaults) {
    const MtpAttnScanSlotsArgs a{.n_rows = 1};
    EXPECT_EQ(a.ring, 0);
    EXPECT_EQ(a.d_hi, nullptr);
    EXPECT_EQ(a.split_part, nullptr);
}

// 14 / 15 fields, the old mtp_feed_rows_multislot{,_dev} parameter lists in order.
TEST(MtpFeedRowsArgs, CarriesEveryFieldInDeclarationOrder) {
    const MtpFeedRowsArgs a{.n_rows = 1, .hidden_dim = 2, .post_norm_eps = 3.0f, .post_norm_offset = 4.0f};
    const auto& [tok, hid, src, nr, slots, pos, mtp, emb, ws, hd, st, pn, eps, off] = a;
    EXPECT_EQ((std::array{nr, hd}), (std::array{1, 2}));
    EXPECT_EQ((std::array{eps, off}), (std::array{3.0f, 4.0f}));
    const MtpFeedRowsDevArgs d{.n_rows = 1, .max_pos = 2, .hidden_dim = 3, .post_norm_offset = 4.0f};
    const auto& [dtok, dhid, dsrc, dnr, dslots, dpos, dmp, dmtp, demb, dws, dhd, dst, dpn, deps, doff] = d;
    EXPECT_EQ((std::array{dnr, dmp, dhd}), (std::array{1, 2, 3}));
    EXPECT_EQ(doff, 4.0f);
}

// Defaults = the old trailing defaults: no post norm, eps 1e-6, offset 0.
TEST(MtpFeedRowsArgs, OmittedFieldsTakeTheOldDefaults) {
    const MtpFeedRowsArgs a{.n_rows = 1};
    const MtpFeedRowsDevArgs d{.n_rows = 1};
    EXPECT_EQ((std::array{a.post_norm, d.post_norm}), (std::array<const imp::Tensor*, 2>{}));
    EXPECT_EQ((std::array{a.post_norm_eps, d.post_norm_eps}), (std::array{1e-6f, 1e-6f}));
    EXPECT_EQ((std::array{a.post_norm_offset, d.post_norm_offset}), (std::array{0.0f, 0.0f}));
}

// 15 fields, the old qsa_gather_kv parameter list in order.
TEST(QsaGatherKvArgs, CarriesEveryFieldInDeclarationOrder) {
    const QsaGatherKvArgs a{.block_size = 1, .n_kv = 2, .hd = 3, .cap = 4, .blocks_per_row = 5, .rows = 6};
    const auto& [k, v, bt, bs, nkv, hd, sel, cnt, cap, ks, vs, bpr, ctx, rows, st] = a;
    EXPECT_EQ((std::array{bs, nkv, hd, cap, bpr, rows}), (std::array{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(ctx, nullptr);
}

// 15 fields, the old ple_conv_add parameter list in order; the snapshot / real-row pointers default
// to nullptr as before.
TEST(PleConvAddArgs, CarriesEveryFieldInDeclarationOrder) {
    const PleConvAddArgs a{.slot_stride = 5, .n_seq = 1, .channels = 2, .kernel = 3, .dilation = 4};
    const auto& [gv, gvn, w, cs, stride, slots, ns, hidden, ch, kern, dil, st, snap, sn, rn] = a;
    EXPECT_EQ((std::array{ns, ch, kern, dil}), (std::array{1, 2, 3, 4}));
    EXPECT_EQ(stride, 5);
    EXPECT_EQ((std::array<const void*, 3>{snap, sn, rn}), (std::array<const void*, 3>{}));
}

// 14 fields, the old dspark_attention_dev parameter list in order; d_first / blocks default to nullptr.
TEST(DsparkAttentionDevArgs, CarriesEveryFieldInDeclarationOrder) {
    const DsparkAttentionDevArgs a{.rows = 1, .n_heads = 2, .n_kv_heads = 3, .max_kv_len = 4, .scale = 5.0f};
    const auto& [q, k, v, o, work, rows, nh, nkv, ctx, mkv, sc, st, first, blocks] = a;
    EXPECT_EQ((std::array{rows, nh, nkv, mkv}), (std::array{1, 2, 3, 4}));
    EXPECT_EQ(sc, 5.0f);
    EXPECT_EQ((std::array{first, blocks}), (std::array<const int*, 2>{}));
}

// 14 fields, the old pli_project_combine parameter list in order.
TEST(PliProjectCombineArgs, CarriesEveryFieldInDeclarationOrder) {
    const PliProjectCombineArgs a{.n = 1,
                                  .n_layers = 2,
                                  .dim = 3,
                                  .vocab = 4,
                                  .mm_token_id = 5,
                                  .pad_id = 6,
                                  .eps = 7.0f,
                                  .tok_scale = 8.0f,
                                  .out_scale = 9.0f};
    const auto& [proj, table, ids, nw, n, nl, dim, vocab, mm, pad, eps, ts, os, st] = a;
    EXPECT_EQ((std::array{n, nl, dim, vocab, mm, pad}), (std::array{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ((std::array{eps, ts, os}), (std::array{7.0f, 8.0f, 9.0f}));
}

}  // namespace
