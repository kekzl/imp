// Launcher argument structs (audit D-11, B-133): every field is named at the call site. The
// structured bindings pin the field count (an added or dropped field fails to compile); the
// int runs a swap would silently corrupt are checked in declaration order.

#include <gtest/gtest.h>

#include "compute/attention_apa.h"
#include "compute/gdn.h"
#include "compute/mtp_forward.h"
#include "compute/rope.h"
#include "compute/ssm.h"
#include "exec/executor_kernels.h"
#include "exec/sparse_attn_select.h"

#include <array>
#include <type_traits>

namespace {

using imp::AttnApaPagedArgs;
using imp::GdnChunkparArgs;
using imp::GdnScanArgs;
using imp::GdnScanBatchedArgs;
using imp::KvWriteRopeArgs;
using imp::MtpMropeArgs;
using imp::QknormRopeArgs;
using imp::SparseKeyMinmaxArgs;
using imp::SparsePrefillSelectArgs;
using imp::SsmConv1dGroupedArgs;
using imp::SsmScanDecodeArgs;
using imp::SsmScanDecodeBatchedArgs;
using imp::SsmScanPrefillArgs;

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

// 26 fields, the old gdn_scan_fused_{f32,bf16}_batched parameter list in order.
TEST(GdnScanBatchedArgs, CarriesEveryFieldInDeclarationOrder) {
    const GdnScanBatchedArgs<__nv_bfloat16> a{.conv_channels = 1,
                                              .h_state_seq_stride = 2,
                                              .n_seq = 3,
                                              .n_tokens = 4,
                                              .n_heads = 5,
                                              .head_dim_ssm = 6,
                                              .state_size = 7,
                                              .n_groups = 8,
                                              .grouped_layout = 9,
                                              .fac_stride = 10};
    const auto& [conv, cc, al, be, alog, dtb, pool, slots, stride, y, ns, nt, nh, hd, ss, ng, st, gl, rn,
                 offs, snap, sn, outs, snaps, fac, fs] = a;
    EXPECT_EQ((std::array{cc, ns, nt, nh, hd, ss, ng, gl, fs}), (std::array{1, 3, 4, 5, 6, 7, 8, 9, 10}));
    EXPECT_EQ(stride, 2);
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(pool)>, __nv_bfloat16*>));
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(snap)>, __nv_bfloat16*>));
}

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

// 22 fields, the old write_kv_cache_rope_fused parameter list in order.
TEST(KvWriteRopeArgs, CarriesEveryFieldInDeclarationOrder) {
    const KvWriteRopeArgs a{.grid = dim3(11, 2),
                            .threads = 1,
                            .block_stride = 2,
                            .row_elems = 3,
                            .block_size = 4,
                            .n_tokens = 5,
                            .max_blocks_per_seq = 6,
                            .n_sequences = 7,
                            .n_kv_heads = 8,
                            .head_dim = 9,
                            .theta = 10.0f,
                            .inv_scaling = 11.0f,
                            .rope_pairs = 12,
                            .neox = true};
    const auto& [grid, thr, st, k, v, pos, bt, kc, vc, bstr, re, bs, nt, mbs, nseq, nkv, hd, th, inv, rp, nx,
                 lr] = a;
    EXPECT_EQ((std::array{thr, bstr, re, bs, nt, mbs, nseq, nkv, hd, rp}),
              (std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 12}));
    EXPECT_EQ((std::array{grid.x, grid.y, grid.z}), (std::array{11u, 2u, 1u}));
    EXPECT_EQ((std::array{th, inv}), (std::array{10.0f, 11.0f}));
    EXPECT_TRUE(nx);
}

TEST(KvWriteRopeArgs, OmittedFieldsTakeTheOldDefaults) {
    const KvWriteRopeArgs a{.threads = 1};
    EXPECT_EQ(a.inv_scaling, 1.0f);
    EXPECT_FALSE(a.neox);
    EXPECT_EQ(a.longrope_inv_freqs, nullptr);
    EXPECT_EQ(a.stream, nullptr);
}

// 21 fields, the old qknorm_rope_fused / qknorm_rope_fused_cta parameter list in order.
TEST(QknormRopeArgs, CarriesEveryFieldInDeclarationOrder) {
    const QknormRopeArgs a{.n_heads = 1,
                           .n_kv_heads = 2,
                           .head_dim = 3,
                           .eps = 4.0f,
                           .theta = 5.0f,
                           .scaling = 6.0f,
                           .rope_dim = 7,
                           .neox = true,
                           .weight_offset = 8.0f,
                           .ext_factor = 9.0f,
                           .attn_factor = 10.0f,
                           .n_tokens = 11};
    const auto& [q, k, qw, kw, nh, nkv, hd, eps, pos, th, sc, rd, nx, st, wo, ef, af, cd, lr, mr, nt] = a;
    EXPECT_EQ((std::array{nh, nkv, hd, rd, nt}), (std::array{1, 2, 3, 7, 11}));
    EXPECT_EQ((std::array{eps, th, sc, wo, ef, af}), (std::array{4.0f, 5.0f, 6.0f, 8.0f, 9.0f, 10.0f}));
    EXPECT_TRUE(nx);
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

// 19 fields, the old gdn_scan_fused_{f32,bf16} parameter list in order (chunkwise: the same plus
// chunk_size as its own argument).
TEST(GdnScanArgs, CarriesEveryFieldInDeclarationOrder) {
    const GdnScanArgs<__nv_bfloat16> a{.conv_channels = 1,
                                       .n_tokens = 2,
                                       .n_heads = 3,
                                       .head_dim_ssm = 4,
                                       .state_size = 5,
                                       .n_groups = 6,
                                       .grouped_layout = 7,
                                       .snap2_off = 8};
    const auto& [conv, cc, al, be, alog, dtb, h, y, nt, nh, hd, ss, ng, st, gl, rn, snap, sn, s2] = a;
    EXPECT_EQ((std::array{cc, nt, nh, hd, ss, ng, gl}), (std::array{1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(s2, 8);
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(h)>, __nv_bfloat16*>));
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(snap)>, __nv_bfloat16*>));
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

// 18 fields, the old gdn_scan_chunkpar_{f32,bf16} parameter list in order; strip_chunks 0 = auto.
TEST(GdnChunkparArgs, CarriesEveryFieldInDeclarationOrder) {
    const GdnChunkparArgs<float> a{.conv_channels = 1,
                                   .n_tokens = 2,
                                   .n_heads = 3,
                                   .head_dim_ssm = 4,
                                   .state_size = 5,
                                   .n_groups = 6,
                                   .grouped_layout = 7,
                                   .ws_bytes = 8,
                                   .strip_chunks = 9};
    const auto& [conv, cc, al, be, alog, dtb, h, y, nt, nh, hd, ss, ng, st, gl, ws, wsb, sc] = a;
    EXPECT_EQ((std::array{cc, nt, nh, hd, ss, ng, gl, sc}), (std::array{1, 2, 3, 4, 5, 6, 7, 9}));
    EXPECT_EQ(wsb, 8u);
    EXPECT_EQ(GdnChunkparArgs<__nv_bfloat16>{}.strip_chunks, 0);
}

// 21 fields, the old ssm_scan_prefill parameter list in order; defaults FP32 state, no SSD.
TEST(SsmScanPrefillArgs, CarriesEveryFieldInDeclarationOrder) {
    const SsmScanPrefillArgs a{
        .n_tokens = 1, .n_heads = 2, .head_dim_ssm = 3, .state_size = 4, .n_groups = 5};
    const auto& [x, b, c, dt, alog, d, dtb, h, y, z, nt, nh, hd, ss, ng, hq, st, rn, snap, sn, ssd] = a;
    EXPECT_EQ((std::array{nt, nh, hd, ss, ng}), (std::array{1, 2, 3, 4, 5}));
    EXPECT_EQ(hq, imp::QType::F32);
    EXPECT_FALSE(ssd);
    EXPECT_EQ(rn, nullptr);
}

// 20 fields, the old sparse_prefill_select_past parameter list in order.
TEST(SparsePrefillSelectArgs, CarriesEveryFieldInDeclarationOrder) {
    const SparsePrefillSelectArgs a{.n_rows = 1,
                                    .sample_rows = 2,
                                    .past_len = 3,
                                    .n_heads = 4,
                                    .n_kv_heads = 5,
                                    .head_dim = 6,
                                    .block_size = 7,
                                    .capacity_blocks = 8,
                                    .budget_blocks = 9,
                                    .sink_blocks = 10,
                                    .recent_blocks = 11,
                                    .std_coef = 12.0f};
    const auto& [q, nr, sr, mm, bt, pl, nh, nkv, hd, bs, cap, bud, sink, rec, sc, scores, agg, out, ctx,
                 st] = a;
    EXPECT_EQ((std::array{nr, sr, pl, nh, nkv, hd, bs, cap, bud, sink, rec}),
              (std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(sc, 12.0f);
    EXPECT_EQ(out, nullptr);
}

// 18 fields, the old sparse_update_key_minmax_all_layers parameter list in order.
TEST(SparseKeyMinmaxArgs, CarriesEveryFieldInDeclarationOrder) {
    const SparseKeyMinmaxArgs a{.cache_dtype = imp::QType::NVFP4,
                                .k_layer_stride_bytes = 1,
                                .sc_layer_stride_bytes = 2,
                                .mm_layer_stride_bytes = 3,
                                .n_layers = 4,
                                .n_kv_heads = 5,
                                .head_dim = 6,
                                .block_size = 7,
                                .n_tokens = 8,
                                .max_blocks_per_seq = 9,
                                .n_sequences = 10};
    const auto& [dt, k, ks, ksc, scs, mm, mms, pos, bt, so, nl, nkv, hd, bs, nt, mbs, nseq, st] = a;
    EXPECT_EQ(dt, imp::QType::NVFP4);
    EXPECT_EQ((std::array{ks, scs, mms}), (std::array<int64_t, 3>{1, 2, 3}));
    EXPECT_EQ((std::array{nl, nkv, hd, bs, nt, mbs, nseq}), (std::array{4, 5, 6, 7, 8, 9, 10}));
    EXPECT_EQ(so, nullptr);
}

// 20 fields, the old attention_apa_prefill_paged parameter list in order; owner -1 = uncached.
TEST(AttnApaPagedArgs, CarriesEveryFieldInDeclarationOrder) {
    const AttnApaPagedArgs a{.block_size = 1,
                             .tail = 2,
                             .n = 3,
                             .kv_len = 4,
                             .nh = 5,
                             .nkv = 6,
                             .hd = 7,
                             .scale = 8.0f,
                             .q_offset = 9,
                             .eps = 10.0f,
                             .kv_layer = 11};
    const auto& [q, kp, vp, bt, bs, kt, vt, tail, o, n, kv, nh, nkv, hd, sc, qo, eps, layer, owner, st] = a;
    EXPECT_EQ((std::array{bs, tail, n, kv, nh, nkv, hd, qo, layer}),
              (std::array{1, 2, 3, 4, 5, 6, 7, 9, 11}));
    EXPECT_EQ((std::array{sc, eps}), (std::array{8.0f, 10.0f}));
    EXPECT_EQ(owner, -1);
}

// 16 fields, the old ssm_scan_decode parameter list in order; defaults FP32 state, default stream.
TEST(SsmScanDecodeArgs, CarriesEveryFieldInDeclarationOrder) {
    const SsmScanDecodeArgs a{.n_heads = 1, .head_dim_ssm = 2, .state_size = 3, .n_groups = 4};
    const auto& [x, b, c, dt, alog, d, dtb, h, y, z, nh, hd, ss, ng, hq, st] = a;
    EXPECT_EQ((std::array{nh, hd, ss, ng}), (std::array{1, 2, 3, 4}));
    EXPECT_EQ(hq, imp::QType::F32);
    EXPECT_EQ(st, nullptr);
}

// 19 fields, the old ssm_scan_decode_batched parameter list in order.
TEST(SsmScanDecodeBatchedArgs, CarriesEveryFieldInDeclarationOrder) {
    const SsmScanDecodeBatchedArgs a{.h_slot_stride_bytes = 6,
                                     .n_seq = 1,
                                     .n_heads = 2,
                                     .head_dim_ssm = 3,
                                     .state_size = 4,
                                     .n_groups = 5,
                                     .h_dtype = imp::QType::F16};
    const auto& [x, b, c, dt, alog, d, dtb, pool, slots, stride, y, z, ns, nh, hd, ss, ng, hq, st] = a;
    EXPECT_EQ((std::array{ns, nh, hd, ss, ng}), (std::array{1, 2, 3, 4, 5}));
    EXPECT_EQ(stride, 6u);
    EXPECT_EQ(hq, imp::QType::F16);
}

// 16 fields, the old ssm_conv1d_prefill_f32_silu_grouped parameter list in order; the six
// trailing pointers default to nullptr as before.
TEST(SsmConv1dGroupedArgs, CarriesEveryFieldInDeclarationOrder) {
    const SsmConv1dGroupedArgs a{.slot_stride = 1, .n_seq = 2, .conv_kernel = 3};
    const auto& [pool, slots, stride, ns, x, w, b, out, ck, st, rn, snap, sn, prev, os, ss] = a;
    EXPECT_EQ(stride, 1);
    EXPECT_EQ((std::array{ns, ck}), (std::array{2, 3}));
    EXPECT_EQ((std::array<const void*, 6>{rn, snap, sn, prev, os, ss}), (std::array<const void*, 6>{}));
}

// 19 fields, the old mtp_apply_mrope parameter list in order.
TEST(MtpMropeArgs, CarriesEveryFieldInDeclarationOrder) {
    const MtpMropeArgs a{.n_heads = 1,
                         .n_kv_heads = 2,
                         .head_dim = 3,
                         .rope_dim = 4,
                         .theta = 5.0f,
                         .sec0 = 6,
                         .sec1 = 7,
                         .sec2 = 8,
                         .pos = 9,
                         .inv_scaling = 10.0f,
                         .ext_factor = 11.0f,
                         .attn_factor = 12.0f,
                         .corr_dim_0 = 13.0f,
                         .corr_dim_1 = 14.0f,
                         .n_rows = 15};
    const auto& [q, nh, k, nkv, hd, rd, th, s0, s1, s2, pos, inv, ext, attn, c0, c1, st, nr, rp] = a;
    EXPECT_EQ((std::array{nh, nkv, hd, rd, s0, s1, s2, pos, nr}), (std::array{1, 2, 3, 4, 6, 7, 8, 9, 15}));
    EXPECT_EQ((std::array{th, inv, ext, attn, c0, c1}),
              (std::array{5.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f}));
}

// Defaults = the old trailing defaults: one row, positions pos + r (no per-row table).
TEST(MtpMropeArgs, OmittedFieldsTakeTheOldDefaults) {
    const MtpMropeArgs a{.pos = 1};
    EXPECT_EQ(a.n_rows, 1);
    EXPECT_EQ(a.d_row_pos, nullptr);
}

}  // namespace
