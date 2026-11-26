// Launcher argument structs (audit D-11, B-133): every field is named at the call site. The
// structured bindings pin the field count (an added or dropped field fails to compile); the
// int runs a swap would silently corrupt are checked in declaration order.

#include <gtest/gtest.h>

#include "compute/gdn.h"
#include "compute/rope.h"
#include "exec/executor_kernels.h"

#include <array>
#include <type_traits>

namespace {

using imp::GdnScanBatchedArgs;
using imp::KvWriteRopeArgs;
using imp::QknormRopeArgs;

static_assert(std::is_aggregate_v<GdnScanBatchedArgs<float>>);
static_assert(std::is_aggregate_v<KvWriteRopeArgs>);
static_assert(std::is_aggregate_v<QknormRopeArgs>);

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

}  // namespace
