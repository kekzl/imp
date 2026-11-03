// max_safe_prefill_chunk must agree with prefill dispatch on which attention family serves a
// head_dim; unclamped, the cuBLAS fallback overran its S-matrix on DeepSeek-V2-Lite (MLA,
// hd=192), aborting the process.

#include "exec/attention_dispatch_rules.h"
#include "compute/attention_paged.h"  // paged_attention_serves_head_dim

#include <gtest/gtest.h>

using namespace imp;

TEST(AttentionDispatchRules, FmhaCoversTheFusedHeadDims) {
    for (int hd : {64, 96, 128, 256, 512})
        EXPECT_TRUE(fmha_serves_head_dim(hd)) << "hd=" << hd;
}

TEST(AttentionDispatchRules, FmhaDoesNotCoverMla) {
    // DeepSeek-V2 MLA: qk_nope 128 + qk_rope 64. Neither family serves it, so
    // the chunk MUST stay bounded by the S-matrix.
    EXPECT_FALSE(fmha_serves_head_dim(192));
    EXPECT_FALSE(fa2_serves_head_dim(192, true));
    EXPECT_FALSE(o_n_attention_serves_head_dim(192, /*fa2_hd256_enabled=*/true));
}

TEST(AttentionDispatchRules, Fa2IsHd128AndOptionallyHd256) {
    EXPECT_TRUE(fa2_serves_head_dim(128, false));
    EXPECT_TRUE(fa2_serves_head_dim(256, true));
    EXPECT_FALSE(fa2_serves_head_dim(256, false));  // gated by attention.fa2_hd256
    EXPECT_FALSE(fa2_serves_head_dim(512, true));   // hd=512 is FMHA/cuBLAS only
}

TEST(AttentionDispatchRules, OnlyUnservedHeadDimsNeedTheSmatrix) {
    // The union is what decides whether a chunk is unconstrained. Anything
    // outside it is served by materialising [n, ctx_len] and must be clamped.
    for (int hd : {128, 256, 512, 64, 96})
        EXPECT_TRUE(o_n_attention_serves_head_dim(hd, true)) << "hd=" << hd;
    for (int hd : {192, 80, 160, 320})
        EXPECT_FALSE(o_n_attention_serves_head_dim(hd, true)) << "hd=" << hd;
}

// A dtype outside paged_attention_serves_head_dim's switch defeated the #1674 guard:
// kv_cache.dtype=mxfp4 at head_dim=96 passed init and threw at first decode (AUDIT_ARCH_2026
// A1-5). MXFP4_KV shares NVFP4's template set (64/128/256/512), no 96.
TEST(AttentionDispatchRules, PagedDecodeRefusesMxfp4KvAtHeadDim96) {
    EXPECT_FALSE(paged_attention_serves_head_dim(QType::MXFP4_KV, 96));
    for (int hd : {64, 128, 256, 512})
        EXPECT_TRUE(paged_attention_serves_head_dim(QType::MXFP4_KV, hd)) << "hd=" << hd;
    EXPECT_TRUE(paged_attention_serves_head_dim(QType::F16, 96));
    EXPECT_TRUE(paged_attention_serves_head_dim(QType::F16, 192));  // MLA (#2374)
    EXPECT_FALSE(paged_attention_serves_head_dim(QType::FP8_E4M3, 192));
}

// F16 multitok split rule on the RTX 5090 (170 SMs, 2 CTAs/SM at HD=128 x 4 heads): one wave,
// unsplit at a budget <= 2, fewest splits with the fewest per-warp block rounds otherwise.
TEST(AttentionDispatchRules, F16MultitokSplitCountIsOneWave) {
    // Qwen3-8B 32/8 heads: 8 CTAs per row; 1131 tokens = 71 blocks of 16.
    EXPECT_EQ(paged_attention_f16_multitok_split_count(16 * 8, 2, 170, 71), 1);   // budget 2
    EXPECT_EQ(paged_attention_f16_multitok_split_count(32 * 8, 2, 170, 71), 1);   // budget 1
    EXPECT_EQ(paged_attention_f16_multitok_split_count(8 * 8, 2, 170, 71), 5);    // 15 blocks, 2 rounds
    EXPECT_EQ(paged_attention_f16_multitok_split_count(1 * 8, 2, 170, 71), 9);    // 8 blocks, 1 round
    EXPECT_EQ(paged_attention_f16_multitok_split_count(1 * 8, 2, 170, 512), 32);  // 16 blocks, 2 rounds
    EXPECT_EQ(paged_attention_f16_multitok_split_count(1 * 8, 2, 170, 11), 2);    // >= 4 blocks per split
    EXPECT_EQ(paged_attention_f16_multitok_split_count(1 * 8, 2, 170, 3), 1);
    // Gemma-4 global HD=512 x 2 heads runs 1 CTA/SM: 16 rows x 8 CTAs never split.
    EXPECT_EQ(paged_attention_f16_multitok_split_count(16 * 8, 1, 170, 71), 1);
    for (int rows : {1, 2, 4, 8, 16, 32})
        for (int blocks : {4, 71, 256, 512, 2048}) {
            const int s = paged_attention_f16_multitok_split_count(rows * 8, 2, 170, blocks);
            EXPECT_TRUE(s == 1 || rows * 8 * s <= 340) << rows << " rows " << blocks << " blocks: " << s;
            EXPECT_LE(s, 64);
        }
}
