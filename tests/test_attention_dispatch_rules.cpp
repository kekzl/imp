// Which attention family serves which head_dim, asked in one place.
//
// The prefill dispatch picks the kernel; max_safe_prefill_chunk decides how big
// a chunk it may hand it. When the second assumes a family that the first
// cannot use, nothing clamps the chunk and the cuBLAS fallback runs past its
// S-matrix. That happened: the clamp trusted `fmha_prefill_threshold` without
// asking whether FMHA serves the head_dim, and a perplexity run on
// DeepSeek-V2-Lite (MLA, head_dim 192) aborted the process with
// "engine should have prevented this".

#include "exec/attention_dispatch_rules.h"

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
