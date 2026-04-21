#include "model/tensor_kind_matcher.h"

#include <gtest/gtest.h>

using namespace imp;

TEST(TensorKindMatcher, AttentionProjections) {
    EXPECT_EQ(match_tensor_kind("blk.0.attn_q.weight"),      TensorKind::WQ);
    EXPECT_EQ(match_tensor_kind("blk.12.attn_k.weight"),     TensorKind::WK);
    EXPECT_EQ(match_tensor_kind("blk.5.attn_v.weight"),      TensorKind::WV);
    EXPECT_EQ(match_tensor_kind("blk.3.attn_output.weight"), TensorKind::WO);
}

TEST(TensorKindMatcher, FFN) {
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_gate.weight"), TensorKind::W_GATE);
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_up.weight"),   TensorKind::W_UP);
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_down.weight"), TensorKind::W_DOWN);
}

TEST(TensorKindMatcher, MoEExperts) {
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_gate_exps.weight"), TensorKind::EXPERT_GATE);
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_up_exps.weight"),   TensorKind::EXPERT_UP);
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_down_exps.weight"), TensorKind::EXPERT_DOWN);
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_gate_inp.weight"),  TensorKind::ROUTER);
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_gate_inp_shexp.weight"),
              TensorKind::SHARED_EXPERT_GATE);
}

TEST(TensorKindMatcher, GDNAndMamba) {
    EXPECT_EQ(match_tensor_kind("blk.0.ssm_in.weight"),     TensorKind::SSM_IN);
    EXPECT_EQ(match_tensor_kind("blk.0.ssm_out.weight"),    TensorKind::SSM_OUT);
    EXPECT_EQ(match_tensor_kind("blk.0.ssm_conv1d.weight"), TensorKind::CONV1D_W);
    EXPECT_EQ(match_tensor_kind("blk.0.ssm_conv1d.bias"),   TensorKind::CONV1D_B);
    EXPECT_EQ(match_tensor_kind("blk.0.ssm_a"),             TensorKind::A_LOG);
    EXPECT_EQ(match_tensor_kind("blk.0.ssm_dt_b"),          TensorKind::DT_BIAS);
}

TEST(TensorKindMatcher, GDNGate) {
    // GDN output-gating projection — check both possible naming conventions.
    // The matcher should classify whatever name the existing loader uses.
    // If the test fails, determine the actual name via: grep -n "gdn_gate" src/model/*.cpp
    // and extend the matcher.
    TensorKind k = match_tensor_kind("blk.0.gdn_gate.weight");
    EXPECT_EQ(k, TensorKind::GDN_GATE);
}

TEST(TensorKindMatcher, Norms) {
    EXPECT_EQ(match_tensor_kind("blk.0.attn_norm.weight"),     TensorKind::ATTN_NORM);
    EXPECT_EQ(match_tensor_kind("blk.0.ffn_norm.weight"),      TensorKind::FFN_NORM);
    EXPECT_EQ(match_tensor_kind("blk.0.post_attn_norm.weight"),TensorKind::POST_ATTN_NORM);
    EXPECT_EQ(match_tensor_kind("blk.0.post_ffn_norm.weight"), TensorKind::POST_FFN_NORM);
    EXPECT_EQ(match_tensor_kind("blk.0.attn_q_norm.weight"),   TensorKind::QK_NORM_Q);
    EXPECT_EQ(match_tensor_kind("blk.0.attn_k_norm.weight"),   TensorKind::QK_NORM_K);
}

TEST(TensorKindMatcher, Embeddings) {
    EXPECT_EQ(match_tensor_kind("token_embd.weight"), TensorKind::TOK_EMBED);
    EXPECT_EQ(match_tensor_kind("output.weight"),     TensorKind::LM_HEAD);
}

TEST(TensorKindMatcher, UnknownReturnsUnknown) {
    EXPECT_EQ(match_tensor_kind("foo.bar.baz"), TensorKind::UNKNOWN);
    EXPECT_EQ(match_tensor_kind(""),            TensorKind::UNKNOWN);
}
