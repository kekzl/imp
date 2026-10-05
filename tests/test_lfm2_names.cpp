// LFM2 tensor names -> HF-canonical names weight_map.cpp matches, NVFP4 scale suffixes kept.

#include <gtest/gtest.h>

#include "model/lfm2_names.h"

using imp::lfm2_canonical_name;

TEST(Lfm2Names, ModulesAndSuffixes) {
    EXPECT_EQ(lfm2_canonical_name("model.embedding_norm.weight"), "model.norm.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.3.operator_norm.weight"),
              "model.layers.3.input_layernorm.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.3.ffn_norm.weight"),
              "model.layers.3.post_attention_layernorm.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.2.self_attn.out_proj.weight_scale_2"),
              "model.layers.2.self_attn.o_proj.weight_scale_2");
    EXPECT_EQ(lfm2_canonical_name("model.layers.2.self_attn.q_layernorm.weight"),
              "model.layers.2.self_attn.q_norm.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.0.conv.in_proj.weight_scale"),
              "model.layers.0.mamba.in_proj.weight_scale");
    EXPECT_EQ(lfm2_canonical_name("model.layers.0.conv.conv.weight"), "model.layers.0.mamba.conv1d.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.1.feed_forward.w3.weight"),
              "model.layers.1.mlp.up_proj.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.5.feed_forward.gate.weight"),
              "model.layers.5.mlp.gate.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.5.feed_forward.expert_bias"), "model.layers.5.mlp.gate.bias");
    EXPECT_EQ(lfm2_canonical_name("model.layers.23.feed_forward.experts.15.w1.weight_scale_2"),
              "model.layers.23.mlp.experts.15.gate_proj.weight_scale_2");
    EXPECT_EQ(lfm2_canonical_name("model.layers.9.feed_forward.experts.0.w2.weight"),
              "model.layers.9.mlp.experts.0.down_proj.weight");
    // Not LFM2-specific: unchanged.
    EXPECT_EQ(lfm2_canonical_name("model.layers.2.self_attn.q_proj.weight"),
              "model.layers.2.self_attn.q_proj.weight");
    EXPECT_EQ(lfm2_canonical_name("model.embed_tokens.weight"), "model.embed_tokens.weight");
    EXPECT_EQ(lfm2_canonical_name("model.layers.1.feed_forward.w1x.weight"),
              "model.layers.1.feed_forward.w1x.weight");
}
