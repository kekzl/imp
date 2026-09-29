// dispatch_mtp_head() layout detection and name mapping on synthetic name lists, no weights.
// Qwen4Exp list = the 3101 mtp.* names of Qwen3.8-Flash-Next-NVFP4's index (29 + 512 x 6).
#include "model/mtp_head.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

uint8_t g_byte = 0;  // non-null data pointer; dispatch never reads through it

Tensor fake(std::initializer_list<int64_t> dims) {
    std::vector<int64_t> s(dims);
    return Tensor(&g_byte, QType::BF16, static_cast<int>(s.size()), s.data(), /*on_device=*/false);
}

using TensorMap = std::unordered_map<std::string, Tensor>;

TensorMap qwen4exp_names(int n_experts) {
    TensorMap m;
    for (const char* n : {
             "mtp.fc_embedding.weight",
             "mtp.fc_hidden.weight",
             "mtp.hyper_connection_mixer.hc_norm.weight",
             "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
             "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
             "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
             "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
             "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
             "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
             "mtp.layers.0.mlp.shared_expert.down_proj.weight",
             "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
             "mtp.layers.0.mlp.shared_expert.up_proj.weight",
             "mtp.layers.0.mlp.shared_expert_gate.weight",
             "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
             "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
             "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
             "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
             "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
             "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
             "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
             "mtp.layers.0.self_attn.k_norm.weight",
             "mtp.layers.0.self_attn.k_proj.weight",
             "mtp.layers.0.self_attn.o_proj.weight",
             "mtp.layers.0.self_attn.q_norm.weight",
             "mtp.layers.0.self_attn.q_proj.weight",
             "mtp.layers.0.self_attn.v_proj.weight",
             "mtp.pre_fc_norm_embedding.weight",
             "mtp.pre_fc_norm_hidden.weight",
         })
        m.emplace(n, fake({1}));
    m.emplace("mtp.layers.0.mlp.gate.weight", fake({n_experts, 2560}));
    for (int e = 0; e < n_experts; ++e) {
        const std::string p = "mtp.layers.0.mlp.experts." + std::to_string(e) + ".";
        for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
            m.emplace(p + proj + ".weight", fake({1}));
            m.emplace(p + proj + ".weight_scale_inv", fake({1}));
        }
    }
    return m;
}

TEST(MtpLayoutDispatch, Qwen4ExpMapsAll3101Names) {
    const TensorMap m = qwen4exp_names(512);
    ASSERT_EQ(m.size(), 3101u);
    const MtpHead h = dispatch_mtp_head(m, "synthetic", 0);
    EXPECT_EQ(h.layout, MtpLayout::Qwen4Exp);
    EXPECT_TRUE(h.loaded);
    EXPECT_EQ(h.info.n_tensors, 3101);
    EXPECT_EQ(h.info.n_mapped, 3101);
    ASSERT_EQ(h.experts_fp8.size(), 512u);
    EXPECT_NE(h.experts_fp8[511].down_scale_inv.data, nullptr);
    EXPECT_NE(h.fc_embedding.data, nullptr);
    EXPECT_NE(h.fc_hidden.data, nullptr);
    EXPECT_EQ(h.fc.data, nullptr) << "split fc: the fused slot stays null";
    EXPECT_NE(h.attn_hc.block_inject.data, nullptr);
    EXPECT_NE(h.mlp_hc.block_inject.data, nullptr);
    EXPECT_EQ(h.final_mixer.block_inject.data, nullptr) << "final mixer has use_combine=False";
    EXPECT_NE(h.indexer_qk_proj.data, nullptr);
    EXPECT_TRUE(h.experts_up.empty()) << "FP8 experts must not reach the Nemotron upload vectors";
    EXPECT_FALSE(mtp_forward_implemented(h));
}

TEST(MtpLayoutDispatch, Qwen4ExpMissingExpertScaleIsIncomplete) {
    TensorMap m = qwen4exp_names(512);
    m.erase("mtp.layers.0.mlp.experts.300.up_proj.weight_scale_inv");
    const MtpHead h = dispatch_mtp_head(m, "synthetic", 0);
    EXPECT_EQ(h.layout, MtpLayout::Qwen4Exp);
    EXPECT_FALSE(h.loaded);
}

TEST(MtpLayoutDispatch, FcLayoutStaysQwen) {
    TensorMap m;
    for (const char* n : {
             "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.fc.weight",
             "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
             "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
             "mtp.layers.0.self_attn.v_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
             "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight", "mtp.norm.weight",
             "mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.experts.gate_up_proj",
             "mtp.layers.0.mlp.experts.down_proj", "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
             "mtp.layers.0.mlp.shared_expert.up_proj.weight", "mtp.layers.0.mlp.shared_expert.down_proj.weight",
             "mtp.layers.0.mlp.shared_expert_gate.weight",
         })
        m.emplace(n, fake({1}));
    ASSERT_EQ(m.size(), 19u);
    const MtpHead h = dispatch_mtp_head(m, "synthetic", 0);
    EXPECT_EQ(h.layout, MtpLayout::Qwen);
    EXPECT_TRUE(h.loaded);
    EXPECT_EQ(h.info.n_mapped, 19);
    EXPECT_NE(h.fc.data, nullptr);
    EXPECT_TRUE(mtp_forward_implemented(h));
}

TEST(MtpLayoutDispatch, EhProjLayoutStaysNemotron) {
    TensorMap m;
    for (const char* n : {
             "mtp.layers.0.enorm.weight", "mtp.layers.0.hnorm.weight", "mtp.layers.0.eh_proj.weight",
             "mtp.layers.0.norm.weight", "mtp.layers.0.mixer.q_proj.weight", "mtp.layers.0.mixer.k_proj.weight",
             "mtp.layers.0.mixer.v_proj.weight", "mtp.layers.0.mixer.o_proj.weight", "mtp.layers.1.norm.weight",
             "mtp.layers.1.final_layernorm.weight", "mtp.layers.1.mixer.shared_experts.up_proj.weight",
             "mtp.layers.1.mixer.shared_experts.down_proj.weight",
         })
        m.emplace(n, fake({1}));
    m.emplace("mtp.layers.1.mixer.gate.weight", fake({2, 2688}));
    for (int e = 0; e < 2; ++e) {
        const std::string p = "mtp.layers.1.mixer.experts." + std::to_string(e) + ".";
        m.emplace(p + "up_proj.weight", fake({1}));
        m.emplace(p + "down_proj.weight", fake({1}));
    }
    const MtpHead h = dispatch_mtp_head(m, "synthetic", 0);
    EXPECT_EQ(h.layout, MtpLayout::Nemotron);
    EXPECT_TRUE(h.loaded);
    EXPECT_TRUE(h.experts_non_gated);
    EXPECT_EQ(h.experts_up.size(), 2u);
    EXPECT_TRUE(mtp_forward_implemented(h));
}

TEST(MtpLayoutDispatch, HeadKeyAcceptsFcEmbedding) {
    EXPECT_TRUE(name_is_mtp_head_key("mtp.fc_embedding.weight"));
    EXPECT_TRUE(name_is_mtp_head_key("model.mtp.fc_embedding.weight"));
    EXPECT_FALSE(name_is_mtp_head_key("mtp.fc_hidden.weight"));
}

}  // namespace
}  // namespace imp
