// Ling-3.0 (bailing_hybrid) load path without the checkpoint: config.json parse (fixture
// tests/fixtures/hf_config_golden/syn_bailing_hybrid.json = Ling-3.0-tiny's config.json) and the
// whole-map tensor rewrite (model/bailing_hybrid_tensors.cpp) on synthetic BF16 tensors.

#include "model/hf_config_loader.h"
#include "model/model_config.h"
#include "model/weight_map_arch.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

namespace fs = std::filesystem;

ModelConfig load_fixture() {
    const fs::path dir = fs::temp_directory_path() / ("imp_test_ling3_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    fs::copy_file(fs::path(IMP_TEST_FIXTURES_DIR) / "hf_config_golden" / "syn_bailing_hybrid.json",
                  dir / "config.json", fs::copy_options::overwrite_existing);
    ModelConfig cfg;
    EXPECT_TRUE(HFConfigLoader::load_config(dir.string(), cfg));
    fs::remove_all(dir);
    return cfg;
}

TEST(BailingHybridConfig, Ling3TinyMapsKdaMlaAndRouter) {
    const ModelConfig cfg = load_fixture();
    EXPECT_EQ(cfg.arch, ModelArch::BAILING_HYBRID);
    // KDA on the GDN slots: 16 heads x 128, conv 4, lower bound -5, sigmoid gated norm.
    EXPECT_TRUE(cfg.gdn_kda);
    EXPECT_TRUE(cfg.gdn_gate_sigmoid);
    EXPECT_EQ(cfg.ssm_dt_rank, 16);
    EXPECT_EQ(cfg.ssm_group_count, 16);
    EXPECT_EQ(cfg.ssm_state_size, 128);
    EXPECT_EQ(cfg.ssm_inner_size, 2048);
    EXPECT_EQ(cfg.ssm_conv_kernel, 4);
    EXPECT_EQ(cfg.ssm_conv_channels(), 3 * 2048);
    EXPECT_FLOAT_EQ(cfg.kda_lower_bound, -5.0f);
    // MLA: qk 128 + 64, v 128, q_lora 256, kv_lora 512.
    EXPECT_EQ(cfg.kv_lora_rank, 512);
    EXPECT_EQ(cfg.q_lora_rank, 256);
    EXPECT_EQ(cfg.head_dim, 192);
    EXPECT_EQ(cfg.rope_dim, 64);
    EXPECT_EQ(cfg.v_head_dim, 128);
    // Router: sigmoid + bias, 8 groups keep 4, normalised, * 2.5; one shared expert of 512.
    EXPECT_TRUE(cfg.moe_sigmoid_gating);
    EXPECT_TRUE(cfg.expert_weights_norm);
    EXPECT_EQ(cfg.moe_n_group, 8);
    EXPECT_EQ(cfg.moe_topk_group, 4);
    EXPECT_FLOAT_EQ(cfg.expert_weights_scale, 2.5f);
    EXPECT_EQ(cfg.n_experts, 128);
    EXPECT_EQ(cfg.n_experts_active, 8);
    EXPECT_EQ(cfg.expert_d_ff, 512);
    EXPECT_EQ(cfg.n_experts_shared, 1);
    EXPECT_EQ(cfg.expert_shared_d_ff, 512);
    EXPECT_EQ(cfg.first_k_dense_replace, 1);
    EXPECT_EQ(cfg.d_ff, 4608);
}

// Synthetic BF16 tensors with a recognisable byte per source tensor.
struct Store {
    std::vector<std::vector<uint16_t>> bufs;
    std::unordered_map<std::string, Tensor> map;
    void add(const std::string& name, std::initializer_list<int64_t> shape, uint16_t fill) {
        int64_t n = 1;
        for (int64_t d : shape)
            n *= d;
        bufs.emplace_back(static_cast<size_t>(n), fill);
        std::vector<int64_t> s(shape);
        map.emplace(name,
                    Tensor(bufs.back().data(), QType::BF16, static_cast<int>(s.size()), s.data(), false));
    }
};

ModelConfig small_cfg() {
    ModelConfig cfg;
    cfg.arch = ModelArch::BAILING_HYBRID;
    cfg.n_layers = 4;
    cfg.n_heads = 2;
    return cfg;
}

void add_kda_layer(Store& s, int layer, int hk, int d) {
    const std::string p = "model.layers." + std::to_string(layer) + ".attention.";
    s.add(p + "q_proj.weight", {hk, d}, 1);
    s.add(p + "k_proj.weight", {hk, d}, 2);
    s.add(p + "v_proj.weight", {hk, d}, 3);
    s.add(p + "q_conv1d.weight", {hk, 1, 4}, 4);
    s.add(p + "k_conv1d.weight", {hk, 1, 4}, 5);
    s.add(p + "v_conv1d.weight", {hk, 1, 4}, 6);
    s.add(p + "f_proj.weight", {hk, d}, 7);
    s.add(p + "g_proj.weight", {hk, d}, 8);
    s.add(p + "o_norm.weight", {8}, 9);
}

TEST(BailingHybridTensors, FusesKdaAndRenamesMla) {
    Store s;
    const int hk = 16, d = 8;  // 2 heads x 8 key dims
    add_kda_layer(s, 0, hk, d);
    s.add("model.layers.3.attention.g_proj.weight", {2, d}, 10);  // MLA head gate [n_heads, d]
    s.add("model.layers.3.attention.q_b_proj.weight", {6, 4}, 11);
    s.add("model.layers.3.attention.dense.weight", {d, 4}, 12);
    s.add("model.layers.1.mlp.gate.expert_bias", {4}, 13);
    s.add("model.word_embeddings.weight", {10, d}, 14);
    std::vector<void*> owned;
    arch_prepare_tensor_map(small_cfg(), s.map, owned);

    const auto& m = s.map;
    ASSERT_TRUE(m.contains("model.layers.0.linear_attn.in_proj_qkv.weight"));
    const Tensor& qkv = m.at("model.layers.0.linear_attn.in_proj_qkv.weight");
    EXPECT_EQ(qkv.shape[0], 3 * hk);
    EXPECT_EQ(qkv.shape[1], d);
    const auto* q = static_cast<const uint16_t*>(qkv.data);
    EXPECT_EQ(q[0], 1);                                          // rows [0, hk): q
    EXPECT_EQ(q[static_cast<size_t>(hk) * d], 2);                // then k
    EXPECT_EQ(q[(2 * static_cast<size_t>(hk) * d) + d - 1], 3);  // then v
    const Tensor& conv = m.at("model.layers.0.linear_attn.conv1d.weight");
    EXPECT_EQ(conv.shape[0], 3 * hk);
    EXPECT_EQ(conv.ndim, 3);
    EXPECT_EQ(static_cast<const uint16_t*>(conv.data)[static_cast<size_t>(2 * hk) * 4], 6);
    EXPECT_TRUE(m.contains("model.layers.0.linear_attn.in_proj_a.weight"));
    EXPECT_TRUE(m.contains("model.layers.0.linear_attn.in_proj_z.weight"));
    EXPECT_TRUE(m.contains("model.layers.0.linear_attn.norm.weight"));
    EXPECT_FALSE(m.contains("model.layers.0.attention.q_proj.weight"));
    EXPECT_TRUE(m.contains("model.layers.3.self_attn.head_gate.weight"));
    EXPECT_TRUE(m.contains("model.layers.3.self_attn.q_proj.weight"));
    EXPECT_TRUE(m.contains("model.layers.3.self_attn.o_proj.weight"));
    EXPECT_TRUE(m.contains("model.layers.1.mlp.gate.bias"));
    EXPECT_TRUE(m.contains("model.embed_tokens.weight"));
    EXPECT_EQ(owned.size(), 2u);
    for (void* p : owned)
        std::free(p);
}

TEST(BailingHybridTensors, RefusesIncompleteKdaLayer) {
    Store s;
    add_kda_layer(s, 0, 16, 8);
    s.map.erase("model.layers.0.attention.v_conv1d.weight");
    std::vector<void*> owned;
    EXPECT_THROW(arch_prepare_tensor_map(small_cfg(), s.map, owned), std::runtime_error);
}

TEST(BailingHybridTensors, OtherArchesUntouched) {
    Store s;
    add_kda_layer(s, 0, 16, 8);
    ModelConfig cfg = small_cfg();
    cfg.arch = ModelArch::DEEPSEEK;
    std::vector<void*> owned;
    arch_prepare_tensor_map(cfg, s.map, owned);
    EXPECT_TRUE(s.map.contains("model.layers.0.attention.q_proj.weight"));
    EXPECT_TRUE(owned.empty());
}

}  // namespace
}  // namespace imp
