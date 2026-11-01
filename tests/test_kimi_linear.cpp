// Kimi-Linear (kimi_linear) load path without the checkpoint: config.json parse (fixture
// tests/fixtures/hf_config_golden/syn_kimi_linear.json = Firworks/Kimi-Linear-48B-A3B-Instruct-nvfp4's
// config.json) and the whole-map tensor rewrite (model/kimi_linear_tensors.cpp) on synthetic tensors
// spelled like that checkpoint's model.safetensors.index.json (dense BF16 and NVFP4 q/k/v).

#include "model/hf_config_loader.h"
#include "model/model_config.h"
#include "model/weight_map_arch.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

namespace fs = std::filesystem;

std::string fixture_json() {
    std::ifstream in(fs::path(IMP_TEST_FIXTURES_DIR) / "hf_config_golden" / "syn_kimi_linear.json");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// load_config on `json` in a scratch dir.
bool load_json(const std::string& json, ModelConfig& cfg) {
    const fs::path dir = fs::temp_directory_path() / ("imp_test_kimi_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::ofstream(dir / "config.json") << json;
    const bool ok = HFConfigLoader::load_config(dir.string(), cfg);
    fs::remove_all(dir);
    return ok;
}

// The fixture with one substring replaced (it must occur).
std::string patched(const std::string& from, const std::string& to) {
    std::string j = fixture_json();
    const size_t at = j.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    if (at != std::string::npos)
        j.replace(at, from.size(), to);
    return j;
}

constexpr int kMlaLayers[] = {3, 7, 11, 15, 19, 23, 26};

TEST(KimiLinearConfig, MapsKdaMlaNopeAndRouter) {
    ModelConfig cfg;
    ASSERT_TRUE(load_json(fixture_json(), cfg));
    EXPECT_EQ(cfg.arch, ModelArch::KIMI_LINEAR);
    EXPECT_EQ(cfg.n_layers, 27);
    EXPECT_EQ(cfg.d_model, 2304);
    // KDA dims from linear_attn_config, not the top-level head_dim 72.
    EXPECT_TRUE(cfg.gdn_kda);
    EXPECT_TRUE(cfg.kda_softplus_gate);
    EXPECT_EQ(cfg.kda_lora_rank, 128);
    EXPECT_TRUE(cfg.gdn_gate_sigmoid);
    EXPECT_EQ(cfg.ssm_dt_rank, 32);
    EXPECT_EQ(cfg.ssm_group_count, 32);
    EXPECT_EQ(cfg.ssm_state_size, 128);
    EXPECT_EQ(cfg.ssm_inner_size, 4096);
    EXPECT_EQ(cfg.ssm_conv_kernel, 4);
    EXPECT_EQ(cfg.ssm_conv_channels(), 3 * 4096);
    // Layer partition: 1-based lists -> 0-based masks; MLA layers are NoPE.
    ASSERT_EQ(cfg.kda_layers.size(), 27u);
    ASSERT_EQ(cfg.nope_layers.size(), 27u);
    int n_kda = 0;
    for (int i = 0; i < 27; ++i) {
        const bool mla = std::find(std::begin(kMlaLayers), std::end(kMlaLayers), i) != std::end(kMlaLayers);
        EXPECT_EQ(cfg.kda_layers[static_cast<size_t>(i)] != 0, !mla) << "layer " << i;
        EXPECT_EQ(cfg.layer_nope(i), mla) << "layer " << i;
        n_kda += cfg.kda_layers[static_cast<size_t>(i)];
    }
    EXPECT_EQ(n_kda, 20);
    // MLA without q_lora: qk 128 + 64, v 128, kv_lora 512.
    EXPECT_EQ(cfg.kv_lora_rank, 512);
    EXPECT_EQ(cfg.q_lora_rank, 0);
    EXPECT_EQ(cfg.head_dim, 192);
    EXPECT_EQ(cfg.rope_dim, 64);
    EXPECT_EQ(cfg.v_head_dim, 128);
    // Router: sigmoid + e_score_correction_bias, renormalised, * 2.446, ungrouped (num_expert_group 1).
    EXPECT_TRUE(cfg.moe_sigmoid_gating);
    EXPECT_TRUE(cfg.expert_weights_norm);
    EXPECT_FLOAT_EQ(cfg.expert_weights_scale, 2.446f);
    EXPECT_EQ(cfg.moe_n_group, 1);
    EXPECT_EQ(cfg.n_experts, 256);
    EXPECT_EQ(cfg.n_experts_active, 8);
    EXPECT_EQ(cfg.expert_d_ff, 1024);
    EXPECT_EQ(cfg.n_experts_shared, 1);
    EXPECT_EQ(cfg.expert_shared_d_ff, 1024);
    EXPECT_EQ(cfg.first_k_dense_replace, 1);
    EXPECT_EQ(cfg.d_ff, 9216);
}

TEST(KimiLinearConfig, RefusesWhatTheScansCannotServe) {
    ModelConfig cfg;
    EXPECT_FALSE(load_json(patched("\"head_dim\": 128,", "\"head_dim\": 64,"), cfg)) << "KDA head_dim 64";
    ModelConfig c2;
    EXPECT_FALSE(load_json(patched("\"kda_layers\": [1, 2, 3,", "\"kda_layers\": [1, 2, 4,"), c2))
        << "layer 4 both KDA and full attention, layer 3 neither";
    ModelConfig c3;
    EXPECT_FALSE(load_json(patched("\"linear_attn_config\"", "\"linear_attn_cfg\""), c3))
        << "no linear_attn_config";
    ModelConfig c4;
    EXPECT_FALSE(load_json(patched("\"moe_layer_freq\": 1", "\"moe_layer_freq\": 2"), c4));
    ModelConfig c5;
    EXPECT_FALSE(load_json(patched("\"moe_router_activation_func\": \"sigmoid\"",
                                   "\"moe_router_activation_func\": \"softmax\""),
                           c5));
}

// Synthetic tensors with a recognisable fill per source tensor.
struct Store {
    std::vector<std::vector<uint16_t>> bufs;
    std::unordered_map<std::string, Tensor> map;
    void add(const std::string& name, std::initializer_list<int64_t> shape, uint16_t fill,
             QType q = QType::BF16) {
        int64_t n = 1;
        for (int64_t d : shape)
            n *= d;
        bufs.emplace_back(static_cast<size_t>(n), fill);
        std::vector<int64_t> s(shape);
        map.emplace(name, Tensor(bufs.back().data(), q, static_cast<int>(s.size()), s.data(), false));
    }
};

// 4 layers: 0-2 KDA, 3 MLA (the fixture's first group).
ModelConfig small_cfg() {
    ModelConfig cfg;
    cfg.arch = ModelArch::KIMI_LINEAR;
    cfg.n_layers = 4;
    cfg.n_heads = 2;
    cfg.kda_layers = {1, 1, 1, 0};
    return cfg;
}

const char* const kNvfp4Suffixes[] = {".weight", ".weight_scale", ".weight_scale_2", ".input_scale"};

// One KDA layer in the checkpoint spelling (after llm-compressor's .weight_packed -> .weight rename).
void add_kda_layer(Store& s, int layer, int hk, int d, bool nvfp4) {
    const std::string p = "model.layers." + std::to_string(layer) + ".self_attn.";
    const char* proj[] = {"q_proj",   "k_proj",   "v_proj", "f_a_proj", "f_b_proj",
                          "g_a_proj", "g_b_proj", "b_proj", "o_proj"};
    uint16_t fill = 1;
    for (const char* m : proj) {
        if (nvfp4) {
            for (const char* sfx : kNvfp4Suffixes)
                s.add(p + m + sfx, {hk, d / 2}, fill,
                      std::string(sfx) == ".weight" ? QType::INT8 : QType::FP8_E4M3);
        } else {
            s.add(p + m + ".weight", {hk, d}, fill);
        }
        ++fill;
    }
    s.add(p + "q_conv1d.weight", {hk, 1, 4}, 20);
    s.add(p + "k_conv1d.weight", {hk, 1, 4}, 21);
    s.add(p + "v_conv1d.weight", {hk, 1, 4}, 22);
    s.add(p + "o_norm.weight", {8}, 23);
    s.add(p + "A_log", {1, 1, 2, 1}, 24, QType::F32);
    s.add(p + "dt_bias", {hk}, 25, QType::F32);
}

void add_mla_and_moe(Store& s, int d) {
    s.add("model.layers.3.self_attn.q_proj.weight", {6, d}, 30);
    s.add("model.layers.3.self_attn.kv_a_proj_with_mqa.weight", {5, d}, 31);
    s.add("model.layers.3.self_attn.kv_a_layernorm.weight", {4}, 32);
    s.add("model.layers.3.self_attn.kv_b_proj.weight", {6, 4}, 33);
    s.add("model.layers.3.self_attn.o_proj.weight", {d, 4}, 34);
    s.add("model.layers.1.block_sparse_moe.gate.weight", {4, d}, 40);
    s.add("model.layers.1.block_sparse_moe.gate.e_score_correction_bias", {4}, 41, QType::F32);
    s.add("model.layers.1.block_sparse_moe.experts.7.w1.weight", {2, d}, 42);
    s.add("model.layers.1.block_sparse_moe.experts.7.w2.weight_scale", {d, 1}, 43);
    s.add("model.layers.1.block_sparse_moe.experts.7.w3.input_scale", {1, 1}, 44, QType::F32);
    s.add("model.layers.1.block_sparse_moe.shared_experts.down_proj.weight", {d, 2}, 45);
}

TEST(KimiLinearTensors, DenseFusesQkvRenamesLoraAndMoe) {
    Store s;
    const int hk = 16, d = 8;
    for (int l = 0; l < 3; ++l)
        add_kda_layer(s, l, hk, d, /*nvfp4=*/false);
    add_mla_and_moe(s, d);
    std::vector<void*> owned;
    arch_prepare_tensor_map(small_cfg(), s.map, owned);

    const auto& m = s.map;
    const std::string la = "model.layers.0.linear_attn.";
    ASSERT_TRUE(m.contains(la + "in_proj_qkv.weight"));
    const Tensor& qkv = m.at(la + "in_proj_qkv.weight");
    EXPECT_EQ(qkv.shape[0], 3 * hk);
    const auto* q = static_cast<const uint16_t*>(qkv.data);
    EXPECT_EQ(q[0], 1);                                          // rows [0, hk): q
    EXPECT_EQ(q[static_cast<size_t>(hk) * d], 2);                // then k
    EXPECT_EQ(q[(2 * static_cast<size_t>(hk) * d) + d - 1], 3);  // then v
    EXPECT_FALSE(m.contains(la + "in_proj_q.weight"));
    const Tensor& conv = m.at(la + "conv1d.weight");
    EXPECT_EQ(conv.shape[0], 3 * hk);
    EXPECT_EQ(static_cast<const uint16_t*>(conv.data)[static_cast<size_t>(2 * hk) * 4], 22);
    // LoRA: down-projections to their own slots, up-projections onto the GDN alpha / gate slots.
    EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "f_a_proj.weight").data)[0], 4);
    EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "in_proj_a.weight").data)[0], 5);
    EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "g_a_proj.weight").data)[0], 6);
    EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "in_proj_z.weight").data)[0], 7);
    EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "in_proj_b.weight").data)[0], 8);
    EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "out_proj.weight").data)[0], 9);
    EXPECT_TRUE(m.contains(la + "norm.weight"));
    EXPECT_TRUE(m.contains(la + "dt_bias"));
    const Tensor& a_log = m.at(la + "A_log");
    EXPECT_EQ(a_log.ndim, 1);
    EXPECT_EQ(a_log.shape[0], 2);
    EXPECT_FALSE(m.contains("model.layers.0.self_attn.q_proj.weight"));
    // MLA layer untouched; MoE in the DeepSeek spelling.
    EXPECT_TRUE(m.contains("model.layers.3.self_attn.q_proj.weight"));
    EXPECT_TRUE(m.contains("model.layers.3.self_attn.kv_a_proj_with_mqa.weight"));
    EXPECT_TRUE(m.contains("model.layers.3.self_attn.o_proj.weight"));
    EXPECT_TRUE(m.contains("model.layers.1.mlp.gate.weight"));
    EXPECT_TRUE(m.contains("model.layers.1.mlp.gate.bias"));
    EXPECT_TRUE(m.contains("model.layers.1.mlp.experts.7.gate_proj.weight"));
    EXPECT_TRUE(m.contains("model.layers.1.mlp.experts.7.down_proj.weight_scale"));
    EXPECT_TRUE(m.contains("model.layers.1.mlp.experts.7.up_proj.input_scale"));
    EXPECT_TRUE(m.contains("model.layers.1.mlp.shared_experts.down_proj.weight"));
    EXPECT_EQ(owned.size(), 6u);  // 3 layers x (q|k|v, conv)
    for (void* p : owned)
        std::free(p);
}

// NVFP4 q/k/v carry three global scales: three tensors with every scale suffix, convs still fused.
TEST(KimiLinearTensors, Nvfp4KeepsQkvSplitWithTheirScales) {
    Store s;
    const int hk = 16, d = 32;
    for (int l = 0; l < 3; ++l)
        add_kda_layer(s, l, hk, d, /*nvfp4=*/true);
    add_mla_and_moe(s, d);
    std::vector<void*> owned;
    arch_prepare_tensor_map(small_cfg(), s.map, owned);
    const auto& m = s.map;
    const std::string la = "model.layers.2.linear_attn.";
    EXPECT_FALSE(m.contains(la + "in_proj_qkv.weight"));
    for (const char* sfx : kNvfp4Suffixes) {
        EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "in_proj_q" + sfx).data)[0], 1) << sfx;
        EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "in_proj_k" + sfx).data)[0], 2) << sfx;
        EXPECT_EQ(static_cast<const uint16_t*>(m.at(la + "in_proj_v" + sfx).data)[0], 3) << sfx;
        EXPECT_TRUE(m.contains(la + "f_a_proj" + sfx)) << sfx;
        EXPECT_TRUE(m.contains(la + "in_proj_z" + sfx)) << sfx;
    }
    EXPECT_TRUE(m.contains(la + "conv1d.weight"));
    EXPECT_EQ(owned.size(), 3u);  // convs only
    for (void* p : owned)
        std::free(p);
}

TEST(KimiLinearTensors, RefusesBrokenLayouts) {
    {
        Store s;
        for (int l = 0; l < 3; ++l)
            add_kda_layer(s, l, 16, 8, false);
        s.map.erase("model.layers.1.self_attn.k_conv1d.weight");
        std::vector<void*> owned;
        EXPECT_THROW(arch_prepare_tensor_map(small_cfg(), s.map, owned), std::runtime_error)
            << "missing conv";
        for (void* p : owned)
            std::free(p);
    }
    {
        Store s;
        for (int l = 0; l < 3; ++l)
            add_kda_layer(s, l, 16, 8, false);
        s.add("model.layers.3.self_attn.A_log", {1, 1, 2, 1}, 1, QType::F32);
        std::vector<void*> owned;
        EXPECT_THROW(arch_prepare_tensor_map(small_cfg(), s.map, owned), std::runtime_error)
            << "KDA tensor in an MLA layer";
    }
    {
        Store s;
        for (int l = 0; l < 3; ++l)
            add_kda_layer(s, l, 16, 8, false);
        s.map.erase("model.layers.0.self_attn.v_proj.weight");
        s.add("model.layers.0.self_attn.v_proj.weight", {16, 4}, 3, QType::INT8);
        s.add("model.layers.0.self_attn.v_proj.weight_scale", {16, 1}, 3, QType::FP8_E4M3);
        std::vector<void*> owned;
        EXPECT_THROW(arch_prepare_tensor_map(small_cfg(), s.map, owned), std::runtime_error)
            << "dense q/k with a quantized v";
        for (void* p : owned)
            std::free(p);
    }
    {
        Store s;
        add_kda_layer(s, 0, 16, 8, false);
        ModelConfig cfg = small_cfg();
        cfg.kda_layers.clear();
        std::vector<void*> owned;
        EXPECT_THROW(arch_prepare_tensor_map(cfg, s.map, owned), std::runtime_error) << "no KDA layer list";
    }
}

// The slots the weight map assigns the rewritten names to, and the NVFP4 scale keys phase 0 resolves.
TEST(KimiLinearTensors, WeightMapSlots) {
    TransformerLayer L;
    EXPECT_EQ(kimi_kda_slot(L, "in_proj_q"), &L.ssm_in);
    EXPECT_EQ(kimi_kda_slot(L, "in_proj_k"), &L.kda_k_proj);
    EXPECT_EQ(kimi_kda_slot(L, "in_proj_v"), &L.kda_v_proj);
    EXPECT_EQ(kimi_kda_slot(L, "f_a_proj"), &L.kda_f_a);
    EXPECT_EQ(kimi_kda_slot(L, "g_a_proj"), &L.kda_g_a);
    EXPECT_EQ(kimi_kda_slot(L, "in_proj_a"), nullptr);
    EXPECT_STREQ(kimi_kda_scale_slot("in_proj_q"), "ssm_in");
    EXPECT_STREQ(kimi_kda_scale_slot("in_proj_k"), "kda_k_proj");
    EXPECT_STREQ(kimi_kda_scale_slot("in_proj_v"), "kda_v_proj");
    EXPECT_STREQ(kimi_kda_scale_slot("f_a_proj"), "kda_f_a");
    EXPECT_STREQ(kimi_kda_scale_slot("g_a_proj"), "kda_g_a");
    EXPECT_EQ(kimi_kda_scale_slot("in_proj_qkv"), nullptr);
}

}  // namespace
}  // namespace imp
