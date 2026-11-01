// Gemma-4 E-variants (google/gemma-4-E4B-it): config parse of the per-layer-input (PLE) and
// KV-sharing keys, and the weight mapping of the PLE tensors / the unused k,v of shared layers.
// Fixture: tests/fixtures/hf_config_golden/syn_gemma4_e4b.json = the checkpoint's config.json.
// Source mapping checked against transformers 5.20 modeling_gemma4.py (Gemma4TextAttention).

#include "model/hf_config_loader.h"
#include "model/model.h"
#include "model/weight_map.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

namespace imp {
namespace {

namespace fs = std::filesystem;

class Gemma4EConfig : public ::testing::Test {
protected:
    fs::path dir_;
    void SetUp() override {
        dir_ = fs::temp_directory_path() / ("imp_test_gemma4_e_" + std::to_string(::getpid()));
        fs::create_directories(dir_);
    }
    void TearDown() override { fs::remove_all(dir_); }

    static std::string fixture() {
        std::ifstream in(fs::path(IMP_TEST_FIXTURES_DIR) / "hf_config_golden" / "syn_gemma4_e4b.json");
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    bool load(const std::string& json, ModelConfig& cfg) const {
        std::ofstream(dir_ / "config.json") << json;
        return HFConfigLoader::load_config(dir_.string(), cfg);
    }
};

TEST_F(Gemma4EConfig, E4BPerLayerInputsAndKvSources) {
    ModelConfig cfg;
    ASSERT_TRUE(load(fixture(), cfg));
    EXPECT_EQ(cfg.arch, ModelArch::GEMMA4);
    EXPECT_EQ(cfg.n_layers, 42);
    EXPECT_EQ(cfg.d_model, 2560);
    EXPECT_EQ(cfg.pli_dim, 256);
    EXPECT_EQ(cfg.pli_vocab, 262144);
    EXPECT_EQ(cfg.head_dim, 512);  // widened to the full-attention head dim
    ASSERT_EQ(cfg.n_kv_heads_per_layer.size(), 42u);
    for (int v : cfg.n_kv_heads_per_layer)
        EXPECT_EQ(v, 2);  // num_global_key_value_heads null -> num_key_value_heads
    ASSERT_EQ(cfg.kv_source_layer.size(), 42u);
    int shared = 0;
    for (int l = 0; l < 42; ++l) {
        const bool full = (l % 6) == 5;
        EXPECT_EQ(cfg.swa_layers[static_cast<size_t>(l)], full ? 0 : 1) << "layer " << l;
        if (l < 24) {
            EXPECT_FALSE(cfg.layer_kv_shared(l)) << "layer " << l;
        } else {
            // last non-shared layer of the same type: 22 sliding, 23 full
            EXPECT_EQ(cfg.kv_source(l), full ? 23 : 22) << "layer " << l;
            ++shared;
        }
    }
    EXPECT_EQ(shared, 18);
}

TEST_F(Gemma4EConfig, DoubleWideMlpIsRefused) {
    std::string json = fixture();
    const std::string key = "\"use_double_wide_mlp\": false";
    const size_t at = json.find(key);
    ASSERT_NE(at, std::string::npos);
    json.replace(at, key.size(), "\"use_double_wide_mlp\": true");
    ModelConfig cfg;
    EXPECT_FALSE(load(json, cfg));
}

TEST_F(Gemma4EConfig, DenseGemma4HasNoPerLayerInputs) {
    std::string json = fixture();
    for (const char* key : {"\"hidden_size_per_layer_input\": 256", "\"num_kv_shared_layers\": 18"}) {
        const size_t at = json.find(key);
        ASSERT_NE(at, std::string::npos) << key;
        json.replace(at, std::string(key).size(),
                     std::string(key).substr(0, std::string(key).find(':')) + ": 0");
    }
    ModelConfig cfg;
    ASSERT_TRUE(load(json, cfg));
    EXPECT_EQ(cfg.pli_dim, 0);
    EXPECT_EQ(cfg.pli_vocab, 0);
    EXPECT_TRUE(cfg.kv_source_layer.empty());
    EXPECT_FALSE(cfg.layer_kv_shared(41));
}

TEST(Gemma4KvSources, EdgeCases) {
    const std::vector<uint8_t> types = {1, 1, 0, 1, 0, 1};  // s s f s f s
    EXPECT_TRUE(gemma4_kv_source_layers(types, 0).empty());
    EXPECT_TRUE(gemma4_kv_source_layers(types, 6).empty());  // nothing left to share from
    const std::vector<int> two = gemma4_kv_source_layers(types, 2);
    EXPECT_EQ(two, (std::vector<int>{-1, -1, -1, -1, 2, 3}));
    // 3 shared: layer 3 (sliding) is shared, sources must come from 0..2
    EXPECT_EQ(gemma4_kv_source_layers(types, 3), (std::vector<int>{-1, -1, -1, 1, 2, 1}));
    // no earlier full layer for a shared full layer
    EXPECT_TRUE(gemma4_kv_source_layers({1, 1, 0}, 1).empty());
}

// Names and roles from the E4B checkpoint header (model.safetensors).
struct WeightFixture {
    std::vector<uint16_t> backing = std::vector<uint16_t>(1024, 0);
    Model model;
    std::unordered_map<std::string, Tensor> tensors;

    WeightFixture() {
        model.config_.arch = ModelArch::GEMMA4;
        model.config_.n_layers = 3;
        model.config_.d_model = 8;
        model.config_.n_experts = 0;
        model.config_.pli_dim = 4;
        model.config_.kv_source_layer = {-1, -1, 1};
        model.layers_.resize(3);
    }
    void add(const std::string& name) {
        Tensor t;
        t.data = backing.data();
        t.qtype = QType::F16;
        t.ndim = 2;
        t.shape[0] = 8;
        t.shape[1] = 8;
        tensors["model.language_model." + name] = t;
    }
};

TEST(Gemma4EWeights, PerLayerInputsAndSharedKvMapping) {
    WeightFixture f;
    f.add("embed_tokens.weight");
    f.add("norm.weight");
    f.add("embed_tokens_per_layer.weight");
    f.add("per_layer_model_projection.weight");
    f.add("per_layer_projection_norm.weight");
    for (int l = 0; l < 3; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        for (const char* s : {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                              "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                              "per_layer_input_gate.weight", "per_layer_projection.weight",
                              "post_per_layer_input_norm.weight", "layer_scalar"})
            f.add(p + s);
    }
    f.tensors["model.audio_tower.layers.0.lconv1d.linear_start.linear.weight"] =
        f.tensors["model.language_model.norm.weight"];

    WeightMap wm(ModelArch::GEMMA4);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));
    EXPECT_NE(f.model.pli_table().data, nullptr);
    EXPECT_NE(f.model.pli_model_proj().data, nullptr);
    EXPECT_NE(f.model.pli_proj_norm().data, nullptr);
    for (int l = 0; l < 3; ++l) {
        const TransformerLayer& L = f.model.layers_[static_cast<size_t>(l)];
        EXPECT_NE(L.pli_gate.data, nullptr) << l;
        EXPECT_NE(L.pli_proj.data, nullptr) << l;
        EXPECT_NE(L.pli_norm.data, nullptr) << l;
        EXPECT_NE(L.wq.data, nullptr) << l;
        EXPECT_NE(L.attn_q_norm.data, nullptr) << l;
        EXPECT_NE(L.layer_out_scale.data, nullptr) << l;
        const bool shared = l == 2;
        EXPECT_EQ(L.wk.data == nullptr, shared) << l;
        EXPECT_EQ(L.wv.data == nullptr, shared) << l;
        EXPECT_EQ(L.attn_k_norm.data == nullptr, shared) << l;
    }
    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.kv_shared, 3);  // k_proj, v_proj, k_norm of layer 2
    EXPECT_EQ(s.audio, 1);
    EXPECT_EQ(s.unrecognised, 0);
    EXPECT_EQ(s.total, 4);
}

}  // namespace
}  // namespace imp
