// Flint (RecursiveSI/flint): config.json -> ModelConfig / ModelProfile, and the two tensor renames.
#include <gtest/gtest.h>

#include "model/hf_config_loader.h"
#include "model/model.h"
#include "model/model_arch.h"
#include "model/model_config.h"
#include "model/model_profile.h"
#include "model/nvfp4_undeclared.h"
#include "model/weight_map_arch.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

using imp::ModelArch;

// The shipped config.json cut to 6 layers: full (NoPE), full (NoPE), sliding, sliding, sliding, full (NoPE).
constexpr const char* kConfig = R"({
    "architectures": ["FlintForCausalLM"], "model_type": "flint",
    "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4, "num_hidden_layers": 6,
    "head_dim": 128, "hidden_act": "silu", "intermediate_size": 6144, "moe_intermediate_size": 512,
    "num_experts": 256, "num_experts_per_tok": 6, "shared_expert_intermediate_size": 1024,
    "norm_topk_prob": true, "router_score_function": "sigmoid", "softmax_type": "learnable",
    "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": 513, "vocab_size": 150144,
    "max_position_embeddings": 262144, "tie_word_embeddings": false, "mlp_only_layers": [0, 1],
    "no_rope_layer_indices": [0, 1, 5],
    "layer_types": ["full_attention", "full_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention"]
})";

class FlintConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("imp_flint_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        std::filesystem::create_directories(dir_);
    }
    void TearDown() override { std::filesystem::remove_all(dir_); }
    void write(const std::string& json) const { std::ofstream(dir_ / "config.json") << json; }
    static std::string replaced(std::string s, const std::string& from, const std::string& to) {
        const size_t at = s.find(from);
        EXPECT_NE(at, std::string::npos) << from;
        return s.replace(at, from.size(), to);
    }
    std::filesystem::path dir_;
};

TEST_F(FlintConfigTest, LayerMapRouterAndSharedExpert) {
    write(kConfig);
    imp::ModelConfig cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_config(dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, ModelArch::FLINT);
    EXPECT_EQ(cfg.swa_layers, (std::vector<uint8_t>{0, 0, 1, 1, 1, 0}));
    EXPECT_EQ(cfg.nope_layers, (std::vector<uint8_t>{1, 1, 0, 0, 0, 1}));
    EXPECT_EQ(cfg.sliding_window, 513) << "the current token and 512 before it";
    EXPECT_TRUE(cfg.moe_sigmoid_gating);
    EXPECT_TRUE(cfg.expert_weights_norm);
    EXPECT_EQ(cfg.n_experts, 256);
    EXPECT_EQ(cfg.n_experts_active, 6);
    EXPECT_EQ(cfg.d_ff, 6144) << "the dense layers";
    EXPECT_EQ(cfg.expert_d_ff, 512);
    EXPECT_EQ(cfg.n_experts_shared, 1);
    EXPECT_EQ(cfg.expert_shared_d_ff, 1024);
    EXPECT_EQ(cfg.head_dim, 128);
    EXPECT_FLOAT_EQ(cfg.rope_theta, 10000.0f);
}

TEST_F(FlintConfigTest, NormTopkOffAndSoftmaxRouterFollowTheConfig) {
    write(replaced(replaced(kConfig, "\"norm_topk_prob\": true", "\"norm_topk_prob\": false"),
                   "\"router_score_function\": \"sigmoid\"", "\"router_score_function\": \"softmax\""));
    imp::ModelConfig cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_config(dir_.string(), cfg));
    imp::apply_arch_defaults(cfg);
    EXPECT_FALSE(cfg.expert_weights_norm) << "the registry row must not force the renormalization on";
}

TEST_F(FlintConfigTest, RefusesWhatTheBuildCannotServe) {
    imp::ModelConfig cfg;
    write(replaced(kConfig, "\"softmax_type\": \"learnable\"", "\"softmax_type\": \"standard\""));
    EXPECT_FALSE(imp::HFConfigLoader::load_config(dir_.string(), cfg)) << "a Flint without sinks";
    write(replaced(kConfig, "\"router_score_function\": \"sigmoid\"",
                   "\"router_score_function\": \"sparsemax\""));
    EXPECT_FALSE(imp::HFConfigLoader::load_config(dir_.string(), cfg));
    write(replaced(kConfig, "\"no_rope_layer_indices\": [0, 1, 5]", "\"no_rope_layer_indices\": [0, 1, 6]"));
    EXPECT_FALSE(imp::HFConfigLoader::load_config(dir_.string(), cfg)) << "layer index past the stack";
    write(replaced(kConfig, "\"sliding_window\": 513", "\"sliding_window\": 0"));
    EXPECT_FALSE(imp::HFConfigLoader::load_config(dir_.string(), cfg)) << "sliding layers without a window";
    write(replaced(kConfig, "\"sliding_attention\", \"full_attention\"]", "\"sliding_attention\"]"));
    EXPECT_FALSE(imp::HFConfigLoader::load_config(dir_.string(), cfg))
        << "layer_types shorter than the stack";
}

TEST(FlintNames, SinkAndExpertBiasTakeTheMapsNames) {
    std::string sink = "model.layers.7.self_attn.softmax_offset";
    ASSERT_TRUE(imp::arch_canonical_name(ModelArch::FLINT, 49, sink));
    EXPECT_EQ(sink, "model.layers.7.self_attn.sinks");
    std::string bias = "model.layers.7.mlp.gate.expert_bias";
    ASSERT_TRUE(imp::arch_canonical_name(ModelArch::FLINT, 49, bias));
    EXPECT_EQ(bias, "model.layers.7.mlp.gate.bias");
    std::string plain = "model.layers.7.mlp.gate.weight";
    ASSERT_TRUE(imp::arch_canonical_name(ModelArch::FLINT, 49, plain));
    EXPECT_EQ(plain, "model.layers.7.mlp.gate.weight");
    std::string other = "model.layers.7.self_attn.softmax_offset";
    ASSERT_TRUE(imp::arch_canonical_name(ModelArch::QWEN3_MOE, 49, other));
    EXPECT_EQ(other, "model.layers.7.self_attn.softmax_offset") << "the rename is Flint's alone";
}

// NVFP4 tensors without a declaration the build reads are refused, not run as weights.
TEST(Nvfp4Undeclared, PackedCodesWithoutADeclarationAreRefused) {
    const std::map<std::string, int> nvfp4 = {{"model.layers.0.self_attn.q_proj.weight", 0},
                                              {"model.layers.0.self_attn.q_proj.weight_scale", 0},
                                              {"model.layers.0.self_attn.q_proj.weight_scale_2", 0},
                                              {"model.norm.weight", 0}};
    EXPECT_EQ(imp::first_undeclared_nvfp4_module(nvfp4), "model.layers.0.self_attn.q_proj");
    EXPECT_TRUE(imp::nvfp4_undeclared(nvfp4, /*declared=*/false, "/models/x"));
    EXPECT_FALSE(imp::nvfp4_undeclared(nvfp4, /*declared=*/true, "/models/x"));
    const std::map<std::string, int> fp8 = {{"model.layers.0.self_attn.q_proj.weight", 0},
                                            {"model.layers.0.self_attn.q_proj.weight_scale", 0}};
    EXPECT_FALSE(imp::nvfp4_undeclared(fp8, false, "/models/x")) << "a per-tensor scale alone is not NVFP4";
    const std::map<std::string, int> stray = {{"lm_head.weight_scale_2", 0}, {"lm_head.weight", 0}};
    EXPECT_FALSE(imp::nvfp4_undeclared(stray, false, "/models/x")) << "no block scales: not the NVFP4 triple";
}

}  // namespace
