// tests/test_mla.cpp — MLA config parsing tests (Task 0.1 + 0.2)
//                      MoE weight-name mapping tests (Task 1.1)
//
// Tests that HFConfigLoader::load_config correctly parses DeepSeek-V2/V3
// Multi-head Latent Attention (MLA) fields from config.json, and that
// derive_model_profile() selects AttnVariant::MLA for MLA configs.
//
// Task 1.1 tests call WeightMap::map_name() directly — a public method —
// to verify DeepSeek-V2 MoE tensor name routing without loading the 30 GB
// model. Seam: WeightMap(ModelArch) + map_name(std::string) (weight_map.h).
//
// Requires a DeepSeek-V2-Lite HF model directory with config.json.
// Set IMP_TEST_MODEL_DEEPSEEK to the directory path, or place the model at
// /models/DeepSeek-V2-Lite (Docker bind-mount fallback). Skipped if absent.

#include "model/hf_config_loader.h"
#include "model/model.h"
#include "model/model_config.h"
#include "model/model_profile.h"
#include "model/tensor_kind_matcher.h"
#include "model/weight_map.h"
#include "test_models.h"

#include <gtest/gtest.h>

#include <filesystem>

using imp::HFConfigLoader;
using imp::ModelConfig;
using imp::ModelArch;
using imp::ModelProfile;
using imp::Model;
using imp::WeightMap;

namespace {

// ---------------------------------------------------------------------------
// Task 2.5: YaRN mscale attention-scale multiplier
// ---------------------------------------------------------------------------

TEST(MLAConfig, YarnMscaleAttentionScaleNonMLA) {
    // Default-constructed config: not MLA, multiplier must be exactly 1.0.
    ModelConfig cfg;
    EXPECT_FLOAT_EQ(imp::mla_attention_scale_multiplier(cfg), 1.0f);
}

TEST(MLAConfig, YarnMscaleAttentionScale) {
    std::string dir = imp_test::env_path_or(imp_test::kEnvModelDeepSeek,
                                            "/models/DeepSeek-V2-Lite");
    if (!std::filesystem::exists(dir)) {
        GTEST_SKIP() << "Set IMP_TEST_MODEL_DEEPSEEK or place model at "
                     << dir << " to run MLA mscale tests";
    }

    ModelConfig cfg;
    bool ok = HFConfigLoader::load_config(dir, cfg);
    ASSERT_TRUE(ok) << "Failed to load config from " << dir;
    ASSERT_TRUE(cfg.is_mla());

    // DeepSeek-V2-Lite: mscale_all_dim=0.707, factor=40.
    // mscale_adj = 0.1 * 0.707 * ln(40) + 1.0 ≈ 1.26076
    // multiplier  = mscale_adj^2 ≈ 1.5895
    float mult = imp::mla_attention_scale_multiplier(cfg);
    EXPECT_NEAR(mult, 1.261f * 1.261f, 1e-2f)
        << "YaRN mscale multiplier mismatch (expected ~1.59)";
}

// ---------------------------------------------------------------------------

TEST(MLAConfig, ParsesDeepSeekV2LiteFields) {
    std::string dir = imp_test::env_path_or(imp_test::kEnvModelDeepSeek,
                                            "/models/DeepSeek-V2-Lite");
    if (!std::filesystem::exists(dir)) {
        GTEST_SKIP() << "Set IMP_TEST_MODEL_DEEPSEEK or place model at "
                     << dir << " to run MLA config tests";
    }

    ModelConfig cfg;
    bool ok = HFConfigLoader::load_config(dir, cfg);
    ASSERT_TRUE(ok) << "Failed to load config from " << dir;

    EXPECT_EQ(cfg.arch, ModelArch::DEEPSEEK);
    EXPECT_TRUE(cfg.is_mla());

    EXPECT_EQ(cfg.kv_lora_rank, 512);
    EXPECT_EQ(cfg.q_lora_rank, 0);       // field absent in V2-Lite -> stays 0
    EXPECT_EQ(cfg.qk_rope_head_dim, 64);
    EXPECT_EQ(cfg.qk_nope_head_dim, 128);
    EXPECT_EQ(cfg.v_head_dim, 128);

    // head_dim overridden to nope+rope = 128+64 = 192 for MLA models
    EXPECT_EQ(cfg.head_dim, 192);

    // mla_mscale: raw 0.707 from rope_scaling.mscale
    // (yarn-adjusted attention scale formula is Task 2.5)
    EXPECT_NEAR(cfg.mla_mscale, 0.707f, 1e-3f);

    // MoE shared experts
    EXPECT_EQ(cfg.n_experts_shared, 2);

    // First k dense layers (hybrid MoE)
    EXPECT_EQ(cfg.first_k_dense_replace, 1);
}

TEST(MLAConfig, IsMlaReturnsFalseForNonMLA) {
    // Default-constructed ModelConfig has kv_lora_rank == 0 -> not MLA
    ModelConfig cfg;
    EXPECT_EQ(cfg.kv_lora_rank, 0);
    EXPECT_FALSE(cfg.is_mla());
}

TEST(MLAConfig, ProfileSelectsMLAVariant) {
    std::string dir = imp_test::env_path_or(imp_test::kEnvModelDeepSeek,
                                            "/models/DeepSeek-V2-Lite");
    if (!std::filesystem::exists(dir)) {
        GTEST_SKIP() << "Set IMP_TEST_MODEL_DEEPSEEK or place model at "
                     << dir << " to run MLA profile tests";
    }

    ModelConfig cfg;
    bool ok = HFConfigLoader::load_config(dir, cfg);
    ASSERT_TRUE(ok) << "Failed to load config from " << dir;
    ASSERT_TRUE(cfg.is_mla()) << "Expected DeepSeek-V2-Lite to be an MLA model";

    // derive_model_profile requires a Model for layer scanning (GDN/SSM
    // classification); an empty model (no layers loaded) is sufficient here
    // because the MLA branch only keys on cfg.is_mla().
    Model m;
    m.config_ = cfg;
    ModelProfile prof = derive_model_profile(m, cfg);
    EXPECT_EQ(prof.attn_variant, ModelProfile::AttnVariant::MLA);
}

// ---------------------------------------------------------------------------
// Task 1.1: DeepSeek-V2 MoE weight-name mapping (WeightMap::map_name seam)
//
// Verifies that map_name() routes all DeepSeek-V2 MoE tensor names to the
// correct internal slot strings without loading any model weights.
// ---------------------------------------------------------------------------

TEST(MLAWeightMap, DeepSeekDenseLayers) {
    // Layer 0 is dense (first_k_dense_replace=1): mlp.{gate,up,down}_proj.weight
    WeightMap wm(ModelArch::DEEPSEEK);
    EXPECT_EQ(wm.map_name("model.layers.0.mlp.gate_proj.weight"), "layer.0.w_gate");
    EXPECT_EQ(wm.map_name("model.layers.0.mlp.up_proj.weight"),   "layer.0.w_up");
    EXPECT_EQ(wm.map_name("model.layers.0.mlp.down_proj.weight"), "layer.0.w_down");
}

TEST(MLAWeightMap, DeepSeekMoERouter) {
    // Layer 1+: MoE router is mlp.gate.weight (NOT gate_proj — that is the
    // expert SwiGLU gate; "gate" alone = router).
    WeightMap wm(ModelArch::DEEPSEEK);
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.gate.weight"), "layer.1.moe_gate");
    // Sanity: dense gate_proj on a MoE layer still maps to w_gate (shouldn't
    // appear in practice for layer>=1, but the mapper must not corrupt it).
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.gate_proj.weight"), "layer.1.w_gate");
}

TEST(MLAWeightMap, DeepSeekRoutedExperts) {
    // Routed experts: mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight
    WeightMap wm(ModelArch::DEEPSEEK);
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.experts.3.gate_proj.weight"),
              "layer.1.expert.3.w_gate");
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.experts.3.up_proj.weight"),
              "layer.1.expert.3.w_up");
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.experts.3.down_proj.weight"),
              "layer.1.expert.3.w_down");
    // Boundary: expert 63 (last of 64 in V2-Lite)
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.experts.63.gate_proj.weight"),
              "layer.1.expert.63.w_gate");
}

TEST(MLAWeightMap, DeepSeekSharedExperts) {
    // Shared experts use PLURAL "shared_experts" (not "shared_expert"):
    // mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight
    WeightMap wm(ModelArch::DEEPSEEK);
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.shared_experts.gate_proj.weight"),
              "layer.1.w_gate_shared");
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.shared_experts.up_proj.weight"),
              "layer.1.w_up_shared");
    EXPECT_EQ(wm.map_name("model.layers.1.mlp.shared_experts.down_proj.weight"),
              "layer.1.w_down_shared");
}

// ---------------------------------------------------------------------------
// Task 2.2: MLA attention projection name mapping (WeightMap::map_name seam)
//
// Verifies that map_name() routes DeepSeek-V2 MLA self_attn tensor names to
// the correct internal slot strings without loading any model weights.
// ---------------------------------------------------------------------------

TEST(MLAWeightMap, MLAAttentionProjectionNames) {
    WeightMap wm(ModelArch::DEEPSEEK);
    // kv_a_proj_with_mqa -> kv_a_proj (packed latent+rope, split happens at runtime)
    EXPECT_EQ(wm.map_name("model.layers.1.self_attn.kv_a_proj_with_mqa.weight"),
              "layer.1.kv_a_proj");
    // kv_a_layernorm -> kv_a_norm (RMSNorm weight on the 512-dim latent)
    EXPECT_EQ(wm.map_name("model.layers.1.self_attn.kv_a_layernorm.weight"),
              "layer.1.kv_a_norm");
    // kv_b_proj -> kv_b_proj (up-proj, output 16*(128+128)=4096)
    EXPECT_EQ(wm.map_name("model.layers.1.self_attn.kv_b_proj.weight"),
              "layer.1.kv_b_proj");
    // q_proj and o_proj remain unchanged (wq/wo)
    EXPECT_EQ(wm.map_name("model.layers.1.self_attn.q_proj.weight"),
              "layer.1.wq");
    EXPECT_EQ(wm.map_name("model.layers.1.self_attn.o_proj.weight"),
              "layer.1.wo");
}

// ---------------------------------------------------------------------------
// Task 2.2: TensorKindMatcher for MLA tensor names
// ---------------------------------------------------------------------------

TEST(MLATensorKindMatcher, MLAProjectionKinds) {
    using imp::match_tensor_kind;
    using imp::TensorKind;
    // kv_a_proj_with_mqa is matched via .kv_a_proj. substring (after name remapping)
    // In tensor_kind_matcher, we test the internal name form used by the matcher:
    EXPECT_EQ(match_tensor_kind("model.layers.1.self_attn.kv_a_proj_with_mqa.weight"),
              TensorKind::KV_A_PROJ);
    EXPECT_EQ(match_tensor_kind("model.layers.1.self_attn.kv_a_layernorm.weight"),
              TensorKind::KV_A_NORM);
    EXPECT_EQ(match_tensor_kind("model.layers.1.self_attn.kv_b_proj.weight"),
              TensorKind::KV_B_PROJ);
}

}  // namespace
