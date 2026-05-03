#include "model/llm_compressor_loader.h"
#include <gtest/gtest.h>

using namespace imp::llm_compressor;

TEST(LlmCompressorTranslate, RenamesWeightPacked) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.self_attn.q_proj.weight_packed", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(c.suffix_renames, 1);
}

TEST(LlmCompressorTranslate, RenamesWeightGlobalScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.mlp.gate_proj.weight_global_scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.mlp.gate_proj.weight_scale_2");
    EXPECT_EQ(c.suffix_renames, 1);
}

TEST(LlmCompressorTranslate, RenamesInputGlobalScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.5.self_attn.k_proj.input_global_scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.5.self_attn.k_proj.input_scale");
    EXPECT_EQ(c.suffix_renames, 1);
}

TEST(LlmCompressorTranslate, WeightScaleUnchanged) {
    // .weight_scale exists in BOTH formats with identical layout, no rename.
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.mlp.up_proj.weight_scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.mlp.up_proj.weight_scale");
    EXPECT_EQ(c.suffix_renames, 0);
    EXPECT_EQ(c.passed_through, 1);
}

TEST(LlmCompressorTranslate, UnknownPassesThrough) {
    TranslationCounters c{};
    auto t = translate_name("model.embed_tokens.weight", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.embed_tokens.weight");
    EXPECT_EQ(c.passed_through, 1);
}

TEST(LlmCompressorTranslate, StripsMultimodalPrefix) {
    TranslationCounters c{};
    auto t = translate_name("model.language_model.layers.0.self_attn.q_proj.weight_packed", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(c.suffix_renames, 1);
    EXPECT_EQ(c.prefix_strips, 1);
}

TEST(LlmCompressorTranslate, SkipsVisionTower) {
    TranslationCounters c{};
    auto t = translate_name("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.vision_skipped, 1);
}

TEST(LlmCompressorTranslate, SkipsVisualPrefix) {
    // Qwen3-VL naming uses model.visual.* instead of model.vision_tower.*
    TranslationCounters c{};
    auto t = translate_name("model.visual.blocks.0.attn.q_proj.weight", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.vision_skipped, 1);
}

// Phase 2: layer_scalar is now emitted (not skipped) — weight_map.cpp routes
// it to layer.layer_out_scale, which executor_forward applies after each MoE
// layer. The counter records emission for the load summary log line.
TEST(LlmCompressorTranslate, EmitsLayerScalar) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.layer_scalar", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.layer_scalar");
    EXPECT_EQ(c.gemma4_extras, 1);
}

// Phase 2: per_expert_scale is now emitted — weight_map routes it to
// layer.expert_down_scale, applied by moe_apply_per_expert_scale_kernel
// before the routing weighted sum.
TEST(LlmCompressorTranslate, EmitsPerExpertScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.5.router.per_expert_scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.5.router.per_expert_scale");
    EXPECT_EQ(c.gemma4_extras, 1);
}

// Phase 2: router.scale (per-input-channel router pre-scale) is now emitted —
// weight_map routes it to layer.ffn_gate_inp_scale, applied to the router
// input before the gating projection.
TEST(LlmCompressorTranslate, EmitsRouterScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.router.scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.router.scale");
    EXPECT_EQ(c.gemma4_extras, 1);
}

// Defensive: .scale on a known projection (q_proj/k_proj/...) is NOT a Gemma-4
// extra and must not increment the gemma4_extras counter.
TEST(LlmCompressorTranslate, ProjScaleIsNotGemma4Extra) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.self_attn.q_proj.scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(c.gemma4_extras, 0);
    EXPECT_EQ(c.passed_through, 1);
}

// Mistral3 layout: `language_model.model.layers.*` (no leading `model.`).
// Strip the `language_model.` wrapper entirely so the rest matches imp's
// canonical `model.layers.*` naming.
TEST(LlmCompressorTranslate, StripsMistral3LanguageModelPrefix) {
    TranslationCounters c{};
    auto t = translate_name("language_model.model.layers.0.self_attn.q_proj.weight_packed", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(c.suffix_renames, 1);
    EXPECT_EQ(c.prefix_strips, 1);
}

// Mistral3 stores lm_head directly under `language_model.` (not under
// `language_model.model.`). After strip → `lm_head.weight` (top-level), which
// is the canonical name imp expects for the output projection.
TEST(LlmCompressorTranslate, StripsMistral3LmHead) {
    TranslationCounters c{};
    auto t = translate_name("language_model.lm_head.weight", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "lm_head.weight");
    EXPECT_EQ(c.prefix_strips, 1);
    EXPECT_EQ(c.passed_through, 1);
}

// Gemma-4-style nesting MUST take precedence over the Mistral3-style strip
// (the Gemma-4 prefix `model.language_model.` is a strict superset of
// `language_model.`, so order matters).
TEST(LlmCompressorTranslate, Gemma4PrefixStillWinsOverMistral3) {
    TranslationCounters c{};
    auto t = translate_name("model.language_model.layers.0.self_attn.q_proj.weight_packed", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(c.prefix_strips, 1);
}

// Mistral3 vision tower at top level (no `model.` wrapper).
TEST(LlmCompressorTranslate, SkipsRawVisionTower) {
    TranslationCounters c{};
    auto t = translate_name("vision_tower.transformer.layers.0.attention.q.weight", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.vision_skipped, 1);
}

// Multimodal projector tensors connect vision → language; we skip them in
// Phase 2 (full multimodal support is a separate Phase 3 effort).
TEST(LlmCompressorTranslate, SkipsMultiModalProjector) {
    TranslationCounters c{};
    auto t = translate_name("multi_modal_projector.linear_1.weight", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.vision_skipped, 1);
}

#include <fstream>
#include <cstdlib>
#include <unistd.h>

namespace {

std::string write_temp_recipe(const std::string& content) {
    std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/recipe_" +
                       std::to_string(::getpid()) + ".yaml";
    // Create a temp dir and place recipe.yaml inside it.
    std::string dir = path + ".d";
    std::string mkdir_cmd = "mkdir -p '" + dir + "'";
    std::system(mkdir_cmd.c_str());
    std::ofstream out(dir + "/recipe.yaml");
    out << content;
    out.close();
    return dir;
}

void cleanup_temp_recipe(const std::string& dir) {
    std::string rm = "rm -rf '" + dir + "'";
    std::system(rm.c_str());
}

}  // namespace

TEST(LlmCompressorRecipe, ParsesGemma4Recipe) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, 're:.*embed.*', 're:.*router', 're:.*vision_tower.*']
      scheme: NVFP4
      bypass_divisibility_checks: false
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.group_size, 16);
    ASSERT_EQ(cfg.exclude_modules.size(), 4u);
    EXPECT_EQ(cfg.exclude_modules[0], "lm_head");
    EXPECT_EQ(cfg.exclude_modules[1], "re:.*embed.*");
    cleanup_temp_recipe(dir);
}

TEST(LlmCompressorRecipe, ParsesQwen36Recipe) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: ['re:.*lm_head', 're:visual.*', 're:model.visual.*', 're:.*mlp.gate$', 're:.*embed_tokens$', 're:.*shared_expert_gate$', 're:.*linear_attn.*']
      scheme: NVFP4
      bypass_divisibility_checks: false
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.exclude_modules.size(), 7u);
    EXPECT_EQ(cfg.exclude_modules[3], "re:.*mlp.gate$");
    cleanup_temp_recipe(dir);
}

TEST(LlmCompressorRecipe, RejectsNonNVFP4Scheme) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: W8A8
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_FALSE(ok);
    cleanup_temp_recipe(dir);
}

TEST(LlmCompressorRecipe, ReturnsFalseOnMissingFile) {
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml("/tmp/nonexistent_dir_xyz", cfg);
    EXPECT_FALSE(ok);
}

// Mistral3-style recipe: no `scheme:` line; NVFP4 is implicit in the
// `config_groups.group_0.weights.{num_bits: 4, type: float}` schema. Also
// exercises multi-line bracket-array `ignore:` parsing.
TEST(LlmCompressorRecipe, ParsesConfigGroupsSchema) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.9
      mappings: []
    QuantizationModifier:
      config_groups:
        group_0:
          targets: [Linear]
          weights:
            num_bits: 4
            type: float
            symmetric: true
            group_size: 16
            strategy: tensor_group
          input_activations:
            num_bits: 4
            type: float
            group_size: 16
          output_activations: null
          format: null
      targets: [Linear]
      ignore: ['re:.*lm_head.*', 're:.*multi_modal_projector.*', 're:.*vision_tower.*', 're:.*model.norm.*',
        're:.*embed_tokens.*']
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.group_size, 16);
    ASSERT_EQ(cfg.exclude_modules.size(), 5u);
    EXPECT_EQ(cfg.exclude_modules[0], "re:.*lm_head.*");
    EXPECT_EQ(cfg.exclude_modules[4], "re:.*embed_tokens.*");
    cleanup_temp_recipe(dir);
}

// Make sure config_groups detection doesn't false-positive on non-NVFP4
// numeric signatures (e.g. W8A8: num_bits=8, type=int).
TEST(LlmCompressorRecipe, RejectsConfigGroupsW8A8) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      config_groups:
        group_0:
          targets: [Linear]
          weights:
            num_bits: 8
            type: int
            group_size: 128
      targets: [Linear]
      ignore: [lm_head]
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_FALSE(ok);
    cleanup_temp_recipe(dir);
}

TEST(LlmCompressorFormatDetect, PrefersModeloptWhenBothPresent) {
    std::string dir = std::string(std::getenv("TMPDIR") ?: "/tmp") + "/fmt_both_" +
                      std::to_string(::getpid());
    std::system(("mkdir -p '" + dir + "'").c_str());
    std::ofstream(dir + "/hf_quant_config.json") << R"({"quantization":{"quant_algo":"NVFP4"}})";
    std::ofstream(dir + "/recipe.yaml")
        << "default_stage:\n  default_modifiers:\n    QuantizationModifier:\n      scheme: NVFP4\n";

    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::MODELOPT);

    std::system(("rm -rf '" + dir + "'").c_str());
}

TEST(LlmCompressorFormatDetect, DetectsLlmCompressorByRecipeYaml) {
    std::string dir = std::string(std::getenv("TMPDIR") ?: "/tmp") + "/fmt_lc_" + std::to_string(::getpid());
    std::system(("mkdir -p '" + dir + "'").c_str());
    std::ofstream(dir + "/recipe.yaml") << R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: NVFP4
)";
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);

    std::system(("rm -rf '" + dir + "'").c_str());
}

TEST(LlmCompressorFormatDetect, ReturnsFalseWhenNoConfigPresent) {
    std::string dir = std::string(std::getenv("TMPDIR") ?: "/tmp") + "/fmt_none_" +
                      std::to_string(::getpid());
    std::system(("mkdir -p '" + dir + "'").c_str());
    // Empty dir, no config files.
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_FALSE(ok);
    std::system(("rm -rf '" + dir + "'").c_str());
}
