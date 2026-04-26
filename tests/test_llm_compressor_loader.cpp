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
    auto t = translate_name(
        "model.language_model.layers.0.self_attn.q_proj.weight_packed", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(c.suffix_renames, 1);
    EXPECT_EQ(c.prefix_strips, 1);
}

TEST(LlmCompressorTranslate, SkipsVisionTower) {
    TranslationCounters c{};
    auto t = translate_name(
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", c);
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

TEST(LlmCompressorTranslate, SkipsLayerScalar) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.layer_scalar", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.gemma4_extra_skipped, 1);
}

TEST(LlmCompressorTranslate, SkipsPerExpertScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.5.experts.per_expert_scale", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.gemma4_extra_skipped, 1);
}

TEST(LlmCompressorTranslate, DoesNotSkipProjScale) {
    // .scale suffix on a recognized projection name is NOT a Gemma-4 extra.
    // (Defensive against false-positive blanket .scale skip.)
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.self_attn.q_proj.scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);  // pass through
    EXPECT_EQ(c.gemma4_extra_skipped, 0);
}

#include <fstream>
#include <cstdlib>
#include <unistd.h>

namespace {

std::string write_temp_recipe(const std::string& content) {
    std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
                       + "/recipe_" + std::to_string(::getpid()) + ".yaml";
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

} // namespace

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
