#include "model/llm_compressor_loader.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace imp::llm_compressor;

namespace {
std::string tmpdir() {
    const char* t = std::getenv("TMPDIR");
    return t ? t : "/tmp";
}
}  // namespace

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

// ---- keep_vision: the tower survives when the checkpoint declares one ----

TEST(LlmCompressorKeepVision, EmitsVisualPrefixVerbatim) {
    TranslationCounters c{};
    auto t = translate_name("model.visual.blocks.0.attn.qkv.weight", c, /*keep_vision=*/true);
    ASSERT_EQ(t.action, NameTranslation::EMIT);
    // Verbatim: the vision mapper strips `model.visual.` itself and then matches
    // the remainder literally, so any rewrite here unmaps the slot silently.
    EXPECT_EQ(t.out_name, "model.visual.blocks.0.attn.qkv.weight");
    EXPECT_EQ(c.vision_kept, 1);
    EXPECT_EQ(c.vision_skipped, 0);
}

// The rename steps run after the vision check on purpose. A vision tensor whose
// name happens to end in a translated suffix must still come out untouched —
// otherwise a quantized tower would load with every slot missing and only the
// "tower incomplete" warning to show for it.
TEST(LlmCompressorKeepVision, DoesNotRenameSuffixesOnVisionTensors) {
    TranslationCounters c{};
    auto t = translate_name("model.visual.blocks.0.attn.qkv.weight_packed", c, /*keep_vision=*/true);
    ASSERT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.visual.blocks.0.attn.qkv.weight_packed");
    EXPECT_EQ(c.suffix_renames, 0);
}

// keep_vision is about the vision tower only — the MTP head has its own gate
// (load_mtp_head) and must not ride in on this one.
TEST(LlmCompressorKeepVision, StillSkipsMtp) {
    TranslationCounters c{};
    EXPECT_EQ(translate_name("mtp.layers.0.self_attn.q_proj.weight", c, /*keep_vision=*/true).action,
              NameTranslation::SKIP);
    EXPECT_EQ(translate_name("model.mtp.fc.weight", c, /*keep_vision=*/true).action, NameTranslation::SKIP);
    EXPECT_EQ(c.vision_kept, 0);
}

TEST(LlmCompressorKeepVision, LeavesTextTensorsIdentical) {
    TranslationCounters off{}, on{};
    const std::string in = "model.language_model.layers.0.self_attn.q_proj.weight_packed";
    auto a = translate_name(in, off, /*keep_vision=*/false);
    auto b = translate_name(in, on, /*keep_vision=*/true);
    EXPECT_EQ(a.action, b.action);
    EXPECT_EQ(a.out_name, b.out_name);
    EXPECT_EQ(off.suffix_renames, on.suffix_renames);
    EXPECT_EQ(off.prefix_strips, on.prefix_strips);
}

// The shard-drop in load_sharded() calls name_is_skipped() while load_shard()
// calls translate_name(). If the two ever disagree, the shard carrying the
// tower is discarded before a single tensor is translated — the exact failure
// this pair of functions exists to prevent.
TEST(LlmCompressorKeepVision, PredicateAgreesWithTranslate) {
    const char* names[] = {
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.bias",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        "vision_tower.transformer.layers.0.attention.q.weight",
        "multi_modal_projector.linear_1.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.mtp.fc.weight",
        "model.layers.0.self_attn.q_proj.weight_packed",
        "model.embed_tokens.weight",
    };
    for (bool keep : {false, true}) {
        for (const char* n : names) {
            TranslationCounters c{};
            const bool translated_skips = translate_name(n, c, keep).action == NameTranslation::SKIP;
            EXPECT_EQ(name_is_skipped(n, keep), translated_skips) << "name=" << n << " keep_vision=" << keep;
        }
    }
}

// Default argument = today's behaviour, so every text-only checkpoint that
// loaded before this flag existed still translates byte-for-byte the same.
TEST(LlmCompressorKeepVision, DefaultsToDropping) {
    EXPECT_TRUE(name_is_skipped("model.visual.blocks.0.attn.qkv.weight"));
    EXPECT_TRUE(name_is_vision("model.visual.blocks.0.attn.qkv.weight"));
    EXPECT_FALSE(name_is_vision("mtp.layers.0.self_attn.q_proj.weight"));
    EXPECT_FALSE(name_is_vision("model.layers.0.self_attn.q_proj.weight_packed"));
}

#include <fstream>
#include <cstdlib>
#include <unistd.h>

namespace {

std::string write_temp_recipe(const std::string& content) {
    std::string dir = tmpdir() + "/recipe_" + std::to_string(::getpid()) + ".yaml.d";
    std::filesystem::create_directories(dir);
    std::ofstream out(dir + "/recipe.yaml");
    out << content;
    out.close();
    return dir;
}

void cleanup_temp_recipe(const std::string& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
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
    std::string dir = tmpdir() + "/fmt_both_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/hf_quant_config.json") << R"({"quantization":{"quant_algo":"NVFP4"}})";
    std::ofstream(dir + "/recipe.yaml")
        << "default_stage:\n  default_modifiers:\n    QuantizationModifier:\n      scheme: NVFP4\n";

    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::MODELOPT);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(LlmCompressorFormatDetect, DetectsLlmCompressorByRecipeYaml) {
    std::string dir = tmpdir() + "/fmt_lc_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
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

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(LlmCompressorFormatDetect, ReturnsFalseWhenNoConfigPresent) {
    std::string dir = tmpdir() + "/fmt_none_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    // Empty dir, no config files.
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_FALSE(ok);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// ---- MIXED_PRECISION (Modelopt): a per-tensor algorithm table ----
//
// Nemotron-3.5-Lightning exports one: the Mamba in/out projections are FP8 and
// the MoE experts NVFP4, so there is no single top-level `quant_algo` to match.

namespace {
std::string write_quant_config(const std::string& tag, const std::string& json) {
    std::string dir = tmpdir() + "/qc_" + tag + "_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/hf_quant_config.json") << json;
    return dir;
}
}  // namespace

TEST(ModeloptMixedPrecision, AcceptsTableThatNamesNvfp4) {
    std::string dir = write_quant_config("mixed", R"({"quantization":{
      "quant_algo": "MIXED_PRECISION",
      "kv_cache_quant_algo": "FP8",
      "quantized_layers": {
        "backbone.layers.0.mixer.in_proj":  {"quant_algo": "FP8"},
        "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
        "backbone.layers.1.mixer.experts.0.up_proj":   {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        "backbone.layers.1.mixer.experts.0.down_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
      }}})");
    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));
    EXPECT_TRUE(cfg.mixed_precision);
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::MODELOPT);
    EXPECT_EQ(cfg.n_nvfp4_tensors, 2);
    EXPECT_EQ(cfg.n_fp8_tensors, 2);
    EXPECT_EQ(cfg.n_other_tensors, 0);
    // group_size rides on the NVFP4 entries, not the top level.
    EXPECT_EQ(cfg.group_size, 16);
    EXPECT_EQ(cfg.kv_cache_quant_algo, "FP8");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// The return value sets `is_nvfp4_prequant`, which drives the MoE expert cache
// and the VRAM budget. An all-FP8 export claiming it would misplan both, so a
// table without a single NVFP4 tensor must NOT be accepted as one.
TEST(ModeloptMixedPrecision, RejectsTableWithoutNvfp4) {
    std::string dir = write_quant_config("mixed_fp8", R"({"quantization":{
      "quant_algo": "MIXED_PRECISION",
      "quantized_layers": {
        "backbone.layers.0.mixer.in_proj":  {"quant_algo": "FP8"},
        "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"}
      }}})");
    imp::HFConfigLoader::NvFP4Config cfg;
    EXPECT_FALSE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(ModeloptMixedPrecision, RejectsMissingTable) {
    std::string dir = write_quant_config("mixed_notable",
                                         R"({"quantization":{"quant_algo":"MIXED_PRECISION"}})");
    imp::HFConfigLoader::NvFP4Config cfg;
    EXPECT_FALSE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// An algorithm this build does not know must be counted and surfaced, not
// folded into the NVFP4 count — silence there is how a half-understood
// checkpoint passes as "some NVFP4 model".
TEST(ModeloptMixedPrecision, CountsUnrecognisedAlgorithmsSeparately) {
    std::string dir = write_quant_config("mixed_unknown", R"({"quantization":{
      "quant_algo": "MIXED_PRECISION",
      "quantized_layers": {
        "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        "backbone.layers.2.mixer.in_proj": {"quant_algo": "W8A8_INT8"},
        "backbone.layers.3.mixer.in_proj": {"quant_algo": "AWQ_W4A16"}
      }}})");
    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));
    EXPECT_EQ(cfg.n_nvfp4_tensors, 1);
    EXPECT_EQ(cfg.n_other_tensors, 2);
    EXPECT_EQ(cfg.n_fp8_tensors, 0);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// Plain NVFP4 must be untouched by the MIXED_PRECISION branch.
TEST(ModeloptMixedPrecision, PlainNvfp4Unchanged) {
    std::string dir = write_quant_config("plain", R"({"quantization":{
      "quant_algo": "NVFP4", "group_size": 16, "exclude_modules": ["lm_head"]}})");
    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));
    EXPECT_FALSE(cfg.mixed_precision);
    EXPECT_EQ(cfg.n_nvfp4_tensors, 0);
    EXPECT_EQ(cfg.exclude_modules.size(), 1u);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// ---- compressed-tensors declared in config.json --------------------------
//
// recipe.yaml is llm-compressor's record of the run, not the checkpoint's
// declaration: plenty of published exports carry only the config.json block,
// and imp used to read those as Modelopt — whose tensor scale is the
// RECIPROCAL of this format's.

TEST(LlmCompressorFormatDetect, DetectsCompressedTensorsFromConfigJson) {
    std::string dir = tmpdir() + "/fmt_ctcfg_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/config.json") << R"({
  "model_type": "qwen3",
  "quantization_config": {
    "config_groups": {"group_0": {"input_activations": null, "targets": ["Linear"],
      "weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group",
                  "symmetric": true}}},
    "format": "nvfp4-pack-quantized",
    "ignore": ["lm_head", "model.layers.0.mlp.gate"],
    "quant_method": "compressed-tensors"
  }
})";
    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);
    EXPECT_EQ(cfg.group_size, 16);
    ASSERT_EQ(cfg.exclude_modules.size(), 2u);
    EXPECT_EQ(cfg.exclude_modules[0], "lm_head");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(LlmCompressorFormatDetect, RejectsCompressedTensorsSchemesThatAreNotNvfp4) {
    // int4 pack-quantized is compressed-tensors too. Claiming it as NVFP4 would
    // read integer weights through the FP4 decoder.
    std::string dir = tmpdir() + "/fmt_ctint_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/config.json") << R"({
  "quantization_config": {
    "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int", "group_size": 128,
                                              "strategy": "group", "symmetric": true}}},
    "format": "pack-quantized",
    "quant_method": "compressed-tensors"
  }
})";
    imp::HFConfigLoader::NvFP4Config cfg;
    EXPECT_FALSE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(LlmCompressorFormatDetect, IgnoresAConfigJsonWithoutAQuantizationBlock) {
    std::string dir = tmpdir() + "/fmt_plain_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/config.json") << R"({"model_type": "qwen3", "quantization_config": null})";
    imp::HFConfigLoader::NvFP4Config cfg;
    EXPECT_FALSE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(LlmCompressorFormatDetect, RecipeYamlStillWinsOverConfigJson) {
    // Both present is the normal llm-compressor upload. The recipe path is the
    // one with history behind it, so it stays first.
    std::string dir = tmpdir() + "/fmt_ctboth_" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/recipe.yaml")
        << "default_stage:\n  default_modifiers:\n    QuantizationModifier:\n"
           "      ignore: [lm_head]\n      scheme: NVFP4\n";
    std::ofstream(dir + "/config.json") << R"({
  "quantization_config": {
    "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16,
                                              "strategy": "tensor_group", "symmetric": true}}},
    "format": "nvfp4-pack-quantized", "ignore": ["a", "b", "c"],
    "quant_method": "compressed-tensors"
  }
})";
    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg));
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);
    EXPECT_EQ(cfg.exclude_modules.size(), 1u) << "the recipe's ignore list, not config.json's";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}
