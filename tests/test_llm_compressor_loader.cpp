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
