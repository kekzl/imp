// Qwen3-VL vision_config parsing.
//
// A wrong geometry here silently mis-shapes every encoder buffer, so the parser
// refuses rather than half-fills, and these tests pin that: on any rejection the
// output must be left untouched.
//
// Oracle: the `vision_config` of the staged Qwen3-VL-4B-Instruct config.json.

#include "vision/qwen3vl_vision_config.h"

#include <gtest/gtest.h>

#include <string>

namespace imp {
namespace {

// The real block, verbatim.
const char* kRealVisionConfig = R"({
  "deepstack_visual_indexes": [5, 11, 17],
  "depth": 24,
  "hidden_act": "gelu_pytorch_tanh",
  "hidden_size": 1024,
  "in_channels": 3,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "model_type": "qwen3_vl",
  "num_heads": 16,
  "num_position_embeddings": 2304,
  "out_hidden_size": 2560,
  "patch_size": 16,
  "spatial_merge_size": 2,
  "temporal_patch_size": 2
})";

JValue parse(const std::string& text) {
    JsonParser p(text);
    return p.parse();
}

TEST(Qwen3VLVisionConfig, ParsesTheRealCheckpoint) {
    auto parsed = parse_qwen3vl_vision_config(parse(kRealVisionConfig));
    ASSERT_TRUE(parsed) << parsed.error();
    const VisionConfig& c = *parsed;

    EXPECT_TRUE(c.is_qwen3vl);
    EXPECT_EQ(c.num_layers, 24);
    EXPECT_EQ(c.hidden_size, 1024);
    EXPECT_EQ(c.num_heads, 16);
    EXPECT_EQ(c.head_dim, 64);  // derived, not read
    EXPECT_EQ(c.intermediate_size, 4096);
    EXPECT_EQ(c.patch_size, 16);
    EXPECT_EQ(c.merge_size, 2);
    EXPECT_EQ(c.temporal_patch_size, 2);
    EXPECT_EQ(c.out_hidden_size, 2560);
    EXPECT_EQ(c.pos_embed_grid, 48) << "2304 = 48^2";
    ASSERT_EQ(c.deepstack_indexes.size(), 3u);
    EXPECT_EQ(c.deepstack_indexes[0], 5);
    EXPECT_EQ(c.deepstack_indexes[2], 17);
}

// Dynamic resolution: there is no fixed image size or patch count. Inheriting
// the SigLIP defaults (896 / 4096 / 256) would be a lie the encoder could read
// and size buffers from.
TEST(Qwen3VLVisionConfig, ClearsTheFixedResolutionFields) {
    const VisionConfig defaults;  // the SigLIP defaults this must not inherit
    ASSERT_NE(defaults.image_size, 0);
    ASSERT_NE(defaults.num_patches, 0);
    auto parsed = parse_qwen3vl_vision_config(parse(kRealVisionConfig));
    ASSERT_TRUE(parsed) << parsed.error();
    EXPECT_EQ(parsed->image_size, 0);
    EXPECT_EQ(parsed->num_patches, 0);
    EXPECT_EQ(parsed->num_image_tokens, 0);
}

// A rejection can no longer hand back a half-filled geometry, because it hands
// back no geometry at all: the signature carries the error instead of writing
// through an out-parameter the caller might read before checking the bool.
TEST(Qwen3VLVisionConfig, RejectionYieldsNoConfigAtAll) {
    // depth missing entirely.
    const char* bad = R"({"hidden_size": 1024, "num_heads": 16})";
    auto parsed = parse_qwen3vl_vision_config(parse(bad));
    ASSERT_FALSE(parsed.has_value());
    EXPECT_FALSE(parsed.error().empty()) << "a rejection must say what was wrong";
    EXPECT_NE(parsed.error().find("depth"), std::string::npos) << parsed.error();
}

TEST(Qwen3VLVisionConfig, RejectsInconsistentGeometry) {
    const std::string base = kRealVisionConfig;

    auto rejected_with = [](const std::string& cfg, const char* needle) {
        auto parsed = parse_qwen3vl_vision_config(parse(cfg));
        ASSERT_FALSE(parsed.has_value());
        EXPECT_NE(parsed.error().find(needle), std::string::npos) << parsed.error();
    };

    // hidden_size not divisible by num_heads.
    auto bad_heads = base;
    bad_heads.replace(bad_heads.find("\"num_heads\": 16"), 15, "\"num_heads\": 15");
    rejected_with(bad_heads, "divisible");

    // num_position_embeddings not a perfect square.
    auto bad_pos = base;
    bad_pos.replace(bad_pos.find("\"num_position_embeddings\": 2304"), 31,
                    "\"num_position_embeddings\": 2305");
    rejected_with(bad_pos, "perfect square");

    // A non-positive dimension.
    auto bad_depth = base;
    bad_depth.replace(bad_depth.find("\"depth\": 24"), 11, "\"depth\": 0 ");
    rejected_with(bad_depth, "positive");
}

// A DeepStack tap pointing past the last block would index out of bounds at
// encode time, long after the config was read.
TEST(Qwen3VLVisionConfig, RejectsDeepstackIndexOutOfRange) {
    std::string bad = kRealVisionConfig;
    bad.replace(bad.find("[5, 11, 17]"), 11, "[5, 11, 24]");
    auto parsed = parse_qwen3vl_vision_config(parse(bad));
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("out of range"), std::string::npos) << parsed.error();
}

// DeepStack is optional; a model without it must still parse.
TEST(Qwen3VLVisionConfig, MissingDeepstackIsFine) {
    std::string no_ds = kRealVisionConfig;
    no_ds.replace(no_ds.find("\"deepstack_visual_indexes\": [5, 11, 17],"), 40, "");
    auto parsed = parse_qwen3vl_vision_config(parse(no_ds));
    ASSERT_TRUE(parsed) << parsed.error();
    EXPECT_TRUE(parsed->deepstack_indexes.empty());
    EXPECT_EQ(parsed->num_layers, 24);
}

// The allowlist is read twice per load — once by the SafeTensors loader to
// decide whether to keep the `model.visual.*` tensors, once here to decide
// whether to parse the geometry. Both go through this one predicate.
TEST(Qwen3VLVisionConfig, TowerAllowlist) {
    EXPECT_TRUE(vision_tower_supported("qwen3_vl"));
    // Qwen3.6 ships the same tower layout under its text model_type.
    EXPECT_TRUE(vision_tower_supported("qwen3_5_moe"));
    // Qwen3.8 is the dense sibling and carries that same tower.
    EXPECT_TRUE(vision_tower_supported("qwen3_5"));

    // Everything else keeps hitting the loud text-only path. Recognising a
    // tower on a resemblance is what this list exists to prevent.
    EXPECT_FALSE(vision_tower_supported(""));
    EXPECT_FALSE(vision_tower_supported("qwen2_vl"));
    EXPECT_FALSE(vision_tower_supported("siglip_vision_model"));
    EXPECT_FALSE(vision_tower_supported("pixtral"));
    EXPECT_FALSE(vision_tower_supported("gemma3"));
}

}  // namespace
}  // namespace imp
