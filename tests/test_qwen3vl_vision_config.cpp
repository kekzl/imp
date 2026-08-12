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
    JsonParser p(text.data(), text.size());
    return p.parse();
}

TEST(Qwen3VLVisionConfig, ParsesTheRealCheckpoint) {
    VisionConfig c;
    std::string err;
    ASSERT_TRUE(parse_qwen3vl_vision_config(parse(kRealVisionConfig), c, err)) << err;

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
    VisionConfig c;  // starts with the SigLIP defaults
    ASSERT_NE(c.image_size, 0);
    ASSERT_NE(c.num_patches, 0);
    std::string err;
    ASSERT_TRUE(parse_qwen3vl_vision_config(parse(kRealVisionConfig), c, err)) << err;
    EXPECT_EQ(c.image_size, 0);
    EXPECT_EQ(c.num_patches, 0);
    EXPECT_EQ(c.num_image_tokens, 0);
}

TEST(Qwen3VLVisionConfig, RejectionLeavesTheOutputUntouched) {
    VisionConfig before;
    VisionConfig c = before;
    std::string err;
    // depth missing entirely.
    const char* bad = R"({"hidden_size": 1024, "num_heads": 16})";
    EXPECT_FALSE(parse_qwen3vl_vision_config(parse(bad), c, err));
    EXPECT_FALSE(err.empty()) << "a rejection must say what was wrong";
    EXPECT_FALSE(c.is_qwen3vl) << "output must not be half-filled on rejection";
    EXPECT_EQ(c.hidden_size, before.hidden_size);
    EXPECT_EQ(c.num_layers, before.num_layers);
}

TEST(Qwen3VLVisionConfig, RejectsInconsistentGeometry) {
    std::string base = kRealVisionConfig;
    VisionConfig c;
    std::string err;

    // hidden_size not divisible by num_heads.
    auto bad_heads = base;
    bad_heads.replace(bad_heads.find("\"num_heads\": 16"), 15, "\"num_heads\": 15");
    EXPECT_FALSE(parse_qwen3vl_vision_config(parse(bad_heads), c, err));
    EXPECT_NE(err.find("divisible"), std::string::npos) << err;

    // num_position_embeddings not a perfect square.
    auto bad_pos = base;
    bad_pos.replace(bad_pos.find("\"num_position_embeddings\": 2304"), 31,
                    "\"num_position_embeddings\": 2305");
    EXPECT_FALSE(parse_qwen3vl_vision_config(parse(bad_pos), c, err));
    EXPECT_NE(err.find("perfect square"), std::string::npos) << err;

    // A non-positive dimension.
    auto bad_depth = base;
    bad_depth.replace(bad_depth.find("\"depth\": 24"), 11, "\"depth\": 0 ");
    EXPECT_FALSE(parse_qwen3vl_vision_config(parse(bad_depth), c, err));
    EXPECT_NE(err.find("positive"), std::string::npos) << err;
}

// A DeepStack tap pointing past the last block would index out of bounds at
// encode time, long after the config was read.
TEST(Qwen3VLVisionConfig, RejectsDeepstackIndexOutOfRange) {
    std::string bad = kRealVisionConfig;
    bad.replace(bad.find("[5, 11, 17]"), 11, "[5, 11, 24]");
    VisionConfig c;
    std::string err;
    EXPECT_FALSE(parse_qwen3vl_vision_config(parse(bad), c, err));
    EXPECT_NE(err.find("out of range"), std::string::npos) << err;
}

// DeepStack is optional; a model without it must still parse.
TEST(Qwen3VLVisionConfig, MissingDeepstackIsFine) {
    std::string no_ds = kRealVisionConfig;
    no_ds.replace(no_ds.find("\"deepstack_visual_indexes\": [5, 11, 17],"), 40, "");
    VisionConfig c;
    std::string err;
    ASSERT_TRUE(parse_qwen3vl_vision_config(parse(no_ds), c, err)) << err;
    EXPECT_TRUE(c.deepstack_indexes.empty());
    EXPECT_EQ(c.num_layers, 24);
}

// The allowlist is read twice per load — once by the SafeTensors loader to
// decide whether to keep the `model.visual.*` tensors, once here to decide
// whether to parse the geometry. Both go through this one predicate.
TEST(Qwen3VLVisionConfig, TowerAllowlist) {
    EXPECT_TRUE(vision_tower_supported("qwen3_vl"));
    // Qwen3.6 ships the same tower layout under its text model_type.
    EXPECT_TRUE(vision_tower_supported("qwen3_5_moe"));

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
