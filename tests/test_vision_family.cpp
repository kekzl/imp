// Vision family registry (vision_family.h): exact model_type names to a family; unknown names and
// families without an encoder in this build stay text-only.

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <fstream>
#include <string>

#include "model/hf_config_loader.h"
#include "model/image_placeholders.h"
#include "vision/qwen3vl_vision_config.h"
#include "vision/vision_family.h"
#include "vision/vision_model.h"

namespace imp {
namespace {

TEST(VisionFamily, RegistryMapsExactNames) {
    EXPECT_EQ(vision_family_of("qwen3_vl"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("qwen3_5"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("qwen3_5_vision"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("qwen3_5_moe"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("qwen4_exp_vision"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("internvl_vision"), VisionFamily::InternVL);
    EXPECT_EQ(vision_family_of("inkling2_vision"), VisionFamily::Inkling2);

    for (const char* unknown : {"", "qwen2_vl", "siglip_vision_model", "pixtral", "gemma3", "intern_vit_6b",
                                "internvl", "Qwen3_VL", "qwen3_vl "})
        EXPECT_EQ(vision_family_of(unknown), VisionFamily::None) << "'" << unknown << "'";
}

TEST(VisionFamily, LoadableOnlyWithAnEncoder) {
    EXPECT_TRUE(vision_family_loadable(VisionFamily::Qwen3VL));
    EXPECT_FALSE(vision_family_loadable(VisionFamily::None));
    EXPECT_TRUE(vision_family_loadable(VisionFamily::InternVL));
    // Inkling2 has its own encoder outside the Qwen3-VL pipeline this predicate gates.
    EXPECT_FALSE(vision_family_loadable(VisionFamily::Inkling2));
    EXPECT_STREQ(vision_family_name(VisionFamily::InternVL), "InternVL");
    EXPECT_STREQ(vision_family_name(VisionFamily::None), "none");
}

// The SafeTensors keep-the-vision-tensors gate reads config.json through probe_vision_tower();
// it must answer from the same registry.
TEST(VisionFamily, ProbeFollowsTheRegistry) {
    const auto dir = std::filesystem::temp_directory_path() / "imp_vision_family_probe";
    std::filesystem::create_directories(dir);
    auto probe = [&](const std::string& vision_type) {
        std::ofstream(dir / "config.json")
            << R"({"vision_config": {"model_type": ")" << vision_type << R"("}})";
        return HFConfigLoader::probe_vision_tower(dir.string());
    };
    EXPECT_TRUE(probe("qwen3_vl"));
    EXPECT_TRUE(probe("qwen3_5_moe"));
    EXPECT_TRUE(probe("internvl_vision"));
    EXPECT_FALSE(probe("pixtral"));
    std::ofstream(dir / "config.json") << R"({"model_type": "qwen3"})";
    EXPECT_FALSE(HFConfigLoader::probe_vision_tower(dir.string())) << "no vision_config: text-only";
    std::filesystem::remove_all(dir);
}

// load_config: an unknown tower and a recognised tower without an encoder both load text-only
// (no VisionModel) and say so on stderr; a Qwen3-VL tower parses.
TEST(VisionFamily, LoadConfigTextOnlyPathsWarn) {
    const auto dir = std::filesystem::temp_directory_path() / "imp_vision_family_load";
    std::filesystem::create_directories(dir);
    const std::string body = R"("hidden_size": 2560, "num_attention_heads": 32, "num_key_value_heads": 8,
        "num_hidden_layers": 36, "vocab_size": 151936, "head_dim": 128)";
    auto load = [&](const std::string& vision_cfg, std::string& err) {
        std::ofstream(dir / "config.json") << "{" << body << R"(, "vision_config": )" << vision_cfg << "}";
        ModelConfig cfg;
        std::unique_ptr<VisionModel> tower;
        testing::internal::CaptureStderr();
        EXPECT_TRUE(HFConfigLoader::load_config(dir.string(), cfg, &tower));
        err = testing::internal::GetCapturedStderr();
        return tower != nullptr;
    };
    std::string err;
    EXPECT_FALSE(load(R"({"model_type": "pixtral"})", err));
    EXPECT_NE(err.find("vision tower will be skipped"), std::string::npos) << err;
    EXPECT_NE(err.find("pixtral"), std::string::npos) << err;

    EXPECT_FALSE(load(R"({"model_type": "internvl_vision", "hidden_size": 1024})", err));
    EXPECT_NE(err.find("InternVL vision_config rejected"), std::string::npos) << err;

    EXPECT_TRUE(load(R"({"model_type": "qwen3_vl", "depth": 24, "hidden_size": 1024, "num_heads": 16,
        "intermediate_size": 4096, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
        "out_hidden_size": 2560, "num_position_embeddings": 2304})",
                     err))
        << err;
    std::filesystem::remove_all(dir);
}

// Inkling2 image parts: toy ids, pad 9, text 1, image 2, end 3, roles 10 (user) and 11 (tool).
std::vector<int32_t> split_parts(std::vector<int32_t> t, bool with_ids = true) {
    InklingPartIds ids{9, 1, with_ids ? 2 : -1, 3, {10, 11}};
    inkling_split_image_parts(t, ids);
    return t;
}
using Ids = std::vector<int32_t>;

TEST(InklingImageParts, APictureInsideATextPartBecomesItsOwnPart) {
    // Picture first, then text: no empty text part ahead of it.
    EXPECT_EQ(split_parts({10, 1, 9, 9, 9, 50, 51, 3}), (Ids{10, 2, 9, 9, 9, 3, 10, 1, 50, 51, 3}));
    // Text first, picture last: no empty text part behind it.
    EXPECT_EQ(split_parts({10, 1, 50, 9, 9, 3}), (Ids{10, 1, 50, 3, 10, 2, 9, 9, 3}));
    // Text, picture, text, picture, text: every part repeats the role header.
    EXPECT_EQ(split_parts({10, 1, 50, 9, 51, 9, 9, 52, 3}),
              (Ids{10, 1, 50, 3, 10, 2, 9, 3, 10, 1, 51, 3, 10, 2, 9, 9, 3, 10, 1, 52, 3}));
    // Two pictures back to back are two parts only when text separates them: one run, one part.
    EXPECT_EQ(split_parts({10, 1, 9, 9, 3}), (Ids{10, 2, 9, 9, 3}));
}

TEST(InklingImageParts, TheHeaderKeepsANameBehindTheRole) {
    EXPECT_EQ(split_parts({11, 60, 61, 1, 70, 9, 3}), (Ids{11, 60, 61, 1, 70, 3, 11, 60, 61, 2, 9, 3}));
}

TEST(InklingImageParts, EverythingElseStaysAsItIs) {
    // A part the template already rendered as an image, a pad outside any text part, other messages.
    const Ids native = {10, 1, 50, 3, 10, 2, 9, 9, 3, 11, 60, 4, 9, 3, 10, 1, 51, 3};
    EXPECT_EQ(split_parts(native), native);
    // A tokenizer without the part markers: untouched.
    const Ids spliced = {10, 1, 9, 9, 50, 3};
    EXPECT_EQ(split_parts(spliced, false), spliced);
}

TEST(InklingImageParts, ExpandGivesEveryPictureItsCountAndItsPart) {
    const auto find = [](std::string_view s) -> int32_t {
        if (s == "<|unused_200054|>")
            return 9;
        if (s == "<|content_text|>")
            return 1;
        if (s == "<|content_image|>")
            return 2;
        if (s == "<|end_message|>")
            return 3;
        if (s == "<|message_user|>")
            return 10;
        if (s == "<|message_tool|>")
            return 11;
        return -1;
    };
    Ids t = {10, 1, 50, 9, 51, 9, 3};
    ASSERT_TRUE(expand_vision_prompt(VisionFamily::Inkling2, t, find, {3, 2}, {}));
    EXPECT_EQ(t, (Ids{10, 1, 50, 3, 10, 2, 9, 9, 9, 3, 10, 1, 51, 3, 10, 2, 9, 9, 3}));
    Ids bad = {10, 1, 9, 3};
    EXPECT_FALSE(expand_vision_prompt(VisionFamily::Inkling2, bad, find, {3, 2}, {}));
    EXPECT_EQ(vision_prompt_blocks(VisionFamily::Inkling2, "ii"), "<|unused_200054|><|unused_200054|>");
}

}  // namespace
}  // namespace imp
