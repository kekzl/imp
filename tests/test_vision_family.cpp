// Vision family registry (vision_family.h): exact model_type names to a family; unknown names and
// families without an encoder in this build stay text-only.

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <fstream>
#include <string>

#include "model/hf_config_loader.h"
#include "vision/qwen3vl_vision_config.h"
#include "vision/vision_family.h"
#include "vision/vision_model.h"

namespace imp {
namespace {

TEST(VisionFamily, RegistryMapsExactNames) {
    EXPECT_EQ(vision_family_of("qwen3_vl"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("qwen3_5"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("qwen3_5_moe"), VisionFamily::Qwen3VL);
    EXPECT_EQ(vision_family_of("internvl_vision"), VisionFamily::InternVL);

    for (const char* unknown : {"", "qwen2_vl", "siglip_vision_model", "pixtral", "gemma3", "intern_vit_6b",
                                "internvl", "Qwen3_VL", "qwen3_vl "})
        EXPECT_EQ(vision_family_of(unknown), VisionFamily::None) << "'" << unknown << "'";
}

TEST(VisionFamily, LoadableOnlyWithAnEncoder) {
    EXPECT_TRUE(vision_family_loadable(VisionFamily::Qwen3VL));
    EXPECT_FALSE(vision_family_loadable(VisionFamily::None));
    EXPECT_TRUE(vision_family_loadable(VisionFamily::InternVL));
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

}  // namespace
}  // namespace imp
