// InternVL3.5 (HF layout, OpenGVLab/InternVL3_5-2B-HF): the LM ships as language_model.model.* +
// language_model.lm_head.weight; vision_tower.* and multi_modal_projector.* are not LM weights.
// Names below are the checkpoint's own (layer 0 of 28, vision layer 0 of 24).

#include "model/hf_config_loader.h"
#include "model/model.h"
#include "model/weight_map.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

Tensor fake_weight(void* backing) {
    Tensor t;
    t.data = backing;
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = 8;
    t.shape[1] = 8;
    return t;
}

struct InternVLFixture {
    std::vector<uint16_t> backing = std::vector<uint16_t>(1024, 0);
    Model model;
    std::unordered_map<std::string, Tensor> tensors;
    int vision_names = 0;

    InternVLFixture() {
        model.config_.arch = ModelArch::QWEN3;
        model.config_.n_layers = 1;
        model.config_.d_model = 8;
        model.config_.multimodal_wrapper = true;  // text_config present
        model.layers_.resize(1);
        for (const char* n : {"language_model.lm_head.weight", "language_model.model.embed_tokens.weight",
                              "language_model.model.norm.weight"})
            add(n);
        for (const char* s :
             {"input_layernorm.weight", "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
              "post_attention_layernorm.weight", "self_attn.k_norm.weight", "self_attn.k_proj.weight",
              "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.q_proj.weight",
              "self_attn.v_proj.weight"})
            add(std::string("language_model.model.layers.0.") + s);
        for (const char* n :
             {"multi_modal_projector.layer_norm.bias", "multi_modal_projector.layer_norm.weight",
              "multi_modal_projector.linear_1.bias", "multi_modal_projector.linear_1.weight",
              "multi_modal_projector.linear_2.bias", "multi_modal_projector.linear_2.weight",
              "vision_tower.embeddings.cls_token", "vision_tower.embeddings.patch_embeddings.projection.bias",
              "vision_tower.embeddings.patch_embeddings.projection.weight",
              "vision_tower.embeddings.position_embeddings",
              "vision_tower.encoder.layer.0.attention.q_proj.weight", "vision_tower.encoder.layer.0.lambda_1",
              "vision_tower.encoder.layer.0.mlp.fc2.bias"}) {
            add(n);
            ++vision_names;
        }
    }
    void add(const std::string& name) { tensors[name] = fake_weight(backing.data()); }
};

TEST(InternVLLmNames, EveryLmTensorIsAssignedAndVisionIsCountedApart) {
    InternVLFixture f;
    WeightMap wm(ModelArch::QWEN3);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));
    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.unrecognised, 0);
    EXPECT_EQ(s.vision, f.vision_names);
    EXPECT_EQ(s.audio, 0);
    EXPECT_EQ(s.total, f.vision_names);
    EXPECT_NE(f.model.tok_emb_.data, nullptr) << "language_model.model.embed_tokens.weight unmapped";
    EXPECT_NE(f.model.out_norm_.data, nullptr) << "language_model.model.norm.weight unmapped";
    EXPECT_NE(f.model.out_proj_.data, nullptr) << "language_model.lm_head.weight unmapped";
}

// The Qwen3-VL spelling still maps (model.language_model.*), so the shared strip did not move.
TEST(InternVLLmNames, Qwen3VLSpellingStillMaps) {
    InternVLFixture f;
    f.tensors.clear();
    for (const char* n : {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight",
                          "model.language_model.layers.0.self_attn.q_proj.weight",
                          "model.visual.blocks.0.attn.qkv.weight", "lm_head.weight"})
        f.add(n);
    WeightMap wm(ModelArch::QWEN3);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));
    EXPECT_EQ(wm.skip_stats().unrecognised, 0);
    EXPECT_EQ(wm.skip_stats().vision, 1);
    EXPECT_NE(f.model.tok_emb_.data, nullptr);
}

// config.json: InternVLForConditionalGeneration resolves its LM from text_config (Qwen3, 28 x 2048).
TEST(InternVLLmNames, ConfigResolvesTheLmFromTextConfig) {
    const auto dir = std::filesystem::temp_directory_path() / "imp_internvl_cfg";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "config.json") << R"({"architectures": ["InternVLForConditionalGeneration"],
        "model_type": "internvl", "image_token_id": 151671, "downsample_ratio": 0.5,
        "text_config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 2048,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 6144, "vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-06,
            "max_position_embeddings": 40960},
        "vision_config": {"model_type": "internvl_vision", "hidden_size": 1024, "num_hidden_layers": 24}})";
    ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(dir.string(), cfg));
    EXPECT_EQ(cfg.arch, ModelArch::QWEN3);
    EXPECT_EQ(cfg.d_model, 2048);
    EXPECT_EQ(cfg.n_layers, 28);
    EXPECT_EQ(cfg.n_kv_heads, 8);
    EXPECT_TRUE(cfg.multimodal_wrapper);
    std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace imp
