// Gemma-4 audio tower config parse and tensor load on a tiny synthetic checkpoint (CPU only):
// every slot filled, clip bounds read, derived tables right, a missing tensor refused.

#include "audio/gemma4_audio_encoder.h"
#include "audio/gemma4_audio_model.h"
#include "core/fp_bits.h"
#include "model/json_util.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

constexpr const char* kConfig = R"({
  "audio_token_id": 258881, "boa_token_id": 256000, "eoa_token_id": 258883,
  "text_config": {"hidden_size": 48},
  "audio_config": {"model_type": "gemma4_audio", "hidden_size": 64, "num_attention_heads": 2,
    "num_hidden_layers": 2, "attention_chunk_size": 12, "attention_context_left": 13,
    "attention_context_right": 0, "attention_logit_cap": 50.0, "conv_kernel_size": 5,
    "output_proj_dims": 40, "residual_weight": 0.5, "rms_norm_eps": 1e-6,
    "subsampling_conv_channels": [8, 4], "use_clipped_linears": true, "hidden_act": "silu"}
})";

JValue parse(const char* text) {
    JsonParser p(text);
    JValue v = p.parse();
    EXPECT_TRUE(p.ok());
    return v;
}

struct Checkpoint {
    std::vector<std::vector<uint16_t>> store;  // BF16 backing
    std::unordered_map<std::string, Tensor> t;

    void add(const std::string& name, std::initializer_list<int64_t> shape, float value = 0.25f) {
        int64_t n = 1;
        int64_t dims[4] = {};
        int nd = 0;
        for (int64_t s : shape) {
            n *= s;
            dims[nd++] = s;
        }
        store.emplace_back(static_cast<size_t>(std::max<int64_t>(n, 1)), float_to_bf16(value));
        t.emplace(name, Tensor(store.back().data(), QType::BF16, nd, dims, false));
    }
    void linear(const std::string& base, int64_t out, int64_t in) {
        add(base + ".linear.weight", {out, in});
        add(base + ".input_min", {}, -4.0f);
        add(base + ".input_max", {}, 4.0f);
        add(base + ".output_min", {}, -8.0f);
        add(base + ".output_max", {}, 8.0f);
    }
};

Checkpoint tiny_checkpoint(int layers) {
    Checkpoint c;
    c.store.reserve(512);
    const std::string t = "model.audio_tower.";
    c.add(t + "subsample_conv_projection.layer0.conv.weight", {8, 1, 3, 3});
    c.add(t + "subsample_conv_projection.layer0.norm.weight", {8});
    c.add(t + "subsample_conv_projection.layer1.conv.weight", {4, 8, 3, 3});
    c.add(t + "subsample_conv_projection.layer1.norm.weight", {4});
    c.add(t + "subsample_conv_projection.input_proj_linear.weight", {64, 32 * 4});
    c.add(t + "output_proj.weight", {40, 64});
    c.add(t + "output_proj.bias", {40});
    c.add("model.embed_audio.embedding_projection.weight", {48, 40});
    for (int i = 0; i < layers; ++i) {
        const std::string p = t + "layers." + std::to_string(i) + ".";
        for (const char* f : {"feed_forward1", "feed_forward2"}) {
            c.add(p + f + ".pre_layer_norm.weight", {64});
            c.add(p + f + ".post_layer_norm.weight", {64});
            c.linear(p + f + ".ffw_layer_1", 256, 64);
            c.linear(p + f + ".ffw_layer_2", 64, 256);
        }
        for (const char* n : {"norm_pre_attn", "norm_post_attn", "norm_out"})
            c.add(p + n + ".weight", {64});
        for (const char* n : {"q_proj", "k_proj", "v_proj", "post"})
            c.linear(p + "self_attn." + n, 64, 64);
        c.add(p + "self_attn.relative_k_proj.weight", {64, 64}, 0.0f);
        c.add(p + "self_attn.per_dim_scale", {32}, 0.0f);
        c.add(p + "lconv1d.pre_layer_norm.weight", {64});
        c.add(p + "lconv1d.conv_norm.weight", {64});
        c.add(p + "lconv1d.depthwise_conv1d.weight", {64, 1, 5});
        c.linear(p + "lconv1d.linear_start", 128, 64);
        c.linear(p + "lconv1d.linear_end", 64, 64);
    }
    return c;
}

TEST(Gemma4AudioLoad, ParsesConfigAndTokenIds) {
    auto c = parse_gemma4_audio_config(parse(kConfig));
    ASSERT_TRUE(c.has_value()) << c.error();
    EXPECT_EQ(c->hidden_size, 64);
    EXPECT_EQ(c->head_dim(), 32);
    EXPECT_EQ(c->rel_positions(), 13);  // distances 12..0
    EXPECT_EQ(c->sub_channels0, 8);
    EXPECT_EQ(c->lm_d_model, 48);
    EXPECT_EQ(c->audio_token_id, 258881);
    EXPECT_EQ(c->boa_token_id, 256000);
    EXPECT_EQ(c->eoa_token_id, 258883);
}

TEST(Gemma4AudioLoad, NoAudioConfigIsNotAnError) {
    auto c = parse_gemma4_audio_config(parse(R"({"audio_config": null})"));
    ASSERT_FALSE(c.has_value());
    EXPECT_TRUE(c.error().empty());
    auto other = parse_gemma4_audio_config(parse(R"({"audio_config": {"model_type": "whisper"}})"));
    ASSERT_FALSE(other.has_value());
    EXPECT_FALSE(other.error().empty());
}

TEST(Gemma4AudioLoad, FillsEverySlotAndDerivesTables) {
    auto c = parse_gemma4_audio_config(parse(kConfig));
    ASSERT_TRUE(c.has_value());
    Checkpoint ck = tiny_checkpoint(2);
    auto m = load_gemma4_audio_tower(*c, ck.t);
    ASSERT_TRUE(m.has_value()) << m.error();
    int slots = 0;
    gemma4_audio_visit_tensors(**m, [&](Tensor& t, const std::string& what, bool) {
        EXPECT_NE(t.data, nullptr) << what;
        ++slots;
    });
    EXPECT_EQ(slots, 5 + (2 * 22) + 3);
    const auto& L = (*m)->layers[1];
    EXPECT_EQ(L.q.in_min, -4.0f);
    EXPECT_EQ(L.q.out_max, 8.0f);
    // per_dim_scale 0 -> softplus = ln 2, so q_scale = hd^-0.5 exactly.
    const float* qs = static_cast<const float*>(L.q_scale.data);
    EXPECT_NEAR(qs[0], 1.0f / std::sqrt(32.0f), 1e-6f);
    // relative_k_proj 0 -> rel_k 0, shape [13, 64].
    EXPECT_EQ(L.rel_k.shape[0], 13);
    EXPECT_EQ(static_cast<const float*>(L.rel_k.data)[5], 0.0f);
    EXPECT_GT(gemma4_audio_tower_device_bytes(**m), 0u);
    EXPECT_GT(Gemma4AudioEncoder::demand_bytes(*c, 2999), 0u);
}

TEST(Gemma4AudioLoad, RefusesMissingTensorAndWrongShape) {
    auto c = parse_gemma4_audio_config(parse(kConfig));
    ASSERT_TRUE(c.has_value());
    Checkpoint ck = tiny_checkpoint(1);  // config says 2 layers
    auto m = load_gemma4_audio_tower(*c, ck.t);
    ASSERT_FALSE(m.has_value());
    EXPECT_NE(m.error().find("layers.1"), std::string::npos) << m.error();

    Checkpoint bad = tiny_checkpoint(2);
    bad.t.erase("model.audio_tower.output_proj.bias");
    bad.add("model.audio_tower.output_proj.bias", {41});
    EXPECT_FALSE(load_gemma4_audio_tower(*c, bad.t).has_value());
}

TEST(Gemma4AudioLoad, MaxFramesCoverThirtySeconds) {
    EXPECT_EQ(Gemma4AudioEncoder::max_frames_for(Gemma4AudioFrontend{}), 2999);
}

}  // namespace
}  // namespace imp
