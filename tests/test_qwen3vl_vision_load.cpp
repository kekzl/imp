// Routing the Qwen3-VL vision weights into the tower.
//
// The failure this guards is silent: a tower that loads with a null slot, or
// with the merger norms swapped, still runs and returns embeddings unrelated to
// the image. So the loader refuses, and these tests pin what it refuses.
//
// Oracle: the tensor names AND shapes of the staged Qwen3-VL-4B-Instruct
// checkpoint, reconstructed here so the test needs neither it nor a GPU.

#include "vision/qwen3vl_vision_load.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

// Descriptors only — nothing dereferences the data, but a null pointer is how
// the loader spells "missing", so it has to be non-null.
char g_dummy[64];

Tensor tensor(std::vector<int64_t> shape) {
    Tensor t;
    t.data = g_dummy;
    t.qtype = QType::F16;
    t.ndim = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
        t.shape[i] = shape[i];
    t.compute_strides();
    return t;
}

VisionConfig real_config() {
    VisionConfig c;
    c.is_qwen3vl = true;
    c.num_layers = 24;
    c.hidden_size = 1024;
    c.num_heads = 16;
    c.head_dim = 64;
    c.intermediate_size = 4096;
    c.patch_size = 16;
    c.merge_size = 2;
    c.temporal_patch_size = 2;
    c.out_hidden_size = 2560;
    c.pos_embed_grid = 48;
    c.deepstack_indexes = {5, 11, 17};
    return c;
}

// The real 315 `model.visual.*` tensors with their real shapes.
std::unordered_map<std::string, Tensor> real_checkpoint() {
    std::unordered_map<std::string, Tensor> m;
    auto add = [&](const std::string& n, std::vector<int64_t> s) { m["model.visual." + n] = tensor(s); };

    // conv3d [1024, 3, 2, 16, 16], flattened to 2-D by the SafeTensors loader.
    add("patch_embed.proj.weight", {1024, 1536});
    add("patch_embed.proj.bias", {1024});
    add("pos_embed.weight", {2304, 1024});
    for (int b = 0; b < 24; ++b) {
        const std::string p = "blocks." + std::to_string(b) + ".";
        add(p + "norm1.weight", {1024});
        add(p + "norm1.bias", {1024});
        add(p + "attn.qkv.weight", {3072, 1024});
        add(p + "attn.qkv.bias", {3072});
        add(p + "attn.proj.weight", {1024, 1024});
        add(p + "attn.proj.bias", {1024});
        add(p + "norm2.weight", {1024});
        add(p + "norm2.bias", {1024});
        add(p + "mlp.linear_fc1.weight", {4096, 1024});
        add(p + "mlp.linear_fc1.bias", {4096});
        add(p + "mlp.linear_fc2.weight", {1024, 4096});
        add(p + "mlp.linear_fc2.bias", {1024});
    }
    // Main merger: norm sits BEFORE the 2x2 concat, so it is hidden-wide.
    add("merger.norm.weight", {1024});
    add("merger.norm.bias", {1024});
    add("merger.linear_fc1.weight", {4096, 4096});
    add("merger.linear_fc1.bias", {4096});
    add("merger.linear_fc2.weight", {2560, 4096});
    add("merger.linear_fc2.bias", {2560});
    // DeepStack mergers: norm sits AFTER the concat, so it is four times wider.
    for (int d = 0; d < 3; ++d) {
        const std::string p = "deepstack_merger_list." + std::to_string(d) + ".";
        add(p + "norm.weight", {4096});
        add(p + "norm.bias", {4096});
        add(p + "linear_fc1.weight", {4096, 4096});
        add(p + "linear_fc1.bias", {4096});
        add(p + "linear_fc2.weight", {2560, 4096});
        add(p + "linear_fc2.bias", {2560});
    }
    return m;
}

VisionModel fresh_tower() {
    VisionModel v;
    v.config = real_config();
    return v;
}

TEST(Qwen3VLVisionLoad, LoadsTheRealCheckpoint) {
    auto tensors = real_checkpoint();
    ASSERT_EQ(tensors.size(), 315u);

    VisionModel v = fresh_tower();
    const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
    ASSERT_TRUE(loaded) << loaded.error().what;
    const Qwen3VLVisionLoadStats& st = *loaded;

    EXPECT_EQ(st.assigned, 315);
    EXPECT_EQ(st.unknown, 0);
    EXPECT_EQ(st.missing, 0);
    ASSERT_EQ(v.layers.size(), 24u);
    ASSERT_EQ(v.deepstack_mergers.size(), 3u);

    // Spot-check that the routing landed where the encoder will look, not just
    // that nothing was null.
    EXPECT_EQ(v.layers[7].wq.shape[0], 3072) << "the fused QKV must stay whole";
    EXPECT_EQ(v.layers[7].ffn_down_w.shape[1], 4096);
    EXPECT_EQ(v.patch_embd_w.shape[1], 1536);
    EXPECT_EQ(v.position_embd.shape[0], 2304);
    EXPECT_EQ(v.merger.norm_w.shape[0], 1024);
    EXPECT_EQ(v.deepstack_mergers[2].norm_w.shape[0], 4096);
}

// The whole reason this returns bool: a tower with a hole must not reach the
// encoder, where it would read as a garbage embedding many layers later.
TEST(Qwen3VLVisionLoad, RefusesWhenASlotStaysNull) {
    auto tensors = real_checkpoint();
    tensors.erase("model.visual.blocks.13.attn.proj.bias");

    VisionModel v = fresh_tower();
    const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
    ASSERT_FALSE(loaded.has_value());
    const std::string& err = loaded.error().what;
    const Qwen3VLVisionLoadStats& st = loaded.error().stats;
    EXPECT_EQ(st.missing, 1);
    EXPECT_NE(err.find("blocks.13.attn.proj.bias"), std::string::npos) << err;
}

// Main merger normalises before the 2x2 concat, DeepStack after it. Swapping
// them normalises the wrong axis and still runs — only the shape says which.
TEST(Qwen3VLVisionLoad, RefusesSwappedMergerNormWidths) {
    auto tensors = real_checkpoint();
    tensors["model.visual.merger.norm.weight"] = tensor({4096});

    VisionModel v = fresh_tower();
    const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
    ASSERT_FALSE(loaded.has_value());
    const std::string& err = loaded.error().what;
    EXPECT_NE(err.find("merger.norm.weight"), std::string::npos) << err;
    EXPECT_NE(err.find("1024"), std::string::npos) << "the message must name the expected width: " << err;
}

TEST(Qwen3VLVisionLoad, RefusesAShapeTheConfigContradicts) {
    struct Case {
        const char* name;
        std::vector<int64_t> shape;
    };
    const Case cases[] = {
        {"model.visual.blocks.0.attn.qkv.weight", {1024, 1024}},  // not fused
        {"model.visual.pos_embed.weight", {2305, 1024}},          // wrong grid
        {"model.visual.patch_embed.proj.weight", {1024, 768}},    // temporal axis dropped
        {"model.visual.merger.linear_fc2.weight", {2560, 1024}},  // pre-concat width
        {"model.visual.blocks.3.norm1.weight", {1024, 1}},        // wrong rank
    };
    for (const auto& c : cases) {
        auto tensors = real_checkpoint();
        tensors[c.name] = tensor(c.shape);
        VisionModel v = fresh_tower();
        const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
        ASSERT_FALSE(loaded.has_value()) << c.name;
        EXPECT_NE(loaded.error().what.find("shape"), std::string::npos)
            << c.name << ": " << loaded.error().what;
    }
}

// LM weights ride in the same shard map. Touching them here would double-assign
// tensors the WeightMap owns.
TEST(Qwen3VLVisionLoad, IgnoresEverythingOutsideTheVisualPrefix) {
    auto tensors = real_checkpoint();
    tensors["model.language_model.layers.0.mlp.up_proj.weight"] = tensor({9728, 2560});
    tensors["lm_head.weight"] = tensor({151936, 2560});
    tensors["visual.blocks.0.norm1.weight"] = tensor({1024});  // prefix-like, not the prefix

    VisionModel v = fresh_tower();
    const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
    ASSERT_TRUE(loaded) << loaded.error().what;
    const Qwen3VLVisionLoadStats& st = *loaded;
    EXPECT_EQ(st.assigned, 315);
    EXPECT_EQ(st.unknown, 0);
}

// An unrecognised vision tensor is reported but not fatal: a future checkpoint
// may carry extras, and refusing the whole tower over one would be worse than
// loading it and saying so.
TEST(Qwen3VLVisionLoad, CountsUnknownVisualTensorsWithoutFailing) {
    auto tensors = real_checkpoint();
    tensors["model.visual.some_future_head.weight"] = tensor({8, 8});

    VisionModel v = fresh_tower();
    const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
    EXPECT_TRUE(loaded) << loaded.error().what;
    const Qwen3VLVisionLoadStats& st = *loaded;
    EXPECT_EQ(st.unknown, 1);
    EXPECT_EQ(st.assigned, 315);
}

// Order matters: the config parse sizes the layer and merger vectors. Running
// this first would resize to zero and route every block into nothing.
TEST(Qwen3VLVisionLoad, RefusesAnUnparsedConfig) {
    VisionModel v;  // is_qwen3vl still false
    const auto loaded = load_qwen3vl_vision_tensors(real_checkpoint(), v);
    ASSERT_FALSE(loaded.has_value());
    const std::string& err = loaded.error().what;
    const Qwen3VLVisionLoadStats& st = loaded.error().stats;
    EXPECT_NE(err.find("not parsed"), std::string::npos) << err;
    EXPECT_EQ(st.assigned, 0);
}

// A DeepStack tap the config does not list has nowhere to go. Silently dropping
// it would leave the LM injecting one embedding fewer than the checkpoint has.
TEST(Qwen3VLVisionLoad, RefusesADeepstackMergerWithNoConfiguredSlot) {
    auto tensors = real_checkpoint();
    // Shaped correctly, so the refusal can only come from the missing slot.
    const std::pair<const char*, std::vector<int64_t>> extra[] = {
        {"norm.weight", {4096}},
        {"norm.bias", {4096}},
        {"linear_fc1.weight", {4096, 4096}},
        {"linear_fc1.bias", {4096}},
        {"linear_fc2.weight", {2560, 4096}},
        {"linear_fc2.bias", {2560}},
    };
    for (const auto& [tail, shape] : extra)
        tensors[std::string("model.visual.deepstack_merger_list.3.") + tail] = tensor(shape);

    VisionModel v = fresh_tower();
    const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
    ASSERT_FALSE(loaded.has_value());
    const std::string& err = loaded.error().what;
    EXPECT_NE(err.find("no slot"), std::string::npos) << err;
}

// Depth comes from the config, so a checkpoint with more blocks is a mismatch,
// not something to grow into.
TEST(Qwen3VLVisionLoad, RefusesABlockIndexPastTheConfiguredDepth) {
    auto tensors = real_checkpoint();
    tensors["model.visual.blocks.24.norm1.weight"] = tensor({1024});

    VisionModel v = fresh_tower();
    const auto loaded = load_qwen3vl_vision_tensors(tensors, v);
    ASSERT_FALSE(loaded.has_value());
    const std::string& err = loaded.error().what;
    EXPECT_NE(err.find("exceeds depth"), std::string::npos) << err;
}

}  // namespace
}  // namespace imp
