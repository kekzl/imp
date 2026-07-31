// Qwen3-VL vision tensor-name mapping.
//
// This is tested exhaustively because a misrouted vision weight does not crash:
// it yields an encoder that runs and returns embeddings unrelated to the image.
// The oracle is the real name list from the staged Qwen3-VL-4B-Instruct
// checkpoint (315 tensors under `model.visual.`), reconstructed here so the test
// needs neither the checkpoint nor a GPU.

#include "vision/qwen3vl_vision_map.h"

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace imp {
namespace {

using Slot = Qwen3VLVisionSlot;

// The checkpoint's `model.visual.*` names with the prefix stripped, for the real
// geometry: depth 24, 3 DeepStack mergers.
std::vector<std::string> real_name_list() {
    std::vector<std::string> names = {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight"};
    for (int b = 0; b < 24; ++b) {
        const std::string p = "blocks." + std::to_string(b) + ".";
        for (const char* t :
             {"norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
              "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
              "mlp.linear_fc2.weight", "mlp.linear_fc2.bias"})
            names.push_back(p + t);
    }
    for (const char* t : {"norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
                          "linear_fc2.weight", "linear_fc2.bias"})
        names.push_back(std::string("merger.") + t);
    for (int d = 0; d < 3; ++d) {
        const std::string p = "deepstack_merger_list." + std::to_string(d) + ".";
        for (const char* t : {"norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
                              "linear_fc2.weight", "linear_fc2.bias"})
            names.push_back(p + t);
    }
    return names;
}

TEST(Qwen3VLVisionMap, EveryRealTensorIsRecognisedExactlyOnce) {
    const auto names = real_name_list();
    ASSERT_EQ(names.size(), 315u) << "the checkpoint has 315 model.visual.* tensors";

    std::set<std::pair<int, int>> seen;  // (slot, index) must be unique
    for (const auto& n : names) {
        const auto r = qwen3vl_map_vision_tensor(n);
        ASSERT_NE(r.slot, Slot::Unknown) << "unmapped: " << n;
        const auto key = std::make_pair(static_cast<int>(r.slot), r.index);
        EXPECT_TRUE(seen.insert(key).second) << "two tensors map to the same slot: " << n;
    }
    EXPECT_EQ(seen.size(), names.size());
}

// The main merger and the DeepStack mergers share sub-names. Routing a
// DeepStack tensor into the main merger would still produce a running encoder.
TEST(Qwen3VLVisionMap, MainMergerAndDeepstackAreDistinct) {
    const auto main_m = qwen3vl_map_vision_tensor("merger.linear_fc1.weight");
    EXPECT_EQ(main_m.slot, Slot::MergerFc1Weight);
    EXPECT_EQ(main_m.index, -1) << "the main merger must be index -1";

    for (int d = 0; d < 3; ++d) {
        const auto ds = qwen3vl_map_vision_tensor("deepstack_merger_list." + std::to_string(d) +
                                                  ".linear_fc1.weight");
        EXPECT_EQ(ds.slot, Slot::MergerFc1Weight);
        EXPECT_EQ(ds.index, d) << "DeepStack merger index must survive";
    }
}

TEST(Qwen3VLVisionMap, BlockIndexIsParsedNotAssumed) {
    for (int b : {0, 1, 9, 10, 23}) {
        const auto r = qwen3vl_map_vision_tensor("blocks." + std::to_string(b) + ".attn.qkv.weight");
        EXPECT_EQ(r.slot, Slot::QkvWeight);
        EXPECT_EQ(r.index, b);
    }
}

// A non-numeric or missing index must NOT silently become block 0 — that would
// stack every layer's weights onto the first one.
TEST(Qwen3VLVisionMap, MalformedIndexIsUnknownNotZero) {
    for (const char* n : {"blocks.x.norm1.weight", "blocks..norm1.weight", "blocks.norm1.weight",
                          "deepstack_merger_list.x.norm.weight"}) {
        const auto r = qwen3vl_map_vision_tensor(n);
        EXPECT_EQ(r.slot, Slot::Unknown) << n;
        EXPECT_EQ(r.index, -1) << n;
    }
}

TEST(Qwen3VLVisionMap, UnrelatedNamesAreUnknown) {
    for (const char* n : {"", "blocks.0.attn.q_proj.weight", "merger.linear_fc3.weight", "patch_embed.proj",
                          "pos_embed", "language_model.layers.0.mlp.up_proj.weight"}) {
        EXPECT_EQ(qwen3vl_map_vision_tensor(n).slot, Slot::Unknown) << n;
    }
}

// Every block slot must be reachable — a typo in one of the twelve tails would
// otherwise only show up as a null tensor much later.
TEST(Qwen3VLVisionMap, AllTwelvePerBlockSlotsAreCovered) {
    std::set<int> slots;
    for (const char* t :
         {"norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
          "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
          "mlp.linear_fc2.weight", "mlp.linear_fc2.bias"}) {
        const auto r = qwen3vl_map_vision_tensor(std::string("blocks.7.") + t);
        ASSERT_NE(r.slot, Slot::Unknown) << t;
        EXPECT_EQ(r.index, 7);
        slots.insert(static_cast<int>(r.slot));
    }
    EXPECT_EQ(slots.size(), 12u) << "two block tails collapsed onto one slot";
}

}  // namespace
}  // namespace imp
