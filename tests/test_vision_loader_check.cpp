// The mmproj GGUF loader's two refusals.
//
// The failure they guard is silent. A checkpoint in a dialect this loader does
// not read still parses: the names it shares land in their slots, the rest are
// dropped at DEBUG, and the model comes back looking loaded — then the encoder
// hands a null slot to vision_gemm. Qwen3-VL's mmproj did exactly that: 247 of
// 316 tensors assigned, no error, no output worth anything.
//
// Oracle: the tensor inventories of the three mmproj files this box actually
// has — gemma-3-4b (projector_type `gemma3`), gemma-4-26b (`gemma4v`) and
// Qwen3-VL-4B (`qwen3vl_merger`) — read off the GGUF headers and reproduced
// here as slot sets, so the test needs neither the files nor a GPU.

#include "vision/vision_loader_check.h"

#include <gtest/gtest.h>

#include <string>

namespace imp {
namespace {

constexpr int kLayers = 4;   // shape of the walk, not of any real tower
constexpr int kHidden = 32;  // any non-zero shape: only ndim is read

// Shaped but not uploaded — which is precisely the probe pass, where the loader
// counts bytes instead of taking arena slabs and every slot's data stays null.
// A test that handed out real pointers would not notice `filled()` regressing
// to a null-pointer test, and the probe would then reject every model.
Tensor shaped(int64_t n = kHidden) {
    Tensor t;
    t.data = nullptr;
    t.qtype = QType::F16;
    t.ndim = 1;
    t.shape[0] = n;
    return t;
}

// The slots gemma-3's mmproj fills: LayerNorm blocks with biases, separate
// q/k/v with biases, one projection.
VisionModel gemma3_model() {
    VisionModel m;
    m.config.is_gemma4v = false;
    m.config.num_layers = kLayers;
    m.layers.resize(kLayers);

    m.patch_embd_w = shaped();
    m.patch_embd_b = shaped();
    m.position_embd = shaped();
    m.post_norm_w = shaped();
    m.post_norm_b = shaped();
    m.mm_proj_w = shaped();

    for (auto& l : m.layers) {
        l.ln1_w = l.ln1_b = l.ln2_w = l.ln2_b = shaped();
        l.wq = l.wk = l.wv = l.wo = shaped();
        l.bq = l.bk = l.bv = l.bo = shaped();
        l.ffn_up_w = l.ffn_up_b = l.ffn_down_w = l.ffn_down_b = shaped();
    }
    return m;
}

// gemma-4v: RMSNorm (no LN biases), no attention biases, no patch_embd bias and
// no post_ln at all — but per-head q/k/v norms, sandwich post-norms and a GeGLU
// gate. Its inventory is a poor subset of gemma-3's, which is the whole reason
// the required set has to branch on is_gemma4v.
VisionModel gemma4v_model() {
    VisionModel m;
    m.config.is_gemma4v = true;
    m.config.num_layers = kLayers;
    m.layers.resize(kLayers);

    m.patch_embd_w = shaped();
    m.position_embd = shaped();
    m.mm_proj_w = shaped();
    m.std_scale = m.std_bias = shaped();

    for (auto& l : m.layers) {
        l.ln1_w = l.ln2_w = shaped();
        l.wq = l.wk = l.wv = l.wo = shaped();
        l.ffn_up_w = l.ffn_down_w = l.ffn_gate_w = shaped();
        l.q_norm = l.k_norm = shaped();
        l.attn_post_norm = l.ffn_post_norm = shaped();
    }
    return m;
}

// ---- Gate 1: named dialects ----

TEST(VisionLoaderCheck, RejectsQwen3VLMerger) {
    const std::string reason = vision_projector_reject_reason("qwen3vl_merger");
    ASSERT_FALSE(reason.empty());
    // The message has to point somewhere. Being told "unsupported" and left to
    // guess is how this cost an afternoon in the first place.
    EXPECT_NE(reason.find("--model"), std::string::npos);
    EXPECT_NE(reason.find("SafeTensors"), std::string::npos);
}

TEST(VisionLoaderCheck, AcceptsTheDialectsItReads) {
    EXPECT_EQ(vision_projector_reject_reason("gemma3"), "");
    EXPECT_EQ(vision_projector_reject_reason("gemma4v"), "");
    // No projector_type key at all is the LLaVA-ish default, not a refusal.
    EXPECT_EQ(vision_projector_reject_reason(""), "");
}

// ---- Gate 2: completeness ----

TEST(VisionLoaderCheck, Gemma3TowerIsComplete) { EXPECT_EQ(vision_model_missing_slot(gemma3_model()), ""); }

TEST(VisionLoaderCheck, Gemma4vTowerIsComplete) { EXPECT_EQ(vision_model_missing_slot(gemma4v_model()), ""); }

// The bug, reproduced at the level the check runs at: Qwen3-VL exports one
// fused attn_qkv per block, so the three slots the loader looks for stay empty
// while everything around them fills.
TEST(VisionLoaderCheck, CatchesFusedQkvLeavingQkvEmpty) {
    VisionModel m = gemma3_model();
    for (auto& l : m.layers)
        l.wq = l.wk = l.wv = Tensor{};

    EXPECT_EQ(vision_model_missing_slot(m), "v.blk.0.attn_q.weight");
}

TEST(VisionLoaderCheck, ReportsTheOffendingBlockIndex) {
    VisionModel m = gemma3_model();
    m.layers[2].ffn_down_w = Tensor{};

    EXPECT_EQ(vision_model_missing_slot(m), "v.blk.2.ffn_down.weight");
}

TEST(VisionLoaderCheck, CatchesMissingGlobals) {
    {
        VisionModel m = gemma3_model();
        m.patch_embd_w = Tensor{};
        EXPECT_EQ(vision_model_missing_slot(m), "v.patch_embd.weight");
    }
    {
        // Qwen3-VL's other half: a two-layer merger, so `mm.0` alone is not the
        // projection and a loader that only knows `mm.0` can end up with none.
        VisionModel m = gemma3_model();
        m.mm_proj_w = Tensor{};
        EXPECT_EQ(vision_model_missing_slot(m), "mm.0.weight / mm.input_projection.weight");
    }
}

TEST(VisionLoaderCheck, CatchesMissingGemma4vBlockTensors) {
    VisionModel m = gemma4v_model();
    m.layers[1].ffn_gate_w = Tensor{};
    EXPECT_EQ(vision_model_missing_slot(m), "v.blk.1.ffn_gate.weight");
}

// ---- Not over-reaching ----
//
// Every slot below is absent from a real mmproj this box loads today, or is
// guarded by an `if (.data)` in the encoder. Requiring any of them would turn a
// working model into a hard failure, which is a worse bug than the one being
// fixed.

TEST(VisionLoaderCheck, OptionalSlotsDoNotBlockLoad) {
    VisionModel m = gemma3_model();
    m.patch_embd_b = Tensor{};  // absent in gemma-4v
    m.post_norm_w = Tensor{};   // absent in gemma-4v
    m.post_norm_b = Tensor{};
    m.mm_proj_b = Tensor{};  // gemma-3 exports no projection bias
    m.mm_post_norm_w = Tensor{};
    m.position_embd = Tensor{};  // encoder adds it only when present
    for (auto& l : m.layers) {
        l.bq = l.bk = l.bv = l.bo = Tensor{};
        l.ffn_up_b = l.ffn_down_b = Tensor{};
    }

    EXPECT_EQ(vision_model_missing_slot(m), "");
}

// gemma4v reads the position table unconditionally (axial RoPE), so there it is
// required — the one slot whose status differs between the two dialects.
TEST(VisionLoaderCheck, PositionEmbedIsRequiredOnlyForGemma4v) {
    VisionModel m = gemma4v_model();
    m.position_embd = Tensor{};
    EXPECT_EQ(vision_model_missing_slot(m), "v.position_embd.weight");
}

TEST(VisionLoaderCheck, Gemma4vDoesNotRequireLayerNormBiases) {
    VisionModel m = gemma4v_model();
    for (auto& l : m.layers)
        l.ln1_b = l.ln2_b = Tensor{};
    EXPECT_EQ(vision_model_missing_slot(m), "");
}

}  // namespace
}  // namespace imp
