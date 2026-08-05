#include "vision/vision_loader_check.h"

namespace imp {

std::string vision_projector_reject_reason(const std::string& projector) {
    if (projector == "qwen3vl_merger") {
        return "projector_type 'qwen3vl_merger': the mmproj GGUF loader reads the LLaVA/SigLIP "
               "layout (separate attn_q/attn_k/attn_v, one projection), and Qwen3-VL exports fused "
               "attn_qkv, DeepStack mergers and a two-layer merger. Drop --mmproj and point --model "
               "at the Qwen3-VL SafeTensors checkpoint, which imp loads natively.";
    }
    return {};
}

namespace {

// A slot counts as filled once the loader has given it a shape. See the header
// for why this is not a null-pointer test.
bool filled(const Tensor& t) { return t.ndim > 0; }

}  // namespace

std::string vision_model_missing_slot(const VisionModel& model) {
    const bool g4v = model.config.is_gemma4v;

    if (!filled(model.patch_embd_w))
        return "v.patch_embd.weight";
    if (!filled(model.mm_proj_w))
        return "mm.0.weight / mm.input_projection.weight";
    // Optional for SigLIP (the encoder adds it only when present), but gemma4v's
    // axial RoPE reads the table unconditionally.
    if (g4v && !filled(model.position_embd))
        return "v.position_embd.weight";

    for (size_t i = 0; i < model.layers.size(); ++i) {
        const VisionLayerWeights& l = model.layers[i];
        const std::string blk = "v.blk." + std::to_string(i) + ".";

        if (!filled(l.ln1_w))
            return blk + "ln1.weight";
        if (!filled(l.ln2_w))
            return blk + "ln2.weight";
        if (!filled(l.wq))
            return blk + "attn_q.weight";
        if (!filled(l.wk))
            return blk + "attn_k.weight";
        if (!filled(l.wv))
            return blk + "attn_v.weight";
        if (!filled(l.wo))
            return blk + "attn_out.weight";
        if (!filled(l.ffn_up_w))
            return blk + "ffn_up.weight";
        if (!filled(l.ffn_down_w))
            return blk + "ffn_down.weight";

        if (g4v) {
            if (!filled(l.q_norm))
                return blk + "attn_q_norm.weight";
            if (!filled(l.k_norm))
                return blk + "attn_k_norm.weight";
            if (!filled(l.attn_post_norm))
                return blk + "attn_post_norm.weight";
            if (!filled(l.ffn_post_norm))
                return blk + "ffn_post_norm.weight";
            if (!filled(l.ffn_gate_w))
                return blk + "ffn_gate.weight";
        } else {
            // vision_layernorm_kernel reads beta unconditionally — a missing LN
            // bias is a null dereference, not a defaulted zero.
            if (!filled(l.ln1_b))
                return blk + "ln1.bias";
            if (!filled(l.ln2_b))
                return blk + "ln2.bias";
        }
    }

    return {};
}

}  // namespace imp
