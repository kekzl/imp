#pragma once
// =============================================================================
// mtp_head.h — Multi-Token Predictor head storage
// =============================================================================
//
// Trained MTP head shipped alongside DeepSeek-V3-family models (Qwen3.6,
// DeepSeek V3, etc.) as `model_mtp.safetensors` in the model directory.
//
// Phase 1.A: detection metadata only.
// Phase 1.B (this expansion): named Tensor fields, weights loaded as BF16.
// Phase 2: forward kernel implementation.
// Phase 3+: verify-loop integration.
//
// Reference architecture (Qwen3.6-NVFP4 MTP, 1.6 GB BF16, 19 tensors):
//
//   FC + pre-FC norms (token-conditioning block):
//     mtp.fc                          [2048, 4096]   project concat(emb, h_prev)
//     mtp.pre_fc_norm_embedding       [2048]         RMSNorm on embedding input
//     mtp.pre_fc_norm_hidden          [2048]         RMSNorm on hidden_state input
//
//   Single transformer layer (mtp.layers.0.*):
//     input_layernorm                 [2048]
//     post_attention_layernorm        [2048]
//     self_attn.q_proj                [8192, 2048]   16 heads × 512 head_dim
//                                                    OR n_heads MQA-style variant
//     self_attn.k_proj                [512, 2048]    2 kv_heads × 256 head_dim
//     self_attn.v_proj                [512, 2048]
//     self_attn.o_proj                [2048, 4096]
//     self_attn.q_norm                [256]          per-head RMSNorm
//     self_attn.k_norm                [256]
//     mlp.gate                        [256, 2048]    256 experts router
//     mlp.experts.gate_up_proj        [256, 1024, 2048]  256 experts × (gate+up) packed
//     mlp.experts.down_proj           [256, 2048, 512]
//     mlp.shared_expert.gate_proj     [512, 2048]
//     mlp.shared_expert.up_proj       [512, 2048]
//     mlp.shared_expert.down_proj     [2048, 512]
//     mlp.shared_expert_gate          [1, 2048]      sigmoid-gated shared expert
//
//   Final norm:
//     mtp.norm                        [2048]
//
//   LM head: SHARED with main model's `model.lm_head.weight` (not stored here).
//
// Second layout (Nemotron-3.5-Lightning, 2.6 GB BF16, 270 tensors). Same idea,
// but it is a miniature Nemotron rather than a miniature Qwen: the head spans
// TWO blocks — attention in `layers.0`, MoE in `layers.1` — mirroring the
// hybrid main model, and every name differs:
//
//     mtp.layers.0.enorm / hnorm      [2688]         = pre_fc_norm_{embedding,hidden}
//     mtp.layers.0.eh_proj            [2688, 5376]   = fc (concat(emb,h) → hidden)
//     mtp.layers.0.norm               [2688]         = input_layernorm
//     mtp.layers.0.mixer.{q,k,v,o}_proj              = self_attn.* (no q/k norm)
//     mtp.layers.1.norm               [2688]         = post_attention_layernorm
//     mtp.layers.1.mixer.gate.weight  [128, 2688]    router
//     mtp.layers.1.mixer.gate.e_score_correction_bias [128]  DeepSeek-style bias
//     mtp.layers.1.mixer.experts.{e}.up_proj   [1856, 2688]  128 experts, PER-EXPERT
//     mtp.layers.1.mixer.experts.{e}.down_proj [2688, 1856]  2-D, not packed 3-D
//     mtp.layers.1.mixer.shared_experts.{up,down}_proj
//     mtp.layers.1.final_layernorm    [2688]         = final norm
//
// Two structural differences the forward pass has to honour, not just the
// names: the experts are NON-GATED (no `gate_proj` — squared-ReLU, like the
// main Nemotron FFN) and there is no sigmoid `shared_expert_gate`.
// =============================================================================

#include "core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Phase 1.A leftover — kept as part of the new MtpHead struct for compatibility
// with the existing Model::mtp_info_ field.
struct MtpHeadInfo {
    std::string path;
    size_t      file_bytes = 0;
    int         n_tensors  = 0;
};

// Full MTP head storage. Populated by safetensors_loader when
// `model_mtp.safetensors` is present alongside the main weights and
// `runtime.mtp_spec_decode > 0` is enabled. Otherwise empty / .loaded=false.
struct MtpHead {
    MtpHeadInfo info;

    // Token-conditioning block:
    //   mtp_in = norm(emb(t)) || norm(h_prev)   then fc projects 4096 → 2048
    Tensor pre_fc_norm_embedding;
    Tensor pre_fc_norm_hidden;
    Tensor fc;

    // Single transformer layer (mtp.layers.0.*):
    Tensor input_layernorm;
    Tensor post_attention_layernorm;

    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor q_norm;
    Tensor k_norm;

    Tensor router;                          // mlp.gate.weight
    Tensor experts_gate_up_packed;          // [256, 1024, 2048] packed gate+up per expert
    Tensor experts_down_packed;             // [256, 2048, 512]

    Tensor shared_expert_gate_proj;
    Tensor shared_expert_up_proj;
    Tensor shared_expert_down_proj;
    Tensor shared_expert_gate;              // [1, hidden] sigmoid gate

    Tensor final_norm;                      // mtp.norm.weight

    // --- Nemotron-3.5 layout (see header comment) -----------------------------
    // Per-expert 2-D weights instead of the packed 3-D pair above. Empty on the
    // Qwen layout; when non-empty the forward pass indexes these directly and
    // ignores experts_*_packed.
    std::vector<Tensor> experts_up;    // [n_experts] each [d_ff_e, hidden]
    std::vector<Tensor> experts_down;  // [n_experts] each [hidden, d_ff_e]
    // DeepSeek-style additive score bias on the router logits. Null when absent.
    Tensor router_score_bias;  // [n_experts] FP32
    // Experts have no gate_proj: activation is squared ReLU, not SwiGLU. This is
    // a property of the checkpoint, so it is recorded rather than re-derived
    // from which tensors happen to be present at each use site.
    bool experts_non_gated = false;
    // Qwen3.6's MTP attention is attn_output_gate=True: q_proj emits
    // [num_heads, 2*head_dim] and the second half gates the output. Nemotron's
    // does not, and its attention is NoPE — the hybrid's Mamba layers carry
    // position, so applying RoPE here would rotate against the main model.
    // Both default to the Qwen behaviour so that path is untouched.
    bool attn_output_gate = true;
    bool attn_rope = true;

    // Status flag set true when ALL of the above tensors are populated.
    bool loaded = false;
};

}  // namespace imp
