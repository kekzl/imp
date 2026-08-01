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

    // Status flag set true when ALL of the above tensors are populated.
    bool loaded = false;
};

}  // namespace imp
