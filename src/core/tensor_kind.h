#pragma once

#include <cstdint>

namespace imp {

using TensorID = int32_t;  // dense index into WeightRegistry::handles_
constexpr TensorID kInvalidTensorID = -1;

enum class TensorKind : uint8_t {
    UNKNOWN = 0,

    // Attention projections
    WQ,
    WK,
    WV,
    WO,
    QKV_FUSED,

    // MLA (Multi-head Latent Attention) projections (DeepSeek-V2/V3)
    KV_A_PROJ,  // kv_a_proj_with_mqa: latent(512)+rope(64) down-projection
    KV_A_NORM,  // kv_a_layernorm: RMSNorm on the 512-dim latent (never quantized)
    KV_B_PROJ,  // kv_b_proj: up-projection, output 16*(128+128)=4096

    // FFN / expert projections
    W_GATE,
    W_UP,
    W_DOWN,
    EXPERT_GATE,
    EXPERT_UP,
    EXPERT_DOWN,

    // Fused variants (populated by planner, not loader)
    FUSED_KV,
    FUSED_GATE_UP,

    // Embeddings
    TOK_EMBED,
    LM_HEAD,

    // MoE routing
    ROUTER,
    SHARED_EXPERT_GATE,

    // GDN / Mamba2 (no quantized path today)
    SSM_IN,
    SSM_OUT,
    CONV1D_W,
    CONV1D_B,
    A_LOG,
    DT_BIAS,
    BETA,
    ALPHA,
    SSM_GROUP_NORM,
    GDN_GATE,
    GDN_ALPHA,
    GDN_BETA,
    GDN_ALPHA_BETA_PACKED,
    GDN_INPUT_PACKED,

    // Norms (always FP32)
    ATTN_NORM,
    FFN_NORM,
    POST_ATTN_NORM,
    POST_FFN_NORM,
    QK_NORM_Q,
    QK_NORM_K,

    // Positional
    ROPE_FREQS,

    // Vision (SigLIP)
    SIGLIP_ATTN,
    SIGLIP_FFN,
    SIGLIP_NORM,
    MM_PROJ,

    COUNT,
};

const char* tensor_kind_name(TensorKind k);

}  // namespace imp
