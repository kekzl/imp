#pragma once

#include "core/tensor.h"
#include <vector>
#include <string>
#include <cmath>

namespace imp {

enum class OpType {
    EMBEDDING,
    RMSNORM,
    RMSNORM_RESIDUAL,
    ROPE,
    QKV_PROJ,
    ATTENTION_PREFILL,
    ATTENTION_DECODE,
    O_PROJ,
    GATE_PROJ,
    UP_PROJ,
    DOWN_PROJ,
    SWIGLU,
    RESIDUAL_ADD,
    MOE_GATE,
    MOE_GATHER,
    GROUPED_GEMM,
    MOE_SCATTER,
    OUTPUT_NORM,
    LM_HEAD,
    SAMPLING
};

const char* op_type_name(OpType type);

// Op-specific parameters

struct RMSNormParams {
    float eps = 1e-5f;
};

struct RoPEParams {
    int head_dim = 128;
    float theta = 10000.0f;
    float scaling = 1.0f;
};

struct AttentionParams {
    float scale = 0.0f;        // 1/sqrt(head_dim), computed at build time
    bool causal = true;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
};

struct PagedAttentionParams {
    float scale = 0.0f;
    int block_size = 16;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
};

struct GEMMParams {
    float alpha = 1.0f;
    float beta = 0.0f;
    // Output dimensions for documentation:
    // M = n_tokens, N = output_dim, K = input_dim
    int output_dim = 0;
};

struct SamplingParams {
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
};

// Union-like parameter storage for OpNode
struct OpParams {
    RMSNormParams rmsnorm;
    RoPEParams rope;
    AttentionParams attention;
    PagedAttentionParams paged_attention;
    GEMMParams gemm;
    SamplingParams sampling;
};

struct OpNode {
    int id = 0;
    OpType type = OpType::EMBEDDING;
    std::vector<int> inputs;   // indices into graph node list
    std::vector<int> outputs;
    int layer = -1;            // transformer layer index (-1 for non-layer ops)
    OpParams params;
};

} // namespace imp
