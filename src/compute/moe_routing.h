#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

struct MoeRoutingResult {
    Tensor expert_indices;   // [n_tokens, top_k] int32
    Tensor expert_weights;   // [n_tokens, top_k] float
    Tensor sorted_token_ids; // tokens sorted by expert
    Tensor expert_offsets;   // [n_experts+1] scan offsets
    int32_t* token_to_expanded = nullptr; // [n_tokens * top_k] inverse map: flat_idx -> expanded row
    bool owns_memory = true; // if true, caller must cudaFree tensors
};

// Pre-allocated buffer pool for MoE routing temporaries.
// Eliminates per-call cudaMalloc during moe_topk_gating.
struct MoeRoutingBuffers {
    void* pool = nullptr;
    size_t pool_size = 0;

    // Pointers into pool
    int32_t* expert_indices = nullptr;   // [max_tokens * top_k]
    float*   expert_weights = nullptr;   // [max_tokens * top_k]
    int32_t* sorted_token_ids = nullptr; // [max_tokens * top_k * 2] (includes flat_idx)
    int32_t* expert_offsets = nullptr;   // [max_experts + 1]
    int32_t* expert_counts = nullptr;    // [max_experts]
    int32_t* expert_write_pos = nullptr; // [max_experts]
    int32_t* token_to_expanded = nullptr; // [max_tokens * top_k] inverse map

    int max_tokens = 0;
    int max_experts = 0;
    int top_k = 0;

    void allocate(int max_tokens, int max_experts, int top_k);
    void free();
    ~MoeRoutingBuffers();
};

void moe_topk_gating(const Tensor& gate_logits, int top_k,
                     MoeRoutingResult& result,
                     cudaStream_t stream = nullptr,
                     bool use_sigmoid = false,
                     bool normalize_weights = true,
                     const void* score_bias = nullptr);

// Variant using pre-allocated buffers (no cudaMalloc inside).
// skip_sorting: when true, only runs the top-k gating kernel and skips
// count/scan/scatter (useful for n=1 decode where gather/scatter is skipped).
void moe_topk_gating(const Tensor& gate_logits, int top_k,
                     MoeRoutingBuffers& buffers,
                     MoeRoutingResult& result,
                     cudaStream_t stream = nullptr,
                     bool use_sigmoid = false,
                     bool normalize_weights = true,
                     const void* score_bias = nullptr,
                     bool skip_sorting = false);

void moe_gather(const Tensor& input, const MoeRoutingResult& routing,
                Tensor& gathered, cudaStream_t stream = nullptr);

void moe_scatter(const Tensor& expert_output, const MoeRoutingResult& routing,
                 Tensor& output, cudaStream_t stream = nullptr);

// Fused token-centric scatter + FP32->FP16 conversion + residual add.
// Replaces: moe_scatter + fp32_to_fp16_kernel + elementwise_add for prefill.
// Uses token_to_expanded inverse map to avoid atomicAdd contention.
// expert_output: [expanded, d_model] FP16. residual: [n_tokens, d_model] FP16 (or nullptr).
// output: [n_tokens, d_model] FP16.
void moe_scatter_fused_residual(const void* expert_output,
                                 const int32_t* token_to_expanded,
                                 const float* expert_weights,
                                 const void* residual, void* output,
                                 int n_tokens, int d_model, int top_k,
                                 cudaStream_t stream = nullptr);

// FP32-input variant for diagnostic IMP_GEMMA4_FP32_EXPERT_DOWN.
// expert_output_fp32: [expanded, d_model] FP32 (down GEMM kept output in FP32).
void moe_scatter_fused_residual_fp32in(const void* expert_output_fp32,
                                        const int32_t* token_to_expanded,
                                        const float* expert_weights,
                                        const void* residual, void* output,
                                        int n_tokens, int d_model, int top_k,
                                        cudaStream_t stream = nullptr);

// Fused weighted sum + FP16 output + optional residual add.
// Combines expert weighted sum + residual add in one kernel. Only this fused
// variant is used by run_moe_ffn; the older non-residual `moe_weighted_sum`
// was removed 2026-04-20 (was declared but never called).
// expert_outputs: [top_k, d_model] FP16. expert_weights: [top_k] FP32 on device.
// residual: [d_model] FP16 (or nullptr to skip). output: [d_model] FP16.
void moe_weighted_sum_residual(const void* expert_outputs, const float* expert_weights,
                               const void* residual, void* output,
                               int d_model, int top_k,
                               cudaStream_t stream = nullptr);

// Fused gate GEMV + softmax/sigmoid + top-k selection in a single kernel.
// For n=1 decode: replaces gemv_gate_fp32 + moe_topk_gating(skip_sorting=true).
// W_gate: [n_experts, d_model] FP16. x: [d_model] FP16.
// Writes to buffers.expert_indices and buffers.expert_weights directly.
void moe_gate_topk_fused(const void* W_gate, const void* x,
                         int n_experts, int d_model, int top_k,
                         MoeRoutingBuffers& buffers,
                         MoeRoutingResult& result,
                         cudaStream_t stream = nullptr,
                         bool use_sigmoid = false,
                         bool normalize_weights = true,
                         const void* score_bias = nullptr);

} // namespace imp
