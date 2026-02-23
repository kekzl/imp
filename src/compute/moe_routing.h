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

// Variant using pre-allocated buffers (no cudaMalloc inside)
void moe_topk_gating(const Tensor& gate_logits, int top_k,
                     MoeRoutingBuffers& buffers,
                     MoeRoutingResult& result,
                     cudaStream_t stream = nullptr,
                     bool use_sigmoid = false,
                     bool normalize_weights = true,
                     const void* score_bias = nullptr);

void moe_gather(const Tensor& input, const MoeRoutingResult& routing,
                Tensor& gathered, cudaStream_t stream = nullptr);

void moe_scatter(const Tensor& expert_output, const MoeRoutingResult& routing,
                 Tensor& output, cudaStream_t stream = nullptr);

} // namespace imp
