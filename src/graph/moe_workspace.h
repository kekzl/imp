#pragma once

#include "core/tensor.h"
#include "compute/moe_routing.h"
#include "memory/vram_allocator.h"
#include <cuda_runtime.h>

namespace imp {

// ---------------------------------------------------------------------------
// MoE workspace: holds all MoE-specific tensors and buffers.
//
// Phase tensors are views into the executor's shared_workspace_ (no separate
// allocation). Separately allocated buffers are owned and freed by this struct.
// All members are public for zero-overhead access in the forward pass.
// ---------------------------------------------------------------------------
struct MoEWorkspace {

    // --- Phase tensors (views into shared_workspace_, set by configure_moe_workspace) ---
    MoeRoutingBuffers routing_buffers;
    Tensor gate_logits;        // [max_tokens, n_experts] FP32
    Tensor gathered;           // [max_tokens * top_k, d_model] compute_dtype
    Tensor expert_gate;        // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor expert_up;          // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor expert_swiglu;      // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor expert_down;        // [max_tokens * top_k, d_model] compute_dtype
    Tensor scatter_out;        // [max_tokens, d_model] FP32 (scatter output)

    // --- Separately allocated buffers ---

    // On-the-fly dequant scratch buffer for quantized expert weights (1 expert).
    void* dequant_buf = nullptr;
    size_t dequant_buf_size = 0;

    // Batch dequant buffer for MoE prefill: holds a chunk of experts' weights
    // dequanted to FP16 for L2-resident chunked processing.
    void* batch_dequant_buf = nullptr;
    size_t batch_dequant_buf_size = 0;

    // Pre-allocated device pointer array for batched MoE GEMM.
    // Layout: [A_ptrs..., B_ptrs..., C_ptrs...] = 3 * n_experts void pointers.
    void** d_work_ptrs = nullptr;
    int d_work_ptrs_count = 0;

    // Per-expert FP8 scale buffer: [n_experts] floats on device.
    float* d_fp8_scales = nullptr;

    // Device-side weight pointer array for device-grouped GEMM.
    void** d_weight_ptrs = nullptr;
    int d_weight_ptrs_count = 0;

    // GPU staging buffer for one expert's raw quantized bytes (H2D copy).
    void* raw_staging_buf = nullptr;
    size_t raw_staging_size = 0;

    // Free all separately allocated buffers (NOT the phase tensor views).
    void free(VRAMAllocator* alloc);
};

} // namespace imp
