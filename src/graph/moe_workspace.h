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
    Tensor gate_logits;    // [max_tokens, n_experts] FP32
    Tensor gathered;       // [max_tokens * top_k, d_model] compute_dtype
    Tensor expert_gate;    // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor expert_up;      // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor expert_swiglu;  // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor expert_down;    // [max_tokens * top_k, d_model] compute_dtype
    Tensor scatter_out;    // [max_tokens, d_model] FP32 (scatter output)

    // --- Separately allocated buffers ---

    // On-the-fly dequant scratch buffer for quantized expert weights (1 expert).
    void* dequant_buf = nullptr;
    size_t dequant_buf_size = 0;

    // Batch dequant buffer for MoE prefill: holds a chunk of experts' weights
    // dequanted to FP16 for L2-resident chunked processing.
    void* batch_dequant_buf = nullptr;
    size_t batch_dequant_buf_size = 0;

    // FP32 scratch for MoE prefill down-projection when fp32_down_active=true.
    // Sized for max_tokens × top_k × d_model × sizeof(float). Pre-allocated once
    // to eliminate per-call cudaMallocAsync at executor_forward_moe.cu:1080.
    void* fp32_down_buf = nullptr;
    size_t fp32_down_buf_size = 0;

    // Pre-allocated device pointer array for batched MoE GEMM.
    // Layout: [A_ptrs..., B_ptrs..., C_ptrs...] = 3 * n_experts void pointers.
    void** d_work_ptrs = nullptr;
    int d_work_ptrs_count = 0;

    // Per-expert FP8 scale buffer: [n_experts] floats on device.
    float* d_fp8_scales = nullptr;

    // Per-expert token-count buffer: [n_experts] int32 on device.
    // Populated by compute_M_per_from_offsets_device from routing.expert_offsets.
    // Replaces the host-side D2H + sync + loop pattern in MoE prefill dispatch
    // (executor_forward_moe.cu). Prerequisite for CUDA-graph capture of the
    // prefill path. See plan moe_prefill_graphs_plan_2026_05_10.
    int32_t* d_M_per = nullptr;
    int d_M_per_count = 0;

    // Device-side weight pointer array for device-grouped GEMM.
    void** d_weight_ptrs = nullptr;
    int d_weight_ptrs_count = 0;

    // GPU staging buffer for one expert's raw quantized bytes (H2D copy).
    void* raw_staging_buf = nullptr;
    size_t raw_staging_size = 0;

    // CUTLASS 3.x NVFP4 grouped GEMM staging:
    //   packed: [max_expanded, max_K/2] — contiguous FP4 activations
    //   sf:     per-expert SfAtom slabs (worst-case padded to 128 rows per expert)
    //   sfa_ptrs: device array of [ne] pointers into sf slab (fused-quantize kernel input)
    void* cutlass3x_packed = nullptr;
    size_t cutlass3x_packed_size = 0;
    void* cutlass3x_sf = nullptr;
    size_t cutlass3x_sf_size = 0;
    uint8_t** cutlass3x_sfa_ptrs = nullptr;  // device [ne] uint8_t* array
    int cutlass3x_sfa_ptrs_count = 0;

    // Free all separately allocated buffers (NOT the phase tensor views).
    void free(VRAMAllocator* alloc);
};

}  // namespace imp
