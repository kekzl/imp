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

    // Expert-activation histogram (diagnostics.moe_expert_hist), off unless the
    // key names a path. [n_layers * n_experts] device counters, incremented once
    // per (token, k) routing decision. Answers "how skewed is expert selection",
    // which is what decides whether a resident/host split of MoE experts can pay
    // (docs/roadmap.md, "CPU-resident cold experts").
    unsigned int* expert_hist = nullptr;
    int hist_layers = 0;
    int hist_experts = 0;
    int hist_top_k = 0;

    // Per-token expert trace (diagnostics.moe_expert_trace). Flat append of
    // records [layer, e0..e_{top_k-1}], one per (token, layer), in stream order.
    // Decode only — a prefill call would append n*top_k at once and break the
    // fixed record stride the reader relies on.
    int* expert_trace = nullptr;
    unsigned int* trace_cursor = nullptr;  // device counter, in ints
    size_t trace_capacity = 0;             // ints
    int trace_top_k = 0;

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

    // Compact-alpha output buffer: [n_experts] floats on device. Populated by
    // compact_alpha_active and consumed by the grouped GEMM dispatch when not
    // all experts are active. First d_na valid entries; rest unused.
    float* d_alpha_compact = nullptr;
    // Active-expert count: [1] int32 on device. Written by compact_alpha_active.
    int32_t* d_na = nullptr;

    // Per-expert SFA byte-offsets into the CUTLASS 3.x SfAtom-padded staging
    // buffer (Phase 3 of moe_prefill_graphs_plan_2026_05_10). Exclusive prefix
    // sum of cutlass_nvfp4_sf_size(M_per[e], K) — populated each forward by
    // compute_sfa_offsets_device. Replaces the host-side sfa_offsets loop in
    // executor_forward_moe.cu's quantize_once lambda. [n_experts+1] int64.
    int64_t* d_sfa_offsets = nullptr;

    // Phase 3c-full Step 1 caches for the device-args wrapper:
    // - d_B_ptrs_cache:   [n_experts] device array of per-expert weight pointers
    // - d_SFB_ptrs_cache: [n_experts] device array of per-expert SFB pointers
    // - d_alpha_full:     [n_experts] device floats per-expert alpha
    // Filled per-call from the registry handles (host) via cudaMemcpyAsync;
    // shared across all three projections (gate / up / down) inside one
    // forward layer. Replaces the per-call cudaMallocAsync of the MVP wire.
    // Superseded by per_layer_da_cache below when da_cache_ready is true —
    // these stay around for any future per-call fallback path.
    const void** d_B_ptrs_cache   = nullptr;
    const void** d_SFB_ptrs_cache = nullptr;
    float*       d_alpha_full     = nullptr;

    // Phase 3c-full Step 3: per-layer pre-cached device-args ptr arrays.
    // Built once at model-load time (pre_dequant_weights) when handle payloads
    // are populated; reused on every forward call with no host iteration and
    // no per-call H2D. Prerequisite for CUDA-graph capture of the MoE prefill.
    struct PerLayerNvfp4DeviceArgsCache {
        const void** d_gate_B_ptrs   = nullptr;
        const void** d_gate_SFB_ptrs = nullptr;
        float*       d_gate_alpha    = nullptr;
        const void** d_up_B_ptrs     = nullptr;
        const void** d_up_SFB_ptrs   = nullptr;
        float*       d_up_alpha      = nullptr;
        const void** d_down_B_ptrs   = nullptr;
        const void** d_down_SFB_ptrs = nullptr;
        float*       d_down_alpha    = nullptr;
        // True if all 9 buffers above are non-null AND populated. Tested in
        // the device-args dispatch to gate whether the pre-cache fast-path
        // can fire (otherwise fall back to per-call H2D into d_B_ptrs_cache).
        bool ready = false;
    };
    std::vector<PerLayerNvfp4DeviceArgsCache> per_layer_da_cache;

    // Device-side weight pointer array for device-grouped GEMM.
    void** d_weight_ptrs = nullptr;
    int d_weight_ptrs_count = 0;

    // GPU staging buffer for one expert's raw quantized bytes (H2D copy).
    void* raw_staging_buf = nullptr;
    size_t raw_staging_size = 0;

    // Whole-layer staging for host-resident NVFP4 experts during PREFILL.
    //
    // The per-expert route issues two H2D per expert per projection — 18 432
    // transfers of ~768 KiB + ~96 KiB for one pass over this model — and small
    // transfers do not reach PCIe bandwidth. Staging a whole projection at once
    // moves the same bytes as ONE transfer, because the pinned slabs from
    // `moe.pin_host_experts` already lay a projection's experts back to back
    // (and the mmap usually does too, which is checked at runtime).
    //
    // Sized for one layer only and reused across layers: the forward is
    // sequential, so layer L+1 overwrites layer L after its kernels have run
    // on the same stream. Null when the experts are device-resident, when the
    // model is not NVFP4-prequant, or when one layer does not fit the budget.
    void* layer_stage_buf = nullptr;      // [kExpertProjCount][n_experts * expert_bytes]
    size_t layer_stage_proj_bytes = 0;    // per-projection span within the buffer
    size_t layer_stage_size = 0;          // total allocation
    int layer_stage_experts = 0;          // n_experts the buffer was sized for

    // Slot indices for the host-offload decode path: [3 * top_k] int32, one
    // block per projection (gate, up, down). The fused MoE decode kernels
    // address an expert as `base + idx * stride`; with host-resident experts
    // the contiguous array is the LRU cache's per-layer slot pool, so `idx` is
    // the expert's slot rather than its id. See docs/roadmap.md.
    int32_t* d_slot_idx = nullptr;
    int d_slot_idx_count = 0;

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
