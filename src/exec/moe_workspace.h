#pragma once

#include <vector>

#include "core/tensor.h"
#include "compute/moe_routing.h"
#include "memory/vram_allocator.h"
#include "memory/host_pinned.h"
#include <cuda_runtime.h>

namespace imp {

class Model;

// Which prefill path serves every on-device expert without batch_dequant_buf.
// cutlass_fp4 (name historical): NVFP4/MXFP4 experts (grouped NVFP4 GEMM); imma_all: K-quant experts
// the MoE IMMA kernel takes (Q8_0 excluded: its planes can hit the plane budget at runtime).
// f16_all: every packed expert tensor is device F16 (nothing to dequant; Ling-3.0 BF16: 192 MiB).
struct MoeExpertServe {
    bool cutlass_fp4 = false;
    bool imma_all = false;
    bool f16_all = false;
};
[[nodiscard]] MoeExpertServe classify_moe_experts(const Model& model, bool moe_imma_prefill,
                                                  QType compute_dtype);

// MoE workspace: holds all MoE-specific tensors and buffers. Phase tensors are views into
// the executor's shared_workspace_ (no separate allocation); separately allocated buffers
// are owned/freed by this struct. All members public for zero-overhead access.
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
    void* router_ws = nullptr;  // fp32 split-K partials of the router GEMM (gemm_f16_splitk)
    size_t router_ws_bytes = 0;
    void* router_fused_ws = nullptr;  // moe_router_fused: split-K tickets (zeroed once) + fp32 partials
    size_t router_fused_ws_bytes = 0;

    // On-the-fly dequant scratch buffer for quantized expert weights (1 expert).
    void* dequant_buf = nullptr;
    size_t dequant_buf_size = 0;

    // Batch dequant buffer for MoE prefill: holds a chunk of experts' weights
    // dequanted to FP16 for L2-resident chunked processing.
    void* batch_dequant_buf = nullptr;
    size_t batch_dequant_buf_size = 0;
    // Every on-device expert tensor fits the MoE IMMA kernel: the FP16 batch path runs without
    // batch_dequant_buf, a runtime decline dequants into a stream-ordered temporary.
    bool imma_serves_all = false;
    [[nodiscard]] bool fp16_batch_buffer_ok(size_t need) const {
        return (batch_dequant_buf != nullptr && batch_dequant_buf_size >= need) || imma_serves_all;
    }

    // Expert-activation histogram (diagnostics.moe_expert_hist), off unless the key names a
    // path. [n_layers*n_experts] device counters, incremented once per (token,k) routing
    // decision. Answers how skewed expert selection is, deciding whether a resident/host
    // split of MoE experts can pay (docs/ROADMAP.md).
    unsigned int* expert_hist = nullptr;
    int hist_layers = 0;
    int hist_experts = 0;
    int hist_top_k = 0;

    // Per-launch expert imbalance (#1548): max(M_e) decides grouped-GEMM cost (one hot expert
    // sets the tile for all, the rest is padding); a whole-process histogram averages this
    // away. Device-side because the NVFP4 prefill path keeps expert args on-card for graph
    // capture; reading them back would trade capture for a diagnostic. Four counters per
    // layer: [peak_max, sum_max, sum_rows, launches], recorded only at n>1 (decode pays nothing).
    unsigned int* imb_acc = nullptr;
    int imb_layers = 0;
    int imb_experts = 0;

    // Per-token expert trace (diagnostics.moe_expert_trace): flat append of
    // [layer, e0..e_{top_k-1}], one per (token, layer), in stream order. Decode only: a
    // prefill call would append n*top_k at once and break the fixed record stride.
    int* expert_trace = nullptr;
    float* trace_weights = nullptr;        // [record * top_k + j], tail of the expert_trace allocation
    unsigned int* trace_cursor = nullptr;  // device counter, in ints
    size_t trace_capacity = 0;             // ints
    int trace_top_k = 0;

    // Pre-allocated device pointer array for batched MoE GEMM.
    // Layout: [A_ptrs..., B_ptrs..., C_ptrs...] = 3 * n_experts void pointers.
    void** d_work_ptrs = nullptr;
    int d_work_ptrs_count = 0;

    // Per-expert FP8 scale buffer: [n_experts] floats on device.
    float* d_fp8_scales = nullptr;

    // Per-expert token-count buffer [n_experts] int32, device. Populated by
    // compute_M_per_from_offsets_device from routing.expert_offsets. Replaces the host-side
    // D2H+sync+loop pattern; prerequisite for CUDA-graph capture of the prefill path.
    int32_t* d_M_per = nullptr;
    int d_M_per_count = 0;

    // Compact-alpha output buffer: [n_experts] floats on device. Populated by
    // compact_alpha_active and consumed by the grouped GEMM dispatch when not
    // all experts are active. First d_na valid entries; rest unused.
    float* d_alpha_compact = nullptr;
    // Active-expert count: [1] int32 on device. Written by compact_alpha_active.
    int32_t* d_na = nullptr;

    // Per-expert SFA byte-offsets into the SfAtom-padded grouped-GEMM staging buffer.
    // Exclusive prefix sum of cutlass_nvfp4_sf_size(M_per[e], K), populated each forward by
    // compute_sfa_offsets_device. [n_experts+1] int64.
    int64_t* d_sfa_offsets = nullptr;

    // Phase 3c-full device-args caches: d_B_ptrs_cache/d_SFB_ptrs_cache/d_alpha_full are
    // [n_experts] device arrays filled per-call from the registry handles via
    // cudaMemcpyAsync, shared across gate/up/down within one forward layer. Replaces the
    // per-call cudaMallocAsync of the MVP wire. Superseded by per_layer_da_cache when
    // da_cache_ready; kept for any future per-call fallback path.
    const void** d_B_ptrs_cache   = nullptr;
    const void** d_SFB_ptrs_cache = nullptr;
    float*       d_alpha_full     = nullptr;

    // Per-layer pre-cached device-args ptr arrays, built once at model-load time
    // (pre_dequant_weights) when handle payloads are populated; reused on every forward call
    // with no host iteration and no per-call H2D. Prerequisite for CUDA-graph capture of the
    // MoE prefill.
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

    // Whole-layer staging for host-resident NVFP4 experts during PREFILL. Per-expert route
    // issues small H2Ds per expert per projection, which don't reach PCIe bandwidth; staging
    // a whole projection moves the same bytes as ONE transfer (moe.pin_host_experts' pinned
    // slabs, or the mmap, already lay a projection's experts back to back).
    // Sized for one layer, reused across layers (forward is sequential). Null when experts
    // are device-resident, model is not NVFP4-prequant, or one layer doesn't fit the budget.
    void* layer_stage_buf = nullptr;      // [layer_stage_slots][layer_stage_experts * expert_bytes]
    size_t layer_stage_proj_bytes = 0;    // per-projection span within the buffer
    size_t layer_stage_size = 0;          // total allocation
    int layer_stage_experts = 0;          // experts per projection slot (a block when chunked)
    int layer_stage_slots = 0;            // projection slots: 3 whole-layer, 2 chunked (gate+up, then down)
    int layer_stage_ptr_stride = 0;       // n_experts: per-projection span of the pointer arrays

    // Device-args view of the staged layer: grouped NVFP4 GEMM wants SfAtom-ordered
    // scale factors and device arrays of per-expert pointers; staged bytes carry the native
    // per-16 layout, converted once per staged layer into layer_stage_sf with rebuilt pointer
    // arrays. Lets a host-resident layer take the same prefill path as a resident one.
    void* layer_stage_sf = nullptr;        // [layer_stage_slots][layer_stage_experts * sf_size]
    size_t layer_stage_sf_proj_bytes = 0;  // per-projection span
    size_t layer_stage_sf_size = 0;
    const void** layer_stage_b_ptrs = nullptr;    // [kExpertProjCount * n_experts]
    const void** layer_stage_sfb_ptrs = nullptr;  // [kExpertProjCount * n_experts]
    float* layer_stage_alpha = nullptr;           // [n_layers][kExpertProjCount * n_experts]
    // Pointer arrays and alpha rows depend only on the staging layout and the load-time tensor
    // scales: uploaded once (pageable H2D per block cost ~1800 copies per Flash-Next prefill).
    std::vector<uint8_t> layer_stage_alpha_filled;  // [n_layers]
    struct StagePtrSig {
        const void* packed = nullptr;
        const void* sf = nullptr;
        size_t pb = 0, sfe = 0;
        int n = 0;
        bool operator==(const StagePtrSig&) const = default;
    };
    std::vector<StagePtrSig> layer_stage_ptr_sig;  // [kExpertProjCount * n_experts], keyed by first expert

    // Slot indices for the host-offload decode path: [3 * kDevExpertMaxRows * top_k] int32, one block per
    // projection (gate, up, down). Fused MoE decode kernels address an expert as
    // base + idx*stride; idx is the expert's LRU-cache slot, not its id.
    int32_t* d_slot_idx = nullptr;
    int d_slot_idx_count = 0;
    // Per-entry miss flags ([kDevExpertMaxRows * top_k] int32) behind the slot indices:
    // the hit GEMVs skip them.
    [[nodiscard]] int32_t* entry_miss() const { return d_slot_idx ? d_slot_idx + d_slot_idx_count : nullptr; }
    // Pinned landing zone for the per-layer routing readback (top_k int32). A D2H into a
    // std::vector is a staged pageable copy: 245 us of copy-engine time per layer on WSL2.
    PinnedBuffer h_routing_readback;

    // Grouped NVFP4 GEMM staging (cutlass3x_* names historical):
    //   packed: [max_expanded, max_K/2] contiguous FP4 activations
    //   sf: per-expert SfAtom slabs, worst-case padded to 128 rows per expert
    //   sfa_ptrs: device array of [ne] pointers into sf slab (fused-quantize kernel input)
    void* cutlass3x_packed = nullptr;
    size_t cutlass3x_packed_size = 0;
    void* cutlass3x_sf = nullptr;
    size_t cutlass3x_sf_size = 0;
    uint8_t** cutlass3x_sfa_ptrs = nullptr;  // device [ne] uint8_t* array
    int cutlass3x_sfa_ptrs_count = 0;

    // smallM NVFP4 prefill scratch (#2446), T2 arena, null = smallM branch off:
    //   scales [3*ne] f32: act tensor scales | alpha | compacted alpha; ptrs [2*ne]: packed | sf
    float* smallM_scales = nullptr;
    void** smallM_ptrs = nullptr;
    int smallM_count = 0;
    void* smallM_tables = nullptr;  // smallM_table_bytes(ne): per-call pointer/M/descriptor tables (#2451)
    size_t smallM_tables_bytes = 0;

    // Free all separately allocated buffers (NOT the phase tensor views).
    void free(VRAMAllocator* alloc);
};

}  // namespace imp
