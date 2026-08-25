#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"               // NvFP4QuantResult, NvFP4MoEQuantResult
#include "compute/gemm_cutlass_sm120.h"      // CutlassNvFP4Weight
#include "compute/gemm_cutlass_mxfp4_sm120.h"// CutlassMxFP4Weight
#include "quant/marlin/marlin_w4a16.h"       // marlin_w4a16::MarlinWeight
#include <cuda_fp16.h>
#include <unordered_map>
#include <cstddef>
#include <vector>

namespace imp {

// ---------------------------------------------------------------------------
// FP8 weight cache entry (used by WeightCaches::fp8).
// ---------------------------------------------------------------------------
struct FP8CacheEntry {
    Tensor weight;     // [N, K] FP8_E4M3 on device
    float host_scale{};  // absmax / 448
    float* d_scale{};    // device-side scale (1 float)
    // Per-row (output-channel) scales — set by the fp8_ssm_proj sidecar
    // (points into fp8_ssm_sidecar_row_scales); null = per-tensor scale.
    const float* d_row_scales{};
    // True when `weight` borrows the checkpoint's own FP8 bytes rather than
    // pointing into a cache this pipeline allocated (Modelopt
    // MIXED_PRECISION — Nemotron-3.5). Three consequences, all load-bearing:
    //   - the entry owns no memory, so nothing here may be freed;
    //   - it is a DECODE sidecar, never the primary tier: sm_120 has no FP8
    //     prefill GEMM, so phase 4 demotes prefill to the FP16 companion;
    //   - it is therefore NOT an alternative to that FP16 copy, and phase 3
    //     must keep it — same reason native NVFP4 keeps its own.
    bool native_source = false;
};

// ---------------------------------------------------------------------------
// WeightCaches: all pre-quantized weight maps for the inference engine.
//
// Replaces the former WeightCacheManager type (Phase 5 cleanup).
// All members are public for zero-overhead access in the forward pass.
// Lifecycle: allocated during pre_dequant_weights(), freed in free_buffers().
// ---------------------------------------------------------------------------
struct WeightCaches {
    // --- FP16 weight cache ---
    std::unordered_map<const void*, Tensor> fp16;
    size_t fp16_bytes = 0;

    // Bulk allocation used by the MXFP4 → FP16 decode fallback. When set,
    // every Tensor::data in `fp16` is a SUB-pointer (offset) into this single
    // cudaMalloc'd buffer — cudaFree on the sub-pointers returns
    // "invalid argument". On shutdown, range-check each fp16 entry against
    // this region (analogous to fp8_migrated_data) and skip the per-tensor
    // cudaFree; the bulk pointer is freed once via raw cudaFree.
    void* fp16_bulk_data = nullptr;
    size_t fp16_bulk_data_size = 0;

    // Fused KV: [wk; wv] per layer for strided batched prefill GEMM.
    std::unordered_map<int, Tensor> fused_kv;
    // Fused gate+up: [w_gate; w_up] per layer.
    std::unordered_map<int, Tensor> fused_gate_up;

    // --- FP8 E4M3 weight cache ---
    std::unordered_map<const void*, FP8CacheEntry> fp8;
    size_t fp8_bytes = 0;
    // The part of fp8_bytes that the FP8 SSM sidecar contributes. It is the one
    // FP8 cache allocated through VRAMAllocator rather than raw
    // cudaMallocAsync, so since the allocator started naming its own charges
    // the WEIGHT_CACHE_FP8 note has to exclude it or the report counts it twice
    // (measured: 963.8 MiB double-counted on Qwen3.6-35B-A3B-NVFP4).
    size_t fp8_sidecar_bytes = 0;
    bool use_fp8 = false;

    // Bulk-allocated buffers for FP16→FP8 migration
    float* fp8_migrated_scales = nullptr;
    int fp8_migrated_count = 0;
    void* fp8_migrated_data = nullptr;
    size_t fp8_migrated_data_size = 0;

    // Overflow FP8 cache
    float* fp8_overflow_scales = nullptr;
    int fp8_overflow_count = 0;
    void* fp8_overflow_data = nullptr;
    size_t fp8_overflow_data_size = 0;

    // FP8 decode sidecar for GDN/SSM projections (gemm.fp8_ssm_proj); entries
    // carry per-row scales (d_row_scales into the bulk below, d_scale = null).
    void* fp8_ssm_sidecar_data = nullptr;
    size_t fp8_ssm_sidecar_data_size = 0;
    float* fp8_ssm_sidecar_row_scales = nullptr;

    // --- NVFP4 decode weight cache ---
    // Mode: 0=off, 1=additive, 2=only
    std::unordered_map<const void*, NvFP4QuantResult> nvfp4;
    size_t nvfp4_bytes = 0;
    int nvfp4_decode_mode = 0;

    // Per-expert NVFP4
    std::unordered_map<const void*, NvFP4MoEQuantResult> nvfp4_moe;
    size_t nvfp4_moe_bytes = 0;

    // --- CUTLASS sm_120 block-scaled NVFP4 ---
    std::unordered_map<const void*, CutlassNvFP4Weight> cutlass_nvfp4;
    size_t cutlass_nvfp4_bytes = 0;

    // --- Marlin W4A16 batched-decode sidecar (gemm.marlin) ---
    // Repacked copy of the 4-bit data + processed scales per dense weight,
    // budget-aware (a full copy of every linear is ~12.8 GiB on the 27B, so
    // coverage is whatever VRAM affords, largest weights first). Entries own
    // their buffers; freed via marlin_w4a16::release() at teardown.
    std::unordered_map<const void*, marlin_w4a16::MarlinWeight> marlin;
    size_t marlin_bytes = 0;

    // Single bulk allocation backing every cutlass_nvfp4 entry's SfAtom scale
    // factors (mirrors fp16_bulk_data). Each entry's scale_factors is a
    // sub-pointer with sf_borrowed=true, so the per-tensor cudaFree is skipped
    // and this slab is freed once at teardown. Replaces ~18k per-tensor
    // cudaMalloc+cudaMemsetAsync on large MoE loads (~600 ms of load time).
    void* cutlass_sf_slab = nullptr;
    size_t cutlass_sf_slab_size = 0;
    // Per-(layer, projection) SfAtom slabs built by the MoE phase. Their
    // per-expert slices become CutlassNvFP4Weight::scale_factors with
    // sf_borrowed=true, so free_cutlass_nvfp4_weight deliberately skips them
    // and the BASE pointers have to be owned here — otherwise nothing frees
    // them at all. They come from vram_alloc_force (plain cudaMalloc), so they
    // must be released through VRAMAllocator, NOT through
    // Model::gpu_allocations_, which frees with cudaFreeAsync (#834).
    std::vector<void*> owned_sf_slabs;

    // --- CUTLASS sm_120 MXFP4 ---
    std::unordered_map<const void*, CutlassMxFP4Weight> cutlass_mxfp4;
    size_t cutlass_mxfp4_bytes = 0;
    bool use_mxfp4 = false;

    // Dual-path mode: FP8 attention + NVFP4 FFN
    bool dual_path_quant = false;
};

}  // namespace imp
