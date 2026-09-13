#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"               // NvFP4QuantResult, NvFP4MoEQuantResult
#include "compute/gemm_cutlass_sm120.h"      // CutlassNvFP4Weight
#include "compute/gemm_cutlass_mxfp4_sm120.h"// CutlassMxFP4Weight
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
    // True when `weight` borrows the checkpoint's own FP8 bytes rather than a cache this
    // pipeline allocated (Modelopt MIXED_PRECISION). Three consequences: the entry owns no
    // memory, so nothing here may be freed; it is a DECODE sidecar, never primary (sm_120
    // has no FP8 prefill GEMM, so phase 4 demotes prefill to the FP16 companion); it is
    // therefore not an alternative to that FP16 copy, and phase 3 must keep it.
    bool native_source = false;
};

// WeightCaches: all pre-quantized weight maps for the inference engine. Replaces the
// former WeightCacheManager type. All members public for zero-overhead forward-pass
// access. Lifecycle: allocated during pre_dequant_weights(), freed in free_buffers().
struct WeightCaches {
    // --- FP16 weight cache ---
    std::unordered_map<const void*, Tensor> fp16;
    size_t fp16_bytes = 0;

    // Bulk allocation for the MXFP4->FP16 decode fallback: every Tensor::data in fp16 is a
    // SUB-pointer into this one cudaMalloc'd buffer (cudaFree on a sub-pointer returns
    // invalid argument). On shutdown, range-check each fp16 entry against this region and
    // skip the per-tensor cudaFree; free the bulk pointer once via raw cudaFree.
    void* fp16_bulk_data = nullptr;
    size_t fp16_bulk_data_size = 0;

    // Fused KV: [wk; wv] per layer for strided batched prefill GEMM.
    std::unordered_map<int, Tensor> fused_kv;
    // Fused gate+up: [w_gate; w_up] per layer.
    std::unordered_map<int, Tensor> fused_gate_up;

    // --- FP8 E4M3 weight cache ---
    std::unordered_map<const void*, FP8CacheEntry> fp8;
    size_t fp8_bytes = 0;
    // The part of fp8_bytes the FP8 SSM sidecar contributes: the one FP8 cache allocated
    // through VRAMAllocator rather than raw cudaMallocAsync, so the WEIGHT_CACHE_FP8 note
    // must exclude it or the report double-counts it.
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

    // Single bulk allocation backing every cutlass_nvfp4 entry's SfAtom scale factors
    // (mirrors fp16_bulk_data). Each entry's scale_factors is a sub-pointer with
    // sf_borrowed=true, so the per-tensor cudaFree is skipped and this slab is freed once at
    // teardown, replacing thousands of per-tensor cudaMalloc+cudaMemsetAsync calls on large
    // MoE loads.
    void* cutlass_sf_slab = nullptr;
    size_t cutlass_sf_slab_size = 0;
    // Per-(layer, projection) SfAtom slabs built by the MoE phase. Their per-expert slices
    // become CutlassNvFP4Weight::scale_factors with sf_borrowed=true, so
    // free_cutlass_nvfp4_weight skips them and the BASE pointers must be owned here.
    // Allocated via vram_alloc_force (plain cudaMalloc), so release through VRAMAllocator,
    // NOT Model::gpu_allocations_ (which frees with cudaFreeAsync, #834).
    std::vector<void*> owned_sf_slabs;

    // --- CUTLASS sm_120 MXFP4 ---
    std::unordered_map<const void*, CutlassMxFP4Weight> cutlass_mxfp4;
    size_t cutlass_mxfp4_bytes = 0;
    bool use_mxfp4 = false;

    // Dual-path mode: FP8 attention + NVFP4 FFN
    bool dual_path_quant = false;
};

}  // namespace imp
