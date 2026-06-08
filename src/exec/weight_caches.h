#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"               // NvFP4QuantResult, NvFP4MoEQuantResult
#include "compute/gemm_cutlass_sm120.h"      // CutlassNvFP4Weight
#include "compute/gemm_cutlass_mxfp4_sm120.h"// CutlassMxFP4Weight
#include <cuda_fp16.h>
#include <unordered_map>
#include <cstddef>

namespace imp {

// ---------------------------------------------------------------------------
// FP8 weight cache entry (used by WeightCaches::fp8).
// ---------------------------------------------------------------------------
struct FP8CacheEntry {
    Tensor weight;     // [N, K] FP8_E4M3 on device
    float host_scale;  // absmax / 448
    float* d_scale;    // device-side scale (1 float)
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

    // --- CUTLASS sm_120 MXFP4 ---
    std::unordered_map<const void*, CutlassMxFP4Weight> cutlass_mxfp4;
    size_t cutlass_mxfp4_bytes = 0;
    bool use_mxfp4 = false;

    // --- Q4_K_M direct INT8 IMMA cache (Phase 2C infrastructure) ---
    // NOTE (2026-06-08): currently INACTIVE. The load-time gate (the former
    // gemm.q4k_imma_enabled flag) was removed as dead — no path populates this
    // map; it is only declared + freed. Was meant to be filled by
    // mmq_q4k_imma_reorder() and consumed by mmq_q4k_imma_tile(). Kept pending a
    // Phase-2C revival or a full removal of the q4k_imma_tile stack.
    // Three device buffers per entry:
    //   w_sym_s8 [N, K]      int8  symmetric-shifted (q - 8)
    //   eff_alpha [N, K/32]  FP16  d_super · sc[j]
    //   eff_beta  [N, K/32]  FP16  8·d_super·sc[j] − dmin_super·m[j]
    // Decode identity: α·q_sym + β  ≡  d·sc·q − dmin·m.
    //
    // The Phase 2C dispatcher (separate PR) gates entries on
    //   M ≥ 1024 && dense && Q4_K_M && !fp16_cache_hit
    // Off by default until E2E A/B against dense Q4_K_M models lands.
    struct Q4kImmaCacheEntry {
        int8_t* w_sym_s8 = nullptr;
        __half* eff_alpha = nullptr;
        __half* eff_beta = nullptr;
        int N = 0;
        int K = 0;
    };
    std::unordered_map<const void*, Q4kImmaCacheEntry> q4k_imma;
    size_t q4k_imma_bytes = 0;
    bool use_q4k_imma = false;

    // Dual-path mode: FP8 attention + NVFP4 FFN
    bool dual_path_quant = false;
};

}  // namespace imp
