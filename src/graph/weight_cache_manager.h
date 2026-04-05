#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "memory/vram_allocator.h"
#include <cuda_runtime.h>
#include <unordered_map>

namespace imp {

// ---------------------------------------------------------------------------
// FP8 weight cache entry: pre-quantized FP8 E4M3 weight + per-tensor scale.
// Used for FP8×FP8 cuBLASLt GEMM (2x tensor core throughput on sm_120).
// ---------------------------------------------------------------------------
struct FP8CacheEntry {
    Tensor weight;       // [N, K] FP8_E4M3 on device
    float host_scale;    // absmax / 448
    float* d_scale;      // device-side scale (1 float, for gemm_cublaslt bScale)
};

// ---------------------------------------------------------------------------
// Manages all pre-quantized weight caches for the inference engine.
//
// Holds FP16, FP8, NVFP4, CUTLASS NVFP4, and MXFP4 weight caches that
// accelerate prefill and decode by avoiding per-layer dequantization.
// All members are public for zero-overhead access in the forward pass.
// ---------------------------------------------------------------------------
struct WeightCacheManager {

    // --- FP16 weight cache (Phase 1: dequant Q8/Q6/Q5 → FP16 for cuBLAS prefill) ---
    std::unordered_map<const void*, Tensor> fp16;
    size_t fp16_bytes = 0;

    // Fused KV: concatenated [wk; wv] as [2*nkv*hd, d_model] FP16 per layer.
    // Enables strided batched GEMM for K+V in a single cuBLAS call during prefill.
    std::unordered_map<int, Tensor> fused_kv;

    // Fused gate+up: concatenated [w_gate; w_up] as [2*d_ff, d_model] FP16 per layer.
    std::unordered_map<int, Tensor> fused_gate_up;

    // --- FP8 E4M3 weight cache (Phase 2: 50% of FP16, 2x tensor core throughput) ---
    std::unordered_map<const void*, FP8CacheEntry> fp8;
    size_t fp8_bytes = 0;
    bool use_fp8 = false;

    // Bulk-allocated buffers for FP16→FP8 migration (single cudaMalloc for all entries)
    float* fp8_migrated_scales = nullptr;
    int fp8_migrated_count = 0;
    void* fp8_migrated_data = nullptr;
    size_t fp8_migrated_data_size = 0;

    // Overflow FP8 cache (entries that didn't fit in the initial bulk allocation)
    float* fp8_overflow_scales = nullptr;
    int fp8_overflow_count = 0;
    void* fp8_overflow_data = nullptr;
    size_t fp8_overflow_data_size = 0;

    // --- NVFP4 decode weight cache (Phase 3: 31-47% bandwidth reduction for decode GEMV) ---
    // Mode: 0=off, 1=additive (FP16 + NVFP4), 2=only (NVFP4 replaces FP16)
    std::unordered_map<const void*, NvFP4QuantResult> nvfp4;
    size_t nvfp4_bytes = 0;
    int nvfp4_decode_mode = 0;

    // Per-expert NVFP4 quantization for MoE models
    std::unordered_map<const void*, NvFP4MoEQuantResult> nvfp4_moe;
    size_t nvfp4_moe_bytes = 0;

    // --- CUTLASS sm_120 block-scaled NVFP4 (native FP4 prefill GEMM) ---
    std::unordered_map<const void*, CutlassNvFP4Weight> cutlass_nvfp4;
    size_t cutlass_nvfp4_bytes = 0;

    // --- CUTLASS sm_120 MXFP4 (alternative to NVFP4 CUTLASS path) ---
    std::unordered_map<const void*, CutlassMxFP4Weight> cutlass_mxfp4;
    size_t cutlass_mxfp4_bytes = 0;
    bool use_mxfp4 = false;

    // --- Dual-path mode: FP8 attention + NVFP4 FFN ---
    // When true, attention weights (WQ/WK/WV/WO) are excluded from NVFP4 cache
    // and kept at FP8 for higher quality. Only FFN weights get NVFP4.
    bool dual_path_quant = false;

    // Free all cached weights and bulk buffers.
    void free(VRAMAllocator* alloc);
};

} // namespace imp
