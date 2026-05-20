#pragma once
// =============================================================================
// tiled_attention_ceiling_bench.h — Säule 3 of Track E gating bench
// =============================================================================
//
// Microbench for the inner-loop ceiling of a hand-written tiled FA2-style
// attention kernel on sm_120a. Measures the maximum throughput achievable
// with the full set of HW features (cp.async double-buffer + ldmatrix.sync
// + mma.sync.m16n8k16 + warp-reduce online softmax + stmatrix.sync).
//
// Output: tile_us per (Br × Bkv) inner iteration, effective TFLOPS.
//
// This is the UPPER BOUND for what Track E's tiled streaming softmax could
// achieve at hd=128 FP16. If this ceiling is ≤ 1.5× current FMHA throughput,
// Track E has no runway. If ≥ 2× FMHA AND > cuBLAS, Track E is worth ~10-15
// dev days.
//
// Scope: FP16 KV only (covers Qwen3/Llama dense + GQA, the production majority).
// FP8/NVFP4 KV-inner-loop ceiling is implicit from mxf4nvf4_mma_bench (already
// measured — 268 TOPS for NVFP4 block-scale, see sm120_mma_variants_2026_04_25).
// =============================================================================

namespace imp {

struct TiledAttnCeilingResult {
    // Time per inner iteration (one full Br×Bkv tile: K+V load → QKᵀ → softmax → PV).
    double tile_ns;
    // Effective TFLOPS over the (Br × Bkv × HD × 4) ops/tile (QKᵀ 2× + PV 2×).
    double effective_tflops;
    // GB/s of K+V tile bandwidth (excludes Q which is loaded once).
    double kv_bandwidth_gb_per_s;
};

// Bench: launch a kernel that loops `iters` × (load K/V tile + QKᵀ + softmax + PV)
// on one CTA per SM, 128 threads per CTA. Fixed Br=64, Bkv=64, HD=128 FP16.
//
// Returns false on kernel-launch failure. Out param is left zero in that case.
bool tiled_attention_ceiling_bench(TiledAttnCeilingResult* out);

}  // namespace imp
