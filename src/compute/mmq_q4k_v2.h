#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// ---------------------------------------------------------------------------
// mmq_q4k v2 — HMMA Tensor-Core path for Q4_K @ FP16 GEMM.
//
// v1 (src/compute/mmq_q4k.cu) uses scalar dp4a (~50 TFLOPS peak) and is
// capped at M ≤ 16 in dispatch because FP16-TC cuBLAS wins above that.
// v2 routes the inner GEMM through mma.sync.m16n8k16.f16/f32 (~838 TFLOPS
// peak), with Q4 weights dequantized to FP16 in registers on the fly.
//
// Design memo: mmq_q4k_v2_hmma_design_2026_05_15. Blueprint pattern:
// NVIDIA/tilus examples/quantization/matmul_a16wx.py.
//
// Phase 1a (this file, shipped): precompute per-sub-block affine scales.
// Phase 1b: weight-layout permutation to MMA-fragment order.
// Phase 2:  HMMA kernel skeleton.
// Phase 3+: dequant-in-registers, tile tuning, integration.
// ---------------------------------------------------------------------------

// Precomputes per-sub-block affine constants for a Q4_K weight tensor.
//
// Q4_K stores weights as W[k] = d · sc[k/32] · q[k]  −  dmin · m[k/32], where
//   d, dmin    : super-block scale/min (FP16, one pair per 256-element block)
//   sc[i], m[i]: 6-bit packed sub-block scale + min (8 pairs per super-block)
//   q[k]       : 4-bit quantized value
//
// v2 hoists the affine constants out of the inner loop:
//   eff_scale[n, k/32] = d_n · sc_n[k/32]   (FP16)
//   eff_min  [n, k/32] = dmin_n · m_n[k/32] (FP16)
//
// Then the GEMM kernel's inner loop dequantizes Q4 → FP16 via a single
// __hfma2: b_fp16 = q_int · eff_scale − eff_min, and feeds FP16 to mma.sync.
//
// Cost: runs ONCE at model load. Output footprint: 2 × FP16 per sub-block per
// output channel — e.g. Qwen3-32B Q4_K_M layer (5120 × 5120): 5120 × 160 × 4 B
// = 3.3 MB per weight tensor. Trivial vs the 16 GB total weights.
//
// Constraints:
//   - K % 256 == 0
//   - eff_scale, eff_min must be pre-allocated with N * (K/32) halves each
//   - W layout: row-major block_q4_K[N][K/256] (canonical GGUF on-device order)

void q4k_precompute_eff_scales(
    const void* W,        // [N, K/256] block_q4_K bytes
    half* eff_scale_out,  // [N, K/32] FP16
    half* eff_min_out,    // [N, K/32] FP16
    int N, int K,
    cudaStream_t stream);

// Convenience: bytes needed for one eff_scale or eff_min tensor of shape [N, K/32].
inline size_t q4k_eff_scale_bytes(int N, int K) {
    return static_cast<size_t>(N) * (K / 32) * sizeof(half);
}

}  // namespace imp
