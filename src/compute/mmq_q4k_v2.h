#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// mmq_q4k v2 — HMMA Tensor-Core path for Q4_K @ FP16 GEMM.
//
// Routes the inner GEMM through mma.sync.m16n8k16.f16/f32 (~838 TFLOPS peak),
// with Q4 weights dequantized to FP16 in registers on the fly.
//
// Frozen as opt-in via IMP_FORCE_Q4K_V2=1: end-to-end on Q4_K_M models, the
// FP16-cache + cuBLAS-FP16-TC baseline wins by 4-6 % because the FP16 cache
// absorbs the hot weights. v2 stays available for VRAM-constrained scenarios
// where the FP16 cache can't fit.
//
// Design memo: mmq_q4k_v2_hmma_design_2026_05_15. Blueprint pattern:
// NVIDIA/tilus examples/quantization/matmul_a16wx.py.
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

// Permute Q4_K nibbles into the v2 consumer layout.
//
// Canonical Q4_K stores qs[128] with 2 sub-blocks sharing 32 bytes:
//   byte i in qs[32*g..32*g+32] holds sub-block (2*g)'s nibble at low 4 bits
//   and sub-block (2*g+1)'s nibble at high 4 bits.
//
// The v2 kernel wants each sub-block's 32 nibbles packed contiguously and
// K-major: 16 bytes per sub-block, byte j = (nibble for K=2j) | (nibble for
// K=2j+1) << 4. Loading is then a single coalesced `int4` per sub-block per
// output row, with 4 consecutive K values held in 16 bits per thread —
// matching the m16n8k16 B-fragment layout (4 K-values per thread per N col).
//
// Output: eff_q4[N, K/32, 16] uint8 — row-major, 16 bytes per sub-block.
// Total footprint = N * K / 2 bytes (same nibble count as canonical qs[]).
void q4k_permute_to_v2_layout(
    const void* W,        // [N, K/256] block_q4_K bytes
    uint8_t* eff_q4_out,  // [N, K/32, 16] uint8
    int N, int K,
    cudaStream_t stream);

// Convenience: bytes needed for one eff_q4 tensor (N rows × K cols).
inline size_t q4k_eff_q4_bytes(int N, int K) {
    return static_cast<size_t>(N) * (K / 32) * 16u;
}

// ---------------------------------------------------------------------------
// HMMA GEMM kernel (mma.sync.m16n8k16.f16/f32 via WMMA).
//
// Computes y = x @ W^T where W is Q4_K-quantized weights, given the
// precomputed v2 inputs (eff_scale, eff_min) + permuted eff_q4 nibbles.
// Activations stay in FP16; weights are dequantized in registers on the
// fly with a 2-stage cp.async pipeline.
//
// Constraints:
//   - K % 32 == 0
//   - eff_q4 / eff_scale / eff_min must come from the same W with same N, K
//   - M and N can be arbitrary; out-of-bounds rows/cols are masked.
void mmq_q4k_v2(
    const half* x,            // [M, K] FP16 activations (row-major)
    const uint8_t* eff_q4,    // [N, K/32, 16] permuted Q4 (Phase 1b output)
    const half* eff_scale,    // [N, K/32] (Phase 1a output)
    const half* eff_min,      // [N, K/32] (Phase 1a output)
    half* y,                  // [M, N] FP16 output (row-major)
    int M, int N, int K,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Phase 6: Q5_K v2 path
//
// Q5_K is Q4_K + a 1-bit "high" overlay per quant. Block layout:
//   d (FP16) | dmin (FP16) | scales[12] (6-bit packed sc + m, SAME as Q4_K)
//                          | qh[32] (high bits, 1 per quant, byte b covers
//                            K=8b..8b+7) | qs[128] (low 4 bits, SAME as Q4_K)
// Total 176 bytes per 256 elements.
//
// Per-quant dequant: q5 = qs.nibble | (qh.bit << 4)  in [0, 31]
//                    w  = q5 · eff_scale - eff_min
// where eff_scale = d · sc[s] and eff_min = dmin · m[s] — IDENTICAL to Q4_K.
//
// Phase 1a (eff_scale + eff_min) is shared with Q4_K, just with a different
// block_stride argument. Phase 1b for Q5_K writes a packed eff_q5 layout:
//   per sub-block (32 elements) → 16 bytes of permuted nibbles (SAME packing
//   as eff_q4) + 4 bytes of qh (sub-block bytes, K-major).
// Tensor shape: [N, K/32, 20] uint8. Total bytes = N·K·20/32 = 0.625·N·K.

void q5k_precompute_eff_scales(
    const void* W,        // [N, K/256] block_q5_K bytes
    half* eff_scale_out,  // [N, K/32]
    half* eff_min_out,    // [N, K/32]
    int N, int K,
    cudaStream_t stream);

void q5k_permute_to_v2_layout(
    const void* W,        // [N, K/256] block_q5_K bytes
    uint8_t* eff_q5_ql_out,  // [N, K/32, 16] permuted low nibbles (Q4_K-style)
    uint8_t* eff_q5_qh_out,  // [N, K/32, 4]  high bits, byte b covers K=8b..8b+7
    int N, int K,
    cudaStream_t stream);

inline size_t q5k_eff_ql_bytes(int N, int K) {
    return static_cast<size_t>(N) * (K / 32) * 16u;
}
inline size_t q5k_eff_qh_bytes(int N, int K) {
    return static_cast<size_t>(N) * (K / 32) * 4u;
}

void mmq_q5k_v2(
    const half* x,
    const uint8_t* eff_q5_ql,
    const uint8_t* eff_q5_qh,
    const half* eff_scale,
    const half* eff_min,
    half* y,
    int M, int N, int K,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Phase 6b: Q6_K v2 path
//
// Q6_K block layout (210 bytes / 256 elements):
//   ql[128]   : low 4 bits (GGML interleaved layout)
//   qh[64]    : high 2 bits (GGML interleaved layout)
//   scales[16]: signed int8, one per 16-element sub-block
//   d (FP16)  : super-block scale
//
// Unlike Q4_K/Q5_K, sub-blocks are **16 elements** (not 32) and the dequant
// formula uses signed quants centered at zero:
//   q_unsigned = ql.nibble | (qh.bits << 4)   in [0, 63]
//   w = d · scales[s] · (q_unsigned - 32)
// Factorizing:
//   eff_scale = d · scales[s]   (FP16, signed via int8 scale)
//   eff_min   = 32 · eff_scale
//   w = eff_scale · q_unsigned - eff_min
// → matches Q4_K/Q5_K formula, so the same Phase 3 MMA core applies. Only
//   differences in this kernel: BK=16 (one sub-block per K-iter), WRK=1
//   (one m16n8k16 per K-step), and we expand the Q6_K bytes into a clean
//   1-byte-per-element layout at Phase 1 time to avoid the GGML interleaved
//   ql/qh bit dance in the hot loop.

void q6k_prepare_v2_layout(
    const void* W,            // [N, K/256] block_q6_K bytes
    uint8_t* eff_q6_out,      // [N, K] uint8, each byte = q_unsigned ∈ [0, 63]
    half* eff_scale_out,      // [N, K/16] (d · scales[s])
    half* eff_min_out,        // [N, K/16] (32 · eff_scale)
    int N, int K,
    cudaStream_t stream);

inline size_t q6k_eff_q6_bytes(int N, int K) {
    return static_cast<size_t>(N) * K;
}
inline size_t q6k_eff_scale_bytes(int N, int K) {
    return static_cast<size_t>(N) * (K / 16) * sizeof(half);
}

void mmq_q6k_v2(
    const half* x,
    const uint8_t* eff_q6,
    const half* eff_scale,
    const half* eff_min,
    half* y,
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace imp
