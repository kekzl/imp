#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Q8_0 INT8 IMMA prefill GEMM — fused dequant on int8 tensor cores.
//
// Sibling of the Q4_K IMMA stack (mmq_q4k_imma_tile.*), redesigned against
// the 2026-05-18 phase-2B ceiling diagnosis (40 TOPS plateau). The three
// structural fixes over that kernel:
//   1. Per-sub-block scales are cp.async-staged in SMEM with the data tiles —
//      the 2B kernel paid 8 GLOBAL loads per MMA for x_scale/α/β/rowsum.
//   2. BLOCK 128×128×64, 8 warps, 32×64 warp tile → 128 MMAs per CTA per
//      K-step with ONE __syncthreads pair (2B: 16 MMAs per 2 syncs).
//   3. Q8_0 is symmetric (w = d·q): the epilogue is a pure
//      (d_a·d_w)·s32 FMA — no β·rowsum correction term at all.
//
// Math:  out[m,n] = Σ_kb  d_a[m,kb] · d_w[n,kb] · Σ_{k∈kb} X_s8[m,k]·W_s8[n,k]
// (kb = 32-wide sub-block = one m16n8k32 IMMA per fragment; measured
// saturated s8.s8.s32 peak on this silicon: 968 TOPS — full rate.)
//
// Weight planes ([N][K] s8 + [N][K/32] half d) are split out of the raw GGUF
// Q8_0 block stream once per weight pointer and cached. Activation s8 planes
// are grow-only scratch. Both allocations are capture-guarded: a cache miss
// during CUDA-graph capture declines (returns false) instead of allocating —
// the warmup prefill populates the caches before capture.

// Attempt out[M,N] (FP16, row-major) = x[M,K]·W^T (beta=0) or += (beta=1)
// from the raw Q8_0 block stream (34-B blocks, [N][K/32]).
// Declines (returns false, output untouched) when: M < 64, N % 128 != 0,
// K % 64 != 0, beta ∉ {0,1}, capture-guarded cache miss, or allocation
// failure.
bool mmq_q8_imma_gemm(const void* w_q8_blocks, const __half* x_f16, __half* out_f16, int M, int N,
                      int K, cudaStream_t stream, float beta = 0.0f);

// Free cached weight planes + activation scratch (tests / teardown).
void mmq_q8_imma_release_all();

}  // namespace imp
