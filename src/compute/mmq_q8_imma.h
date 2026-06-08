#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// INT8 IMMA prefill GEMM family (sm_120a) — fused dequant on int8 tensor
// cores (s8.s8.s32 mma.sync, measured 968 TOPS saturated — full rate).
//
// Weight forms:
//   Q8_0 — native s8, SoA-split once per tensor (qs [N][K] + (α=d, β=0)
//          interleaved scale plane [N][K/32][2]).
//   Q4_K — reordered once via mmq_q4k_imma_reorder (symmetric s8 + per-sub
//          α/β, see mmq_q4k_imma_layout.h), interleaved the same way.
//
// Unified epilogue per 32-wide sub-block kb:
//   out[m,n] += d_a[m,kb] · ( α[n,kb] · Σ s8·s8 + β[n,kb] · rowsum_a[m,kb] )
//
// Activations are s8-quantized per 32-block (half scale + float rowsum),
// memoized per (pointer, M, K) so MoE gate/up GEMMs sharing one gathered
// batch quantize once. All caches are capture-guarded: a miss during CUDA-
// graph capture declines instead of allocating (warmup populates them).

// Dense Q8_0: out[M,N] = x·W^T (beta=0) or += (beta=1).
// Declines: M < 64, N % 128, K % 64, beta ∉ {0,1}, capture-guarded miss.
bool mmq_q8_imma_gemm(const void* w_q8_blocks, const __half* x_f16, __half* out_f16, int M, int N,
                      int K, cudaStream_t stream, float beta = 0.0f);

// Dense Q4_K (new stack — distinct from the retired 2026-05 64x32 q4k_imma
// kernel): same contract; K % 256 == 0 (Q4_K super-block).
bool mmq_q4k_imma_gemm(const void* w_q4k_blocks, const __half* x_f16, __half* out_f16, int M,
                       int N, int K, cudaStream_t stream, float beta = 0.0f);

// Dense Q6_K: per-16 scales via half-MMA split (symmetric, no beta term);
// one-time 224-B-aligned repack (+6.7% of the Q6_K bytes — the 210-B blocks
// are only 2-aligned, forge 2026-05-28 finding). K % 256 == 0.
bool mmq_q6k_imma_gemm(const void* w_q6k_blocks, const __half* x_f16, __half* out_f16, int M,
                       int N, int K, cudaStream_t stream, float beta = 0.0f);

// MoE grouped prefill GEMM over ne experts in ONE launch (gridDim.z = ne).
//   w_blocks     : packed expert weights [ne][N][K] (GGUF blocks, contiguous)
//   x_f16        : gathered activations [expanded][K] (expert-contiguous)
//   out_f16      : [expanded][N]
//   d_offsets    : device int32 [ne+1] expert row offsets
//   h_max_rows   : host-known max rows per expert (sizes grid.y; < 96 picks
//                  the BM=32 small-M tile — pp512 top-8/128 routing averages
//                  ~32 rows per expert)
//   expanded     : total gathered rows (activation quantize span)
//   qkind: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K
bool mmq_imma_moe_gemm(const void* w_blocks, int qkind, const __half* x_f16, __half* out_f16,
                       const int32_t* d_offsets, int h_max_rows, int expanded, int ne, int N,
                       int K, cudaStream_t stream);

// Free cached weight planes + activation scratch (tests / teardown).
void mmq_q8_imma_release_all();

}  // namespace imp
