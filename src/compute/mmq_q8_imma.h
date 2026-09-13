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

// MoE grouped prefill GEMM over ne experts in one launch (gridDim.z=ne).
// w_blocks[ne][N][K] packed GGUF blocks; x_f16[expanded][K] gathered activations
// (expert-contiguous); out_f16[expanded][N]; d_offsets device int32[ne+1] row offsets.
// h_max_rows: host max rows/expert, sizes grid.y (<96 picks BM=32 small-M tile;
// pp512 top-8/128 routing averages ~32 rows/expert). qkind: 0=Q8_0, 1=Q4_K, 2=Q6_K.
bool mmq_imma_moe_gemm(const void* w_blocks, int qkind, const __half* x_f16, __half* out_f16,
                       const int32_t* d_offsets, int h_max_rows, int expanded, int ne, int N,
                       int K, cudaStream_t stream);

// Free cached weight planes + activation scratch (tests / teardown).
void mmq_q8_imma_release_all();

// Per-weight prefill cache VRAM budget (Q8_0 SoA planes, Q6_K repack), taken lazily on
// first prefill. Uncharged planes grew into whatever KV pool left free, making the spill
// victim a lottery (#1899). Engine::init sets the plan's reserved figure
// (vram_budget's imma_plane_bytes); a take past it declines to the dequant path.
// SIZE_MAX (default) = uncapped; 0 = no planes.
void mmq_q8_imma_set_plane_budget(size_t bytes);
size_t mmq_q8_imma_plane_bytes_used();

// What the caches above will take for one (N, K) Q8_0 weight — the planner
// charges this exact arithmetic, so the two cannot drift.
size_t imma_q8_plane_bytes(int64_t N, int64_t K);

// Takes the IMMA prefill activation scratch (+ split-K slice) from the T2 arena once, at
// the bound exec_t2_demand charged as `imma_scratch`. Call from Engine::init after the
// arena opens. No-op if the model has no IMMA-eligible weights (rows or k == 0).
void mmq_q8_imma_preallocate(int rows, int k);

}  // namespace imp
