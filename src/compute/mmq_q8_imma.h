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
// Declines: M < 2, N % 2, K % 64, beta ∉ {0,1}, capture-guarded miss.
// allow_splitk=false: no M<=32 split-K, so a row matches its value at any M (prompt rows, #2152).
// reuse_act: x is unchanged since this thread's last IMMA quantize; it is skipped when that quantize
// wrote exactly (x, M, K, stream) into the current scratch (q/k/v share one input, #2416).
[[nodiscard]] bool mmq_q8_imma_gemm(const void* w_q8_blocks, const __half* x_f16, __half* out_f16, int M,
                                    int N, int K, cudaStream_t stream, float beta = 0.0f,
                                    bool allow_splitk = true, bool reuse_act = false);

// Drops the last-quantize tag: a caller about to rely on reuse_act clears it first, so a tag left by
// an earlier layer on the same workspace pointer can never be matched.
void mmq_imma_act_forget();

// Dense Q4_K (new stack — distinct from the retired 2026-05 64x32 q4k_imma
// kernel): same contract; K % 256 == 0 (Q4_K super-block).
[[nodiscard]] bool mmq_q4k_imma_gemm(const void* w_q4k_blocks, const __half* x_f16, __half* out_f16, int M, int N, int K,
                       cudaStream_t stream, float beta = 0.0f, bool allow_splitk = true);

// Dense Q6_K: per-16 scales via half-MMA split (symmetric, no beta term), GGUF 210-B blocks read
// in place. K % 256 == 0. Not routed: dense Q6_K loses to dequant + fp16-acc cuBLAS
// (Qwen3-14B-Q6_K pp512 6957 -> 3747 tok/s); kept as the BM=128 test entry.
[[nodiscard]] bool mmq_q6k_imma_gemm(const void* w_q6k_blocks, const __half* x_f16, __half* out_f16, int M, int N, int K,
                       cudaStream_t stream, float beta = 0.0f);

// MoE grouped prefill GEMM over ne experts in one launch (gridDim.z=ne).
// w_blocks[ne][N][K] packed GGUF blocks; x_f16[expanded][K] gathered activations
// (expert-contiguous); out_f16[expanded][N]; d_offsets device int32[ne+1] row offsets.
// h_max_rows: upper bound on rows/expert, sizes grid.y (surplus CTAs exit on their offsets);
// rows_hint (0 = h_max_rows) picks the tile: <96 selects BM=32 small-M (
// pp512 top-8/128 routing averages ~32 rows/expert). qkind: 0=Q8_0, 1=Q4_K, 2=Q6_K, 3=Q5_1, 4=Q5_K.
// reuse_act: x_f16 is unchanged since the previous call quantized it (gate then up, #2469); the
// quantize is skipped only if that call was the last one on this scratch.
// Shape gate of mmq_imma_moe_gemm: N even, K a multiple of 256 (K-quants) or 64 (Q8_0, Q5_1).
[[nodiscard]] constexpr bool mmq_imma_moe_shape_ok(int qkind, int N, int K) {
    return N % 2 == 0 && K % ((qkind == 1 || qkind == 2 || qkind == 4) ? 256 : 64) == 0;
}

[[nodiscard]] bool mmq_imma_moe_gemm(const void* w_blocks, int qkind, const __half* x_f16, __half* out_f16,
                                     const int32_t* d_offsets, int h_max_rows, int expanded, int ne, int N,
                                     int K, cudaStream_t stream, int rows_hint = 0, bool reuse_act = false);

// Free cached weight planes + activation scratch (tests / teardown).
void mmq_q8_imma_release_all();

// Builds the Q8_0 planes of one dense weight now (Phase 4c, before any capture); the caller may
// then free the Q8_0 source: the planes live until release_all (engine teardown, one context per
// process). False on a shape the IMMA kernel declines (N odd, K % 64), a budget hit or a failed malloc.
[[nodiscard]] bool mmq_q8_imma_build_planes(const void* src, int N, int K, cudaStream_t stream);
// FP16 [N][K] from the planes keyed by `src`, bit-equal to dequant_gpu on the Q8_0 source.
[[nodiscard]] bool mmq_q8_imma_planes_to_f16(const void* src, __half* dst, cudaStream_t stream);

// Per-weight prefill cache VRAM budget (Q8_0 SoA planes), taken lazily on
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
