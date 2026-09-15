#pragma once
// Narrow FP16 GEMM for prefill rows (M > 32): up to two (W,C) pairs in one launch.
//   C_p[M,N_p] = A[M,K] @ W_p[N_p,K]^T   p = 0,1
// FP16 operands, FP32 accumulate, FP16 out; N_0+N_1 <= 128, N_p%8==0, K%32==0, A/W 16 B
// aligned. Built for the
// GDN alpha/beta projections at prefill (N=n_heads), where cuBLAS runs two split-K GEMMs plus
// two reduce launches per layer. Split-K across CTAs, the partials summed in index order by a
// second launch: bitwise deterministic per shape. Workspace (device, 16 B aligned) holds the
// FP32 partials; the split adapts to what fits, split 1 needs no workspace and no second launch.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// Bytes for a split-K of `split` over M rows and N_0+N_1 = n_total columns.
size_t gemm_f16_narrow_prefill_workspace_bytes(int M, int n_total, int split);

// Returns false (nothing launched) when a shape does not fit; the caller keeps its previous
// path. N1 == 0 runs pair 0 only. ws may be null (split 1).
bool gemm_f16_narrow_prefill(const half* A, int M, int K, const half* W0, half* C0, int N0, const half* W1,
                             half* C1, int N1, void* ws, size_t ws_bytes, cudaStream_t stream);

// Grid-shape knobs for the sweep in tests/test_gemm_f16_narrow_prefill.cu: the split-K cap
// (power of two, <= 32) and the CTA count the split reaches for. 0 = the shipped default.
void gemm_f16_narrow_prefill_tune(int max_split, int target_ctas);

}  // namespace imp
