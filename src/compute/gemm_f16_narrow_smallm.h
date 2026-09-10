#pragma once
// Narrow FP16 GEMM for batched decode: up to two (W, C) pairs in one launch.
//
//   C_p[M, N_p] = A[M, K] @ W_p[N_p, K]^T     p = 0, 1
//
// FP16 operands, FP32 accumulate, FP16 out; M <= 32, N_p % 16 == 0, K % 128 == 0.
// Built for the GDN alpha/beta projections (N = n_heads, 48 on Qwen3.8-27B):
// at M = 32 they ran as two cuBLAS GEMMs of nvjet + splitKreduce each, 4
// launches per GDN layer. Split-K across CTAs with a fixed-order final
// reduce: bitwise deterministic per shape.
//
// Workspace (device, zero-initialised once): gemm_f16_narrow_smallm_workspace_bytes()
// for the largest N_0 + N_1 the caller will pass. The tickets in it are reset
// by the kernel itself, so one zeroing at allocation is enough.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

size_t gemm_f16_narrow_smallm_workspace_bytes(int n_total_max);

// Returns false (nothing launched) when a shape or the workspace does not
// fit; the caller keeps its previous path. N1 == 0 runs pair 0 only.
bool gemm_f16_narrow_smallm(const half* A, int M, int K, const half* W0, half* C0, int N0, const half* W1,
                            half* C1, int N1, void* ws, size_t ws_bytes, cudaStream_t stream);

}  // namespace imp
