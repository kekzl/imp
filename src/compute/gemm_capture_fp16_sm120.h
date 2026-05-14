#pragma once

// Capture-safe sm_120 FP16 dense GEMM. cuBLASLt fails with
// CUBLAS_STATUS_INTERNAL_ERROR (status 14) under stream capture on
// consumer Blackwell (sm_120). CUTLASS 4.5's sm_120 CollectiveBuilder
// only ships F8F6F4 MMA, so dense FP16 must be hand-tuned for sm_120
// directly. This file declares the dispatch entry point.
//
// Implementation uses nvcuda::wmma (HMMA m16n8k16) — the same tensor
// core path that compiles to mma.sync on sm_120. All decisions are
// device-side, no host-side cuBLAS heuristics — fully graph-safe.
//
// Layout convention matches cuBLAS's GEMM with OP_T on B:
//   A: [M, K] row-major FP16
//   B: [N, K] row-major FP16 (semantically B^T in the GEMM)
//   D: [M, N] row-major FP16
//   D = alpha * A @ B^T + beta * D

#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// True on sm_120+ hardware. Cached.
bool capture_gemm_fp16_sm120_available();

// Returns false if the GEMM cannot be implemented for the requested
// shape (M, N, K must be positive; M and N must be tile-aligned for
// the v1 kernel). Caller must fall back to the existing path if false.
bool gemm_capture_fp16_sm120(const void* A, const void* B, void* D, int M, int N, int K, float alpha,
                              float beta, cudaStream_t stream);

}  // namespace imp
