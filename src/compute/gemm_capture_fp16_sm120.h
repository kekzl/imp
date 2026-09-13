#pragma once

// Capture-safe sm_120 FP16 dense GEMM: cuBLASLt fails with CUBLAS_STATUS_INTERNAL_ERROR
// under stream capture on sm_120. CUTLASS 4.5's sm_120 CollectiveBuilder only ships
// F8F6F4 MMA, so dense FP16 needs a hand-tuned kernel here (nvcuda::wmma HMMA m16n8k16,
// compiling to mma.sync). All decisions are device-side, fully graph-safe.
// Layout matches cuBLAS OP_T on B: A[M,K] row-major, B[N,K] row-major (semantically
// B^T), D[M,N] row-major; D = alpha*A@B^T + beta*D.

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
