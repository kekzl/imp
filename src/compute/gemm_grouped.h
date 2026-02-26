#pragma once

#include "core/tensor.h"
#include <vector>
#include <cuda_runtime.h>

namespace imp {

// cuBLASLt Grouped GEMM for MoE expert parallelism
void gemm_grouped(const std::vector<Tensor>& A,
                  const std::vector<Tensor>& B,
                  std::vector<Tensor>& C,
                  cudaStream_t stream = nullptr);

// Batched GEMM for MoE expert parallelism via cublasGemmBatchedEx.
// All active experts are dispatched in a single cuBLAS call.
//
// For each expert e (0..n_experts-1) with count[e] > 0:
//   C_e = A_e @ B_e^T
//   A_e: [count_e, K]  — gathered tokens for expert e (from gathered buffer)
//   B_e: [N, K]        — expert weight (from packed or unpacked tensor)
//   C_e: [count_e, N]  — output
//
// a_base:    base pointer of gathered input buffer [expanded, K] compute_dtype
// c_base:    base pointer of output buffer [expanded, N] compute_dtype
// offsets:   host array [n_experts+1] with start offsets into expanded dimension
// b_ptrs:    host array [n_experts] with device pointers to each expert's weight [N, K]
// K, N:      inner and output dimensions
// dtype:     element type (FP16/BF16)
// n_experts: total number of experts
// d_work_ptrs: optional pre-allocated device memory for pointer arrays.
// If non-null, must hold 3 * n_experts void* entries (A, B, C pointer arrays).
// If null, device arrays are allocated/freed per call via cudaMallocAsync.
void gemm_moe_batched(const void* a_base, void* c_base,
                      const int32_t* offsets,
                      const void* const* b_ptrs,
                      int K, int N, DType dtype,
                      int n_experts,
                      cudaStream_t stream = nullptr,
                      void** d_work_ptrs = nullptr,
                      DType output_dtype = DType(255));

} // namespace imp
