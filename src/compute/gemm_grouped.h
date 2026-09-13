#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Batched GEMM for MoE expert parallelism via cublasGemmBatchedEx; all active experts
// dispatched in one cuBLAS call. For each expert e with count[e]>0: C_e = A_e @ B_e^T,
// A_e [count_e,K] gathered tokens, B_e [N,K] expert weight, C_e [count_e,N] output.
//   a_base/c_base: gathered input/output buffers [expanded,K]/[expanded,N]
//   offsets: host [n_experts+1] start offsets into expanded dim
//   b_ptrs: host [n_experts] device pointers to each expert's weight [N,K]
//   d_work_ptrs: optional pre-allocated device memory, 3*n_experts void* entries (A,B,C
//   pointer arrays); if null, device arrays are allocated/freed per call via cudaMallocAsync.
void gemm_moe_batched(const void* a_base, void* c_base, const int32_t* offsets, const void* const* b_ptrs,
                      int K, int N, QType dtype, int n_experts, cudaStream_t stream = nullptr,
                      void** d_work_ptrs = nullptr, QType output_dtype = QType(255),
                      const float* a_scales = nullptr, const float* b_scales = nullptr);

}  // namespace imp
