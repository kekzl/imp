#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

enum class DType : uint8_t;

// CUTLASS FP16 batched GEMM for MoE expert parallelism.
// Drop-in replacement for gemm_moe_batched() with ~3us launch overhead
// (vs ~27us for cuBLAS), making L2-chunked processing viable.
//
// For each expert e (0..n_experts-1) with count[e] > 0:
//   C_e = A_e @ B_e^T
//   A_e: [count_e, K]  -- gathered tokens in compute_dtype
//   B_e: [N, K]        -- expert weight in compute_dtype
//   C_e: [count_e, N]  -- output in compute_dtype
//
// a_base:      base pointer of gathered input buffer [expanded, K]
// c_base:      base pointer of output buffer [expanded, N]
// offsets:     host array [n_experts+1] with start offsets into expanded dimension
// b_ptrs:      host array [n_experts] with device pointers to each expert's weight [N, K]
// K, N:        inner and output dimensions
// dtype:       element type (FP16)
// n_experts:   total number of experts in this chunk
// stream:      CUDA stream
// d_work_ptrs: optional pre-allocated device memory for pointer arrays.
//              If non-null, must hold 3 * n_experts void* entries (A, B, C arrays).
void gemm_moe_cutlass(const void* a_base, void* c_base,
                      const int32_t* offsets,
                      const void* const* b_ptrs,
                      int K, int N, DType dtype, int n_experts,
                      cudaStream_t stream = nullptr,
                      void** d_work_ptrs = nullptr);

} // namespace imp
