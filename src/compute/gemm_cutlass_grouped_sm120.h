#pragma once

#ifdef IMP_USE_CUTLASS

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

// Run grouped GEMM for MoE: one CUTLASS persistent kernel launch
// processes all active experts. Uses CUTLASS 2.x GemmGrouped with
// kDeviceOnly scheduling (cp.async, SM80-compatible, runs on SM120).
//
// For each expert i (0..n_problems-1):
//   C[i] = A[i] @ B[i]^T
//   A[i]: [M_i, K] FP16 (routed tokens for expert i)
//   B[i]: [N, K]   FP16 (expert i's weight matrix)
//   C[i]: [M_i, N] FP16
//
// All pointer arrays and problem_m must be device-resident.
bool gemm_grouped_cutlass_sm120(
    const void* const* d_A_ptrs,    // Device array of activation pointers [n_problems]
    const void* const* d_B_ptrs,    // Device array of weight pointers [n_problems]
    void* const* d_C_ptrs,          // Device array of output pointers [n_problems]
    const int* d_problem_m,         // Device array of M values per expert [n_problems]
    int N, int K,                   // Shared across all experts
    int n_problems,                 // Number of active experts
    void* workspace,                // Pre-allocated workspace
    size_t workspace_size,
    cudaStream_t stream);

// Query workspace size for given problem count.
size_t gemm_grouped_cutlass_sm120_workspace(int max_problems, int max_M, int N, int K);

// Check if CUTLASS grouped GEMM is available on this device (sm_80+).
bool cutlass_grouped_gemm_sm120_available();

} // namespace imp

#endif // IMP_USE_CUTLASS
