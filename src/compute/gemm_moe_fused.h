#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Fused Q6_K × FP16 GEMM for MoE expert prefill projections.
//
// Reads Q6_K weights directly from packed expert tensors and multiplies with
// FP16 activations, completely eliminating the intermediate FP16 dequant buffer.
// Reduces DRAM traffic by ~5x compared to separate dequant + GEMM.
//
// Each warp processes one output row for one expert. All lanes cooperatively
// dequant the same Q6_K block (coalesced reads), then multiply with FP16
// activations that naturally cache in L1 (shared across rows of the same expert).
//
// For each expert e with M_e tokens:
//   C_e[M_e, N] = A_e[M_e, K] @ W_e[N, K]^T
//
// packed_weights:      all experts' Q6_K data contiguous [n_experts, N, K]
// activations:         FP16 input [total_expanded, K]
// output:              FP16 output [total_expanded, N]
// d_offsets:           device array [n_experts+1] cumulative token offsets
// N:                   output dimension (rows per expert weight matrix)
// K:                   inner dimension (cols, must be multiple of 256)
// expert_stride_bytes: bytes between experts in packed_weights
// n_experts:           total number of experts
void gemm_q6k_fused_moe_prefill(const void* packed_weights,
                                const void* activations,
                                void* output,
                                const int32_t* d_offsets,
                                int N, int K,
                                size_t expert_stride_bytes,
                                int n_experts,
                                cudaStream_t stream = nullptr);

} // namespace imp
