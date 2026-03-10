#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Fused Q6_K × FP16 GEMM with tensor cores (WMMA) for MoE expert prefill.
//
// Reads Q6_K weights directly, dequants to FP16 in shared memory, and
// multiplies with FP16 activations using WMMA 16×16×16 instructions.
// Eliminates the global memory dequant buffer entirely.
//
// Compared to the scalar fused kernel (gemm_moe_fused.cu):
// - ~30x higher compute throughput via tensor cores
// - Same Q6_K→direct memory savings (no intermediate FP16 buffer)
// - Better activation reuse via explicit shared memory tiling
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
// sorted_token_ids: optional [total_expanded] int32. When non-null, activations
// are read via indirection (gather-free): activations[sorted_token_ids[i], :].
// When null, reads activations[i, :] directly (requires pre-gathered input).
void gemm_q6k_fused_moe_prefill_tc(const void* packed_weights,
                                    const void* activations,
                                    void* output,
                                    const int32_t* d_offsets,
                                    int N, int K,
                                    size_t expert_stride_bytes,
                                    int n_experts,
                                    cudaStream_t stream = nullptr,
                                    const int32_t* sorted_token_ids = nullptr);

} // namespace imp
