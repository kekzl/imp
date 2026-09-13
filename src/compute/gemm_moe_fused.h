#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Fused Q6_K x FP16 GEMM for MoE expert prefill projections: reads Q6_K weights directly,
// eliminating the intermediate FP16 dequant buffer. Each warp processes one output row per
// expert; lanes cooperatively dequant the same Q6_K block, then multiply with FP16 activations
// cached in L1 across rows of the same expert.
// For each expert e with M_e tokens: C_e[M_e,N] = A_e[M_e,K] @ W_e[N,K]^T
//   packed_weights: [n_experts,N,K] contiguous; activations: [total_expanded,K] FP16
//   output: [total_expanded,N] FP16; d_offsets: [n_experts+1] cumulative token offsets
//   N: output dim; K: inner dim (multiple of 256); expert_stride_bytes: bytes between experts
void gemm_q6k_fused_moe_prefill(const void* packed_weights, const void* activations, void* output,
                                const int32_t* d_offsets, int N, int K, size_t expert_stride_bytes,
                                int n_experts, cudaStream_t stream = nullptr);

}  // namespace imp
