#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Fused Q6_K x FP16 GEMM with tensor cores (WMMA) for MoE expert prefill: dequants Q6_K to
// FP16 in shared memory, multiplies via WMMA 16x16x16. Higher compute throughput than the
// scalar fused kernel (gemm_moe_fused.cu) via tensor cores, same Q6_K->direct memory savings,
// better activation reuse via explicit shared-memory tiling.
// For each expert e with M_e tokens: C_e[M_e,N] = A_e[M_e,K] @ W_e[N,K]^T
//   packed_weights [n_experts,N,K]; activations [total_expanded,K] FP16; output
//   [total_expanded,N] FP16; d_offsets [n_experts+1]; K multiple of 256
// sorted_token_ids: optional [total_expanded] int32; when non-null, activations are read via
// indirection (gather-free) as activations[sorted_token_ids[i],:]; else reads activations[i,:]
// directly (requires pre-gathered input).
void gemm_q6k_fused_moe_prefill_tc(const void* packed_weights, const void* activations, void* output,
                                   const int32_t* d_offsets, int N, int K, size_t expert_stride_bytes,
                                   int n_experts, cudaStream_t stream = nullptr,
                                   const int32_t* sorted_token_ids = nullptr);

}  // namespace imp
