#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

struct block_q8_1;  // forward declaration (defined in gemm.h)

// Fused Q6_K x Q8_1 dp4a GEMM for MoE expert prefill: weight-stationary kernel reading Q6_K
// weights from DRAM and Q8_1 activations from L2, eliminating the FP16 intermediate entirely.
// Uses dp4a, same numerical path as the gemv_q6k_q8_1 decode kernel.
// For each expert e with M_e tokens: C_e[M_e,N] = Q8_1(A_e)[M_e,K] x dequant(B_q6k_e[N,K])^T
//   packed_weight [n_experts,N,K] raw Q6_K; q8_base/d8_base [expanded,K/32] pre-quantized
//   activations/scales; c_base [expanded,N] FP16 output; offsets [n_experts+1] DEVICE (no
//   D2H); K must be a multiple of 256; weight_stride: bytes between experts.
void gemm_q6k_moe_fused(const void* packed_weight, const block_q8_1* q8_base, const float* d8_base,
                        void* c_base, const int32_t* offsets, int K, int N, int n_experts,
                        size_t weight_stride, cudaStream_t stream = nullptr);

}  // namespace imp
