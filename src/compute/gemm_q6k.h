#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imp {

struct block_q8_1;  // forward declaration (defined in gemm.h)

// Fused Q6_K × Q8_1 dp4a GEMM for MoE expert prefill.
//
// Weight-stationary kernel that reads Q6_K weights from DRAM and Q8_1
// activations from L2 cache, eliminating the FP16 intermediate buffer entirely.
// Uses dp4a (INT8x4 dot product) for accumulation — same numerical path as
// the proven gemv_q6k_q8_1 decode kernel.
//
// For each expert e with M_e tokens (determined by offsets):
//   C_e[M_e, N] = Q8_1(A_e)[M_e, K] × dequant(B_q6k_e[N, K])^T
//
// packed_weight: [n_experts, N, K] raw Q6_K bytes on device
// q8_base:       [expanded, K/32] block_q8_1 pre-quantized activations (device)
// d8_base:       [expanded, K/32] float block scales (device)
// c_base:        [expanded, N] FP16 output (device)
// offsets:       [n_experts+1] int32 expert token offsets (DEVICE — no D2H needed!)
// K, N:          inner and output dimensions (K must be multiple of 256)
// n_experts:     total number of experts
// weight_stride: bytes between consecutive experts in packed_weight
// stream:        CUDA stream
void gemm_q6k_moe_fused(
    const void* packed_weight,
    const block_q8_1* q8_base,
    const float* d8_base,
    void* c_base,
    const int32_t* offsets,
    int K, int N,
    int n_experts,
    size_t weight_stride,
    cudaStream_t stream = nullptr);

} // namespace imp
