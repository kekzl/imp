#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

struct block_q8_1;

// Fused Q4_K/Q5_K × FP16 scalar GEMM for MoE expert prefill.
// Same interface as gemm_q6k_fused_moe_prefill. Uses FP16 activations directly.
void gemm_q4k_fused_moe_prefill(const void* packed_weights, const void* activations, void* output,
                                const int32_t* d_offsets, int N, int K, size_t expert_stride_bytes,
                                int n_experts, cudaStream_t stream = nullptr);
void gemm_q5k_fused_moe_prefill(const void* packed_weights, const void* activations, void* output,
                                const int32_t* d_offsets, int N, int K, size_t expert_stride_bytes,
                                int n_experts, cudaStream_t stream = nullptr);

// Fused Q4_K/Q5_K × Q8_1 dp4a GEMM for MoE expert prefill.
// Same interface as gemm_q6k_moe_fused. Activations must be pre-quantized to Q8_1.
// Uses shared memory tiling for activation reuse across output columns.
void gemm_q4k_dp4a_moe_fused(const void* packed_weight, const block_q8_1* q8_base,
                              const float* d8_base, void* c_base, const int32_t* offsets,
                              int K, int N, int n_experts, size_t weight_stride,
                              cudaStream_t stream = nullptr);
void gemm_q5k_dp4a_moe_fused(const void* packed_weight, const block_q8_1* q8_base,
                              const float* d8_base, void* c_base, const int32_t* offsets,
                              int K, int N, int n_experts, size_t weight_stride,
                              cudaStream_t stream = nullptr);

}  // namespace imp
