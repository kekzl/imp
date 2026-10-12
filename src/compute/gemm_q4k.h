#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

struct block_q8_1;

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


// Dense Q4_K/Q5_K x Q8_1 dp4a GEMM for non-MoE prefill: quantizes FP16 activations [M,K] to
// Q8_1, then computes directly from Q4_K/Q5_K blocks via dp4a, avoiding the FP16 weight cache
// intermediate (0.55 B/elem vs 2.0 B/elem). q8_scratch: [M*ceil(K/32)] block_q8_1, d8_scratch:
// [M*ceil(K/32)] float. beta must be 0 (no residual accumulation).
void gemm_q4k_dp4a_dense(const void* packed_q4k, const half* activations, half* output,
                          void* q8_scratch, float* d8_scratch,
                          int M, int N, int K, cudaStream_t stream = nullptr);
void gemm_q5k_dp4a_dense(const void* packed_q5k, const half* activations, half* output,
                          void* q8_scratch, float* d8_scratch,
                          int M, int N, int K, cudaStream_t stream = nullptr);

// Q4_K dense dp4a for decode rows: any K % 256 (16/8/4-row activation tile by K), beta 0 or 1
// (out += for 1). False (nothing launched) when no tile fits or beta is neither (#2688).
[[nodiscard]] bool gemm_q4k_dp4a_dense_rows(const void* packed_q4k, const half* activations, half* output,
                                            void* q8_scratch, float* d8_scratch, int M, int N, int K,
                                            float beta, cudaStream_t stream);

}  // namespace imp
