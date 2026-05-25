#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

namespace imp {

struct block_q8_1;

// Fused Q4_K × Q8_1 dp4a GEMM for MoE expert prefill.
//
// Same architecture as gemm_q6k_moe_fused: weight-stationary, Q8_1 activations
// in shared memory, dp4a accumulation. Eliminates the FP16 intermediate buffer
// that causes the 8.3× bandwidth overhead vs llama.cpp's MMQ.
//
// Q4_K specifics: unsigned 4-bit nibbles with per-sub-block scale+min. The
// min correction uses the Q8_1 sum field (block_q8_1::s) for the bias term.
void gemm_q4k_moe_fused(const void* packed_weight, const block_q8_1* q8_base, const float* d8_base,
                         const half* s8_base, void* c_base, const int32_t* offsets, int K, int N,
                         int n_experts, size_t weight_stride, cudaStream_t stream = nullptr);

}  // namespace imp
