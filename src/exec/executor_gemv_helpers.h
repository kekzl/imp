#pragma once

#include "compute/gemm.h"
#include "compute/gemm_q6k.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Residual-fused GEMV dispatch by quant type: y[i] = dot(W[i], x) + residual[i].
// Shared by executor_attention.cu and executor_ffn.cu.
static void dispatch_gemv_residual(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8,
                                   half* y, const half* residual, int M, int K, cudaStream_t stream) {
    switch (qtype) {
        case QType::Q6_K:
            gemv_q6k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream);
            break;
        case QType::Q4_0:
            gemv_q4_0_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream);
            break;
        case QType::Q4_K:
            gemv_q4_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream);
            break;
        case QType::Q5_K:
            gemv_q5_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream);
            break;
        case QType::Q2_K:
            gemv_q2_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream);
            break;
        case QType::Q3_K:
            gemv_q3_k_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream);
            break;
        default:
            gemv_q8_0_q8_1_residual(W, q8_1, d8, y, residual, M, K, stream);
            break;
    }
}

// LM head dp4a GEMV dispatch: y = W @ x with FP32 output (logits).
// Shared by executor_forward.cu (production forward_logits) and
// executor_perplexity.cu (teacher-forced eval) — the eval path MUST use the
// same kernels as production or PPL stops measuring what serving does.
static void dispatch_gemv_fp32(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8,
                               float* y, int M, int K, cudaStream_t stream) {
    switch (qtype) {
        case QType::Q6_K:
            gemv_q6k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_0:
            gemv_q4_0_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_K:
            gemv_q4_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q5_K:
            gemv_q5_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q2_K:
            gemv_q2_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q3_K:
            gemv_q3_k_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
        default:
            gemv_q8_0_q8_1_fp32(W, q8_1, d8, y, M, K, stream);
            break;
    }
}

}  // namespace imp
