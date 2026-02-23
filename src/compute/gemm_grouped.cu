#include "compute/gemm_grouped.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace imp {

// ---------------------------------------------------------------------------
// cuBLAS handle (lazily initialized, process-lifetime)
// Uses cublasGemmEx — the same proven path as gemm.cu's gemm().
// ---------------------------------------------------------------------------
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm_grouped: cublasCreate failed (status %d)\n", (int)st);
            abort();
        }
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

// ---------------------------------------------------------------------------
// Helper: map DType -> cudaDataType
// ---------------------------------------------------------------------------
static cudaDataType_t dtype_to_cuda(DType dt) {
    switch (dt) {
        case DType::FP32:     return CUDA_R_32F;
        case DType::FP16:     return CUDA_R_16F;
        case DType::BF16:     return CUDA_R_16BF;
        case DType::FP8_E4M3: return CUDA_R_8F_E4M3;
        case DType::FP8_E5M2: return CUDA_R_8F_E5M2;
        case DType::INT8:     return CUDA_R_8I;
        case DType::INT32:    return CUDA_R_32I;
        default:
            fprintf(stderr, "imp::gemm_grouped: unsupported dtype %d\n", (int)dt);
            abort();
    }
}

// ---------------------------------------------------------------------------
// Helper: choose cuBLAS compute type for a given operand dtype
// ---------------------------------------------------------------------------
static cublasComputeType_t dtype_to_compute(DType dt) {
    switch (dt) {
        case DType::FP32:     return CUBLAS_COMPUTE_32F;
        case DType::FP16:     return CUBLAS_COMPUTE_32F;
        case DType::BF16:     return CUBLAS_COMPUTE_32F;
        case DType::FP8_E4M3: return CUBLAS_COMPUTE_32F;
        case DType::FP8_E5M2: return CUBLAS_COMPUTE_32F;
        case DType::INT8:     return CUBLAS_COMPUTE_32I;
        default:              return CUBLAS_COMPUTE_32F;
    }
}

// ---------------------------------------------------------------------------
// Run a single expert GEMM via cublasGemmEx (same as gemm.cu's gemm()).
//   C_i = A_i @ B_i^T    (alpha=1, beta=0)
//   A_i [M_i, K],  B_i [N, K],  C_i [M_i, N]   -- row-major
//
// Column-major trick (identical to gemm.cu):
//   B row-major [N,K] = col-major [K,N,ld=K]; CUBLAS_OP_T → [N,K]
//   A row-major [Mi,K] = col-major [K,Mi,ld=K]; CUBLAS_OP_N
//   Result: [N,Mi] col-major = [Mi,N] row-major
// ---------------------------------------------------------------------------
static void run_expert_matmul(cublasHandle_t handle,
                              const Tensor& Ai, const Tensor& Bi, Tensor& Ci,
                              cudaStream_t stream) {
    const int64_t Mi = Ai.shape[0];   // tokens routed to this expert
    const int64_t K  = Ai.shape[1];
    const int64_t N  = Bi.shape[0];   // B is [N, K], we compute A @ B^T

    if (Mi == 0) return;

    cudaDataType_t cuda_dt_A = dtype_to_cuda(Ai.dtype);
    cudaDataType_t cuda_dt_B = dtype_to_cuda(Bi.dtype);
    cudaDataType_t cuda_dt_C = dtype_to_cuda(Ci.dtype);
    cublasComputeType_t compute_type = dtype_to_compute(Ai.dtype);

    cublasSetStream(handle, stream);

    if (compute_type == CUBLAS_COMPUTE_32I) {
        int32_t alpha = 1;
        int32_t beta  = 0;
        cublasStatus_t st = cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)N, (int)Mi, (int)K,
            &alpha,
            Bi.data, cuda_dt_B, (int)K,
            Ai.data, cuda_dt_A, (int)K,
            &beta,
            Ci.data, cuda_dt_C, (int)N,
            compute_type, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm_grouped: cublasGemmEx failed for expert "
                            "(M=%lld, K=%lld, N=%lld, status %d)\n",
                    (long long)Mi, (long long)K, (long long)N, (int)st);
        }
    } else {
        float alpha = 1.0f;
        float beta  = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)N, (int)Mi, (int)K,
            &alpha,
            Bi.data, cuda_dt_B, (int)K,
            Ai.data, cuda_dt_A, (int)K,
            &beta,
            Ci.data, cuda_dt_C, (int)N,
            compute_type, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm_grouped: cublasGemmEx failed for expert "
                            "(M=%lld, K=%lld, N=%lld, status %d)\n",
                    (long long)Mi, (long long)K, (long long)N, (int)st);
        }
    }
}

// ---------------------------------------------------------------------------
// gemm_grouped: batched GEMM for MoE expert parallelism
//
// For each expert i:
//   C[i] = A[i] @ B[i]^T
//   A[i]: [tokens_for_expert_i, K]
//   B[i]: [N, K]              (weight matrix: [out_features, in_features])
//   C[i]: [tokens_for_expert_i, N]
//
// All GEMMs are issued on the same stream to allow the GPU scheduler to
// overlap kernel execution where possible.
//
// When IMP_CUDA_13_1 is defined, we could use a true grouped/batched GEMM
// API.  For now we fall back to a loop of individual cublasLtMatmul calls
// which is the safe portable path.
// ---------------------------------------------------------------------------
void gemm_grouped(const std::vector<Tensor>& A,
                  const std::vector<Tensor>& B,
                  std::vector<Tensor>& C,
                  cudaStream_t stream) {
    const size_t num_experts = A.size();
    if (num_experts == 0) return;
    if (B.size() != num_experts || C.size() != num_experts) {
        fprintf(stderr, "imp::gemm_grouped: A/B/C vector sizes must match "
                        "(got %zu, %zu, %zu)\n", A.size(), B.size(), C.size());
        return;
    }

    cublasHandle_t handle = get_cublas_handle();

    for (size_t i = 0; i < num_experts; ++i) {
        run_expert_matmul(handle, A[i], B[i], C[i], stream);
    }
}

} // namespace imp
