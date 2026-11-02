#pragma once

// cuBLASLt stand-ins for cublasGemmEx / cublasGemmStridedBatchedEx / cublasGemmBatchedEx: the same
// column-major arguments without a cuBLAS handle, so imp links cuBLASLt only. Descriptors and the
// heuristic's first algo are cached per (shape, types, pointer alignment); workspace = gemm()'s.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <source_location>

namespace imp {

// FP32 operands run TF32, as under the CUBLAS_TF32_TENSOR_OP_MATH handles these calls replace.
[[nodiscard]] constexpr cublasComputeType_t lt_compute_32f(cudaDataType_t a_type) {
    return a_type == CUDA_R_32F ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

[[nodiscard]] cublasStatus_t lt_gemm_strided_batched(
    cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A,
    cudaDataType_t a_type, int lda, long long stride_a, const void* B, cudaDataType_t b_type, int ldb,
    long long stride_b, const void* beta, void* C, cudaDataType_t c_type, int ldc, long long stride_c,
    int batch, cublasComputeType_t compute, cudaStream_t stream,
    const std::source_location& loc = std::source_location::current());

[[nodiscard]] inline cublasStatus_t lt_gemm_ex(
    cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A,
    cudaDataType_t a_type, int lda, const void* B, cudaDataType_t b_type, int ldb, const void* beta, void* C,
    cudaDataType_t c_type, int ldc, cublasComputeType_t compute, cudaStream_t stream,
    const std::source_location& loc = std::source_location::current()) {
    return lt_gemm_strided_batched(transa, transb, m, n, k, alpha, A, a_type, lda, 0, B, b_type, ldb, 0, beta,
                                   C, c_type, ldc, 0, 1, compute, stream, loc);
}

// Pointer-array batch: a / b / c are device arrays of batch matrix pointers.
[[nodiscard]] cublasStatus_t lt_gemm_batched(
    cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha,
    const void* const* a, cudaDataType_t a_type, int lda, const void* const* b, cudaDataType_t b_type,
    int ldb, const void* beta, void* const* c, cudaDataType_t c_type, int ldc, int batch,
    cublasComputeType_t compute, cudaStream_t stream,
    const std::source_location& loc = std::source_location::current());

// Destroys the cached descriptors (gemm_cleanup()).
void lt_gemm_ex_cleanup();

}  // namespace imp
