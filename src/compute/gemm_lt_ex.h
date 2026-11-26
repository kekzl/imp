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

// Arguments of lt_gemm_strided_batched / lt_gemm_ex (audit D-11), the old parameter list in order:
// built with designated initializers, so a swapped m / n / k or lda / ldb / ldc reads wrong at the
// call site. Strides 0 and batch 1 = one GEMM; the operand fields have no meaningful default.
struct LtGemmArgs {
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    const void* alpha = nullptr;
    const void* A = nullptr;
    cudaDataType_t a_type = CUDA_R_16F;
    int lda = 0;
    long long stride_a = 0;
    const void* B = nullptr;
    cudaDataType_t b_type = CUDA_R_16F;
    int ldb = 0;
    long long stride_b = 0;
    const void* beta = nullptr;
    void* C = nullptr;
    cudaDataType_t c_type = CUDA_R_16F;
    int ldc = 0;
    long long stride_c = 0;
    int batch = 1;
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    cudaStream_t stream = nullptr;
};

// loc stays a separate trailing parameter: its default names the caller's file:line.
[[nodiscard]] cublasStatus_t lt_gemm_strided_batched(
    const LtGemmArgs& a, const std::source_location& loc = std::source_location::current());

// lt_gemm_ex's operands: strides 0 and batch 1 whatever the caller set.
[[nodiscard]] constexpr LtGemmArgs lt_gemm_ex_args(LtGemmArgs a) {
    a.stride_a = a.stride_b = a.stride_c = 0;
    a.batch = 1;
    return a;
}

[[nodiscard]] inline cublasStatus_t lt_gemm_ex(
    const LtGemmArgs& a, const std::source_location& loc = std::source_location::current()) {
    return lt_gemm_strided_batched(lt_gemm_ex_args(a), loc);
}

// Arguments of lt_gemm_batched (audit D-11), the old parameter list in order. Pointer-array batch:
// a / b / c are device arrays of batch matrix pointers.
struct LtGemmBatchedArgs {
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    const void* alpha = nullptr;
    const void* const* a = nullptr;
    cudaDataType_t a_type = CUDA_R_16F;
    int lda = 0;
    const void* const* b = nullptr;
    cudaDataType_t b_type = CUDA_R_16F;
    int ldb = 0;
    const void* beta = nullptr;
    void* const* c = nullptr;
    cudaDataType_t c_type = CUDA_R_16F;
    int ldc = 0;
    int batch = 1;
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    cudaStream_t stream = nullptr;
};

[[nodiscard]] cublasStatus_t lt_gemm_batched(
    const LtGemmBatchedArgs& args, const std::source_location& loc = std::source_location::current());

// Destroys the cached descriptors (gemm_cleanup()).
void lt_gemm_ex_cleanup();

}  // namespace imp
