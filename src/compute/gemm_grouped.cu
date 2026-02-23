#include "compute/gemm_grouped.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace imp {

// ---------------------------------------------------------------------------
// cuBLASLt handle (lazily initialized, process-lifetime)
// ---------------------------------------------------------------------------
static cublasLtHandle_t get_cublaslt_handle() {
    static cublasLtHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t st = cublasLtCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm_grouped: cublasLtCreate failed (status %d)\n", (int)st);
            abort();
        }
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
// Helper: scalar type matching compute type
// ---------------------------------------------------------------------------
static cudaDataType_t compute_to_scale_type(cublasComputeType_t ct) {
    if (ct == CUBLAS_COMPUTE_32I) return CUDA_R_32I;
    return CUDA_R_32F;
}

// ---------------------------------------------------------------------------
// Run a single expert GEMM via cublasLtMatmul.
//   C_i = A_i @ B_i      (alpha=1, beta=0)
//   A_i [M_i, K],  B_i [K, N],  C_i [M_i, N]   -- row-major
//
// Same column-major trick as gemm.cu:
//   col-major: C_i^T = B_i^T @ A_i^T
//   => cublasLt(N, N, N_col=N, M_col=M_i, K_col=K,
//               B_ptr ldb=N, A_ptr lda=K, C_ptr ldc=N)
// ---------------------------------------------------------------------------
static void run_expert_matmul(cublasLtHandle_t lt_handle,
                              const Tensor& Ai, const Tensor& Bi, Tensor& Ci,
                              cudaStream_t stream) {
    const int64_t Mi = Ai.shape[0];   // tokens routed to this expert
    const int64_t K  = Ai.shape[1];
    const int64_t N  = Bi.shape[1];

    if (Mi == 0) return;  // no tokens for this expert

    cudaDataType_t cuda_dt_A = dtype_to_cuda(Ai.dtype);
    cudaDataType_t cuda_dt_B = dtype_to_cuda(Bi.dtype);
    cudaDataType_t cuda_dt_C = dtype_to_cuda(Ci.dtype);
    cublasComputeType_t compute_type = dtype_to_compute(Ai.dtype);
    cudaDataType_t scale_type = compute_to_scale_type(compute_type);

    // --- Create matmul descriptor ---
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type);

    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // --- Create matrix layouts (column-major interpretation for row-major trick) ---
    // B^T in col-major: N rows, K cols, leading dim N
    cublasLtMatrixLayout_t layout_B = nullptr;
    cublasLtMatrixLayoutCreate(&layout_B, cuda_dt_B, (uint64_t)N, (uint64_t)K, (uint64_t)N);

    // A^T in col-major: K rows, Mi cols, leading dim K
    cublasLtMatrixLayout_t layout_A = nullptr;
    cublasLtMatrixLayoutCreate(&layout_A, cuda_dt_A, (uint64_t)K, (uint64_t)Mi, (uint64_t)K);

    // C^T in col-major: N rows, Mi cols, leading dim N
    cublasLtMatrixLayout_t layout_C = nullptr;
    cublasLtMatrixLayoutCreate(&layout_C, cuda_dt_C, (uint64_t)N, (uint64_t)Mi, (uint64_t)N);

    // --- Scalars ---
    float alpha_f = 1.0f;
    float beta_f  = 0.0f;
    int32_t alpha_i = 1;
    int32_t beta_i  = 0;
    const void* alpha_ptr = (compute_type == CUBLAS_COMPUTE_32I)
                            ? static_cast<const void*>(&alpha_i)
                            : static_cast<const void*>(&alpha_f);
    const void* beta_ptr  = (compute_type == CUBLAS_COMPUTE_32I)
                            ? static_cast<const void*>(&beta_i)
                            : static_cast<const void*>(&beta_f);

    // --- Execute ---
    cublasStatus_t st = cublasLtMatmul(
        lt_handle,
        matmul_desc,
        alpha_ptr,
        Bi.data, layout_B,   // first operand  = B (N x K col-major)
        Ai.data, layout_A,   // second operand = A (K x Mi col-major)
        beta_ptr,
        Ci.data, layout_C,   // C output
        Ci.data, layout_C,   // D output (same as C for in-place beta=0)
        nullptr,              // algo (nullptr = default heuristic)
        nullptr,              // workspace
        0,                    // workspace size
        stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm_grouped: cublasLtMatmul failed for expert "
                        "(M=%lld, K=%lld, N=%lld, status %d)\n",
                (long long)Mi, (long long)K, (long long)N, (int)st);
    }

    // --- Cleanup descriptors ---
    cublasLtMatrixLayoutDestroy(layout_A);
    cublasLtMatrixLayoutDestroy(layout_B);
    cublasLtMatrixLayoutDestroy(layout_C);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// ---------------------------------------------------------------------------
// gemm_grouped: batched GEMM for MoE expert parallelism
//
// For each expert i:
//   C[i] = A[i] @ B[i]
//   A[i]: [tokens_for_expert_i, K]
//   B[i]: [K, N]
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

    cublasLtHandle_t lt_handle = get_cublaslt_handle();

#ifdef IMP_CUDA_13_1
    // -----------------------------------------------------------------------
    // Grouped GEMM path (CUDA 13.1+ with cublasLt grouped GEMM support).
    // We use cublasLtMatmulDescSetAttribute with GROUPED_GEMM attributes
    // to batch all expert GEMMs into a single API call.
    // -----------------------------------------------------------------------

    // All experts share the same K, N but may differ in M (token count).
    // We need to collect per-expert pointers and dimensions.
    const int64_t K = A[0].shape[1];
    const int64_t N = B[0].shape[1];

    cudaDataType_t cuda_dt_A = dtype_to_cuda(A[0].dtype);
    cudaDataType_t cuda_dt_B = dtype_to_cuda(B[0].dtype);
    cudaDataType_t cuda_dt_C = dtype_to_cuda(C[0].dtype);
    cublasComputeType_t compute_type = dtype_to_compute(A[0].dtype);
    cudaDataType_t scale_type = compute_to_scale_type(compute_type);

    // Collect per-expert pointers and M values.
    std::vector<const void*> A_ptrs(num_experts);
    std::vector<const void*> B_ptrs(num_experts);
    std::vector<void*> C_ptrs(num_experts);
    std::vector<int64_t> M_vals(num_experts);

    for (size_t i = 0; i < num_experts; ++i) {
        A_ptrs[i] = A[i].data;
        B_ptrs[i] = B[i].data;
        C_ptrs[i] = C[i].data;
        M_vals[i] = A[i].shape[0];
    }

    // Device arrays for batched pointers.
    const void** d_A_ptrs = nullptr;
    const void** d_B_ptrs = nullptr;
    void** d_C_ptrs = nullptr;
    cudaMalloc(&d_A_ptrs, num_experts * sizeof(void*));
    cudaMalloc(&d_B_ptrs, num_experts * sizeof(void*));
    cudaMalloc(&d_C_ptrs, num_experts * sizeof(void*));

    cudaMemcpyAsync(d_A_ptrs, A_ptrs.data(), num_experts * sizeof(void*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_ptrs, B_ptrs.data(), num_experts * sizeof(void*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_ptrs, C_ptrs.data(), num_experts * sizeof(void*),
                    cudaMemcpyHostToDevice, stream);

    // Fall back to the loop if the grouped API isn't available at link time.
    // This section is a placeholder for when the grouped GEMM extension is
    // fully supported in cublasLt.  For now, the loop below handles it.
    cudaFree(d_A_ptrs);
    cudaFree(d_B_ptrs);
    cudaFree(d_C_ptrs);

    // Until the full grouped API is linked, fall through to the loop path.
    for (size_t i = 0; i < num_experts; ++i) {
        run_expert_matmul(lt_handle, A[i], B[i], C[i], stream);
    }

#else
    // -----------------------------------------------------------------------
    // Fallback: loop of individual cublasLtMatmul calls on the same stream.
    // This still benefits from the GPU overlapping small kernels issued
    // back-to-back on one stream.
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < num_experts; ++i) {
        run_expert_matmul(lt_handle, A[i], B[i], C[i], stream);
    }
#endif  // IMP_CUDA_13_1
}

} // namespace imp
