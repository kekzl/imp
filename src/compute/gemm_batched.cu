#include "compute/gemm.h"
#include "compute/gemm_internal.cuh"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace imp {

// Bridge the file-local cuBLAS internals (defined in gemm.cu) to the original
// symbol names so the moved function bodies stay byte-identical.
static constexpr auto kGemmAlgo = CUBLAS_GEMM_AUTOTUNE;

static inline cublasHandle_t get_cublas_handle() { return gemm_internal_cublas_handle(); }
static inline cublasLtHandle_t get_cublaslt_handle() { return gemm_internal_cublaslt_handle(); }
static inline cudaDataType_t dtype_to_cuda(QType dt) { return gemm_internal_dtype_to_cuda(dt); }

// ---------------------------------------------------------------------------
// Batched K/V projection via cublasGemmStridedBatchedEx
// ---------------------------------------------------------------------------

void gemm_kv_batched(const Tensor& input, const Tensor& weight_kv, Tensor& k_out, Tensor& v_out,
                     cudaStream_t stream) {
    int M = static_cast<int>(input.shape[0]);  // n_tokens
    int K = static_cast<int>(input.shape[1]);  // d_model
    int N = static_cast<int>(k_out.shape[1]);  // nkv * hd

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cudaDataType_t dt = dtype_to_cuda(input.qtype);
    float alpha = 1.0f, beta = 0.0f;

    // Col-major interpretation (same trick as gemm()):
    //   weight [N,K] row-major = [K,N] col-major; CUBLAS_OP_T → [N,K]
    //   input  [M,K] row-major = [K,M] col-major; CUBLAS_OP_N
    //   result [N,M] col-major = [M,N] row-major
    long long weight_stride = static_cast<long long>(N) * K;  // stride between wk and wv in weight_kv
    // strideC: derive from the ACTUAL pointer distance between the two output
    // views, like gemm_pair_batched. The old `M*N` only matched the workspace
    // layout when M == the buffer's max_tokens (the engine maintains that via
    // resize_workspace, the raw-executor path does not) — for M < max_tokens
    // the V batch landed inside the K buffer and v_out stayed stale (#677:
    // first-forward V was silently zero/garbage).
    long long output_stride = (static_cast<const char*>(v_out.data) -
                               static_cast<const char*>(k_out.data)) /
                              static_cast<long long>(dtype_size(input.qtype));
    if (output_stride < static_cast<long long>(M) * N) {
        // Outputs not laid out batched-compatibly (or overlapping): fall back
        // to two separate GEMMs rather than corrupting V.
        int64_t w_shape[2] = {N, K};
        Tensor wk(weight_kv.data, weight_kv.qtype, 2, w_shape, true);
        Tensor wv(static_cast<char*>(weight_kv.data) +
                      weight_stride * static_cast<long long>(dtype_size(weight_kv.qtype)),
                  weight_kv.qtype, 2, w_shape, true);
        gemm(input, wk, k_out, alpha, beta, stream);
        gemm(input, wv, v_out, alpha, beta, stream);
        return;
    }

    cublasStatus_t st = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M,
                                                   K,                              // cuBLAS m, n, k
                                                   &alpha, weight_kv.data, dt, K,  // A (weight), lda=K
                                                   weight_stride,                  // strideA: offset to wv
                                                   input.data, dt, K,              // B (input), ldb=K
                                                   0,  // strideB: 0 (same input for both)
                                                   &beta, k_out.data, dt, N,  // C (output), ldc=N
                                                   output_stride,             // strideC: offset to v_out
                                                   2,                         // batch_count = 2 (K and V)
                                                   CUBLAS_COMPUTE_32F, kGemmAlgo);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm_kv_batched: cublasGemmStridedBatchedEx failed (status %d)\n", (int)st);
    }
}

void gemm_pair_batched(const Tensor& input, const Tensor& weight_fused, Tensor& out1, Tensor& out2,
                       cudaStream_t stream) {
    int M = static_cast<int>(input.shape[0]);  // n_tokens
    int K = static_cast<int>(input.shape[1]);  // d_model
    int N = static_cast<int>(out1.shape[1]);   // d_ff (or nkv*hd)

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cudaDataType_t dt = dtype_to_cuda(input.qtype);
    float alpha = 1.0f, beta = 0.0f;

    long long weight_stride = static_cast<long long>(N) * K;
    // Compute actual byte offset between out1 and out2, then convert to element offset
    long long output_stride = (static_cast<const char*>(out2.data) - static_cast<const char*>(out1.data)) /
                              dtype_size(input.qtype);

    cublasStatus_t st = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                                                   weight_fused.data, dt, K, weight_stride, input.data, dt, K,
                                                   0, &beta, out1.data, dt, N, output_stride, 2,
                                                   CUBLAS_COMPUTE_32F, kGemmAlgo);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm_pair_batched: cublasGemmStridedBatchedEx failed (status %d)\n", (int)st);
    }
}

bool gemm_cublaslt_fp8_probe() {
    // Test at M=15 (non-power-of-2, common short-prompt length after chat
    // template wrapping) because cuBLAS 13.4 FP8 on sm_120 returns
    // NOT_SUPPORTED for certain M values even when M=16/32/64 work.
    constexpr int M = 15, K = 4096, N = 12288;
    void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    float *d_sa = nullptr, *d_sb = nullptr;
    if (cudaMalloc(&d_a, M * K) != cudaSuccess) return false;
    if (cudaMalloc(&d_b, N * K) != cudaSuccess) { cudaFree(d_a); return false; }
    if (cudaMalloc(&d_c, M * N * 2) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return false; }
    cudaMalloc(&d_sa, sizeof(float));
    cudaMalloc(&d_sb, sizeof(float));
    float one = 1.0f;
    cudaMemcpy(d_sa, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_a, 0, M * K);
    cudaMemset(d_b, 0, N * K);

    cublasLtHandle_t lt = get_cublaslt_handle();
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sa, sizeof(d_sa));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sb, sizeof(d_sb));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, N, M, N);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasLtMatmul(lt, opDesc, &alpha, d_b, Bdesc, d_a, Adesc, &beta,
                                        d_c, Cdesc, d_c, Cdesc, nullptr,
                                        gemm_internal_workspace(), gemm_internal_workspace_size(), nullptr);

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_sa); cudaFree(d_sb);
    cudaGetLastError();
    return st == CUBLAS_STATUS_SUCCESS;
}

}  // namespace imp
