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

// ---------------------------------------------------------------------------
// gemm_moe_batched: single cublasGemmBatchedEx call for all active MoE experts.
//
// Groups all experts with non-zero token counts into one batched call.
// This eliminates per-expert kernel launch overhead (critical for 128-expert
// models like Qwen3-Coder where serial dispatch causes 9,216 launches/pass).
//
// All experts share the same K and N dimensions (same weight shape), but have
// different M (token count). cublasGemmBatchedEx requires uniform M across
// batches, so we group experts by M and issue one call per unique M value.
// In practice during prefill, most experts have similar M (tokens are spread
// roughly evenly), so this is typically 2-5 calls instead of 128.
// ---------------------------------------------------------------------------
void gemm_moe_batched(const void* a_base, void* c_base,
                      const int32_t* offsets,
                      const void* const* b_ptrs,
                      int K, int N, DType dtype,
                      int n_experts,
                      cudaStream_t stream,
                      void** d_work_ptrs,
                      DType output_dtype)
{
    if (n_experts == 0) return;

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cudaDataType_t cuda_dt_ab = dtype_to_cuda(dtype);
    cublasComputeType_t compute_type = dtype_to_compute(dtype);
    size_t elem_sz = dtype_size(dtype);

    // Output type: defaults to input type if not specified
    DType out_dt = (output_dtype != DType(255)) ? output_dtype : dtype;
    cudaDataType_t cuda_dt_c = dtype_to_cuda(out_dt);
    size_t out_elem_sz = dtype_size(out_dt);

    float alpha = 1.0f;
    float beta  = 0.0f;

    const char* a_bytes = static_cast<const char*>(a_base);
    char* c_bytes = static_cast<char*>(c_base);

    // Count active experts and group by M (token count).
    // Build flat host pointer arrays sorted by group for efficient upload.
    int n_active = 0;
    struct GroupInfo { int M; int start; int count; };
    std::vector<GroupInfo> groups;

    // First pass: count active experts and identify groups
    for (int e = 0; e < n_experts; e++) {
        int count = offsets[e + 1] - offsets[e];
        if (count == 0) continue;
        n_active++;
        bool found = false;
        for (auto& g : groups) {
            if (g.M == count) { g.count++; found = true; break; }
        }
        if (!found) groups.push_back({count, 0, 1});
    }

    if (n_active == 0) return;

    // Compute group start offsets (prefix sum)
    int offset_acc = 0;
    for (auto& g : groups) {
        g.start = offset_acc;
        offset_acc += g.count;
        g.count = 0;  // reset for second pass
    }

    // Second pass: build flat pointer arrays in group order
    std::vector<const void*> h_A(n_active);
    std::vector<const void*> h_B(n_active);
    std::vector<void*> h_C(n_active);

    for (int e = 0; e < n_experts; e++) {
        int count = offsets[e + 1] - offsets[e];
        if (count == 0) continue;

        // Find this expert's group
        for (auto& g : groups) {
            if (g.M == count) {
                int idx = g.start + g.count;
                g.count++;
                int start = offsets[e];
                h_A[idx] = a_bytes + static_cast<size_t>(start) * K * elem_sz;
                h_B[idx] = b_ptrs[e];
                h_C[idx] = c_bytes + static_cast<size_t>(start) * N * out_elem_sz;
                break;
            }
        }
    }

    // Device pointer arrays: use pre-allocated or allocate once
    void** d_A_ptrs;
    void** d_B_ptrs;
    void** d_C_ptrs;
    bool owns_ptrs = false;

    if (d_work_ptrs && n_active <= n_experts) {
        // Use pre-allocated device memory: [A..., B..., C...]
        d_A_ptrs = d_work_ptrs;
        d_B_ptrs = d_work_ptrs + n_experts;
        d_C_ptrs = d_work_ptrs + 2 * n_experts;
    } else {
        // Allocate once for all groups (not per-group)
        size_t ptr_bytes = n_active * sizeof(void*);
        cudaMallocAsync(&d_A_ptrs, ptr_bytes, stream);
        cudaMallocAsync(&d_B_ptrs, ptr_bytes, stream);
        cudaMallocAsync(&d_C_ptrs, ptr_bytes, stream);
        owns_ptrs = true;
    }

    // Single upload of all pointer arrays
    size_t active_bytes = n_active * sizeof(void*);
    cudaMemcpyAsync(d_A_ptrs, h_A.data(), active_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_ptrs, h_B.data(), active_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_ptrs, h_C.data(), active_bytes, cudaMemcpyHostToDevice, stream);

    // Use cublasGemmGroupedBatchedEx: single cuBLAS call for ALL groups.
    // This eliminates per-group launch overhead (critical for 61+ groups with 128 experts).
    {
        int group_count = static_cast<int>(groups.size());

        // Per-group descriptor arrays (host)
        std::vector<cublasOperation_t> transa_arr(group_count, CUBLAS_OP_T);
        std::vector<cublasOperation_t> transb_arr(group_count, CUBLAS_OP_N);
        std::vector<int> m_arr(group_count);    // cuBLAS m = N (col-major trick: result cols)
        std::vector<int> n_arr(group_count);    // cuBLAS n = M (result rows)
        std::vector<int> k_arr(group_count);
        std::vector<int> lda_arr(group_count);  // B leading dim
        std::vector<int> ldb_arr(group_count);  // A leading dim
        std::vector<int> ldc_arr(group_count);  // C leading dim
        std::vector<int> group_size_arr(group_count);
        std::vector<float> alpha_arr(group_count, 1.0f);
        std::vector<float> beta_arr(group_count, 0.0f);

        for (int gi = 0; gi < group_count; gi++) {
            m_arr[gi] = N;              // output columns (row-major N)
            n_arr[gi] = groups[gi].M;   // output rows (token count for this group)
            k_arr[gi] = K;
            lda_arr[gi] = K;            // B stride (weight matrix [N,K], col-major [K,N])
            ldb_arr[gi] = K;            // A stride (activation [M,K], col-major [K,M])
            ldc_arr[gi] = N;            // C stride (output [M,N], col-major [N,M])
            group_size_arr[gi] = groups[gi].count;
        }

        cublasStatus_t st = cublasGemmGroupedBatchedEx(
            handle,
            transa_arr.data(), transb_arr.data(),
            m_arr.data(), n_arr.data(), k_arr.data(),
            alpha_arr.data(),
            (const void* const*)d_B_ptrs, cuda_dt_ab, lda_arr.data(),
            (const void* const*)d_A_ptrs, cuda_dt_ab, ldb_arr.data(),
            beta_arr.data(),
            (void* const*)d_C_ptrs, cuda_dt_c, ldc_arr.data(),
            group_count,
            group_size_arr.data(),
            compute_type);

        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm_moe_batched: cublasGemmGroupedBatchedEx failed "
                            "(groups=%d, active=%d, K=%d, N=%d, status %d)\n",
                    group_count, n_active, K, N, (int)st);
            // Fallback: per-group cublasGemmBatchedEx
            for (const auto& g : groups) {
                cublasGemmBatchedEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, g.M, K, &alpha,
                    (const void**)(d_B_ptrs + g.start), cuda_dt_ab, K,
                    (const void**)(d_A_ptrs + g.start), cuda_dt_ab, K,
                    &beta,
                    (void**)(d_C_ptrs + g.start), cuda_dt_c, N,
                    g.count, compute_type, CUBLAS_GEMM_DEFAULT);
            }
        }
    }

    if (owns_ptrs) {
        cudaFreeAsync(d_A_ptrs, stream);
        cudaFreeAsync(d_B_ptrs, stream);
        cudaFreeAsync(d_C_ptrs, stream);
    }
}

} // namespace imp
