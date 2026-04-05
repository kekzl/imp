#include "compute/gemm_grouped.h"
#include "core/logging.h"

#include "compute/gemm_cutlass_grouped_sm120.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace imp {

// cuBLAS 13.1 autotune: benchmarks algorithms internally on first call
// and caches the winner. Falls back to default on older toolkits.
#if IMP_CUDA_13_1
static constexpr auto kGemmAlgo = CUBLAS_GEMM_AUTOTUNE;
#else
static constexpr auto kGemmAlgo = CUBLAS_GEMM_DEFAULT;
#endif

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
            compute_type, kGemmAlgo);
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
            compute_type, kGemmAlgo);
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
                      DType output_dtype,
                      const float* a_scales,
                      const float* b_scales)
{
    // a_scales/b_scales: optional host arrays [n_experts] for per-expert FP8 scaling.
    // When provided, alpha for expert e = a_scales[e] * b_scales[e] (or just a_scales[e]
    // if b_scales is null). This gives per-expert de-scaling for FP8 GEMMs.
    (void)b_scales;  // reserved for future cuBLASLt per-expert B scale support
    if (n_experts == 0) return;

    // -----------------------------------------------------------------------
    // CUTLASS grouped GEMM: single persistent kernel for all experts.
    // Only applies to FP16 without per-expert scales (non-FP8 path).
    // -----------------------------------------------------------------------
    if (cutlass_grouped_gemm_sm120_available() && dtype == DType::FP16 &&
        !a_scales && (output_dtype == DType(255) || output_dtype == DType::FP16))
    {
        // Count active experts and build host pointer/M arrays
        int n_active = 0;
        for (int e = 0; e < n_experts; e++) {
            if (offsets[e + 1] > offsets[e]) n_active++;
        }

        if (n_active > 0) {
            size_t elem_sz_fp16 = sizeof(half);
            std::vector<const void*> h_A(n_active);
            std::vector<const void*> h_B(n_active);
            std::vector<void*>       h_C(n_active);
            std::vector<int>         h_M(n_active);

            const char* a_bytes = static_cast<const char*>(a_base);
            char* c_bytes = static_cast<char*>(c_base);
            int gi = 0;
            for (int e = 0; e < n_experts; e++) {
                int count = offsets[e + 1] - offsets[e];
                if (count <= 0) continue;
                h_A[gi] = a_bytes + static_cast<size_t>(offsets[e]) * K * elem_sz_fp16;
                h_B[gi] = b_ptrs[e];
                h_C[gi] = c_bytes + static_cast<size_t>(offsets[e]) * N * elem_sz_fp16;
                h_M[gi] = count;
                gi++;
            }

            // Upload pointer arrays and M values to device
            // Use pre-allocated d_work_ptrs if available, else allocate
            void** d_A_ptrs_cu;
            void** d_B_ptrs_cu;
            void** d_C_ptrs_cu;
            int*   d_M_values;
            bool owns_cu_ptrs = false;

            size_t ptr_bytes = n_active * sizeof(void*);
            size_t m_bytes = n_active * sizeof(int);

            if (d_work_ptrs && n_active <= n_experts) {
                // Reuse pre-allocated workspace: [A..., B..., C..., M...]
                // d_work_ptrs has space for 3 * n_experts void* entries.
                // We borrow additional space for M values after the 3 pointer arrays.
                d_A_ptrs_cu = d_work_ptrs;
                d_B_ptrs_cu = d_work_ptrs + n_experts;
                d_C_ptrs_cu = d_work_ptrs + 2 * n_experts;
                // M values: allocate separately (d_work_ptrs may not have extra room)
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_M_values, m_bytes, stream));
                owns_cu_ptrs = true;  // only owns d_M_values
            } else {
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(reinterpret_cast<void**>(&d_A_ptrs_cu), ptr_bytes, stream));
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(reinterpret_cast<void**>(&d_B_ptrs_cu), ptr_bytes, stream));
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(reinterpret_cast<void**>(&d_C_ptrs_cu), ptr_bytes, stream));
                IMP_CUDA_CHECK_LOG(cudaMallocAsync(reinterpret_cast<void**>(&d_M_values), m_bytes, stream));
                owns_cu_ptrs = true;
            }

            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_A_ptrs_cu, h_A.data(), ptr_bytes, cudaMemcpyHostToDevice, stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_B_ptrs_cu, h_B.data(), ptr_bytes, cudaMemcpyHostToDevice, stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_C_ptrs_cu, h_C.data(), ptr_bytes, cudaMemcpyHostToDevice, stream));
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_M_values, h_M.data(), m_bytes, cudaMemcpyHostToDevice, stream));

            bool ok = gemm_grouped_cutlass_sm120(
                reinterpret_cast<const void* const*>(d_A_ptrs_cu),
                reinterpret_cast<const void* const*>(d_B_ptrs_cu),
                reinterpret_cast<void* const*>(d_C_ptrs_cu),
                d_M_values,
                N, K, n_active,
                nullptr, 0,  // workspace auto-managed internally
                stream);

            // Cleanup async allocations
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_M_values, stream));
            if (owns_cu_ptrs && !d_work_ptrs) {
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_A_ptrs_cu, stream));
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_B_ptrs_cu, stream));
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_C_ptrs_cu, stream));
            }

            if (ok) return;  // Success — skip cuBLAS path
            // Fall through to cuBLAS on failure
        }
    }

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cudaDataType_t cuda_dt_ab = dtype_to_cuda(dtype);
    cublasComputeType_t compute_type = dtype_to_compute(dtype);
    size_t elem_sz = dtype_size(dtype);

    // Output type: defaults to input type if not specified
    DType out_dt = (output_dtype != DType(255)) ? output_dtype : dtype;
    cudaDataType_t cuda_dt_c = dtype_to_cuda(out_dt);
    size_t out_elem_sz = dtype_size(out_dt);

    float beta  = 0.0f;

    const char* a_bytes = static_cast<const char*>(a_base);
    char* c_bytes = static_cast<char*>(c_base);

    // When per-expert scales are provided (FP8 path), each expert needs its own
    // alpha = a_scale * b_scale, so we make each active expert its own group.
    // Otherwise, we group experts by M for fewer cuBLAS groups.
    const bool has_per_expert_scales = (a_scales != nullptr);

    // Count active experts and group by M (token count).
    // Build flat host pointer arrays sorted by group for efficient upload.
    int n_active = 0;
    struct GroupInfo { int M; int start; int count; float alpha; };
    std::vector<GroupInfo> groups;
    // Per-expert alpha values (only used when has_per_expert_scales)
    std::vector<float> expert_alphas;
    if (has_per_expert_scales) expert_alphas.reserve(n_experts);

    // First pass: count active experts and identify groups
    for (int e = 0; e < n_experts; e++) {
        int count = offsets[e + 1] - offsets[e];
        if (count == 0) continue;
        n_active++;

        if (has_per_expert_scales) {
            // Each active expert is its own group for per-expert alpha
            float a_alpha = a_scales[e] * (b_scales ? b_scales[e] : 1.0f);
            expert_alphas.push_back(a_alpha);
            groups.push_back({count, 0, 1, a_alpha});
        } else {
            bool found = false;
            for (auto& g : groups) {
                if (g.M == count) { g.count++; found = true; break; }
            }
            if (!found) groups.push_back({count, 0, 1, 1.0f});
        }
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

    if (has_per_expert_scales) {
        // Each active expert is its own group (1:1 mapping, already in order)
        int active_idx = 0;
        for (int e = 0; e < n_experts; e++) {
            int count = offsets[e + 1] - offsets[e];
            if (count == 0) continue;
            int start = offsets[e];
            h_A[active_idx] = a_bytes + static_cast<size_t>(start) * K * elem_sz;
            h_B[active_idx] = b_ptrs[e];
            h_C[active_idx] = c_bytes + static_cast<size_t>(start) * N * out_elem_sz;
            groups[active_idx].count = 1;
            active_idx++;
        }
    } else {
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
        IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_A_ptrs, ptr_bytes, stream));
        IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_B_ptrs, ptr_bytes, stream));
        IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_C_ptrs, ptr_bytes, stream));
        owns_ptrs = true;
    }

    // Single upload of all pointer arrays
    size_t active_bytes = n_active * sizeof(void*);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_A_ptrs, h_A.data(), active_bytes, cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_B_ptrs, h_B.data(), active_bytes, cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_C_ptrs, h_C.data(), active_bytes, cudaMemcpyHostToDevice, stream));

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
        std::vector<float> alpha_arr(group_count);
        std::vector<float> beta_arr(group_count, 0.0f);

        for (int gi = 0; gi < group_count; gi++) {
            m_arr[gi] = N;              // output columns (row-major N)
            n_arr[gi] = groups[gi].M;   // output rows (token count for this group)
            k_arr[gi] = K;
            lda_arr[gi] = K;            // B stride (weight matrix [N,K], col-major [K,N])
            ldb_arr[gi] = K;            // A stride (activation [M,K], col-major [K,M])
            ldc_arr[gi] = N;            // C stride (output [M,N], col-major [N,M])
            group_size_arr[gi] = groups[gi].count;
            alpha_arr[gi] = groups[gi].alpha;
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
                float g_alpha = g.alpha;
                cublasGemmBatchedEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, g.M, K, &g_alpha,
                    (const void**)(d_B_ptrs + g.start), cuda_dt_ab, K,
                    (const void**)(d_A_ptrs + g.start), cuda_dt_ab, K,
                    &beta,
                    (void**)(d_C_ptrs + g.start), cuda_dt_c, N,
                    g.count, compute_type, kGemmAlgo);
            }
        }
    }

    if (owns_ptrs) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_A_ptrs, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_B_ptrs, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_C_ptrs, stream));
    }
}

#if IMP_CUDA_13_1

// gemm_grouped_cleanup: no-op now that we route through cublasGemmGroupedBatchedEx
// instead of cublasLt per-expert loop. Kept for API compatibility (called from engine shutdown).
void gemm_grouped_cleanup() {
    // Previously freed cublasLt workspace. Now a no-op.
}

// ---------------------------------------------------------------------------
// Helper kernel: compute per-expert M values and A/C pointer arrays from
// device-side offsets. Runs 1 thread per expert.
//
// offsets:      [n_experts+1] int32 — start offsets into the gathered buffer
// a_base/c_base: base pointers for gathered input / output
// M_values:     [n_experts] int32 — per-expert token counts (M_i = offsets[i+1] - offsets[i])
// a_ptrs/c_ptrs: [n_experts] void* — per-expert pointers into a_base/c_base
// ---------------------------------------------------------------------------
__global__ void compute_expert_params_kernel(
    const int32_t* __restrict__ offsets,
    const void* a_base, void* c_base,
    int32_t* __restrict__ M_values,
    const void** __restrict__ a_ptrs,
    void** __restrict__ c_ptrs,
    int K, int N,
    size_t elem_sz, size_t out_elem_sz,
    int n_experts)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_experts) return;

    int start = offsets[e];
    int count = offsets[e + 1] - start;
    M_values[e] = count;

    const char* a_bytes = static_cast<const char*>(a_base);
    char* c_bytes = static_cast<char*>(c_base);
    a_ptrs[e] = a_bytes + static_cast<int64_t>(start) * K * elem_sz;
    c_ptrs[e] = c_bytes + static_cast<int64_t>(start) * N * out_elem_sz;
}

// ---------------------------------------------------------------------------
// gemm_moe_device_grouped: MoE GEMM dispatch with GPU-side offset computation.
// Computes per-expert M values and pointer arrays on GPU via a small kernel,
// then routes through cublasGemmGroupedBatchedEx (single call for all experts).
// Still requires one D2H sync per call (for offsets + pointers), but uses
// batched async copies to minimize sync overhead.
// ---------------------------------------------------------------------------
void gemm_moe_device_grouped(
    const void* d_a_base, void* d_c_base,
    const int32_t* d_offsets,
    const void* const* d_b_ptrs,
    int K, int N, DType dtype,
    int n_experts, int max_tokens_per_expert,
    cudaStream_t stream,
    const float* a_scales,
    const float* b_scales)
{
    (void)max_tokens_per_expert;  // no longer needed — gemm_moe_batched handles variable M
    (void)b_scales;  // reserved for future per-expert B scale support
    if (n_experts == 0) return;

    size_t elem_sz = dtype_size(dtype);

    // Output in FP16 for FP8 inputs, same dtype otherwise
    DType out_dt = (dtype == DType::FP8_E4M3 || dtype == DType::FP8_E5M2)
                       ? DType::FP16 : dtype;
    size_t out_elem_sz = dtype_size(out_dt);

    // Device buffers for per-expert params: M_values, a_ptrs, c_ptrs
    // These are small (~3 KB for 128 experts) and allocated on the stream.
    int32_t* d_M_values = nullptr;
    const void** d_a_ptrs = nullptr;
    void** d_c_ptrs = nullptr;

    IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_M_values, n_experts * sizeof(int32_t), stream));
    IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_a_ptrs, n_experts * sizeof(void*), stream));
    IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_c_ptrs, n_experts * sizeof(void*), stream));

    // Compute per-expert params on GPU (no host sync!)
    int threads = 256;
    int blocks = (n_experts + threads - 1) / threads;
    compute_expert_params_kernel<<<blocks, threads, 0, stream>>>(
        d_offsets, d_a_base, d_c_base,
        d_M_values, d_a_ptrs, d_c_ptrs,
        K, N, elem_sz, out_elem_sz, n_experts);

    // Route through cublasGemmGroupedBatchedEx instead of per-expert cublasLtMatmul loop.
    // This replaces n_experts individual launches with a single grouped call.
    // Requires a single D2H sync for offsets + pointers (batched), but eliminates
    // per-expert kernel launch overhead (critical for 128-expert models).
    {
        // Batch all D2H copies, then single sync
        std::vector<int32_t> h_M(n_experts);
        std::vector<const void*> h_a_ptrs(n_experts);
        std::vector<void*> h_c_ptrs(n_experts);
        std::vector<const void*> h_b_ptrs(n_experts);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_M.data(), d_M_values,
                         n_experts * sizeof(int32_t),
                         cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_a_ptrs.data(), d_a_ptrs,
                         n_experts * sizeof(void*),
                         cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_c_ptrs.data(), d_c_ptrs,
                         n_experts * sizeof(void*),
                         cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_b_ptrs.data(), d_b_ptrs,
                         n_experts * sizeof(void*),
                         cudaMemcpyDeviceToHost, stream));

        std::vector<float> h_a_scales;
        if (a_scales) {
            h_a_scales.resize(n_experts);
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_a_scales.data(), a_scales,
                             n_experts * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
        }

        cudaStreamSynchronize(stream);  // single sync for all D2H copies

        // Build host offset array from M values for gemm_moe_batched
        std::vector<int32_t> h_offsets(n_experts + 1);
        h_offsets[0] = 0;
        for (int e = 0; e < n_experts; e++)
            h_offsets[e + 1] = h_offsets[e] + h_M[e];

        // Route through gemm_moe_batched which uses cublasGemmGroupedBatchedEx
        // (single call for all experts, grouped by M value).
        gemm_moe_batched(d_a_base, d_c_base,
                         h_offsets.data(), h_b_ptrs.data(),
                         K, N, dtype, n_experts, stream,
                         nullptr,  // no pre-allocated work ptrs (uses internal alloc)
                         out_dt,
                         a_scales ? h_a_scales.data() : nullptr,
                         nullptr);  // b_scales not supported yet
    }

    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_M_values, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_a_ptrs, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_c_ptrs, stream));
}

#endif  // IMP_CUDA_13_1

} // namespace imp
