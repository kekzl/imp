#include "compute/gemm_grouped.h"
#include "compute/gemm_lt_ex.h"
#include "core/cuda_errors.h"
#include "core/cublas_check.h"
#include "core/logging.h"
#include "memory/device_alloc_all.h"
#include "memory/device_owned.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <utility>

namespace imp {


// ---------------------------------------------------------------------------
// Helper: map QType -> cudaDataType
// ---------------------------------------------------------------------------
static cudaDataType_t dtype_to_cuda(QType dt) {
    switch (dt) {
        case QType::F32:
            return CUDA_R_32F;
        case QType::F16:
            return CUDA_R_16F;
        case QType::BF16:
            return CUDA_R_16BF;
        case QType::FP8_E4M3:
            return CUDA_R_8F_E4M3;
        case QType::FP8_E5M2:
            return CUDA_R_8F_E5M2;
        case QType::INT8:
            return CUDA_R_8I;
        case QType::INT32:
            return CUDA_R_32I;
        default:
            IMP_LOG_ERROR("imp::gemm_grouped: unsupported dtype %d", std::to_underlying(dt));
            abort();
    }
}

// ---------------------------------------------------------------------------
// Helper: choose cuBLAS compute type for a given operand dtype
// ---------------------------------------------------------------------------
static cublasComputeType_t dtype_to_compute(QType dt) {
    switch (dt) {
        case QType::F32:
            return CUBLAS_COMPUTE_32F;
        case QType::F16:
            return CUBLAS_COMPUTE_32F;
        case QType::BF16:
            return CUBLAS_COMPUTE_32F;
        case QType::FP8_E4M3:
            return CUBLAS_COMPUTE_32F;
        case QType::FP8_E5M2:
            return CUBLAS_COMPUTE_32F;
        case QType::INT8:
            return CUBLAS_COMPUTE_32I;
        default:
            return CUBLAS_COMPUTE_32F;
    }
}

// gemm_moe_batched: one pointer-array GEMM batch per distinct expert M (typically 2-5 calls
// instead of one per expert); all experts share K, N.
void gemm_moe_batched(const void* a_base, void* c_base, const int32_t* offsets, const void* const* b_ptrs,
                      int K, int N, QType dtype, int n_experts, cudaStream_t stream, void** d_work_ptrs,
                      QType output_dtype, const float* a_scales, const float* b_scales) {
    // a_scales/b_scales: optional host arrays [n_experts] for per-expert FP8 scaling.
    // When provided, alpha for expert e = a_scales[e] * b_scales[e] (or just a_scales[e]
    // if b_scales is null). This gives per-expert de-scaling for FP8 GEMMs.
    (void)b_scales;  // reserved for future cuBLASLt per-expert B scale support
    if (n_experts == 0)
        return;

    // All experts run on cuBLASLt below: lt_gemm_ex per single-expert group, lt_gemm_batched per
    // group of equal M. The CUTLASS 2.x GemmGrouped wrapper it replaced was removed: cuBLAS grouped
    // was faster on prefill and equivalent on decode on sm_120 (CUDA 13.2+).

    cudaDataType_t cuda_dt_ab = dtype_to_cuda(dtype);
    cublasComputeType_t compute_type = dtype_to_compute(dtype);
    size_t elem_sz = dtype_size(dtype);

    // Output type: defaults to input type if not specified
    QType out_dt = (output_dtype != QType(255)) ? output_dtype : dtype;
    cudaDataType_t cuda_dt_c = dtype_to_cuda(out_dt);
    size_t out_elem_sz = dtype_size(out_dt);

    float beta = 0.0f;

    const char* a_bytes = static_cast<const char*>(a_base);
    char* c_bytes = static_cast<char*>(c_base);

    // When per-expert scales are provided (FP8 path), each expert needs its own
    // alpha = a_scale * b_scale, so we make each active expert its own group.
    // Otherwise, we group experts by M for fewer cuBLAS groups.
    const bool has_per_expert_scales = (a_scales != nullptr);

    // Count active experts and group by M (token count).
    // Build flat host pointer arrays sorted by group for efficient upload.
    int n_active = 0;
    struct GroupInfo {
        int M;
        int start;
        int count;
        float alpha;
    };
    std::vector<GroupInfo> groups;
    // Per-expert alpha values (only used when has_per_expert_scales)
    std::vector<float> expert_alphas;
    if (has_per_expert_scales)
        expert_alphas.reserve(n_experts);

    // First pass: count active experts and identify groups
    for (int e = 0; e < n_experts; e++) {
        int count = offsets[e + 1] - offsets[e];
        if (count == 0)
            continue;
        n_active++;

        if (has_per_expert_scales) {
            // Each active expert is its own group for per-expert alpha
            float a_alpha = a_scales[e] * (b_scales ? b_scales[e] : 1.0f);
            expert_alphas.push_back(a_alpha);
            groups.push_back({count, 0, 1, a_alpha});
        } else {
            bool found = false;
            for (auto& g : groups) {
                if (g.M == count) {
                    g.count++;
                    found = true;
                    break;
                }
            }
            if (!found)
                groups.push_back({count, 0, 1, 1.0f});
        }
    }

    if (n_active == 0)
        return;

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
            if (count == 0)
                continue;
            int start = offsets[e];
            h_A[active_idx] = a_bytes + (static_cast<size_t>(start) * K * elem_sz);
            h_B[active_idx] = b_ptrs[e];
            h_C[active_idx] = c_bytes + (static_cast<size_t>(start) * N * out_elem_sz);
            groups[active_idx].count = 1;
            active_idx++;
        }
    } else {
        for (int e = 0; e < n_experts; e++) {
            int count = offsets[e + 1] - offsets[e];
            if (count == 0)
                continue;

            // Find this expert's group
            for (auto& g : groups) {
                if (g.M == count) {
                    int idx = g.start + g.count;
                    g.count++;
                    int start = offsets[e];
                    h_A[idx] = a_bytes + (static_cast<size_t>(start) * K * elem_sz);
                    h_B[idx] = b_ptrs[e];
                    h_C[idx] = c_bytes + (static_cast<size_t>(start) * N * out_elem_sz);
                    break;
                }
            }
        }
    }

    // Device pointer arrays: use pre-allocated or allocate once
    void** d_A_ptrs = nullptr;
    void** d_B_ptrs = nullptr;
    void** d_C_ptrs = nullptr;
    DeviceOwned<void*> a_own, b_own, c_own;  // hold the arrays only when allocated here

    if (d_work_ptrs && n_active <= n_experts) {
        // Use pre-allocated device memory: [A..., B..., C...]
        d_A_ptrs = d_work_ptrs;
        d_B_ptrs = d_work_ptrs + n_experts;
        d_C_ptrs = d_work_ptrs + (static_cast<ptrdiff_t>(2) * n_experts);
    } else {
        // Allocate once for all groups (not per-group)
        size_t ptr_bytes = n_active * sizeof(void*);
        cuda_alloc_or_throw(device_alloc_all_async(stream, dev_req(d_A_ptrs, ptr_bytes),
                                                   dev_req(d_B_ptrs, ptr_bytes),
                                                   dev_req(d_C_ptrs, ptr_bytes)),
                            "gemm_moe_batched pointer arrays");
        a_own.adopt_async(d_A_ptrs, ptr_bytes, stream);
        b_own.adopt_async(d_B_ptrs, ptr_bytes, stream);
        c_own.adopt_async(d_C_ptrs, ptr_bytes, stream);
    }

    // Single upload of all pointer arrays
    size_t active_bytes = n_active * sizeof(void*);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<void*>(d_A_ptrs), static_cast<const void*>(h_A.data()),
                                       active_bytes, cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<void*>(d_B_ptrs), static_cast<const void*>(h_B.data()),
                                       active_bytes, cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<void*>(d_C_ptrs), static_cast<const void*>(h_C.data()),
                                       active_bytes, cudaMemcpyHostToDevice, stream));

    // One pointer-array cuBLASLt batch per M group (cuBLASLt has no grouped algo on sm_120,
    // docs/internals/SM120.md); a single expert takes its host pointers and exact alignment.
    for (const auto& g : groups) {
        float g_alpha = g.alpha;
        cublasStatus_t st = g.count == 1 ? lt_gemm_ex({.transa = CUBLAS_OP_T,
                                                       .transb = CUBLAS_OP_N,
                                                       .m = N,
                                                       .n = g.M,
                                                       .k = K,
                                                       .alpha = &g_alpha,
                                                       .A = h_B[g.start],
                                                       .a_type = cuda_dt_ab,
                                                       .lda = K,
                                                       .B = h_A[g.start],
                                                       .b_type = cuda_dt_ab,
                                                       .ldb = K,
                                                       .beta = &beta,
                                                       .C = h_C[g.start],
                                                       .c_type = cuda_dt_c,
                                                       .ldc = N,
                                                       .compute = compute_type,
                                                       .stream = stream})
                                         : lt_gemm_batched({.transa = CUBLAS_OP_T,
                                                            .transb = CUBLAS_OP_N,
                                                            .m = N,
                                                            .n = g.M,
                                                            .k = K,
                                                            .alpha = &g_alpha,
                                                            .a = d_B_ptrs + g.start,
                                                            .a_type = cuda_dt_ab,
                                                            .lda = K,
                                                            .b = d_A_ptrs + g.start,
                                                            .b_type = cuda_dt_ab,
                                                            .ldb = K,
                                                            .beta = &beta,
                                                            .c = d_C_ptrs + g.start,
                                                            .c_type = cuda_dt_c,
                                                            .ldc = N,
                                                            .batch = g.count,
                                                            .compute = compute_type,
                                                            .stream = stream});
        if (st != CUBLAS_STATUS_SUCCESS)
            IMP_LOG_ERROR("imp::gemm_moe_batched: cuBLASLt failed (M=%d, experts=%d, K=%d, N=%d, status %d)",
                          g.M, g.count, K, N, (int)st);
    }

    a_own.reset();
    b_own.reset();
    c_own.reset();
}

}  // namespace imp
