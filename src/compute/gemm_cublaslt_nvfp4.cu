// cuBLASLt block-scaled NVFP4×NVFP4→FP16 GEMM for prefill acceleration.
//
// Uses cublasLtMatmul with CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 block-scale
// mode for native Blackwell (sm_120) FP4 tensor core acceleration.
//
// Scale factor layout (VEC16_UE4M3) is identical to the CUTLASS SfAtom format
// already used by convert_nvfp4_to_cutlass() and quantize_fp16_to_nvfp4_cutlass().
// No additional weight conversion is needed — CutlassNvFP4Weight is used directly.
//
// tensor_scale is applied as the GEMM alpha parameter (same approach as the
// CUTLASS path) to avoid precision loss from absorbing it into UE4M3 range.

#include "compute/gemm_cublaslt_nvfp4.h"
#include "core/logging.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <mutex>

// CUDA_R_4F_E2M1 and VEC16_UE4M3 block-scale enums were added in CUDA 12.8.
// Use CUDART_VERSION to guard since these are enum values (not preprocessor macros).
#define IMP_HAS_CUBLASLT_NVFP4 (CUDART_VERSION >= 12080)

namespace imp {

#if IMP_HAS_CUBLASLT_NVFP4

// Guard for one-time availability probe
static std::once_flag s_probe_flag;
static bool s_available = false;

// Persistent cuBLASLt handle (created once)
static cublasLtHandle_t s_lt_handle = nullptr;

// Persistent workspace (grows as needed, never freed)
static void* s_workspace = nullptr;
static size_t s_workspace_size = 0;

static cublasLtHandle_t get_handle() {
    if (!s_lt_handle) {
        cublasStatus_t st = cublasLtCreate(&s_lt_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_ERROR("gemm_cublaslt_nvfp4: cublasLtCreate failed (%d)", (int)st);
            s_lt_handle = nullptr;
        }
    }
    return s_lt_handle;
}

// Probe whether cuBLASLt NVFP4 GEMM actually works on this GPU by running a
// heuristic query.  This catches runtime issues (driver too old, GPU not
// Blackwell, cuBLAS build without FP4 support).
static void probe_availability() {
    cublasLtHandle_t lt = get_handle();
    if (!lt) { s_available = false; return; }

    // Probe: M=128, N=128, K=128 — realistic dimensions for FP4 GEMM.
    // FP4 tensor cores require CUBLAS_OP_N for both operands (no transpose).
    // Layout: A = weight [N,K] col-major, B = activation [K,M] col-major.
    constexpr int pM = 128, pN = 128, pK = 128;

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasStatus_t st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) { s_available = false; return; }

    // FP4 requires OP_N for both matrices (no transpose support)
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));

    // Dummy scale pointers (NULL ok for heuristic query)
    void* dummy = nullptr;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy, sizeof(dummy));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy, sizeof(dummy));

    // cuBLAS with OP_N for both:
    //   A = weight [N,K] col-major, OP_N → op(A) = [N,K]
    //   B = activation [K,M] col-major, OP_N → op(B) = [K,M]
    //   D = op(A) × op(B) = [N,K]×[K,M] = [N,M] col-major = [M,N] row-major
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, pN, pK, pN);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, pK, pM, pK);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, pN, pM, pN);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = 4 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t result = {};
    int n_result = 0;
    st = cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                         pref, 1, &result, &n_result);

    s_available = (st == CUBLAS_STATUS_SUCCESS && n_result > 0);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);

    if (s_available) {
        IMP_LOG_INFO("cuBLASLt NVFP4 GEMM: available (probe passed, OP_N layout)");
    } else {
        IMP_LOG_INFO("cuBLASLt NVFP4 GEMM: not available (probe: status=%d n_result=%d)"
                     " — using CUTLASS NVFP4 path instead",
                     (int)st, n_result);
    }
}

bool cublaslt_nvfp4_available() {
    std::call_once(s_probe_flag, probe_availability);
    return s_available;
}

bool gemm_nvfp4_cublaslt(const void* a_data, const void* a_sf,
                          const CutlassNvFP4Weight& b,
                          void* d_fp16, int M, int N, int K,
                          cudaStream_t stream)
{
    cublasLtHandle_t lt = get_handle();
    if (!lt) return false;

    // --- Create operation descriptor ---
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasStatus_t st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) return false;

    // FP4 requires OP_N for both matrices (transpose not supported on FP4 tensor cores).
    // Weight must be pre-transposed to col-major [N,K] layout.
    //   cuBLAS "A" = weight  [N,K] col-major, OP_N → op(A)=[N,K]
    //   cuBLAS "B" = activation [K,M] col-major (= [M,K] row-major), OP_N → op(B)=[K,M]
    //   D = op(A) × op(B) = [N,K]×[K,M] = [N,M] col-major = [M,N] row-major
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // Block-scale mode: VEC16_UE4M3 (1 unsigned E4M3 scale per 16 FP4 elements)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));

    // Scale pointers (device pointers to SfAtom layout UE4M3 tensors):
    //   A_SCALE = weight scales (b.scale_factors)
    //   B_SCALE = activation scales (a_sf)
    const void* a_scale_ptr = b.scale_factors;
    const void* b_scale_ptr = a_sf;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                    &a_scale_ptr, sizeof(a_scale_ptr));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                    &b_scale_ptr, sizeof(b_scale_ptr));

    // --- Matrix layouts (OP_N for both, weight must be col-major [N,K]) ---
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, N, K, N);   // weight  [N,K] col-major
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, M, K);   // activation [K,M] col-major
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, N, M, N);        // C = D buffer (beta=0)
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16F, N, M, N);        // output [N,M] col-major

    // --- Algorithm heuristic ---
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t max_ws = 4 * 1024 * 1024;  // 4 MiB workspace limit
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &max_ws, sizeof(max_ws));

    cublasLtMatmulHeuristicResult_t result = {};
    int n_result = 0;
    st = cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                         pref, 1, &result, &n_result);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != CUBLAS_STATUS_SUCCESS || n_result == 0) {
        IMP_LOG_WARN("cuBLASLt NVFP4: no algorithm for M=%d N=%d K=%d (status=%d)",
                     M, N, K, (int)st);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatrixLayoutDestroy(Ddesc);
        cublasLtMatmulDescDestroy(opDesc);
        return false;
    }

    // --- Workspace ---
    if (result.workspaceSize > s_workspace_size) {
        if (s_workspace) cudaFree(s_workspace);
        cudaError_t err = cudaMalloc(&s_workspace, result.workspaceSize);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("cuBLASLt NVFP4: workspace alloc failed (%zu bytes): %s",
                          result.workspaceSize, cudaGetErrorString(err));
            s_workspace = nullptr;
            s_workspace_size = 0;
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            cublasLtMatrixLayoutDestroy(Ddesc);
            cublasLtMatmulDescDestroy(opDesc);
            return false;
        }
        s_workspace_size = result.workspaceSize;
    }

    // --- Execute GEMM ---
    // D = alpha * op(A) * op(B) + beta * C
    // alpha = tensor_scale (compensates deferred global scale from weight quantization)
    float alpha = b.tensor_scale;
    float beta = 0.0f;

    st = cublasLtMatmul(lt, opDesc,
                         &alpha,
                         b.data, Adesc,        // cuBLAS A = weight
                         a_data, Bdesc,         // cuBLAS B = activation
                         &beta,
                         d_fp16, Cdesc,         // C = D buffer (beta=0, never read)
                         d_fp16, Ddesc,         // D = output
                         &result.algo,
                         s_workspace, s_workspace_size,
                         stream);

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(opDesc);

    if (st != CUBLAS_STATUS_SUCCESS) {
        IMP_LOG_WARN("cuBLASLt NVFP4: matmul failed (status=%d) M=%d N=%d K=%d",
                     (int)st, M, N, K);
        return false;
    }

    return true;
}

#else // CUDART_VERSION < 12080

bool cublaslt_nvfp4_available() { return false; }

bool gemm_nvfp4_cublaslt(const void*, const void*,
                          const CutlassNvFP4Weight&,
                          void*, int, int, int,
                          cudaStream_t) {
    return false;
}

#endif // IMP_HAS_CUBLASLT_NVFP4

} // namespace imp
