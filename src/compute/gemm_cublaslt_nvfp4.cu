// cuBLASLt block-scaled NVFP4×NVFP4→BF16 GEMM for prefill acceleration.
//
// Uses cublasLtMatmul with CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 block-scale
// mode for native Blackwell FP4 tensor core acceleration.
//
// Scale factor layout (VEC16_UE4M3) is identical to the CUTLASS SfAtom format
// already used by convert_nvfp4_to_cutlass() and quantize_fp16_to_nvfp4_cutlass().
// No additional weight conversion is needed — CutlassNvFP4Weight is used directly.
//
// cuBLASLt FP4 GEMM outputs BF16 (FP16 not in the supported type combination
// table). A lightweight BF16→FP16 conversion kernel runs after the GEMM.
//
// tensor_scale is applied as the GEMM alpha parameter (same approach as the
// CUTLASS path) to avoid precision loss from absorbing it into UE4M3 range.
//
// NOTE: As of cuBLAS 13.2.1.1, FP4 GEMM kernels are only compiled for sm_100
// (data center Blackwell). sm_120 (consumer Blackwell / RTX 5090) returns
// CUBLAS_STATUS_INVALID_VALUE from the heuristic query. The CUTLASS NVFP4 path
// is the primary path for sm_120 (compiled from source). This cuBLASLt path
// will activate automatically when NVIDIA adds sm_120 support.

#include "compute/gemm_cublaslt_nvfp4.h"
#include "core/logging.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// Persistent BF16 temp buffer for GEMM output (reused, grows as needed)
static void* s_bf16_buf = nullptr;
static size_t s_bf16_buf_size = 0;

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

// BF16→FP16 conversion kernel
__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                     half* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(__bfloat162float(in[idx]));
}

// Probe whether cuBLASLt NVFP4 GEMM actually works on this GPU by running a
// heuristic query. This catches runtime issues (no FP4 kernels for this SM,
// driver too old, or cuBLAS build without FP4 support).
//
// Probe config matches NVIDIA's LtNvfp4Matmul sample:
//   OP_T for weight (A), OP_N for activation (B)
//   BF16 output, VEC16_UE4M3 input scales, SCALAR_32F output scale
static void probe_availability() {
    cublasLtHandle_t lt = get_handle();
    if (!lt) { s_available = false; return; }

    constexpr int pM = 128, pN = 128, pK = 128;

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasStatus_t st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) { s_available = false; return; }

    cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // Input block-scale modes (VEC16_UE4M3: 1 unsigned E4M3 scale per 16 FP4 elements)
    cublasLtMatmulMatrixScale_t sm_vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm_vec16, sizeof(sm_vec16));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm_vec16, sizeof(sm_vec16));

    // Output scale mode (scalar FP32)
    cublasLtMatmulMatrixScale_t sm_scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &sm_scalar, sizeof(sm_scalar));

    void* dummy = nullptr;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy, sizeof(dummy));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy, sizeof(dummy));

    // A stored as [K,N] col-major, OP_T → logical [N,K]
    // B stored as [K,M] col-major, OP_N → logical [K,M]
    // D = [N,M] col-major = [M,N] row-major (BF16)
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, pK, pN, pK);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, pK, pM, pK);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, pN, pM, pN);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = 32ULL * 1024 * 1024;
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
        IMP_LOG_INFO("cuBLASLt NVFP4 GEMM: available (OP_T, BF16 output)");
    } else {
        IMP_LOG_INFO("cuBLASLt NVFP4 GEMM: not available (status=%d) — using CUTLASS NVFP4 path",
                     (int)st);
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

    // --- Ensure BF16 temp buffer is large enough ---
    size_t bf16_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
    if (bf16_bytes > s_bf16_buf_size) {
        if (s_bf16_buf) cudaFree(s_bf16_buf);
        cudaError_t err = cudaMalloc(&s_bf16_buf, bf16_bytes);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("cuBLASLt NVFP4: BF16 buffer alloc failed (%zu bytes): %s",
                          bf16_bytes, cudaGetErrorString(err));
            s_bf16_buf = nullptr;
            s_bf16_buf_size = 0;
            return false;
        }
        s_bf16_buf_size = bf16_bytes;
    }

    // --- Create operation descriptor ---
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasStatus_t st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) return false;

    // OP_T for weight (stored as [K,N] col-major, transposed → logical [N,K])
    // OP_N for activation ([K,M] col-major)
    cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // Block-scale mode: VEC16_UE4M3 (1 unsigned E4M3 scale per 16 FP4 elements)
    cublasLtMatmulMatrixScale_t sm_vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm_vec16, sizeof(sm_vec16));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm_vec16, sizeof(sm_vec16));

    // Output scale mode (scalar FP32)
    cublasLtMatmulMatrixScale_t sm_scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &sm_scalar, sizeof(sm_scalar));

    // Scale pointers:
    //   A_SCALE = weight scales (b.scale_factors)
    //   B_SCALE = activation scales (a_sf)
    const void* a_scale_ptr = b.scale_factors;
    const void* b_scale_ptr = a_sf;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                    &a_scale_ptr, sizeof(a_scale_ptr));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                    &b_scale_ptr, sizeof(b_scale_ptr));

    // --- Matrix layouts ---
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, K, N, K);   // stored [K,N], OP_T → [N,K]
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, M, K);   // activation [K,M] col-major
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N);       // C = BF16 (beta=0)
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, N, M, N);       // D = BF16 output

    // --- Algorithm heuristic ---
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t max_ws = 32ULL * 1024 * 1024;
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

    // --- Execute GEMM → BF16 ---
    float alpha = b.tensor_scale;
    float beta = 0.0f;

    st = cublasLtMatmul(lt, opDesc,
                         &alpha,
                         b.data, Adesc,           // cuBLAS A = weight
                         a_data, Bdesc,            // cuBLAS B = activation
                         &beta,
                         s_bf16_buf, Cdesc,        // C = BF16 (beta=0, never read)
                         s_bf16_buf, Ddesc,        // D = BF16 output
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

    // --- Convert BF16 → FP16 ---
    int n_elem = M * N;
    int threads = 256;
    int blocks = (n_elem + threads - 1) / threads;
    bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(s_bf16_buf),
        static_cast<half*>(d_fp16), n_elem);

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
