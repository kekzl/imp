#include "compute/gemm.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

namespace imp {

// ---------------------------------------------------------------------------
// cuBLAS / cuBLASLt handles (lazily initialized, process-lifetime)
// ---------------------------------------------------------------------------
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm: cublasCreate failed (status %d)\n", (int)st);
            abort();
        }
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

static cublasLtHandle_t get_cublaslt_handle() {
    static cublasLtHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t st = cublasLtCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm: cublasLtCreate failed (status %d)\n", (int)st);
            abort();
        }
    }
    return handle;
}

// ---------------------------------------------------------------------------
// Static workspace for cuBLASLt — allocated once via gemm_init(), shared by
// all GEMM calls.  Avoids per-call cudaMalloc which fails when GPU memory is
// saturated (e.g. 30B MoE models on 32 GB cards).
// ---------------------------------------------------------------------------
static void* s_workspace = nullptr;
static size_t s_workspace_size = 0;

void gemm_init() {
    // Force handle creation early.
    get_cublas_handle();
    get_cublaslt_handle();

    // Pre-allocate cuBLASLt workspace while GPU memory is still available.
    if (!s_workspace) {
        constexpr size_t kTrySizes[] = {
            32ULL << 20,   // 32 MiB
             8ULL << 20,   //  8 MiB
             2ULL << 20,   //  2 MiB
        };
        for (size_t sz : kTrySizes) {
            cudaError_t err = cudaMalloc(&s_workspace, sz);
            if (err == cudaSuccess) {
                s_workspace_size = sz;
                break;
            }
            s_workspace = nullptr;
        }
    }

    // Also let legacy cuBLAS API use the same workspace.
    if (s_workspace) {
        cublasSetWorkspace(get_cublas_handle(), s_workspace, s_workspace_size);
    }
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
            fprintf(stderr, "imp::gemm: unsupported dtype %d\n", (int)dt);
            abort();
    }
}

// ---------------------------------------------------------------------------
// Helper: choose cuBLAS compute type for a given operand dtype
// ---------------------------------------------------------------------------
static cublasComputeType_t dtype_to_compute(DType dt) {
    switch (dt) {
        case DType::FP32:     return CUBLAS_COMPUTE_32F;
        case DType::FP16:     return CUBLAS_COMPUTE_32F;   // accumulate in FP32 for accuracy
        case DType::BF16:     return CUBLAS_COMPUTE_32F;
        case DType::FP8_E4M3: return CUBLAS_COMPUTE_32F;
        case DType::FP8_E5M2: return CUBLAS_COMPUTE_32F;
        case DType::INT8:     return CUBLAS_COMPUTE_32I;
        default:              return CUBLAS_COMPUTE_32F;
    }
}

// ---------------------------------------------------------------------------
// gemm:  C = alpha * A @ B^T + beta * C
//   A [M, K]  B [N, K]  C [M, N]   -- all row-major
//
// Weight matrices from GGUF are [out_features, in_features] = [N, K].
// cuBLAS is column-major.  For row-major C = A @ B^T:
//   C^T = B @ A^T  (in col-major)
// So we call cuBLAS with (transa=T, transb=N, m=N, n=M, k=K,
//   lda=K (for B), ldb=K (for A), ldc=N (for C)).
// ---------------------------------------------------------------------------
void gemm(const Tensor& A, const Tensor& B, Tensor& C,
          float alpha, float beta, cudaStream_t stream) {
    // --- dimension extraction ---
    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];  // B is [N, K], we compute A @ B^T

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    // --- FP32 fast path using cublasSgemm ---
    // B is [N,K] row-major = [K,N] col-major. We need B transposed → CUBLAS_OP_T.
    // A is [M,K] row-major = [K,M] col-major. We need A as-is    → CUBLAS_OP_N.
    if (A.dtype == DType::FP32 && B.dtype == DType::FP32 && C.dtype == DType::FP32) {
        cublasStatus_t st = cublasSgemm(
            handle,
            CUBLAS_OP_T,    // transa: transpose B_col [K,N] → [N,K]
            CUBLAS_OP_N,    // transb: A_col [K,M] used as-is
            (int)N,         // m
            (int)M,         // n
            (int)K,         // k
            &alpha,
            static_cast<const float*>(B.data), (int)K,  // lda = K (leading dim of B before transpose)
            static_cast<const float*>(A.data), (int)K,  // ldb = K (leading dim of A)
            &beta,
            static_cast<float*>(C.data),       (int)N   // ldc = N
        );
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm: cublasSgemm failed (status %d)\n", (int)st);
        }
        return;
    }

    // --- Generic path via cuBLASLt (uses pre-allocated static workspace) ---
    cudaDataType_t cuda_dtype_A = dtype_to_cuda(A.dtype);
    cudaDataType_t cuda_dtype_B = dtype_to_cuda(B.dtype);
    cudaDataType_t cuda_dtype_C = dtype_to_cuda(C.dtype);
    cublasComputeType_t compute_type = dtype_to_compute(A.dtype);
    cudaDataType_t scale_type = (compute_type == CUBLAS_COMPUTE_32I)
                                    ? CUDA_R_32I : CUDA_R_32F;

    cublasLtHandle_t lt = get_cublaslt_handle();

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatmulDescCreate(&opDesc, compute_type, scale_type);

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &transB, sizeof(transB));

    cublasLtMatrixLayout_t Bdesc = nullptr, Adesc = nullptr, Cdesc = nullptr;
    cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype_B, (int)K, (int)N, (int)K);
    cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype_A, (int)K, (int)M, (int)K);
    cublasLtMatrixLayoutCreate(&Cdesc, cuda_dtype_C, (int)N, (int)M, (int)N);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &s_workspace_size, sizeof(s_workspace_size));

    cublasLtMatmulHeuristicResult_t result = {};
    int nresults = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Bdesc, Adesc, Cdesc, Cdesc,
                                    pref, 1, &result, &nresults);

    size_t ws = (nresults > 0 && result.workspaceSize <= s_workspace_size)
                    ? result.workspaceSize : 0;

    if (compute_type == CUBLAS_COMPUTE_32I) {
        int32_t ialpha = (int32_t)alpha;
        int32_t ibeta  = (int32_t)beta;
        cublasStatus_t st = cublasLtMatmul(lt, opDesc,
            &ialpha, B.data, Bdesc, A.data, Adesc,
            &ibeta,  C.data, Cdesc, C.data, Cdesc,
            (nresults > 0) ? &result.algo : nullptr,
            s_workspace, ws, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm: cublasLtMatmul (INT) failed (status %d)\n", (int)st);
        }
    } else {
        cublasStatus_t st = cublasLtMatmul(lt, opDesc,
            &alpha, B.data, Bdesc, A.data, Adesc,
            &beta,  C.data, Cdesc, C.data, Cdesc,
            (nresults > 0) ? &result.algo : nullptr,
            s_workspace, ws, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            static int err_count = 0;
            if (++err_count <= 10) {
                fprintf(stderr, "imp::gemm: cublasLtMatmul failed (status %d) "
                        "M=%ld K=%ld N=%ld nresults=%d ws=%zu/%zu "
                        "dtA=%d dtB=%d dtC=%d\n",
                        (int)st, (long)M, (long)K, (long)N,
                        nresults, ws, s_workspace_size,
                        (int)cuda_dtype_A, (int)cuda_dtype_B, (int)cuda_dtype_C);
            }
        }
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);
}

// ---------------------------------------------------------------------------
// GEMV kernels -- each warp computes one output element (dot product of a row)
// ---------------------------------------------------------------------------

// --- FP32 GEMV kernel ---
__global__ void gemv_fp32_kernel(const float* __restrict__ A,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int M, int K) {
    // Each warp handles one row of A.
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const float* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: float4 = 4 floats per load.
    const int K_vec = K / 4;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v     = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a = A_row_v[i];
        float4 xv = x_v[i];
        sum += a.x * xv.x + a.y * xv.y + a.z * xv.z + a.w * xv.w;
    }

    // Handle remainder elements (K not divisible by 4).
    int base = K_vec * 4;
    for (int i = base + lane; i < K; i += 32) {
        sum += A_row[i] * x[i];
    }

    // Warp-level reduction via shuffle.
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// --- FP16 GEMV kernel ---
__global__ void gemv_fp16_kernel(const half* __restrict__ A,
                                  const half* __restrict__ x,
                                  half* __restrict__ y,
                                  int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const half* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: load 4 halves (8 bytes) at a time via float2-sized loads
    // half2 packing: process 2 halves at a time via half2 multiply-add.
    const int K_vec = K / 8;  // 8 halves = 16 bytes = sizeof(float4)
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v     = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a_raw = A_row_v[i];
        float4 x_raw = x_v[i];

        // Reinterpret as half2 arrays (4 half2 per float4).
        const half2* a_h2 = reinterpret_cast<const half2*>(&a_raw);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_raw);

        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2[j], x_h2[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
    }

    // Remainder.
    int base = K_vec * 8;
    for (int i = base + lane; i < K; i += 32) {
        sum += __half2float(A_row[i]) * __half2float(x[i]);
    }

    // Warp shuffle reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// --- BF16 GEMV kernel ---
__global__ void gemv_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ x,
                                  __nv_bfloat16* __restrict__ y,
                                  int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const __nv_bfloat16* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: 8 bf16 per float4.
    const int K_vec = K / 8;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v     = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a_raw = A_row_v[i];
        float4 x_raw = x_v[i];

        const __nv_bfloat162* a_h2 = reinterpret_cast<const __nv_bfloat162*>(&a_raw);
        const __nv_bfloat162* x_h2 = reinterpret_cast<const __nv_bfloat162*>(&x_raw);

        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 prod = __hmul2(a_h2[j], x_h2[j]);
            sum += __bfloat162float(prod.x) + __bfloat162float(prod.y);
        }
    }

    // Remainder.
    int base = K_vec * 8;
    for (int i = base + lane; i < K; i += 32) {
        sum += __bfloat162float(A_row[i]) * __bfloat162float(x[i]);
    }

    // Warp shuffle reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        y[row] = __float2bfloat16(sum);
    }
}

// ---------------------------------------------------------------------------
// gemv:  y = A @ x
//   A [M, K],  x [K] or [K, batch],  y [M] or [M, batch]
//   Custom CUDA kernels for the memory-bandwidth-bound case.
//   For batched case (x has 2 dims), we loop over batch columns.
// ---------------------------------------------------------------------------
void gemv(const Tensor& A, const Tensor& x, Tensor& y,
          cudaStream_t stream) {
    const int M = (int)A.shape[0];
    const int K = (int)A.shape[1];

    // Determine batch size from x's shape.
    int batch = 1;
    if (x.ndim == 2) {
        batch = (int)x.shape[1];
    }

    const int threads_per_block = 256;  // 8 warps per block
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;

    for (int b = 0; b < batch; ++b) {
        switch (A.dtype) {
            case DType::FP32: {
                const float* A_ptr = static_cast<const float*>(A.data);
                const float* x_ptr = static_cast<const float*>(x.data) + (int64_t)b * K;
                float* y_ptr       = static_cast<float*>(y.data)       + (int64_t)b * M;
                gemv_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    A_ptr, x_ptr, y_ptr, M, K);
                break;
            }
            case DType::FP16: {
                const half* A_ptr = static_cast<const half*>(A.data);
                const half* x_ptr = static_cast<const half*>(x.data) + (int64_t)b * K;
                half* y_ptr       = static_cast<half*>(y.data)       + (int64_t)b * M;
                gemv_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    A_ptr, x_ptr, y_ptr, M, K);
                break;
            }
            case DType::BF16: {
                const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data);
                const __nv_bfloat16* x_ptr = static_cast<const __nv_bfloat16*>(x.data) + (int64_t)b * K;
                __nv_bfloat16* y_ptr       = static_cast<__nv_bfloat16*>(y.data)       + (int64_t)b * M;
                gemv_bf16_kernel<<<blocks, threads_per_block, 0, stream>>>(
                    A_ptr, x_ptr, y_ptr, M, K);
                break;
            }
            default: {
                // Fallback: use cuBLAS gemv for other dtypes via gemm with N=1.
                // Construct a temporary Tensor view for the column vectors.
                Tensor x_col;
                x_col.data = static_cast<char*>(x.data) + b * K * dtype_size(x.dtype);
                x_col.dtype = x.dtype;
                x_col.ndim = 2;
                x_col.shape[0] = K;
                x_col.shape[1] = 1;
                x_col.stride[0] = 1;
                x_col.stride[1] = K;
                x_col.on_device = true;

                Tensor y_col;
                y_col.data = static_cast<char*>(y.data) + b * M * dtype_size(y.dtype);
                y_col.dtype = y.dtype;
                y_col.ndim = 2;
                y_col.shape[0] = M;
                y_col.shape[1] = 1;
                y_col.stride[0] = 1;
                y_col.stride[1] = M;
                y_col.on_device = true;

                gemm(A, x_col, y_col, 1.0f, 0.0f, stream);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// gemm_cublaslt: cuBLASLt GEMM with explicit algorithm selection + FP8 scales
//   Uses the same static workspace as gemm().
// ---------------------------------------------------------------------------
void gemm_cublaslt(const Tensor& A, const Tensor& B, Tensor& C,
                   float alpha, float beta,
                   const float* aScale, const float* bScale,
                   cudaStream_t stream) {
    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];

    cublasLtHandle_t lt = get_cublaslt_handle();

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &transB, sizeof(transB));

    if (aScale) {
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                        &aScale, sizeof(aScale));
    }
    if (bScale) {
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                        &bScale, sizeof(bScale));
    }

    cudaDataType_t cuda_dtype_A = dtype_to_cuda(A.dtype);
    cudaDataType_t cuda_dtype_B = dtype_to_cuda(B.dtype);
    cudaDataType_t cuda_dtype_C = dtype_to_cuda(C.dtype);

    cublasLtMatrixLayout_t Bdesc = nullptr, Adesc = nullptr, Cdesc = nullptr;
    cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype_B, (int)K, (int)N, (int)K);
    cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype_A, (int)K, (int)M, (int)K);
    cublasLtMatrixLayoutCreate(&Cdesc, cuda_dtype_C, (int)N, (int)M, (int)N);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &s_workspace_size, sizeof(s_workspace_size));

    cublasLtMatmulHeuristicResult_t result = {};
    int nresults = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Bdesc, Adesc, Cdesc, Cdesc,
                                    pref, 1, &result, &nresults);

    size_t ws = (nresults > 0 && result.workspaceSize <= s_workspace_size)
                    ? result.workspaceSize : 0;

    cublasStatus_t st = cublasLtMatmul(lt, opDesc,
        &alpha, B.data, Bdesc, A.data, Adesc,
        &beta,  C.data, Cdesc, C.data, Cdesc,
        (nresults > 0) ? &result.algo : nullptr,
        s_workspace, ws, stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm_cublaslt: cublasLtMatmul failed (status %d)\n", (int)st);
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);
}

// ---------------------------------------------------------------------------
// FP8 E4M3 GEMV kernel -- 16 FP8 values per load (16 bytes)
// Each warp handles one row. Dequant on-the-fly with per-tensor scale.
// ---------------------------------------------------------------------------
__global__ void gemv_fp8_e4m3_kernel(const uint8_t* __restrict__ A,
                                      const half* __restrict__ x,
                                      half* __restrict__ y,
                                      int M, int K, float scale) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const uint8_t* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: 16 FP8 values per load (16 bytes = sizeof(float4))
    const int K_vec = K / 16;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);

    // x is FP16 -- load 8 halves at a time (16 bytes)
    const float4* x_v = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a_raw = A_row_v[i];
        float4 x_raw = x_v[i];

        // Reinterpret FP8 bytes
        const uint8_t* a_bytes = reinterpret_cast<const uint8_t*>(&a_raw);
        const half* x_halves = reinterpret_cast<const half*>(&x_raw);

        // Manually dequant and accumulate 16 FP8 values
        for (int j = 0; j < 16; ++j) {
            // Simple FP8 E4M3 dequant: interpret as native type if available
#if defined(__CUDA_FP8_TYPES_EXIST__)
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &a_bytes[j], 1);
            float a_val = (float)fp8_val * scale;
#else
            // Software fallback: quick dequant
            uint8_t bits = a_bytes[j];
            uint8_t sign = (bits >> 7) & 1;
            int exp = (int)((bits >> 3) & 0x0F);
            uint8_t man = bits & 0x07;
            float a_val;
            if (exp == 0 && man == 0) {
                a_val = 0.0f;
            } else if (exp == 0) {
                a_val = ldexpf((float)man / 8.0f, -6);
            } else if (exp == 15 && man != 0) {
                a_val = 0.0f; // NaN -> 0
            } else {
                a_val = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
            }
            if (sign) a_val = -a_val;
            a_val *= scale;
#endif
            float x_val = (j < 8) ? __half2float(x_halves[j]) : __half2float(x_halves[j]);
            sum += a_val * x_val;
        }
    }

    // Handle remainder
    int base = K_vec * 16;
    for (int i = base + lane; i < K; i += 32) {
#if defined(__CUDA_FP8_TYPES_EXIST__)
        __nv_fp8_e4m3 fp8_val;
        memcpy(&fp8_val, &A_row[i], 1);
        float a_val = (float)fp8_val * scale;
#else
        uint8_t bits = A_row[i];
        uint8_t sign = (bits >> 7) & 1;
        int exp = (int)((bits >> 3) & 0x0F);
        uint8_t man = bits & 0x07;
        float a_val;
        if (exp == 0 && man == 0) a_val = 0.0f;
        else if (exp == 0) a_val = ldexpf((float)man / 8.0f, -6);
        else a_val = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
        if (sign) a_val = -a_val;
        a_val *= scale;
#endif
        sum += a_val * __half2float(*(reinterpret_cast<const half*>(x) + i));
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

void gemv_fp8(const Tensor& A, const Tensor& x, Tensor& y,
              float scale, cudaStream_t stream) {
    const int M = (int)A.shape[0];
    const int K = (int)A.shape[1];

    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;

    gemv_fp8_e4m3_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(A.data),
        static_cast<const half*>(x.data),
        static_cast<half*>(y.data),
        M, K, scale);
}

} // namespace imp
