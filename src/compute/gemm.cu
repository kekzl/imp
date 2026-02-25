#include "compute/gemm.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>

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
        // Use TF32 tensor ops for FP32 inputs (19-bit mantissa, ~8x faster
        // than FP32 on Hopper/Blackwell tensor cores, accuracy sufficient for
        // inference).  cuBLAS will also select FP16/BF16 tensor core paths
        // automatically when operands match.
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
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
            64ULL << 20,   // 64 MiB — RTX 5090 (32 GB) has headroom
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
// cuBLASLt descriptor + algorithm cache
// ---------------------------------------------------------------------------
struct GemmCacheKey {
    cudaDataType_t dtA, dtB, dtC;
    cublasComputeType_t compute;
    int64_t M, K, N;

    bool operator==(const GemmCacheKey& o) const {
        return dtA == o.dtA && dtB == o.dtB && dtC == o.dtC &&
               compute == o.compute && M == o.M && K == o.K && N == o.N;
    }
};

struct GemmCacheKeyHash {
    size_t operator()(const GemmCacheKey& k) const {
        size_t h = 14695981039346656037ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        mix(static_cast<uint64_t>(k.dtA));
        mix(static_cast<uint64_t>(k.dtB));
        mix(static_cast<uint64_t>(k.dtC));
        mix(static_cast<uint64_t>(k.compute));
        mix(static_cast<uint64_t>(k.M));
        mix(static_cast<uint64_t>(k.K));
        mix(static_cast<uint64_t>(k.N));
        return h;
    }
};

struct GemmCacheEntry {
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulAlgo_t algo;
    size_t workspace_size;
    bool has_algo;
};

static std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheKeyHash> s_gemm_cache;
static std::mutex s_gemm_cache_mutex;

void gemm_cleanup() {
    std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
    for (auto& [key, entry] : s_gemm_cache) {
        cublasLtMatrixLayoutDestroy(entry.Adesc);
        cublasLtMatrixLayoutDestroy(entry.Bdesc);
        cublasLtMatrixLayoutDestroy(entry.Cdesc);
        cublasLtMatmulDescDestroy(entry.opDesc);
    }
    s_gemm_cache.clear();
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

    // --- GEMV fast path for M=1 decode (memory-bandwidth-bound) ---
    // Applies when all operands share the same dtype (excludes LM head: FP16→FP32).
    if (M == 1 && alpha == 1.0f && beta == 0.0f &&
        A.dtype == B.dtype && A.dtype == C.dtype &&
        (A.dtype == DType::FP16 || A.dtype == DType::FP32 || A.dtype == DType::BF16)) {
        Tensor x_vec;
        x_vec.data = A.data;
        x_vec.dtype = A.dtype;
        x_vec.ndim = 1;
        x_vec.shape[0] = K;
        x_vec.stride[0] = 1;
        x_vec.on_device = true;

        Tensor y_vec;
        y_vec.data = C.data;
        y_vec.dtype = C.dtype;
        y_vec.ndim = 1;
        y_vec.shape[0] = N;
        y_vec.stride[0] = 1;
        y_vec.on_device = true;

        gemv(B, x_vec, y_vec, stream);
        return;
    }

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

    // --- Generic path via cuBLASLt (uses pre-allocated static workspace + descriptor cache) ---
    cudaDataType_t cuda_dtype_A = dtype_to_cuda(A.dtype);
    cudaDataType_t cuda_dtype_B = dtype_to_cuda(B.dtype);
    cudaDataType_t cuda_dtype_C = dtype_to_cuda(C.dtype);
    cublasComputeType_t compute_type = dtype_to_compute(A.dtype);

    cublasLtHandle_t lt = get_cublaslt_handle();

    GemmCacheKey cache_key{cuda_dtype_A, cuda_dtype_B, cuda_dtype_C, compute_type, M, K, N};

    GemmCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
        auto it = s_gemm_cache.find(cache_key);
        if (it != s_gemm_cache.end()) {
            entry = &it->second;
        } else {
            GemmCacheEntry new_entry{};
            cudaDataType_t scale_type = (compute_type == CUBLAS_COMPUTE_32I)
                                            ? CUDA_R_32I : CUDA_R_32F;

            cublasLtMatmulDescCreate(&new_entry.opDesc, compute_type, scale_type);

            cublasOperation_t transA = CUBLAS_OP_T;
            cublasOperation_t transB = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(new_entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transA, sizeof(transA));
            cublasLtMatmulDescSetAttribute(new_entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transB, sizeof(transB));

            cublasLtMatrixLayoutCreate(&new_entry.Bdesc, cuda_dtype_B, (int)K, (int)N, (int)K);
            cublasLtMatrixLayoutCreate(&new_entry.Adesc, cuda_dtype_A, (int)K, (int)M, (int)K);
            cublasLtMatrixLayoutCreate(&new_entry.Cdesc, cuda_dtype_C, (int)N, (int)M, (int)N);

            cublasLtMatmulPreference_t pref = nullptr;
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &s_workspace_size, sizeof(s_workspace_size));

            cublasLtMatmulHeuristicResult_t result = {};
            int nresults = 0;
            cublasLtMatmulAlgoGetHeuristic(lt, new_entry.opDesc, new_entry.Bdesc,
                                            new_entry.Adesc, new_entry.Cdesc, new_entry.Cdesc,
                                            pref, 1, &result, &nresults);

            cublasLtMatmulPreferenceDestroy(pref);

            new_entry.has_algo = (nresults > 0);
            if (nresults > 0) {
                new_entry.algo = result.algo;
                new_entry.workspace_size = (result.workspaceSize <= s_workspace_size)
                                               ? result.workspaceSize : 0;
            } else {
                new_entry.workspace_size = 0;
            }

            auto [inserted_it, _] = s_gemm_cache.emplace(cache_key, new_entry);
            entry = &inserted_it->second;
        }
    }

    if (compute_type == CUBLAS_COMPUTE_32I) {
        int32_t ialpha = (int32_t)alpha;
        int32_t ibeta  = (int32_t)beta;
        cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc,
            &ialpha, B.data, entry->Bdesc, A.data, entry->Adesc,
            &ibeta,  C.data, entry->Cdesc, C.data, entry->Cdesc,
            entry->has_algo ? &entry->algo : nullptr,
            s_workspace, entry->workspace_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm: cublasLtMatmul (INT) failed (status %d)\n", (int)st);
        }
    } else {
        cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc,
            &alpha, B.data, entry->Bdesc, A.data, entry->Adesc,
            &beta,  C.data, entry->Cdesc, C.data, entry->Cdesc,
            entry->has_algo ? &entry->algo : nullptr,
            s_workspace, entry->workspace_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            static int err_count = 0;
            if (++err_count <= 10) {
                fprintf(stderr, "imp::gemm: cublasLtMatmul failed (status %d) "
                        "M=%ld K=%ld N=%ld has_algo=%d ws=%zu/%zu "
                        "dtA=%d dtB=%d dtC=%d\n",
                        (int)st, (long)M, (long)K, (long)N,
                        entry->has_algo, entry->workspace_size, s_workspace_size,
                        (int)cuda_dtype_A, (int)cuda_dtype_B, (int)cuda_dtype_C);
            }
        }
    }
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

        // 16 FP8 values need 16 FP16 values = 2 float4 loads from x
        float4 x_raw0 = x_v[2 * i];
        float4 x_raw1 = x_v[2 * i + 1];

        // Reinterpret FP8 bytes
        const uint8_t* a_bytes = reinterpret_cast<const uint8_t*>(&a_raw);
        const half* x_lo = reinterpret_cast<const half*>(&x_raw0);  // x[0..7]
        const half* x_hi = reinterpret_cast<const half*>(&x_raw1);  // x[8..15]

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
            float x_val = (j < 8) ? __half2float(x_lo[j]) : __half2float(x_hi[j - 8]);
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

// ---------------------------------------------------------------------------
// Fused Q6_K GEMV kernel -- dequant-and-dot in one pass.
// Q6_K block = 210 bytes for 256 elements: ql[128] + qh[64] + scales[16] + d[2].
// Each warp computes one output row's dot product.
// ---------------------------------------------------------------------------
__global__ void gemv_q6k_kernel(const uint8_t* __restrict__ W,
                                 const half* __restrict__ x,
                                 half* __restrict__ y,
                                 int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 210;
        const uint8_t* ql = bp;          // ql[128]
        const uint8_t* qh = bp + 128;    // qh[64]
        const int8_t* sc  = (const int8_t*)(bp + 192);  // scales[16]
        float d = __half2float(*(const half*)(bp + 208)); // d[2]
        const int base = b * 256;

        // Coalesced loads: 4 ql bytes + 2 qh bytes per thread
        uint8_t ql_a = ql[lane];           // [0..31]
        uint8_t ql_b = ql[lane + 32];      // [32..63]
        uint8_t ql_c = ql[64 + lane];      // [64..95]
        uint8_t ql_d = ql[64 + lane + 32]; // [96..127]
        uint8_t qh0  = qh[lane];           // [0..31]
        uint8_t qh1  = qh[32 + lane];      // [32..63]

        // Dequant 8 values per thread (elements at lane, lane+32, ..., lane+224)
        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        // Scale lookups: 16 scales per block, 2 sub-blocks of 32 elements each
        // lane/16 selects between two scale groups within each 32-lane sub-block
        int sc_idx = lane >> 4;  // 0 or 1
        sum += d * (
            (float)sc[sc_idx]      * (float)q0 * __half2float(x[base + lane]) +
            (float)sc[sc_idx + 2]  * (float)q1 * __half2float(x[base + lane + 32]) +
            (float)sc[sc_idx + 4]  * (float)q2 * __half2float(x[base + lane + 64]) +
            (float)sc[sc_idx + 6]  * (float)q3 * __half2float(x[base + lane + 96]) +
            (float)sc[sc_idx + 8]  * (float)q4 * __half2float(x[base + lane + 128]) +
            (float)sc[sc_idx + 10] * (float)q5 * __half2float(x[base + lane + 160]) +
            (float)sc[sc_idx + 12] * (float)q6 * __half2float(x[base + lane + 192]) +
            (float)sc[sc_idx + 14] * (float)q7 * __half2float(x[base + lane + 224]));
    }

    // Warp shuffle reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[row] = __float2half(sum);
}

void gemv_q6k(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;  // 8 warps
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;
    gemv_q6k_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(W), x, y, M, K);
}

// ---------------------------------------------------------------------------
// Fused Q8_0 GEMV kernel -- dequant-and-dot in one pass.
// Q8_0 block = 34 bytes for 32 elements: d[2] + qs[32].
// Each warp computes one output row's dot product. Each thread handles one
// element per block (32 threads = 32 elements = 1 block).
// ---------------------------------------------------------------------------
__global__ void gemv_q8_0_kernel(const uint8_t* __restrict__ W,
                                  const half* __restrict__ x,
                                  half* __restrict__ y,
                                  int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 34;
        float d = __half2float(*(const half*)bp);
        int8_t q = ((const int8_t*)(bp + 2))[lane];
        sum += d * (float)q * __half2float(x[b * 32 + lane]);
    }

    // Warp shuffle reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[row] = __float2half(sum);
}

void gemv_q8_0(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;  // 8 warps
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;
    gemv_q8_0_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(W), x, y, M, K);
}

// ---------------------------------------------------------------------------
// MoE decode GEMV: processes all top_k experts in a single kernel launch.
// expert_indices[slot] selects which expert's weights to read from packed_weights.
// Grid: top_k * blocks_per_expert blocks. Each block group handles one expert slot.
// x_stride: 0 = shared input for all experts (gate/up), >0 = per-expert input (down).
// ---------------------------------------------------------------------------

__global__ void gemv_q6k_moe_decode_kernel(
        const uint8_t* __restrict__ packed_weights,
        const int32_t* __restrict__ expert_indices,
        const half* __restrict__ x,
        half* __restrict__ y,
        int rows, int K,
        size_t expert_stride_bytes,
        int x_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 210;
        const uint8_t* ql = bp;
        const uint8_t* qh = bp + 128;
        const int8_t* sc  = (const int8_t*)(bp + 192);
        float d = __half2float(*(const half*)(bp + 208));
        const int base = b * 256;

        uint8_t ql_a = ql[lane];
        uint8_t ql_b = ql[lane + 32];
        uint8_t ql_c = ql[64 + lane];
        uint8_t ql_d = ql[64 + lane + 32];
        uint8_t qh0  = qh[lane];
        uint8_t qh1  = qh[32 + lane];

        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        int sc_idx = lane >> 4;
        sum += d * (
            (float)sc[sc_idx]      * (float)q0 * __half2float(x_ptr[base + lane]) +
            (float)sc[sc_idx + 2]  * (float)q1 * __half2float(x_ptr[base + lane + 32]) +
            (float)sc[sc_idx + 4]  * (float)q2 * __half2float(x_ptr[base + lane + 64]) +
            (float)sc[sc_idx + 6]  * (float)q3 * __half2float(x_ptr[base + lane + 96]) +
            (float)sc[sc_idx + 8]  * (float)q4 * __half2float(x_ptr[base + lane + 128]) +
            (float)sc[sc_idx + 10] * (float)q5 * __half2float(x_ptr[base + lane + 160]) +
            (float)sc[sc_idx + 12] * (float)q6 * __half2float(x_ptr[base + lane + 192]) +
            (float)sc[sc_idx + 14] * (float)q7 * __half2float(x_ptr[base + lane + 224]));
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_moe_decode(const void* packed_weights,
                          const int32_t* expert_indices,
                          const half* x, half* y,
                          int rows, int K,
                          size_t expert_stride_bytes,
                          int x_stride, int top_k,
                          cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    const int total_blocks = top_k * blocks_per_expert;
    gemv_q6k_moe_decode_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        expert_indices, x, y, rows, K,
        expert_stride_bytes, x_stride, blocks_per_expert);
}

__global__ void gemv_q8_0_moe_decode_kernel(
        const uint8_t* __restrict__ packed_weights,
        const int32_t* __restrict__ expert_indices,
        const half* __restrict__ x,
        half* __restrict__ y,
        int rows, int K,
        size_t expert_stride_bytes,
        int x_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 34;
        float d = __half2float(*(const half*)bp);
        int8_t q = ((const int8_t*)(bp + 2))[lane];
        sum += d * (float)q * __half2float(x_ptr[b * 32 + lane]);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_moe_decode(const void* packed_weights,
                           const int32_t* expert_indices,
                           const half* x, half* y,
                           int rows, int K,
                           size_t expert_stride_bytes,
                           int x_stride, int top_k,
                           cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    const int total_blocks = top_k * blocks_per_expert;
    gemv_q8_0_moe_decode_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        expert_indices, x, y, rows, K,
        expert_stride_bytes, x_stride, blocks_per_expert);
}

// ===========================================================================
// MMVQ: dp4a-accelerated quantized GEMV
// ===========================================================================
//
// Instead of dequantizing weights to FP16 and doing float*float per element,
// we quantize the input vector to Q8_1 (INT8 + scale) and use dp4a (native
// INT8x4 dot product) for accumulation. This processes 4 elements per
// instruction and halves input bandwidth (INT8 vs FP16).
//
// Q8_1 block: 32 values quantized to INT8 with per-block scale (d) and sum (s).
// ===========================================================================

// Quantize K FP16 values to Q8_1 blocks.
// Grid: K/32 blocks, 32 threads each. One block per Q8_1 output block.
__global__ void quantize_fp16_to_q8_1_kernel(const half* __restrict__ x,
                                              block_q8_1* __restrict__ q8_1_out,
                                              float* __restrict__ d8_out,
                                              int K) {
    const int block_idx = blockIdx.x;
    const int lane = threadIdx.x;  // 0..31
    const int base = block_idx * 32;

    if (base + lane >= K) return;

    // Load one FP16 value per thread
    float val = __half2float(x[base + lane]);

    // Find max absolute value across the 32-element block via warp shuffle
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, amax, offset);
        amax = fmaxf(amax, other);
    }

    // Compute scale: d = max / 127
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    // Quantize: round to nearest int8
    int8_t q = static_cast<int8_t>(__float2int_rn(val * id));

    // Write output: lane 0 writes the header, all lanes write their qs byte
    block_q8_1* out = &q8_1_out[block_idx];
    out->qs[lane] = q;
    if (lane == 0) {
        out->d = __float2half(d);
        d8_out[block_idx] = d;
    }
}

void quantize_fp16_to_q8_1(const half* x, block_q8_1* q8_1_out, float* d8_out,
                            int K, cudaStream_t stream) {
    int n_blocks = K / 32;
    if (n_blocks <= 0) return;
    quantize_fp16_to_q8_1_kernel<<<n_blocks, 32, 0, stream>>>(x, q8_1_out, d8_out, K);
}

// ---------------------------------------------------------------------------
// Fused SwiGLU + Q8_1 quantization kernel.
// Computes silu(gate) * up and quantizes the result to Q8_1 in a single pass.
// Eliminates the intermediate FP16 activation buffer write+read.
//
// Each block handles 32 contiguous elements (one Q8_1 block).
// 32 threads per block (one full warp).
// ---------------------------------------------------------------------------
__global__ void swiglu_quantize_q8_1_kernel(
        const half* __restrict__ gate,       // [total_elements] FP16
        const half* __restrict__ up,         // [total_elements] FP16
        block_q8_1* __restrict__ q8_out,     // [total_elements/32] Q8_1 blocks
        float* __restrict__ d8_out,          // [total_elements/32] block scales
        int total_elements) {
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    const int idx = blk * 32 + tid;

    // SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    float val = 0.0f;
    if (idx < total_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        val = g / (1.0f + expf(-g)) * u;
    }

    // Warp-cooperative quantization: find max abs
    float amax = fabsf(val);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));

    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    int8_t q = static_cast<int8_t>(__float2int_rn(val * id));

    q8_out[blk].qs[tid] = q;
    if (tid == 0) {
        q8_out[blk].d = __float2half(d);
        d8_out[blk] = d;
    }
}

void swiglu_quantize_q8_1(const half* gate, const half* up,
                            block_q8_1* q8_out, float* d8_out,
                            int total_elements, cudaStream_t stream) {
    int n_blocks = total_elements / 32;
    if (n_blocks <= 0) return;
    swiglu_quantize_q8_1_kernel<<<n_blocks, 32, 0, stream>>>(
        gate, up, q8_out, d8_out, total_elements);
}

// ---------------------------------------------------------------------------
// Fused RMSNorm + Q8_1 quantization kernel.
//
// Combines RMSNorm (with weight) and FP16→Q8_1 quantization in one kernel,
// eliminating the intermediate norm_out FP16 buffer write+read.
//
// Single-row only (n=1 decode). One CUDA block, 256 threads.
// Phase 1: Load hidden, compute sum of squares, block-reduce for RMS.
// Phase 2: Normalize (multiply by inv_rms * weight), write to shared memory.
// Phase 3: Quantize from shared memory to Q8_1 blocks (32 elements per warp).
//
// Also writes the FP16 norm_out if norm_out_ptr is non-null (needed when
// the GEMV path doesn't consume Q8_1, e.g. non-quantized weights).
// ---------------------------------------------------------------------------
__global__ void rmsnorm_quantize_q8_1_kernel(
        const half* __restrict__ x,         // [d_model] input hidden state
        const half* __restrict__ weight,    // [d_model] RMSNorm weight
        block_q8_1* __restrict__ q8_out,    // [d_model/32] Q8_1 output blocks
        float* __restrict__ d8_out,         // [d_model/32] Q8_1 block scales
        half* __restrict__ norm_out_ptr,    // [d_model] optional FP16 output (can be null)
        int d_model,
        float eps) {
    // Minimal shared memory: 8 floats for cross-warp reduction + 1 for inv_rms.
    // All intermediate values kept in registers — no d_model-sized shared buffer.
    __shared__ float warp_reduce[8];
    __shared__ float s_inv_rms;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int n_q8_blocks = d_model >> 5;  // d_model / 32

    // Phase 1: Load x values with warp-aligned Q8_1 block access (coalesced),
    // cache in registers, and compute sum of squares.
    // Each warp handles blocks in stride-n_warps order.
    // Max blocks per warp = d_model / (32 * n_warps) = d_model / 256.
    // For d_model=8192, that's 32 — fits in registers easily.
    float x_cache[32];
    float sum_sq = 0.0f;
    int n_cached = 0;

    for (int b = warp_id; b < n_q8_blocks; b += n_warps) {
        float v = __half2float(x[b * 32 + lane]);
        x_cache[n_cached++] = v;
        sum_sq += v * v;
    }

    // Warp reduce sum_sq
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0) warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            s_inv_rms = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Phase 2+3: Normalize (from register cache) + Quantize to Q8_1 (fused).
    // No intermediate shared memory buffer needed.
    int cache_idx = 0;
    for (int b = warp_id; b < n_q8_blocks; b += n_warps) {
        float val = x_cache[cache_idx++] * inv_rms * __half2float(weight[b * 32 + lane]);

        if (norm_out_ptr) norm_out_ptr[b * 32 + lane] = __float2half(val);

        // Warp-level amax for Q8_1 quantization
        float amax = fabsf(val);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));

        float d = amax / 127.0f;
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

        int8_t q = static_cast<int8_t>(__float2int_rn(val * id));

        q8_out[b].qs[lane] = q;
        if (lane == 0) {
            q8_out[b].d = __float2half(d);
            d8_out[b] = d;
        }
    }
}

void rmsnorm_quantize_q8_1(const half* x, const half* weight,
                             block_q8_1* q8_out, float* d8_out,
                             half* norm_out,
                             int d_model, float eps,
                             cudaStream_t stream) {
    const int threads = 256;
    // Shared memory: only warp_reduce[8] + s_inv_rms declared as __shared__ in kernel
    // (no dynamic shared memory needed).
    rmsnorm_quantize_q8_1_kernel<<<1, threads, 0, stream>>>(
        x, weight, q8_out, d8_out, norm_out, d_model, eps);
}

// ---------------------------------------------------------------------------
// FP16 GEMV with FP32 output for MoE gate logits: y = W @ x
// W: [M, K] FP16 (row-major), x: [K] FP16, y: [M] FP32.
// Designed for M=n_experts (64-256), K=d_model (2048-8192), n=1 decode.
// Replaces cuBLAS gemm() + fp16_to_fp32 cast for tiny M=1 GEMMs.
// Each warp handles one output row. Uses half2 vectorized loads for 2x bandwidth.
// ---------------------------------------------------------------------------
__global__ void gemv_gate_fp32_kernel(const half* __restrict__ W,
                                       const half* __restrict__ x,
                                       float* __restrict__ y,
                                       int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const half* W_row = W + (size_t)row * K;
    float sum = 0.0f;

    // Process 2 elements per thread per iteration using half2
    const int K2 = K / 2;
    const half2* W2 = reinterpret_cast<const half2*>(W_row);
    const half2* x2 = reinterpret_cast<const half2*>(x);

    for (int i = lane; i < K2; i += 32) {
        half2 w = W2[i];
        half2 v = x2[i];
        sum += __half2float(w.x) * __half2float(v.x);
        sum += __half2float(w.y) * __half2float(v.y);
    }

    // Handle odd K (unlikely but safe)
    if ((K & 1) && lane == 0) {
        sum += __half2float(W_row[K - 1]) * __half2float(x[K - 1]);
    }

    // Warp shuffle reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[row] = sum;
}

void gemv_gate_fp32(const half* W, const half* x, float* y,
                    int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;
    gemv_gate_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(W, x, y, M, K);
}

// ---------------------------------------------------------------------------
// dp4a Q6_K × Q8_1 GEMV kernel
//
// Q6_K block = 210 bytes for 256 elements: ql[128] + qh[64] + scales[16] + d[2].
// Each warp computes one output row.
// Inner loop: for each Q6_K block (256 elems), dequant 8 groups of 32 ints,
// pack into int8x4, dp4a with pre-quantized Q8_1 input.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Q6_K × Q8_1 dp4a inner loop: processes one group (32 elements = 1 Q8_1 block)
// from a Q6_K block using dp4a. Returns the accumulated weighted sum.
//
// Q6_K memory layout per 256-element block:
//   ql[128]: lower 4 bits (8 groups of 32 elements, packed as nybbles)
//   qh[64]:  upper 2 bits (8 groups of 32 elements, packed 4 per byte)
//   sc[16]:  8-bit sub-block scales (16 sub-blocks of 16 elements)
//   d[2]:    FP16 block scale
//
// Group g (0-7) maps to elements [g*32, g*32+31]:
//   ql offset: (g/4)*64 + (g%2)*32, use high nybble if (g%4) >= 2
//   qh offset: (g<4)?0:32, shift by (g%4)*2
// ---------------------------------------------------------------------------
__device__ __forceinline__ float q6k_q8_1_dp4a_group(
        const uint8_t* __restrict__ ql,
        const uint8_t* __restrict__ qh,
        const int8_t* __restrict__ sc,
        float d_w,
        const int8_t* __restrict__ xqs,
        float d_x,
        int g) {
    const int ql_base = (g / 4) * 64 + (g % 2) * 32;
    const int is_high = ((g % 4) >= 2);
    const int qh_base = (g < 4) ? 0 : 32;
    const int qh_shift = (g % 4) * 2;

    float group_sum = 0.0f;

    // 2 sub-blocks of 16 elements each
    for (int sb = 0; sb < 2; sb++) {
        const int8_t sc_val = sc[2 * g + sb];
        const int sub_off = sb * 16;
        int32_t sumi = 0;

        // 4 dp4a operations per sub-block (16 elements / 4)
        #pragma unroll
        for (int d4 = 0; d4 < 4; d4++) {
            const int k = sub_off + d4 * 4;

            // Load 4 ql bytes and extract low/high nybbles
            uint32_t ql4;
            memcpy(&ql4, ql + ql_base + k, 4);
            const uint32_t lo4 = is_high ? ((ql4 >> 4) & 0x0F0F0F0FU)
                                         : (ql4 & 0x0F0F0F0FU);

            // Load 4 qh bytes and extract 2-bit fields
            uint32_t qh4;
            memcpy(&qh4, qh + qh_base + k, 4);
            const uint32_t hi4 = ((qh4 >> qh_shift) & 0x03030303U) << 4;

            // Pack 4 Q6_K values as int8 and subtract bias (32) per byte
            const int vi = __vsubss4(lo4 | hi4, 0x20202020U);

            // Load 4 Q8_1 values as packed int32
            int xi;
            memcpy(&xi, xqs + k, 4);

            sumi = __dp4a(vi, xi, sumi);
        }

        group_sum += d_w * d_x * (float)sc_val * (float)sumi;
    }

    return group_sum;
}

// ---------------------------------------------------------------------------
// Multi-row Q6_K dp4a GEMV: each warp processes N_ROWS output rows
// simultaneously, loading Q8_1 input once and reusing across all rows.
// This increases work per thread by N_ROWS×, improving memory latency hiding.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float q6k_dp4a_group_preloaded(
        const uint8_t* __restrict__ ql,
        const uint8_t* __restrict__ qh,
        const int8_t* __restrict__ sc,
        float d_w,
        const int* __restrict__ xqs_packed,  // [8] pre-loaded int32 from Q8_1
        float d_x,
        int g) {
    const int ql_base = (g / 4) * 64 + (g % 2) * 32;
    const int is_high = ((g % 4) >= 2);
    const int qh_base = (g < 4) ? 0 : 32;
    const int qh_shift = (g % 4) * 2;

    float group_sum = 0.0f;

    #pragma unroll
    for (int sb = 0; sb < 2; sb++) {
        const int8_t sc_val = sc[2 * g + sb];
        const int sub_off = sb * 16;
        int32_t sumi = 0;

        #pragma unroll
        for (int d4 = 0; d4 < 4; d4++) {
            const int k = sub_off + d4 * 4;

            uint32_t ql4;
            memcpy(&ql4, ql + ql_base + k, 4);
            const uint32_t lo4 = is_high ? ((ql4 >> 4) & 0x0F0F0F0FU)
                                         : (ql4 & 0x0F0F0F0FU);
            uint32_t qh4;
            memcpy(&qh4, qh + qh_base + k, 4);
            const uint32_t hi4 = ((qh4 >> qh_shift) & 0x03030303U) << 4;
            const int vi = __vsubss4(lo4 | hi4, 0x20202020U);
            sumi = __dp4a(vi, xqs_packed[sb * 4 + d4], sumi);
        }
        group_sum += d_w * d_x * (float)sc_val * (float)sumi;
    }
    return group_sum;
}

template<int N_ROWS, bool ADD_RESIDUAL>
__global__ void gemv_q6k_q8_1_kernel(const uint8_t* __restrict__ W,
                                      const block_q8_1* __restrict__ q8_1,
                                      const float* __restrict__ d8,
                                      half* y,
                                      const half* residual,
                                      int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    if (row_base >= M) return;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const int total_q8 = blocks_per_row * 8;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int q8_idx = lane; q8_idx < total_q8; q8_idx += 32) {
        const int q6k_blk = q8_idx / 8;
        const int g = q8_idx % 8;

        // Pre-load Q8_1 data into registers (shared across all rows)
        int xqs_packed[8];
        memcpy(xqs_packed, q8_1[q8_idx].qs, 32);
        float dq = d8[q8_idx];

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) break;
            const uint8_t* bp = W + (size_t)row * row_bytes + q6k_blk * 210;
            float d_w = __half2float(*(const half*)(bp + 208));
            sum[r] += q6k_dp4a_group_preloaded(
                bp, bp + 128, (const int8_t*)(bp + 192),
                d_w, xqs_packed, dq, g);
        }
    }

    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0) {
            const int row = row_base + r;
            if (row < M) {
                float s = sum[r];
                if constexpr (ADD_RESIDUAL) s += __half2float(residual[row]);
                y[row] = __float2half(s);
            }
        }
    }
}

// Choose N_ROWS to balance work-per-warp vs grid occupancy.
// More rows = better latency hiding but fewer blocks for parallelism.
static void launch_gemv_q6k_q8_1(const uint8_t* W, const block_q8_1* q8_1, const float* d8,
                                   half* y, const half* residual, bool add_residual,
                                   int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (M + rows_per_block - 1) / rows_per_block;
        if (add_residual)
            gemv_q6k_q8_1_kernel<NR, true><<<blocks, threads_per_block, 0, stream>>>(
                W, q8_1, d8, y, residual, M, K);
        else
            gemv_q6k_q8_1_kernel<NR, false><<<blocks, threads_per_block, 0, stream>>>(
                W, q8_1, d8, y, nullptr, M, K);
    };

    // Grid-size aware N_ROWS selection: use N_ROWS=2 only when it yields enough
    // blocks for good SM utilization (at least 256 blocks ≈ 1.5× SM count).
    // For O-proj (M=2048) with N_ROWS=2, blocks=128 → only 75% SM utilization.
    // Switching to N_ROWS=1 gives blocks=256 → 94% SM utilization.
    int nr2_blocks = (M + warps_per_block * 2 - 1) / (warps_per_block * 2);
    if (nr2_blocks >= 256) launch(std::integral_constant<int, 2>{});
    else                   launch(std::integral_constant<int, 1>{});
}

void gemv_q6k_q8_1(const void* W, const block_q8_1* q8_1, const float* d8,
                    half* y, int M, int K, cudaStream_t stream) {
    launch_gemv_q6k_q8_1(static_cast<const uint8_t*>(W), q8_1, d8,
                           y, nullptr, false, M, K, stream);
}

void gemv_q6k_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8,
                              half* y, const half* residual,
                              int M, int K, cudaStream_t stream) {
    launch_gemv_q6k_q8_1(static_cast<const uint8_t*>(W), q8_1, d8,
                           y, residual, true, M, K, stream);
}

// FP32-output Q6_K dp4a GEMV — for LM head (logits must be FP32 for sampling precision).
// Uses multi-row (N_ROWS=2) for large vocab to improve latency hiding.
template<int N_ROWS>
__global__ void gemv_q6k_q8_1_fp32_kernel(const uint8_t* __restrict__ W,
                                           const block_q8_1* __restrict__ q8_1,
                                           const float* __restrict__ d8,
                                           float* __restrict__ y,
                                           int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    if (row_base >= M) return;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const int total_q8 = blocks_per_row * 8;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int q8_idx = lane; q8_idx < total_q8; q8_idx += 32) {
        const int q6k_blk = q8_idx / 8;
        const int g = q8_idx % 8;
        int xqs_packed[8];
        memcpy(xqs_packed, q8_1[q8_idx].qs, 32);
        float dq = d8[q8_idx];

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) break;
            const uint8_t* bp = W + (size_t)row * row_bytes + q6k_blk * 210;
            float d_w = __half2float(*(const half*)(bp + 208));
            sum[r] += q6k_dp4a_group_preloaded(
                bp, bp + 128, (const int8_t*)(bp + 192), d_w, xqs_packed, dq, g);
        }
    }

    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0 && row_base + r < M) y[row_base + r] = sum[r];
    }
}

void gemv_q6k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8,
                          float* y, int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    // LM head: M=vocab_size (large), always use N_ROWS=2
    const int rows_per_block = warps_per_block * 2;
    const int blocks = (M + rows_per_block - 1) / rows_per_block;
    gemv_q6k_q8_1_fp32_kernel<2><<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(W), q8_1, d8, y, M, K);
}

// FP32-output Q8_0 dp4a GEMV — for LM head.
__global__ void gemv_q8_0_q8_1_fp32_kernel(const uint8_t* __restrict__ W,
                                            const block_q8_1* __restrict__ q8_1,
                                            const float* __restrict__ d8,
                                            float* __restrict__ y,
                                            int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row     = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const uint8_t* bp = W_row + b * 34;
        const float d_w = __half2float(*(const half*)bp);

        int wi;
        memcpy(&wi, bp + 2 + (0 * 4), 4);
        int xi0;
        memcpy(&xi0, q8_1[b].qs + 0, 4);
        int32_t sumi = __dp4a(wi, xi0, 0);

        memcpy(&wi, bp + 2 + (1 * 4), 4);
        int xi1;
        memcpy(&xi1, q8_1[b].qs + 4, 4);
        sumi = __dp4a(wi, xi1, sumi);

        memcpy(&wi, bp + 2 + (2 * 4), 4);
        int xi2;
        memcpy(&xi2, q8_1[b].qs + 8, 4);
        sumi = __dp4a(wi, xi2, sumi);

        memcpy(&wi, bp + 2 + (3 * 4), 4);
        int xi3;
        memcpy(&xi3, q8_1[b].qs + 12, 4);
        sumi = __dp4a(wi, xi3, sumi);

        memcpy(&wi, bp + 2 + (4 * 4), 4);
        int xi4;
        memcpy(&xi4, q8_1[b].qs + 16, 4);
        sumi = __dp4a(wi, xi4, sumi);

        memcpy(&wi, bp + 2 + (5 * 4), 4);
        int xi5;
        memcpy(&xi5, q8_1[b].qs + 20, 4);
        sumi = __dp4a(wi, xi5, sumi);

        memcpy(&wi, bp + 2 + (6 * 4), 4);
        int xi6;
        memcpy(&xi6, q8_1[b].qs + 24, 4);
        sumi = __dp4a(wi, xi6, sumi);

        memcpy(&wi, bp + 2 + (7 * 4), 4);
        int xi7;
        memcpy(&xi7, q8_1[b].qs + 28, 4);
        sumi = __dp4a(wi, xi7, sumi);

        sum += d_w * d8[b] * (float)sumi;
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[row] = sum;
}

void gemv_q8_0_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8,
                           float* y, int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;
    gemv_q8_0_q8_1_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(W), q8_1, d8, y, M, K);
}

// ---------------------------------------------------------------------------
// dp4a Q8_0 × Q8_1 GEMV kernel
//
// Q8_0 block = 34 bytes for 32 elements: d[2] + qs[32].
// Both weight and input are INT8, so dp4a is a natural fit.
// Each warp handles one row. Each thread processes one Q8_0 block per iteration,
// using dp4a to compute the 32-element dot product in 8 dp4a instructions.
// ---------------------------------------------------------------------------
// Multi-row Q8_0 dp4a GEMV: each warp processes N_ROWS simultaneously.
template<int N_ROWS, bool ADD_RESIDUAL>
__global__ void gemv_q8_0_q8_1_kernel(const uint8_t* __restrict__ W,
                                       const block_q8_1* __restrict__ q8_1,
                                       const float* __restrict__ d8,
                                       half* y,
                                       const half* residual,
                                       int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;

    if (row_base >= M) return;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        // Pre-load Q8_1 input data (shared across all rows)
        int xi[8];
        memcpy(xi, q8_1[b].qs, 32);
        float dq = d8[b];

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int row = row_base + r;
            if (row >= M) break;

            const uint8_t* bp = W + (size_t)row * row_bytes + b * 34;
            // Q8_0 blocks are 34 bytes (not 4-aligned), use memcpy for safe access
            half d_w_h;
            memcpy(&d_w_h, bp, sizeof(half));
            float d_w = __half2float(d_w_h);
            int wi[8];
            memcpy(wi, bp + 2, 32);

            int32_t sumi = 0;
            sumi = __dp4a(wi[0], xi[0], sumi);
            sumi = __dp4a(wi[1], xi[1], sumi);
            sumi = __dp4a(wi[2], xi[2], sumi);
            sumi = __dp4a(wi[3], xi[3], sumi);
            sumi = __dp4a(wi[4], xi[4], sumi);
            sumi = __dp4a(wi[5], xi[5], sumi);
            sumi = __dp4a(wi[6], xi[6], sumi);
            sumi = __dp4a(wi[7], xi[7], sumi);

            sum[r] += d_w * dq * (float)sumi;
        }
    }

    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0) {
            const int row = row_base + r;
            if (row < M) {
                float s = sum[r];
                if constexpr (ADD_RESIDUAL) s += __half2float(residual[row]);
                y[row] = __float2half(s);
            }
        }
    }
}

static void launch_gemv_q8_0_q8_1(const uint8_t* W, const block_q8_1* q8_1, const float* d8,
                                    half* y, const half* residual, bool add_residual,
                                    int M, int K, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    auto launch = [&](auto n_rows_tag) {
        constexpr int NR = decltype(n_rows_tag)::value;
        const int rows_per_block = warps_per_block * NR;
        const int blocks = (M + rows_per_block - 1) / rows_per_block;
        if (add_residual)
            gemv_q8_0_q8_1_kernel<NR, true><<<blocks, threads_per_block, 0, stream>>>(
                W, q8_1, d8, y, residual, M, K);
        else
            gemv_q8_0_q8_1_kernel<NR, false><<<blocks, threads_per_block, 0, stream>>>(
                W, q8_1, d8, y, nullptr, M, K);
    };

    if (M >= 1024) launch(std::integral_constant<int, 2>{});
    else           launch(std::integral_constant<int, 1>{});
}

void gemv_q8_0_q8_1(const void* W, const block_q8_1* q8_1, const float* d8,
                     half* y, int M, int K, cudaStream_t stream) {
    launch_gemv_q8_0_q8_1(static_cast<const uint8_t*>(W), q8_1, d8,
                            y, nullptr, false, M, K, stream);
}

void gemv_q8_0_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8,
                               half* y, const half* residual,
                               int M, int K, cudaStream_t stream) {
    launch_gemv_q8_0_q8_1(static_cast<const uint8_t*>(W), q8_1, d8,
                            y, residual, true, M, K, stream);
}

// ---------------------------------------------------------------------------
// Fused QKV GEMV: single kernel computes Q, K, V projections.
// Grid: ceil((q_rows + k_rows + v_rows) / warps_per_block) blocks.
// Each warp determines which projection it belongs to based on its global row.
// The pre-quantized Q8_1 input is shared — read once, used for all 3 projections.
// ---------------------------------------------------------------------------

template<int N_ROWS>
__global__ void gemv_qkv_fused_q6k_q8_1_kernel(
        const uint8_t* __restrict__ W_q,
        const uint8_t* __restrict__ W_k,
        const uint8_t* __restrict__ W_v,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_q,
        half* __restrict__ y_k,
        half* __restrict__ y_v,
        int q_rows, int k_rows, int v_rows, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int row_base = (blockIdx.x * warps_per_block + warp_id) * N_ROWS;
    const int total_rows = q_rows + k_rows + v_rows;

    if (row_base >= total_rows) return;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const int total_q8 = blocks_per_row * 8;

    float sum[N_ROWS];
    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) sum[r] = 0.0f;

    for (int q8_idx = lane; q8_idx < total_q8; q8_idx += 32) {
        const int q6k_blk = q8_idx / 8;
        const int g = q8_idx % 8;

        // Load Q8_1 data once (shared across all N_ROWS rows)
        int xqs_packed[8];
        memcpy(xqs_packed, q8_1[q8_idx].qs, 32);
        float dq = d8[q8_idx];

        #pragma unroll
        for (int r = 0; r < N_ROWS; r++) {
            const int global_row = row_base + r;
            if (global_row >= total_rows) break;

            // Determine which projection and local row
            const uint8_t* W;
            int local_row;
            if (global_row < q_rows) {
                local_row = global_row;
                W = W_q;
            } else if (global_row < q_rows + k_rows) {
                local_row = global_row - q_rows;
                W = W_k;
            } else {
                local_row = global_row - q_rows - k_rows;
                W = W_v;
            }

            const uint8_t* bp = W + (size_t)local_row * row_bytes + q6k_blk * 210;
            float d_w = __half2float(*(const half*)(bp + 208));
            sum[r] += q6k_dp4a_group_preloaded(
                bp, bp + 128, (const int8_t*)(bp + 192), d_w, xqs_packed, dq, g);
        }
    }

    #pragma unroll
    for (int r = 0; r < N_ROWS; r++) {
        for (int off = 16; off > 0; off >>= 1)
            sum[r] += __shfl_down_sync(0xFFFFFFFF, sum[r], off);
        if (lane == 0) {
            const int global_row = row_base + r;
            if (global_row >= total_rows) break;
            half* y;
            int local_row;
            if (global_row < q_rows) {
                local_row = global_row;
                y = y_q;
            } else if (global_row < q_rows + k_rows) {
                local_row = global_row - q_rows;
                y = y_k;
            } else {
                local_row = global_row - q_rows - k_rows;
                y = y_v;
            }
            y[local_row] = __float2half(sum[r]);
        }
    }
}

void gemv_qkv_fused_q6k_q8_1(const void* W_q, const void* W_k, const void* W_v,
                               const block_q8_1* q8_1, const float* d8,
                               half* y_q, half* y_k, half* y_v,
                               int q_rows, int k_rows, int v_rows, int K,
                               cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int total = q_rows + k_rows + v_rows;
    // Use N_ROWS=2 when grid is large enough (>= 256 blocks)
    int nr2_blocks = (total + warps_per_block * 2 - 1) / (warps_per_block * 2);
    if (nr2_blocks >= 256) {
        gemv_qkv_fused_q6k_q8_1_kernel<2><<<nr2_blocks, threads_per_block, 0, stream>>>(
            static_cast<const uint8_t*>(W_q),
            static_cast<const uint8_t*>(W_k),
            static_cast<const uint8_t*>(W_v),
            q8_1, d8, y_q, y_k, y_v,
            q_rows, k_rows, v_rows, K);
    } else {
        int blocks = (total + warps_per_block - 1) / warps_per_block;
        gemv_qkv_fused_q6k_q8_1_kernel<1><<<blocks, threads_per_block, 0, stream>>>(
            static_cast<const uint8_t*>(W_q),
            static_cast<const uint8_t*>(W_k),
            static_cast<const uint8_t*>(W_v),
            q8_1, d8, y_q, y_k, y_v,
            q_rows, k_rows, v_rows, K);
    }
}

__global__ void gemv_qkv_fused_q8_0_q8_1_kernel(
        const uint8_t* __restrict__ W_q,
        const uint8_t* __restrict__ W_k,
        const uint8_t* __restrict__ W_v,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_q,
        half* __restrict__ y_k,
        half* __restrict__ y_v,
        int q_rows, int k_rows, int v_rows, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int global_row = blockIdx.x * warps_per_block + warp_id;
    const int total_rows = q_rows + k_rows + v_rows;

    if (global_row >= total_rows) return;

    const uint8_t* W;
    half* y;
    int local_row;

    if (global_row < q_rows) {
        local_row = global_row;
        y = y_q;
    } else if (global_row < q_rows + k_rows) {
        local_row = global_row - q_rows;
        y = y_k;
    } else {
        local_row = global_row - q_rows - k_rows;
        y = y_v;
    }

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;

    if (global_row < q_rows) W = W_q + (size_t)local_row * row_bytes;
    else if (global_row < q_rows + k_rows) W = W_k + (size_t)local_row * row_bytes;
    else W = W_v + (size_t)local_row * row_bytes;

    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const uint8_t* bp = W + b * 34;
        // Q8_0 blocks are 34 bytes (not 4-aligned), use memcpy for safe access
        half d_w_h;
        memcpy(&d_w_h, bp, sizeof(half));
        float d_w = __half2float(d_w_h);
        int wi[8];
        memcpy(wi, bp + 2, 32);
        int xi[8];
        memcpy(xi, q8_1[b].qs, 32);

        int32_t sumi = 0;
        sumi = __dp4a(wi[0], xi[0], sumi);
        sumi = __dp4a(wi[1], xi[1], sumi);
        sumi = __dp4a(wi[2], xi[2], sumi);
        sumi = __dp4a(wi[3], xi[3], sumi);
        sumi = __dp4a(wi[4], xi[4], sumi);
        sumi = __dp4a(wi[5], xi[5], sumi);
        sumi = __dp4a(wi[6], xi[6], sumi);
        sumi = __dp4a(wi[7], xi[7], sumi);

        sum += d_w * d8[b] * (float)sumi;
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[local_row] = __float2half(sum);
}

void gemv_qkv_fused_q8_0_q8_1(const void* W_q, const void* W_k, const void* W_v,
                                const block_q8_1* q8_1, const float* d8,
                                half* y_q, half* y_k, half* y_v,
                                int q_rows, int k_rows, int v_rows, int K,
                                cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int total = q_rows + k_rows + v_rows;
    const int blocks = (total + warps_per_block - 1) / warps_per_block;
    gemv_qkv_fused_q8_0_q8_1_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(W_q),
        static_cast<const uint8_t*>(W_k),
        static_cast<const uint8_t*>(W_v),
        q8_1, d8, y_q, y_k, y_v,
        q_rows, k_rows, v_rows, K);
}

// ---------------------------------------------------------------------------
// dp4a MoE decode GEMV variants (Q6_K × Q8_1 and Q8_0 × Q8_1)
// ---------------------------------------------------------------------------

__global__ void gemv_q6k_q8_1_moe_decode_kernel(
        const uint8_t* __restrict__ packed_weights,
        const int32_t* __restrict__ expert_indices,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y,
        int rows, int K,
        size_t expert_stride_bytes,
        int q8_1_stride,
        int d8_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;
    const int total_q8 = blocks_per_row * 8;
    float sum = 0.0f;

    for (int q8_idx = lane; q8_idx < total_q8; q8_idx += 32) {
        const int q6k_blk = q8_idx / 8;
        const int g = q8_idx % 8;
        int xqs_packed[8];
        memcpy(xqs_packed, x_q8[q8_idx].qs, 32);
        float dq = x_d8[q8_idx];

        const uint8_t* bp = W_row + q6k_blk * 210;
        float d_w = __half2float(*(const half*)(bp + 208));
        sum += q6k_dp4a_group_preloaded(
            bp, bp + 128, (const int8_t*)(bp + 192), d_w, xqs_packed, dq, g);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_q8_1_moe_decode(const void* packed_weights,
                                const int32_t* expert_indices,
                                const block_q8_1* q8_1, const float* d8,
                                half* y, int rows, int K,
                                size_t expert_stride_bytes,
                                int q8_1_stride, int d8_stride, int top_k,
                                cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    const int total_blocks = top_k * blocks_per_expert;
    gemv_q6k_q8_1_moe_decode_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        expert_indices, q8_1, d8, y, rows, K,
        expert_stride_bytes, q8_1_stride, d8_stride, blocks_per_expert);
}

__global__ void gemv_q8_0_q8_1_moe_decode_kernel(
        const uint8_t* __restrict__ packed_weights,
        const int32_t* __restrict__ expert_indices,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y,
        int rows, int K,
        size_t expert_stride_bytes,
        int q8_1_stride,
        int d8_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;
    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const uint8_t* bp = W_row + b * 34;
        // Q8_0 blocks are 34 bytes (not 4-aligned), use memcpy for safe access
        half d_w_h;
        memcpy(&d_w_h, bp, sizeof(half));
        float d_w = __half2float(d_w_h);
        int wi[8];
        memcpy(wi, bp + 2, 32);
        int xi[8];
        memcpy(xi, x_q8[b].qs, 32);

        int32_t sumi = 0;
        sumi = __dp4a(wi[0], xi[0], sumi);
        sumi = __dp4a(wi[1], xi[1], sumi);
        sumi = __dp4a(wi[2], xi[2], sumi);
        sumi = __dp4a(wi[3], xi[3], sumi);
        sumi = __dp4a(wi[4], xi[4], sumi);
        sumi = __dp4a(wi[5], xi[5], sumi);
        sumi = __dp4a(wi[6], xi[6], sumi);
        sumi = __dp4a(wi[7], xi[7], sumi);

        sum += d_w * x_d8[b] * (float)sumi;
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_q8_1_moe_decode(const void* packed_weights,
                                 const int32_t* expert_indices,
                                 const block_q8_1* q8_1, const float* d8,
                                 half* y, int rows, int K,
                                 size_t expert_stride_bytes,
                                 int q8_1_stride, int d8_stride, int top_k,
                                 cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    const int total_blocks = top_k * blocks_per_expert;
    gemv_q8_0_q8_1_moe_decode_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        expert_indices, q8_1, d8, y, rows, K,
        expert_stride_bytes, q8_1_stride, d8_stride, blocks_per_expert);
}

// ---------------------------------------------------------------------------
// Fused gate+up MoE GEMV: computes both gate and up projections in a single
// kernel launch. blockIdx.y selects projection: 0=gate, 1=up.
// Saves one kernel launch per MoE layer (48 launches for Qwen3-Coder).
// ---------------------------------------------------------------------------

__global__ void gemv_q6k_moe_gate_up_fused_kernel(
        const uint8_t* __restrict__ gate_weights,
        const uint8_t* __restrict__ up_weights,
        const int32_t* __restrict__ expert_indices,
        const half* __restrict__ x,
        half* __restrict__ y_gate,
        half* __restrict__ y_up,
        int rows, int K,
        size_t gate_stride_bytes,
        size_t up_stride_bytes,
        int x_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    // blockIdx.y: 0 = gate, 1 = up
    const bool is_up = (blockIdx.y == 1);
    const uint8_t* packed = is_up ? up_weights : gate_weights;
    size_t stride = is_up ? up_stride_bytes : gate_stride_bytes;
    half* y = is_up ? y_up : y_gate;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed + (size_t)expert_id * stride;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 210;
        const uint8_t* ql = bp;
        const uint8_t* qh = bp + 128;
        const int8_t* sc  = (const int8_t*)(bp + 192);
        float d = __half2float(*(const half*)(bp + 208));
        const int base = b * 256;

        uint8_t ql_a = ql[lane];
        uint8_t ql_b = ql[lane + 32];
        uint8_t ql_c = ql[64 + lane];
        uint8_t ql_d = ql[64 + lane + 32];
        uint8_t qh0  = qh[lane];
        uint8_t qh1  = qh[32 + lane];

        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        int sc_idx = lane >> 4;
        sum += d * (
            (float)sc[sc_idx]      * (float)q0 * __half2float(x_ptr[base + lane]) +
            (float)sc[sc_idx + 2]  * (float)q1 * __half2float(x_ptr[base + lane + 32]) +
            (float)sc[sc_idx + 4]  * (float)q2 * __half2float(x_ptr[base + lane + 64]) +
            (float)sc[sc_idx + 6]  * (float)q3 * __half2float(x_ptr[base + lane + 96]) +
            (float)sc[sc_idx + 8]  * (float)q4 * __half2float(x_ptr[base + lane + 128]) +
            (float)sc[sc_idx + 10] * (float)q5 * __half2float(x_ptr[base + lane + 160]) +
            (float)sc[sc_idx + 12] * (float)q6 * __half2float(x_ptr[base + lane + 192]) +
            (float)sc[sc_idx + 14] * (float)q7 * __half2float(x_ptr[base + lane + 224]));
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices, const half* x,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int x_stride, int top_k, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q6k_moe_gate_up_fused_kernel<<<grid, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights),
        static_cast<const uint8_t*>(up_weights),
        expert_indices, x, y_gate, y_up, rows, K,
        gate_stride_bytes, up_stride_bytes, x_stride, blocks_per_expert);
}

__global__ void gemv_q8_0_moe_gate_up_fused_kernel(
        const uint8_t* __restrict__ gate_weights,
        const uint8_t* __restrict__ up_weights,
        const int32_t* __restrict__ expert_indices,
        const half* __restrict__ x,
        half* __restrict__ y_gate,
        half* __restrict__ y_up,
        int rows, int K,
        size_t gate_stride_bytes,
        size_t up_stride_bytes,
        int x_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* packed = is_up ? up_weights : gate_weights;
    size_t stride = is_up ? up_stride_bytes : gate_stride_bytes;
    half* y = is_up ? y_up : y_gate;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed + (size_t)expert_id * stride;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 34;
        float d = __half2float(*(const half*)bp);
        int8_t q = ((const int8_t*)(bp + 2))[lane];
        sum += d * (float)q * __half2float(x_ptr[b * 32 + lane]);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices, const half* x,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int x_stride, int top_k, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q8_0_moe_gate_up_fused_kernel<<<grid, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights),
        static_cast<const uint8_t*>(up_weights),
        expert_indices, x, y_gate, y_up, rows, K,
        gate_stride_bytes, up_stride_bytes, x_stride, blocks_per_expert);
}

// dp4a Q8_1 variants of fused gate+up

__global__ void gemv_q6k_q8_1_moe_gate_up_fused_kernel(
        const uint8_t* __restrict__ gate_weights,
        const uint8_t* __restrict__ up_weights,
        const int32_t* __restrict__ expert_indices,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_gate,
        half* __restrict__ y_up,
        int rows, int K,
        size_t gate_stride_bytes,
        size_t up_stride_bytes,
        int q8_1_stride,
        int d8_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* packed = is_up ? up_weights : gate_weights;
    size_t stride = is_up ? up_stride_bytes : gate_stride_bytes;
    half* y = is_up ? y_up : y_gate;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed + (size_t)expert_id * stride;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;
    const int total_q8 = blocks_per_row * 8;
    float sum = 0.0f;

    for (int q8_idx = lane; q8_idx < total_q8; q8_idx += 32) {
        const int q6k_blk = q8_idx / 8;
        const int g = q8_idx % 8;
        int xqs_packed[8];
        memcpy(xqs_packed, x_q8[q8_idx].qs, 32);
        float dq = x_d8[q8_idx];

        const uint8_t* bp = W_row + q6k_blk * 210;
        float d_w = __half2float(*(const half*)(bp + 208));
        sum += q6k_dp4a_group_preloaded(
            bp, bp + 128, (const int8_t*)(bp + 192), d_w, xqs_packed, dq, g);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_q8_1_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices,
        const block_q8_1* q8_1, const float* d8,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int q8_1_stride, int d8_stride, int top_k,
        cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q6k_q8_1_moe_gate_up_fused_kernel<<<grid, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights),
        static_cast<const uint8_t*>(up_weights),
        expert_indices, q8_1, d8, y_gate, y_up, rows, K,
        gate_stride_bytes, up_stride_bytes,
        q8_1_stride, d8_stride, blocks_per_expert);
}

__global__ void gemv_q8_0_q8_1_moe_gate_up_fused_kernel(
        const uint8_t* __restrict__ gate_weights,
        const uint8_t* __restrict__ up_weights,
        const int32_t* __restrict__ expert_indices,
        const block_q8_1* __restrict__ q8_1,
        const float* __restrict__ d8,
        half* __restrict__ y_gate,
        half* __restrict__ y_up,
        int rows, int K,
        size_t gate_stride_bytes,
        size_t up_stride_bytes,
        int q8_1_stride,
        int d8_stride,
        int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows) return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* packed = is_up ? up_weights : gate_weights;
    size_t stride = is_up ? up_stride_bytes : gate_stride_bytes;
    half* y = is_up ? y_up : y_gate;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed + (size_t)expert_id * stride;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const block_q8_1* x_q8 = q8_1 + expert_slot * q8_1_stride;
    const float* x_d8 = d8 + expert_slot * d8_stride;
    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const uint8_t* bp = W_row + b * 34;
        // Q8_0 blocks are 34 bytes (not 4-aligned), use memcpy for safe access
        half d_w_h;
        memcpy(&d_w_h, bp, sizeof(half));
        float d_w = __half2float(d_w_h);
        int wi[8];
        memcpy(wi, bp + 2, 32);
        int xi[8];
        memcpy(xi, x_q8[b].qs, 32);

        int32_t sumi = 0;
        sumi = __dp4a(wi[0], xi[0], sumi);
        sumi = __dp4a(wi[1], xi[1], sumi);
        sumi = __dp4a(wi[2], xi[2], sumi);
        sumi = __dp4a(wi[3], xi[3], sumi);
        sumi = __dp4a(wi[4], xi[4], sumi);
        sumi = __dp4a(wi[5], xi[5], sumi);
        sumi = __dp4a(wi[6], xi[6], sumi);
        sumi = __dp4a(wi[7], xi[7], sumi);

        sum += d_w * x_d8[b] * (float)sumi;
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_q8_1_moe_gate_up_fused(
        const void* gate_weights, const void* up_weights,
        const int32_t* expert_indices,
        const block_q8_1* q8_1, const float* d8,
        half* y_gate, half* y_up,
        int rows, int K,
        size_t gate_stride_bytes, size_t up_stride_bytes,
        int q8_1_stride, int d8_stride, int top_k,
        cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_per_expert = (rows + warps_per_block - 1) / warps_per_block;
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q8_0_q8_1_moe_gate_up_fused_kernel<<<grid, threads_per_block, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights),
        static_cast<const uint8_t*>(up_weights),
        expert_indices, q8_1, d8, y_gate, y_up, rows, K,
        gate_stride_bytes, up_stride_bytes,
        q8_1_stride, d8_stride, blocks_per_expert);
}

// ---------------------------------------------------------------------------
// FP8 E4M3 GEMV
// ---------------------------------------------------------------------------

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
