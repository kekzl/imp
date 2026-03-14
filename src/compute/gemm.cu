#include "compute/gemm.h"
#include "runtime/pdl.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>

#define CUBLASLT_CHECK(call) do { \
    cublasStatus_t _st = (call); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "imp::gemm: %s failed (status %d)\n", #call, (int)_st); \
    } \
} while(0)

namespace imp {

#if IMP_CUDA_13_1
static constexpr auto kGemmAlgo = CUBLAS_GEMM_AUTOTUNE;
#else
static constexpr auto kGemmAlgo = CUBLAS_GEMM_DEFAULT;
#endif

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

// Static benchmark scratch buffer for algo selection (allocated once in gemm_init).
// Avoids per-cache-miss cudaMalloc/cudaFree which fragment GPU memory.
static void* s_bench_scratch = nullptr;
static size_t s_bench_scratch_size = 0;
static constexpr size_t kBenchScratchSize = 32ULL << 20;  // 32 MiB

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

    // Pre-allocate benchmark scratch buffer for algo selection.
    if (!s_bench_scratch) {
        if (cudaMalloc(&s_bench_scratch, kBenchScratchSize) == cudaSuccess) {
            s_bench_scratch_size = kBenchScratchSize;
        }
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
    bool has_scales;  // FP8 scale pointers present (affects opDesc attributes)

    bool operator==(const GemmCacheKey& o) const {
        return dtA == o.dtA && dtB == o.dtB && dtC == o.dtC &&
               compute == o.compute && M == o.M && K == o.K && N == o.N &&
               has_scales == o.has_scales;
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
        mix(static_cast<uint64_t>(k.has_scales));
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

// ---------------------------------------------------------------------------
// Algorithm benchmarking: request top-N candidates, time each, pick fastest.
// Uses a temporary output buffer to avoid corrupting C during live inference.
// Eliminates 2.6x prefill variance from non-deterministic cuBLAS autotuning.
// ---------------------------------------------------------------------------
static constexpr int kMaxAlgoCandidates = 8;
static constexpr int kBenchmarkIters = 5;

static void benchmark_and_select_algo(
        cublasLtHandle_t lt, GemmCacheEntry& entry,
        const void* A_data, const void* B_data, size_t C_bytes,
        float alpha, float beta, bool is_int_compute, cudaStream_t stream) {
    cublasLtMatmulPreference_t pref = nullptr;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &s_workspace_size, sizeof(s_workspace_size)));

    cublasLtMatmulHeuristicResult_t results[kMaxAlgoCandidates];
    int nresults = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, entry.opDesc, entry.Bdesc,
        entry.Adesc, entry.Cdesc, entry.Cdesc,
        pref, kMaxAlgoCandidates, results, &nresults);
    cublasLtMatmulPreferenceDestroy(pref);

    if (nresults <= 0) { entry.has_algo = false; entry.workspace_size = 0; return; }
    if (nresults == 1) {
        entry.algo = results[0].algo;
        entry.workspace_size = (results[0].workspaceSize <= s_workspace_size)
                                   ? results[0].workspaceSize : 0;
        entry.has_algo = true;
        return;
    }

    // Use pre-allocated scratch buffer to avoid fragmenting GPU memory
    if (!s_bench_scratch || C_bytes > s_bench_scratch_size) {
        entry.algo = results[0].algo;
        entry.workspace_size = (results[0].workspaceSize <= s_workspace_size)
                                   ? results[0].workspaceSize : 0;
        entry.has_algo = true;
        return;
    }
    void* temp_c = s_bench_scratch;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best_ms = 1e30f;
    int best_idx = 0;

    for (int i = 0; i < nresults; i++) {
        if (results[i].workspaceSize > s_workspace_size) continue;
        float zero = 0.0f;
        // Warmup
        cublasLtMatmul(lt, entry.opDesc, &alpha, B_data, entry.Bdesc,
            A_data, entry.Adesc, &zero, temp_c, entry.Cdesc,
            temp_c, entry.Cdesc, &results[i].algo,
            s_workspace, results[i].workspaceSize, stream);
        // Timed
        cudaEventRecord(start, stream);
        for (int r = 0; r < kBenchmarkIters; r++)
            cublasLtMatmul(lt, entry.opDesc, &alpha, B_data, entry.Bdesc,
                A_data, entry.Adesc, &zero, temp_c, entry.Cdesc,
                temp_c, entry.Cdesc, &results[i].algo,
                s_workspace, results[i].workspaceSize, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) { best_ms = ms; best_idx = i; }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    entry.algo = results[best_idx].algo;
    entry.workspace_size = results[best_idx].workspaceSize;
    entry.has_algo = true;
}

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

    GemmCacheKey cache_key{cuda_dtype_A, cuda_dtype_B, cuda_dtype_C, compute_type, M, K, N, false};

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

            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&new_entry.Bdesc, cuda_dtype_B, (int)K, (int)N, (int)K));
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&new_entry.Adesc, cuda_dtype_A, (int)K, (int)M, (int)K));
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&new_entry.Cdesc, cuda_dtype_C, (int)N, (int)M, (int)N));

            size_t c_bytes = (size_t)M * N * dtype_size(C.dtype);
            benchmark_and_select_algo(lt, new_entry,
                A.data, B.data, c_bytes, alpha, beta,
                (compute_type == CUBLAS_COMPUTE_32I), stream);

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

#if __CUDA_ARCH__ >= 1200
    // Blackwell (sm_120+): 256-bit loads via paired float4 (16 halves per iteration).
    // 2× wider than the default 128-bit path, better saturating memory bandwidth.
    const int K_vec16 = K / 16;  // 16 halves = 32 bytes = 2 × sizeof(float4)
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v     = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec16; i += 32) {
        float4 a0 = A_row_v[2*i];
        float4 a1 = A_row_v[2*i + 1];
        float4 x0 = x_v[2*i];
        float4 x1 = x_v[2*i + 1];

        const half2* a_h2_0 = reinterpret_cast<const half2*>(&a0);
        const half2* x_h2_0 = reinterpret_cast<const half2*>(&x0);
        const half2* a_h2_1 = reinterpret_cast<const half2*>(&a1);
        const half2* x_h2_1 = reinterpret_cast<const half2*>(&x1);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2_0[j], x_h2_0[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2_1[j], x_h2_1[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
    }

    // Handle elements between K_vec16*16 and K_vec8*8 (0 or 8 elements).
    int base16 = K_vec16 * 16;
    if (base16 + 8 <= K) {
        int K_vec8_rem = (K - base16) / 8;
        const float4* A_rem = reinterpret_cast<const float4*>(A_row + base16);
        const float4* x_rem = reinterpret_cast<const float4*>(x + base16);
        for (int i = lane; i < K_vec8_rem; i += 32) {
            float4 a_raw = A_rem[i];
            float4 x_raw = x_rem[i];
            const half2* a_h2 = reinterpret_cast<const half2*>(&a_raw);
            const half2* x_h2 = reinterpret_cast<const half2*>(&x_raw);
            for (int j = 0; j < 4; ++j) {
                half2 prod = __hmul2(a_h2[j], x_h2[j]);
                sum += __half2float(prod.x) + __half2float(prod.y);
            }
        }
        base16 = base16 + K_vec8_rem * 8;
    }

    // Scalar remainder.
    for (int i = base16 + lane; i < K; i += 32) {
        sum += __half2float(A_row[i]) * __half2float(x[i]);
    }
#else
    // Default path: 128-bit loads (8 halves per float4).
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
#endif

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

    // Cache key: (M, K, N, dtypes, beta) — scale pointers vary per-call but
    // don't affect descriptor/algo selection, only set via opDesc attribute.
    cudaDataType_t cuda_dtype_A = dtype_to_cuda(A.dtype);
    cudaDataType_t cuda_dtype_B = dtype_to_cuda(B.dtype);
    cudaDataType_t cuda_dtype_C = dtype_to_cuda(C.dtype);
    GemmCacheKey cache_key{cuda_dtype_A, cuda_dtype_B, cuda_dtype_C, CUBLAS_COMPUTE_32F,
                           M, K, N, (aScale != nullptr)};

    GemmCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
        auto it = s_gemm_cache.find(cache_key);
        if (it != s_gemm_cache.end()) {
            entry = &it->second;
        } else {
            GemmCacheEntry new_entry{};
            cublasLtMatmulDescCreate(&new_entry.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

            cublasOperation_t transA = CUBLAS_OP_T;
            cublasOperation_t transB = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(new_entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transA, sizeof(transA));
            cublasLtMatmulDescSetAttribute(new_entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transB, sizeof(transB));

            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&new_entry.Bdesc, cuda_dtype_B, (int)K, (int)N, (int)K));
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&new_entry.Adesc, cuda_dtype_A, (int)K, (int)M, (int)K));
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&new_entry.Cdesc, cuda_dtype_C, (int)N, (int)M, (int)N));

            // Set scale pointers before benchmarking so FP8 algos run correctly
            if (aScale) {
                cublasLtMatmulDescSetAttribute(new_entry.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &aScale, sizeof(aScale));
            }
            if (bScale) {
                cublasLtMatmulDescSetAttribute(new_entry.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &bScale, sizeof(bScale));
            }

            size_t c_bytes = (size_t)M * N * dtype_size(C.dtype);
            benchmark_and_select_algo(lt, new_entry,
                A.data, B.data, c_bytes, alpha, beta, false, stream);

            auto [ins_it, _] = s_gemm_cache.emplace(cache_key, new_entry);
            entry = &ins_it->second;
        }
    }

    // Set per-call scale pointers (vary by weight tensor, not cached)
    if (aScale) {
        cublasLtMatmulDescSetAttribute(entry->opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                        &aScale, sizeof(aScale));
    }
    if (bScale) {
        cublasLtMatmulDescSetAttribute(entry->opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                        &bScale, sizeof(bScale));
    }

    cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc,
        &alpha, B.data, entry->Bdesc, A.data, entry->Adesc,
        &beta,  C.data, entry->Cdesc, C.data, entry->Cdesc,
        entry->has_algo ? &entry->algo : nullptr,
        s_workspace, entry->workspace_size, stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm_cublaslt: cublasLtMatmul failed (status %d)\n", (int)st);
    }
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
// Fused gate+up MoE GEMV (scalar FP16 variants — NOT dp4a, kept as-is)
// ---------------------------------------------------------------------------


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

// ---------------------------------------------------------------------------
// Batched K/V projection via cublasGemmStridedBatchedEx
// ---------------------------------------------------------------------------

void gemm_kv_batched(const Tensor& input, const Tensor& weight_kv,
                     Tensor& k_out, Tensor& v_out, cudaStream_t stream) {
    int M = static_cast<int>(input.shape[0]);       // n_tokens
    int K = static_cast<int>(input.shape[1]);       // d_model
    int N = static_cast<int>(k_out.shape[1]);       // nkv * hd

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cudaDataType_t dt = dtype_to_cuda(input.dtype);
    float alpha = 1.0f, beta = 0.0f;

    // Col-major interpretation (same trick as gemm()):
    //   weight [N,K] row-major = [K,N] col-major; CUBLAS_OP_T → [N,K]
    //   input  [M,K] row-major = [K,M] col-major; CUBLAS_OP_N
    //   result [N,M] col-major = [M,N] row-major
    long long weight_stride = static_cast<long long>(N) * K;  // stride between wk and wv in weight_kv
    long long output_stride = static_cast<long long>(M) * N;  // stride between k_out and v_out

    cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,                                    // cuBLAS m, n, k
        &alpha,
        weight_kv.data, dt, K,                      // A (weight), lda=K
        weight_stride,                               // strideA: offset to wv
        input.data, dt, K,                           // B (input), ldb=K
        0,                                           // strideB: 0 (same input for both)
        &beta,
        k_out.data, dt, N,                           // C (output), ldc=N
        output_stride,                               // strideC: offset to v_out
        2,                                           // batch_count = 2 (K and V)
        CUBLAS_COMPUTE_32F,
        kGemmAlgo);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm_kv_batched: cublasGemmStridedBatchedEx failed (status %d)\n",
                (int)st);
    }
}

void gemm_pair_batched(const Tensor& input, const Tensor& weight_fused,
                       Tensor& out1, Tensor& out2, cudaStream_t stream) {
    int M = static_cast<int>(input.shape[0]);       // n_tokens
    int K = static_cast<int>(input.shape[1]);       // d_model
    int N = static_cast<int>(out1.shape[1]);        // d_ff (or nkv*hd)

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cudaDataType_t dt = dtype_to_cuda(input.dtype);
    float alpha = 1.0f, beta = 0.0f;

    long long weight_stride = static_cast<long long>(N) * K;
    // Compute actual byte offset between out1 and out2, then convert to element offset
    long long output_stride = (static_cast<const char*>(out2.data) -
                               static_cast<const char*>(out1.data)) /
                              dtype_size(input.dtype);

    cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight_fused.data, dt, K,
        weight_stride,
        input.data, dt, K,
        0,
        &beta,
        out1.data, dt, N,
        output_stride,
        2,
        CUBLAS_COMPUTE_32F,
        kGemmAlgo);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm_pair_batched: cublasGemmStridedBatchedEx failed (status %d)\n",
                (int)st);
    }
}


} // namespace imp
