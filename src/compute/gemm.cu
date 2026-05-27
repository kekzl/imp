#include "compute/gemm.h"
#include "compute/gemm_capture_fp16_sm120.h"
#include "core/logging.h"
#include "runtime/pdl.h"
#include "runtime/process_diag.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "quant/fp8_quant.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <vector>

#define CUBLASLT_CHECK(call)                                                        \
    do {                                                                            \
        cublasStatus_t _st = (call);                                                \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                         \
            fprintf(stderr, "imp::gemm: %s failed (status %d)\n", #call, (int)_st); \
        }                                                                           \
    } while (0)

namespace imp {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Warp-level sum reduction via __shfl_down_sync. Result valid in lane 0 only.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// GEMV launch constants: 256 threads = 8 warps per block.
static constexpr int kGemvThreads = 256;
static constexpr int kGemvWarps = kGemvThreads / 32;

// Compute the number of blocks needed to cover M rows at kGemvWarps rows/block.
static inline int gemv_blocks(int M) { return (M + kGemvWarps - 1) / kGemvWarps; }

static constexpr auto kGemmAlgo = CUBLAS_GEMM_AUTOTUNE;

// ---------------------------------------------------------------------------
// cuBLAS / cuBLASLt handles (lazily initialized)
// ---------------------------------------------------------------------------
static cublasHandle_t s_cublas_handle = nullptr;
static cublasLtHandle_t s_cublaslt_handle = nullptr;

static cublasHandle_t get_cublas_handle() {
    if (!s_cublas_handle) {
        cublasStatus_t st = cublasCreate(&s_cublas_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm: cublasCreate failed (status %d)\n", (int)st);
            abort();
        }
        cublasSetMathMode(s_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return s_cublas_handle;
}

static cublasLtHandle_t get_cublaslt_handle() {
    if (!s_cublaslt_handle) {
        cublasStatus_t st = cublasLtCreate(&s_cublaslt_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::gemm: cublasLtCreate failed (status %d)\n", (int)st);
            abort();
        }
    }
    return s_cublaslt_handle;
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
            64ULL << 20,  // 64 MiB — RTX 5090 (32 GB) has headroom
            32ULL << 20,  // 32 MiB
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
            fprintf(stderr, "imp::gemm: unsupported dtype %d\n", (int)dt);
            return CUDA_R_16F;  // fallback (caller guard should prevent reaching here)
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
            return CUBLAS_COMPUTE_32F;  // accumulate in FP32 for accuracy
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

// ---------------------------------------------------------------------------
// Bucket M for cache key: exact for decode (M<=1), bucketed for prefill.
// cuBLASLt algorithm selection is stable across nearby M values.
// ---------------------------------------------------------------------------
static int64_t bucket_m(int64_t m) {
    if (m <= 1)
        return m;  // decode: exact
    if (m <= 64)
        return 64;
    if (m <= 128)
        return 128;
    if (m <= 256)
        return 256;
    if (m <= 512)
        return 512;
    // For larger M, round up to next multiple of 128
    return ((m + 127) / 128) * 128;
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
        return dtA == o.dtA && dtB == o.dtB && dtC == o.dtC && compute == o.compute && M == o.M && K == o.K &&
               N == o.N && has_scales == o.has_scales;
    }
};

struct GemmCacheKeyHash {
    size_t operator()(const GemmCacheKey& k) const {
        size_t h = 14695981039346656037ULL;
        auto mix = [&](uint64_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
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
    int64_t desc_M;  // M dimension baked into layout descriptors
};

static std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheKeyHash> s_gemm_cache;
static std::mutex s_gemm_cache_mutex;

// ---------------------------------------------------------------------------
// cuBLASLt descriptor creation helpers
// ---------------------------------------------------------------------------

// Create matmul descriptor + 3 matrix layouts for C^T = B @ A^T (row-major convention).
// B [K,N] col-major with TRANSA=T, A [K,M] col-major with TRANSB=N, C [N,M] col-major.
static void create_gemm_descriptors(GemmCacheEntry& entry, cublasComputeType_t compute_type,
                                    cudaDataType_t scale_type, cudaDataType_t dtype_A, cudaDataType_t dtype_B,
                                    cudaDataType_t dtype_C, int K, int M, int N) {
    cublasLtMatmulDescCreate(&entry.opDesc, compute_type, scale_type);

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&entry.Bdesc, dtype_B, K, N, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&entry.Adesc, dtype_A, K, M, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&entry.Cdesc, dtype_C, N, M, N));
}

// Rebuild A and C layout descriptors when actual M differs from cached M.
// Cheap CPU-only operation (no GPU sync).
static void rebuild_layouts_for_m(GemmCacheEntry& entry, cudaDataType_t dtype_A, cudaDataType_t dtype_C,
                                  int K, int M, int N) {
    cublasLtMatrixLayoutDestroy(entry.Adesc);
    cublasLtMatrixLayoutDestroy(entry.Cdesc);
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&entry.Adesc, dtype_A, K, M, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&entry.Cdesc, dtype_C, N, M, N));
    entry.desc_M = M;
}

// Re-select algorithm via heuristic after a cublasLtMatmul failure.
// Called when the cached algo (benchmarked for a different M within the
// same bucket) is invalid for the current M — e.g. FP8 algos on sm_120
// are sensitive to exact dimensions.
static void reselect_algo_for_entry(GemmCacheEntry& entry) {
    cublasLtHandle_t lt = get_cublaslt_handle();
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &s_workspace_size,
                                         sizeof(s_workspace_size));

    cublasLtMatmulHeuristicResult_t results[1];
    int nresults = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, entry.opDesc, entry.Bdesc, entry.Adesc, entry.Cdesc, entry.Cdesc, pref,
                                   1, results, &nresults);
    cublasLtMatmulPreferenceDestroy(pref);

    if (nresults > 0) {
        entry.algo = results[0].algo;
        entry.workspace_size = (results[0].workspaceSize <= s_workspace_size) ? results[0].workspaceSize : 0;
        entry.has_algo = true;
    } else {
        entry.has_algo = false;
        entry.workspace_size = 0;
    }
}

// Set per-call FP8 scale pointers on a matmul descriptor.
static inline void set_gemm_scale_pointers(cublasLtMatmulDesc_t opDesc, const float* aScale,
                                           const float* bScale) {
    if (aScale) {
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale, sizeof(aScale));
    }
    if (bScale) {
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, sizeof(bScale));
    }
}

// ---------------------------------------------------------------------------
// Algorithm benchmarking: request top-N candidates, time each, pick fastest.
// Uses a temporary output buffer to avoid corrupting C during live inference.
// Eliminates 2.6x prefill variance from non-deterministic cuBLAS autotuning.
// ---------------------------------------------------------------------------
static constexpr int kMaxAlgoCandidates = 8;
static constexpr int kBenchmarkIters = 5;

// Diagnostic: when diagnostics.log_gemm_algo (legacy IMP_LOG_GEMM_ALGO=1)
// is set, log shape + per-candidate algoId/tileId + chosen algo for every
// benchmark_and_select_algo call. Used to enumerate which exact GEMM shapes
// select cuBLAS legacy WMMA kernels (Finding 1/5).
static int gemm_algo_log_enabled() { return imp::process_diag_log_gemm_algo() ? 1 : 0; }

static void benchmark_and_select_algo(cublasLtHandle_t lt, GemmCacheEntry& entry, const void* A_data,
                                      const void* B_data, size_t C_bytes, float alpha, float beta,
                                      bool is_int_compute, cudaStream_t stream, int M = 0, int N = 0,
                                      int K = 0) {
    cublasLtMatmulPreference_t pref = nullptr;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &s_workspace_size, sizeof(s_workspace_size)));

    cublasLtMatmulHeuristicResult_t results[kMaxAlgoCandidates];
    int nresults = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, entry.opDesc, entry.Bdesc, entry.Adesc, entry.Cdesc, entry.Cdesc, pref,
                                   kMaxAlgoCandidates, results, &nresults);
    cublasLtMatmulPreferenceDestroy(pref);

    if (nresults <= 0) {
        entry.has_algo = false;
        entry.workspace_size = 0;
        return;
    }
    // [diag] IMP_LOG_GEMM_ALGO: dump shape + per-candidate algoId/tileId.
    // Helps identify which shapes are stuck on legacy WMMA candidates.
    if (gemm_algo_log_enabled() && M > 0) {
        fprintf(stderr, "[gemm-algo] shape M=%d N=%d K=%d  candidates=%d\n", M, N, K, nresults);
        for (int i = 0; i < nresults; i++) {
            int algo_id = -1, tile_id = -1;
            cublasLtMatmulAlgoCapGetAttribute(&results[i].algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                              &algo_id, sizeof(algo_id), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id,
                                                 sizeof(tile_id), nullptr);
            fprintf(stderr, "[gemm-algo]   cand[%d]: numImplFlags=0x%x tile=%d ws=%zu\n", i, algo_id, tile_id,
                    results[i].workspaceSize);
        }
    }
    // [runtime] deterministic_gemm = true skips timing-based selection so
    // repeat runs produce bitwise-identical prefill outputs.
    const bool s_deterministic_gemm = imp::process_diag_deterministic_gemm();
    if (s_deterministic_gemm || nresults == 1) {
        entry.algo = results[0].algo;
        entry.workspace_size = (results[0].workspaceSize <= s_workspace_size) ? results[0].workspaceSize : 0;
        entry.has_algo = true;
        if (gemm_algo_log_enabled() && M > 0) {
            fprintf(stderr, "[gemm-algo]   PICKED cand[0] (deterministic or only candidate)\n");
        }
        return;
    }

    // Use pre-allocated scratch buffer to avoid fragmenting GPU memory
    if (!s_bench_scratch || C_bytes > s_bench_scratch_size) {
        entry.algo = results[0].algo;
        entry.workspace_size = (results[0].workspaceSize <= s_workspace_size) ? results[0].workspaceSize : 0;
        entry.has_algo = true;
        return;
    }
    void* temp_c = s_bench_scratch;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best_ms = 1e30f;
    int best_idx = 0;

    // Warmup all candidates first so steady-state caches are warm before any
    // candidate is timed. One warmup call per candidate (the previous policy)
    // let single-rep bench mode pick legacy WMMA paths whose first-call cost
    // is competitive but whose steady-state cost is 3-9× higher than modern
    // m16n8k16 (`s16816gemm`) tiles. Three warmups per candidate covers
    // cuBLASLt's per-algo lazy compile + L2 fill so the timed loop reflects
    // hot-path behavior, eliminating WMMA-fallback selection (Finding 1).
    // Track which algos actually work — cuBLAS heuristic can return algos
    // that fail at runtime (e.g. FP8 on sm_120 at certain M values).
    std::vector<bool> algo_ok(nresults, true);
    constexpr int kWarmupIters = 3;
    for (int i = 0; i < nresults; i++) {
        if (results[i].workspaceSize > s_workspace_size) {
            algo_ok[i] = false;
            continue;
        }
        float zero = 0.0f;
        for (int w = 0; w < kWarmupIters; w++) {
            cublasStatus_t wst = cublasLtMatmul(lt, entry.opDesc, &alpha, B_data, entry.Bdesc, A_data,
                                                 entry.Adesc, &zero, temp_c, entry.Cdesc, temp_c, entry.Cdesc,
                                                 &results[i].algo, s_workspace, results[i].workspaceSize, stream);
            if (wst != CUBLAS_STATUS_SUCCESS) {
                algo_ok[i] = false;
                break;
            }
        }
    }

    for (int i = 0; i < nresults; i++) {
        if (!algo_ok[i])
            continue;
        float zero = 0.0f;
        cudaEventRecord(start, stream);
        for (int r = 0; r < kBenchmarkIters; r++)
            cublasLtMatmul(lt, entry.opDesc, &alpha, B_data, entry.Bdesc, A_data, entry.Adesc, &zero, temp_c,
                           entry.Cdesc, temp_c, entry.Cdesc, &results[i].algo, s_workspace,
                           results[i].workspaceSize, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) {
            best_ms = ms;
            best_idx = i;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (!algo_ok[best_idx]) {
        entry.has_algo = false;
        entry.workspace_size = 0;
        return;
    }
    entry.algo = results[best_idx].algo;
    entry.workspace_size = results[best_idx].workspaceSize;
    entry.has_algo = true;
    if (gemm_algo_log_enabled() && M > 0) {
        int picked_tile = -1;
        cublasLtMatmulAlgoConfigGetAttribute(&entry.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &picked_tile,
                                             sizeof(picked_tile), nullptr);
        fprintf(stderr, "[gemm-algo]   PICKED cand[%d] tile=%d  best_ms=%.3f\n", best_idx, picked_tile,
                best_ms);
    }
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

// --- GEMV fast path for M=1 decode (memory-bandwidth-bound) ---
// Applies when all operands share the same dtype (excludes LM head: FP16→FP32).
// Returns true if handled.
static bool gemm_try_gemv(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                          cudaStream_t stream) {
    const int64_t M = A.shape[0];
    if (M != 1 || alpha != 1.0f || beta != 0.0f)
        return false;
    if (A.qtype != B.qtype || A.qtype != C.qtype)
        return false;
    if (A.qtype != QType::F16 && A.qtype != QType::F32 && A.qtype != QType::BF16)
        return false;

    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];

    Tensor x_vec;
    x_vec.data = A.data;
    x_vec.qtype = A.qtype;
    x_vec.ndim = 1;
    x_vec.shape[0] = K;
    x_vec.stride[0] = 1;
    x_vec.on_device = true;

    Tensor y_vec;
    y_vec.data = C.data;
    y_vec.qtype = C.qtype;
    y_vec.ndim = 1;
    y_vec.shape[0] = N;
    y_vec.stride[0] = 1;
    y_vec.on_device = true;

    gemv(B, x_vec, y_vec, stream);
    return true;
}

// --- FP32 fast path using cublasSgemm ---
// B is [N,K] row-major = [K,N] col-major. We need B transposed → CUBLAS_OP_T.
// A is [M,K] row-major = [K,M] col-major. We need A as-is    → CUBLAS_OP_N.
// Returns true if handled.
static bool gemm_try_sgemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                           cudaStream_t stream) {
    if (A.qtype != QType::F32 || B.qtype != QType::F32 || C.qtype != QType::F32)
        return false;

    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cublasStatus_t st = cublasSgemm(handle,
                                    CUBLAS_OP_T,  // transa: transpose B_col [K,N] → [N,K]
                                    CUBLAS_OP_N,  // transb: A_col [K,M] used as-is
                                    (int)N,       // m
                                    (int)M,       // n
                                    (int)K,       // k
                                    &alpha, static_cast<const float*>(B.data),
                                    (int)K,  // lda = K (leading dim of B before transpose)
                                    static_cast<const float*>(A.data), (int)K,  // ldb = K (leading dim of A)
                                    &beta, static_cast<float*>(C.data), (int)N  // ldc = N
    );
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "imp::gemm: cublasSgemm failed (status %d)\n", (int)st);
    }
    return true;
}

// --- Generic path via cuBLASLt (uses pre-allocated static workspace + descriptor cache) ---
static void gemm_cublaslt_generic(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                                  cudaStream_t stream) {
    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];

    if (N == 0 || !B.data) {
        return;  // Skip empty weight
    }

    cudaDataType_t cuda_dtype_A = dtype_to_cuda(A.qtype);
    cudaDataType_t cuda_dtype_B = dtype_to_cuda(B.qtype);
    cudaDataType_t cuda_dtype_C = dtype_to_cuda(C.qtype);
    cublasComputeType_t compute_type = dtype_to_compute(A.qtype);

    // Capture-safe path: cuBLASLt fails with CUBLAS_STATUS_INTERNAL_ERROR
    // (status 14) under stream capture on sm_120 — heuristic + workspace
    // allocation paths aren't graph-safe. Route FP16×FP16→FP16 GEMMs to the
    // hand-tuned sm_120 WMMA kernel when the stream is in capture mode.
    if (A.qtype == QType::F16 && B.qtype == QType::F16 && C.qtype == QType::F16) {
        cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess &&
            cap_status == cudaStreamCaptureStatusActive) {
            if (gemm_capture_fp16_sm120(A.data, B.data, C.data, (int)M, (int)N, (int)K, alpha, beta,
                                         stream)) {
                return;
            }
        }
    }

    // Mixed-precision output (e.g. FP16×FP16 → FP32 for diagnostic precision
    // probes): bypass cuBLASLt and use cublasGemmEx directly. cuBLASLt's
    // descriptor + algo selection produces wildly wrong results with our
    // FP16→FP32 dimensions on sm_120 (sums in the billions while real
    // attention output is ±100). cublasGemmEx is the legacy, well-tested API
    // that handles FP16×FP16→FP32 with CUBLAS_COMPUTE_32F correctly.
    if (A.qtype != C.qtype && A.qtype == QType::F16 && B.qtype == QType::F16 && C.qtype == QType::F32) {
        cublasHandle_t fb_handle = get_cublas_handle();
        cublasSetStream(fb_handle, stream);
        cublasStatus_t st = cublasGemmEx(fb_handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                                         B.data, cuda_dtype_B, (int)K, A.data, cuda_dtype_A, (int)K, &beta,
                                         C.data, cuda_dtype_C, (int)N, CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_WARN("gemm: cublasGemmEx FP16→FP32 failed status=%d M=%ld K=%ld N=%ld", (int)st, (long)M,
                         (long)K, (long)N);
        }
        return;
    }

    cublasLtHandle_t lt = get_cublaslt_handle();

    GemmCacheKey cache_key{cuda_dtype_A, cuda_dtype_B, cuda_dtype_C, compute_type, bucket_m(M), K, N, false};

    GemmCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
        auto it = s_gemm_cache.find(cache_key);
        if (it != s_gemm_cache.end()) {
            entry = &it->second;
        } else {
            GemmCacheEntry new_entry{};
            new_entry.desc_M = M;
            cudaDataType_t scale_type = (compute_type == CUBLAS_COMPUTE_32I) ? CUDA_R_32I : CUDA_R_32F;

            create_gemm_descriptors(new_entry, compute_type, scale_type, cuda_dtype_A, cuda_dtype_B,
                                    cuda_dtype_C, (int)K, (int)M, (int)N);

            size_t c_bytes = (size_t)M * N * dtype_size(C.qtype);
            benchmark_and_select_algo(lt, new_entry, A.data, B.data, c_bytes, alpha, beta,
                                      (compute_type == CUBLAS_COMPUTE_32I), stream, (int)M, (int)N, (int)K);

            auto [inserted_it, _] = s_gemm_cache.emplace(cache_key, new_entry);
            entry = &inserted_it->second;
        }

        // Rebuild layout descriptors if actual M differs from cached M
        // (bucketed key matched but exact M changed).
        if (entry->desc_M != M) {
            rebuild_layouts_for_m(*entry, cuda_dtype_A, cuda_dtype_C, (int)K, (int)M, (int)N);
        }
    }

    if (compute_type == CUBLAS_COMPUTE_32I) {
        int32_t ialpha = (int32_t)alpha;
        int32_t ibeta = (int32_t)beta;
        cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc, &ialpha, B.data, entry->Bdesc, A.data,
                                           entry->Adesc, &ibeta, C.data, entry->Cdesc, C.data, entry->Cdesc,
                                           entry->has_algo ? &entry->algo : nullptr, s_workspace,
                                           entry->workspace_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_WARN(
                "gemm: cublasLtMatmul (INT) failed (status %d) M=%ld K=%ld N=%ld, "
                "falling back to cublasGemmEx",
                (int)st, (long)M, (long)K, (long)N);
            cublasHandle_t fb_handle = get_cublas_handle();
            cublasSetStream(fb_handle, stream);
            cublasGemmEx(fb_handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &ialpha, B.data,
                         cuda_dtype_B, (int)K, A.data, cuda_dtype_A, (int)K, &ibeta, C.data, cuda_dtype_C,
                         (int)N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        }
    } else {
        cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc, &alpha, B.data, entry->Bdesc, A.data,
                                           entry->Adesc, &beta, C.data, entry->Cdesc, C.data, entry->Cdesc,
                                           entry->has_algo ? &entry->algo : nullptr, s_workspace,
                                           entry->workspace_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            // Stale algo from a different M within the same bucket.
            // Re-select via heuristic and retry before falling back.
            {
                std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
                reselect_algo_for_entry(*entry);
            }
            st = cublasLtMatmul(lt, entry->opDesc, &alpha, B.data, entry->Bdesc, A.data, entry->Adesc, &beta,
                                C.data, entry->Cdesc, C.data, entry->Cdesc,
                                entry->has_algo ? &entry->algo : nullptr, s_workspace, entry->workspace_size,
                                stream);
            if (st != CUBLAS_STATUS_SUCCESS) {
                static int fallback_count = 0;
                if (++fallback_count <= 10) {
                    IMP_LOG_WARN(
                        "gemm: cublasLtMatmul failed (status %d) M=%ld K=%ld N=%ld "
                        "after algo reselect, falling back to cublasGemmEx",
                        (int)st, (long)M, (long)K, (long)N);
                }
                cublasHandle_t fb_handle = get_cublas_handle();
                cublasSetStream(fb_handle, stream);
                cublasStatus_t fb_st = cublasGemmEx(fb_handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M,
                                                    (int)K, &alpha, B.data, cuda_dtype_B, (int)K, A.data,
                                                    cuda_dtype_A, (int)K, &beta, C.data, cuda_dtype_C, (int)N,
                                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                if (fb_st != CUBLAS_STATUS_SUCCESS) {
                    // Both cublasLt and cublasGemmEx failed. Output buffer holds
                    // garbage; downstream kernels will likely IMA on its values
                    // or produce silent NaN. Surface the failure here instead.
                    IMP_LOG_ERROR(
                        "gemm: cublasGemmEx fallback also failed (status %d) "
                        "M=%ld K=%ld N=%ld dtA=%d dtB=%d dtC=%d. Output "
                        "buffer is garbage; expect downstream IMA.",
                        (int)fb_st, (long)M, (long)K, (long)N, (int)cuda_dtype_A, (int)cuda_dtype_B,
                        (int)cuda_dtype_C);
                }
            }
        }
    }
}

void gemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta, cudaStream_t stream) {
    // Guard against quantized weight tensors (e.g. MXFP4 with dtype=INT4)
    // that should have been handled by the FP16 weight cache path.
    // Passing raw quantized data to cuBLAS causes illegal memory access
    // (cuBLAS reads sizeof(FP16)*numel bytes but only sizeof(quant)*numel exist).
    if (B.qtype == QType::INT4) {
        // This should never be reached — FP16 cache or gemm_dispatch should
        // handle quantized weights. If we get here, output will be zero (safe).
        return;
    }
    if (gemm_try_gemv(A, B, C, alpha, beta, stream))
        return;
    if (gemm_try_sgemm(A, B, C, alpha, beta, stream))
        return;
    gemm_cublaslt_generic(A, B, C, alpha, beta, stream);
}

// ---------------------------------------------------------------------------
// GEMV kernels -- each warp computes one output element (dot product of a row)
// ---------------------------------------------------------------------------

// --- FP32 GEMV kernel ---
__global__ void gemv_fp32_kernel(const float* __restrict__ A, const float* __restrict__ x,
                                 float* __restrict__ y, int M, int K) {
    // Each warp handles one row of A.
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const float* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: float4 = 4 floats per load.
    const int K_vec = K / 4;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);

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
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = sum;
    }
}

// --- FP16 GEMV kernel ---
__global__ void gemv_fp16_kernel(const half* __restrict__ A, const half* __restrict__ x, half* __restrict__ y,
                                 int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const half* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

#if __CUDA_ARCH__ >= 1200
    // Blackwell (sm_120+): 256-bit loads via paired float4 (16 halves per iteration).
    // 2× wider than the default 128-bit path, better saturating memory bandwidth.
    const int K_vec16 = K / 16;  // 16 halves = 32 bytes = 2 × sizeof(float4)
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec16; i += 32) {
        float4 a0 = A_row_v[2 * i];
        float4 a1 = A_row_v[2 * i + 1];
        float4 x0 = x_v[2 * i];
        float4 x1 = x_v[2 * i + 1];

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
    const float4* x_v = reinterpret_cast<const float4*>(x);

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
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// --- BF16 GEMV kernel ---
__global__ void gemv_bf16_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const __nv_bfloat16* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: 8 bf16 per float4.
    const int K_vec = K / 8;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);

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
    sum = warp_reduce_sum(sum);

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
void gemv(const Tensor& A, const Tensor& x, Tensor& y, cudaStream_t stream) {
    const int M = (int)A.shape[0];
    const int K = (int)A.shape[1];

    // Determine batch size from x's shape.
    int batch = 1;
    if (x.ndim == 2) {
        batch = (int)x.shape[1];
    }

    const int blocks = gemv_blocks(M);

    for (int b = 0; b < batch; ++b) {
        switch (A.qtype) {
            case QType::F32: {
                const float* A_ptr = static_cast<const float*>(A.data);
                const float* x_ptr = static_cast<const float*>(x.data) + (int64_t)b * K;
                float* y_ptr = static_cast<float*>(y.data) + (int64_t)b * M;
                gemv_fp32_kernel<<<blocks, kGemvThreads, 0, stream>>>(A_ptr, x_ptr, y_ptr, M, K);
                break;
            }
            case QType::F16: {
                const half* A_ptr = static_cast<const half*>(A.data);
                const half* x_ptr = static_cast<const half*>(x.data) + (int64_t)b * K;
                half* y_ptr = static_cast<half*>(y.data) + (int64_t)b * M;
                gemv_fp16_kernel<<<blocks, kGemvThreads, 0, stream>>>(A_ptr, x_ptr, y_ptr, M, K);
                break;
            }
            case QType::BF16: {
                const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data);
                const __nv_bfloat16* x_ptr = static_cast<const __nv_bfloat16*>(x.data) + (int64_t)b * K;
                __nv_bfloat16* y_ptr = static_cast<__nv_bfloat16*>(y.data) + (int64_t)b * M;
                gemv_bf16_kernel<<<blocks, kGemvThreads, 0, stream>>>(A_ptr, x_ptr, y_ptr, M, K);
                break;
            }
            default: {
                // Fallback: use cuBLAS gemv for other dtypes via gemm with N=1.
                // Construct a temporary Tensor view for the column vectors.
                Tensor x_col;
                x_col.data = static_cast<char*>(x.data) + b * K * dtype_size(x.qtype);
                x_col.qtype = x.qtype;
                x_col.ndim = 2;
                x_col.shape[0] = K;
                x_col.shape[1] = 1;
                x_col.stride[0] = 1;
                x_col.stride[1] = K;
                x_col.on_device = true;

                Tensor y_col;
                y_col.data = static_cast<char*>(y.data) + b * M * dtype_size(y.qtype);
                y_col.qtype = y.qtype;
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
void gemm_cublaslt(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta, const float* aScale,
                   const float* bScale, cudaStream_t stream) {
    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];

    cublasLtHandle_t lt = get_cublaslt_handle();

    // Cache key: (M, K, N, dtypes, beta) — scale pointers vary per-call but
    // don't affect descriptor/algo selection, only set via opDesc attribute.
    cudaDataType_t cuda_dtype_A = dtype_to_cuda(A.qtype);
    cudaDataType_t cuda_dtype_B = dtype_to_cuda(B.qtype);
    cudaDataType_t cuda_dtype_C = dtype_to_cuda(C.qtype);
    // FP8 algos on sm_120 are sensitive to exact M — an algo benchmarked at
    // one M within a bucket can return CUBLAS_STATUS_NOT_SUPPORTED at another
    // M in the same bucket. Use exact M for FP8 to avoid stale algo reuse.
    bool is_fp8 = (cuda_dtype_A == CUDA_R_8F_E4M3 || cuda_dtype_B == CUDA_R_8F_E4M3);
    GemmCacheKey cache_key{
        cuda_dtype_A, cuda_dtype_B, cuda_dtype_C, CUBLAS_COMPUTE_32F,
        is_fp8 ? M : bucket_m(M), K, N, (aScale != nullptr)};

    GemmCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
        auto it = s_gemm_cache.find(cache_key);
        if (it != s_gemm_cache.end()) {
            entry = &it->second;
        } else {
            GemmCacheEntry new_entry{};
            new_entry.desc_M = M;

            create_gemm_descriptors(new_entry, CUBLAS_COMPUTE_32F, CUDA_R_32F, cuda_dtype_A, cuda_dtype_B,
                                    cuda_dtype_C, (int)K, (int)M, (int)N);

            set_gemm_scale_pointers(new_entry.opDesc, aScale, bScale);

            size_t c_bytes = (size_t)M * N * dtype_size(C.qtype);
            benchmark_and_select_algo(lt, new_entry, A.data, B.data, c_bytes, alpha, beta, false, stream,
                                      (int)M, (int)N, (int)K);

            auto [ins_it, _] = s_gemm_cache.emplace(cache_key, new_entry);
            entry = &ins_it->second;
        }

        // Rebuild layout descriptors if actual M differs from cached M
        // (bucketed key matched but exact M changed).
        if (entry->desc_M != M) {
            rebuild_layouts_for_m(*entry, cuda_dtype_A, cuda_dtype_C, (int)K, (int)M, (int)N);
        }
    }

    // Set per-call scale pointers (vary by weight tensor, not cached).
    // SAFETY: This mutates the cached opDesc WITHOUT holding s_gemm_cache_mutex.
    // This is safe because imp enforces single-stream GEMM execution — all GEMM
    // calls are serialized on a single CUDA stream, so no two threads can race
    // on the same opDesc concurrently. If multi-stream GEMM is ever added, this
    // section must be protected by the mutex (or opDesc must be duplicated per-call).
    set_gemm_scale_pointers(entry->opDesc, aScale, bScale);

    cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc, &alpha, B.data, entry->Bdesc, A.data, entry->Adesc,
                                       &beta, C.data, entry->Cdesc, C.data, entry->Cdesc,
                                       entry->has_algo ? &entry->algo : nullptr, s_workspace,
                                       entry->workspace_size, stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        // The cached algo (benchmarked for a different M within the same bucket)
        // may be invalid for the current M. Re-select via heuristic and retry.
        {
            std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
            reselect_algo_for_entry(*entry);
        }
        set_gemm_scale_pointers(entry->opDesc, aScale, bScale);
        st = cublasLtMatmul(lt, entry->opDesc, &alpha, B.data, entry->Bdesc, A.data, entry->Adesc, &beta,
                            C.data, entry->Cdesc, C.data, entry->Cdesc,
                            entry->has_algo ? &entry->algo : nullptr, s_workspace, entry->workspace_size,
                            stream);

        if (st != CUBLAS_STATUS_SUCCESS) {
            static int fallback_count = 0;
            if (++fallback_count <= 10) {
                IMP_LOG_WARN(
                    "gemm_cublaslt: cublasLtMatmul failed (status %d) M=%ld K=%ld N=%ld "
                    "after algo reselect, retrying with default heuristic",
                    (int)st, (long)M, (long)K, (long)N);
            }
            set_gemm_scale_pointers(entry->opDesc, aScale, bScale);
            st = cublasLtMatmul(lt, entry->opDesc, &alpha, B.data, entry->Bdesc, A.data, entry->Adesc,
                                &beta, C.data, entry->Cdesc, C.data, entry->Cdesc, nullptr, s_workspace,
                                entry->workspace_size, stream);
            if (st != CUBLAS_STATUS_SUCCESS) {
                static bool s_fp8_warned = false;
                if (!s_fp8_warned) {
                    s_fp8_warned = true;
                    IMP_LOG_WARN(
                        "gemm_cublaslt: FP8 GEMM unsupported by cuBLASLt on this GPU/driver. "
                        "Falling back to FP8->FP16 dequant + FP16 GEMM (slower but correct). "
                        "Consider --set attention.fp8_prefill=never to skip FP8 overhead.");
                }
                float a_scale_h = 1.0f, b_scale_h = 1.0f;
                if (aScale)
                    cudaMemcpy(&a_scale_h, aScale, sizeof(float), cudaMemcpyDeviceToHost);
                if (bScale)
                    cudaMemcpy(&b_scale_h, bScale, sizeof(float), cudaMemcpyDeviceToHost);

                int a_elems = static_cast<int>(M * K);
                int b_elems = static_cast<int>(N * K);
                half *d_a16 = nullptr, *d_b16 = nullptr;
                cudaMallocAsync(&d_a16, static_cast<size_t>(a_elems) * sizeof(half), stream);
                cudaMallocAsync(&d_b16, static_cast<size_t>(b_elems) * sizeof(half), stream);
                if (d_a16 && d_b16) {
                    dequantize_fp8_e4m3_to_fp16(A.data, d_a16, a_elems, a_scale_h, stream);
                    dequantize_fp8_e4m3_to_fp16(B.data, d_b16, b_elems, b_scale_h, stream);
                    int64_t a16_shape[2] = {M, K};
                    int64_t b16_shape[2] = {N, K};
                    Tensor A16(d_a16, QType::F16, 2, a16_shape, true);
                    Tensor B16(d_b16, QType::F16, 2, b16_shape, true);
                    gemm(A16, B16, C, alpha, beta, stream);
                } else {
                    IMP_LOG_ERROR("gemm_cublaslt: FP8 dequant fallback alloc failed M=%ld K=%ld N=%ld",
                                  (long)M, (long)K, (long)N);
                }
                if (d_a16) cudaFreeAsync(d_a16, stream);
                if (d_b16) cudaFreeAsync(d_b16, stream);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 GEMV kernel -- 16 FP8 values per load (16 bytes)
// Each warp handles one row. Dequant on-the-fly with per-tensor scale.
// ---------------------------------------------------------------------------
__global__ void gemv_fp8_e4m3_kernel(const uint8_t* __restrict__ A, const half* __restrict__ x,
                                     half* __restrict__ y, int M, int K, float scale) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

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

// Dequant and accumulate 16 FP8 values in two groups of 8,
// avoiding per-element j<8 branch for x_lo vs x_hi selection.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &a_bytes[j], 1);
            float a_val = (float)fp8_val * scale;
            sum += a_val * __half2float(x_lo[j]);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &a_bytes[8 + j], 1);
            float a_val = (float)fp8_val * scale;
            sum += a_val * __half2float(x_hi[j]);
        }
    }

    // Handle remainder
    int base = K_vec * 16;
    for (int i = base + lane; i < K; i += 32) {
        __nv_fp8_e4m3 fp8_val;
        memcpy(&fp8_val, &A_row[i], 1);
        float a_val = (float)fp8_val * scale;
        sum += a_val * __half2float(*(reinterpret_cast<const half*>(x) + i));
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// ---------------------------------------------------------------------------
// Fused Q6_K GEMV kernel -- dequant-and-dot in one pass.
// Q6_K block = 210 bytes for 256 elements: ql[128] + qh[64] + scales[16] + d[2].
// Each warp computes one output row's dot product.
// ---------------------------------------------------------------------------
__global__ void gemv_q6k_kernel(const uint8_t* __restrict__ W, const half* __restrict__ x,
                                half* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 210;
        const uint8_t* ql = bp;                            // ql[128]
        const uint8_t* qh = bp + 128;                      // qh[64]
        const int8_t* sc = (const int8_t*)(bp + 192);      // scales[16]
        float d = __half2float(*(const half*)(bp + 208));  // d[2]
        const int base = b * 256;

        // Coalesced loads: 4 ql bytes + 2 qh bytes per thread
        uint8_t ql_a = ql[lane];            // [0..31]
        uint8_t ql_b = ql[lane + 32];       // [32..63]
        uint8_t ql_c = ql[64 + lane];       // [64..95]
        uint8_t ql_d = ql[64 + lane + 32];  // [96..127]
        uint8_t qh0 = qh[lane];             // [0..31]
        uint8_t qh1 = qh[32 + lane];        // [32..63]

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
        sum += d * ((float)sc[sc_idx] * (float)q0 * __half2float(x[base + lane]) +
                    (float)sc[sc_idx + 2] * (float)q1 * __half2float(x[base + lane + 32]) +
                    (float)sc[sc_idx + 4] * (float)q2 * __half2float(x[base + lane + 64]) +
                    (float)sc[sc_idx + 6] * (float)q3 * __half2float(x[base + lane + 96]) +
                    (float)sc[sc_idx + 8] * (float)q4 * __half2float(x[base + lane + 128]) +
                    (float)sc[sc_idx + 10] * (float)q5 * __half2float(x[base + lane + 160]) +
                    (float)sc[sc_idx + 12] * (float)q6 * __half2float(x[base + lane + 192]) +
                    (float)sc[sc_idx + 14] * (float)q7 * __half2float(x[base + lane + 224]));
    }

    // Warp shuffle reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[row] = __float2half(sum);
}

void gemv_q6k(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream) {
    gemv_q6k_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(static_cast<const uint8_t*>(W), x, y, M, K);
}

// ---------------------------------------------------------------------------
// Fused Q8_0 GEMV kernel -- dequant-and-dot in one pass.
// Q8_0 block = 34 bytes for 32 elements: d[2] + qs[32].
// Each warp computes one output row's dot product. Each thread handles one
// element per block (32 threads = 32 elements = 1 block).
// ---------------------------------------------------------------------------
__global__ void gemv_q8_0_kernel(const uint8_t* __restrict__ W, const half* __restrict__ x,
                                 half* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

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
    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[row] = __float2half(sum);
}

void gemv_q8_0(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream) {
    gemv_q8_0_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(static_cast<const uint8_t*>(W), x, y, M, K);
}

// ---------------------------------------------------------------------------
// MoE decode GEMV: processes all top_k experts in a single kernel launch.
// expert_indices[slot] selects which expert's weights to read from packed_weights.
// Grid: top_k * blocks_per_expert blocks. Each block group handles one expert slot.
// x_stride: 0 = shared input for all experts (gate/up), >0 = per-expert input (down).
// ---------------------------------------------------------------------------

__global__ void gemv_q6k_moe_decode_kernel(const uint8_t* __restrict__ packed_weights,
                                           const int32_t* __restrict__ expert_indices,
                                           const half* __restrict__ x, half* __restrict__ y, int rows, int K,
                                           size_t expert_stride_bytes, int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

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
        const int8_t* sc = (const int8_t*)(bp + 192);
        float d = __half2float(*(const half*)(bp + 208));
        const int base = b * 256;

        uint8_t ql_a = ql[lane];
        uint8_t ql_b = ql[lane + 32];
        uint8_t ql_c = ql[64 + lane];
        uint8_t ql_d = ql[64 + lane + 32];
        uint8_t qh0 = qh[lane];
        uint8_t qh1 = qh[32 + lane];

        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        int sc_idx = lane >> 4;
        sum += d * ((float)sc[sc_idx] * (float)q0 * __half2float(x_ptr[base + lane]) +
                    (float)sc[sc_idx + 2] * (float)q1 * __half2float(x_ptr[base + lane + 32]) +
                    (float)sc[sc_idx + 4] * (float)q2 * __half2float(x_ptr[base + lane + 64]) +
                    (float)sc[sc_idx + 6] * (float)q3 * __half2float(x_ptr[base + lane + 96]) +
                    (float)sc[sc_idx + 8] * (float)q4 * __half2float(x_ptr[base + lane + 128]) +
                    (float)sc[sc_idx + 10] * (float)q5 * __half2float(x_ptr[base + lane + 160]) +
                    (float)sc[sc_idx + 12] * (float)q6 * __half2float(x_ptr[base + lane + 192]) +
                    (float)sc[sc_idx + 14] * (float)q7 * __half2float(x_ptr[base + lane + 224]));
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_moe_decode(const void* packed_weights, const int32_t* expert_indices, const half* x, half* y,
                         int rows, int K, size_t expert_stride_bytes, int x_stride, int top_k,
                         cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    gemv_q6k_moe_decode_kernel<<<top_k * blocks_per_expert, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights), expert_indices, x, y, rows, K, expert_stride_bytes,
        x_stride, blocks_per_expert);
}

__global__ void gemv_q8_0_moe_decode_kernel(const uint8_t* __restrict__ packed_weights,
                                            const int32_t* __restrict__ expert_indices,
                                            const half* __restrict__ x, half* __restrict__ y, int rows, int K,
                                            size_t expert_stride_bytes, int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

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

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_moe_decode(const void* packed_weights, const int32_t* expert_indices, const half* x, half* y,
                          int rows, int K, size_t expert_stride_bytes, int x_stride, int top_k,
                          cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    gemv_q8_0_moe_decode_kernel<<<top_k * blocks_per_expert, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights), expert_indices, x, y, rows, K, expert_stride_bytes,
        x_stride, blocks_per_expert);
}

// ---------------------------------------------------------------------------
// FP16 GEMV with FP32 output for MoE gate logits: y = W @ x
// W: [M, K] FP16 (row-major), x: [K] FP16, y: [M] FP32.
// Designed for M=n_experts (64-256), K=d_model (2048-8192), n=1 decode.
// Replaces cuBLAS gemm() + fp16_to_fp32 cast for tiny M=1 GEMMs.
// Each warp handles one output row. Uses half2 vectorized loads for 2x bandwidth.
// ---------------------------------------------------------------------------
__global__ void gemv_gate_fp32_kernel(const half* __restrict__ W, const half* __restrict__ x,
                                      float* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

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
    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[row] = sum;
}

void gemv_gate_fp32(const half* W, const half* x, float* y, int M, int K, cudaStream_t stream) {
    gemv_gate_fp32_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(W, x, y, M, K);
}

// FP32-input variant: avoids FP16 truncation of router input for MoE precision.
__global__ void gemv_gate_fp32_fp32input_kernel(const half* __restrict__ W, const float* __restrict__ x,
                                                float* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const half* W_row = W + (size_t)row * K;
    float sum = 0.0f;

    // Process 2 weight elements per iteration (half2), read FP32 input directly
    const int K2 = K / 2;
    const half2* W2 = reinterpret_cast<const half2*>(W_row);

    for (int i = lane; i < K2; i += 32) {
        half2 w = W2[i];
        sum += __half2float(w.x) * x[i * 2];
        sum += __half2float(w.y) * x[i * 2 + 1];
    }

    if ((K & 1) && lane == 0) {
        sum += __half2float(W_row[K - 1]) * x[K - 1];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        y[row] = sum;
}

void gemv_gate_fp32_fp32input(const half* W, const float* x, float* y, int M, int K, cudaStream_t stream) {
    gemv_gate_fp32_fp32input_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(W, x, y, M, K);
}

// ---------------------------------------------------------------------------
// Fused gate+up MoE GEMV (scalar FP16 variants — NOT dp4a, kept as-is)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fused gate+up MoE GEMV: computes both gate and up projections in a single
// kernel launch. blockIdx.y selects projection: 0=gate, 1=up.
// Saves one kernel launch per MoE layer (48 launches for Qwen3-Coder).
// ---------------------------------------------------------------------------

__global__ void gemv_q6k_moe_gate_up_fused_kernel(const uint8_t* __restrict__ gate_weights,
                                                  const uint8_t* __restrict__ up_weights,
                                                  const int32_t* __restrict__ expert_indices,
                                                  const half* __restrict__ x, half* __restrict__ y_gate,
                                                  half* __restrict__ y_up, int rows, int K,
                                                  size_t gate_stride_bytes, size_t up_stride_bytes,
                                                  int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

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
        const int8_t* sc = (const int8_t*)(bp + 192);
        float d = __half2float(*(const half*)(bp + 208));
        const int base = b * 256;

        uint8_t ql_a = ql[lane];
        uint8_t ql_b = ql[lane + 32];
        uint8_t ql_c = ql[64 + lane];
        uint8_t ql_d = ql[64 + lane + 32];
        uint8_t qh0 = qh[lane];
        uint8_t qh1 = qh[32 + lane];

        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        int sc_idx = lane >> 4;
        sum += d * ((float)sc[sc_idx] * (float)q0 * __half2float(x_ptr[base + lane]) +
                    (float)sc[sc_idx + 2] * (float)q1 * __half2float(x_ptr[base + lane + 32]) +
                    (float)sc[sc_idx + 4] * (float)q2 * __half2float(x_ptr[base + lane + 64]) +
                    (float)sc[sc_idx + 6] * (float)q3 * __half2float(x_ptr[base + lane + 96]) +
                    (float)sc[sc_idx + 8] * (float)q4 * __half2float(x_ptr[base + lane + 128]) +
                    (float)sc[sc_idx + 10] * (float)q5 * __half2float(x_ptr[base + lane + 160]) +
                    (float)sc[sc_idx + 12] * (float)q6 * __half2float(x_ptr[base + lane + 192]) +
                    (float)sc[sc_idx + 14] * (float)q7 * __half2float(x_ptr[base + lane + 224]));
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                int rows, int K, size_t gate_stride_bytes, size_t up_stride_bytes,
                                int x_stride, int top_k, cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q6k_moe_gate_up_fused_kernel<<<grid, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights), static_cast<const uint8_t*>(up_weights), expert_indices, x,
        y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes, x_stride, blocks_per_expert);
}

__global__ void gemv_q8_0_moe_gate_up_fused_kernel(const uint8_t* __restrict__ gate_weights,
                                                   const uint8_t* __restrict__ up_weights,
                                                   const int32_t* __restrict__ expert_indices,
                                                   const half* __restrict__ x, half* __restrict__ y_gate,
                                                   half* __restrict__ y_up, int rows, int K,
                                                   size_t gate_stride_bytes, size_t up_stride_bytes,
                                                   int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

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

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                 const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                 int rows, int K, size_t gate_stride_bytes, size_t up_stride_bytes,
                                 int x_stride, int top_k, cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q8_0_moe_gate_up_fused_kernel<<<grid, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights), static_cast<const uint8_t*>(up_weights), expert_indices, x,
        y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes, x_stride, blocks_per_expert);
}

// ---------------------------------------------------------------------------
// FP8 E4M3 GEMV
// ---------------------------------------------------------------------------

void gemv_fp8(const Tensor& A, const Tensor& x, Tensor& y, float scale, cudaStream_t stream) {
    const int M = (int)A.shape[0];
    const int K = (int)A.shape[1];

    gemv_fp8_e4m3_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(static_cast<const uint8_t*>(A.data),
                                                                      static_cast<const half*>(x.data),
                                                                      static_cast<half*>(y.data), M, K,
                                                                      scale);
}

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
    long long output_stride = static_cast<long long>(M) * N;  // stride between k_out and v_out

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
                                        s_workspace, s_workspace_size, nullptr);

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
