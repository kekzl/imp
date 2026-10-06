#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include <atomic>
#include "compute/gemm_algo_capture.h"
#include "compute/gemm_capture_fp16_sm120.h"
#include "core/cuda_raii.h"
#include "core/cuda_static_reset.h"
#include "compute/gemm_internal.cuh"
#include "core/logging.h"
#include "core/tensor_kind.h"
#include "memory/engine_arena.h"
#include "core/pdl.h"
#include "core/process_diag.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "quant/fp8_quant.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <span>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <utility>

#define CUBLASLT_CHECK(call)                                                    \
    do {                                                                        \
        cublasStatus_t _st = (call);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            IMP_LOG_ERROR("imp::gemm: %s failed (status %d)", #call, (int)_st); \
        }                                                                       \
    } while (0)

namespace imp {

// #847 graph-captured verify: see gemm.h. Single engine thread writes it;
// relaxed atomic keeps tsan honest without cost.
static std::atomic<bool> g_lt_capture_allowed{false};
void gemm_set_lt_capture_allowed(bool allowed) {
    g_lt_capture_allowed.store(allowed, std::memory_order_relaxed);
}
bool gemm_lt_capture_allowed() {
    return g_lt_capture_allowed.load(std::memory_order_relaxed);
}

// kGemvThreads / kGemvWarps / gemv_blocks live in
// gemm_internal.cuh, warp_reduce_sum in gemm_internal_device.cuh (shared with gemm_gemv_dtype.cu + gemm_moe_gemv.cu).
// cublasGemm*Ex algorithm: cublas_gemm_algo() (compute/cublas_gemm_algo.h, #2168).

// ---------------------------------------------------------------------------
// cuBLAS / cuBLASLt handles (lazily initialized)
// ---------------------------------------------------------------------------
static cublasHandle_t s_cublas_handle = nullptr;
static cublasLtHandle_t s_cublaslt_handle = nullptr;

static cublasHandle_t get_cublas_handle() {
    if (!s_cublas_handle) {
        cublasStatus_t st = cublasCreate(&s_cublas_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_ERROR("imp::gemm: cublasCreate failed (status %d)", (int)st);
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
            IMP_LOG_ERROR("imp::gemm: cublasLtCreate failed (status %d)", (int)st);
            abort();
        }
    }
    return s_cublaslt_handle;
}

// Shared cuBLASLt workspace, taken once from the engine-persistent (T2) arena via
// gemm_init(), used by all GEMM calls. cuBLASLt takes the workspace as an argument, so
// one slice sized at the plan's max serves every call (docs/internals/MEMORY.md A5.3).
static void* s_workspace = nullptr;
static size_t s_workspace_size = 0;

// Bench scratch for algo selection — the C buffer the candidate algos write
// while being timed. Also T2: the alternative is a cudaMalloc/cudaFree per
// cache miss, which fragments and can run while serving.
static void* s_bench_scratch = nullptr;
static size_t s_bench_scratch_size = 0;
// Arena generation of both slices. reset()/close() bumps it: a slice from an older generation is
// handed out again, and a stale workspace aliased the new bench scratch (cuBLASLt hang, #2611).
static uint64_t s_slices_gen = 0;

void gemm_init() {
    // Force handle creation early.
    get_cublas_handle();
    get_cublaslt_handle();

    const uint64_t gen = engine_arena().generation();
    const size_t prev_workspace_size = s_workspace_size;
    if ((s_workspace || s_bench_scratch) && s_slices_gen != gen) {
        s_workspace = nullptr;
        s_workspace_size = 0;
        s_bench_scratch = nullptr;
        s_bench_scratch_size = 0;
    }
    s_slices_gen = gen;

    // T2 (A7 step 8). Both buffers are engine-lifetime and degrade cleanly to null: a
    // 0-byte workspace makes cuBLASLt's heuristic return only algos needing none, a null
    // bench scratch takes the heuristic's first choice. That is why they leave the I1
    // allowlist instead of a cudaMalloc fallback the gate cannot see (AUDIT B47). Outside an
    // Engine (GPU test binaries calling gemm_init() directly), take_bytes() returns empty
    // and both stay null.
    if (!s_workspace) {
        // Size ladder kept from the pre-arena code, but degrades against the ARENA rather than
        // free VRAM: exec_t2_demand charges the full kGemmCublasWorkspaceBytes, so anything
        // below it means the plan under-reserved, worth a log line rather than a silent halving.
        constexpr size_t kTrySizes[] = {
            kGemmCublasWorkspaceBytes,  // 64 MiB — the charged size
            32ULL << 20,                // 32 MiB
            8ULL << 20,                 //  8 MiB
            2ULL << 20,                 //  2 MiB
        };
        for (size_t sz : kTrySizes) {
            auto slab = engine_arena().take_bytes(sz);
            if (!slab.empty()) {
                s_workspace = slab.data();
                s_workspace_size = sz;
                break;
            }
        }
        if (s_workspace_size < kGemmCublasWorkspaceBytes && engine_arena().is_open()) {
            IMP_LOG_WARN(
                "cuBLASLt workspace: the T2 arena served %.1f MiB of the %.1f MiB charged "
                "(%.1f MiB free of %.1f MiB) — GEMM algo choice is restricted",
                s_workspace_size / (1024.0 * 1024.0), kGemmCublasWorkspaceBytes / (1024.0 * 1024.0),
                engine_arena().remaining() / (1024.0 * 1024.0),
                engine_arena().capacity() / (1024.0 * 1024.0));
        }
    }

    // Cached algos were picked against the old size and pass their own workspace_size per call.
    if (s_workspace_size < prev_workspace_size)
        gemm_cleanup();

    // Also let legacy cuBLAS API use the same workspace; null after a dropped slice, never stale.
    if (s_workspace || prev_workspace_size > 0)
        cublasSetWorkspace(get_cublas_handle(), s_workspace, s_workspace_size);

    if (!s_bench_scratch) {
        auto slab = engine_arena().take_bytes(kGemmBenchScratchBytes);
        if (!slab.empty()) {
            s_bench_scratch = slab.data();
            s_bench_scratch_size = kGemmBenchScratchBytes;
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
            IMP_LOG_ERROR("imp::gemm: unsupported dtype %d", std::to_underlying(dt));
            return CUDA_R_16F;  // fallback (caller guard should prevent reaching here)
    }
}

// Non-static accessors exposing the file-local cuBLAS internals to the
// batched-GEMM TU (gemm_batched.cu) via gemm_internal.cuh.
cublasHandle_t gemm_internal_cublas_handle() { return get_cublas_handle(); }
cublasLtHandle_t gemm_internal_cublaslt_handle() { return get_cublaslt_handle(); }
cudaDataType_t gemm_internal_dtype_to_cuda(QType dt) { return dtype_to_cuda(dt); }
void* gemm_internal_workspace() { return s_workspace; }
size_t gemm_internal_workspace_size() { return s_workspace_size; }

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

// Bucket M for the cache key: exact for decode (M<=1), bucketed for prefill. cuBLASLt
// algorithm selection is stable across nearby M values.
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
    bool exact_m = false;  // M is exact: this M failed the bucket pin's algo check (#2346)

    bool operator==(const GemmCacheKey& o) const {
        return dtA == o.dtA && dtB == o.dtB && dtC == o.dtC && compute == o.compute && M == o.M && K == o.K &&
               N == o.N && has_scales == o.has_scales && exact_m == o.exact_m;
    }
};

struct GemmCacheKeyHash {
    static size_t operator()(const GemmCacheKey& k) {
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
        mix(static_cast<uint64_t>(k.exact_m));
        return h;
    }
};

struct GemmCacheEntry {
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulAlgo_t algo;
    size_t workspace_size;
    bool has_algo;
    // The algo came from the TIMED probe, not the heuristic. Without this flag the reselect
    // path cannot tell which of the two it is about to overwrite, so discarding a
    // benchmarked pin looked identical to re-picking a heuristic one (#1545).
    bool benchmarked = false;
    int64_t desc_M;  // M dimension baked into layout descriptors
    // The ACTUAL M this algo was chosen at, which is not desc_M: the layouts are rebuilt for
    // every M that lands in the bucket, so desc_M always equals the current call's M and says
    // nothing about where the algo came from. A cuBLASLt algo is dimension-sensitive, so a pin
    // taken at one M can fail at another M in the same bucket; keeping the two apart is what
    // lets the failure path leave a pin that is still correct for its own M (2026-09-21).
    int64_t algo_M = 0;
    // cublasLtMatmulAlgoCheck of the pin at each other M seen in the bucket (#2346).
    std::unordered_map<int64_t, bool> pin_ok_at_m;
    // The algo a stream capture recorded for this entry is logged once (#2396).
    bool capture_logged = false;
};

static std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheKeyHash> s_gemm_cache;
static std::mutex s_gemm_cache_mutex;
// #2346: matmul failures with a pin chosen at another M, and Ms that got an exact-M entry instead.
// Reported and reset by gemm_cleanup.
static std::atomic<int> s_pin_other_m_failures{0};
static std::atomic<int> s_exact_m_entries{0};

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

// Re-selects the algo via heuristic after a runtime cublasLtMatmul failure (a stale algo
// from a different M in the same bucket; e.g. FP8 algos on sm_120 are dimension-sensitive).
// Returns false without touching the entry when deterministic mode is on: the
// heuristic's results[0] is exactly what the deterministic branch refuses to trust
// without a warmup probe (#1574); the caller falls back instead of retrying unvalidated.
static bool reselect_algo_for_entry(GemmCacheEntry& entry, int64_t M, int64_t K, int64_t N) {
    if (imp::process_diag_deterministic_gemm()) {
        IMP_LOG_DEBUG(
            "[gemm-algo] matmul failed and deterministic_gemm is on: not re-picking by "
            "heuristic (that would drop the warmup-validated algo)");
        return false;
    }
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
        // Logs once when the benchmarked pin is replaced by heuristic results[0] for the rest
        // of the process: previously only a second failure logged anything, so a shape could
        // silently run on a different algo than the probe chose, differing per process start
        // (#1545). `benchmarked` is cleared with it, so this fires once per pin, not per failure.
        if (entry.benchmarked) {
            IMP_LOG_WARN(
                "[gemm-algo] matmul failed with the BENCHMARKED algo for M=%ld K=%ld N=%ld "
                "(bucket desc_M=%ld) — replacing it with the heuristic pick for the rest of the "
                "process; the probe's choice is gone until restart",
                (long)M, (long)K, (long)N, (long)entry.desc_M);
        }
        entry.algo = results[0].algo;
        entry.algo_M = M;
        entry.workspace_size = (results[0].workspaceSize <= s_workspace_size) ? results[0].workspaceSize : 0;
        entry.has_algo = true;
        entry.benchmarked = false;
    } else {
        entry.has_algo = false;
        entry.workspace_size = 0;
    }
    return true;
}

// A cached pin is dimension-sensitive, and bucket_m puts several M behind one key. When the
// pin was chosen at a DIFFERENT M, replacing it here would change every later call at that
// other M too: two identical greedy requests answered differently once a third request had
// failed once (degen_suite kv-growth, 2026-09-21). Returns true when the pin was left alone,
// and the caller then runs this one call on the default heuristic.
static bool keep_pin_chosen_at_other_m(GemmCacheEntry& entry, int64_t M, int64_t K, int64_t N) {
    bool other_m = false;
    {
        std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
        other_m = entry.has_algo && entry.algo_M != M;
        if (!other_m)
            reselect_algo_for_entry(entry, M, K, N);
    }
    if (other_m) {
        const int n = s_pin_other_m_failures.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 5)
            IMP_LOG_WARN(
                "[gemm-algo] matmul failed at M=%ld K=%ld N=%ld with the pin chosen at "
                "M=%ld (same bucket) — this call takes the heuristic, the pin stays so "
                "calls at its own M keep answering identically (failure %d; total at "
                "gemm_cleanup)",
                (long)M, (long)K, (long)N, (long)entry.algo_M, n);
    }
    return other_m;
}

// True when the bucket pin may run at this M. A pin is chosen at one M and is dimension-sensitive
// (#2346: M=784 pin failed at M=816, M=16 pin at M=32); cublasLtMatmulAlgoCheck answers on the
// host before the launch. Cached per M. Caller holds s_gemm_cache_mutex; layouts are at M.
static bool pin_supports_m(cublasLtHandle_t lt, GemmCacheEntry& entry, int64_t M) {
    if (!entry.has_algo || entry.algo_M == M)
        return true;
    auto it = entry.pin_ok_at_m.find(M);
    if (it != entry.pin_ok_at_m.end())
        return it->second;
    cublasLtMatmulHeuristicResult_t res{};
    const bool ok = cublasLtMatmulAlgoCheck(lt, entry.opDesc, entry.Bdesc, entry.Adesc, entry.Cdesc,
                                            entry.Cdesc, &entry.algo, &res) == CUBLAS_STATUS_SUCCESS &&
                    res.workspaceSize <= entry.workspace_size;
    entry.pin_ok_at_m.emplace(M, ok);
    if (!ok)
        s_exact_m_entries.fetch_add(1, std::memory_order_relaxed);
    return ok;
}

// The entry for this call: the bucket entry, or an exact-M entry chosen at M when the bucket pin
// fails pin_supports_m. `make` builds and selects a new entry at M. Caller holds the mutex.
template <class Make>
static GemmCacheEntry* entry_for_m(cublasLtHandle_t lt, GemmCacheKey key, int64_t M, cudaDataType_t dtA,
                                   cudaDataType_t dtC, int64_t K, int64_t N, Make&& make) {
    auto it = s_gemm_cache.find(key);
    if (it == s_gemm_cache.end())
        it = s_gemm_cache.emplace(key, make()).first;
    GemmCacheEntry* entry = &it->second;
    if (entry->desc_M != M)
        rebuild_layouts_for_m(*entry, dtA, dtC, (int)K, (int)M, (int)N);
    if (pin_supports_m(lt, *entry, M))
        return entry;
    key.M = M;
    key.exact_m = true;
    auto ex = s_gemm_cache.find(key);
    if (ex == s_gemm_cache.end())
        ex = s_gemm_cache.emplace(key, make()).first;
    return &ex->second;
}

// Set per-call FP8 scale pointers on a matmul descriptor.
static inline void set_gemm_scale_pointers(cublasLtMatmulDesc_t opDesc, const float* aScale,
                                           const float* bScale) {
    // The attribute value is the device pointer itself, so its address is passed.
    for (auto [attr, p] : {std::pair{CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale},
                           std::pair{CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale}}) {
        if (*p)
            cublasLtMatmulDescSetAttribute(opDesc, attr, static_cast<const void*>(p), sizeof(*p));
    }
}

// Algorithm benchmarking: request top-N candidates, time each, pick fastest. Uses a
// temporary output buffer to avoid corrupting C during live inference.
static constexpr int kMaxAlgoCandidates = 8;
static constexpr int kBenchmarkIters = 5;

// Selection stability (F-9): a single timed sample per candidate is unstable at small M
// because the timed window is dominated by scheduling noise; the two fixes are on the
// estimator, not persistence (an on-disk cache would freeze a mispick permanently and
// needs driver/cuBLAS version invalidation).
//   1. Size the timed window per candidate to ~kTargetWindowMs (not a fixed rep count),
//      so every shape gets comparable measurement quality; costs compared per rep.
//   2. kAlgoMargin hysteresis toward heuristic order: a candidate replaces the incumbent
//      only by beating it by more than the margin, else the lower heuristic index wins
//      (deterministic per shape/device). Margin is set from measured residual spread.
static constexpr int kBenchmarkRounds = 3;
static constexpr float kTargetWindowMs = 0.5f;
static constexpr int kMaxBenchIters = 512;
static constexpr float kAlgoMargin = 0.10f;

// Diagnostic: when diagnostics.log_gemm_algo is set, log shape + per-candidate algoId/tileId + chosen algo for every
// benchmark_and_select_algo call. Used to enumerate which exact GEMM shapes
// select cuBLAS legacy WMMA kernels (Finding 1/5).
static int gemm_algo_log_enabled() { return imp::process_diag_log_gemm_algo() ? 1 : 0; }

// ms per run() over `iters` runs; 1e30f (= unmeasured, as cand_ms starts) on an event error, never 0.
template <typename Run>
static float time_on_stream(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream, int iters, Run&& run) {
    cudaError_t err = cudaEventRecord(start, stream);
    for (int r = 0; r < iters; r++) run();
    float ms = 0;
    if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
    if (err == cudaSuccess) err = cudaEventSynchronize(stop);
    if (err == cudaSuccess) err = cudaEventElapsedTime(&ms, start, stop);
    if (err == cudaSuccess) return ms / static_cast<float>(iters);
    IMP_LOG_WARN("[gemm-algo] candidate timing failed: %s", cudaGetErrorString(err));
    return 1e30f;
}

static void benchmark_and_select_algo(cublasLtHandle_t lt, GemmCacheEntry& entry, const void* A_data,
                                      const void* B_data, size_t C_bytes, float alpha, float beta,
                                      bool is_int_compute, cudaStream_t stream, int M = 0, int N = 0,
                                      int K = 0, bool fp16_scale = false) {
    (void)beta;            // CCN pin key is the signature text (tools/check_complexity.py),
    (void)is_int_compute;  // so the dead params stay named until the function is split.
    // COMPUTE_16F descriptors take __half alpha/beta (scale type CUDA_R_16F).
    const __half h_alpha = __float2half(alpha);
    const __half h_zero = __float2half(0.0f);
    const float f_zero = 0.0f;
    const void* p_alpha = fp16_scale ? static_cast<const void*>(&h_alpha) : static_cast<const void*>(&alpha);
    const void* p_zero = fp16_scale ? static_cast<const void*>(&h_zero) : static_cast<const void*>(&f_zero);
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
    // [diag] diagnostics.log_gemm_algo: dump shape + per-candidate algoId/tileId.
    // Helps identify which shapes are stuck on legacy WMMA candidates.
    if (gemm_algo_log_enabled() && M > 0) {
        IMP_LOG_DEBUG("[gemm-algo] shape M=%d N=%d K=%d  candidates=%d", M, N, K, nresults);
        for (int i = 0; i < nresults; i++) {
            int algo_id = -1, tile_id = -1;
            cublasLtMatmulAlgoCapGetAttribute(&results[i].algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                              &algo_id, sizeof(algo_id), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id,
                                                 sizeof(tile_id), nullptr);
            IMP_LOG_DEBUG("[gemm-algo]   cand[%d]: numImplFlags=0x%x tile=%d ws=%zu", i, algo_id, tile_id,
                          results[i].workspaceSize);
        }
    }
    // [runtime] deterministic_gemm = true skips timing-based selection so
    // repeat runs produce bitwise-identical prefill outputs.
    const bool s_deterministic_gemm = imp::process_diag_deterministic_gemm();
    if (s_deterministic_gemm || nresults == 1) {
        // Deterministic selection must still be VALID: the cuBLASLt heuristic can return a
        // candidate that faults at runtime on sm_120 (status 14 NOT_SUPPORTED at certain M) even
        // though deterministic mode skips the timing warmup that would normally reject it
        // (FP8-KV forces this path on models FA2 doesn't serve; pre-#932 that included all
        // hd=256 models, surfacing as silent repeated-token garbage on FFN GEMMs). Warmup-probe
        // candidates in heuristic order and pick the first that survives: that order is stable.
        int pick = 0;
        if (nresults > 1 && s_bench_scratch && C_bytes <= s_bench_scratch_size) {
            void* temp_c = s_bench_scratch;
            constexpr int kDetWarmupIters = 2;
            pick = -1;
            for (int i = 0; i < nresults && pick < 0; i++) {
                if (results[i].workspaceSize > s_workspace_size)
                    continue;
                bool ok = true;
                for (int w = 0; w < kDetWarmupIters; w++) {
                    cublasStatus_t wst = cublasLtMatmul(
                        lt, entry.opDesc, p_alpha, B_data, entry.Bdesc, A_data, entry.Adesc, p_zero,
                        temp_c, entry.Cdesc, temp_c, entry.Cdesc, &results[i].algo, s_workspace,
                        results[i].workspaceSize, stream);
                    if (wst != CUBLAS_STATUS_SUCCESS) {
                        ok = false;
                        break;
                    }
                }
                if (ok)
                    pick = i;
            }
            if (pick < 0) {
                // Every candidate faults at runtime — leave has_algo=false so
                // the matmul uses cuBLASLt's own default (and the caller's
                // fallback chain / fatal guard handles a genuine dead shape).
                entry.has_algo = false;
                entry.workspace_size = 0;
                return;
            }
        }
        entry.algo = results[pick].algo;
        entry.algo_M = M;
        entry.workspace_size =
            (results[pick].workspaceSize <= s_workspace_size) ? results[pick].workspaceSize : 0;
        entry.has_algo = true;
        entry.benchmarked = false;
        if (gemm_algo_log_enabled() && M > 0) {
            IMP_LOG_DEBUG("[gemm-algo]   PICKED cand[%d] (deterministic, warmup-validated)", pick);
        }
        return;
    }

    // Use pre-allocated scratch buffer to avoid fragmenting GPU memory
    if (!s_bench_scratch || C_bytes > s_bench_scratch_size) {
        entry.algo = results[0].algo;
        entry.algo_M = M;
        entry.workspace_size = (results[0].workspaceSize <= s_workspace_size) ? results[0].workspaceSize : 0;
        entry.has_algo = true;
        entry.benchmarked = false;
        return;
    }
    void* temp_c = s_bench_scratch;

    CudaEvent start, stop;  // timing enabled for cudaEventElapsedTime
    (void)start.create(cudaEventDefault);
    (void)stop.create(cudaEventDefault);
    std::vector<float> cand_ms(nresults, 1e30f);

    // Warmup all candidates first so steady-state caches are warm before any is timed: a
    // single warmup per candidate let single-rep bench mode pick legacy WMMA paths whose
    // first-call cost is competitive but whose steady-state cost is much higher than modern
    // m16n8k16 tiles. Three warmups covers cuBLASLt's per-algo lazy compile + L2 fill.
    // Tracks algo_ok per candidate since heuristic algos can fail at runtime.
    std::vector<bool> algo_ok(nresults, true);
    constexpr int kWarmupIters = 3;
    for (int i = 0; i < nresults; i++) {
        if (results[i].workspaceSize > s_workspace_size) {
            algo_ok[i] = false;
            continue;
        }
        for (int w = 0; w < kWarmupIters; w++) {
            cublasStatus_t wst = cublasLtMatmul(lt, entry.opDesc, p_alpha, B_data, entry.Bdesc, A_data,
                                                 entry.Adesc, p_zero, temp_c, entry.Cdesc, temp_c, entry.Cdesc,
                                                 &results[i].algo, s_workspace, results[i].workspaceSize, stream);
            if (wst != CUBLAS_STATUS_SUCCESS) {
                algo_ok[i] = false;
                break;
            }
        }
    }

    auto time_candidate = [&](int i, int iters) {
        return time_on_stream(start, stop, stream, iters, [&] {
            cublasLtMatmul(lt, entry.opDesc, p_alpha, B_data, entry.Bdesc, A_data, entry.Adesc, p_zero, temp_c,
                           entry.Cdesc, temp_c, entry.Cdesc, &results[i].algo, s_workspace,
                           results[i].workspaceSize, stream);
        });
    };

    // Probe round — its timings are thrown away, they only size the real rounds.
    std::vector<int> reps(nresults, kBenchmarkIters);
    for (int i = 0; i < nresults; i++) {
        if (!algo_ok[i])
            continue;
        const float per_rep = time_candidate(i, kBenchmarkIters);
        const int want = (per_rep > 1e-5f) ? static_cast<int>(kTargetWindowMs / per_rep) : kMaxBenchIters;
        reps[i] = std::clamp(want, kBenchmarkIters, kMaxBenchIters);
    }

    // The heuristic's own first choice, among candidates that survived warmup.
    // Everything below is measured against it.
    int base = -1;
    for (int i = 0; i < nresults && base < 0; i++) {
        if (algo_ok[i])
            base = i;
    }

    // Interleaved rounds: each round compares every candidate against `base` timed in the
    // SAME round, and a candidate keeps its claim only by beating base by kAlgoMargin in
    // every round. Pairing the comparison within a round makes it survive a bad one (cold
    // clocks / a busy host inflate base and challenger together, so the ratio still carries
    // signal). Comparing per-candidate minima across rounds does not have this property.
    std::vector<bool> beats_base(nresults, base >= 0);
    for (int round = 0; round < kBenchmarkRounds && base >= 0; round++) {
        // Base is timed first and explicitly — every other candidate in this round
        // is judged against this number, so it must exist before they are timed.
        const float base_ms = time_candidate(base, reps[base]);
        if (base_ms < cand_ms[base])
            cand_ms[base] = base_ms;
        beats_base[base] = false;
        for (int i = 0; i < nresults; i++) {
            if (i == base)
                continue;
            if (!algo_ok[i]) {
                beats_base[i] = false;
                continue;
            }
            const float per_rep = time_candidate(i, reps[i]);
            if (per_rep < cand_ms[i])
                cand_ms[i] = per_rep;
            if (per_rep >= base_ms * (1.0f - kAlgoMargin))
                beats_base[i] = false;
        }
    }

    start.reset();
    stop.reset();

    // [diag] per-candidate cost, so the margin below can be chosen from measured
    // spread instead of guessed. Without this only the winner's time is logged,
    // which cannot answer "how much does the choice actually matter for this shape".
    if (gemm_algo_log_enabled() && M > 0) {
        for (int i = 0; i < nresults; i++)
            IMP_LOG_DEBUG("[gemm-algo]   cost[%d] per_ms=%.5f reps=%d%s", i, cand_ms[i], reps[i],
                          algo_ok[i] ? "" : " REJECTED");
    }

    // Lowest-indexed candidate that beat base in every round wins, not the smallest time:
    // when two challengers tie with each other but both beat base, picking by time flips
    // between them run to run. Index order is stable across processes, so this keeps the
    // gain and stays reproducible. A legacy WMMA candidate whose steady-state cost is much
    // worse never beats base by the margin in any round.
    int best_idx = base;
    for (int i = 0; i < nresults; i++) {
        if (beats_base[i]) {
            best_idx = i;
            break;
        }
    }

    const float best_ms = (best_idx >= 0) ? cand_ms[best_idx] : 0.0f;

    if (best_idx < 0 || !algo_ok[best_idx]) {
        entry.has_algo = false;
        entry.workspace_size = 0;
        return;
    }
    entry.algo = results[best_idx].algo;
    entry.algo_M = M;
    entry.workspace_size = results[best_idx].workspaceSize;
    entry.has_algo = true;
    entry.benchmarked = true;
    if (gemm_algo_log_enabled() && M > 0) {
        int picked_tile = -1;
        cublasLtMatmulAlgoConfigGetAttribute(&entry.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &picked_tile,
                                             sizeof(picked_tile), nullptr);
        IMP_LOG_DEBUG("[gemm-algo]   PICKED cand[%d] tile=%d  best_ms=%.3f", best_idx, picked_tile, best_ms);
    }
}

// True when `stream` is not capturing, so the probe may launch and time candidates (#2396).
static bool probe_allowed_on(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    const bool ok = cudaStreamIsCapturing(stream, &st) == cudaSuccess;
    if (!ok)
        (void)cudaGetLastError();  // the failed query's own error, not a launch error
    return gemm_algo_probe_allowed(ok, st != cudaStreamCaptureStatusNone);
}

static int algo_tile_id(const cublasLtMatmulAlgo_t& algo) {
    int tile = -1;
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr);
    return tile;
}

// Cold shape inside a capture: heuristic order, first candidate cublasLtMatmulAlgoCheck accepts.
// Host-only, no launch: probing there invalidated the capture and dropped the size's graph (#2396).
static void select_algo_host_only(cublasLtHandle_t lt, GemmCacheEntry& entry, int64_t M, int64_t K,
                                  int64_t N) {
    cublasLtMatmulPreference_t pref = nullptr;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &s_workspace_size, sizeof(s_workspace_size)));
    cublasLtMatmulHeuristicResult_t results[kMaxAlgoCandidates];
    int nresults = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, entry.opDesc, entry.Bdesc, entry.Adesc, entry.Cdesc, entry.Cdesc, pref,
                                   kMaxAlgoCandidates, results, &nresults);
    cublasLtMatmulPreferenceDestroy(pref);

    std::array<GemmAlgoHostCheck, kMaxAlgoCandidates> checks{};
    for (int i = 0; i < nresults; i++) {
        cublasLtMatmulHeuristicResult_t res{};
        checks[i].supported = cublasLtMatmulAlgoCheck(lt, entry.opDesc, entry.Bdesc, entry.Adesc, entry.Cdesc,
                                                      entry.Cdesc, &results[i].algo,
                                                      &res) == CUBLAS_STATUS_SUCCESS;
        checks[i].workspace = res.workspaceSize;
    }
    const int pick = gemm_capture_safe_pick(std::span(checks.data(), (size_t)std::max(nresults, 0)),
                                            s_workspace_size);
    entry.benchmarked = false;
    entry.algo_M = M;
    entry.has_algo = pick >= 0;
    entry.workspace_size = pick >= 0 ? checks[pick].workspace : 0;
    if (pick >= 0)
        entry.algo = results[pick].algo;
    IMP_LOG_INFO(
        "[gemm-algo] M=%ld K=%ld N=%ld first seen inside a stream capture: no timing, heuristic "
        "cand[%d] of %d (host-checked, tile=%d)",
        (long)M, (long)K, (long)N, pick, nresults, pick >= 0 ? algo_tile_id(entry.algo) : -1);
}

// Probe outside a capture, host-only pick inside one. Same signature as the probe.
static void select_algo(cublasLtHandle_t lt, GemmCacheEntry& entry, const void* A_data, const void* B_data,
                        size_t C_bytes, float alpha, float beta, bool is_int_compute, cudaStream_t stream,
                        int M, int N, int K, bool fp16_scale = false) {
    if (probe_allowed_on(stream))
        benchmark_and_select_algo(lt, entry, A_data, B_data, C_bytes, alpha, beta, is_int_compute, stream, M,
                                  N, K, fp16_scale);
    else
        select_algo_host_only(lt, entry, M, K, N);
}

// Names the algo a capture records for this entry, once per entry. Caller holds the mutex.
static void log_captured_algo_once(GemmCacheEntry& entry, cudaStream_t stream, int64_t M, int64_t K,
                                   int64_t N) {
    if (entry.capture_logged)
        return;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    if (st != cudaStreamCaptureStatusActive)
        return;
    entry.capture_logged = true;
    IMP_LOG_INFO("[gemm-algo] captured GEMM M=%ld K=%ld N=%ld uses the %s algo (chosen at M=%ld, tile=%d)",
                 (long)M, (long)K, (long)N,
                 !entry.has_algo ? "default" : (entry.benchmarked ? "benchmarked" : "heuristic"),
                 (long)entry.algo_M, entry.has_algo ? algo_tile_id(entry.algo) : -1);
}

void gemm_cleanup() {
    // Say it before the caches go: a clipped activation scale is silent
    // otherwise, and the flag lives on the device (#1544).
    nvfp4_report_scale_clipping();
    if (const int f = s_pin_other_m_failures.exchange(0), x = s_exact_m_entries.exchange(0); f > 0 || x > 0)
        IMP_LOG_INFO(
            "[gemm-algo] %d matmul failure(s) with a pin chosen at another M; %d M(s) got an exact-M "
            "entry because the bucket pin failed cublasLtMatmulAlgoCheck",
            f, x);
    std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
    for (auto& [key, entry] : s_gemm_cache) {
        cublasLtMatrixLayoutDestroy(entry.Adesc);
        cublasLtMatrixLayoutDestroy(entry.Bdesc);
        cublasLtMatrixLayoutDestroy(entry.Cdesc);
        cublasLtMatmulDescDestroy(entry.opDesc);
    }
    s_gemm_cache.clear();
}

// Pre-cudaDeviceReset hook (see core/cuda_static_reset.h): frees + nulls the
// lazily-created cuBLAS/cuBLASLt handles and static workspaces so their
// guards re-arm on the next use after the reset.
void gemm_reset_static_cuda_state() {
    gemm_cleanup();  // idempotent: clears the cuBLASLt descriptor/algo cache
    if (s_cublas_handle) {
        (void)cublasDestroy(s_cublas_handle);
        s_cublas_handle = nullptr;
    }
    if (s_cublaslt_handle) {
        (void)cublasLtDestroy(s_cublaslt_handle);
        s_cublaslt_handle = nullptr;
    }
    // Arena-owned since A7 step 8 — the region belongs to the T2 arena, which
    // ~Engine closes. Only the pointers are re-armed here, so the next
    // gemm_init() takes a fresh slice.
    s_workspace = nullptr;
    s_workspace_size = 0;
    s_bench_scratch = nullptr;
    s_bench_scratch_size = 0;
}

// Registered as a pre-cudaDeviceReset hook (#1207); see core/cuda_static_reset.h.
namespace {
IMP_REGISTER_CUDA_STATIC_RESET(gemm_reset_static_cuda_state);
}  // namespace

// gemm: C = alpha*A@B^T + beta*C, A[M,K] B[N,K] C[M,N] all row-major. GGUF weights are
// [out_features,in_features] = [N,K]. cuBLAS is column-major; for row-major C=A@B^T,
// C^T = B@A^T (col-major), so cuBLAS is called with transa=T, transb=N, m=N, n=M, k=K,
// lda=K (B), ldb=K (A), ldc=N (C).

// gemm_try_gemv() (M=1 decode fast path) and gemm_try_sgemm() (FP32 fast path)
// live in gemm_gemv_dtype.cu (declared in gemm_internal.cuh).

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

    // FP16-accumulate prefill fast path (gemm.cublas_fp16_acc): sm_120 runs FP16 TC with
    // FP32 accumulate at 1/4 rate; COMPUTE_16F restores full rate. FP16-only, M>1 (decode
    // M==1 routes through gemm_try_gemv before this, stays FP32 accumulate).
    const bool use_fp16_acc = process_diag_cublas_fp16_acc() && M > 1 && A.qtype == QType::F16 &&
                              B.qtype == QType::F16 && C.qtype == QType::F16;
    if (use_fp16_acc)
        compute_type = CUBLAS_COMPUTE_16F;

    // Capture-safe path: cuBLASLt fails with CUBLAS_STATUS_INTERNAL_ERROR under stream
    // capture on sm_120 for cold shapes (heuristic + workspace allocation aren't graph-safe).
    // Routes FP16 GEMMs to the hand-tuned sm_120 WMMA kernel when the stream is capturing,
    // unless the capturer opted in via gemm_set_lt_capture_allowed() (graph-captured verify,
    // #847, warms every shape eagerly first). A residual status-14 fails that one capture;
    // the engine falls back to eager verify and disables capture after repeated failures.
    if (A.qtype == QType::F16 && B.qtype == QType::F16 && C.qtype == QType::F16 &&
        !gemm_lt_capture_allowed()) {
        cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess &&
            cap_status == cudaStreamCaptureStatusActive) {
            if (gemm_capture_fp16_sm120(A.data, B.data, C.data, (int)M, (int)N, (int)K, alpha, beta,
                                         stream)) {
                return;
            }
        }
    }

    // Mixed-precision output (e.g. FP16xFP16->FP32 for diagnostic precision probes): bypasses
    // cuBLASLt and uses cublasGemmEx directly. cuBLASLt's descriptor + algo selection produces
    // wildly wrong results for FP16->FP32 dims on sm_120; cublasGemmEx with CUBLAS_COMPUTE_32F
    // handles it correctly.
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
        entry = entry_for_m(lt, cache_key, M, cuda_dtype_A, cuda_dtype_C, K, N, [&] {
            GemmCacheEntry new_entry{};
            new_entry.desc_M = M;
            cudaDataType_t scale_type = (compute_type == CUBLAS_COMPUTE_32I)   ? CUDA_R_32I
                                        : (compute_type == CUBLAS_COMPUTE_16F) ? CUDA_R_16F
                                                                                : CUDA_R_32F;

            create_gemm_descriptors(new_entry, compute_type, scale_type, cuda_dtype_A, cuda_dtype_B,
                                    cuda_dtype_C, (int)K, (int)M, (int)N);

            size_t c_bytes = (size_t)M * N * dtype_size(C.qtype);
            select_algo(lt, new_entry, A.data, B.data, c_bytes, alpha, beta,
                        (compute_type == CUBLAS_COMPUTE_32I), stream, (int)M, (int)N, (int)K, use_fp16_acc);
            return new_entry;
        });
        log_captured_algo_once(*entry, stream, M, K, N);
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
            cublasStatus_t fb_st = cublasGemmEx(
                fb_handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &ialpha, B.data, cuda_dtype_B,
                (int)K, A.data, cuda_dtype_A, (int)K, &ibeta, C.data, cuda_dtype_C, (int)N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            if (fb_st != CUBLAS_STATUS_SUCCESS) {
                // Both cublasLt and the cublasGemmEx fallback failed: C holds garbage. Continuing
                // corrupts the forward pass silently (repeated-token gibberish + downstream IMA).
                // Fail loudly instead.
                char msg[192];
                snprintf(msg, sizeof(msg),
                         "gemm(INT): cublasLtMatmul + cublasGemmEx fallback both failed (status %d) "
                         "M=%ld K=%ld N=%ld — aborting rather than emitting garbage",
                         (int)fb_st, (long)M, (long)K, (long)N);
                throw std::runtime_error(msg);
            }
        }
    } else {
        // COMPUTE_16F descriptors take __half alpha/beta (scale type R_16F).
        const __half h_alpha = __float2half(alpha);
        const __half h_beta = __float2half(beta);
        const void* p_alpha = use_fp16_acc ? static_cast<const void*>(&h_alpha)
                                           : static_cast<const void*>(&alpha);
        const void* p_beta = use_fp16_acc ? static_cast<const void*>(&h_beta)
                                          : static_cast<const void*>(&beta);
        cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc, p_alpha, B.data, entry->Bdesc, A.data,
                                           entry->Adesc, p_beta, C.data, entry->Cdesc, C.data, entry->Cdesc,
                                           entry->has_algo ? &entry->algo : nullptr, s_workspace,
                                           entry->workspace_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            const bool pin_belongs_to_other_m = keep_pin_chosen_at_other_m(*entry, M, K, N);
            const cublasLtMatmulAlgo_t* retry_algo =
                (pin_belongs_to_other_m || !entry->has_algo) ? nullptr : &entry->algo;
            st = cublasLtMatmul(lt, entry->opDesc, p_alpha, B.data, entry->Bdesc, A.data, entry->Adesc,
                                p_beta, C.data, entry->Cdesc, C.data, entry->Cdesc, retry_algo, s_workspace,
                                pin_belongs_to_other_m ? 0 : entry->workspace_size, stream);
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
                    // Both cublasLt and cublasGemmEx failed. Output buffer holds garbage; continuing
                    // corrupts the forward pass silently (repeated-token gibberish + downstream IMA). Fail
                    // loudly: under CUDA-graph capture this aborts the capture (per-step fallback) rather
                    // than baking in garbage.
                    char msg[192];
                    snprintf(msg, sizeof(msg),
                             "gemm: cublasLtMatmul + cublasGemmEx fallback both failed (status %d) "
                             "M=%ld K=%ld N=%ld dtA=%d dtB=%d dtC=%d — aborting rather than "
                             "emitting garbage",
                             (int)fb_st, (long)M, (long)K, (long)N, (int)cuda_dtype_A,
                             (int)cuda_dtype_B, (int)cuda_dtype_C);
                    throw std::runtime_error(msg);
                }
            }
        }
    }
}

void gemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta, cudaStream_t stream) {
    // Defensive guard: a packed NVFP4 weight (or INT8-typed packed payload) must never reach
    // the generic FP16 cuBLAS path. cuBLAS rejects FP16xINT8/NVFP4 with
    // CUBLAS_STATUS_NOT_SUPPORTED and leaves the output uninitialised (silent garbage + IMA).
    // Correct dispatch is the NVFP4 decode/CUTLASS cache (pre_dequant_phase3_*); if one slips
    // through, fail loud and skip the multiply (output is pre-zeroed by the caller) rather
    // than corrupt.
    if ((B.qtype == QType::INT8 || B.qtype == QType::NVFP4) && A.qtype == QType::F16) {
        static std::atomic<int> leak_count{0};
        int n = leak_count.fetch_add(1, std::memory_order_relaxed);
        if (n < 20) {
            IMP_LOG_ERROR(
                "gemm(): packed NVFP4 weight reached the generic cuBLAS path — "
                "kind=%s B.qtype=%d scales=%s M=%ld K=%ld N=%ld — skipping "
                "(routing/tier bug; expected NVFP4 decode/CUTLASS dispatch)",
                tensor_kind_name(B.kind), std::to_underlying(B.qtype),
                B.scales ? "SET" : "NULL",
                (long)A.shape[0], (long)A.shape[1], (long)B.shape[0]);
        }
        return;
    }
    // Guards against quantized weight tensors (e.g. MXFP4 with dtype=INT4) that should have
    // been handled by the FP16 weight cache path: passing raw quantized data to cuBLAS
    // causes illegal memory access (cuBLAS reads sizeof(FP16)*numel but only
    // sizeof(quant)*numel bytes exist).
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

// gemv() and the dtype GEMV kernels (fp32/fp16/bf16) live in
// gemm_gemv_dtype.cu.

// gemm_cublaslt: cuBLASLt GEMM with explicit algorithm selection + FP8 scales. Uses the
// same static workspace as gemm().
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
        entry = entry_for_m(lt, cache_key, M, cuda_dtype_A, cuda_dtype_C, K, N, [&] {
            GemmCacheEntry new_entry{};
            new_entry.desc_M = M;

            create_gemm_descriptors(new_entry, CUBLAS_COMPUTE_32F, CUDA_R_32F, cuda_dtype_A, cuda_dtype_B,
                                    cuda_dtype_C, (int)K, (int)M, (int)N);

            set_gemm_scale_pointers(new_entry.opDesc, aScale, bScale);

            size_t c_bytes = (size_t)M * N * dtype_size(C.qtype);
            select_algo(lt, new_entry, A.data, B.data, c_bytes, alpha, beta, false, stream, (int)M, (int)N,
                        (int)K);
            return new_entry;
        });
        log_captured_algo_once(*entry, stream, M, K, N);
    }

    // Sets per-call scale pointers (vary by weight tensor, not cached) on the cached opDesc
    // WITHOUT holding s_gemm_cache_mutex. Safe because imp serializes all GEMM calls on a
    // single CUDA stream, so no two threads race on the same opDesc. If multi-stream GEMM
    // is added, this needs the mutex or a per-call opDesc duplicate.
    set_gemm_scale_pointers(entry->opDesc, aScale, bScale);

    cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc, &alpha, B.data, entry->Bdesc, A.data, entry->Adesc,
                                       &beta, C.data, entry->Cdesc, C.data, entry->Cdesc,
                                       entry->has_algo ? &entry->algo : nullptr, s_workspace,
                                       entry->workspace_size, stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        const bool pin_belongs_to_other_m = keep_pin_chosen_at_other_m(*entry, M, K, N);
        set_gemm_scale_pointers(entry->opDesc, aScale, bScale);
        const cublasLtMatmulAlgo_t* retry_algo =
            (pin_belongs_to_other_m || !entry->has_algo) ? nullptr : &entry->algo;
        st = cublasLtMatmul(lt, entry->opDesc, &alpha, B.data, entry->Bdesc, A.data, entry->Adesc, &beta,
                            C.data, entry->Cdesc, C.data, entry->Cdesc, retry_algo, s_workspace,
                            pin_belongs_to_other_m ? 0 : entry->workspace_size, stream);

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
                cudaError_t serr = cudaSuccess;
                if (aScale)
                    serr = cudaMemcpy(&a_scale_h, aScale, sizeof(float), cudaMemcpyDeviceToHost);
                if (bScale && serr == cudaSuccess)
                    serr = cudaMemcpy(&b_scale_h, bScale, sizeof(float), cudaMemcpyDeviceToHost);
                if (serr != cudaSuccess) {  // a wrong scale would be a silent wrong result
                    IMP_LOG_ERROR("gemm_cublaslt: FP8 fallback scale read failed M=%ld K=%ld N=%ld: %s",
                                  (long)M, (long)K, (long)N, cudaGetErrorString(serr));
                    return;
                }

                int a_elems = static_cast<int>(M * K);
                int b_elems = static_cast<int>(N * K);
                half *d_a16 = nullptr, *d_b16 = nullptr;
                if (cudaMallocAsync(&d_a16, static_cast<size_t>(a_elems) * sizeof(half), stream) != cudaSuccess)
                    d_a16 = nullptr;
                if (cudaMallocAsync(&d_b16, static_cast<size_t>(b_elems) * sizeof(half), stream) != cudaSuccess)
                    d_b16 = nullptr;
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
                if (d_a16) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_a16, stream));
                if (d_b16) IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_b16, stream));
            }
        }
    }
}

// Block-scaled NVFP4 via cuBLASLt: TN, COMPUTE_32F, VEC16_UE4M3 on both sides. Same algo cache
// and timed probe as gemm(); scale pointers are set per call like gemm_cublaslt's FP8 scales.
bool gemm_nvfp4_cublaslt(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf,
                         float alpha, void* d_fp16, int M, int N, int K, cudaStream_t stream) {
    cublasLtHandle_t lt = get_cublaslt_handle();
    if (!lt)
        return false;
    // cuBLAS "A" operand is the weight (Bdesc of the row-major convention), "B" the activation.
    auto set_scales = [&](cublasLtMatmulDesc_t desc) {
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       static_cast<const void*>(&w_sf), sizeof(w_sf));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       static_cast<const void*>(&a_sf), sizeof(a_sf));
    };
    const GemmCacheKey cache_key{
        CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16F, CUBLAS_COMPUTE_32F, bucket_m(M), K, N, true};
    GemmCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
        entry = entry_for_m(lt, cache_key, M, CUDA_R_4F_E2M1, CUDA_R_16F, K, N, [&] {
            GemmCacheEntry new_entry{};
            new_entry.desc_M = M;
            create_gemm_descriptors(new_entry, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_4F_E2M1, CUDA_R_4F_E2M1,
                                    CUDA_R_16F, K, M, N);
            const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            for (auto attr : {CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_MODE})
                cublasLtMatmulDescSetAttribute(new_entry.opDesc, attr, &mode, sizeof(mode));
            set_scales(new_entry.opDesc);
            select_algo(lt, new_entry, a_data, w_data, (size_t)M * N * sizeof(half), alpha, 0.0f, false,
                        stream, M, N, K);
            return new_entry;
        });
        log_captured_algo_once(*entry, stream, M, K, N);
    }
    if (!entry->has_algo)
        return false;
    // Single GEMM stream, as in gemm_cublaslt: the shared opDesc takes this call's scales.
    set_scales(entry->opDesc);
    const float beta = 0.0f;
    cublasStatus_t st = cublasLtMatmul(lt, entry->opDesc, &alpha, w_data, entry->Bdesc, a_data, entry->Adesc,
                                       &beta, d_fp16, entry->Cdesc, d_fp16, entry->Cdesc, &entry->algo,
                                       s_workspace, entry->workspace_size, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        const bool pin_belongs_to_other_m = keep_pin_chosen_at_other_m(*entry, M, K, N);
        set_scales(entry->opDesc);
        st = cublasLtMatmul(lt, entry->opDesc, &alpha, w_data, entry->Bdesc, a_data, entry->Adesc, &beta,
                            d_fp16, entry->Cdesc, d_fp16, entry->Cdesc,
                            (pin_belongs_to_other_m || !entry->has_algo) ? nullptr : &entry->algo,
                            s_workspace, pin_belongs_to_other_m ? 0 : entry->workspace_size, stream);
    }
    return st == CUBLAS_STATUS_SUCCESS;
}

// dtype GEMV kernels (fp8/q6k/q8_0) + gemv_fp8 live in gemm_gemv_dtype.cu. MoE
// gate/decode/gate-up-fused GEMV kernels live in gemm_moe_gemv.cu. gemm_kv_batched /
// gemm_pair_batched / gemm_cublaslt_fp8_probe live in gemm_batched.cu.

}  // namespace imp
