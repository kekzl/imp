#include "compute/gemm.h"
#include <atomic>
#include "compute/gemm_capture_fp16_sm120.h"
#include "core/cuda_static_reset.h"
#include "compute/gemm_internal.cuh"
#include "core/logging.h"
#include "core/tensor_kind.h"
#include "memory/engine_arena.h"
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
#include <algorithm>
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

// warp_reduce_sum / kGemvThreads / kGemvWarps / gemv_blocks live in
// gemm_internal.cuh (shared with gemm_gemv_dtype.cu + gemm_moe_gemv.cu).
// kGemmAlgo moved to gemm_batched.cu (its only remaining user).

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

// ---------------------------------------------------------------------------
// Shared workspace for cuBLASLt — taken once from the engine-persistent (T2)
// arena via gemm_init(), used by all GEMM calls.  cuBLASLt takes the workspace
// as an argument, so ONE slice sized at the plan's maximum serves every call:
// per-handle would multiply it by handle count for no benefit
// (docs/MEMORY_ARCHITECTURE.md A5.3).
// ---------------------------------------------------------------------------
static void* s_workspace = nullptr;
static size_t s_workspace_size = 0;

// Bench scratch for algo selection — the C buffer the candidate algos write
// while being timed. Also T2: the alternative is a cudaMalloc/cudaFree per
// cache miss, which fragments and can run while serving.
static void* s_bench_scratch = nullptr;
static size_t s_bench_scratch_size = 0;

void gemm_init() {
    // Force handle creation early.
    get_cublas_handle();
    get_cublaslt_handle();

    // T2 (A7 step 8). Both buffers are engine-lifetime and both already
    // DEGRADE cleanly to null — a 0-byte workspace makes cuBLASLt's heuristic
    // return only algos that need none, and a null bench scratch skips the
    // timing loop and takes the heuristic's first choice. That is why they can
    // leave the I1 allowlist outright instead of keeping a cudaMalloc fallback
    // the gate cannot see (AUDIT B47). Outside an Engine — the GPU test
    // binaries call gemm_init() directly — the arena is not open, take_bytes()
    // returns empty, and both stay null.
    if (!s_workspace) {
        // The size ladder is kept from the pre-arena code, but it now degrades
        // against the ARENA rather than against free VRAM: exec_t2_demand
        // charges the full kGemmCublasWorkspaceBytes, so anything below it
        // means the plan under-reserved, which is worth a line in the log
        // rather than a silent halving.
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

    // Also let legacy cuBLAS API use the same workspace.
    if (s_workspace) {
        cublasSetWorkspace(get_cublas_handle(), s_workspace, s_workspace_size);
    }

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

// Selection stability (F-9). A single timed sample per candidate made the choice
// a coin flip at small M: measured over 5 fresh processes on Qwen3-1.7B, all four
// M=512 shapes picked the same tile every time, while all four M=16 shapes picked
// 3-4 *different* tiles, and the winning time for one shape ranged 0.055-0.321 ms
// (5.8x). At M=16 a candidate runs ~0.05 ms, so kBenchmarkIters spans ~0.25 ms —
// short enough that one scheduling hiccup decides the winner.
//
// Two fixes, both on the estimator rather than on persistence. An on-disk algo
// cache (the other candidate fix) would freeze whatever the first noisy run chose
// and hand it to every later process, turning a per-process mispick into a
// permanent one — on top of needing invalidation against driver/cuBLAS version.
//
// Logging every candidate's cost (not just the winner's) showed where the
// instability actually lives: shapes whose best candidate is genuinely ahead pick
// the same one every time — M=512 N=6144 K=2048 spans 0.196-0.449 ms and chose
// cand[0] in 4/4 runs, its cost reproducing to 0.3 % (0.1961/0.1964/0.1961/0.1966),
// and it chose right even in a run where cold clocks inflated everything 5x. Every
// unstable shape instead has its top candidates bunched within ~5-10 %, i.e. inside
// the measurement's own error. So the flips are ties resolved by noise, and the
// throughput at stake in a flip is bounded by how close the tie is.
//
// Two fixes, both on the estimator rather than on persistence. An on-disk algo
// cache (the other candidate fix) would freeze whatever the first noisy run chose
// and hand it to every later process, turning a per-process mispick into a
// permanent one — on top of needing invalidation against driver/cuBLAS version.
//
//  1. Size the timed window instead of fixing the rep count. A fixed
//     kBenchmarkIters makes the window scale with the shape, and at M=16 a
//     candidate runs ~6-24 us, so five reps time ~30-120 us — mostly launch
//     overhead. A probe round now sizes each candidate's reps for a
//     ~kTargetWindowMs window, which puts every shape at comparable measurement
//     quality. Costs are then compared per rep.
//  2. kAlgoMargin hysteresis toward heuristic order. A candidate replaces the
//     incumbent only if it beats it by more than the margin; otherwise the lower
//     heuristic index wins. Heuristic order is deterministic for a given shape and
//     device, so candidates inside the margin resolve the same way in every process
//     instead of by measurement noise. The margin is set from the measured residual
//     spread, not guessed — an earlier 3 % attempt sat below the noise and left the
//     picks as unstable as before.
static constexpr int kBenchmarkRounds = 3;
static constexpr float kTargetWindowMs = 0.5f;
static constexpr int kMaxBenchIters = 512;
static constexpr float kAlgoMargin = 0.10f;

// Diagnostic: when diagnostics.log_gemm_algo is set, log shape + per-candidate algoId/tileId + chosen algo for every
// benchmark_and_select_algo call. Used to enumerate which exact GEMM shapes
// select cuBLAS legacy WMMA kernels (Finding 1/5).
static int gemm_algo_log_enabled() { return imp::process_diag_log_gemm_algo() ? 1 : 0; }

static void benchmark_and_select_algo(cublasLtHandle_t lt, GemmCacheEntry& entry, const void* A_data,
                                      const void* B_data, size_t C_bytes, float alpha, float beta,
                                      bool is_int_compute, cudaStream_t stream, int M = 0, int N = 0,
                                      int K = 0, bool fp16_scale = false) {
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
        // Deterministic selection must still be VALID: the cuBLASLt heuristic
        // can return a top candidate that faults at RUNTIME on sm_120 (status
        // 14 / NOT_SUPPORTED at certain M) — the timing path below rejects
        // those in its warmup, but deterministic mode skips timing. Blindly
        // trusting results[0] here is what let an unvalidated algo reach the
        // real matmul and corrupt the forward pass (FP8-KV forces this path
        // model-wide on models FA2 doesn't serve — pre-#932 that included all
        // hd=256 models; the failure surfaced as silent repeated-token garbage
        // on Qwen3.6-35B FFN GEMMs). Warmup-probe the
        // candidates in heuristic order and pick the FIRST that survives —
        // that order is stable across runs, so determinism is preserved.
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
        entry.workspace_size =
            (results[pick].workspaceSize <= s_workspace_size) ? results[pick].workspaceSize : 0;
        entry.has_algo = true;
        if (gemm_algo_log_enabled() && M > 0) {
            IMP_LOG_DEBUG("[gemm-algo]   PICKED cand[%d] (deterministic, warmup-validated)", pick);
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
    std::vector<float> cand_ms(nresults, 1e30f);

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
        cudaEventRecord(start, stream);
        for (int r = 0; r < iters; r++)
            cublasLtMatmul(lt, entry.opDesc, p_alpha, B_data, entry.Bdesc, A_data, entry.Adesc, p_zero, temp_c,
                           entry.Cdesc, temp_c, entry.Cdesc, &results[i].algo, s_workspace,
                           results[i].workspaceSize, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms / static_cast<float>(iters);
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

    // Interleaved rounds. Each round compares every candidate against `base`
    // timed in that SAME round, and a candidate keeps its claim only by beating
    // base by kAlgoMargin in every round. Pairing the comparison inside a round is
    // what makes it survive a bad one: cold clocks or a busy host inflate base and
    // challenger together, so their ratio still carries signal where their absolute
    // times carry none. Comparing per-candidate minima across rounds does not have
    // this property, and measurably did not work.
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // [diag] per-candidate cost, so the margin below can be chosen from measured
    // spread instead of guessed. Without this only the winner's time is logged,
    // which cannot answer "how much does the choice actually matter for this shape".
    if (gemm_algo_log_enabled() && M > 0) {
        for (int i = 0; i < nresults; i++)
            IMP_LOG_DEBUG("[gemm-algo]   cost[%d] per_ms=%.5f reps=%d%s", i, cand_ms[i], reps[i],
                          algo_ok[i] ? "" : " REJECTED");
    }

    // Lowest-indexed candidate that beat base in every round wins; otherwise base.
    // Taking the lowest index rather than the smallest time matters: where two
    // challengers are tied with each other but both clearly ahead of base — which is
    // exactly what M=512 N=1024 K=2048 does here, cand[3] and cand[5] both ~8-45 %
    // ahead — picking by time flips between them run to run, and demanding a single
    // undisputed winner throws away a real gain to keep the slower base. Index order
    // is stable across processes, so this takes the gain and stays reproducible.
    //
    // The reason the timing loop exists is preserved: a legacy WMMA candidate whose
    // steady-state cost is 3-9x worse never beats base by the margin in any round,
    // and a base that bad loses to every challenger in every round.
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
    entry.workspace_size = results[best_idx].workspaceSize;
    entry.has_algo = true;
    if (gemm_algo_log_enabled() && M > 0) {
        int picked_tile = -1;
        cublasLtMatmulAlgoConfigGetAttribute(&entry.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &picked_tile,
                                             sizeof(picked_tile), nullptr);
        IMP_LOG_DEBUG("[gemm-algo]   PICKED cand[%d] tile=%d  best_ms=%.3f", best_idx, picked_tile, best_ms);
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

    // FP16-accumulate prefill fast path (gemm.cublas_fp16_acc): GeForce
    // sm_120 runs FP16 TC with FP32 accumulate at 1/4 rate; COMPUTE_16F
    // restores full rate (~2x measured on prefill shapes). F16-only, M>1 —
    // decode (M==1) routes through gemm_try_gemv before this and stays 32F.
    const bool use_fp16_acc = process_diag_cublas_fp16_acc() && M > 1 && A.qtype == QType::F16 &&
                              B.qtype == QType::F16 && C.qtype == QType::F16;
    if (use_fp16_acc)
        compute_type = CUBLAS_COMPUTE_16F;

    // Capture-safe path: cuBLASLt fails with CUBLAS_STATUS_INTERNAL_ERROR
    // (status 14) under stream capture on sm_120 — heuristic + workspace
    // allocation paths aren't graph-safe on a COLD shape. Route FP16×FP16→FP16
    // GEMMs to the hand-tuned sm_120 WMMA kernel when the stream is in capture
    // mode — unless the capturer opted in via gemm_set_lt_capture_allowed():
    // the graph-captured verify chunk (#847) warms every shape eagerly before
    // capturing, so Lt's heuristic cache and handle workspace are populated
    // and the call records cleanly (the WMMA fallback measured ~5x slower than
    // Lt's nvjet kernels on the verify GEMMs — 167 ms/1400 tok on Q8-8B). A
    // residual status-14 fails that one capture; the engine falls back to the
    // eager verify and disables capture after repeated failures.
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
            cudaDataType_t scale_type = (compute_type == CUBLAS_COMPUTE_32I)   ? CUDA_R_32I
                                        : (compute_type == CUBLAS_COMPUTE_16F) ? CUDA_R_16F
                                                                                : CUDA_R_32F;

            create_gemm_descriptors(new_entry, compute_type, scale_type, cuda_dtype_A, cuda_dtype_B,
                                    cuda_dtype_C, (int)K, (int)M, (int)N);

            size_t c_bytes = (size_t)M * N * dtype_size(C.qtype);
            benchmark_and_select_algo(lt, new_entry, A.data, B.data, c_bytes, alpha, beta,
                                      (compute_type == CUBLAS_COMPUTE_32I), stream, (int)M, (int)N, (int)K,
                                      use_fp16_acc);

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
            cublasStatus_t fb_st = cublasGemmEx(
                fb_handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &ialpha, B.data, cuda_dtype_B,
                (int)K, A.data, cuda_dtype_A, (int)K, &ibeta, C.data, cuda_dtype_C, (int)N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            if (fb_st != CUBLAS_STATUS_SUCCESS) {
                // Both cublasLt and the cublasGemmEx fallback failed: C holds
                // garbage. Continuing corrupts the forward pass silently
                // (repeated-token gibberish + downstream IMA). Fail loudly —
                // the throw is translated to ImpError at the API boundary.
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
            // Stale algo from a different M within the same bucket.
            // Re-select via heuristic and retry before falling back.
            {
                std::lock_guard<std::mutex> lock(s_gemm_cache_mutex);
                reselect_algo_for_entry(*entry);
            }
            st = cublasLtMatmul(lt, entry->opDesc, p_alpha, B.data, entry->Bdesc, A.data, entry->Adesc,
                                p_beta, C.data, entry->Cdesc, C.data, entry->Cdesc,
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
                    // garbage; continuing corrupts the forward pass silently
                    // (repeated-token gibberish + downstream IMA). Fail loudly
                    // instead — the throw is translated to ImpError at the API
                    // boundary, and under CUDA-graph capture it aborts the
                    // capture (per-step fallback) rather than baking in garbage.
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
    // Defensive guard: a packed NVFP4 weight (or its INT8-typed packed
    // payload) must never reach the generic FP16 cuBLAS path. cuBLAS rejects
    // the FP16xINT8/NVFP4 operand mix with CUBLAS_STATUS_NOT_SUPPORTED
    // (status 15) and leaves an uninitialised output buffer — silent
    // repeated-token garbage plus downstream illegal-memory-access crashes.
    // The correct dispatch for these weights is the NVFP4 decode/CUTLASS cache
    // (see pre_dequant_phase3_*), which the budget planner now reserves for
    // (the double-counted-reserve fix). If one still slips through — a future
    // budget regression or an un-tiered weight — fail LOUD and skip the
    // multiply instead of corrupting the forward pass. The output buffer is
    // pre-zeroed by the caller, so an early return is safe (wrong-but-bounded,
    // never an IMA). Log the exact identity, bounded to avoid flooding.
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

// gemv() and the dtype GEMV kernels (fp32/fp16/bf16) live in
// gemm_gemv_dtype.cu.

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

// The dtype GEMV kernels (fp8/q6k/q8_0) + gemv_fp8 live in gemm_gemv_dtype.cu.
// The MoE gate/decode/gate-up-fused GEMV kernels live in gemm_moe_gemv.cu.
// gemm_kv_batched / gemm_pair_batched / gemm_cublaslt_fp8_probe live in
// gemm_batched.cu.

}  // namespace imp
