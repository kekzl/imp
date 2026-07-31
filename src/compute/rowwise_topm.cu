#include "compute/rowwise_topm.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

#include <cfloat>

namespace imp {

// Two-stage split top-M (the rowwise_argmax_partial pattern — the original
// one-block-per-row form measured 1.4 ms per verify at 151k vocab: 4 blocks
// on 170 SMs re-scanning the row m times).
//
// Stage 1: grid (rows, kTopMSplits); each block computes the local top-M of
// its V/kTopMSplits slice via m sequential masked block-reductions.
// Stage 2: one block per row merges kTopMSplits * m candidates.
constexpr int kTopMSplits = 32;

__global__ void rowwise_topm_partial_kernel(const float* __restrict__ logits, int V, int m,
                                            float* __restrict__ pvals,
                                            int32_t* __restrict__ pidxs) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    const int chunk = (V + kTopMSplits - 1) / kTopMSplits;
    const int begin = split * chunk;
    const int end = min(V, begin + chunk);
    const float* lg = logits + static_cast<int64_t>(row) * V;
    const int tid = threadIdx.x;
    __shared__ int s_sel[kRowwiseTopMMax];
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    const int64_t out_base = (static_cast<int64_t>(row) * kTopMSplits + split) * m;
    for (int pass = 0; pass < m; ++pass) {
        float best = -FLT_MAX;
        int best_idx = V;  // sentinel: loses every tie-break
        for (int i = begin + tid; i < end; i += blockDim.x) {
            bool taken = false;
            for (int e = 0; e < pass; ++e)
                if (s_sel[e] == i) {
                    taken = true;
                    break;
                }
            if (taken)
                continue;
            const float v = lg[i];
            if (v > best || (v == best && i < best_idx)) {
                best = v;
                best_idx = i;
            }
        }
        s_val[tid] = best;
        s_idx[tid] = best_idx;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_val[tid + s] > s_val[tid] ||
                    (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_sel[pass] = s_idx[0];
            pvals[out_base + pass] = s_val[0];
            pidxs[out_base + pass] = s_idx[0];
        }
        __syncthreads();
    }
}

// One block per row, one thread: serial selection over kTopMSplits*m
// candidates (<= 512 values — trivial next to the stage-1 scan).
__global__ void rowwise_topm_reduce_kernel(const float* __restrict__ pvals,
                                           const int32_t* __restrict__ pidxs, int V, int m,
                                           int32_t* __restrict__ out) {
    if (threadIdx.x != 0)
        return;
    const int row = blockIdx.x;
    const int n = kTopMSplits * m;
    const float* pv = pvals + static_cast<int64_t>(row) * n;
    const int32_t* pi = pidxs + static_cast<int64_t>(row) * n;
    int32_t sel[kRowwiseTopMMax];
    for (int pass = 0; pass < m; ++pass) {
        float best = -FLT_MAX;
        int best_idx = V;
        for (int i = 0; i < n; ++i) {
            const int32_t idx = pi[i];
            bool taken = false;
            for (int e = 0; e < pass; ++e)
                if (sel[e] == idx) {
                    taken = true;
                    break;
                }
            if (taken)
                continue;
            const float v = pv[i];
            if (v > best || (v == best && idx < best_idx)) {
                best = v;
                best_idx = idx;
            }
        }
        sel[pass] = best_idx;
        out[static_cast<int64_t>(row) * m + pass] = best_idx;
    }
}

// Persistent partial scratch (rows*kTopMSplits*m floats+ints). Graph-safe:
// allocated once at first use for the max shape and never freed mid-run.
namespace {
float* g_topm_pvals = nullptr;
int32_t* g_topm_pidxs = nullptr;
size_t g_topm_cap = 0;
uint64_t g_topm_gen = 0;  // arena generation; a model swap invalidates the pair

// T2 (A7 step 8). Same shape as AUDIT B13's family and for the same reason: the
// two pointers below are kernel parameters, the old grow freed them, and the
// comment in rowwise_topm() already conceded that "growing under stream capture
// would abort the capture — callers warm the shape up eagerly first". An arena
// take is not a CUDA call, so it cannot abort a capture, and it does not free
// the slice a captured graph may still name.
//
// Uncharged in exec_t2_demand: the one live caller in the engine is
// executor_perplexity.cu (the --perplexity tool path, rows = the eval chunk),
// and rowwise_topm_reserve() has no caller at all today — the header's claim
// that "callers warm the shape up eagerly first (spec-capture warmup does)" is
// stale, which the arena move makes harmless rather than latent. Charging the
// max_tokens worst case would reserve ~16 MiB for a path that is not on the
// serving hot path at all, so it draws on the arena's slack and returns without
// a result if that runs out.
bool topm_ensure(size_t need) {
    const uint64_t g = engine_arena().generation();
    if (g_topm_pvals && g_topm_gen == g && need <= g_topm_cap)
        return true;
    if (g_topm_gen != g) {
        g_topm_pvals = nullptr;
        g_topm_pidxs = nullptr;
        g_topm_cap = 0;
        g_topm_gen = g;
    }
    auto pv = engine_arena().take_bytes(need * sizeof(float));
    auto pi = engine_arena().take_bytes(need * sizeof(int32_t));
    if (pv.empty() || pi.empty()) {
        IMP_LOG_WARN(
            "rowwise_topm: %.2f MiB unavailable from the T2 arena (%.1f MiB free) — top-M "
            "returns empty for this call",
            need * 8.0 / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0));
        g_topm_pvals = nullptr;
        g_topm_pidxs = nullptr;
        g_topm_cap = 0;
        return false;
    }
    g_topm_pvals = reinterpret_cast<float*>(pv.data());
    g_topm_pidxs = reinterpret_cast<int32_t*>(pi.data());
    g_topm_cap = need;
    return true;
}
}  // namespace

void rowwise_topm_reserve(int rows, int m) {
    if (rows <= 0 || m <= 0)
        return;
    if (m > kRowwiseTopMMax)
        m = kRowwiseTopMMax;
    (void)topm_ensure(static_cast<size_t>(rows) * kTopMSplits * m);
}

void rowwise_topm(const float* d_logits, int rows, int vocab, int m, int32_t* d_out,
                  cudaStream_t stream) {
    if (rows <= 0 || vocab <= 0 || m <= 0 || !d_logits || !d_out)
        return;
    if (m > kRowwiseTopMMax)
        m = kRowwiseTopMMax;
    const size_t need = static_cast<size_t>(rows) * kTopMSplits * m;
    if (!topm_ensure(need))
        return;
    dim3 grid(rows, kTopMSplits);
    rowwise_topm_partial_kernel<<<grid, 256, 0, stream>>>(d_logits, vocab, m, g_topm_pvals,
                                                          g_topm_pidxs);
    IMP_CUDA_CHECK_LAUNCH();
    rowwise_topm_reduce_kernel<<<rows, 32, 0, stream>>>(g_topm_pvals, g_topm_pidxs, vocab, m,
                                                        d_out);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
