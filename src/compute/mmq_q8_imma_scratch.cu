// Owns IMMA prefill family memory (recompile-blast-radius split from mmq_q8_imma.cu):
// Q8_0 weight planes, Q6_K repack, activation triple, split-K partials, and the
// preallocation that takes the latter two from the T2 arena at their planned bound.
// Q8_0/Q6_K weight caches are direct allocations (MODEL-resident, T1/A7 step 6);
// activation/split-K scratches are engine-persistent (T2, moved in A7 step 8/AUDIT B13).
// State lives in mmq_q8_imma_internal.cuh (not file-static): mmq_q8_imma.cu's dispatch
// reads the scratch pointers directly for kernel args.

#include "compute/mmq_q8_imma.h"
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include "memory/mem_account.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace imp {

// Kernels that PREPARE these buffers (Q8_0 plane split, Q6_K 224-B repack, activation
// quantizer) live here with their data, not with the dispatch.
namespace {
__global__ void q6k_repack_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                  size_t n_blocks) {
    const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n_blocks * 105;  // 210 B as 105 u16 copies
    if (b >= total) return;
    const size_t blk = b / 105;
    const size_t off = (b % 105) * 2;
    uint16_t v;
    memcpy(&v, src + blk * 210 + off, 2);
    memcpy(dst + blk * kQ6Stride + off, &v, 2);
}


// Q8_0 SoA split: raw 34-B blocks {half d; int8 qs[32]} (2-aligned, memcpy only) ->
// qs plane[N][K] s8 + interleaved (alpha=d, beta=0) plane[N][K/32][2].
__global__ void q8_split_kernel(const uint8_t* __restrict__ src, int8_t* __restrict__ qs_plane,
                                __half* __restrict__ sc_plane, int n_blocks_total) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks_total) return;
    const uint8_t* blk = src + static_cast<size_t>(b) * 34;
    __half d;
    memcpy(&d, blk, 2);
    sc_plane[static_cast<size_t>(b) * 2] = d;
    sc_plane[static_cast<size_t>(b) * 2 + 1] = __float2half(0.0f);
    int8_t* dst = qs_plane + static_cast<size_t>(b) * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = static_cast<int8_t>(blk[2 + i]);
}

// Activation quantizer: 8 warps per block, grid-stride over (m, sub) pairs
// (the shared 32-thread-block version ran at ~150 GB/s). Emits s8 + half
// scale + float rowsum (the rowsum couples to the Q4_K β term; ~free here).
__global__ void quantize_act_fast_kernel(const __half* __restrict__ X, int M, int K,
                                         int8_t* __restrict__ xs8, __half* __restrict__ xscale,
                                         float* __restrict__ xrowsum) {
    const int subs = K / 32;
    const int total = M * subs;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int nwarps = gridDim.x * (blockDim.x >> 5);
    for (int idx = warp; idx < total; idx += nwarps) {
        const int m = idx / subs;
        const int s = idx - m * subs;
        const size_t off = static_cast<size_t>(m) * K + s * 32;
        const float v = __half2float(X[off + lane]);
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = __float2int_rn(v * inv);
        q = max(-127, min(127, q));
        xs8[off + lane] = static_cast<int8_t>(q);
        int sq = q;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xFFFFFFFFu, sq, o);
        if (lane == 0) {
            xscale[static_cast<size_t>(m) * subs + s] =
                __float2half((amax > 0.0f) ? (amax / 127.0f) : 0.0f);
            xrowsum[static_cast<size_t>(m) * subs + s] = static_cast<float>(sq);
        }
    }
}

}  // namespace



std::mutex g_imma_mtx;
std::unordered_map<const void*, WeightPlanes> g_imma_weights;
std::unordered_map<const void*, Q6kRepack> g_imma_q6k;
ActScratch g_imma_act;

// Q8_0 weight caches (up to 8.6 GiB, 1.125B/element) are taken lazily on first prefill,
// best-effort: cudaMalloc failure drops that GEMM to the dequant path (#1899).
// Size is decided by the planner (vram_budget's imma_plane_bytes) and enforced here:
// a take past the budget declines like an allocation failure. SIZE_MAX (default) =
// uncapped (tests/standalone tools); a planned budget of 0 means "no planes".
size_t g_imma_plane_budget = SIZE_MAX;
size_t g_imma_plane_used = 0;
bool g_imma_plane_budget_hit = false;

// Plane bytes for one (N, K) Q8_0 weight: the s8 plane plus the interleaved
// (alpha, beta) half pair per 32-block. Public so the planner charges the
// number this allocator will actually take.
size_t imma_q8_plane_bytes(int64_t N, int64_t K) {
    return static_cast<size_t>(N) * static_cast<size_t>(K) +
           static_cast<size_t>(N) * (static_cast<size_t>(K) / 32) * 2 * sizeof(__half);
}

namespace {
// Charge `need` against the budget. Caller holds g_imma_mtx.
bool imma_plane_budget_take(size_t need, const char* what, int64_t N, int64_t K) {
    if (g_imma_plane_budget == SIZE_MAX)
        return true;
    if (g_imma_plane_used + need <= g_imma_plane_budget)
        return true;
    if (!g_imma_plane_budget_hit) {
        g_imma_plane_budget_hit = true;
        // Shape is named in the message deliberately: planner scans the tensor list, allocator
        // keys on source pointers, and the two can disagree (unscanned tensor, re-registered
        // weight). Naming the take that overran makes it debuggable.
        IMP_LOG_WARN(
            "mmq_q8_imma: %s cache reached its planned budget (%.0f MiB used, %.0f MiB "
            "planned, %.0f MiB wanted for N=%lld K=%lld) — the remaining prefill GEMMs run "
            "the dequant path. Raise it by giving the plan more room (smaller KV pool) or "
            "set gemm.q8_imma_enabled=false to drop the cache entirely.",
            what, g_imma_plane_used / (1024.0 * 1024.0), g_imma_plane_budget / (1024.0 * 1024.0),
            need / (1024.0 * 1024.0), static_cast<long long>(N), static_cast<long long>(K));
    }
    return false;
}
}  // namespace

void mmq_q8_imma_set_plane_budget(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_imma_mtx);
    g_imma_plane_budget = bytes;
    g_imma_plane_budget_hit = false;
}

size_t mmq_q8_imma_plane_bytes_used() {
    std::lock_guard<std::mutex> lk(g_imma_mtx);
    return g_imma_plane_used;
}

bool imma_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(stream, &st) == cudaSuccess &&
           st == cudaStreamCaptureStatusActive;
}

bool imma_ensure_weight(const void* src, int N, int K, cudaStream_t stream, bool capturing) {
    // Q8_0 only: SoA planes cost 1.06x source, fine for dense Q8 models. Q4_K (esp. MoE
    // experts) reads raw blocks in-kernel instead: the plane variant duplicated all expert
    // weights and hit the 32-GB VRAM wall on Qwen3-30B (pp512 8x slower under UVM paging).
    auto it = g_imma_weights.find(src);
    if (it != g_imma_weights.end() && it->second.N == N && it->second.K == K) return true;
    if (capturing) return false;  // never allocate inside graph capture

    WeightPlanes w;
    w.N = N;
    w.K = K;
    const size_t subs = static_cast<size_t>(K) / 32;
    const size_t need = imma_q8_plane_bytes(N, K);
    if (!imma_plane_budget_take(need, "Q8_0 plane", N, K))
        return false;
    if (cudaMalloc(&w.qs, static_cast<size_t>(N) * K) != cudaSuccess) return false;
    if (cudaMalloc(&w.sc, static_cast<size_t>(N) * subs * 2 * sizeof(__half)) != cudaSuccess) {
        cudaFree(w.qs);
        return false;
    }
    g_imma_plane_used += need;
    MemAccount::instance().note("imma_q8_planes", static_cast<std::ptrdiff_t>(need));
    const int total = N * static_cast<int>(subs);
    q8_split_kernel<<<(total + 255) / 256, 256, 0, stream>>>(static_cast<const uint8_t*>(src),
                                                             w.qs, w.sc, total);
    IMP_CUDA_CHECK_LAUNCH();
    g_imma_weights[src] = w;
    return true;
}

bool imma_ensure_q6k(const void* src, size_t n_blocks, cudaStream_t stream, bool capturing) {
    auto it = g_imma_q6k.find(src);
    if (it != g_imma_q6k.end() && it->second.n_blocks == n_blocks) return true;
    if (capturing) return false;
    Q6kRepack r;
    r.n_blocks = n_blocks;
    const size_t need = n_blocks * kQ6Stride;
    if (!imma_plane_budget_take(need, "Q6_K repack", static_cast<int64_t>(n_blocks), 256))
        return false;
    if (cudaMalloc(&r.blocks, need) != cudaSuccess)
        return false;
    g_imma_plane_used += need;
    MemAccount::instance().note("imma_q8_planes", static_cast<std::ptrdiff_t>(need));
    const size_t total = n_blocks * 105;
    q6k_repack_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t*>(src), r.blocks, n_blocks);
    IMP_CUDA_CHECK_LAUNCH();
    g_imma_q6k[src] = r;
    return true;
}

// T2 (A7 step 8), closes AUDIT B13: these 3 buffers are kernel params baked into
// instantiated CUDA graphs. cudaFree+cudaMalloc made a replayed graph read freed memory
// on grow; a bump arena never frees, so a grow hands out a new slice, old one stays valid.
// Growing strands the previous slice, so exec_t2_demand charges the bound up front
// (`imma_scratch`); a grow past it is logged.
bool imma_ensure_act(int M, int K, bool capturing) {
    const size_t mk = static_cast<size_t>(M) * K;
    const size_t msubs = static_cast<size_t>(M) * (K / 32);
    const uint64_t gen = engine_arena().generation();
    if (g_imma_act.xs8 && g_imma_act.gen == gen && g_imma_act.cap_mk >= mk && g_imma_act.cap_msubs >= msubs)
        return true;
    if (g_imma_act.xs8 && g_imma_act.gen != gen)
        g_imma_act = ActScratch{};  // the arena was closed under us
    if (capturing) return false;
    if (g_imma_act.xs8) {
        // Reachable only if mmq_q8_imma_preallocate()'s bound was too small: preallocation takes
        // the charged (rows, K) up front so a staircase of intermediate takes (each stranded in
        // the bump arena) cannot happen.
        IMP_LOG_WARN("mmq_q8_imma: activation scratch regrew to M=%d K=%d (%.1f MiB) past the "
                     "preallocated bound — re-measure exec_imma_scratch_shape()",
                     M, K, (mk + msubs * 6) / (1024.0 * 1024.0));
    }
    auto xs8 = engine_arena().take_bytes(mk);
    auto xscale = engine_arena().take_bytes(msubs * sizeof(__half));
    auto xrowsum = engine_arena().take_bytes(msubs * sizeof(float));
    if (xs8.empty() || xscale.empty() || xrowsum.empty()) {
        IMP_LOG_WARN("mmq_q8_imma: activation scratch for M=%d K=%d unavailable from the T2 arena "
                     "(%.1f MiB free) — this GEMM falls back to the dequant path",
                     M, K, engine_arena().remaining() / (1024.0 * 1024.0));
        return false;
    }
    g_imma_act.xs8 = reinterpret_cast<int8_t*>(xs8.data());
    g_imma_act.xscale = reinterpret_cast<__half*>(xscale.data());
    g_imma_act.xrowsum = reinterpret_cast<float*>(xrowsum.data());
    g_imma_act.cap_mk = mk;
    g_imma_act.cap_msubs = msubs;
    g_imma_act.gen = gen;
    return true;
}

SplitKScratch g_imma_splitk;

// T2, same B13 argument as imma_ensure_act: this buffer is also a graph parameter.
// Bound is provable: caller only takes this path at M<=32, tile guard caps N*used at
// 512*kBN, so kExecImmaSplitkBytes covers every shape (see exec/workspace_sizes.h).
bool imma_ensure_splitk(size_t floats, bool capturing) {
    const uint64_t gen = engine_arena().generation();
    if (g_imma_splitk.buf && g_imma_splitk.gen == gen && g_imma_splitk.cap >= floats) return true;
    if (g_imma_splitk.buf && g_imma_splitk.gen != gen) g_imma_splitk = SplitKScratch{};
    if (capturing) return false;
    if (g_imma_splitk.buf) {
        IMP_LOG_WARN("mmq_q8_imma: split-K scratch regrew to %.1f MiB — the N*used bound in "
                     "exec/workspace_sizes.h (kExecImmaSplitkBytes) no longer holds",
                     floats * sizeof(float) / (1024.0 * 1024.0));
    }
    auto slab = engine_arena().take_bytes(floats * sizeof(float));
    if (slab.empty()) {
        g_imma_splitk.buf = nullptr;
        g_imma_splitk.cap = 0;
        return false;  // caller drops to the non-split-K path
    }
    g_imma_splitk.buf = reinterpret_cast<float*>(slab.data());
    g_imma_splitk.cap = floats;
    g_imma_splitk.gen = gen;
    return true;
}

void imma_quantize_act(const __half* x, int M, int K, cudaStream_t stream) {
    // NO memoization: workspace buffers (moe gathered, layer activations) are REUSED across
    // layers with the same pointer; a (ptr,M,K) memo served layer-1 activations to every
    // later layer (PPL 31.6 -> 441k). Kernel costs ~7us; quantize unconditionally.
    const int total_warps = M * (K / 32);
    const int blocks = min(2048, (total_warps + 7) / 8);
    quantize_act_fast_kernel<<<blocks, 256, 0, stream>>>(x, M, K, g_imma_act.xs8, g_imma_act.xscale,
                                                         g_imma_act.xrowsum);
    IMP_CUDA_CHECK_LAUNCH();
}


// Takes activation triple + split-K slice ONCE at the exec_t2_demand bound (A7 step 8),
// called from Engine::init after the T2 arena opens. Without it, imma_ensure_act/
// imma_ensure_splitk climb a staircase of ever-larger takes; a bump arena has no free,
// so each intermediate one is stranded and the sum exceeds what the plan reserved.
void mmq_q8_imma_preallocate(int rows, int k) {
    if (rows <= 0 || k <= 0)
        return;
    std::lock_guard<std::mutex> lk(g_imma_mtx);
    if (!imma_ensure_act(rows, k, /*capturing=*/false)) {
        IMP_LOG_WARN("mmq_q8_imma: could not preallocate the %dx%d activation scratch — the IMMA "
                     "prefill path will fall back to dequant",
                     rows, k);
        return;
    }
    // kExecImmaSplitkBytes / sizeof(float), mirrored here so this TU stays free
    // of the exec/ headers; the bound's derivation lives in workspace_sizes.h.
    (void)imma_ensure_splitk((8ull << 20) / sizeof(float), /*capturing=*/false);
    IMP_LOG_DEBUG("mmq_q8_imma: preallocated %dx%d activation scratch (%.1f MiB)", rows, k,
                  (static_cast<size_t>(rows) * k * 19 / 16) / (1024.0 * 1024.0));
}


}  // namespace imp
