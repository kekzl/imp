// Next-layer expert prefetch for the device expert cache (always on). Layer L's MoE input
// through layer L+1's router predicts L+1's routing; up to `budget` predicted keys that are not
// resident get LRU victims and are copied on the gather stream behind L's own misses, so PCIe
// works through L's GEMVs and L+1's attention. Design: expert_cache_device.h.
#include "exec/expert_cache_device.h"
#include "exec/expert_cache_device_kernels.cuh"
#include "exec/executor_helpers.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <climits>

namespace imp {

namespace {

using namespace expert_cache_kernels;  // NOLINT(google-build-using-namespace): shared resolve helpers

constexpr int kTopkThreads = 256;
constexpr int kTopkPerThread = 4;    // ne <= 1024
constexpr int kPredTraceTag = 1000;  // trace record layer field: 1000 + predicted layer

// (value, index) max, ties to the lower index.
__device__ __forceinline__ void argmax_merge(float& v, int& i, float ov, int oi) {
    if (oi >= 0 && (i < 0 || ov > v || (ov == v && oi < i))) {
        v = ov;
        i = oi;
    }
}

// One block: top_k largest of logits[ne] into pred (descending), optional trace record.
__global__ void expert_predict_topk_kernel(const float* __restrict__ logits, int ne, int top_k,
                                           int32_t* __restrict__ pred, int* trace, float* trace_w,
                                           unsigned int* cursor, unsigned long long capacity, int tag) {
    __shared__ float s_v[kTopkThreads / 32];
    __shared__ int s_i[kTopkThreads / 32];
    __shared__ int s_win;
    __shared__ float s_winv[kDevExpertMaxTopK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float v[kTopkPerThread];
    for (int q = 0; q < kTopkPerThread; ++q) {
        const int e = tid + (q * kTopkThreads);
        v[q] = e < ne ? logits[e] : -FLT_MAX;
    }
    for (int k = 0; k < top_k; ++k) {
        float bv = -FLT_MAX;
        int bi = -1;
        for (int q = 0; q < kTopkPerThread; ++q) {
            const int e = tid + (q * kTopkThreads);
            if (e < ne && v[q] > -FLT_MAX)
                argmax_merge(bv, bi, v[q], e);
        }
        for (int off = 16; off > 0; off >>= 1)
            argmax_merge(bv, bi, __shfl_xor_sync(0xffffffffu, bv, off),
                         __shfl_xor_sync(0xffffffffu, bi, off));
        if (lane == 0) {
            s_v[warp] = bv;
            s_i[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < kTopkThreads / 32; ++w)
                argmax_merge(bv, bi, s_v[w], s_i[w]);
            s_win = bi;
            s_winv[k] = bv;
            pred[k] = bi;
        }
        __syncthreads();
        const int win = s_win;
        if (win >= 0 && win % kTopkThreads == tid)
            v[win / kTopkThreads] = -FLT_MAX;
    }
    if (tid != 0 || trace == nullptr)
        return;
    const unsigned int rec = 1u + static_cast<unsigned int>(top_k);
    const unsigned int off = atomicAdd(cursor, rec);
    if (static_cast<unsigned long long>(off) + rec > capacity)
        return;
    trace[off] = tag;
    const unsigned int w_off = (off / rec) * static_cast<unsigned int>(top_k);
    for (int k = 0; k < top_k; ++k) {
        trace[off + 1 + k] = pred[k];
        trace_w[w_off + k] = s_winv[k];
    }
}

// One block: entries rank-major (expert k, then gate / up / down). Predicted keys that are resident
// get restamped (no prefetch victim takes them); the first `budget` absent ones take LRU victims, are
// committed like misses and listed as pending for the layer's resolve; pf_done restarts at 0.
__global__ void expert_prefetch_resolve_kernel(DevExpertLayer L, const int32_t* __restrict__ pred, int top_k,
                                               int budget, DevMissEntry* __restrict__ misses,
                                               int* __restrict__ n_miss_out) {
    constexpr int kN = 3 * kDevExpertMaxTopK;
    __shared__ int s_key[kN];
    __shared__ int s_victim[kN];
    __shared__ int s_ord[kN];  // prefetch ordinal, -1 = resident or past the budget
    __shared__ int s_best[kResolveThreads / 32];
    __shared__ uint32_t s_bestv[kResolveThreads / 32];
    __shared__ uint32_t s_stamp[kResolveSmemSlots];
    __shared__ uint32_t s_clock;
    __shared__ int s_n;
    const int tid = threadIdx.x;
    const int n = 3 * top_k;
    if (tid == 0) {
        s_clock = ++(*L.clock);
        s_n = 0;
    }
    __syncthreads();
    for (int i = tid; i < n; i += kResolveThreads) {
        const int e = pred[i / 3];
        const int key = ((i % 3) * L.n_experts) + e;
        s_key[i] = key;
        s_ord[i] = -1;
        const bool valid = e >= 0 && e < L.n_experts;
        const int s = valid ? L.slot_of[key] : INT_MAX;  // invalid id: skipped like a resident key
        s_victim[i] = s;
        if (valid && s >= 0)
            L.stamp[s] = s_clock;
    }
    __syncthreads();
    const bool stamps_in_smem = L.slots <= kResolveSmemSlots;
    if (stamps_in_smem)
        for (int s = tid; s < L.slots; s += kResolveThreads)
            s_stamp[s] = L.stamp[s];
    __syncthreads();
    for (int i = 0; i < n && s_n < budget; ++i) {
        if (s_victim[i] >= 0)
            continue;
        const int best = lru_victim(L, s_stamp, stamps_in_smem, s_clock, s_best, s_bestv);
        if (tid == 0 && best >= 0) {
            s_victim[i] = best;
            if (stamps_in_smem)
                s_stamp[best] = s_clock;
            else
                L.stamp[best] = s_clock;
            s_ord[i] = s_n++;
        }
        __syncthreads();
    }
    for (int i = tid; i < n; i += kResolveThreads) {
        if (s_ord[i] < 0)
            continue;
        commit_miss(L, s_key[i], s_victim[i], s_ord[i], s_clock, misses);
        L.pf_slots[s_ord[i]] = s_victim[i];
    }
    __syncthreads();
    if (tid == 0) {
        *n_miss_out = s_n;
        *L.pf_count = s_n;
        *L.pf_done = 0;  // the previous round of this layer finished: its gather precedes this fork
        atomicAdd(&L.stats[2], static_cast<unsigned long long>(s_n));
    }
}

constexpr int kPfCopyUnroll = 4;  // int4 loads in flight per thread (host and pool never alias)

__device__ __forceinline__ void pf_copy(const int4* __restrict__ s, int4* __restrict__ d, size_t n16,
                                        size_t t0, size_t stride) {
    for (size_t j = t0; j < n16; j += kPfCopyUnroll * stride) {
        int4 v[kPfCopyUnroll];
#pragma unroll
        for (int u = 0; u < kPfCopyUnroll; ++u)
            if (j + (u * stride) < n16)
                v[u] = s[j + (u * stride)];
#pragma unroll
        for (int u = 0; u < kPfCopyUnroll; ++u)
            if (j + (u * stride) < n16)
                d[j + (u * stride)] = v[u];
    }
}

// grid (blocks per key, budget): block y copies pending key y (non-tiered layers only: no write-back),
// then counts itself in done once its stores are fenced.
__global__ void expert_prefetch_gather_kernel(const DevMissEntry* __restrict__ misses,
                                              const int* __restrict__ n_miss, DevGatherBytes sz,
                                              uint32_t* __restrict__ done) {
    const int i = blockIdx.y;
    if (i >= *n_miss)
        return;
    const DevMissEntry m = misses[i];
    const int proj = m.key / sz.n_experts;
    const auto pick = [proj](const size_t (&v)[3]) { return proj == 0 ? v[0] : (proj == 1 ? v[1] : v[2]); };
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t t0 = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    pf_copy(reinterpret_cast<const int4*>(m.packed), reinterpret_cast<int4*>(m.dst), pick(sz.packed) / 16, t0,
            stride);
    if (m.ms != nullptr)
        pf_copy(reinterpret_cast<const int4*>(m.ms), reinterpret_cast<int4*>(m.dst + pick(sz.ms_off)),
                pick(sz.ms) / 16, t0, stride);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(done, 1u);
}

}  // namespace

bool DeviceExpertCache::enable_prefetch(int budget, int blocks_per_key) {
    free_prefetch_();
    if (budget <= 0 || layers_ready_ == 0 || n_experts_ > kTopkThreads * kTopkPerThread)
        return false;
    budget = std::min(budget, kDevExpertPrefetchMax);
    const int n_layers = static_cast<int>(layers_.size());
    auto al = [](size_t v) { return (v + 255) / 256 * 256; };
    const size_t slots_b = al(static_cast<size_t>(n_layers) * kDevExpertPrefetchMax * sizeof(int32_t));
    const size_t count_b = al(static_cast<size_t>(n_layers) * sizeof(int32_t));
    const size_t miss_b = al(static_cast<size_t>(2) * kDevExpertPrefetchMax * sizeof(DevMissEntry));
    const size_t logits_b = al(static_cast<size_t>(n_experts_) * sizeof(float));
    const size_t pred_b = al(kDevExpertMaxTopK * sizeof(int32_t));
    const size_t bytes = slots_b + count_b + miss_b + 256 + logits_b + pred_b + count_b;  // last: pf_done
    pf_tables_ = vram_alloc(alloc_, bytes, "expert_prefetch");
    if (!pf_tables_)
        return false;
    IMP_CUDA_CHECK_LOG(cudaMemset(pf_tables_, 0, bytes));
    char* base = static_cast<char*>(pf_tables_);
    auto* slots = reinterpret_cast<int32_t*>(base);
    auto* counts = reinterpret_cast<int32_t*>(base + slots_b);
    pf_misses_ = reinterpret_cast<DevMissEntry*>(base + slots_b + count_b);
    pf_n_miss_ = reinterpret_cast<int*>(base + slots_b + count_b + miss_b);
    pf_logits_ = reinterpret_cast<float*>(base + slots_b + count_b + miss_b + 256);
    pf_pred_ = reinterpret_cast<int32_t*>(base + slots_b + count_b + miss_b + 256 + logits_b);
    auto* done = reinterpret_cast<uint32_t*>(base + slots_b + count_b + miss_b + 256 + logits_b + pred_b);
    for (int l = 0; l < n_layers; ++l) {
        if (!layer_ready(l) || layer_tiered(l))
            continue;
        layers_[l].pf_slots = slots + (static_cast<size_t>(l) * kDevExpertPrefetchMax);
        layers_[l].pf_count = counts + l;
        layers_[l].pf_done = done + l;
        layers_[l].pf_blocks = std::clamp(blocks_per_key, 1, kDevExpertGatherBlocks * 4);
    }
    pf_budget_ = budget;
    IMP_LOG_INFO(
        "expert prefetch: up to %d keys per layer predicted from the previous layer's router input, "
        "%d copy blocks per key",
        budget, std::clamp(blocks_per_key, 1, kDevExpertGatherBlocks * 4));
    return true;
}

void DeviceExpertCache::free_prefetch_() {
    if (pf_tables_)
        vram_free(alloc_, pf_tables_);
    pf_tables_ = nullptr;
    pf_misses_ = nullptr;
    pf_n_miss_ = nullptr;
    pf_logits_ = nullptr;
    pf_pred_ = nullptr;
    pf_budget_ = 0;
    for (DevExpertLayer& L : layers_) {
        L.pf_slots = nullptr;
        L.pf_count = nullptr;
        L.pf_done = nullptr;
    }
}

void DeviceExpertCache::predict_topk(const float* logits, int ne, int top_k, int layer, PredTrace trace,
                                     cudaStream_t stream) {
    expert_predict_topk_kernel<<<1, kTopkThreads, 0, stream>>>(logits, ne, top_k, pf_pred_, trace.trace,
                                                               trace.weights, trace.cursor, trace.capacity,
                                                               kPredTraceTag + layer);
    IMP_CUDA_CHECK_LAUNCH();
}

void DeviceExpertCache::prefetch_resolve(int layer, int top_k, cudaStream_t stream) {
    expert_prefetch_resolve_kernel<<<1, kResolveThreads, 0, stream>>>(layers_[layer], pf_pred_, top_k,
                                                                      pf_budget_, pf_misses_of_(layer),
                                                                      pf_n_miss_of_(layer));
    IMP_CUDA_CHECK_LAUNCH();
}

void DeviceExpertCache::prefetch_gather(int layer, cudaStream_t stream) {
    const DevExpertLayer& L = layers_[layer];
    const dim3 grid(L.pf_blocks, pf_budget_);
    expert_prefetch_gather_kernel<<<grid, 256, 0, stream>>>(pf_misses_of_(layer), pf_n_miss_of_(layer),
                                                            gather_bytes_(layer), L.pf_done);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
