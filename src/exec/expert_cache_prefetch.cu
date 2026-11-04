// Next-layer expert prefetch for the device expert cache (moe.expert_prefetch). Layer L's MoE input
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
// committed like misses and listed as pending for the layer's resolve.
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
        atomicAdd(&L.stats[2], static_cast<unsigned long long>(s_n));
    }
}

}  // namespace

bool DeviceExpertCache::enable_prefetch(int budget) {
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
    const size_t bytes = slots_b + count_b + miss_b + 256 + logits_b +
                         al(kDevExpertMaxTopK * sizeof(int32_t));
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
    for (int l = 0; l < n_layers; ++l) {
        if (!layer_ready(l) || layer_tiered(l))
            continue;
        layers_[l].pf_slots = slots + (static_cast<size_t>(l) * kDevExpertPrefetchMax);
        layers_[l].pf_count = counts + l;
    }
    pf_budget_ = budget;
    IMP_LOG_INFO(
        "moe.expert_prefetch: up to %d keys per layer predicted from the previous layer's router input",
        budget);
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

}  // namespace imp
