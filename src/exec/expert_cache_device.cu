// Device-driven expert cache: resolve kernel (lookup + LRU eviction + slot/scale writes)
// and gather kernel (zero-copy from mapped pinned host memory). Design: expert_cache_device.h.
#include "exec/expert_cache_device.h"
#include "exec/executor_helpers.h"
#include "exec/nvfp4_expert_offload.h"
#include "model/model.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstring>

namespace imp {

namespace {

constexpr int kResolveThreads = 256;
constexpr int kMaxEntries = 3 * kDevExpertMaxTopK;

// One block. Phase 1: every routed (proj, expert) looks itself up; hits restamp their slot.
// Phase 2: misses, in order, each take the least recently stamped slot not used this step.
__global__ void expert_cache_resolve_kernel(DevExpertLayer L, const int32_t* __restrict__ expert_indices,
                                            int top_k, int32_t* __restrict__ slot_idx_out,
                                            DevMissEntry* __restrict__ misses,
                                            int* __restrict__ n_miss_out) {
    __shared__ int s_key[kMaxEntries];
    __shared__ int s_miss[kMaxEntries];
    __shared__ int s_best[kResolveThreads];
    __shared__ uint32_t s_bestv[kResolveThreads];
    __shared__ uint32_t s_clock;
    __shared__ int s_nmiss;

    const int tid = threadIdx.x;
    const int n = 3 * top_k;
    if (tid == 0) {
        s_clock = ++(*L.clock);
        s_nmiss = 0;
    }
    __syncthreads();

    if (tid < n) {
        const int proj = tid / top_k;
        const int e = expert_indices[tid - proj * top_k];
        const int key = proj * L.n_experts + e;
        s_key[tid] = key;
        const int s = L.slot_of[key];
        if (s >= 0) {
            L.stamp[s] = s_clock;
            slot_idx_out[tid] = s;
            s_miss[tid] = 0;
        } else {
            s_miss[tid] = 1;
        }
    }
    __syncthreads();

    for (int i = 0; i < n; ++i) {
        if (!s_miss[i])
            continue;
        // argmin stamp over the layer's slots, excluding slots stamped this step.
        int best = -1;
        uint32_t bestv = UINT_MAX;
        for (int s = tid; s < L.slots; s += kResolveThreads) {
            const uint32_t v = L.stamp[s];
            if (v != s_clock && v < bestv) {
                bestv = v;
                best = s;
            }
        }
        s_best[tid] = best;
        s_bestv[tid] = bestv;
        __syncthreads();
        // Tree-reduce the per-thread argmins. This was a serial scan of all kResolveThreads
        // entries on thread 0, once per miss: 30 x 256 single-threaded iterations per layer,
        // 1.10 ms per token over 48 layers. Ties go to the lower slot index; any slot holding
        // the minimum stamp is an LRU victim, so the pick stays correct either way.
        for (int off = kResolveThreads / 2; off > 0; off >>= 1) {
            if (tid < off) {
                const int o = tid + off;
                const bool take = s_best[o] >= 0 &&
                                  (s_best[tid] < 0 || s_bestv[o] < s_bestv[tid] ||
                                   (s_bestv[o] == s_bestv[tid] && s_best[o] < s_best[tid]));
                if (take) {
                    s_bestv[tid] = s_bestv[o];
                    s_best[tid] = s_best[o];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int victim = s_best[0];
            const int key = s_key[i];
            if (victim >= 0) {
                const int old = L.key_of_slot[victim];
                if (old >= 0)
                    L.slot_of[old] = -1;
                L.slot_of[key] = victim;
                L.key_of_slot[victim] = key;
                L.stamp[victim] = s_clock;
                L.slot_scales[victim] = L.src[key].scale;
                misses[s_nmiss] = DevMissEntry{L.src[key].packed, L.src[key].ms,
                                               L.pool + static_cast<size_t>(victim) * L.slot_bytes};
                ++s_nmiss;
            }
            // victim < 0 cannot happen: slots >= 3 * top_k is checked at init.
            slot_idx_out[i] = victim;
        }
        __syncthreads();
    }
    if (tid == 0) {
        *n_miss_out = s_nmiss;
        atomicAdd(&L.stats[0], static_cast<unsigned long long>(n - s_nmiss));
        atomicAdd(&L.stats[1], static_cast<unsigned long long>(s_nmiss));
    }
}

// grid (blocks_per_expert, n_experts, 3): block (x, e, p) copies its share of expert e's
// projection p into the stage buffer when the routing touched e (prefill staging).
__global__ void expert_stage_touched_kernel(const DevExpertSrc* __restrict__ src,
                                            const int32_t* __restrict__ expert_offsets,
                                            char* __restrict__ stage_buf, size_t proj_bytes,
                                            size_t pb, size_t mb, int n_experts) {
    const int e = blockIdx.y, p = blockIdx.z;
    if (expert_offsets[e + 1] == expert_offsets[e])
        return;
    const DevExpertSrc s = src[p * n_experts + e];
    if (!s.packed || !s.ms)
        return;
    char* base = stage_buf + static_cast<size_t>(p) * proj_bytes;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t t0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    {
        const int4* sp = reinterpret_cast<const int4*>(s.packed);
        int4* dp = reinterpret_cast<int4*>(base + static_cast<size_t>(e) * pb);
        for (size_t j = t0; j < pb / 16; j += stride)
            dp[j] = sp[j];
    }
    {
        const int4* sm = reinterpret_cast<const int4*>(s.ms);
        int4* dm = reinterpret_cast<int4*>(base + static_cast<size_t>(n_experts) * pb +
                                          static_cast<size_t>(e) * mb);
        for (size_t j = t0; j < mb / 16; j += stride)
            dm[j] = sm[j];
    }
}

// Expert block [e0, e0 + gridDim.y) of up to two projections: grid (blocks_per_expert, n, slots),
// slot z holds projection proj_of_slot[z] at block-local index y (chunked prefill staging).
__global__ void expert_stage_touched_range_kernel(const DevExpertSrc* __restrict__ src,
                                                  const int32_t* __restrict__ expert_offsets,
                                                  char* __restrict__ stage_buf, size_t proj_bytes,
                                                  size_t pb, size_t mb, int n_experts, int e0,
                                                  int2 proj_of_slot) {
    const int i = blockIdx.y, e = e0 + i, z = blockIdx.z;
    const int p = z == 0 ? proj_of_slot.x : proj_of_slot.y;
    if (e >= n_experts || expert_offsets[e + 1] == expert_offsets[e])
        return;
    const DevExpertSrc s = src[p * n_experts + e];
    if (!s.packed || !s.ms)
        return;
    char* base = stage_buf + static_cast<size_t>(z) * proj_bytes;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t t0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int4* sp = reinterpret_cast<const int4*>(s.packed);
    int4* dp = reinterpret_cast<int4*>(base + static_cast<size_t>(i) * pb);
    for (size_t j = t0; j < pb / 16; j += stride)
        dp[j] = sp[j];
    const int4* sm = reinterpret_cast<const int4*>(s.ms);
    int4* dm = reinterpret_cast<int4*>(base + static_cast<size_t>(gridDim.y) * pb + static_cast<size_t>(i) * mb);
    for (size_t j = t0; j < mb / 16; j += stride)
        dm[j] = sm[j];
}

// grid (blocks_per_copy, kMaxEntries): block y copies miss y, packed block then micro-scales.
__global__ void expert_cache_gather_kernel(const DevMissEntry* __restrict__ misses,
                                           const int* __restrict__ n_miss, size_t packed_bytes,
                                           size_t ms_bytes, size_t ms_off) {
    const int i = blockIdx.y;
    if (i >= *n_miss)
        return;
    const DevMissEntry m = misses[i];
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t t0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    {
        const int4* s = reinterpret_cast<const int4*>(m.packed);
        int4* d = reinterpret_cast<int4*>(m.dst);
        for (size_t j = t0; j < packed_bytes / 16; j += stride)
            d[j] = s[j];
    }
    {
        const int4* s = reinterpret_cast<const int4*>(m.ms);
        int4* d = reinterpret_cast<int4*>(m.dst + ms_off);
        for (size_t j = t0; j < ms_bytes / 16; j += stride)
            d[j] = s[j];
    }
}

// Device view of a pointer inside one of the model's mapped pinned slabs, or null.
const char* device_view(const Model& model, const void* host_ptr) {
    const char* p = static_cast<const char*>(host_ptr);
    for (const PinnedBuffer& pin : model.host_pinned_allocs()) {
        const char* b = static_cast<const char*>(pin.data());
        if (!b || !pin.device() || p < b || p >= b + pin.bytes())
            continue;
        return static_cast<const char*>(pin.device()) + (p - b);
    }
    return nullptr;
}

}  // namespace

bool DeviceExpertCache::init(const Model& model, ExpertLRUCache& cache, VRAMAllocator* alloc,
                             int top_k) {
    destroy();
    if (!cache.pool_ || !cache.nvfp4_slots_ || !cache.d_slot_scales_ || cache.n_experts_ <= 0)
        return false;
    if (top_k <= 0 || top_k > kDevExpertMaxTopK || cache.slots_per_layer_ < 3 * top_k)
        return false;
    cache_ = &cache;
    alloc_ = alloc;
    const int n_layers = cache.n_layers_;
    const int ne = cache.n_experts_;
    const int slots = cache.slots_per_layer_;

    // Host-side build of the source table, one entry per (layer, proj, expert).
    std::vector<DevExpertSrc> src(static_cast<size_t>(n_layers) * 3 * ne, DevExpertSrc{});
    std::vector<char> ready(n_layers, 0);
    ms_off_.assign(static_cast<size_t>(n_layers) * 3, 0);
    packed_bytes_.assign(static_cast<size_t>(n_layers) * 3, 0);
    ms_bytes_.assign(static_cast<size_t>(n_layers) * 3, 0);
    for (int l = 0; l < n_layers; ++l) {
        const TransformerLayer& ly = model.layer(l);
        if (ly.expert_w_up.empty() || !ly.expert_w_up[0].data || ly.expert_w_up[0].on_device)
            continue;
        const std::vector<Tensor>* projs[3] = {&ly.expert_w_gate, &ly.expert_w_up, &ly.expert_w_down};
        bool ok = true;
        for (int p = 0; p < 3 && ok; ++p) {
            const std::vector<Tensor>& ex = *projs[p];
            if (p == 0 && (ex.empty() || !ex[0].data))
                continue;  // non-gated model
            if (!nvfp4_host_experts_servable(ex) || static_cast<int>(ex.size()) != ne) {
                ok = false;
                break;
            }
            const auto layout = nvfp4_slot_layout(ex[0].shape[0], ex[0].shape[1] * 2);
            if (layout.slot_bytes() == 0 || layout.slot_bytes() > cache.slot_size_ ||
                layout.packed_bytes % 16 != 0 || layout.ms_bytes % 16 != 0 ||
                layout.packed_off() % 16 != 0) {
                ok = false;
                break;
            }
            ms_off_[static_cast<size_t>(l) * 3 + p] = layout.packed_off();
            packed_bytes_[static_cast<size_t>(l) * 3 + p] = layout.packed_bytes;
            ms_bytes_[static_cast<size_t>(l) * 3 + p] = layout.ms_bytes;
            for (int e = 0; e < ne; ++e) {
                const char* dp = device_view(model, ex[e].data);
                const char* dm = device_view(model, ex[e].scales);
                if (!dp || !dm) {
                    ok = false;  // not in a mapped slab: the host path keeps this layer
                    break;
                }
                src[(static_cast<size_t>(l) * 3 + p) * ne + e] = DevExpertSrc{dp, dm, ex[e].tensor_scale};
            }
        }
        // One miss list, one size triple per layer: gate, up and down must share the slot
        // layout (N x K and K x N have the same byte counts; a gated model always does).
        const size_t b = static_cast<size_t>(l) * 3;
        for (int p = 0; p < 3 && ok; ++p) {
            if (packed_bytes_[b + p] == 0)
                continue;  // absent gate projection
            ok = packed_bytes_[b + p] == packed_bytes_[b + 1] && ms_bytes_[b + p] == ms_bytes_[b + 1] &&
                 ms_off_[b + p] == ms_off_[b + 1];
        }
        ready[l] = ok ? 1 : 0;
    }
    layers_ready_ = 0;
    for (int l = 0; l < n_layers; ++l)
        layers_ready_ += ready[l];
    if (layers_ready_ == 0) {
        destroy();
        return false;
    }

    // One device allocation: src table, slot_of, key_of_slot, stamp, clock, stats, misses.
    const size_t src_b = src.size() * sizeof(DevExpertSrc);
    const size_t slot_of_b = static_cast<size_t>(n_layers) * 3 * ne * sizeof(int32_t);
    const size_t key_b = static_cast<size_t>(n_layers) * slots * sizeof(int32_t);
    const size_t stamp_b = static_cast<size_t>(n_layers) * slots * sizeof(uint32_t);
    const size_t clock_b = static_cast<size_t>(n_layers) * sizeof(uint32_t);
    const size_t stats_b = 2 * sizeof(unsigned long long);
    const size_t miss_b = static_cast<size_t>(kMaxEntries) * sizeof(DevMissEntry) + 256;
    auto al = [](size_t v) { return (v + 255) / 256 * 256; };
    tables_bytes_ = al(src_b) + al(slot_of_b) + al(key_b) + al(stamp_b) + al(clock_b) + al(stats_b) +
                    al(miss_b);
    tables_ = vram_alloc(alloc_, tables_bytes_, "expert_cache_dev");
    if (!tables_) {
        destroy();
        return false;
    }
    char* base = static_cast<char*>(tables_);
    DevExpertSrc* d_src = reinterpret_cast<DevExpertSrc*>(base);
    base += al(src_b);
    int32_t* d_slot_of = reinterpret_cast<int32_t*>(base);
    base += al(slot_of_b);
    int32_t* d_key = reinterpret_cast<int32_t*>(base);
    base += al(key_b);
    uint32_t* d_stamp = reinterpret_cast<uint32_t*>(base);
    base += al(stamp_b);
    uint32_t* d_clock = reinterpret_cast<uint32_t*>(base);
    base += al(clock_b);
    d_stats_ = reinterpret_cast<unsigned long long*>(base);
    base += al(stats_b);
    d_misses_ = reinterpret_cast<DevMissEntry*>(base);
    d_n_miss_ = reinterpret_cast<int*>(base + static_cast<size_t>(kMaxEntries) * sizeof(DevMissEntry));

    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_src, src.data(), src_b, cudaMemcpyHostToDevice));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_slot_of, 0xFF, slot_of_b));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_key, 0xFF, key_b));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_stamp, 0, stamp_b));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_clock, 0, clock_b));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_stats_, 0, stats_b));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_n_miss_, 0, sizeof(int)));

    layers_.assign(n_layers, DevExpertLayer{});
    for (int l = 0; l < n_layers; ++l) {
        if (!ready[l])
            continue;
        DevExpertLayer& L = layers_[l];
        L.src = d_src + static_cast<size_t>(l) * 3 * ne;
        L.slot_of = d_slot_of + static_cast<size_t>(l) * 3 * ne;
        L.key_of_slot = d_key + static_cast<size_t>(l) * slots;
        L.stamp = d_stamp + static_cast<size_t>(l) * slots;
        L.clock = d_clock + l;
        L.stats = d_stats_;
        L.pool = static_cast<char*>(cache.pool_) +
                 static_cast<size_t>(l) * slots * cache.slot_size_;
        L.slot_scales = cache.layer_slot_scales(l);
        L.slot_bytes = cache.slot_size_;
        L.n_experts = ne;
        L.slots = slots;
    }
    seen_host_generation_ = cache.host_generation_;
    IMP_LOG_INFO("Device expert cache: %d/%d MoE layer(s) resolved and staged on the device "
                 "(%d slots/layer, tables %.1f MiB, zero-copy gather from mapped pinned slabs)",
                 layers_ready_, n_layers, slots, tables_bytes_ / (1024.0 * 1024.0));
    return true;
}

void DeviceExpertCache::invalidate_(cudaStream_t stream) {
    for (const DevExpertLayer& L : layers_) {
        if (!L.src)
            continue;
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(L.slot_of, 0xFF, static_cast<size_t>(3) * L.n_experts * sizeof(int32_t), stream));
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(L.key_of_slot, 0xFF, static_cast<size_t>(L.slots) * sizeof(int32_t), stream));
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(L.stamp, 0, static_cast<size_t>(L.slots) * sizeof(uint32_t), stream));
    }
}

void DeviceExpertCache::take_over(cudaStream_t stream) {
    if (!cache_ || layers_ready_ == 0)
        return;
    if (cache_->host_generation_ != seen_host_generation_) {
        // The host path filled slots since we last looked: our tables name occupants that
        // are gone. Forget everything; the pool is ours again from here.
        invalidate_(stream);
        seen_host_generation_ = cache_->host_generation_;
    }
    cache_->device_dirty_ = true;
}

bool DeviceExpertCache::stage_touched(int layer, const int32_t* expert_offsets, char* stage_buf,
                                      size_t proj_bytes, size_t pb, size_t mb,
                                      cudaStream_t stream) {
    if (!layer_ready(layer) || !expert_offsets || !stage_buf || pb % 16 != 0 || mb % 16 != 0)
        return false;
    const DevExpertLayer& L = layers_[layer];
    for (int p = 0; p < 3; ++p) {
        const size_t i = static_cast<size_t>(layer) * 3 + p;
        if (packed_bytes_[i] != pb || ms_bytes_[i] != mb)
            return false;
    }
    if (static_cast<size_t>(L.n_experts) * (pb + mb) > proj_bytes)
        return false;
    const dim3 grid(16, L.n_experts, 3);
    expert_stage_touched_kernel<<<grid, 256, 0, stream>>>(L.src, expert_offsets, stage_buf,
                                                          proj_bytes, pb, mb, L.n_experts);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

bool DeviceExpertCache::stage_touched_range(int layer, const int32_t* expert_offsets, char* stage_buf,
                                            size_t proj_bytes, size_t pb, size_t mb, int e0, int n,
                                            int proj0, int proj1, cudaStream_t stream) {
    if (!layer_ready(layer) || !expert_offsets || !stage_buf || pb % 16 != 0 || mb % 16 != 0 || n <= 0)
        return false;
    const DevExpertLayer& L = layers_[layer];
    for (int p : {proj0, proj1}) {
        if (p < 0)
            continue;
        const size_t i = static_cast<size_t>(layer) * 3 + p;
        if (packed_bytes_[i] != pb || ms_bytes_[i] != mb)
            return false;
    }
    if (static_cast<size_t>(n) * (pb + mb) > proj_bytes || e0 < 0 || e0 >= L.n_experts)
        return false;
    const dim3 grid(16, n, proj1 < 0 ? 1 : 2);
    expert_stage_touched_range_kernel<<<grid, 256, 0, stream>>>(L.src, expert_offsets, stage_buf, proj_bytes, pb,
                                                                mb, L.n_experts, e0, make_int2(proj0, proj1));
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

void DeviceExpertCache::resolve_and_stage(int layer, const int32_t* expert_indices, int top_k,
                                          int32_t* slot_idx_out, cudaStream_t stream) {
    take_over(stream);
    const DevExpertLayer& L = layers_[layer];
    expert_cache_resolve_kernel<<<1, kResolveThreads, 0, stream>>>(L, expert_indices, top_k,
                                                                    slot_idx_out, d_misses_, d_n_miss_);
    IMP_CUDA_CHECK_LAUNCH();
    // gate, up and down share one slot layout per model (the pool was sized for the largest);
    // the gather takes the up projection's sizes, identical for the three in every NVFP4 MoE
    // seen so far (N x K and K x N have the same byte counts).
    const size_t i = static_cast<size_t>(layer) * 3 + 1;
    const dim3 grid(16, kMaxEntries);
    expert_cache_gather_kernel<<<grid, 256, 0, stream>>>(d_misses_, d_n_miss_, packed_bytes_[i],
                                                         ms_bytes_[i], ms_off_[i]);
    IMP_CUDA_CHECK_LAUNCH();
}

void DeviceExpertCache::destroy() {
    if (d_stats_) {
        unsigned long long h[2] = {0, 0};
        if (cudaMemcpy(h, d_stats_, sizeof(h), cudaMemcpyDeviceToHost) == cudaSuccess &&
            h[0] + h[1] > 0)
            IMP_LOG_INFO("Device expert cache stats: %llu hits, %llu misses (%.1f%% hit rate)", h[0],
                         h[1], 100.0 * static_cast<double>(h[0]) / static_cast<double>(h[0] + h[1]));
    }
    if (tables_)
        vram_free(alloc_, tables_);
    tables_ = nullptr;
    tables_bytes_ = 0;
    d_misses_ = nullptr;
    d_n_miss_ = nullptr;
    d_stats_ = nullptr;
    layers_.clear();
    ms_off_.clear();
    packed_bytes_.clear();
    ms_bytes_.clear();
    layers_ready_ = 0;
    cache_ = nullptr;
}

}  // namespace imp
