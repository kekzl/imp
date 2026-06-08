#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace imp {

class VRAMAllocator;

// ---------------------------------------------------------------------------
// LRU cache for GPU-resident expert weights.
// When MoE experts don't fit in VRAM, they reside on host (mmap/pinned).
// This cache keeps recently-used experts on GPU to avoid repeated H2D copies.
// Key = (packed_tensor_ptr, expert_index), Value = GPU slot with raw bytes.
// ---------------------------------------------------------------------------
struct ExpertCacheKey {
    const void* packed_ptr;  // pointer to packed tensor (identifies weight matrix)
    int expert_idx;
    bool operator==(const ExpertCacheKey& o) const {
        return packed_ptr == o.packed_ptr && expert_idx == o.expert_idx;
    }
};

struct ExpertCacheKeyHash {
    size_t operator()(const ExpertCacheKey& k) const {
        // Combine pointer hash and expert index
        size_t h = std::hash<const void*>{}(k.packed_ptr);
        h ^= std::hash<int>{}(k.expert_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// MoE projection enum — every layer has up to three: gate, up, down.
// Used to index the per-layer × per-projection × per-expert device-side
// lookup mirror (Phase 2 of the MoE host-offload + CUDA Graphs design,
// see docs/plans/moe_host_offload_graphs_design_2026_05_17.md §3a).
enum class ExpertProj { Gate = 0, Up = 1, Down = 2 };
inline constexpr int kExpertProjCount = 3;

// Per-layer LRU state — Phase 3 partitions the global pool so layer L's
// cache state can't be evicted by layer M's misses. Each layer has its own
// recency list + key→slot map; slot indices are layer-relative.
struct PerLayerLRU {
    std::list<int> lru_order;  // slot indices WITHIN this layer (0..slots_per_layer-1)
    using LRUIter = std::list<int>::iterator;
    std::unordered_map<ExpertCacheKey, std::pair<int, LRUIter>, ExpertCacheKeyHash> lookup;
};

// Per-layer access history ring — Phase 4 prefetch signal. Records every
// get_or_load(layer, proj, expert) hit OR miss into a fixed-size ring, so
// `prefetch_layer(layer)` can consult the most-recent (proj, expert) pairs
// even after they've been evicted from the LRU. Recency-resilient — survives
// across-token churn that the per-layer LRU's "currently cached" view doesn't.
struct PerLayerAccessRing {
    // Capacity is per-layer; sized at init so writes never reallocate.
    std::vector<std::pair<int, int>> entries;  // (proj, expert) pairs
    int head = 0;       // next write position (mod entries.size())
    int filled = 0;     // entries actually populated (clamped to capacity)
};

struct ExpertLRUCache {
    // Each slot holds one expert's raw quantized bytes on GPU. Slots are
    // partitioned per-layer (Phase 3): `slots_[layer * slots_per_layer_ +
    // slot_idx_within_layer]`. GPU pointer = `pool_ + flat_index * slot_size_`.
    struct Slot {
        void* gpu_ptr = nullptr;  // points into pool_
        ExpertCacheKey key = {};
        bool occupied = false;
        // Mirror coords — set on get_or_load, used to invalidate the device
        // lookup cell on eviction. `layer` matches the flat-index layer.
        int layer = -1;
        int proj = -1;   // ExpertProj
        int expert = -1;
    };

    void* pool_ = nullptr;     // contiguous GPU allocation for all slots
    size_t slot_size_ = 0;     // bytes per slot (max expert raw size)
    int slots_per_layer_ = 0;  // S — uniform per-layer pool depth (Phase 3)
    int n_slots_ = 0;          // = n_layers_ * slots_per_layer_ (total)
    std::vector<Slot> slots_;  // flat array, layer-major (size = n_slots_)

    // Per-layer LRU state (Phase 3). `per_layer_lru_[layer]` holds layer L's
    // recency list + key→slot map. Slot indices are layer-relative (0..S-1).
    std::vector<PerLayerLRU> per_layer_lru_;
    using LRUIter = PerLayerLRU::LRUIter;

    // Phase 4: per-layer access history ring + prefetch infrastructure.
    // - `per_layer_history_[layer]` records every get_or_load() (proj, expert)
    //   into a fixed-size ring. Even after eviction the access is remembered,
    //   so prefetch_layer() can pre-warm the cache with experts likely to
    //   recur next token.
    // - `prefetch_stream_` is a dedicated CUDA stream where prefetch H2Ds run
    //   concurrent with the engine's compute stream. Compute waits on
    //   `prefetch_done_[layer]` before dispatching layer L's reads.
    std::vector<PerLayerAccessRing> per_layer_history_;
    int history_capacity_ = 0;        // entries per layer (sized at init)
    cudaStream_t prefetch_stream_ = nullptr;
    std::vector<cudaEvent_t> prefetch_done_;  // one event per layer, signaled by prefetch_layer
    std::vector<bool> prefetch_issued_;       // per-layer flag — has prefetch_layer been called?
    int64_t prefetch_h2ds_ = 0;       // count of async H2Ds the prefetcher actually issued
    int64_t prefetch_skipped_cached_ = 0;  // ring entries already cached at prefetch time

    // Phase 3: pre-computed host source pointers for capture-safe memcpy.
    // host_expert_addrs_[layer][proj * n_experts + expert] = packed.data +
    // expert_idx * expert_raw. Populated lazily on first get_or_load per
    // cell — the value is stable for the lifetime of the model load (packed
    // tensors don't move on host). Phase 5 will use these as the fixed
    // `src` argument of cudaGraphAddMemcpyNode so the graph can replay
    // without recomputing host pointers each iteration.
    std::vector<std::vector<const void*>> host_expert_addrs_;

    // Phase 4: canonical packed_ptr per (layer, proj). The ExpertCacheKey
    // hashes on packed_ptr, so prefetch must rebuild keys that match what
    // dispatch will pass on its next get_or_load. Sized [n_layers × 3];
    // populated on first get_or_load per (layer, proj).
    std::vector<const void*> host_packed_ptrs_;

    // Phase 5: per-(layer, proj) expert byte size. The prefetch path can't
    // reuse the global slot_size_ (max across projs) because smaller
    // projections — e.g. Qwen3.6 gate @ 144 MiB vs down @ 176 MiB — would
    // overflow the pinned host region during cudaMemcpyAsync and fail with
    // "invalid argument". Populated on first get_or_load per (layer, proj)
    // when the caller passes the actual per-expert byte size.
    std::vector<size_t> host_expert_bytes_;

    int64_t hits_ = 0;
    int64_t misses_ = 0;

    VRAMAllocator* alloc_ = nullptr;

    // Device-side lookup mirror (Phase 2 → Phase 3). Sized [n_layers × 3 ×
    // n_experts] int32 cells; cell value = **layer-relative** slot_idx
    // (0..slots_per_layer-1) or -1 if not cached. Read pattern (Phase 5 will
    // do this from inside dispatch kernels):
    //   slot = d_lookup_[layer * 3 * n_experts + proj * n_experts + expert];
    //   if (slot >= 0) src = pool_ + (layer * slots_per_layer + slot) * slot_size;
    int* d_lookup_ = nullptr;
    int n_layers_ = 0;
    int n_experts_ = 0;
    bool debug_parity_ = false;
    mutable int64_t parity_checks_ok_ = 0;  // exposed for tests; bumped by const check_parity()

    // Initialize: allocate the slot pool (partitioned per-layer) + the
    // device mirror + the host source-pointer table. Returns false if GPU
    // allocation fails (cache disabled).
    bool init(size_t max_expert_raw, size_t budget_bytes, VRAMAllocator* alloc,
              int n_layers, int n_experts, bool debug_parity = false);

    // Lookup or insert an expert. Returns GPU pointer to cached expert data.
    // If cache miss: copies from host, evicts LRU entry within the layer's
    // pool if needed.
    // src_host = host pointer to this expert's raw bytes. The first call per
    // (layer, proj, expert) records src_host into host_expert_addrs_;
    // subsequent calls reuse the stored pointer (the cache key disambiguates).
    void* get_or_load(int layer, ExpertProj proj, ExpertCacheKey key,
                      const void* src_host, size_t expert_bytes, cudaStream_t stream);

    // Check if (layer, expert) is cached (no insertion). Updates LRU recency
    // within the layer.
    void* find(int layer, ExpertCacheKey key);

    void destroy();

    float hit_rate() const {
        int64_t total = hits_ + misses_;
        return total > 0 ? static_cast<float>(hits_) / total : 0.0f;
    }

    // Returns the device pointer for layer L, slot_idx_within_layer s, or
    // nullptr if the slot is not occupied. Used by Phase 3 tests and as the
    // future Phase 5 kernel-side resolve helper. Phase 5 will inline this
    // logic into the dispatch kernels.
    void* slot_ptr(int layer, int slot_idx_within_layer) const {
        if (!pool_ || slot_idx_within_layer < 0 ||
            slot_idx_within_layer >= slots_per_layer_)
            return nullptr;
        size_t flat = static_cast<size_t>(layer) * slots_per_layer_ + slot_idx_within_layer;
        return static_cast<char*>(pool_) + flat * slot_size_;
    }

    // Phase 4 — async prefetch.
    //
    // prefetch_layer(layer, top_k, expert_bytes): walk per_layer_history_[layer]
    // from the most-recent entry backward, gather up to top_k unique
    // (proj, expert) pairs that AREN'T currently cached, and kick off an
    // async H2D into a freshly-allocated slot for each on prefetch_stream_.
    // The host LRU + device mirror are updated synchronously on the host
    // side, so a follow-up get_or_load(layer, …) on the compute stream sees
    // the slot as "cached". Records prefetch_done_[layer] when the H2Ds
    // finish — the compute stream must wait on this before reading any
    // affected slot.
    //
    // expert_bytes is the H2D copy size; pass `slot_size_` if you don't
    // know it precisely (over-copy is wasted bandwidth but correct).
    // Returns the number of async H2Ds issued (0..top_k). No-op if
    // prefetch_stream_ is null or top_k <= 0.
    int prefetch_layer(int layer, int top_k, size_t expert_bytes);

    // Compute stream waits on prefetch_done_[layer]. Safe to call even if
    // prefetch_layer wasn't called for `layer` (skip on no-issue).
    void await_prefetch(int layer, cudaStream_t compute_stream);

    // Phase 2 debug helper: copy the device lookup mirror back to host and
    // assert every cell matches the host-side LRU state. Returns true on
    // match. When `debug_parity_` is enabled, get_or_load() runs this after
    // every cache mutation and bumps `parity_checks_ok_` on success / aborts
    // (via IMP_LOG_FATAL) on mismatch. Tests can call it directly.
    bool check_parity(cudaStream_t stream) const;
};

}  // namespace imp
