#include "executor.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace imp {

bool ExpertLRUCache::init(size_t max_expert_raw, size_t budget_bytes, VRAMAllocator* alloc,
                          int n_layers, int n_experts, bool debug_parity) {
    if (max_expert_raw == 0 || budget_bytes == 0)
        return false;

    alloc_ = alloc;
    slot_size_ = max_expert_raw;

    // Phase 3: partition the pool per-layer. When n_layers == 0 (tests that
    // don't care about the per-layer split or callers that haven't migrated
    // yet) fall back to a single virtual layer holding the whole pool, so
    // get_or_load(layer=0, …) keeps working.
    n_layers_ = (n_layers > 0) ? n_layers : 1;
    n_experts_ = std::max(0, n_experts);
    debug_parity_ = debug_parity;
    parity_checks_ok_ = 0;

    // slots_per_layer = floor(budget / (slot_size × n_layers)). We need ≥ 1
    // slot per layer (else a single miss in any layer can't be cached) and
    // ideally ≥ 2 total so an eviction-aware test isn't trivial.
    size_t per_layer_budget = budget_bytes / static_cast<size_t>(n_layers_);
    slots_per_layer_ = static_cast<int>(per_layer_budget / slot_size_);
    if (slots_per_layer_ < 1) {
        IMP_LOG_WARN(
            "Expert LRU cache: budget too small for even 1 slot per layer "
            "(need %zu bytes/slot × %d layers, budget %zu bytes)",
            slot_size_, n_layers_, budget_bytes);
        return false;
    }
    n_slots_ = n_layers_ * slots_per_layer_;
    if (n_slots_ < 2) {
        IMP_LOG_WARN(
            "Expert LRU cache: total slot count too small (%d slots from %d layers × %d per-layer)",
            n_slots_, n_layers_, slots_per_layer_);
        return false;
    }

    size_t total = static_cast<size_t>(n_slots_) * slot_size_;
    if (alloc_) {
        pool_ = alloc_->allocate(total, "expert_cache");
    } else {
        cudaError_t err = cudaMalloc(&pool_, total);
        if (err != cudaSuccess)
            pool_ = nullptr;
    }
    if (!pool_) {
        IMP_LOG_WARN("Expert LRU cache: allocation failed for %zu bytes (%d slots)", total, n_slots_);
        n_slots_ = 0;
        slots_per_layer_ = 0;
        return false;
    }

    slots_.resize(n_slots_);
    for (int i = 0; i < n_slots_; i++) {
        slots_[i].gpu_ptr = static_cast<char*>(pool_) + static_cast<size_t>(i) * slot_size_;
        slots_[i].layer = i / slots_per_layer_;
    }

    per_layer_lru_.assign(n_layers_, PerLayerLRU{});
    for (auto& plru : per_layer_lru_)
        plru.lookup.reserve(slots_per_layer_ * 2);

    hits_ = 0;
    misses_ = 0;

    // Device-side lookup mirror. Cell value is **layer-relative** slot_idx
    // (0..slots_per_layer_-1) or -1. n_experts == 0 means caller does not
    // want the mirror — skip the alloc entirely.
    if (n_experts_ > 0) {
        size_t cells = static_cast<size_t>(n_layers_) * kExpertProjCount * n_experts_;
        size_t bytes = cells * sizeof(int);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_lookup_), bytes);
        if (err != cudaSuccess || !d_lookup_) {
            IMP_LOG_WARN(
                "Expert LRU cache: device-side mirror alloc failed (%zu bytes, err=%s) — "
                "disabling mirror, host-side LRU still functional.",
                bytes, cudaGetErrorString(err));
            d_lookup_ = nullptr;
        } else {
            // (int32_t)-1 = 0xFFFFFFFF — memset 0xFF gives -1 in every cell.
            IMP_CUDA_CHECK_LOG(cudaMemset(d_lookup_, 0xFF, bytes));
        }

        // Host source-pointer table: [layer][proj * n_experts + expert].
        // Lazy-populated by get_or_load() — the value is stable for the
        // model's lifetime once stamped.
        host_expert_addrs_.assign(n_layers_,
                                  std::vector<const void*>(static_cast<size_t>(kExpertProjCount) *
                                                              n_experts_,
                                                          nullptr));
    }

    IMP_LOG_INFO("Expert LRU cache: %d layers × %d slots × %.2f MiB = %.2f MiB GPU memory%s%s",
                 n_layers_, slots_per_layer_, slot_size_ / (1024.0 * 1024.0),
                 total / (1024.0 * 1024.0), d_lookup_ ? " (+ device mirror)" : "",
                 debug_parity_ ? " [parity-check on]" : "");
    return true;
}

namespace {

inline int flat_slot(int layer, int slot_in_layer, int slots_per_layer) {
    return layer * slots_per_layer + slot_in_layer;
}

// Write a single int32 cell to the device-side lookup mirror.
// Async on the supplied stream — Phase 3 mirror is write-only from the
// engine's perspective; ordering against subsequent kernels is whatever
// the stream provides.
inline void write_lookup_cell(int* d_lookup, int n_experts, int layer, int proj, int expert,
                              int slot_idx_in_layer, cudaStream_t stream) {
    if (!d_lookup)
        return;
    if (layer < 0 || proj < 0 || expert < 0)
        return;
    size_t off = (static_cast<size_t>(layer) * kExpertProjCount + proj) * n_experts + expert;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_lookup + off, &slot_idx_in_layer, sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
}

}  // namespace

void* ExpertLRUCache::find(int layer, ExpertCacheKey key) {
    if (layer < 0 || layer >= n_layers_)
        return nullptr;
    auto& plru = per_layer_lru_[layer];
    auto it = plru.lookup.find(key);
    if (it == plru.lookup.end())
        return nullptr;

    // Move to front (most recently used) within this layer's recency list.
    auto& [slot_in_layer, lru_it] = it->second;
    plru.lru_order.erase(lru_it);
    plru.lru_order.push_front(slot_in_layer);
    it->second.second = plru.lru_order.begin();
    hits_++;
    return slots_[flat_slot(layer, slot_in_layer, slots_per_layer_)].gpu_ptr;
}

void* ExpertLRUCache::get_or_load(int layer, ExpertProj proj, ExpertCacheKey key,
                                  const void* src_host, size_t expert_bytes,
                                  cudaStream_t stream) {
    const int proj_idx = static_cast<int>(proj);
    if (layer < 0 || layer >= n_layers_) {
        IMP_LOG_ERROR("ExpertLRUCache::get_or_load: layer %d out of range [0, %d)", layer,
                      n_layers_);
        return nullptr;
    }
    auto& plru = per_layer_lru_[layer];

    // Stamp the host source pointer for capture-safe memcpy (Phase 5).
    // Skipped if either the mirror is disabled or n_experts == 0 (tests).
    if (!host_expert_addrs_.empty() && n_experts_ > 0 && key.expert_idx >= 0 &&
        key.expert_idx < n_experts_) {
        auto& table = host_expert_addrs_[layer];
        size_t off = static_cast<size_t>(proj_idx) * n_experts_ + key.expert_idx;
        if (off < table.size())
            table[off] = src_host;
    }

    // Check cache hit — find() handles per-layer LRU front-update. Hits
    // don't change which slot holds (layer, proj, expert), so the device
    // mirror cell remains correct.
    void* cached = find(layer, key);
    if (cached) {
        if (debug_parity_)
            check_parity(stream);
        return cached;
    }

    misses_++;

    // Find a slot WITHIN this layer's pool: use an unoccupied one, or evict
    // the layer's LRU entry.
    int slot_in_layer = -1;
    if (static_cast<int>(plru.lookup.size()) < slots_per_layer_) {
        for (int s = 0; s < slots_per_layer_; ++s) {
            if (!slots_[flat_slot(layer, s, slots_per_layer_)].occupied) {
                slot_in_layer = s;
                break;
            }
        }
    }
    if (slot_in_layer < 0) {
        // Evict layer-LRU (back of this layer's list).
        slot_in_layer = plru.lru_order.back();
        plru.lru_order.pop_back();
        const int flat = flat_slot(layer, slot_in_layer, slots_per_layer_);
        plru.lookup.erase(slots_[flat].key);
        // Invalidate the evicted slot's device mirror cell before reusing.
        write_lookup_cell(d_lookup_, n_experts_, slots_[flat].layer, slots_[flat].proj,
                          slots_[flat].expert, -1, stream);
        slots_[flat].occupied = false;
    }

    const int flat = flat_slot(layer, slot_in_layer, slots_per_layer_);
    Slot& slot = slots_[flat];
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(slot.gpu_ptr, src_host, expert_bytes, cudaMemcpyHostToDevice,
                                       stream));

    slot.key = key;
    slot.occupied = true;
    slot.layer = layer;
    slot.proj = proj_idx;
    slot.expert = key.expert_idx;
    plru.lru_order.push_front(slot_in_layer);
    plru.lookup[key] = {slot_in_layer, plru.lru_order.begin()};
    write_lookup_cell(d_lookup_, n_experts_, layer, proj_idx, key.expert_idx, slot_in_layer,
                      stream);

    if (debug_parity_)
        check_parity(stream);

    return slot.gpu_ptr;
}

bool ExpertLRUCache::check_parity(cudaStream_t stream) const {
    if (!d_lookup_ || n_layers_ <= 0 || n_experts_ <= 0)
        return true;
    size_t cells = static_cast<size_t>(n_layers_) * kExpertProjCount * n_experts_;
    std::vector<int> host(cells, -1);

    // Re-derive the expected device-mirror state from the authoritative
    // host-side slot table — this is what every "+1 every update" path
    // should converge to. Cell value is layer-relative slot_idx.
    for (size_t flat = 0; flat < slots_.size(); ++flat) {
        const Slot& slot = slots_[flat];
        if (!slot.occupied)
            continue;
        if (slot.layer < 0 || slot.proj < 0 || slot.expert < 0)
            continue;
        if (slot.layer >= n_layers_ || slot.proj >= kExpertProjCount || slot.expert >= n_experts_)
            continue;
        const int slot_in_layer = static_cast<int>(flat) - slot.layer * slots_per_layer_;
        size_t off = (static_cast<size_t>(slot.layer) * kExpertProjCount + slot.proj) *
                         n_experts_ +
                     slot.expert;
        host[off] = slot_in_layer;
    }

    std::vector<int> dev(cells);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dev.data(), d_lookup_, cells * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < cells; ++i) {
        if (host[i] != dev[i]) {
            int layer = static_cast<int>(i / (kExpertProjCount * n_experts_));
            int rest = static_cast<int>(i % (kExpertProjCount * n_experts_));
            int proj = rest / n_experts_;
            int expert = rest % n_experts_;
            IMP_LOG_FATAL(
                "ExpertLRUCache parity check failed at (layer=%d, proj=%d, expert=%d): "
                "host=%d device=%d. The host-side LRU and device-side mirror have diverged — "
                "Phase 3 invariant broken.",
                layer, proj, expert, host[i], dev[i]);
            return false;
        }
    }
    parity_checks_ok_++;
    return true;
}

void ExpertLRUCache::destroy() {
    if (pool_) {
        int64_t total = hits_ + misses_;
        if (total > 0) {
            IMP_LOG_INFO("Expert LRU cache stats: %ld hits, %ld misses (%.1f%% hit rate)", (long)hits_,
                         (long)misses_, hit_rate() * 100.0f);
        }
        if (alloc_)
            alloc_->free(pool_);
        else
            cudaFree(pool_);
        pool_ = nullptr;
    }
    if (d_lookup_) {
        cudaFree(d_lookup_);
        d_lookup_ = nullptr;
    }
    slots_.clear();
    per_layer_lru_.clear();
    host_expert_addrs_.clear();
    n_slots_ = 0;
    slots_per_layer_ = 0;
    n_layers_ = 0;
    n_experts_ = 0;
    hits_ = 0;
    misses_ = 0;
    parity_checks_ok_ = 0;
    debug_parity_ = false;
}

}  // namespace imp
