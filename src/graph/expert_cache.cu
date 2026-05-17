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
    n_slots_ = static_cast<int>(budget_bytes / slot_size_);
    if (n_slots_ < 2) {
        IMP_LOG_WARN(
            "Expert LRU cache: budget too small for even 2 slots "
            "(need %zu bytes/slot, budget %zu bytes)",
            slot_size_, budget_bytes);
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
        return false;
    }

    slots_.resize(n_slots_);
    for (int i = 0; i < n_slots_; i++) {
        slots_[i].gpu_ptr = static_cast<char*>(pool_) + static_cast<size_t>(i) * slot_size_;
    }

    lookup_.reserve(n_slots_ * 2);
    hits_ = 0;
    misses_ = 0;

    // Phase 2: device-side lookup mirror. Size [n_layers × 3 × n_experts]
    // int32. n_layers/n_experts can be 0 in tests that don't care about the
    // mirror — in that case we skip the allocation entirely.
    n_layers_ = std::max(0, n_layers);
    n_experts_ = std::max(0, n_experts);
    debug_parity_ = debug_parity;
    parity_checks_ok_ = 0;
    if (n_layers_ > 0 && n_experts_ > 0) {
        size_t cells = static_cast<size_t>(n_layers_) * kExpertProjCount * n_experts_;
        size_t bytes = cells * sizeof(int);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_lookup_), bytes);
        if (err != cudaSuccess || !d_lookup_) {
            IMP_LOG_WARN(
                "Expert LRU cache: device-side mirror alloc failed (%zu bytes, err=%s) — "
                "disabling Phase 2 mirror, host-side LRU still functional.",
                bytes, cudaGetErrorString(err));
            d_lookup_ = nullptr;
        } else {
            // (int32_t)-1 = 0xFFFFFFFF — memset 0xFF gives -1 in every cell.
            IMP_CUDA_CHECK_LOG(cudaMemset(d_lookup_, 0xFF, bytes));
        }
    }

    IMP_LOG_INFO("Expert LRU cache: %d slots x %.2f MiB = %.2f MiB GPU memory%s%s", n_slots_,
                 slot_size_ / (1024.0 * 1024.0), total / (1024.0 * 1024.0),
                 d_lookup_ ? " (+ device mirror)" : "",
                 debug_parity_ ? " [parity-check on]" : "");
    return true;
}

void* ExpertLRUCache::find(ExpertCacheKey key) {
    auto it = lookup_.find(key);
    if (it == lookup_.end())
        return nullptr;

    // Move to front (most recently used)
    auto& [slot_idx, lru_it] = it->second;
    lru_order_.erase(lru_it);
    lru_order_.push_front(slot_idx);
    it->second.second = lru_order_.begin();
    hits_++;
    return slots_[slot_idx].gpu_ptr;
}

namespace {

// Write a single int32 cell to the device-side lookup mirror.
// Async on the supplied stream — Phase 2 mirror is write-only, so ordering
// against subsequent kernels is whatever the stream provides.
inline void write_lookup_cell(int* d_lookup, int n_experts, int layer, int proj, int expert,
                              int slot_idx, cudaStream_t stream) {
    if (!d_lookup)
        return;
    if (layer < 0 || proj < 0 || expert < 0)
        return;
    size_t off = (static_cast<size_t>(layer) * kExpertProjCount + proj) * n_experts + expert;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_lookup + off, &slot_idx, sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
}

}  // namespace

void* ExpertLRUCache::get_or_load(int layer, ExpertProj proj, ExpertCacheKey key,
                                  const void* src_host, size_t expert_bytes,
                                  cudaStream_t stream) {
    const int proj_idx = static_cast<int>(proj);

    // Check cache hit — find() handles LRU front-update. Hits don't change
    // which slot holds (layer, proj, expert), so the device mirror cell
    // remains correct.
    void* cached = find(key);
    if (cached) {
        if (debug_parity_)
            check_parity(stream);
        return cached;
    }

    misses_++;

    // Find a slot: use an unoccupied one, or evict LRU
    int slot_idx = -1;

    if (static_cast<int>(lookup_.size()) < n_slots_) {
        // Find first unoccupied slot
        for (int i = 0; i < n_slots_; i++) {
            if (!slots_[i].occupied) {
                slot_idx = i;
                break;
            }
        }
    }

    if (slot_idx < 0) {
        // Evict LRU (back of list)
        slot_idx = lru_order_.back();
        lru_order_.pop_back();
        // Remove old entry from lookup
        lookup_.erase(slots_[slot_idx].key);
        // Invalidate the evicted slot's device mirror cell before reusing.
        write_lookup_cell(d_lookup_, n_experts_, slots_[slot_idx].layer, slots_[slot_idx].proj,
                          slots_[slot_idx].expert, -1, stream);
        slots_[slot_idx].occupied = false;
    }

    // Load expert from host to GPU slot
    Slot& slot = slots_[slot_idx];
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(slot.gpu_ptr, src_host, expert_bytes, cudaMemcpyHostToDevice, stream));

    // Register in LRU + device mirror
    slot.key = key;
    slot.occupied = true;
    slot.layer = layer;
    slot.proj = proj_idx;
    slot.expert = key.expert_idx;
    lru_order_.push_front(slot_idx);
    lookup_[key] = {slot_idx, lru_order_.begin()};
    write_lookup_cell(d_lookup_, n_experts_, layer, proj_idx, key.expert_idx, slot_idx, stream);

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
    // should converge to.
    for (const auto& slot : slots_) {
        if (!slot.occupied)
            continue;
        if (slot.layer < 0 || slot.proj < 0 || slot.expert < 0)
            continue;
        if (slot.layer >= n_layers_ || slot.proj >= kExpertProjCount || slot.expert >= n_experts_)
            continue;
        size_t off = (static_cast<size_t>(slot.layer) * kExpertProjCount + slot.proj) *
                         n_experts_ +
                     slot.expert;
        int slot_idx = static_cast<int>(&slot - slots_.data());
        host[off] = slot_idx;
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
                "Phase 2 invariant broken.",
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
    lru_order_.clear();
    lookup_.clear();
    n_slots_ = 0;
    n_layers_ = 0;
    n_experts_ = 0;
    hits_ = 0;
    misses_ = 0;
    parity_checks_ok_ = 0;
    debug_parity_ = false;
}

}  // namespace imp
