#include "executor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

bool ExpertLRUCache::init(size_t max_expert_raw, size_t budget_bytes) {
    if (max_expert_raw == 0 || budget_bytes == 0) return false;

    slot_size_ = max_expert_raw;
    n_slots_ = static_cast<int>(budget_bytes / slot_size_);
    if (n_slots_ < 2) {
        IMP_LOG_WARN("Expert LRU cache: budget too small for even 2 slots "
                     "(need %zu bytes/slot, budget %zu bytes)",
                     slot_size_, budget_bytes);
        return false;
    }

    size_t total = static_cast<size_t>(n_slots_) * slot_size_;
    cudaError_t err = cudaMalloc(&pool_, total);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("Expert LRU cache: cudaMalloc failed for %zu bytes (%d slots): %s",
                     total, n_slots_, cudaGetErrorString(err));
        pool_ = nullptr;
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

    IMP_LOG_INFO("Expert LRU cache: %d slots x %.2f MiB = %.2f MiB GPU memory",
                 n_slots_, slot_size_ / (1024.0 * 1024.0),
                 total / (1024.0 * 1024.0));
    return true;
}

void* ExpertLRUCache::find(ExpertCacheKey key) {
    auto it = lookup_.find(key);
    if (it == lookup_.end()) return nullptr;

    // Move to front (most recently used)
    auto& [slot_idx, lru_it] = it->second;
    lru_order_.erase(lru_it);
    lru_order_.push_front(slot_idx);
    it->second.second = lru_order_.begin();
    hits_++;
    return slots_[slot_idx].gpu_ptr;
}

void* ExpertLRUCache::get_or_load(ExpertCacheKey key, const void* src_host,
                                   size_t expert_bytes, cudaStream_t stream) {
    // Check cache hit
    void* cached = find(key);
    if (cached) return cached;

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
        slots_[slot_idx].occupied = false;
    }

    // Load expert from host to GPU slot
    Slot& slot = slots_[slot_idx];
    cudaMemcpyAsync(slot.gpu_ptr, src_host, expert_bytes,
                    cudaMemcpyHostToDevice, stream);

    // Register in LRU
    slot.key = key;
    slot.occupied = true;
    lru_order_.push_front(slot_idx);
    lookup_[key] = {slot_idx, lru_order_.begin()};

    return slot.gpu_ptr;
}

void ExpertLRUCache::destroy() {
    if (pool_) {
        int64_t total = hits_ + misses_;
        if (total > 0) {
            IMP_LOG_INFO("Expert LRU cache stats: %ld hits, %ld misses (%.1f%% hit rate)",
                         (long)hits_, (long)misses_, hit_rate() * 100.0f);
        }
        cudaFree(pool_);
        pool_ = nullptr;
    }
    slots_.clear();
    lru_order_.clear();
    lookup_.clear();
    n_slots_ = 0;
    hits_ = 0;
    misses_ = 0;
}

} // namespace imp
