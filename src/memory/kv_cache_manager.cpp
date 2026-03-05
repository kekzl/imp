#include "memory/kv_cache_manager.h"

#include <algorithm>
#include <cassert>

namespace imp {

// ─── Construction / destruction ──────────────────────────────────────

KVCacheManager::KVCacheManager(std::unique_ptr<KVCache> cache)
    : cache_(std::move(cache)) {
}

KVCacheManager::~KVCacheManager() = default;

// ─── Sequence management ─────────────────────────────────────────────

bool KVCacheManager::allocate_blocks(int seq_id, int num_blocks) {
    if (num_blocks <= 0) return true;

    auto& blocks = seq_blocks_[seq_id];
    const size_t original_size = blocks.size();

    for (int i = 0; i < num_blocks; ++i) {
        int block_id = cache_->allocate_block();
        if (block_id < 0) {
            // Rollback: free every block we allocated in *this* call.
            for (size_t j = original_size; j < blocks.size(); ++j) {
                cache_->free_block(blocks[j]);
            }
            blocks.resize(original_size);

            // If the sequence had no blocks before and we failed to
            // allocate any, remove the empty entry we just created.
            if (blocks.empty()) {
                seq_blocks_.erase(seq_id);
            }
            return false;
        }
        blocks.push_back(block_id);
    }

    // Make sure the sequence is tracked in the LRU list.
    if (lru_map_.find(seq_id) == lru_map_.end()) {
        lru_order_.push_back(seq_id);
        lru_map_[seq_id] = std::prev(lru_order_.end());
    }

    return true;
}

int KVCacheManager::append_block(int seq_id) {
    // The sequence must already exist.
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return -1;

    int block_id = cache_->allocate_block();
    if (block_id < 0) return -1;

    it->second.push_back(block_id);
    return block_id;
}

void KVCacheManager::free_sequence(int seq_id) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return;

    for (int block_id : it->second) {
        cache_->free_block(block_id);
    }
    seq_blocks_.erase(it);

    // Remove from LRU tracking.
    auto lru_it = lru_map_.find(seq_id);
    if (lru_it != lru_map_.end()) {
        lru_order_.erase(lru_it->second);
        lru_map_.erase(lru_it);
    }
}

const std::vector<int>& KVCacheManager::block_table(int seq_id) const {
    static const std::vector<int> empty;
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return empty;
    return it->second;
}

int KVCacheManager::num_free_blocks() const {
    return cache_->num_free_blocks();
}

// ─── LRU eviction ────────────────────────────────────────────────────

void KVCacheManager::touch(int seq_id) {
    auto it = lru_map_.find(seq_id);
    if (it == lru_map_.end()) {
        // Sequence is not tracked yet -- add it at the tail.
        lru_order_.push_back(seq_id);
        lru_map_[seq_id] = std::prev(lru_order_.end());
        return;
    }
    // Splice to the tail (most recently used).
    lru_order_.splice(lru_order_.end(), lru_order_, it->second);
}

int KVCacheManager::evict_lru() {
    if (lru_order_.empty()) return -1;

    int victim = lru_order_.front();
    free_sequence(victim);  // also removes from lru_order_ / lru_map_
    return victim;
}

bool KVCacheManager::can_allocate(int num_blocks) const {
    if (num_blocks <= 0) return true;
    if (cache_->num_free_blocks() >= num_blocks) return true;

    // Count how many blocks we *could* reclaim by evicting LRU sequences.
    int reclaimable = cache_->num_free_blocks();
    for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
        auto seq_it = seq_blocks_.find(*it);
        if (seq_it != seq_blocks_.end()) {
            reclaimable += static_cast<int>(seq_it->second.size());
        }
        if (reclaimable >= num_blocks) return true;
    }
    return false;
}

// ─── Prefix caching ─────────────────────────────────────────────────

void KVCacheManager::register_prefix(int seq_id, size_t prefix_hash) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return;

    // Store a copy of the sequence's current block table under this hash.
    prefix_cache_[prefix_hash] = it->second;
}

std::vector<int> KVCacheManager::find_prefix(size_t prefix_hash) const {
    auto it = prefix_cache_.find(prefix_hash);
    if (it == prefix_cache_.end()) return {};
    return it->second;
}

void KVCacheManager::share_prefix(int source_seq_id, int target_seq_id,
                                  int num_blocks) {
    auto src_it = seq_blocks_.find(source_seq_id);
    if (src_it == seq_blocks_.end()) return;

    const auto& src_blocks = src_it->second;
    int to_share = std::min(num_blocks, static_cast<int>(src_blocks.size()));
    if (to_share <= 0) return;

    auto& tgt_blocks = seq_blocks_[target_seq_id];

    for (int i = 0; i < to_share; ++i) {
        int block_id = src_blocks[i];
        cache_->inc_ref(block_id);
        tgt_blocks.push_back(block_id);
    }

    // Make sure the target sequence is in the LRU list.
    if (lru_map_.find(target_seq_id) == lru_map_.end()) {
        lru_order_.push_back(target_seq_id);
        lru_map_[target_seq_id] = std::prev(lru_order_.end());
    }
}

// ─── Speculative decoding rollback ───────────────────────────────────

void KVCacheManager::rollback(int seq_id, int new_seq_len) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return;

    auto& blocks = it->second;
    if (blocks.empty() || new_seq_len < 0) return;

    // Keep exactly ceil(new_seq_len / kKVBlockSize) blocks.
    // A partially-filled last block is retained (its stale slots
    // will be overwritten on subsequent writes).
    int blocks_needed = (new_seq_len + kKVBlockSize - 1) / kKVBlockSize;
    while (static_cast<int>(blocks.size()) > blocks_needed) {
        cache_->free_block(blocks.back());
        blocks.pop_back();
    }
}

// ─── Stats ───────────────────────────────────────────────────────────

int KVCacheManager::num_active_sequences() const {
    return static_cast<int>(seq_blocks_.size());
}

int KVCacheManager::total_allocated_blocks() const {
    int total = 0;
    for (const auto& [seq_id, blocks] : seq_blocks_) {
        total += static_cast<int>(blocks.size());
    }
    return total;
}

} // namespace imp
