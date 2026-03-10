#include "memory/kv_cache_manager.h"
#include "core/logging.h"

#include <algorithm>
#include <cassert>
#include <functional>

namespace imp {

// ─── Construction / destruction ──────────────────────────────────────

KVCacheManager::KVCacheManager(std::unique_ptr<KVCache> cache)
    : cache_(std::move(cache)) {
}

KVCacheManager::~KVCacheManager() = default;

// ─── Hashing utility ─────────────────────────────────────────────────

size_t KVCacheManager::compute_block_hash(const int32_t* tokens, int count,
                                           size_t parent_hash) {
    // FNV-1a inspired hash that chains with the parent block's hash.
    // This ensures that block N's hash depends on all preceding blocks,
    // so two sequences must share an identical prefix to match.
    size_t hash = parent_hash ^ 0xcbf29ce484222325ULL;
    for (int i = 0; i < count; ++i) {
        hash ^= static_cast<size_t>(static_cast<uint32_t>(tokens[i]));
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// ─── Sequence management ─────────────────────────────────────────────

bool KVCacheManager::allocate_blocks(int seq_id, int num_blocks) {
    if (num_blocks <= 0) return true;

    auto& blocks = seq_blocks_[seq_id];
    const size_t original_size = blocks.size();

    for (int i = 0; i < num_blocks; ++i) {
        int block_id = allocate_block_with_eviction();
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

    int block_id = allocate_block_with_eviction();
    if (block_id < 0) return -1;

    it->second.push_back(block_id);
    return block_id;
}

void KVCacheManager::free_sequence(int seq_id) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return;

    for (int block_id : it->second) {
        if (prefix_caching_enabled_ && cache_->ref_count(block_id) == 1) {
            // This sequence is the last reference. Check if the block is
            // registered in the hash table for potential reuse.
            auto hash_it = block_id_to_hash_.find(block_id);
            if (hash_it != block_id_to_hash_.end()) {
                // Don't free — keep ref_count=1 and move to cached LRU.
                if (cached_blocks_map_.find(block_id) == cached_blocks_map_.end()) {
                    cached_blocks_lru_.push_back(block_id);
                    cached_blocks_map_[block_id] = std::prev(cached_blocks_lru_.end());
                }
                continue; // Skip cache_->free_block()
            }
        }

        // Normal free: decrement ref_count, return to pool if it hits 0.
        cache_->free_block(block_id);
    }

    seq_blocks_.erase(it);
    seq_block_hashes_.erase(seq_id);

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

    // Count cached (unreferenced) blocks that can be evicted.
    reclaimable += static_cast<int>(cached_blocks_lru_.size());
    if (reclaimable >= num_blocks) return true;

    for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
        auto seq_it = seq_blocks_.find(*it);
        if (seq_it != seq_blocks_.end()) {
            reclaimable += static_cast<int>(seq_it->second.size());
        }
        if (reclaimable >= num_blocks) return true;
    }
    return false;
}

// ─── Prefix caching (legacy) ─────────────────────────────────────────

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

// ─── Content-addressed prefix caching ────────────────────────────────

int KVCacheManager::allocate_blocks_with_prefix(int seq_id,
                                                 const int32_t* tokens,
                                                 int num_tokens) {
    if (num_tokens <= 0) return 0;

    int total_blocks = (num_tokens + kKVBlockSize - 1) / kKVBlockSize;
    auto& blocks = seq_blocks_[seq_id];
    const size_t original_size = blocks.size();

    // We should only be called for a fresh sequence (no existing blocks).
    if (!blocks.empty()) {
        // Fall back to plain allocation for the remaining blocks.
        int existing = static_cast<int>(blocks.size());
        int additional = total_blocks - existing;
        if (additional > 0) {
            return allocate_blocks(seq_id, additional) ? 0 : -1;
        }
        return 0;
    }

    auto& hashes = seq_block_hashes_[seq_id];
    int reused_blocks = 0;
    size_t parent_hash = 0;

    for (int b = 0; b < total_blocks; ++b) {
        int block_start = b * kKVBlockSize;
        int block_tokens = std::min(kKVBlockSize, num_tokens - block_start);

        // Only full blocks are cacheable.
        bool is_full_block = (block_tokens == kKVBlockSize);

        if (prefix_caching_enabled_ && is_full_block) {
            size_t block_hash = compute_block_hash(tokens + block_start,
                                                    block_tokens, parent_hash);

            // Check if this block already exists in the hash table.
            auto hit = block_hash_to_id_.find(block_hash);
            if (hit != block_hash_to_id_.end()) {
                int cached_block = hit->second;

                // Remove from cached LRU if it was unreferenced.
                auto cached_it = cached_blocks_map_.find(cached_block);
                if (cached_it != cached_blocks_map_.end()) {
                    cached_blocks_lru_.erase(cached_it->second);
                    cached_blocks_map_.erase(cached_it);
                    // Block already has ref_count=1 from being retained.
                } else {
                    // Block is actively referenced by another sequence — share it.
                    cache_->inc_ref(cached_block);
                }

                blocks.push_back(cached_block);
                hashes.push_back(block_hash);
                parent_hash = block_hash;
                ++reused_blocks;
                continue;
            }

            // No cache hit — allocate a fresh block and register it.
            int block_id = allocate_block_with_eviction();
            if (block_id < 0) {
                // Rollback everything we allocated/shared in this call.
                for (size_t j = original_size; j < blocks.size(); ++j) {
                    cache_->free_block(blocks[j]);
                }
                blocks.resize(original_size);
                hashes.resize(original_size);
                if (blocks.empty()) {
                    seq_blocks_.erase(seq_id);
                    seq_block_hashes_.erase(seq_id);
                }
                return -1;
            }

            blocks.push_back(block_id);
            hashes.push_back(block_hash);
            // Don't register in hash table yet — KV data hasn't been computed.
            // register_block_hashes() will be called after prefill.
            parent_hash = block_hash;
        } else {
            // Partial block or prefix caching disabled — plain allocation.
            int block_id = allocate_block_with_eviction();
            if (block_id < 0) {
                for (size_t j = original_size; j < blocks.size(); ++j) {
                    cache_->free_block(blocks[j]);
                }
                blocks.resize(original_size);
                hashes.resize(original_size);
                if (blocks.empty()) {
                    seq_blocks_.erase(seq_id);
                    seq_block_hashes_.erase(seq_id);
                }
                return -1;
            }

            blocks.push_back(block_id);
            if (is_full_block) {
                size_t block_hash = compute_block_hash(tokens + block_start,
                                                        block_tokens, parent_hash);
                hashes.push_back(block_hash);
                parent_hash = block_hash;
            } else {
                hashes.push_back(0); // Partial block, not cacheable.
            }
        }
    }

    // Track in LRU.
    if (lru_map_.find(seq_id) == lru_map_.end()) {
        lru_order_.push_back(seq_id);
        lru_map_[seq_id] = std::prev(lru_order_.end());
    }

    if (reused_blocks > 0) {
        IMP_LOG_DEBUG("PrefixCache: seq %d reused %d/%d blocks (%d tokens skippable)",
                      seq_id, reused_blocks, total_blocks,
                      reused_blocks * kKVBlockSize);
    }
    return reused_blocks;
}

void KVCacheManager::register_block_hashes(int seq_id,
                                            const int32_t* tokens,
                                            int num_tokens) {
    if (!prefix_caching_enabled_) return;

    auto blocks_it = seq_blocks_.find(seq_id);
    if (blocks_it == seq_blocks_.end()) return;

    const auto& blocks = blocks_it->second;
    int total_blocks = static_cast<int>(blocks.size());

    auto& hashes = seq_block_hashes_[seq_id];
    size_t parent_hash = 0;

    for (int b = 0; b < total_blocks; ++b) {
        int block_start = b * kKVBlockSize;
        int block_tokens = std::min(kKVBlockSize, num_tokens - block_start);

        if (block_tokens < kKVBlockSize) break; // Only full blocks are cacheable.

        size_t block_hash = compute_block_hash(tokens + block_start,
                                                block_tokens, parent_hash);

        // Ensure hash vector is populated.
        if (b < static_cast<int>(hashes.size())) {
            hashes[b] = block_hash;
        } else {
            hashes.push_back(block_hash);
        }

        // Register in hash table if not already there.
        int block_id = blocks[b];
        if (block_hash_to_id_.find(block_hash) == block_hash_to_id_.end()) {
            block_hash_to_id_[block_hash] = block_id;
            block_id_to_hash_[block_id] = block_hash;
        }

        parent_hash = block_hash;
    }
}

int KVCacheManager::num_cached_blocks() const {
    return static_cast<int>(cached_blocks_lru_.size());
}

bool KVCacheManager::evict_cached_block() {
    int block_id = reclaim_cached_block();
    return block_id >= 0;
}

int KVCacheManager::reclaim_cached_block() {
    if (cached_blocks_lru_.empty()) return -1;

    int block_id = cached_blocks_lru_.front();
    cached_blocks_lru_.pop_front();
    cached_blocks_map_.erase(block_id);

    // Remove from hash tables.
    auto hash_it = block_id_to_hash_.find(block_id);
    if (hash_it != block_id_to_hash_.end()) {
        block_hash_to_id_.erase(hash_it->second);
        block_id_to_hash_.erase(hash_it);
    }

    // The block has ref_count=1 (retained from free_sequence).
    // Free it to return it to the pool.
    cache_->free_block(block_id);
    return block_id;
}

int KVCacheManager::allocate_block_with_eviction() {
    int block_id = cache_->allocate_block();
    if (block_id >= 0) return block_id;

    // Try reclaiming a cached block first (cheaper than evicting a sequence).
    if (reclaim_cached_block() >= 0) {
        block_id = cache_->allocate_block();
        if (block_id >= 0) return block_id;
    }

    return -1; // Caller should try evict_lru() if needed.
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

    // Trim the block hash vector too.
    auto hash_it = seq_block_hashes_.find(seq_id);
    if (hash_it != seq_block_hashes_.end()) {
        auto& hashes = hash_it->second;
        while (static_cast<int>(hashes.size()) > blocks_needed) {
            hashes.pop_back();
        }
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
