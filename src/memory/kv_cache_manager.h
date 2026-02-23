#pragma once

#include "memory/kv_cache.h"
#include <cstddef>
#include <list>
#include <unordered_map>
#include <vector>
#include <memory>

namespace imp {

class KVCacheManager {
public:
    explicit KVCacheManager(std::unique_ptr<KVCache> cache);
    ~KVCacheManager();

    // ── Sequence management ──────────────────────────────────────────

    // Allocate `num_blocks` new blocks for a sequence. Appends to the
    // existing block table if the sequence already has blocks.  On failure
    // (pool exhausted mid-allocation) all blocks allocated during *this*
    // call are rolled back and the function returns false.
    bool allocate_blocks(int seq_id, int num_blocks);

    // Allocate and append a single block to an existing sequence.
    // Returns the new block_id, or -1 on failure.
    int append_block(int seq_id);

    // Free every block owned by a sequence (respecting ref-counts via
    // cache_->free_block) and remove it from all tracking structures.
    void free_sequence(int seq_id);

    // Return the block table for a sequence (empty vector if unknown).
    const std::vector<int>& block_table(int seq_id) const;

    // Number of free blocks in the underlying cache.
    int num_free_blocks() const;

    // ── LRU eviction ─────────────────────────────────────────────────

    // Move a sequence to the most-recently-used position (tail of list).
    void touch(int seq_id);

    // Evict the least-recently-used sequence, freeing its blocks.
    // Returns the evicted seq_id, or -1 if there is nothing to evict.
    int evict_lru();

    // Check whether `num_blocks` blocks are available.  Returns true if
    // the free pool already has enough blocks *or* if evicting LRU
    // sequences could free enough.
    bool can_allocate(int num_blocks) const;

    // ── Prefix caching ───────────────────────────────────────────────

    // Associate `prefix_hash` with the current block table of `seq_id`.
    void register_prefix(int seq_id, size_t prefix_hash);

    // Look up blocks previously registered under `prefix_hash`.
    // Returns the cached block-id vector (empty if not found).
    std::vector<int> find_prefix(size_t prefix_hash) const;

    // Share the first `num_blocks` blocks from `source_seq_id` to
    // `target_seq_id` by incrementing their reference counts.
    void share_prefix(int source_seq_id, int target_seq_id, int num_blocks);

    // ── Speculative decoding rollback ────────────────────────────────

    // Roll back the last `n_tokens` tokens for a sequence by truncating
    // its block table. Frees any blocks that become empty after rollback.
    // This is used when speculative decoding rejects draft tokens.
    void rollback(int seq_id, int n_tokens);

    // ── Stats ────────────────────────────────────────────────────────

    // Number of sequences that currently have allocated blocks.
    int num_active_sequences() const;

    // Total number of blocks across all active sequences.
    int total_allocated_blocks() const;

private:
    // Underlying block-level cache (owns the memory pool).
    std::unique_ptr<KVCache> cache_;

    // seq_id -> ordered list of block ids.
    std::unordered_map<int, std::vector<int>> seq_blocks_;

    // ── LRU tracking ─────────────────────────────────────────────────
    // Doubly-linked list of seq_ids; most recently used at the *tail*.
    std::list<int> lru_order_;
    // O(1) lookup from seq_id to its position in lru_order_.
    std::unordered_map<int, std::list<int>::iterator> lru_map_;

    // ── Prefix caching ───────────────────────────────────────────────
    // prefix_hash -> block ids that hold the cached KV data.
    std::unordered_map<size_t, std::vector<int>> prefix_cache_;
};

} // namespace imp
