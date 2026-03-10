#pragma once

#include "memory/kv_cache.h"
#include <cstddef>
#include <cstdint>
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
    [[nodiscard]] bool allocate_blocks(int seq_id, int num_blocks);

    // Allocate and append a single block to an existing sequence.
    // Returns the new block_id, or -1 on failure.
    [[nodiscard]] int append_block(int seq_id);

    // Free every block owned by a sequence (respecting ref-counts via
    // cache_->free_block) and remove it from all tracking structures.
    // With prefix caching enabled, blocks whose ref_count drops to 0
    // are kept in the block hash table for potential reuse.
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
    [[nodiscard]] bool can_allocate(int num_blocks) const;

    // ── Prefix caching ───────────────────────────────────────────────

    // Associate `prefix_hash` with the current block table of `seq_id`.
    void register_prefix(int seq_id, size_t prefix_hash);

    // Look up blocks previously registered under `prefix_hash`.
    // Returns the cached block-id vector (empty if not found).
    std::vector<int> find_prefix(size_t prefix_hash) const;

    // Share the first `num_blocks` blocks from `source_seq_id` to
    // `target_seq_id` by incrementing their reference counts.
    void share_prefix(int source_seq_id, int target_seq_id, int num_blocks);

    // ── Content-addressed prefix caching ─────────────────────────────

    // Enable or disable automatic content-addressed prefix caching.
    // When enabled, freed blocks are retained in a hash table keyed by
    // token content, and allocate_blocks_with_prefix() reuses them.
    void set_prefix_caching_enabled(bool enabled) { prefix_caching_enabled_ = enabled; }
    bool prefix_caching_enabled() const { return prefix_caching_enabled_; }

    // Allocate blocks for a sequence, reusing cached KV blocks that
    // match the token prefix. `tokens` is the full input token sequence.
    // Returns the number of prefix blocks that were reused (i.e., the
    // number of blocks whose KV data is already computed). The caller
    // should skip prefill for the first `result * kKVBlockSize` tokens.
    // Returns -1 on allocation failure.
    [[nodiscard]] int allocate_blocks_with_prefix(int seq_id,
                                                   const int32_t* tokens,
                                                   int num_tokens);

    // Register the block hashes for a sequence after prefill completes.
    // This must be called so that future sequences can match against
    // these blocks. `tokens` is the full token sequence.
    void register_block_hashes(int seq_id, const int32_t* tokens, int num_tokens);

    // Number of cached (unreferenced) blocks in the hash table.
    int num_cached_blocks() const;

    // Evict a single cached block (LRU order). Returns true if a block
    // was evicted, false if no cached blocks remain.
    bool evict_cached_block();

    // ── Speculative decoding rollback ────────────────────────────────

    // Truncate a sequence's block table to fit `new_seq_len` tokens.
    // Frees any blocks beyond what's needed. The caller must also
    // truncate its own token vectors to match. This correctly handles
    // partial blocks (only frees blocks that are entirely past the
    // new length, keeping the last partially-filled block).
    void rollback(int seq_id, int new_seq_len);

    // ── Stats ────────────────────────────────────────────────────────

    // Number of sequences that currently have allocated blocks.
    int num_active_sequences() const;

    // Total number of blocks across all active sequences.
    int total_allocated_blocks() const;

    // ── Hashing utility (public for testing) ─────────────────────────

    // Compute the hash for a block of tokens. `parent_hash` is the hash
    // of the preceding block (0 for the first block). If the block has
    // fewer than kKVBlockSize tokens, it is NOT cacheable (partial block).
    static size_t compute_block_hash(const int32_t* tokens, int count,
                                     size_t parent_hash);

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

    // ── Prefix caching (legacy hash-to-block-table) ──────────────────
    // prefix_hash -> block ids that hold the cached KV data.
    std::unordered_map<size_t, std::vector<int>> prefix_cache_;

    // ── Content-addressed prefix caching ─────────────────────────────
    bool prefix_caching_enabled_ = false;

    // block_hash -> block_id. A block is in this map as long as its KV
    // data is valid (either actively referenced or cached for reuse).
    std::unordered_map<size_t, int> block_hash_to_id_;

    // Reverse map: block_id -> block_hash. Used to remove entries from
    // block_hash_to_id_ when a cached block is evicted.
    std::unordered_map<int, size_t> block_id_to_hash_;

    // LRU list of cached (unreferenced) block IDs. When a block's
    // ref_count drops to 0, it goes to the tail. Eviction pops from head.
    std::list<int> cached_blocks_lru_;
    std::unordered_map<int, std::list<int>::iterator> cached_blocks_map_;

    // seq_id -> vector of block hashes (parallel to seq_blocks_).
    // Used to maintain hash chain state for append_block operations.
    std::unordered_map<int, std::vector<size_t>> seq_block_hashes_;

    // Try to reclaim a cached block. Returns the block_id, or -1.
    int reclaim_cached_block();

    // Internal: allocate a fresh block, reclaiming cached blocks if needed.
    int allocate_block_with_eviction();
};

} // namespace imp
