#pragma once

#include "memory/kv_cache.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <list>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <utility>

namespace imp {

struct KVCacheStats {
    int active_sequences = 0;
    int total_blocks = 0;
    int free_blocks = 0;
    int cached_blocks = 0;
    int pinned_blocks = 0;
};

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

    // Access the underlying KV cache (for runtime block_size, dtype, etc.).
    KVCache* kv_cache() const { return cache_.get(); }

    // Number of free blocks in the underlying cache.
    int num_free_blocks() const;

    // ── LRU eviction ─────────────────────────────────────────────────

    // Move a sequence to the most-recently-used position (tail of list).
    void touch(int seq_id);

    // Evict the least-recently-used sequence, freeing its blocks.
    // Returns the evicted seq_id, or -1 if there is nothing to evict.
    // WARNING: every lru_order_ entry is a LIVE sequence and imp has no
    // recompute path, so the engine must NOT use this to free room under KV
    // pressure (it reject-newests instead). Manager primitive only.
    int evict_lru();

    // Check whether `num_blocks` blocks are available.  Returns true if
    // the free pool already has enough blocks *or* if evicting LRU
    // sequences could free enough.
    [[nodiscard]] bool can_allocate(int num_blocks) const;

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
    // Reuse stops at the first non-cached block (a hole after an evicted
    // block must not count — skipped tokens need a contiguous KV prefix).
    // `max_reuse_blocks` caps reuse (-1 = unlimited); hybrid models pass
    // the recurrent-snapshot boundary so prefill never re-writes shared
    // blocks beyond the restorable state position.
    // Returns -1 on allocation failure.
    //
    // `content_salt` seeds the hash chain. It exists because the cache is
    // addressed by TOKEN IDS, and that is not enough for a multimodal prompt:
    // every image token carries the same id, so two requests with different
    // pictures produce identical prefixes and the second one would inherit the
    // first one's KV. Passing a hash of the image content makes the two chains
    // diverge from block 0, so a hit requires the same tokens AND the same
    // picture. 0 for text, which is the unchanged behaviour.
    [[nodiscard]] int allocate_blocks_with_prefix(int seq_id, std::span<const int32_t> tokens,
                                                  int max_reuse_blocks = -1, size_t content_salt = 0);

    // Read-only probe: length (in blocks) of the longest fully-cached
    // contiguous block prefix for `tokens`, without allocating anything.
    // Fills `chain_hashes` with the chained hash of every FULL block of
    // `tokens` (independent of cache state). Used by the hybrid recurrent-
    // snapshot lookup to pick a restore boundary before allocation.
    int longest_cached_prefix_blocks(std::span<const int32_t> tokens, std::vector<size_t>& chain_hashes,
                                     size_t content_salt = 0) const;

    // Register the block hashes for a sequence after prefill completes.
    // This must be called so that future sequences can match against
    // these blocks. `tokens` is the full token sequence.
    void register_block_hashes(int seq_id, std::span<const int32_t> tokens, size_t content_salt = 0);

    // Number of cached (unreferenced) blocks in the hash table.
    int num_cached_blocks() const;

    // Evict a single cached block (LRU order). Returns true if a block
    // was evicted, false if no cached blocks remain.
    bool evict_cached_block();

    // ── Prefix block pinning (agentic workloads) ────────────────────

    // Pin the first `num_blocks` blocks of `seq_id`. Pinned blocks are
    // never freed by evict_lru(), reclaim_cached_block(), or
    // free_sequence(). After free_sequence(), pinned blocks stay in the
    // cache pool with ref_count=1 for reuse by future sequences.
    // Re-pinning the same seq_id replaces its previous pin set. When a
    // pin budget is set, the oldest pin owners are unpinned (FIFO) until
    // the new pin fits; their blocks degrade to normal cached blocks.
    void pin_prefix(int seq_id, int num_blocks);

    // Remove pinning for all blocks that were pinned via pin_prefix()
    // for the given sequence. Does NOT free the blocks — they remain
    // allocated as normal (or cached if the sequence was already freed).
    // Blocks shared with other pin owners stay pinned.
    void unpin_prefix(int seq_id);

    // Number of blocks currently pinned across all sequences.
    int num_pinned_blocks() const;

    // Cap on unique pin_prefix-pinned blocks. 0 = unlimited (default).
    void set_pin_budget_blocks(int blocks) { pin_budget_blocks_ = blocks; }
    int pin_budget_blocks() const { return pin_budget_blocks_; }

    // Cached (unreferenced) blocks that are actually reclaimable, i.e.
    // excluding pinned blocks. O(1).
    int num_reclaimable_cached_blocks() const { return reclaimable_cached_count_; }

    // ── SWA-aware sizing (kv_cache.swa_sizing) ───────────────────────
    //
    // Sliding-window layers read/write a small dedicated block group
    // (KVCache SWA group); the manager keeps a second positional table per
    // sequence, same length/indexing as the global table, with -1 holes
    // for positions outside the trailing window. Engine contract:
    //   - swa_prepare(seq, from_tokens, upto_tokens) BEFORE forwarding the
    //     token range [from, upto) (prefill chunk / decode step / on-device
    //     burst incl. spec drafts): pads the SWA table to the global
    //     table's length and allocates live entries for blocks intersecting
    //     [from - window - slack, upto) — every write in the range plus the
    //     window each query reads. Earlier slots stay/become -1.
    //   - swa_trim(seq, committed_tokens) after the step commits: frees
    //     entries fully below committed - window - slack. The slack must
    //     cover the deepest spec-decode rollback (rollback never restores
    //     positions older than slack tokens behind the prepared tail).
    //   - rollback() / free_sequence() handle the SWA table automatically.
    // SWA blocks are never hashed, pinned, persisted, or shared.
    void enable_swa_sizing(int window_tokens, int slack_tokens);
    bool swa_sizing_enabled() const { return swa_window_ > 0; }
    [[nodiscard]] bool swa_prepare(int seq_id, int from_tokens, int upto_tokens);
    [[nodiscard]] bool swa_prepare(int seq_id, int upto_tokens) {
        return swa_prepare(seq_id, upto_tokens, upto_tokens);
    }
    void swa_trim(int seq_id, int committed_tokens);
    // Positional SWA block table (parallel to block_table; -1 = hole).
    // Empty vector if unknown seq or SWA sizing disabled.
    const std::vector<int>& swa_block_table(int seq_id) const;

    // ── SWA window snapshots (prefix caching for SWA-sized models) ───
    //
    // A snapshot is the packed KV content of the live SWA window blocks at
    // a block-aligned position P: for each SWA layer, blocks
    // [swa_first_live_block(P), P/bs). Restoring it into a fresh sequence
    // makes a prefix-cache hit at P valid for windowed layers (their
    // earlier blocks were trailing-freed and cannot back reuse).
    // enable_swa_snapshots() precomputes the slab layout and allocates the
    // copy-desc staging; must be called after enable_swa_sizing.
    bool enable_swa_snapshots();
    // Fixed slab size (bytes) of one snapshot; 0 until enabled.
    size_t swa_snapshot_bytes() const { return swa_snap_bytes_; }
    int swa_first_live_block(int upto_tokens) const;
    // Pack seq's live window blocks at exactly upto_tokens into slab
    // (device, >= swa_snapshot_bytes()). False if a needed block is a hole.
    bool swa_snapshot_pack(int seq_id, int upto_tokens, void* slab, cudaStream_t stream);
    // Allocate window blocks for a sequence whose global prefix [0, upto)
    // was reused from the prefix cache and fill them from slab. Table slots
    // before the window stay -1 holes. False on SWA-group exhaustion
    // (allocated blocks are released; caller falls back to full prefill).
    bool swa_snapshot_restore(int seq_id, int upto_tokens, const void* slab, cudaStream_t stream);

    // ── Speculative decoding rollback ────────────────────────────────

    // Truncate a sequence's block table to fit `new_seq_len` tokens.
    // Frees any blocks beyond what's needed. The caller must also
    // truncate its own token vectors to match. This correctly handles
    // partial blocks (only frees blocks that are entirely past the
    // new length, keeping the last partially-filled block).
    void rollback(int seq_id, int new_seq_len);

    // ── StreamingLLM middle eviction ─────────────────────────────────

    // Free the "middle" blocks of a sequence, keeping the first
    // `n_sink_tokens` tokens (attention sinks) and the last
    // `n_window_tokens` tokens (sliding window). Sink blocks become
    // pinned to prevent later eviction. Freed slots in the block table
    // are replaced with sentinel `-1` so the block table length (and
    // therefore positional alignment in attention kernels) is unchanged.
    //
    // The caller must use a streaming-aware decode kernel (`n_sinks > 0`
    // passed to paged_attention_decode) so that the sentinel `-1` slots
    // are skipped during attention.
    //
    // This is idempotent: calling repeatedly with the same parameters
    // is a no-op once the sequence has been streamed.
    //
    // Returns the number of physical blocks freed (0 if nothing to evict).
    int evict_middle_blocks(int seq_id, int n_sink_tokens, int n_window_tokens);

    // ── Stats ────────────────────────────────────────────────────────

    // Snapshot of all cache statistics.
    KVCacheStats stats() const;

    // Individual stats (convenience wrappers).
    int num_active_sequences() const;
    int total_allocated_blocks() const;

    // ── Persistent prefix cache ─────────────────────────────────────

    // Save all cached (unreferenced) blocks to disk. Includes KV data
    // (GPU→host copy), hash mappings, and metadata for validation.
    // model_fingerprint identifies the producing model/tokenizer/quant.
    // Returns number of blocks saved, or -1 on error.
    int save_prefix_cache(const std::string& path, uint64_t model_fingerprint,
                          cudaStream_t stream = nullptr);

    // Load cached blocks from disk. Validates metadata + model_fingerprint
    // against the current model. Uploads KV data to GPU and registers hash
    // mappings. Returns number of blocks restored, or -1 on error/mismatch.
    int load_prefix_cache(const std::string& path, uint64_t model_fingerprint,
                          cudaStream_t stream = nullptr);

    // ── BitDecoding Phase 3: residual FP16 cache ─────────────────────
    //
    // Optional per-sequence ring buffer of FP16 K/V for the newest N tokens.
    // Decode kernel reads attention from BOTH the NVFP4 paged cache and the
    // FP16 residual; the K/V write path appends FP16 to the ring first and
    // only quantizes-to-NVFP4 when the ring fills (the to-be-overwritten
    // entry is then evicted to the paged cache).
    //
    // Phase 3a (this PR): allocation + accessor surface only. Phase 3b
    // (kernel) and Phase 3c (write path) build on top.
    //
    // Layout: `[max_seqs, n_layers, 2 (K|V), residual_n, n_kv_heads, head_dim]`
    // FP16 contiguous. Indexed by (seq_slot, layer, k_or_v).

    struct ResidualRingState {
        int write_idx = 0;    // next slot to write into [0, residual_n)
        int fill_count = 0;   // populated entries [0, residual_n]
    };

    // Allocate the residual pool. Idempotent: returns false if already enabled.
    // Returns true on success, false on alloc failure or if residual_n==0.
    [[nodiscard]] bool enable_residual_buffer(int max_seqs, int residual_n, VRAMAllocator* alloc);

    bool residual_enabled() const { return residual_pool_ != nullptr; }
    int residual_n_tokens() const { return residual_n_tokens_; }
    int residual_max_seqs() const { return residual_max_seqs_; }
    int residual_n_kv_heads() const { return residual_n_kv_heads_; }
    int residual_head_dim() const { return residual_head_dim_; }

    // Per-(seq, layer) K/V pointer into the residual pool. Returns nullptr if
    // residual is not enabled, seq_id has no allocated slot, or layer is out
    // of range. Uses the slot allocator (allocate_residual_slot) — falls back
    // to nullptr if the seq has not been admitted yet.
    void* residual_k_ptr(int seq_id, int layer) const;
    void* residual_v_ptr(int seq_id, int layer) const;

    // Layer-base pointers (pointer to slot 0's K or V data for the given
    // layer). Combined with `residual_seq_stride_bytes()` this lets a kernel
    // address per-batch-idx data: `K_for_slot_s = K_layer_base + s * stride`.
    // Returns nullptr if residual not enabled or layer out of range.
    void* residual_k_layer_base(int layer) const;
    void* residual_v_layer_base(int layer) const;

    // Stride (in bytes) between consecutive seq slots in the residual pool.
    // Same value for K and V — the (slot, layer, K|V) layout makes this
    // the per-slot row stride independent of layer.
    size_t residual_seq_stride_bytes() const { return residual_per_seq_bytes_; }

    // Device-resident ring state buffers ([max_seqs] ints each). Replaces the
    // per-step host upload of write_idx/fill_count — kernels read directly,
    // and an `advance_residual_state_kernel` updates them once per decode step
    // (graph-capture-safe). Zeroed on slot allocation and release.
    int* d_residual_widx_ptr() const { return d_residual_widx_; }
    int* d_residual_fc_ptr() const { return d_residual_fc_; }

    // ── Slot allocation (multi-seq batch support) ────────────────────
    //
    // Each active sequence holds one slot in [0, residual_max_seqs_) for
    // the duration of its decode. Allocation is FIFO from a free-list.
    // Returns the assigned slot, or -1 if residual not enabled / pool full.
    // Idempotent: re-allocating for the same seq returns the existing slot.
    int allocate_residual_slot(int seq_id);

    // Release a sequence's slot (no-op if not allocated). Called on
    // free_sequence; safe to call eagerly. Resets the ring state too.
    void release_residual_slot(int seq_id);

    // Look up a sequence's slot, or -1 if not allocated.
    int residual_slot_of(int seq_id) const;

    // Per-sequence ring state. Returns {0, 0} for unknown / out-of-range seq.
    ResidualRingState residual_state(int seq_id) const;

    // Advance the ring after a write at the current write_idx.
    // Increments write_idx (mod residual_n) and fill_count (capped at residual_n).
    void advance_residual(int seq_id);

    // Reset ring state for a sequence (called from free_sequence). Does NOT
    // release the slot — use release_residual_slot for that.
    void reset_residual(int seq_id);

    // ── Hashing utility (public for testing) ─────────────────────────

    // Compute the hash for a block of tokens. `parent_hash` is the hash
    // of the preceding block (0 for the first block). If the block has
    // fewer than kKVBlockSize tokens, it is NOT cacheable (partial block).
    static size_t compute_block_hash(std::span<const int32_t> tokens, size_t parent_hash);

private:
    // A sequence's positional block table. refs_[i] is the OWNING reference;
    // an empty ref is a hole, which StreamingLLM's evict_middle_blocks()
    // creates deliberately — the table LENGTH must not change, because the
    // attention kernels derive positional alignment from it.
    //
    // block_table() still hands out a plain vector<int> so the ~10 consumers
    // in runtime/ are untouched; it is derived from refs_ and rebuilt lazily,
    // so the two cannot drift.
    class SeqBlocks {
    public:
        size_t size() const { return refs_.size(); }
        bool empty() const { return refs_.empty(); }
        int id_at(size_t i) const { return i < refs_.size() && refs_[i] ? refs_[i].id() : -1; }
        int back_id() const { return refs_.empty() ? -1 : id_at(refs_.size() - 1); }

        void push(BlockRef r) {
            refs_.push_back(std::move(r));
            dirty_ = true;
        }
        void pop_back() {
            refs_.pop_back();
            dirty_ = true;
        }
        // Shrinking drops the references, which is what frees the blocks.
        void resize(size_t n) {
            refs_.resize(n);
            dirty_ = true;
        }
        // Move the reference out, leaving a hole. Used when ownership passes
        // to the prefix cache.
        BlockRef take(size_t i) {
            dirty_ = true;
            return i < refs_.size() ? std::move(refs_[i]) : BlockRef();
        }
        // Drop the reference, leaving a hole (StreamingLLM eviction).
        void make_hole(size_t i) {
            if (i < refs_.size())
                refs_[i].reset();
            dirty_ = true;
        }

        const std::vector<int>& ids() const {
            if (dirty_) {
                ids_.resize(refs_.size());
                for (size_t i = 0; i < refs_.size(); ++i)
                    ids_[i] = refs_[i] ? refs_[i].id() : -1;
                dirty_ = false;
            }
            return ids_;
        }

    private:
        std::vector<BlockRef> refs_;
        mutable std::vector<int> ids_;
        mutable bool dirty_ = true;
    };

    // Erase a block's prefix-hash entries when this is its last reference.
    // A stale hash->id entry on a block that is about to be free-listed lets
    // a later prefix match share a block the allocator still hands out.
    void drop_stale_hash_if_last(int block_id);

    // Rollback a partial block allocation: drop the references added after
    // original_size, trim the hashes vector, and erase the sequence entries
    // if empty.
    void rollback_partial_allocation(int seq_id, SeqBlocks& blocks, std::vector<size_t>& hashes,
                                     size_t original_size);

    // Underlying block-level cache (owns the memory pool).
    std::unique_ptr<KVCache> cache_;

    // seq_id -> ordered list of block ids.
    std::unordered_map<int, SeqBlocks> seq_blocks_;

    // ── SWA-aware sizing state ───────────────────────────────────────
    // seq_id -> positional SWA-group block table (parallel to seq_blocks_,
    // -1 holes outside the trailing window). Only populated when
    // enable_swa_sizing() armed the feature.
    std::unordered_map<int, std::vector<int>> seq_swa_blocks_;
    // seq_id -> first not-yet-trimmed SWA table index (amortizes swa_trim
    // to O(new dead blocks) per call instead of O(table)).
    std::unordered_map<int, int> swa_trim_cursor_;
    int swa_window_ = 0;  // 0 = SWA sizing disabled
    int swa_slack_ = 0;

    // ── SWA snapshot state (enable_swa_snapshots) ────────────────────
    std::vector<int> swa_snap_layers_;      // SWA layer indices, ascending
    std::vector<size_t> swa_snap_layer_off_;  // slab byte offset per dense idx
    int swa_snap_win_blocks_ = 0;           // max live blocks per layer
    size_t swa_snap_bytes_ = 0;             // total slab bytes
    KVCache::CopyDesc* h_swa_descs_ = nullptr;  // pinned staging
    KVCache::CopyDesc* d_swa_descs_ = nullptr;
    int swa_desc_cap_ = 0;
    // Build pack (to_slab=true) or restore (to_slab=false) descs for the
    // seq's SWA table slots [swa_first_live_block(upto), upto/bs) across
    // all SWA layers and run them in one launch.
    bool swa_snapshot_copy_(int seq_id, int upto_tokens, void* slab, bool to_slab,
                            cudaStream_t stream);

    // ── LRU tracking ─────────────────────────────────────────────────
    // Doubly-linked list of seq_ids; most recently used at the *tail*.
    std::list<int> lru_order_;
    // O(1) lookup from seq_id to its position in lru_order_.
    std::unordered_map<int, std::list<int>::iterator> lru_map_;

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
    // The cache's OWN reference to each entry, alongside its LRU position.
    // Before A7 step 3.2 the cache held nothing: free_sequence() deliberately
    // skipped the free and the block survived at refcount 1 — liveness by
    // omission. Erasing an entry now drops the reference, which is what
    // returns the block to the pool.
    struct CachedEntry {
        std::list<int>::iterator lru_it;
        BlockRef ref;
    };
    std::unordered_map<int, CachedEntry> cached_blocks_map_;

    // seq_id -> vector of block hashes (parallel to seq_blocks_).
    // Used to maintain hash chain state for append_block operations.
    std::unordered_map<int, std::vector<size_t>> seq_block_hashes_;

    // ── Prefix pinning state ─────────────────────────────────────────
    // Block IDs that are pinned and must never be evicted or freed.
    // Two writers: pin_prefix() (refcounted below) and evict_middle_blocks()
    // (StreamingLLM sinks, registered through pin_prefix as well).
    std::unordered_set<int> pinned_blocks_;
    // Pin owner -> exact block IDs it pinned. Survives free_sequence()
    // (seq_blocks_ is erased there; pins usually belong to finished seqs).
    std::unordered_map<int, std::vector<int>> pinned_seq_blocks_;
    // block_id -> number of pin owners. A block leaves pinned_blocks_
    // only when its last owner unpins.
    std::unordered_map<int, int> pin_refcount_;
    // Pin owners in pin order; budget eviction unpins from the front.
    std::list<int> pin_fifo_;
    // Cap on unique pinned blocks (0 = unlimited).
    int pin_budget_blocks_ = 0;

    // Incrementally maintained count of reclaimable cached blocks
    // (cached_blocks_lru_.size() minus pinned cached blocks).
    // Avoids O(C) linear scan in can_allocate().
    int reclaimable_cached_count_ = 0;

    // Try to reclaim a cached block. Returns the block_id, or -1.
    int reclaim_cached_block();

    // Internal: allocate a fresh block, reclaiming cached blocks if needed.
    [[nodiscard]] BlockRef allocate_block_ref_with_eviction();

    // ── Residual FP16 cache state ────────────────────────────────────
    // Single contiguous pool. Layout per element:
    //   pool[seq_slot, layer, k_or_v, ring_idx, kv_head, hd_elem]
    // where seq_slot ∈ [0, residual_max_seqs_), layer ∈ [0, n_layers),
    // k_or_v ∈ {0,1}, ring_idx ∈ [0, residual_n_tokens_).
    void* residual_pool_ = nullptr;
    VRAMAllocator* residual_alloc_ = nullptr;
    int residual_n_tokens_ = 0;
    int residual_max_seqs_ = 0;
    size_t residual_per_layer_bytes_ = 0;     // bytes per (seq, layer, K-or-V)
    size_t residual_per_seq_bytes_ = 0;       // bytes per seq across all layers
    int residual_n_layers_ = 0;               // cached from cache_->n_layers()
    int residual_n_kv_heads_ = 0;             // cached from cache_->n_kv_heads()
    int residual_head_dim_ = 0;               // cached from cache_->head_dim()
    std::unordered_map<int, ResidualRingState> seq_residual_state_;
    // ── Slot allocator (multi-seq batch) ─────────────────────────────
    // Free list of unused slot indices in [0, residual_max_seqs_). When
    // a sequence is admitted we pop one off the back (LIFO for cache-
    // friendliness); on release we push back. seq_slot_map_ tracks the
    // currently-assigned slot per active seq_id.
    std::vector<int> residual_free_slots_;
    std::unordered_map<int, int> residual_seq_slot_;
    // Device-resident ring state, [max_seqs] ints each. Zeroed at alloc.
    int* d_residual_widx_ = nullptr;
    int* d_residual_fc_ = nullptr;
};

}  // namespace imp
