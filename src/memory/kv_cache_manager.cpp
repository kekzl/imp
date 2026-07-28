#include "memory/kv_cache_manager.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <functional>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <utility>

namespace imp {

void KVCacheManager::free_block_dropping_stale_hash(int block_id) {
    if (block_id >= 0 && cache_->ref_count(block_id) == 1) {
        auto hit = block_id_to_hash_.find(block_id);
        if (hit != block_id_to_hash_.end()) {
            block_hash_to_id_.erase(hit->second);
            block_id_to_hash_.erase(hit);
        }
    }
    cache_->free_block(block_id);
}

void KVCacheManager::rollback_partial_allocation(int seq_id, std::vector<int>& blocks,
                                                 std::vector<size_t>& hashes, size_t original_size) {
    for (size_t j = original_size; j < blocks.size(); ++j) {
        free_block_dropping_stale_hash(blocks[j]);
    }
    blocks.resize(original_size);
    hashes.resize(original_size);
    if (blocks.empty()) {
        seq_blocks_.erase(seq_id);
        seq_block_hashes_.erase(seq_id);
    }
}

// ─── Construction / destruction ──────────────────────────────────────

KVCacheManager::KVCacheManager(std::unique_ptr<KVCache> cache) : cache_(std::move(cache)) {}

KVCacheManager::~KVCacheManager() {
    if (h_swa_descs_) {
        cudaFreeHost(h_swa_descs_);
        h_swa_descs_ = nullptr;
    }
    if (d_swa_descs_) {
        cudaFree(d_swa_descs_);
        d_swa_descs_ = nullptr;
    }
    if (residual_pool_ && residual_alloc_) {
        residual_alloc_->free(residual_pool_);
        residual_pool_ = nullptr;
    }
    if (d_residual_widx_) {
        cudaFree(d_residual_widx_);
        d_residual_widx_ = nullptr;
    }
    if (d_residual_fc_) {
        cudaFree(d_residual_fc_);
        d_residual_fc_ = nullptr;
    }
}

// ─── BitDecoding Phase 3: residual FP16 cache ────────────────────────

bool KVCacheManager::enable_residual_buffer(int max_seqs, int residual_n, VRAMAllocator* alloc) {
    if (residual_pool_) {
        IMP_LOG_WARN("KVCacheManager: residual buffer already enabled, ignoring re-enable");
        return false;
    }
    if (max_seqs <= 0 || residual_n <= 0 || alloc == nullptr) {
        return false;
    }

    residual_n_layers_   = cache_->n_layers();
    residual_n_kv_heads_ = cache_->n_kv_heads();
    residual_head_dim_   = cache_->head_dim();
    if (residual_n_kv_heads_ <= 0 || residual_head_dim_ <= 0) {
        IMP_LOG_WARN("KVCacheManager: residual buffer needs uniform per-layer head_dim — disabled (multi-head_dim model)");
        return false;
    }

    // Per (seq, layer, K|V) live region:
    //   residual_n × n_kv_heads × head_dim FP16 elems
    residual_per_layer_bytes_ = static_cast<size_t>(residual_n) *
                                residual_n_kv_heads_ * residual_head_dim_ * sizeof(__half);
    // Per (seq) across all layers, both K and V:
    residual_per_seq_bytes_ = static_cast<size_t>(residual_n_layers_) * 2 *
                              residual_per_layer_bytes_;
    size_t total = static_cast<size_t>(max_seqs) * residual_per_seq_bytes_;

    void* pool = alloc->allocate(total, "kv_residual");
    if (!pool) {
        IMP_LOG_ERROR("KVCacheManager: residual buffer allocation failed (%.2f MiB)",
                      static_cast<double>(total) / (1024.0 * 1024.0));
        return false;
    }

    residual_pool_ = pool;
    residual_alloc_ = alloc;
    residual_n_tokens_ = residual_n;
    residual_max_seqs_ = max_seqs;

    // Initialize the slot free list. LIFO push so slot 0 is allocated first.
    residual_free_slots_.clear();
    residual_free_slots_.reserve(max_seqs);
    for (int i = max_seqs - 1; i >= 0; i--) {
        residual_free_slots_.push_back(i);
    }
    residual_seq_slot_.clear();

    // Allocate device-resident ring state buffers (graph-capture-safe path).
    // Zero-initialized so that newly-allocated slots start with write_idx=0,
    // fill_count=0 without an extra reset call.
    const size_t state_bytes = static_cast<size_t>(max_seqs) * sizeof(int);
    if (cudaMalloc(&d_residual_widx_, state_bytes) != cudaSuccess ||
        cudaMalloc(&d_residual_fc_, state_bytes) != cudaSuccess) {
        IMP_LOG_ERROR("KVCacheManager: residual state buffer alloc failed (%zu bytes)", state_bytes);
        if (d_residual_widx_) { cudaFree(d_residual_widx_); d_residual_widx_ = nullptr; }
        if (d_residual_fc_) { cudaFree(d_residual_fc_); d_residual_fc_ = nullptr; }
        alloc->free(residual_pool_);
        residual_pool_ = nullptr;
        return false;
    }
    cudaMemset(d_residual_widx_, 0, state_bytes);
    cudaMemset(d_residual_fc_, 0, state_bytes);

    IMP_LOG_INFO("KVCacheManager: residual buffer enabled — max_seqs=%d, residual_n=%d, %.2f MiB",
                 max_seqs, residual_n, static_cast<double>(total) / (1024.0 * 1024.0));
    return true;
}

int KVCacheManager::allocate_residual_slot(int seq_id) {
    if (!residual_pool_) return -1;
    auto it = residual_seq_slot_.find(seq_id);
    if (it != residual_seq_slot_.end()) return it->second;
    if (residual_free_slots_.empty()) return -1;
    int slot = residual_free_slots_.back();
    residual_free_slots_.pop_back();
    residual_seq_slot_[seq_id] = slot;
    // Zero device-resident ring state for this slot. Synchronous — runs once
    // per request admission, not on the hot decode path.
    if (d_residual_widx_) {
        int zero = 0;
        cudaMemcpy(d_residual_widx_ + slot, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_residual_fc_ + slot, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }
    return slot;
}

void KVCacheManager::release_residual_slot(int seq_id) {
    if (!residual_pool_) return;
    auto it = residual_seq_slot_.find(seq_id);
    if (it == residual_seq_slot_.end()) return;
    int slot = it->second;
    residual_free_slots_.push_back(slot);
    residual_seq_slot_.erase(it);
    seq_residual_state_.erase(seq_id);
    if (d_residual_widx_) {
        int zero = 0;
        cudaMemcpy(d_residual_widx_ + slot, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_residual_fc_ + slot, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }
}

int KVCacheManager::residual_slot_of(int seq_id) const {
    auto it = residual_seq_slot_.find(seq_id);
    if (it == residual_seq_slot_.end()) return -1;
    return it->second;
}

void* KVCacheManager::residual_k_ptr(int seq_id, int layer) const {
    if (!residual_pool_) return nullptr;
    int slot = residual_slot_of(seq_id);
    if (slot < 0) return nullptr;
    if (layer < 0 || layer >= residual_n_layers_) return nullptr;
    auto* base = static_cast<char*>(residual_pool_);
    base += static_cast<size_t>(slot) * residual_per_seq_bytes_;
    base += static_cast<size_t>(layer) * 2 * residual_per_layer_bytes_;
    // K is k_or_v=0 → offset 0
    return base;
}

void* KVCacheManager::residual_v_ptr(int seq_id, int layer) const {
    if (!residual_pool_) return nullptr;
    int slot = residual_slot_of(seq_id);
    if (slot < 0) return nullptr;
    if (layer < 0 || layer >= residual_n_layers_) return nullptr;
    auto* base = static_cast<char*>(residual_pool_);
    base += static_cast<size_t>(slot) * residual_per_seq_bytes_;
    base += static_cast<size_t>(layer) * 2 * residual_per_layer_bytes_;
    base += residual_per_layer_bytes_;  // V is k_or_v=1
    return base;
}

void* KVCacheManager::residual_k_layer_base(int layer) const {
    if (!residual_pool_) return nullptr;
    if (layer < 0 || layer >= residual_n_layers_) return nullptr;
    auto* base = static_cast<char*>(residual_pool_);
    base += static_cast<size_t>(layer) * 2 * residual_per_layer_bytes_;
    return base;
}

void* KVCacheManager::residual_v_layer_base(int layer) const {
    if (!residual_pool_) return nullptr;
    if (layer < 0 || layer >= residual_n_layers_) return nullptr;
    auto* base = static_cast<char*>(residual_pool_);
    base += static_cast<size_t>(layer) * 2 * residual_per_layer_bytes_;
    base += residual_per_layer_bytes_;
    return base;
}

KVCacheManager::ResidualRingState KVCacheManager::residual_state(int seq_id) const {
    auto it = seq_residual_state_.find(seq_id);
    if (it == seq_residual_state_.end()) return {0, 0};
    return it->second;
}

void KVCacheManager::advance_residual(int seq_id) {
    if (!residual_pool_ || residual_n_tokens_ <= 0) return;
    auto& s = seq_residual_state_[seq_id];
    s.write_idx = (s.write_idx + 1) % residual_n_tokens_;
    if (s.fill_count < residual_n_tokens_) s.fill_count++;
}

void KVCacheManager::reset_residual(int seq_id) {
    seq_residual_state_.erase(seq_id);
}

// ─── Hashing utility ─────────────────────────────────────────────────

size_t KVCacheManager::compute_block_hash(std::span<const int32_t> tokens, size_t parent_hash) {
    // FNV-1a inspired hash that chains with the parent block's hash.
    // This ensures that block N's hash depends on all preceding blocks,
    // so two sequences must share an identical prefix to match.
    size_t hash = parent_hash ^ 0xcbf29ce484222325ULL;
    for (int32_t tok : tokens) {
        hash ^= static_cast<size_t>(static_cast<uint32_t>(tok));
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// ─── Sequence management ─────────────────────────────────────────────

bool KVCacheManager::allocate_blocks(int seq_id, int num_blocks) {
    if (num_blocks <= 0)
        return true;

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
    if (it == seq_blocks_.end())
        return -1;

    int block_id = allocate_block_with_eviction();
    if (block_id < 0)
        return -1;

    it->second.push_back(block_id);
    return block_id;
}

void KVCacheManager::free_sequence(int seq_id) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end())
        return;

    for (int block_id : it->second) {
        // -1 sentinel marks a slot whose physical block was freed by
        // evict_middle_blocks (StreamingLLM). Skip — nothing to free.
        if (block_id < 0)
            continue;
        // Pinned blocks survive free_sequence: keep ref_count=1, add to
        // cached LRU for reuse. They remain in pinned_blocks_ and cannot
        // be evicted until unpin_prefix() is called.
        if (pinned_blocks_.find(block_id) != pinned_blocks_.end()) {
            // Pinned blocks: keep alive with ref_count=1, add to cached LRU
            // so num_cached_blocks() reports them, but do NOT count as
            // reclaimable (pinned blocks cannot be evicted until unpinned).
            if (cached_blocks_map_.find(block_id) == cached_blocks_map_.end()) {
                // The cache does not own this block yet: hand it THIS
                // sequence's reference instead of dropping it. Same refcount
                // as before, now tracked by an owner rather than retained by
                // an omitted free.
                cached_blocks_lru_.push_back(block_id);
                cached_blocks_map_.emplace(
                    block_id, CachedEntry{std::prev(cached_blocks_lru_.end()),
                                          cache_->adopt_block(block_id)});
                // NOT incrementing reclaimable_cached_count_ — pinned blocks excluded
            } else {
                // Already owned by the cache — drop the sequence's reference.
                cache_->free_block(block_id);
            }
            continue;
        }

        if (prefix_caching_enabled_ && cache_->ref_count(block_id) == 1) {
            // This sequence is the last reference. Check if the block is
            // registered in the hash table for potential reuse.
            auto hash_it = block_id_to_hash_.find(block_id);
            if (hash_it != block_id_to_hash_.end()) {
                // Transfer this sequence's reference to the cache.
                if (cached_blocks_map_.find(block_id) == cached_blocks_map_.end()) {
                    cached_blocks_lru_.push_back(block_id);
                    cached_blocks_map_.emplace(
                        block_id, CachedEntry{std::prev(cached_blocks_lru_.end()),
                                              cache_->adopt_block(block_id)});
                    reclaimable_cached_count_++;
                } else {
                    cache_->free_block(block_id);
                }
                continue;
            }
        }

        // Normal free: decrement ref_count, return to pool if it hits 0.
        cache_->free_block(block_id);
    }

    seq_blocks_.erase(it);
    seq_block_hashes_.erase(seq_id);

    // SWA table: free every live SWA-group block (never hashed/pinned/shared).
    if (auto sit = seq_swa_blocks_.find(seq_id); sit != seq_swa_blocks_.end()) {
        for (int bid : sit->second)
            if (bid >= 0)
                cache_->free_swa_block(bid);
        seq_swa_blocks_.erase(sit);
        swa_trim_cursor_.erase(seq_id);
    }

    // Phase 3: reset residual ring state AND return the slot to the free list.
    // release_residual_slot is a no-op if no slot was allocated for this seq.
    release_residual_slot(seq_id);

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
    if (it == seq_blocks_.end())
        return empty;
    return it->second;
}

int KVCacheManager::num_free_blocks() const { return cache_->num_free_blocks(); }

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
    if (lru_order_.empty())
        return -1;

    // Skip pinned sequences — find the first unpinned LRU victim.
    //
    // NOTE: every sequence in lru_order_ is LIVE (free_sequence removes finished
    // ones), and imp has no recompute-on-resume path, so freeing a live
    // sequence's KV here corrupts it. The engine therefore no longer calls this
    // to make room under KV pressure (it reject-newests instead — see
    // prefill_allocate_kv_blocks_/step_decode/step_spec_verify). Kept as a
    // manager primitive + unit-tested; do NOT reintroduce engine-side eviction
    // of live sequences without a preempt-and-recompute path.
    for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
        int candidate = *it;
        if (pinned_seq_blocks_.find(candidate) != pinned_seq_blocks_.end())
            continue;
        free_sequence(candidate);  // also removes from lru_order_ / lru_map_
        return candidate;
    }

    return -1;  // All sequences are pinned.
}

bool KVCacheManager::can_allocate(int num_blocks) const {
    if (num_blocks <= 0)
        return true;

    // Fast path: free pool + reclaimable cached blocks (O(1) via counter)
    int reclaimable = cache_->num_free_blocks() + reclaimable_cached_count_;
    if (reclaimable >= num_blocks)
        return true;

    // Slow path: count blocks from evictable LRU sequences
    for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
        if (pinned_seq_blocks_.find(*it) != pinned_seq_blocks_.end())
            continue;
        auto seq_it = seq_blocks_.find(*it);
        if (seq_it != seq_blocks_.end()) {
            reclaimable += static_cast<int>(seq_it->second.size());
        }
        if (reclaimable >= num_blocks)
            return true;
    }
    return false;
}

// ─── Content-addressed prefix caching ────────────────────────────────

int KVCacheManager::longest_cached_prefix_blocks(std::span<const int32_t> tokens,
                                                 std::vector<size_t>& chain_hashes) const {
    chain_hashes.clear();
    const int num_tokens = static_cast<int>(tokens.size());
    const int full_blocks = num_tokens / cache_->block_size();
    int cached = 0;
    bool chain_intact = true;
    size_t parent_hash = 0;
    for (int b = 0; b < full_blocks; ++b) {
        size_t h = compute_block_hash(
            tokens.subspan(static_cast<size_t>(b) * cache_->block_size(), cache_->block_size()),
            parent_hash);
        chain_hashes.push_back(h);
        parent_hash = h;
        if (chain_intact && block_hash_to_id_.find(h) != block_hash_to_id_.end())
            cached = b + 1;
        else
            chain_intact = false;
    }
    return cached;
}

int KVCacheManager::allocate_blocks_with_prefix(int seq_id, std::span<const int32_t> tokens,
                                                int max_reuse_blocks) {
    const int num_tokens = static_cast<int>(tokens.size());
    if (num_tokens <= 0)
        return 0;

    int total_blocks = (num_tokens + cache_->block_size() - 1) / cache_->block_size();
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
    // Reuse must form a contiguous prefix: once a block misses (or the
    // caller's cap is reached), later hash hits must NOT be shared — the
    // caller skips prefill for reused*block_size tokens, so a hole would
    // leave uncomputed KV inside the "skipped" range (possible when LRU
    // eviction removed an early block while later chain blocks survive).
    bool reuse_open = true;

    for (int b = 0; b < total_blocks; ++b) {
        int block_start = b * cache_->block_size();
        int block_tokens = std::min(cache_->block_size(), num_tokens - block_start);

        // Only full blocks are cacheable.
        bool is_full_block = (block_tokens == cache_->block_size());

        if (max_reuse_blocks >= 0 && b >= max_reuse_blocks)
            reuse_open = false;

        if (prefix_caching_enabled_ && is_full_block) {
            size_t block_hash = compute_block_hash(tokens.subspan(block_start, block_tokens), parent_hash);

            // Check if this block already exists in the hash table.
            auto hit = reuse_open ? block_hash_to_id_.find(block_hash) : block_hash_to_id_.end();
            if (hit != block_hash_to_id_.end() && cache_->ref_count(hit->second) == 0 &&
                cached_blocks_map_.find(hit->second) == cached_blocks_map_.end()) {
                // Stale entry: the mapped block is free-listed (ref 0, not
                // cached). Reusing it would double-own the block. Drop the
                // entry loudly and treat as a miss.
                IMP_LOG_WARN("prefix cache: stale hash entry for free block %d — dropping", hit->second);
                block_id_to_hash_.erase(hit->second);
                block_hash_to_id_.erase(hit);
                hit = block_hash_to_id_.end();
            }
            if (hit != block_hash_to_id_.end()) {
                int cached_block = hit->second;

                // Remove from cached LRU if it was unreferenced.
                auto cached_it = cached_blocks_map_.find(cached_block);
                if (cached_it != cached_blocks_map_.end()) {
                    cached_blocks_lru_.erase(cached_it->second.lru_it);
                    // The reference moves from the cache to this sequence:
                    // relinquish without dropping, so the count stays 1.
                    (void)cached_it->second.ref.release();
                    cached_blocks_map_.erase(cached_it);
                    // Leaving the LRU means leaving the reclaimable count —
                    // pinned blocks were never counted to begin with.
                    if (pinned_blocks_.find(cached_block) == pinned_blocks_.end())
                        reclaimable_cached_count_--;
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

            // No cache hit (or reuse closed) — allocate a fresh block.
            reuse_open = false;
            int block_id = allocate_block_with_eviction();
            if (block_id < 0) {
                // Rollback everything we allocated/shared in this call.
                rollback_partial_allocation(seq_id, blocks, hashes, original_size);
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
                rollback_partial_allocation(seq_id, blocks, hashes, original_size);
                return -1;
            }

            blocks.push_back(block_id);
            if (is_full_block) {
                size_t block_hash = compute_block_hash(tokens.subspan(block_start, block_tokens),
                                                       parent_hash);
                hashes.push_back(block_hash);
                parent_hash = block_hash;
            } else {
                hashes.push_back(0);  // Partial block, not cacheable.
            }
        }
    }

    // Track in LRU.
    if (lru_map_.find(seq_id) == lru_map_.end()) {
        lru_order_.push_back(seq_id);
        lru_map_[seq_id] = std::prev(lru_order_.end());
    }

    if (reused_blocks > 0) {
        IMP_LOG_DEBUG("PrefixCache: seq %d reused %d/%d blocks (%d tokens skippable)", seq_id, reused_blocks,
                      total_blocks, reused_blocks * cache_->block_size());
    }
    return reused_blocks;
}

void KVCacheManager::register_block_hashes(int seq_id, std::span<const int32_t> tokens) {
    const int num_tokens = static_cast<int>(tokens.size());
    if (!prefix_caching_enabled_)
        return;

    auto blocks_it = seq_blocks_.find(seq_id);
    if (blocks_it == seq_blocks_.end())
        return;

    const auto& blocks = blocks_it->second;
    int total_blocks = static_cast<int>(blocks.size());

    auto& hashes = seq_block_hashes_[seq_id];
    size_t parent_hash = 0;

    for (int b = 0; b < total_blocks; ++b) {
        int block_start = b * cache_->block_size();
        int block_tokens = std::min(cache_->block_size(), num_tokens - block_start);

        if (block_tokens < cache_->block_size())
            break;  // Only full blocks are cacheable.

        // Skip recomputation if hash was already computed during allocate_blocks_with_prefix
        int block_id = blocks[b];
        if (b < static_cast<int>(hashes.size()) && hashes[b] != 0) {
            size_t existing_hash = hashes[b];
            if (block_hash_to_id_.find(existing_hash) == block_hash_to_id_.end()) {
                block_hash_to_id_[existing_hash] = block_id;
                block_id_to_hash_[block_id] = existing_hash;
            }
            parent_hash = existing_hash;
            continue;
        }

        size_t block_hash = compute_block_hash(tokens.subspan(block_start, block_tokens), parent_hash);

        if (b < static_cast<int>(hashes.size())) {
            hashes[b] = block_hash;
        } else {
            hashes.push_back(block_hash);
        }

        if (block_hash_to_id_.find(block_hash) == block_hash_to_id_.end()) {
            block_hash_to_id_[block_hash] = block_id;
            block_id_to_hash_[block_id] = block_hash;
        }

        parent_hash = block_hash;
    }
}

int KVCacheManager::num_cached_blocks() const { return static_cast<int>(cached_blocks_lru_.size()); }

bool KVCacheManager::evict_cached_block() {
    int block_id = reclaim_cached_block();
    return block_id >= 0;
}

int KVCacheManager::reclaim_cached_block() {
    if (reclaimable_cached_count_ <= 0)
        return -1;

    // Skip pinned blocks at the front of the LRU. Bounded by the list size
    // so a drifted reclaimable count can never spin on a pinned-only LRU.
    for (size_t n = cached_blocks_lru_.size(); n > 0; --n) {
        if (pinned_blocks_.find(cached_blocks_lru_.front()) == pinned_blocks_.end())
            break;
        // Move pinned block to the back so we don't re-scan it
        cached_blocks_lru_.splice(cached_blocks_lru_.end(), cached_blocks_lru_, cached_blocks_lru_.begin());
    }
    if (cached_blocks_lru_.empty() ||
        pinned_blocks_.find(cached_blocks_lru_.front()) != pinned_blocks_.end())
        return -1;

    int block_id = cached_blocks_lru_.front();
    cached_blocks_lru_.pop_front();
    reclaimable_cached_count_--;

    // Remove from hash tables.
    auto hash_it = block_id_to_hash_.find(block_id);
    if (hash_it != block_id_to_hash_.end()) {
        block_hash_to_id_.erase(hash_it->second);
        block_id_to_hash_.erase(hash_it);
    }

    // Dropping the cache's reference is what returns the block to the pool —
    // there is no separate free_block() call any more.
    cached_blocks_map_.erase(block_id);
    return block_id;
}

int KVCacheManager::allocate_block_with_eviction() {
    int block_id = cache_->allocate_block();
    if (block_id >= 0)
        return block_id;

    // Try reclaiming cached blocks (cheaper than evicting a sequence).
    // Loop until we get a usable block or exhaust all reclaimable cached blocks.
    while (!cached_blocks_lru_.empty()) {
        if (reclaim_cached_block() < 0)
            break;  // All remaining cached blocks are pinned.
        block_id = cache_->allocate_block();
        if (block_id >= 0)
            return block_id;
    }

    return -1;  // Caller should try evict_lru() if needed.
}

// ─── Prefix block pinning ────────────────────────────────────────

void KVCacheManager::pin_prefix(int seq_id, int num_blocks) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end())
        return;

    int to_pin = std::min(num_blocks, static_cast<int>(it->second.size()));
    if (to_pin <= 0)
        return;

    // Re-pinning replaces the owner's previous pin set.
    if (pinned_seq_blocks_.contains(seq_id))
        unpin_prefix(seq_id);

    // Budget: cap the request to the budget, then unpin the oldest owners
    // (FIFO) until the new pin fits. Evicted pins degrade to normal cached
    // blocks — still reusable, just no longer eviction-protected.
    if (pin_budget_blocks_ > 0) {
        to_pin = std::min(to_pin, pin_budget_blocks_);
        while (static_cast<int>(pin_refcount_.size()) + to_pin > pin_budget_blocks_ &&
               !pin_fifo_.empty()) {
            unpin_prefix(pin_fifo_.front());
        }
    }

    const auto& blocks = it->second;
    std::vector<int> owned;
    owned.reserve(to_pin);
    for (int i = 0; i < to_pin; ++i) {
        int bid = blocks[i];
        if (bid < 0)
            continue;  // StreamingLLM sentinel — physical block already freed
        bool already_pinned = pinned_blocks_.contains(bid);
        if (++pin_refcount_[bid] == 1 && !already_pinned) {
            pinned_blocks_.insert(bid);
            // Pinned blocks stay in the cached LRU (if there) so reuse and
            // reporting keep working, but they are excluded from the
            // reclaimable count; reclaim_cached_block() rotates past them.
            if (cached_blocks_map_.contains(bid))
                reclaimable_cached_count_--;
        }
        owned.push_back(bid);
    }
    if (owned.empty())
        return;

    int n_owned = static_cast<int>(owned.size());
    pinned_seq_blocks_[seq_id] = std::move(owned);
    pin_fifo_.push_back(seq_id);

    IMP_LOG_DEBUG("PinPrefix: seq %d pinned %d blocks (%zu unique pinned total)", seq_id, n_owned,
                  pin_refcount_.size());
}

void KVCacheManager::unpin_prefix(int seq_id) {
    auto it = pinned_seq_blocks_.find(seq_id);
    if (it == pinned_seq_blocks_.end())
        return;

    for (int bid : it->second) {
        auto rc = pin_refcount_.find(bid);
        if (rc == pin_refcount_.end())
            continue;
        if (--rc->second > 0)
            continue;  // still pinned by another owner
        pin_refcount_.erase(rc);
        pinned_blocks_.erase(bid);
        // While pinned the block was excluded from the reclaimable count;
        // if it sits in the cached LRU it is reclaimable again now. Blocks
        // currently referenced by an active seq are not in the LRU — their
        // free_sequence() takes the normal hashed-block path later.
        if (cached_blocks_map_.contains(bid))
            reclaimable_cached_count_++;
    }

    pinned_seq_blocks_.erase(it);
    pin_fifo_.remove(seq_id);
    IMP_LOG_DEBUG("UnpinPrefix: seq %d unpinned (%zu unique pinned remain)", seq_id,
                  pin_refcount_.size());
}

int KVCacheManager::num_pinned_blocks() const { return static_cast<int>(pinned_blocks_.size()); }

// ─── SWA-aware sizing (kv_cache.swa_sizing) ──────────────────────────

void KVCacheManager::enable_swa_sizing(int window_tokens, int slack_tokens) {
    if (!cache_->swa_enabled() || window_tokens <= 0) {
        IMP_LOG_WARN("KVCacheManager: enable_swa_sizing ignored (cache swa group %s, window=%d)",
                     cache_->swa_enabled() ? "present" : "absent", window_tokens);
        return;
    }
    swa_window_ = window_tokens;
    // Slack floor of one block covers the partially-filled boundary block;
    // the caller adds the spec-decode rollback depth on top.
    swa_slack_ = std::max(slack_tokens, cache_->block_size());
    IMP_LOG_INFO("KVCacheManager: SWA sizing enabled (window=%d, slack=%d, group=%d blocks)",
                 swa_window_, swa_slack_, cache_->swa_total_blocks());
}

bool KVCacheManager::swa_prepare(int seq_id, int from_tokens, int upto_tokens) {
    if (swa_window_ <= 0 || upto_tokens <= 0)
        return true;
    auto git = seq_blocks_.find(seq_id);
    if (git == seq_blocks_.end())
        return true;  // no global blocks yet — nothing to mirror
    auto& swa = seq_swa_blocks_[seq_id];
    const int bs = cache_->block_size();
    // Pad to the global table's length: new slots start as holes; only the
    // live tail below gets physical blocks.
    if (swa.size() < git->second.size())
        swa.resize(git->second.size(), -1);

    const long long live_start_tok =
        static_cast<long long>(std::min(from_tokens, upto_tokens)) - swa_window_ - swa_slack_;
    const int first_live = live_start_tok > 0 ? static_cast<int>(live_start_tok / bs) : 0;
    const int end_block = std::min(static_cast<int>((upto_tokens + bs - 1) / bs),
                                   static_cast<int>(swa.size()));
    for (int b = first_live; b < end_block; ++b) {
        if (swa[b] >= 0)
            continue;
        int id = cache_->allocate_swa_block();
        if (id < 0) {
            IMP_LOG_ERROR(
                "KVCacheManager: SWA block group exhausted (seq %d, block %d/%d, 0 free) — "
                "group undersized for the live span",
                seq_id, b, end_block);
            return false;
        }
        swa[b] = id;
    }
    return true;
}

void KVCacheManager::swa_trim(int seq_id, int committed_tokens) {
    if (swa_window_ <= 0)
        return;
    auto it = seq_swa_blocks_.find(seq_id);
    if (it == seq_swa_blocks_.end())
        return;
    auto& swa = it->second;
    const int bs = cache_->block_size();
    const long long dead_end_tok =
        static_cast<long long>(committed_tokens) - swa_window_ - swa_slack_;
    if (dead_end_tok <= 0)
        return;
    // Block b is dead when it ends at or before dead_end_tok: (b+1)*bs <= dead_end.
    const int dead_blocks = static_cast<int>(
        std::min<long long>(dead_end_tok / bs, static_cast<long long>(swa.size())));
    int& cursor = swa_trim_cursor_[seq_id];
    for (int b = cursor; b < dead_blocks; ++b) {
        if (swa[b] >= 0) {
            cache_->free_swa_block(swa[b]);
            swa[b] = -1;
        }
    }
    cursor = std::max(cursor, dead_blocks);
}

const std::vector<int>& KVCacheManager::swa_block_table(int seq_id) const {
    static const std::vector<int> empty;
    auto it = seq_swa_blocks_.find(seq_id);
    if (it == seq_swa_blocks_.end())
        return empty;
    return it->second;
}

// ─── SWA window snapshots (prefix caching for SWA-sized models) ──────

bool KVCacheManager::enable_swa_snapshots() {
    if (swa_window_ <= 0 || !cache_->swa_enabled())
        return false;
    if (swa_snap_bytes_ > 0)
        return true;  // already enabled
    const int bs = cache_->block_size();
    swa_snap_layers_.clear();
    for (int l = 0; l < cache_->n_layers(); ++l)
        if (cache_->layer_is_swa(l))
            swa_snap_layers_.push_back(l);
    if (swa_snap_layers_.empty())
        return false;
    // Max live blocks per layer at any block-aligned position: the trailing
    // window plus slack, ceil-aligned, plus the partial boundary block.
    swa_snap_win_blocks_ = (swa_window_ + swa_slack_ + bs - 1) / bs + 1;
    // Slab layout per dense layer j: [K slots][V slots][K scale][V scale],
    // each region swa_snap_win_blocks_ entries of the layer's byte sizes.
    swa_snap_layer_off_.clear();
    size_t off = 0;
    for (int l : swa_snap_layers_) {
        swa_snap_layer_off_.push_back(off);
        const size_t kvb = cache_->block_bytes(l);
        const size_t scb = cache_->k_scale_ptr(l, 0) ? cache_->scale_block_bytes(l) : 0;
        off += static_cast<size_t>(swa_snap_win_blocks_) * 2 * (kvb + scb);
    }
    swa_snap_bytes_ = off;
    const int cap =
        static_cast<int>(swa_snap_layers_.size()) * swa_snap_win_blocks_ * 4;
    if (cudaMallocHost(reinterpret_cast<void**>(&h_swa_descs_),
                       static_cast<size_t>(cap) * sizeof(KVCache::CopyDesc)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_swa_descs_),
                   static_cast<size_t>(cap) * sizeof(KVCache::CopyDesc)) != cudaSuccess) {
        IMP_LOG_WARN("KVCacheManager: SWA snapshot desc alloc failed — snapshots disabled");
        if (h_swa_descs_) {
            cudaFreeHost(h_swa_descs_);
            h_swa_descs_ = nullptr;
        }
        swa_snap_bytes_ = 0;
        return false;
    }
    swa_desc_cap_ = cap;
    IMP_LOG_INFO("KVCacheManager: SWA snapshots enabled (%zu KiB/snapshot, %d layers x %d blocks)",
                 swa_snap_bytes_ >> 10, static_cast<int>(swa_snap_layers_.size()),
                 swa_snap_win_blocks_);
    return true;
}

int KVCacheManager::swa_first_live_block(int upto_tokens) const {
    const int bs = cache_->block_size();
    const long long dead = static_cast<long long>(upto_tokens) - swa_window_ - swa_slack_;
    return dead > 0 ? static_cast<int>(dead / bs) : 0;
}

bool KVCacheManager::swa_snapshot_copy_(int seq_id, int upto_tokens, void* slab, bool to_slab,
                                        cudaStream_t stream) {
    const int bs = cache_->block_size();
    if (upto_tokens <= 0 || upto_tokens % bs != 0)
        return false;
    const int end_b = upto_tokens / bs;
    const int first = swa_first_live_block(upto_tokens);
    const int count = end_b - first;
    if (count <= 0 || count > swa_snap_win_blocks_)
        return false;
    auto it = seq_swa_blocks_.find(seq_id);
    if (it == seq_swa_blocks_.end() || static_cast<int>(it->second.size()) < end_b)
        return false;
    const auto& swa = it->second;
    // Generation-end saves pack at the block-FLOOR of the live context, so
    // the lowest slack blocks may already be trimmed. Restore-time queries
    // at positions >= upto never read below floor((upto - window)/bs); keep
    // one extra boundary block (the #963 floor/ceil lesson) and zero-fill
    // tolerated holes below that — a masked position contributes exactly 0
    // regardless of KV bytes, and zeros can never produce NaN scores.
    const int needed_start =
        std::max(first, (upto_tokens - swa_window_) / bs - 1);
    bool zero_filled = false;
    if (to_slab) {
        for (int b = first; b < needed_start; ++b) {
            if (swa[b] < 0) {
                IMP_CUDA_CHECK_LOG(cudaMemsetAsync(slab, 0, swa_snap_bytes_, stream));
                zero_filled = true;
                break;
            }
        }
    }
    int n = 0;
    char* slab_c = static_cast<char*>(slab);
    for (size_t j = 0; j < swa_snap_layers_.size(); ++j) {
        const int l = swa_snap_layers_[j];
        const size_t kvb = cache_->block_bytes(l);
        const size_t scb = cache_->k_scale_ptr(l, 0) ? cache_->scale_block_bytes(l) : 0;
        char* base = slab_c + swa_snap_layer_off_[j];
        char* k_area = base;
        char* v_area = base + static_cast<size_t>(swa_snap_win_blocks_) * kvb;
        char* ks_area = v_area + static_cast<size_t>(swa_snap_win_blocks_) * kvb;
        char* vs_area = ks_area + static_cast<size_t>(swa_snap_win_blocks_) * scb;
        for (int i = 0; i < count; ++i) {
            const int id = swa[first + i];
            if (id < 0) {
                if (to_slab && zero_filled && first + i < needed_start)
                    continue;  // tolerated trimmed slack block — stays zero
                return false;  // hole inside the read-relevant window
            }
            auto put = [&](void* cache_ptr, char* area, size_t bytes) {
                if (!bytes)
                    return;
                KVCache::CopyDesc& d = h_swa_descs_[n++];
                char* slab_ptr = area + static_cast<size_t>(i) * bytes;
                d.src = to_slab ? cache_ptr : static_cast<void*>(slab_ptr);
                d.dst = to_slab ? static_cast<void*>(slab_ptr) : cache_ptr;
                d.bytes = bytes;
            };
            put(cache_->k_ptr(l, id), k_area, kvb);
            put(cache_->v_ptr(l, id), v_area, kvb);
            if (scb) {
                put(cache_->k_scale_ptr(l, id), ks_area, scb);
                put(cache_->v_scale_ptr(l, id), vs_area, scb);
            }
        }
    }
    if (n == 0 || n > swa_desc_cap_)
        return false;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_swa_descs_, h_swa_descs_,
                                       static_cast<size_t>(n) * sizeof(KVCache::CopyDesc),
                                       cudaMemcpyHostToDevice, stream));
    KVCache::batched_copy_device(d_swa_descs_, n, stream);
    return true;
}

bool KVCacheManager::swa_snapshot_pack(int seq_id, int upto_tokens, void* slab,
                                       cudaStream_t stream) {
    if (swa_snap_bytes_ == 0 || !slab)
        return false;
    return swa_snapshot_copy_(seq_id, upto_tokens, slab, /*to_slab=*/true, stream);
}

bool KVCacheManager::swa_snapshot_restore(int seq_id, int upto_tokens, const void* slab,
                                          cudaStream_t stream) {
    if (swa_snap_bytes_ == 0 || !slab)
        return false;
    const int bs = cache_->block_size();
    if (upto_tokens <= 0 || upto_tokens % bs != 0)
        return false;
    const int end_b = upto_tokens / bs;
    auto git = seq_blocks_.find(seq_id);
    if (git == seq_blocks_.end() || static_cast<int>(git->second.size()) < end_b)
        return false;
    auto& swa = seq_swa_blocks_[seq_id];
    if (swa.size() < git->second.size())
        swa.resize(git->second.size(), -1);
    const int first = swa_first_live_block(upto_tokens);
    std::vector<int> fresh;
    fresh.reserve(static_cast<size_t>(end_b - first));
    for (int b = first; b < end_b; ++b) {
        if (swa[b] >= 0)
            continue;  // already prepared (shouldn't happen on a fresh seq)
        int id = cache_->allocate_swa_block();
        if (id < 0) {
            for (int f : fresh)
                cache_->free_swa_block(f);
            for (int b2 = first; b2 < end_b; ++b2)
                if (std::find(fresh.begin(), fresh.end(), swa[b2]) != fresh.end())
                    swa[b2] = -1;
            IMP_LOG_WARN("KVCacheManager: SWA snapshot restore failed (group exhausted) — "
                         "seq %d falls back to full prefill",
                         seq_id);
            return false;
        }
        swa[b] = id;
        fresh.push_back(id);
    }
    int& cursor = swa_trim_cursor_[seq_id];
    cursor = std::max(cursor, first);
    if (!swa_snapshot_copy_(seq_id, upto_tokens, const_cast<void*>(slab), /*to_slab=*/false,
                            stream)) {
        for (int b = first; b < end_b; ++b) {
            if (swa[b] >= 0 && std::find(fresh.begin(), fresh.end(), swa[b]) != fresh.end()) {
                cache_->free_swa_block(swa[b]);
                swa[b] = -1;
            }
        }
        return false;
    }
    return true;
}

// ─── Speculative decoding rollback ───────────────────────────────────

int KVCacheManager::evict_middle_blocks(int seq_id, int n_sink_tokens, int n_window_tokens) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end())
        return 0;
    if (n_sink_tokens <= 0 || n_window_tokens <= 0)
        return 0;

    auto& blocks = it->second;
    const int block_size = cache_->block_size();
    const int total_blocks = static_cast<int>(blocks.size());
    if (total_blocks == 0)
        return 0;

    const int sink_end_block = (n_sink_tokens + block_size - 1) / block_size;
    const int window_block_count = (n_window_tokens + block_size - 1) / block_size;
    // Retain ONE extra boundary block beyond the ceil-aligned window (#963):
    // the paged decode kernels compute their window start as
    // floor((ctx_len - window) / block_size), which for non-block-aligned
    // ctx_len (every decode step after an aligned prefill) lands one block
    // BEFORE ceil-aligned tail retention — the kernels then read a -1
    // sentinel, and phys_block = -1 is an out-of-bounds KV access (IMA on a
    // full-VRAM card, silent garbage attention otherwise). One 32-token
    // block of extra KV is the price of keeping host eviction and device
    // window math aligned regardless of ctx alignment or call ordering.
    const int window_start_block = std::max(0, total_blocks - window_block_count - 1);

    if (sink_end_block >= window_start_block) {
        // Sinks and window overlap (sequence too short) — nothing to free.
        return 0;
    }

    // Pin sink blocks so LRU / cached-block eviction never touches them
    // while the sequence is alive. Same owner bookkeeping as cache_control
    // pins; pin_prefix replaces the owner's pin set, so only re-pin when
    // the sink range grew.
    auto pit = pinned_seq_blocks_.find(seq_id);
    int already = (pit != pinned_seq_blocks_.end()) ? static_cast<int>(pit->second.size()) : 0;
    if (sink_end_block > already)
        pin_prefix(seq_id, sink_end_block);

    // Free middle blocks and replace with sentinel.
    int freed = 0;
    for (int i = sink_end_block; i < window_start_block; ++i) {
        int bid = blocks[i];
        if (bid < 0)
            continue;  // already evicted
        // Drop from prefix-hash table — chain is broken anyway.
        auto hit = block_id_to_hash_.find(bid);
        if (hit != block_id_to_hash_.end()) {
            block_hash_to_id_.erase(hit->second);
            block_id_to_hash_.erase(hit);
        }
        cache_->free_block(bid);
        blocks[i] = -1;
        ++freed;
    }

    // Invalidate the prefix-hash chain for window blocks too — once a gap
    // exists, downstream blocks can no longer be reused as a prefix.
    auto hash_it = seq_block_hashes_.find(seq_id);
    if (hash_it != seq_block_hashes_.end()) {
        auto& hashes = hash_it->second;
        for (int i = sink_end_block; i < total_blocks && i < static_cast<int>(hashes.size()); ++i) {
            // Clear hash entry for window blocks too: their original hash
            // chained through the now-freed middle.
            int bid = (i < static_cast<int>(blocks.size())) ? blocks[i] : -1;
            if (bid >= 0) {
                auto hit = block_id_to_hash_.find(bid);
                if (hit != block_id_to_hash_.end()) {
                    block_hash_to_id_.erase(hit->second);
                    block_id_to_hash_.erase(hit);
                }
            }
            hashes[i] = 0;
        }
    }

    return freed;
}

void KVCacheManager::rollback(int seq_id, int new_seq_len) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end())
        return;

    auto& blocks = it->second;
    if (blocks.empty() || new_seq_len < 0)
        return;

    // Keep exactly ceil(new_seq_len / cache_->block_size()) blocks.
    // A partially-filled last block is retained (its stale slots
    // will be overwritten on subsequent writes).
    int blocks_needed = (new_seq_len + cache_->block_size() - 1) / cache_->block_size();
    while (static_cast<int>(blocks.size()) > blocks_needed) {
        // -1 sentinels (StreamingLLM evicted middle slots) need no free.
        // Hash-registered blocks must leave the prefix-hash table when this
        // free drops them to ref 0 (stale entry = double-ownership bug).
        if (blocks.back() >= 0)
            free_block_dropping_stale_hash(blocks.back());
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

    // SWA table: drop the tail in lockstep (trimmed holes stay holes — the
    // slack guarantees anything a post-rollback forward reads is still live).
    if (auto sit = seq_swa_blocks_.find(seq_id); sit != seq_swa_blocks_.end()) {
        auto& swa = sit->second;
        while (static_cast<int>(swa.size()) > blocks_needed) {
            if (swa.back() >= 0)
                cache_->free_swa_block(swa.back());
            swa.pop_back();
        }
        if (auto cit = swa_trim_cursor_.find(seq_id); cit != swa_trim_cursor_.end())
            cit->second = std::min(cit->second, blocks_needed);
    }
}

// ─── Stats ───────────────────────────────────────────────────────────

KVCacheStats KVCacheManager::stats() const {
    KVCacheStats s;
    s.active_sequences = static_cast<int>(seq_blocks_.size());
    for (const auto& [seq_id, blocks] : seq_blocks_) {
        s.total_blocks += static_cast<int>(blocks.size());
    }
    s.free_blocks = cache_->num_free_blocks();
    s.cached_blocks = static_cast<int>(cached_blocks_lru_.size());
    s.pinned_blocks = static_cast<int>(pinned_blocks_.size());
    return s;
}

int KVCacheManager::num_active_sequences() const { return static_cast<int>(seq_blocks_.size()); }

int KVCacheManager::total_allocated_blocks() const {
    int total = 0;
    for (const auto& [seq_id, blocks] : seq_blocks_) {
        total += static_cast<int>(blocks.size());
    }
    return total;
}

// ─── Persistent prefix cache ────────────────────────────────────────

// Binary format:
//   Header: magic(4) version(4) n_blocks(4) n_layers(4) n_kv_heads(4)
//           head_dim(4) dtype(4) block_bytes(8) model_fingerprint(8)
//   Per block: hash(8) + KV data (n_layers * 2 * block_bytes)
//
// model_fingerprint (v2): identifies the model+tokenizer+quant that produced
// the KV. Block hashes are content-addressed over token IDs ONLY — two
// different models with identical KV geometry (common across same-family
// fine-tunes) would otherwise match each other's token-hashes and serve the
// WRONG model's KV silently. The geometry checks below cannot catch that.
// Rejecting on fingerprint mismatch degrades to a clean recompute.

static constexpr uint32_t kPrefixCacheMagic = 0x494D5043;  // "IMPC"
static constexpr uint32_t kPrefixCacheVersion = 2;

struct PrefixCacheHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_blocks;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t dtype;
    uint64_t block_bytes;
    uint64_t model_fingerprint;
};

int KVCacheManager::save_prefix_cache(const std::string& path, uint64_t model_fingerprint,
                                      cudaStream_t stream) {
    if (!prefix_caching_enabled_ || cached_blocks_lru_.empty()) {
        IMP_LOG_INFO("Prefix cache: nothing to save (0 cached blocks)");
        return 0;
    }

    // 0600: the snapshot holds raw KV blocks (conversation contents) — keep it
    // out of reach of other local users.
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE* f = (fd >= 0) ? fdopen(fd, "wb") : nullptr;
    if (!f) {
        if (fd >= 0)
            ::close(fd);
        IMP_LOG_ERROR("Failed to open %s for writing", path.c_str());
        return -1;
    }

    int n_blocks = static_cast<int>(cached_blocks_lru_.size());
    size_t bb = cache_->block_bytes();
    int nl = cache_->n_layers();

    PrefixCacheHeader hdr = {};
    hdr.magic = kPrefixCacheMagic;
    hdr.version = kPrefixCacheVersion;
    hdr.n_blocks = static_cast<uint32_t>(n_blocks);
    hdr.n_layers = static_cast<uint32_t>(nl);
    hdr.n_kv_heads = static_cast<uint32_t>(cache_->n_kv_heads());
    hdr.head_dim = static_cast<uint32_t>(cache_->head_dim());
    hdr.dtype = std::to_underlying(cache_->qtype());
    hdr.block_bytes = bb;
    hdr.model_fingerprint = model_fingerprint;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        IMP_LOG_ERROR("Failed to write prefix cache header to %s", path.c_str());
        fclose(f);
        return -1;
    }

    // Allocate host buffer for ALL blocks' KV data so we can pipeline
    // all D2H transfers with cudaMemcpyAsync and sync once.
    size_t per_block_total = static_cast<size_t>(nl) * 2 * bb;

    // First pass: collect valid block IDs and their hashes.
    struct BlockEntry {
        int block_id;
        size_t hash;
    };
    std::vector<BlockEntry> entries;
    entries.reserve(n_blocks);
    for (int block_id : cached_blocks_lru_) {
        auto hash_it = block_id_to_hash_.find(block_id);
        if (hash_it == block_id_to_hash_.end())
            continue;
        entries.push_back({block_id, hash_it->second});
    }

    if (entries.empty()) {
        fclose(f);
        IMP_LOG_INFO("Prefix cache: nothing to save (0 valid cached blocks)");
        return 0;
    }

    // Allocate a contiguous host buffer for all valid blocks.
    std::vector<uint8_t> host_buf(entries.size() * per_block_total);

    // Issue all D2H copies asynchronously.
    std::vector<bool> block_ok(entries.size(), true);
    for (size_t bi = 0; bi < entries.size(); ++bi) {
        int block_id = entries[bi].block_id;
        size_t buf_offset = bi * per_block_total;
        size_t offset = 0;
        for (int l = 0; l < nl; l++) {
            cudaError_t err = cudaMemcpyAsync(host_buf.data() + buf_offset + offset,
                                              cache_->k_ptr(l, block_id), bb, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed for block %d layer %d K: %s", block_id, l,
                              cudaGetErrorString(err));
                block_ok[bi] = false;
                break;
            }
            offset += bb;
            err = cudaMemcpyAsync(host_buf.data() + buf_offset + offset, cache_->v_ptr(l, block_id), bb,
                                  cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed for block %d layer %d V: %s", block_id, l,
                              cudaGetErrorString(err));
                block_ok[bi] = false;
                break;
            }
            offset += bb;
        }
    }

    // Single sync to wait for all D2H transfers before writing to disk.
    cudaStreamSynchronize(stream);

    int saved = 0;
    for (size_t bi = 0; bi < entries.size(); ++bi) {
        if (!block_ok[bi])
            continue;  // Skip blocks with failed copies.

        size_t hash = entries[bi].hash;
        if (fwrite(&hash, sizeof(hash), 1, f) != 1) {
            IMP_LOG_ERROR("Failed to write block hash to %s", path.c_str());
            fclose(f);
            return -1;
        }
        if (fwrite(host_buf.data() + bi * per_block_total, per_block_total, 1, f) != 1) {
            IMP_LOG_ERROR("Failed to write block data to %s", path.c_str());
            fclose(f);
            return -1;
        }
        saved++;
    }

    fclose(f);
    IMP_LOG_INFO("Prefix cache: saved %d blocks (%.1f MiB) to %s", saved,
                 (sizeof(hdr) + saved * (8 + per_block_total)) / (1024.0 * 1024.0), path.c_str());
    return saved;
}

int KVCacheManager::load_prefix_cache(const std::string& path, uint64_t model_fingerprint,
                                      cudaStream_t stream) {
    if (!prefix_caching_enabled_) {
        IMP_LOG_WARN("Prefix cache: loading disabled (prefix caching not enabled)");
        return -1;
    }

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        IMP_LOG_INFO("Prefix cache: no cache file at %s", path.c_str());
        return 0;
    }

    PrefixCacheHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != kPrefixCacheMagic) {
        IMP_LOG_WARN("Prefix cache: invalid header in %s", path.c_str());
        fclose(f);
        return -1;
    }

    if (hdr.version != kPrefixCacheVersion) {
        IMP_LOG_WARN("Prefix cache: version mismatch (%u vs %u)", hdr.version, kPrefixCacheVersion);
        fclose(f);
        return -1;
    }

    // Validate config matches current KV cache
    if (hdr.n_layers != static_cast<uint32_t>(cache_->n_layers()) ||
        hdr.n_kv_heads != static_cast<uint32_t>(cache_->n_kv_heads()) ||
        hdr.head_dim != static_cast<uint32_t>(cache_->head_dim()) ||
        hdr.dtype != std::to_underlying(cache_->qtype()) || hdr.block_bytes != cache_->block_bytes()) {
        IMP_LOG_WARN(
            "Prefix cache: config mismatch (layers=%u/%d, heads=%u/%d, "
            "dim=%u/%d, dtype=%u/%d)",
            hdr.n_layers, cache_->n_layers(), hdr.n_kv_heads, cache_->n_kv_heads(), hdr.head_dim,
            cache_->head_dim(), hdr.dtype, std::to_underlying(cache_->qtype()));
        fclose(f);
        return -1;
    }

    // Reject KV produced by a different model/tokenizer/quant. Same-geometry
    // models share token-hash keys, so without this gate an evicted-and-reused
    // cache file would serve another model's KV silently (wrong output).
    if (hdr.model_fingerprint != model_fingerprint) {
        IMP_LOG_WARN("Prefix cache: model fingerprint mismatch (cache=0x%016llx, current=0x%016llx) "
                     "— discarding cache from a different model/tokenizer/quant",
                     static_cast<unsigned long long>(hdr.model_fingerprint),
                     static_cast<unsigned long long>(model_fingerprint));
        fclose(f);
        return -1;
    }

    int n_blocks = static_cast<int>(hdr.n_blocks);
    int nl = cache_->n_layers();
    size_t bb = cache_->block_bytes();
    size_t per_block_total = static_cast<size_t>(nl) * 2 * bb;

    int loaded = 0;
    int skipped = 0;

    // Read all blocks from disk first, then issue all H2D copies asynchronously.
    struct LoadEntry {
        size_t hash;
        int block_id;
        size_t buf_offset;  // offset into host_buf for this block's data
        bool copy_ok;
    };
    std::vector<LoadEntry> load_entries;
    load_entries.reserve(n_blocks);

    // We need separate host memory regions per block for async copies,
    // so allocate a buffer large enough for all blocks we'll actually load.
    std::vector<uint8_t> all_host_buf;
    // Temporary buffer for reading + skipping
    std::vector<uint8_t> read_buf(per_block_total);

    for (int i = 0; i < n_blocks; i++) {
        size_t hash;
        if (fread(&hash, sizeof(hash), 1, f) != 1)
            break;
        if (fread(read_buf.data(), per_block_total, 1, f) != 1)
            break;

        // Skip if hash already exists (shouldn't happen on fresh start)
        if (block_hash_to_id_.contains(hash)) {
            skipped++;
            continue;
        }

        // Allocate a fresh block
        int block_id = cache_->allocate_block();
        if (block_id < 0) {
            IMP_LOG_INFO("Prefix cache: pool exhausted after %d blocks", loaded);
            break;
        }

        // Append this block's data to the persistent host buffer
        size_t buf_offset = all_host_buf.size();
        all_host_buf.insert(all_host_buf.end(), read_buf.begin(), read_buf.end());
        load_entries.push_back({hash, block_id, buf_offset, true});
        loaded++;
    }

    fclose(f);

    // Issue all H2D copies asynchronously.
    for (auto& entry : load_entries) {
        size_t offset = 0;
        for (int l = 0; l < nl; l++) {
            cudaError_t err = cudaMemcpyAsync(cache_->k_ptr(l, entry.block_id),
                                              all_host_buf.data() + entry.buf_offset + offset, bb,
                                              cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed loading block %d layer %d K: %s", entry.block_id, l,
                              cudaGetErrorString(err));
                entry.copy_ok = false;
                break;
            }
            offset += bb;
            err = cudaMemcpyAsync(cache_->v_ptr(l, entry.block_id),
                                  all_host_buf.data() + entry.buf_offset + offset, bb, cudaMemcpyHostToDevice,
                                  stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed loading block %d layer %d V: %s", entry.block_id, l,
                              cudaGetErrorString(err));
                entry.copy_ok = false;
                break;
            }
            offset += bb;
        }
    }

    // Single sync to wait for all H2D transfers.
    cudaStreamSynchronize(stream);

    // Register successfully loaded blocks. Free blocks with failed copies.
    int actual_loaded = 0;
    for (const auto& entry : load_entries) {
        if (!entry.copy_ok) {
            cache_->free_block(entry.block_id);
            continue;
        }

        block_hash_to_id_[entry.hash] = entry.block_id;
        block_id_to_hash_[entry.block_id] = entry.hash;

        cached_blocks_lru_.push_back(entry.block_id);
        reclaimable_cached_count_++;
        cached_blocks_map_.emplace(entry.block_id,
                                   CachedEntry{std::prev(cached_blocks_lru_.end()),
                                               cache_->adopt_block(entry.block_id)});

        actual_loaded++;
    }
    loaded = actual_loaded;
    IMP_LOG_INFO("Prefix cache: restored %d blocks (%.1f MiB) from %s%s", loaded,
                 (loaded * per_block_total) / (1024.0 * 1024.0), path.c_str(),
                 skipped > 0 ? " (some skipped)" : "");
    return loaded;
}

}  // namespace imp
