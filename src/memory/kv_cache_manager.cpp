#include "memory/kv_cache_manager.h"
#include "core/logging.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <functional>
#include <cuda_runtime.h>

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
        // Pinned blocks survive free_sequence: keep ref_count=1, add to
        // cached LRU for reuse. They remain in pinned_blocks_ and cannot
        // be evicted until unpin_prefix() is called.
        if (pinned_blocks_.find(block_id) != pinned_blocks_.end()) {
            // If ref_count > 1 (shared), just decrement.
            if (cache_->ref_count(block_id) > 1) {
                cache_->free_block(block_id);
            } else {
                // ref_count == 1 and pinned — keep alive in cached LRU.
                if (cached_blocks_map_.find(block_id) == cached_blocks_map_.end()) {
                    cached_blocks_lru_.push_back(block_id);
                    cached_blocks_map_[block_id] = std::prev(cached_blocks_lru_.end());
                }
            }
            continue;
        }

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

    // Skip pinned sequences — find the first unpinned LRU victim.
    for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
        int candidate = *it;
        if (pinned_seqs_.find(candidate) != pinned_seqs_.end()) continue;
        free_sequence(candidate);  // also removes from lru_order_ / lru_map_
        return candidate;
    }

    return -1; // All sequences are pinned.
}

bool KVCacheManager::can_allocate(int num_blocks) const {
    if (num_blocks <= 0) return true;
    if (cache_->num_free_blocks() >= num_blocks) return true;

    // Count how many blocks we *could* reclaim by evicting LRU sequences.
    int reclaimable = cache_->num_free_blocks();

    // Count cached (unreferenced) blocks that can be evicted, excluding pinned.
    for (int block_id : cached_blocks_lru_) {
        if (pinned_blocks_.find(block_id) == pinned_blocks_.end()) {
            ++reclaimable;
        }
    }
    if (reclaimable >= num_blocks) return true;

    for (auto it = lru_order_.begin(); it != lru_order_.end(); ++it) {
        // Skip pinned sequences — their blocks are not reclaimable.
        if (pinned_seqs_.find(*it) != pinned_seqs_.end()) continue;
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
    // Validate that cached block IDs are still allocated (not stale).
    // Legacy prefix cache doesn't track block lifecycle, so blocks may
    // have been freed and reused by other sequences since registration.
    for (int bid : it->second) {
        if (bid < 0 || bid >= cache_->total_blocks() || cache_->ref_count(bid) <= 0) {
            IMP_LOG_WARN("find_prefix: stale block_id %d in legacy prefix cache (hash=%zu)",
                         bid, prefix_hash);
            return {};  // invalidate entire prefix — partial reuse is unsafe
        }
    }
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

    // Find the first non-pinned cached block.
    int block_id = -1;
    for (auto it = cached_blocks_lru_.begin(); it != cached_blocks_lru_.end(); ++it) {
        if (pinned_blocks_.find(*it) == pinned_blocks_.end()) {
            block_id = *it;
            cached_blocks_lru_.erase(it);
            cached_blocks_map_.erase(block_id);
            break;
        }
    }
    if (block_id < 0) return -1;

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

    // Try reclaiming cached blocks (cheaper than evicting a sequence).
    // Loop until we get a usable block or exhaust all reclaimable cached blocks.
    while (!cached_blocks_lru_.empty()) {
        if (reclaim_cached_block() < 0) break;  // All remaining cached blocks are pinned.
        block_id = cache_->allocate_block();
        if (block_id >= 0) return block_id;
    }

    return -1; // Caller should try evict_lru() if needed.
}

// ─── Prefix block pinning ────────────────────────────────────────

void KVCacheManager::pin_prefix(int seq_id, int num_blocks) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return;

    const auto& blocks = it->second;
    int to_pin = std::min(num_blocks, static_cast<int>(blocks.size()));
    if (to_pin <= 0) return;

    for (int i = 0; i < to_pin; ++i) {
        pinned_blocks_.insert(blocks[i]);
    }
    pinned_seqs_[seq_id] = to_pin;

    IMP_LOG_DEBUG("PinPrefix: seq %d pinned %d blocks", seq_id, to_pin);
}

void KVCacheManager::unpin_prefix(int seq_id) {
    auto pin_it = pinned_seqs_.find(seq_id);
    if (pin_it == pinned_seqs_.end()) return;

    int num_pinned = pin_it->second;

    // Try to find the blocks in the active sequence first.
    auto seq_it = seq_blocks_.find(seq_id);
    if (seq_it != seq_blocks_.end()) {
        const auto& blocks = seq_it->second;
        int to_unpin = std::min(num_pinned, static_cast<int>(blocks.size()));
        for (int i = 0; i < to_unpin; ++i) {
            pinned_blocks_.erase(blocks[i]);
        }
    } else {
        // Sequence was already freed — pinned blocks might be in the
        // cached LRU or hash table. We stored block IDs in pinned_blocks_,
        // but we don't know which ones belong to this seq_id anymore.
        // Scan pinned_blocks_ for blocks that are in cached_blocks_map_.
        // This is O(pinned) which is small for prefix pinning.
        //
        // Actually, we need a reverse mapping. Since pinned blocks survive
        // free_sequence, they are in cached_blocks_lru_. We remove them
        // from pinned_blocks_ by iterating cached blocks.
        // Simpler: just remove ALL block IDs from pinned_blocks_ that are
        // no longer in any active sequence. But that could unpin blocks
        // from other sequences. Instead, we do nothing here — the blocks
        // are already orphaned. They'll be in cached_blocks_lru_ and can
        // now be evicted normally since we remove the seq from pinned_seqs_.
        //
        // The correct approach: iterate pinned_blocks_ and remove any that
        // aren't in any active sequence's block table AND aren't pinned by
        // another sequence. For simplicity and correctness, rebuild
        // pinned_blocks_ from pinned_seqs_ (minus this one).
        pinned_blocks_.clear();
        for (const auto& [other_seq, other_count] : pinned_seqs_) {
            if (other_seq == seq_id) continue;
            auto other_it = seq_blocks_.find(other_seq);
            if (other_it == seq_blocks_.end()) continue;
            const auto& other_blocks = other_it->second;
            int n = std::min(other_count, static_cast<int>(other_blocks.size()));
            for (int i = 0; i < n; ++i) {
                pinned_blocks_.insert(other_blocks[i]);
            }
        }
    }

    pinned_seqs_.erase(pin_it);
    IMP_LOG_DEBUG("UnpinPrefix: seq %d unpinned", seq_id);
}

int KVCacheManager::num_pinned_blocks() const {
    return static_cast<int>(pinned_blocks_.size());
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

// ─── Persistent prefix cache ────────────────────────────────────────

// Binary format:
//   Header: magic(4) version(4) n_blocks(4) n_layers(4) n_kv_heads(4)
//           head_dim(4) dtype(4) block_bytes(8)
//   Per block: hash(8) + KV data (n_layers * 2 * block_bytes)

static constexpr uint32_t kPrefixCacheMagic = 0x494D5043;  // "IMPC"
static constexpr uint32_t kPrefixCacheVersion = 1;

struct PrefixCacheHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_blocks;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t dtype;
    uint64_t block_bytes;
};

int KVCacheManager::save_prefix_cache(const std::string& path, cudaStream_t stream) {
    if (!prefix_caching_enabled_ || cached_blocks_lru_.empty()) {
        IMP_LOG_INFO("Prefix cache: nothing to save (0 cached blocks)");
        return 0;
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
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
    hdr.dtype = static_cast<uint32_t>(cache_->dtype());
    hdr.block_bytes = bb;

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
        if (hash_it == block_id_to_hash_.end()) continue;
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
                                              cache_->k_ptr(l, block_id),
                                              bb, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed for block %d layer %d K: %s",
                              block_id, l, cudaGetErrorString(err));
                block_ok[bi] = false;
                break;
            }
            offset += bb;
            err = cudaMemcpyAsync(host_buf.data() + buf_offset + offset,
                                  cache_->v_ptr(l, block_id),
                                  bb, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed for block %d layer %d V: %s",
                              block_id, l, cudaGetErrorString(err));
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
        if (!block_ok[bi]) continue;  // Skip blocks with failed copies.

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
    IMP_LOG_INFO("Prefix cache: saved %d blocks (%.1f MiB) to %s",
                 saved, (sizeof(hdr) + saved * (8 + per_block_total)) / (1024.0 * 1024.0),
                 path.c_str());
    return saved;
}

int KVCacheManager::load_prefix_cache(const std::string& path, cudaStream_t stream) {
    if (!prefix_caching_enabled_) {
        IMP_LOG_WARN("Prefix cache: loading disabled (prefix caching not enabled)");
        return -1;
    }

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        IMP_LOG_INFO("Prefix cache: no cache file at %s", path.c_str());
        return 0;
    }

    PrefixCacheHeader hdr;
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
        hdr.dtype != static_cast<uint32_t>(cache_->dtype()) ||
        hdr.block_bytes != cache_->block_bytes()) {
        IMP_LOG_WARN("Prefix cache: config mismatch (layers=%u/%d, heads=%u/%d, "
                     "dim=%u/%d, dtype=%u/%d)",
                     hdr.n_layers, cache_->n_layers(), hdr.n_kv_heads, cache_->n_kv_heads(),
                     hdr.head_dim, cache_->head_dim(), hdr.dtype, static_cast<int>(cache_->dtype()));
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
        if (fread(&hash, sizeof(hash), 1, f) != 1) break;
        if (fread(read_buf.data(), per_block_total, 1, f) != 1) break;

        // Skip if hash already exists (shouldn't happen on fresh start)
        if (block_hash_to_id_.count(hash)) { skipped++; continue; }

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
                                              all_host_buf.data() + entry.buf_offset + offset,
                                              bb, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed loading block %d layer %d K: %s",
                              entry.block_id, l, cudaGetErrorString(err));
                entry.copy_ok = false;
                break;
            }
            offset += bb;
            err = cudaMemcpyAsync(cache_->v_ptr(l, entry.block_id),
                                  all_host_buf.data() + entry.buf_offset + offset,
                                  bb, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("cudaMemcpyAsync failed loading block %d layer %d V: %s",
                              entry.block_id, l, cudaGetErrorString(err));
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
        cached_blocks_map_[entry.block_id] = std::prev(cached_blocks_lru_.end());

        actual_loaded++;
    }
    loaded = actual_loaded;
    IMP_LOG_INFO("Prefix cache: restored %d blocks (%.1f MiB) from %s%s",
                 loaded, (loaded * per_block_total) / (1024.0 * 1024.0), path.c_str(),
                 skipped > 0 ? " (some skipped)" : "");
    return loaded;
}

} // namespace imp
