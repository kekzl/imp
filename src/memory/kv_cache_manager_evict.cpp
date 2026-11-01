// Score-based block eviction of KVCacheManager (kv_cache.evict_keep_ratio, roadmap row 48), split out
// of kv_cache_manager.cpp (file-size gate). Design: docs/plans/2026-11-01-kv-eviction.md.

#include "memory/kv_cache_manager.h"

#include <algorithm>

namespace imp {

int KVCacheManager::evict_blocks(int seq_id, std::span<const int> logical_blocks) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end() || logical_blocks.empty())
        return 0;
    auto& blocks = it->second;
    const int n = static_cast<int>(blocks.size());
    int freed = 0;
    int first_freed = n;
    for (const int i : logical_blocks) {
        if (i < 0 || i >= n)
            continue;
        const int bid = blocks.id_at(static_cast<size_t>(i));
        if (bid < 0 || pinned_blocks_.contains(bid) || cache_->ref_count(bid) != 1)
            continue;  // already a hole, pinned, or shared with another sequence / the prefix cache
        auto hit = block_id_to_hash_.find(bid);
        if (hit != block_id_to_hash_.end()) {
            block_hash_to_id_.erase(hit->second);
            block_id_to_hash_.erase(hit);
        }
        blocks.make_hole(static_cast<size_t>(i));
        first_freed = std::min(first_freed, i);
        ++freed;
    }
    // Downstream blocks chained their hash through the freed ones: none is a valid prefix now.
    auto hash_it = seq_block_hashes_.find(seq_id);
    if (freed > 0 && hash_it != seq_block_hashes_.end()) {
        auto& hashes = hash_it->second;
        for (int i = first_freed; i < n && i < static_cast<int>(hashes.size()); ++i) {
            const int bid = blocks.id_at(static_cast<size_t>(i));
            if (bid >= 0 && cache_->ref_count(bid) == 1) {
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

}  // namespace imp
