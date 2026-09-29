#pragma once

// Where a prefill takes its prefix-cache snapshot (recurrent slab on a
// hybrid, SWA window on a windowed model): the largest block-aligned prompt
// position, or nowhere when that position is shorter than `min_tokens`.
//
// Splitting the prefill at the boundary costs two eager chunks plus a
// stream sync instead of one. A snapshot pays for itself only when the
// prefix it saves is long.

#include "memory/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace imp {

// Key of the transcript snapshot a hybrid request saves at finish: the state after
// exactly tokens.size() forwarded tokens (prompt + generated minus the final sample).
// Full blocks chain like the KV prefix hashes (parent 0, same as the prefill-boundary
// snapshot, so an aligned transcript dedups against it); a partial tail block chains
// on top with its length mixed in, so a tail of m tokens never keys like a full block
// or like a tail of another length. 0 for fewer than one full block.
inline size_t transcript_tail_key(size_t full_chain_key, std::span<const int32_t> tail) {
    return KVCacheManager::compute_block_hash(tail, full_chain_key) ^
           (static_cast<size_t>(tail.size()) * 0x9E3779B97F4A7C15ULL);
}

inline size_t transcript_snapshot_key(std::span<const int32_t> tokens, int block_size) {
    if (block_size <= 0)
        return 0;
    const int n = static_cast<int>(tokens.size());
    const int full = n / block_size;
    if (full == 0)
        return 0;
    size_t key = 0;
    for (int b = 0; b < full; ++b)
        key = KVCacheManager::compute_block_hash(
            tokens.subspan(static_cast<size_t>(b) * block_size, block_size), key);
    if (n > full * block_size)
        key = transcript_tail_key(key, tokens.subspan(static_cast<size_t>(full) * block_size));
    return key;
}

// A restore has to leave at least one prompt token to forward (the model
// needs logits), so the admission cap is (prompt_tokens - 1) / block_size
// blocks (Engine::hybrid_prefix_reuse_limit_). A block-aligned prompt
// therefore snapshots one block short of its length: at full length the
// snapshot could never be matched, and every aligned prompt got zero reuse.
inline int snapshot_boundary(int prompt_tokens, int block_size, int min_tokens) {
    if (block_size <= 0 || prompt_tokens <= 1)
        return 0;
    const int end = ((prompt_tokens - 1) / block_size) * block_size;
    return end >= min_tokens ? end : 0;
}

// Save position for the chunk starting at `offset`: the hint's block floor when it is a valid
// boundary before the prompt one (shared prefix of /v1/decide items, #2198), else the prompt
// boundary. A restore matches only a snapshot at an exact block, so the shared prefix needs its own.
inline int next_snapshot_boundary(int prompt_tokens, int block_size, int min_tokens, int hint_tokens,
                                  int offset) {
    const int end = snapshot_boundary(prompt_tokens, block_size, min_tokens);
    if (end > 0 && hint_tokens > 0) {
        const int h = (hint_tokens / block_size) * block_size;
        if (h >= block_size && h >= min_tokens && h > offset && h < end)
            return h;
    }
    return end;
}

// Length of the token prefix every sequence shares (0 for fewer than two sequences).
inline int common_prefix_tokens(const std::vector<std::vector<int32_t>>& seqs) {
    if (seqs.size() < 2)
        return 0;
    size_t n = seqs[0].size();
    for (size_t i = 1; i < seqs.size(); ++i) {
        size_t k = 0;
        const size_t lim = std::min(n, seqs[i].size());
        while (k < lim && seqs[i][k] == seqs[0][k])
            ++k;
        n = k;
    }
    return static_cast<int>(n);
}

}  // namespace imp
