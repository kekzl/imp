#pragma once

// Where a prefill takes its prefix-cache snapshot (recurrent slab on a
// hybrid, SWA window on a windowed model): the largest block-aligned prompt
// position, or nowhere when that position is shorter than `min_tokens`.
//
// Splitting the prefill at the boundary costs two eager chunks plus a
// stream sync instead of one. A snapshot pays for itself only when the
// prefix it saves is long.

#include "memory/kv_cache_manager.h"
#include "runtime/prompt_tail.h"

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

// Largest block-aligned position leaving >= tail_rows prompt rows after it. tail_rows 1: a restore
// needs one token to forward (a block-aligned prompt snapshots one block short of its length).
// Hybrids pass kMinPromptChunkRows: the rows after the snapshot are a prompt chunk (#2560).
inline int snapshot_boundary(int prompt_tokens, int block_size, int min_tokens, int tail_rows = 1) {
    if (block_size <= 0 || tail_rows < 1 || prompt_tokens <= tail_rows)
        return 0;
    const int end = ((prompt_tokens - tail_rows) / block_size) * block_size;
    return end >= min_tokens ? end : 0;
}

// Save position for the chunk starting at `offset`: the hint's block floor when it is a valid
// boundary before the prompt one (shared prefix of /v1/decide items, #2198), else the prompt
// boundary. A restore matches only a snapshot at an exact block, so the shared prefix needs its own.
// The hint keeps >= tail_rows rows on both sides.
inline int next_snapshot_boundary(int prompt_tokens, int block_size, int min_tokens, int hint_tokens,
                                  int offset, int tail_rows = 1) {
    const int end = snapshot_boundary(prompt_tokens, block_size, min_tokens, tail_rows);
    if (end > 0 && hint_tokens > 0) {
        const int h = (hint_tokens / block_size) * block_size;
        if (h >= block_size && h >= min_tokens && h - offset >= tail_rows && end - h >= tail_rows)
            return h;
    }
    return end;
}

// Length of the prefill chunk at `offset` given the snapshot boundary `snap_end` (0 = none). Both
// sides of the split keep >= kMinPromptChunkRows rows: a chunk ending that many rows short of the
// boundary, or no split (the boundary is skipped) (#2560, #2562). `last`: the chunk ends the prompt.
constexpr int snapshot_chunk_len(int offset, int chunk_len, bool last, int snap_end) {
    constexpr int min_rows = kMinPromptChunkRows;
    const int snap_rows = snap_end - offset;
    if (snap_rows >= min_rows && snap_rows < chunk_len)
        return snap_rows;
    if (!last && snap_rows > chunk_len && snap_rows - chunk_len < min_rows &&
        snap_rows - min_rows >= min_rows)
        return snap_rows - min_rows;
    return chunk_len;
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
