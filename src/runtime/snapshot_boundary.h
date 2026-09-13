#pragma once

// Where a prefill takes its prefix-cache snapshot (recurrent slab on a
// hybrid, SWA window on a windowed model): the largest block-aligned prompt
// position, or nowhere when that position is shorter than `min_tokens`.
//
// Splitting the prefill at the boundary costs two eager chunks plus a
// stream sync instead of one. A snapshot pays for itself only when the
// prefix it saves is long.

namespace imp {

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

}  // namespace imp
