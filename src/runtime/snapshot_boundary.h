#pragma once

// Where a prefill takes its prefix-cache snapshot (recurrent slab on a
// hybrid, SWA window on a windowed model): the largest block-aligned prompt
// position, or nowhere when that position is shorter than `min_tokens`.
//
// The snapshot costs every first turn a prefill split at the boundary (two
// eager chunks of ~1500 launches each instead of one) plus a stream sync
// between them. Measured 2026-09-08 on Qwen3.8-27B-NVFP4, one stream, a
// 35-token prompt: server-side TTFT floor 36 ms split against 22 ms unsplit,
// while the turn-2 saving of a 32-token prefix is under 3 ms of GPU time.
// A snapshot pays for itself only when the prefix it saves is long.

namespace imp {

inline int snapshot_boundary(int prompt_tokens, int block_size, int min_tokens) {
    if (block_size <= 0 || prompt_tokens <= 0)
        return 0;
    const int end = (prompt_tokens / block_size) * block_size;
    return end >= min_tokens ? end : 0;
}

}  // namespace imp
