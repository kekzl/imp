#pragma once

// Minimum rows of a prompt chunk. Below 33 rows a prompt row takes decode kernels (M=1 GEMV,
// M<=32 small-M/split-K/LM-head paths), so its value depends on the chunk it lands in (#2152).
// Every prompt chunk after a prefix restore or a chunk split keeps at least this many rows.

namespace imp {

inline constexpr int kMinPromptChunkRows = 33;

// Chunk length for a non-final chunk: shortened so the rows left after it are 0 or
// >= kMinPromptChunkRows. Unchanged when shortening would itself go below the minimum.
constexpr int keep_prompt_tail(int chunk_len, int remaining) {
    const int tail = remaining - chunk_len;
    if (tail <= 0 || tail >= kMinPromptChunkRows)
        return chunk_len;
    const int shortened = remaining - kMinPromptChunkRows;
    return shortened >= kMinPromptChunkRows ? shortened : chunk_len;
}

// Prefill chunk before the decode-aware cap: the resolved size (0 = executor max_tokens), capped at
// max_tokens, floored to whole KV blocks. The chunk grid a fresh prefill splits on.
constexpr int base_prefill_chunk(int resolved, int max_tokens, int block_size) {
    int chunk = resolved > 0 && resolved < max_tokens ? resolved : max_tokens;
    if (block_size > 0 && chunk > block_size)
        chunk = (chunk / block_size) * block_size;
    return chunk;
}
// Most full KV blocks a prefix-cache hit may reuse on a prompt of `total` tokens so the
// re-prefilled tail keeps kMinPromptChunkRows rows. 0 = no reuse.
constexpr int prompt_reuse_cap_blocks(int total, int block_size) {
    return (block_size > 0 && total > kMinPromptChunkRows) ? (total - kMinPromptChunkRows) / block_size : 0;
}

}  // namespace imp
