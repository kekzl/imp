#pragma once

#include <algorithm>
#include <cstddef>

namespace imp {

// Prompt-logprobs logits chunk (#2257): rows = min(n_rows, 1024, (avail_bytes / 2) / (4 * vocab)).
// Cap 1024 rows = 594 MiB at vocab 151936. 0 = not one row fits: the per-batch LM-head driver.
inline constexpr int kPromptLpMaxChunkRows = 1024;

inline int prompt_lp_chunk_rows(int n_rows, int vocab, size_t avail_bytes) {
    if (n_rows <= 0 || vocab <= 0)
        return 0;
    const size_t fit = avail_bytes / 2 / (sizeof(float) * static_cast<size_t>(vocab));
    return static_cast<int>(
        std::min({static_cast<size_t>(n_rows), static_cast<size_t>(kPromptLpMaxChunkRows), fit}));
}

}  // namespace imp
