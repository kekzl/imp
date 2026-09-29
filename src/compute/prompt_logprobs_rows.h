#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// Prompt logprobs (#2207, #2257): top-N cap of the fused row kernel (= kMaxPromptLogprobs).
inline constexpr int kPlpMaxTopN = 20;

// One block per row, one pass over the row. Layout: vocab slabs of slab_w columns, slab at column
// n0 stored row-major [rows x min(slab_w, V - n0)] at logits + n0 * rows; slab_w >= V = row-major.
// lp = x[t] - logsumexp(x); rank = 1 + #{j : x[j] > x[t] or (x[j] == x[t] and j < t)};
// top_n (<= kPlpMaxTopN) by (value desc, index asc). top_ids/top_lp may be null when top_n == 0.
void prompt_logprobs_rows(const float* logits, int rows, int V, int slab_w, const int32_t* targets, int top_n,
                          float* lp, int32_t* rank, int32_t* top_ids, float* top_lp, cudaStream_t stream);

}  // namespace imp
