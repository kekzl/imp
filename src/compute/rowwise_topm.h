#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// Row-wise top-M logit indices (best first, ties -> lowest index), d_out[row*m+rank].
// M capped at kRowwiseTopMMax. Feeds the Token-Recycling adjacency table
// (speculative.token_recycling) from the spec-verify chunk: m masked argmax passes per
// row, one block per row (rows <= chunk_pad ~32).
inline constexpr int kRowwiseTopMMax = 16;

void rowwise_topm(const float* d_logits, int rows, int vocab, int m, int32_t* d_out,
                  cudaStream_t stream);
// Pre-allocate the two-stage scratch for this shape — REQUIRED before the
// first rowwise_topm call that runs under stream capture (a lazy cudaMalloc
// mid-capture aborts the capture).
void rowwise_topm_reserve(int rows, int m);

}  // namespace imp
