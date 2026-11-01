#pragma once

// EVS video token pruning, device half (host half: video_evs.h).

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// One video's merged embeddings emb [groups * tokens_per_group, dim] -> dissim[j]: 255 for pair 0,
// else 1 - cos(row j, row j - tokens_per_group) with each norm clamped to >= 1e-8 (torch
// cosine_similarity). One warp per row, FP32 accumulate.
void launch_evs_dissimilarity(const half* emb, int groups, int tokens_per_group, int dim, float* dissim,
                              cudaStream_t stream);

// out[r, :] = in[rows[r], :] for r < n_out, rows of `dim` halves; in and out must not overlap.
void launch_gather_rows(const half* in, const int32_t* rows, int n_out, int dim, half* out,
                        cudaStream_t stream);

}  // namespace imp
