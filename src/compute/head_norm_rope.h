#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Head dim of head_norm_rope_fp16: 64 threads per (row, head), thread i owns the pair (i, i + 64).
inline constexpr int kHeadNormRopeDim = 128;

// Per (row, head) of x [rows, row_stride] (heads of kHeadNormRopeDim from column 0), in place:
// RMSNorm over the head times w (w == nullptr: no norm), then NeoX RoPE at d_pos[row] (device).
// Shared by the block drafters (audit D-14); inkling_draft.cu draft_norm_rope_k is the next caller.
void head_norm_rope_fp16(half* x, int rows, int row_stride, int heads, const half* w, const int* d_pos,
                         float log2_theta, float eps, cudaStream_t stream);

}  // namespace imp
