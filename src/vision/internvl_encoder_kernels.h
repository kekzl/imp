#pragma once

// InternVL-specific encoder kernels. LayerNorm, bias, GELU, softmax and head merge are shared
// with the Qwen3-VL encoder (qwen3vl_encoder_kernels.h).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {

// out[r, c] = row[c] for r < rows: the bias a beta=1 GEMM then accumulates onto.
void launch_internvl_fill_rows(half* out, const half* row, int rows, int dim, cudaStream_t stream);

// hidden [1 + patches, dim]: row 0 = cls; then every row += pos[row] (table has 1 + patches rows).
void launch_internvl_cls_pos(half* hidden, const half* cls, const half* pos, int tokens, int dim,
                             cudaStream_t stream);

// qkv [tokens, 3*heads*head_dim] (q | k | v) -> q, k, v [heads][tokens][head_dim].
void launch_internvl_split_heads(const half* qkv, half* q, half* k, half* v, int tokens, int heads,
                                 int head_dim, cudaStream_t stream);

// h[t, c] += scale[c] * x[t, c] (layer scale lambda_1 / lambda_2), FP32 math.
void launch_internvl_scaled_residual(half* h, const half* x, const half* scale, int tokens, int dim,
                                     cudaStream_t stream);

// Drops the CLS row of hidden [1 + side*side, dim], then 2x2 pixel shuffle (HF InternVLModel.pixel_shuffle,
// downsample 0.5): out token (i, j) of the (side/2)^2 raster = [x(2i,2j) | x(2i,2j+1) | x(2i+1,2j) |
// x(2i+1,2j+1)], each `dim` wide.
void launch_internvl_pixel_shuffle(const half* hidden, half* out, int side, int dim, cudaStream_t stream);

}  // namespace imp
