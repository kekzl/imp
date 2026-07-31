#pragma once

// Element-wise and layout kernels of the Qwen3-VL vision encoder.
//
// Split from the forward orchestration so that editing a kernel does not
// re-`ptxas` the cuBLAS plumbing, and vice versa.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// hidden[i, :] += sum_t weights[i, t] * table[taps[i, t], :]
// The learned position table is a fixed square grid; this is its bilinear
// resample onto this image's grid, precomputed on the host as a gather.
void launch_qwen3vl_pos_embed_add(half* hidden, const half* table, const int32_t* taps, const float* weights,
                                  int tokens, int dim, int taps_per_token, cudaStream_t stream);

// LayerNorm with weight AND bias (eps 1e-6), over the last dimension.
void launch_qwen3vl_layernorm(const half* x, const half* weight, const half* bias, half* out, int rows,
                              int dim, float eps, cudaStream_t stream);

void launch_qwen3vl_add_bias(half* x, const half* bias, int rows, int dim, cudaStream_t stream);
void launch_qwen3vl_residual_add(half* dst, const half* src, int64_t n, cudaStream_t stream);

// The encoder uses BOTH GELU variants, and not by accident: the block MLP uses
// `gelu_pytorch_tanh` (config `hidden_act`) while the patch mergers use
// `nn.GELU()`, which is the exact erf form.
void launch_qwen3vl_gelu_tanh(half* x, int64_t n, cudaStream_t stream);
void launch_qwen3vl_gelu_erf(half* x, int64_t n, cudaStream_t stream);

// Split the fused QKV projection [tokens, 3*heads*head_dim] into per-head
// [heads][tokens][head_dim] and rotate q/k by the encoder's 2-D RoPE in one
// pass. The first half of each head's rotary dims is driven by the patch ROW,
// the second half by its COLUMN; the rotation itself is the half-split (NeoX)
// form, not interleaved.
void launch_qwen3vl_split_qkv_rope(const half* qkv, const int32_t* row, const int32_t* col, half* q, half* k,
                                   half* v, int tokens, int heads, int head_dim, float theta,
                                   cudaStream_t stream);

// Row-wise softmax in FP32 over `cols`, in place.
void launch_qwen3vl_softmax_rows(half* scores, int rows, int cols, cudaStream_t stream);

// [heads][tokens][head_dim] -> [tokens, heads*head_dim]
void launch_qwen3vl_merge_heads(const half* per_head, half* out, int tokens, int heads, int head_dim,
                                cudaStream_t stream);

}  // namespace imp
