#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Paged attention for decode (single-query per sequence)
// Q: [batch, 1, n_heads, head_dim]
// block_tables: [batch, max_blocks] int32
// context_lens: [batch] int32
// sliding_window: 0 = disabled, >0 = only attend to last N KV positions
// softcap: 0 = disabled, >0 = apply tanh(score/cap)*cap (Gemma-2/3)
void paged_attention_decode(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const int* block_tables, const int* context_lens,
    int block_size, float scale, int max_context_len,
    int sliding_window = 0, float softcap = 0.0f,
    cudaStream_t stream = nullptr);

// Set split-K scratch buffer for paged attention. Must be called before
// paged_attention_decode if split-K is desired. The scratch buffer holds
// partial softmax state: size = batch * n_heads * num_splits * (2 + head_dim) * sizeof(float).
// Pass nullptr to disable split-K.
void paged_attention_set_splitk_scratch(void* ptr, size_t size);

// FP8 E4M3 Paged attention for decode: KV cache stored in FP8 with on-the-fly dequant.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, n_kv_heads, block_size, head_dim] FP8_E4M3
// O: [batch, 1, n_heads, head_dim] FP16
// kv_scale: per-tensor FP32 scale for FP8 dequantization (val = fp8_val * kv_scale)
void paged_attention_decode_fp8(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const int* block_tables, const int* context_lens,
    int block_size, float scale, float kv_scale,
    int max_context_len, int sliding_window = 0,
    float softcap = 0.0f, cudaStream_t stream = nullptr);

// INT8 dp4a Paged attention for decode: KV cache stored in INT8 with per-head scales.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, block_size, n_kv_heads, head_dim] INT8
// K_scales/V_scales: [num_blocks, block_size, n_kv_heads] FP16 per-head scales
// O: [batch, 1, n_heads, head_dim] FP16
void paged_attention_decode_int8(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O,
    const half* K_scales, const half* V_scales,
    const int* block_tables, const int* context_lens,
    int block_size, float scale,
    int max_context_len, int sliding_window = 0,
    float softcap = 0.0f, cudaStream_t stream = nullptr);

} // namespace imp
