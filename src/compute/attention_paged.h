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
// n_sinks: 0 = disabled, >0 = StreamingLLM — also attend to tokens [0, n_sinks).
//          Requires sliding_window > 0 and ctx_len > n_sinks + sliding_window to
//          activate; otherwise behaves as plain sliding-window / full attention.
// softcap: 0 = disabled, >0 = apply tanh(score/cap)*cap (Gemma-2/3)
void paged_attention_decode(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O, const int* block_tables, const int* context_lens,
    int block_size, float scale, int max_context_len,
    int sliding_window = 0, float softcap = 0.0f,
    cudaStream_t stream = nullptr,
    int max_blocks_per_seq = 0,
    int n_sinks = 0);

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
    float softcap = 0.0f, cudaStream_t stream = nullptr,
    int max_blocks_per_seq = 0,
    int n_sinks = 0);

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
    float softcap = 0.0f, cudaStream_t stream = nullptr,
    int max_blocks_per_seq = 0,
    int n_sinks = 0);

// INT4 Paged attention for decode: KV cache stored in packed INT4 with per-head scales.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, block_size, n_kv_heads, head_dim/2] packed uint8
// K_scales/V_scales: [num_blocks, block_size, n_kv_heads] FP16 per-head scales
// O: [batch, 1, n_heads, head_dim] FP16
void paged_attention_decode_int4(
    const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache,
    Tensor& O,
    const half* K_scales, const half* V_scales,
    const int* block_tables, const int* context_lens,
    int block_size, float scale,
    int max_context_len, int sliding_window = 0,
    float softcap = 0.0f, cudaStream_t stream = nullptr,
    int max_blocks_per_seq = 0,
    int n_sinks = 0);

// TurboQuant Paged attention for decode: PolarQuant K + QJL sketch + INT4 V.
// K_dir_cache: packed directions — INT4 uniform (K_mscales=nullptr) or FP4 E2M1 (K_mscales!=nullptr)
// K_mscales: nullptr → uniform INT4 dequant (/7.0), non-null → MXFP4 FP4 E2M1 + UE8M0 micro-scales
void paged_attention_decode_turboquant(
    const Tensor& Q, const Tensor& K_dir_cache, const Tensor& V_cache,
    Tensor& O,
    const half* K_norms, const half* V_scales,
    const uint8_t* K_sketches, const uint8_t* qjl_matrix,
    const int* block_tables, const int* context_lens,
    int block_size, float scale, int sketch_dim,
    int max_context_len, int sliding_window = 0,
    float softcap = 0.0f, cudaStream_t stream = nullptr,
    int max_blocks_per_seq = 0,
    const uint8_t* K_mscales = nullptr,
    int n_sinks = 0);

// TurboQuant Lite Paged attention for decode: QJL sketch-only K + INT4 V.
// K is represented only by QJL sketches + FP16 norms (no INT4 directions in pool).
// Q.K estimated purely via QJL: ||q|| * ||k|| * (2*popcount(XNOR(sketch_q, sketch_k)) - sketch_dim) / sketch_dim
// V accumulation: standard INT4 with per-head FP16 scales.
void paged_attention_decode_turboquant_lite(
    const Tensor& Q, const Tensor& V_cache,
    Tensor& O,
    const half* K_norms, const half* V_scales,
    const uint8_t* K_sketches, const uint8_t* qjl_matrix,
    const int* block_tables, const int* context_lens,
    int block_size, float scale, int sketch_dim,
    int max_context_len, int sliding_window = 0,
    float softcap = 0.0f, cudaStream_t stream = nullptr,
    int max_blocks_per_seq = 0,
    int n_sinks = 0);

// Split-K scratch buffer accessor (for use by FP8/INT8 launcher TUs).
// Returns pointer + size. Either can be nullptr/0 if unset.
void paged_attention_get_splitk_scratch(void** out_ptr, size_t* out_size);

// Launch the split-K reduce kernel (shared across FP16/FP8/INT8).
void paged_attention_launch_reduce(float* partial, half* O,
                                   int batch_size, int n_heads, int head_dim,
                                   int num_splits, cudaStream_t stream);

} // namespace imp
