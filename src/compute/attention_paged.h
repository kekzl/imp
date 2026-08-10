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
// attn_sinks (gpt-oss, #547): per-head learned sink logits [n_heads] FP16 —
//          virtual extra softmax column (denominator += exp(sink - max),
//          column dropped). Unrelated to the positional n_sinks above.
// v_head_dim: 0 = same as head_dim (standard path). For MLA with asymmetric
// QK (head_dim=192) vs V (v_head_dim=128), pass v_head_dim=128 to read only
// v_head_dim elements from each V slot and write O as [batch, 1, n_heads, v_head_dim].
// V slots are allocated with head_dim-sized strides (over-allocation); the kernel
// reads only v_head_dim elements. Non-MLA models: leave at 0.
void paged_attention_decode(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                            const int* block_tables, const int* context_lens, int block_size, float scale,
                            int max_context_len, int sliding_window = 0, float softcap = 0.0f,
                            cudaStream_t stream = nullptr, int max_blocks_per_seq = 0, int n_sinks = 0,
                            const void* attn_sinks = nullptr, int v_head_dim = 0);

// Set split-K scratch buffer for paged attention. Must be called before
// paged_attention_decode if split-K is desired. The scratch buffer holds
// partial softmax state: size = batch * n_heads * num_splits * (2 + head_dim) * sizeof(float).
// Pass nullptr to disable split-K.
void paged_attention_set_splitk_scratch(void* ptr, size_t size);

// Token-tiled FP8 split-K kernel (attention_paged_fp8_tile.cu). Dispatched from
// paged_attention_decode_fp8 for head_dim=128 / block_size % 16 == 0; writes the same
// partial_out layout as the pipeline kernel (reduce kernel shared).
bool paged_attention_splitk_fp8_tile_supported(int head_dim, int block_size);
void paged_attention_splitk_fp8_tile_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                            float* partial_out, const int* block_tables,
                                            const int* context_lens, int batch_size, int n_heads,
                                            int n_kv_heads, int block_size, float scale, float kv_scale,
                                            int max_num_blocks, int num_splits, int sliding_window,
                                            float softcap, cudaStream_t stream);

// GQA-batched tile variant: one block per KV head computes all G Q heads from a
// shared smem tile (L2 KV traffic /G). Grid.y = n_kv_heads; the launcher's split
// count should be raised accordingly (see paged_attention_decode_fp8).
bool paged_attention_splitk_fp8_tile_gqa_supported(int head_dim, int block_size, int n_heads,
                                                   int n_kv_heads);
int paged_attention_splitk_fp8_tile_gqa_splits(int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                               int block_size, int max_context_len);
void paged_attention_splitk_fp8_tile_gqa_launch(const half* Q, const uint8_t* K_cache,
                                                const uint8_t* V_cache, float* partial_out,
                                                const int* block_tables, const int* context_lens,
                                                int batch_size, int n_heads, int n_kv_heads, int block_size,
                                                float scale, float kv_scale, int max_num_blocks,
                                                int num_splits, int sliding_window, float softcap,
                                                cudaStream_t stream);

// FP8 E4M3 Paged attention for decode: KV cache stored in FP8 with on-the-fly dequant.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, n_kv_heads, block_size, head_dim] FP8_E4M3
// O: [batch, 1, n_heads, head_dim] FP16
// kv_scale: per-tensor FP32 scale for FP8 dequantization (val = fp8_val * kv_scale)
void paged_attention_decode_fp8(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                const int* block_tables, const int* context_lens, int block_size, float scale,
                                float kv_scale, int max_context_len, int sliding_window = 0,
                                float softcap = 0.0f, cudaStream_t stream = nullptr,
                                int max_blocks_per_seq = 0, int n_sinks = 0,
                                const void* attn_sinks = nullptr);

// INT8 dp4a Paged attention for decode: KV cache stored in INT8 with per-head scales.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, block_size, n_kv_heads, head_dim] INT8
// K_scales/V_scales: [num_blocks, block_size, n_kv_heads] FP16 per-head scales
// O: [batch, 1, n_heads, head_dim] FP16
void paged_attention_decode_int8(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                 const half* K_scales, const half* V_scales, const int* block_tables,
                                 const int* context_lens, int block_size, float scale, int max_context_len,
                                 int sliding_window = 0, float softcap = 0.0f, cudaStream_t stream = nullptr,
                                 int max_blocks_per_seq = 0, int n_sinks = 0);

// INT4 Paged attention for decode: KV cache stored in packed INT4 with per-head scales.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, block_size, n_kv_heads, head_dim/2] packed uint8
// K_scales/V_scales: [num_blocks, block_size, n_kv_heads] FP16 per-head scales
// O: [batch, 1, n_heads, head_dim] FP16
void paged_attention_decode_int4(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                 const half* K_scales, const half* V_scales, const int* block_tables,
                                 const int* context_lens, int block_size, float scale, int max_context_len,
                                 int sliding_window = 0, float softcap = 0.0f, cudaStream_t stream = nullptr,
                                 int max_blocks_per_seq = 0, int n_sinks = 0);

// NVFP4 Paged attention for decode: KV cache stored as packed FP4 (E2M1) with
// per-token-head-group_of_16 UE4M3 (FP8 E4M3) scales.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, block_size, n_kv_heads, head_dim/2] packed uint8
// K_scales/V_scales: [num_blocks, block_size, n_kv_heads, head_dim/16] UE4M3 bytes
// O: [batch, 1, n_heads, head_dim] FP16
void paged_attention_decode_nvfp4(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                  const uint8_t* K_scales, const uint8_t* V_scales, const int* block_tables,
                                  const int* context_lens, int block_size, float scale, int max_context_len,
                                  int sliding_window = 0, float softcap = 0.0f,
                                  cudaStream_t stream = nullptr, int max_blocks_per_seq = 0, int n_sinks = 0);

// MXFP4-KV paged attention for decode: same layout as NVFP4 but scales are
// UE8M0 bytes instead of E4M3. Structurally identical to NVFP4 per design
// memo §3.1.2 — only the scale byte semantics differ.
// Q: [batch, 1, n_heads, head_dim] FP16
// K_cache/V_cache: [num_blocks, block_size, n_kv_heads, head_dim/2] packed uint8
// K_scales/V_scales: [num_blocks, block_size, n_kv_heads, head_dim/16] UE8M0 bytes
// O: [batch, 1, n_heads, head_dim] FP16
void paged_attention_decode_mxfp4_kv(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                     const uint8_t* K_scales, const uint8_t* V_scales,
                                     const int* block_tables, const int* context_lens, int block_size,
                                     float scale, int max_context_len, int sliding_window = 0,
                                     float softcap = 0.0f, cudaStream_t stream = nullptr,
                                     int max_blocks_per_seq = 0, int n_sinks = 0);

// BitDecoding-style TC variant: same signature + semantics as
// paged_attention_decode_nvfp4 but routes the inner Q.K dot through
// nvcuda::wmma 16×16×16 Tensor Core MMA.
//
// Phase 3b residual addendum: optional FP16 ring-buffer holding the newest
// `residual_count` KV tokens (write-through duplicate of paged tail). When
// active, the kernel splits attention into paged tokens [0, ctx_len -
// residual_count) (NVFP4 dequant path) and residual tokens [ctx_len -
// residual_count, ctx_len) (direct FP16 path), merging via the same online-
// softmax invariant as the paged loop.
//
// Two activation forms:
//   1. Single-seq scalar form (batch_size==1):
//      pass K_residual / V_residual + residual_count / residual_write_idx
//      as scalars. d_residual_seq_slots stays nullptr.
//
//   2. Multi-seq array form (batch_size >= 1):
//      pass K_residual_base / V_residual_base = layer-base pointers (slot 0),
//      residual_seq_stride_elems = FP16 elems between slots,
//      d_residual_seq_slots / d_residual_counts / d_residual_write_idxes =
//      device arrays of length batch_size. Kernel reads per blockIdx.x and
//      computes K_for_seq = K_residual_base + slot * stride.
//
// Layout of FP16 residual data per slot:
//   [residual_n_tokens, n_kv_heads, head_dim] half, ring-indexed; the
//   chronologically-i-th most recent token is at slot
//   `(write_idx + residual_n_tokens - residual_count + i) % residual_n_tokens`.
//
// Split-K is forced off when residual is active (residual reads only the
// non-split kernel; split kernel ignores the args).
void paged_attention_decode_nvfp4_tc(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                     const uint8_t* K_scales, const uint8_t* V_scales, const int* block_tables,
                                     const int* context_lens, int block_size, float scale, int max_context_len,
                                     int sliding_window = 0, float softcap = 0.0f,
                                     cudaStream_t stream = nullptr, int max_blocks_per_seq = 0, int n_sinks = 0,
                                     // Single-seq scalar form
                                     const half* K_residual = nullptr, const half* V_residual = nullptr,
                                     int residual_count = 0, int residual_n_tokens = 0,
                                     int residual_write_idx = 0,
                                     // Multi-seq array form (overrides the scalars when d_residual_seq_slots != nullptr)
                                     const half* K_residual_base = nullptr,
                                     const half* V_residual_base = nullptr,
                                     int residual_seq_stride_elems = 0,
                                     const int* d_residual_seq_slots = nullptr,
                                     const int* d_residual_counts = nullptr,
                                     const int* d_residual_write_idxes = nullptr,
                                     // Graph-safe per-slot ring state (KVCacheManager's persistent
                                     // device buffers). When non-null AND d_residual_seq_slots is set,
                                     // kernel reads fc/widx via slot indirection — graph-capture-safe
                                     // because nothing is rebuilt per step. Replaces d_residual_counts/
                                     // d_residual_write_idxes when both pairs are set.
                                     const int* d_residual_fc_per_slot = nullptr,
                                     const int* d_residual_widx_per_slot = nullptr);

// Split-K scratch buffer accessor (for use by FP8/INT8 launcher TUs).
// Returns pointer + size. Either can be nullptr/0 if unset.
void paged_attention_get_splitk_scratch(void** out_ptr, size_t* out_size);

// Does the paged DECODE path for this KV dtype apply learned attention sinks
// (gpt-oss, #547)? This is the single source of truth for the question, and it
// lives here because the answer is a property of the kernels, not of the model.
//
// It exists because the resolver used to answer it with a hand-maintained dtype
// list (#1339, #1345): the sink term reaches the output only if the decode
// kernel takes an `attn_sinks` pointer and hands it to
// crosswarp_reduce_and_write, and every dtype wired up since had to be
// remembered in a second place. Wire a dtype, add it here, and the fallback
// stops firing for it — one edit, not two.
//
// FP16 has applied sinks since #547; FP8 since #1346. The rest route their
// NON-split decode through crosswarp_reduce_and_write with no sink pointer, so
// the shared split-K reduce being wired is not enough to make them correct.
bool paged_attention_applies_sinks(QType kv_dtype);

// Launch the split-K reduce kernel (shared across FP16/FP8/INT8).
void paged_attention_launch_reduce(float* partial, half* O, int batch_size, int n_heads, int head_dim,
                                   int num_splits, cudaStream_t stream,
                                   const half* attn_sinks = nullptr);

}  // namespace imp
