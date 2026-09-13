#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Paged attention decode (single query/sequence). Q:[batch,1,n_heads,hd]. block_tables:
// [batch,max_blocks] int32. context_lens:[batch] int32. sliding_window: 0=off, >0=last N KV.
// n_sinks (StreamingLLM): >0 also attends [0,n_sinks); needs sliding_window>0 and
// ctx_len>n_sinks+sliding_window, else behaves as plain sliding-window/full attention.
// softcap: 0=off, >0 applies tanh(score/cap)*cap. attn_sinks (gpt-oss #547): per-head learned
// logits [n_heads] FP16, virtual softmax column (unrelated to n_sinks above).
// v_head_dim: 0=same as head_dim. MLA asymmetric QK/V: pass v_head_dim to read only that many
// V elements per slot (V allocated at head_dim stride) and write O at v_head_dim width.
void paged_attention_decode(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                            const int* block_tables, const int* context_lens, int block_size, float scale,
                            int max_context_len, int sliding_window = 0, float softcap = 0.0f,
                            cudaStream_t stream = nullptr, int max_blocks_per_seq = 0, int n_sinks = 0,
                            const void* attn_sinks = nullptr, int v_head_dim = 0);

// Sets the split-K scratch buffer; must be called before paged_attention_decode if split-K is
// wanted. Size = batch * n_heads * num_splits * (2 + head_dim) * sizeof(float). nullptr disables it.
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

// FP8 E4M3 paged attention decode, on-the-fly dequant. Q:[batch,1,n_heads,hd] FP16.
// K/V_cache:[num_blocks,n_kv_heads,block_size,hd] FP8_E4M3. kv_scale: per-tensor FP32
// (val = fp8_val * kv_scale). NVFP4 4-token variant (attention.paged_nvfp4_multitok), HD=128/256,
// E4M3 scales; num_splits>1 fills `partial` for the reduce. Returns false for unsupported shapes.
bool paged_attention_decode_nvfp4_multitok_launch(const half* Q, const uint8_t* K_cache,
                                                  const uint8_t* V_cache, const uint8_t* K_scales,
                                                  const uint8_t* V_scales, half* O, float* partial,
                                                  const int* block_tables, const int* context_lens,
                                                  int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                                  int block_size, float scale, int max_num_blocks,
                                                  int num_splits, int sliding_window, float softcap,
                                                  const half* attn_sinks, cudaStream_t stream);

// NVFP4 multitok with Q-head grouping: a CTA converts each K/V row once for heads_per_cta Q
// heads of one KV head. HD=128/256. heads_per_cta 0=auto (largest of 4/3/2 dividing GQA ratio),
// 1=not served. Returns false when the shape is not served.
bool paged_attention_nvfp4_multitok_gqa_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                               const uint8_t* K_scales, const uint8_t* V_scales, half* O,
                                               float* partial, const int* block_tables,
                                               const int* context_lens, int batch_size, int n_heads,
                                               int n_kv_heads, int head_dim, int block_size, float scale,
                                               int max_num_blocks, int num_splits, int sliding_window,
                                               float softcap, const half* attn_sinks, int heads_per_cta,
                                               cudaStream_t stream);
int paged_attention_nvfp4_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested);

// HD=128 FP8 decode with four tokens per warp iteration
// (attention.paged_fp8_multitok, attention_paged_fp8_multitok.cu). Called by
// paged_attention_decode_fp8 when the knob is on and split-K is off.
void paged_attention_decode_fp8_multitok_hd128(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                               half* O, const int* block_tables, const int* context_lens,
                                               int batch_size, int n_heads, int n_kv_heads, int block_size,
                                               float scale, float kv_scale, int max_num_blocks,
                                               int sliding_window, float softcap, const half* attn_sinks,
                                               cudaStream_t stream);

// HD=128 FP8 decode, 16 lanes/KV row, heads_per_cta Q heads/CTA sharing KV loads. Called before
// the four-token kernel when the knob is on and split-K is off; heads_per_cta 0=auto (largest of
// 5/4/3/2/1 dividing the ratio). Returns false for unsupported shapes (HD!=128, ratio>16).
bool paged_attention_fp8_multitok_gqa_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                             half* O, const int* block_tables, const int* context_lens,
                                             int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                             int block_size, float scale, float kv_scale, int max_num_blocks,
                                             int sliding_window, float softcap, const half* attn_sinks,
                                             int heads_per_cta, cudaStream_t stream);
int paged_attention_fp8_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested);

// F16 decode, 4 tokens/warp iteration, heads_per_cta Q heads/CTA sharing KV loads (HD=128/256,
// GQA ratio 1..8). Called when the knob is on, split-K is off and n_sinks==0; heads_per_cta
// 0=auto (largest of 4/2/1 dividing the ratio, HD=256 caps at 2). Returns false otherwise.
bool paged_attention_decode_f16_multitok_launch(const half* Q, const half* K_cache, const half* V_cache,
                                                half* O, const int* block_tables, const int* context_lens,
                                                int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                                int block_size, float scale, int max_num_blocks,
                                                int sliding_window, float softcap, const half* attn_sinks,
                                                int heads_per_cta, cudaStream_t stream);

// Split-K instance of the F16 multitok kernel: grid (batch, n_kv_heads x
// groups, num_splits), one (m, l, o) partial per head in the shared reduce
// layout. Same shape rules as the plain launch; num_splits from the caller.
bool paged_attention_splitk_f16_multitok_launch(const half* Q, const half* K_cache, const half* V_cache,
                                                float* partial, const int* block_tables,
                                                const int* context_lens, int batch_size, int n_heads,
                                                int n_kv_heads, int head_dim, int block_size, float scale,
                                                int max_num_blocks, int num_splits, int sliding_window,
                                                float softcap, int heads_per_cta, cudaStream_t stream);

// Q heads per CTA the F16 multitok kernels will use for this shape (0 =
// shape not served): the largest of 4 / 2 / 1 dividing the GQA ratio, HD=256
// capped at 2; `requested` (1/2/4) wins when it divides.
int paged_attention_f16_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested);

// F16 decode, 4 tokens/warp iteration, heads_per_cta Q heads/CTA sharing KV loads (HD=128/256,
// GQA ratio 1..8). Called when the knob is on, split-K is off and n_sinks==0; heads_per_cta
// 0=auto (largest of 4/2/1 dividing the ratio, HD=256 caps at 2). Returns false otherwise.
bool paged_attention_decode_f16_multitok_launch(const half* Q, const half* K_cache, const half* V_cache,
                                                half* O, const int* block_tables, const int* context_lens,
                                                int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                                int block_size, float scale, int max_num_blocks,
                                                int sliding_window, float softcap, const half* attn_sinks,
                                                int heads_per_cta, cudaStream_t stream);

void paged_attention_decode_fp8(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                const int* block_tables, const int* context_lens, int block_size, float scale,
                                float kv_scale, int max_context_len, int sliding_window = 0,
                                float softcap = 0.0f, cudaStream_t stream = nullptr,
                                int max_blocks_per_seq = 0, int n_sinks = 0,
                                const void* attn_sinks = nullptr);

// INT8 dp4a paged attention decode, per-head scales. Q:[batch,1,n_heads,hd] FP16.
// K/V_cache:[num_blocks,block_size,n_kv_heads,hd] INT8. K/V_scales:[...,n_kv_heads] FP16.
void paged_attention_decode_int8(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                 const half* K_scales, const half* V_scales, const int* block_tables,
                                 const int* context_lens, int block_size, float scale, int max_context_len,
                                 int sliding_window = 0, float softcap = 0.0f, cudaStream_t stream = nullptr,
                                 int max_blocks_per_seq = 0, int n_sinks = 0,
                                 const void* attn_sinks = nullptr);

// INT4 paged attention decode, packed 2/byte, per-head scales. Q:[batch,1,n_heads,hd] FP16.
// K/V_cache:[num_blocks,block_size,n_kv_heads,hd/2] packed uint8. K/V_scales FP16.
void paged_attention_decode_int4(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                 const half* K_scales, const half* V_scales, const int* block_tables,
                                 const int* context_lens, int block_size, float scale, int max_context_len,
                                 int sliding_window = 0, float softcap = 0.0f, cudaStream_t stream = nullptr,
                                 int max_blocks_per_seq = 0, int n_sinks = 0,
                                 const void* attn_sinks = nullptr);

// NVFP4 paged attention decode: packed FP4 (E2M1), per-token-head-group_of_16 UE4M3 scales.
// Q:[batch,1,n_heads,hd] FP16. K/V_cache:[num_blocks,block_size,n_kv_heads,hd/2] packed uint8.
// K/V_scales:[...,hd/16] UE4M3 bytes.
void paged_attention_decode_nvfp4(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                  const uint8_t* K_scales, const uint8_t* V_scales, const int* block_tables,
                                  const int* context_lens, int block_size, float scale, int max_context_len,
                                  int sliding_window = 0, float softcap = 0.0f, cudaStream_t stream = nullptr,
                                  int max_blocks_per_seq = 0, int n_sinks = 0,
                                  const void* attn_sinks = nullptr);

// MXFP4-KV paged attention decode: same layout as NVFP4 but scales are UE8M0 instead of E4M3
// (design memo 3.1.2). Q:[batch,1,n_heads,hd] FP16. K/V_cache packed uint8 [.../hd/2].
// K/V_scales:[...,hd/16] UE8M0 bytes.
void paged_attention_decode_mxfp4_kv(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                     const uint8_t* K_scales, const uint8_t* V_scales,
                                     const int* block_tables, const int* context_lens, int block_size,
                                     float scale, int max_context_len, int sliding_window = 0,
                                     float softcap = 0.0f, cudaStream_t stream = nullptr,
                                     int max_blocks_per_seq = 0, int n_sinks = 0,
                                     const void* attn_sinks = nullptr);

// BitDecoding-style TC variant of paged_attention_decode_nvfp4: routes QK dot through
// nvcuda::wmma 16x16x16 MMA.
// Phase 3b residual: optional FP16 ring of the newest residual_count tokens (write-through
// duplicate of the paged tail). Splits attention into paged [0,ctx_len-residual_count) (NVFP4
// dequant) and residual [ctx_len-residual_count,ctx_len) (direct FP16), merged via the same
// online-softmax invariant.
// Single-seq (batch_size==1): pass K/V_residual + residual_count/write_idx as scalars.
// Multi-seq: pass K/V_residual_base (slot 0), residual_seq_stride_elems, and device arrays
// d_residual_seq_slots/_counts/_write_idxes of length batch_size; slot = base + slot*stride.
// Residual layout per slot: [residual_n_tokens,n_kv_heads,hd] half, ring-indexed; i-th most
// recent token at slot (write_idx+residual_n_tokens-residual_count+i) % residual_n_tokens.
// Split-K forced off when residual is active (only the non-split kernel reads it).
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
                                     // Graph-safe per-slot ring state (KVCacheManager's persistent device
                                     // buffers). When set along
                                     // with d_residual_seq_slots, the kernel reads fc/widx via slot
                                     // indirection (capture-safe, nothing
                                     // rebuilt per step); replaces d_residual_counts/_write_idxes when both
                                     // pairs are set.
                                     const int* d_residual_fc_per_slot = nullptr,
                                     const int* d_residual_widx_per_slot = nullptr);

// Split-K scratch buffer accessor (for use by FP8/INT8 launcher TUs).
// Returns pointer + size. Either can be nullptr/0 if unset.
void paged_attention_get_splitk_scratch(void** out_ptr, size_t* out_size);

// Single source of truth for whether paged DECODE applies attn sinks (gpt-oss #547) for a KV
// dtype - a property of the kernels, not the model. Wire a new dtype here too, or the fallback
// silently misses it (#1339, #1345). FP16 sinks since #547, FP8 since #1346.
bool paged_attention_applies_sinks(QType kv_dtype);

// Which head_dims a KV dtype's paged DECODE launchers template for (#1674); same wire-a-dtype
// contract as paged_attention_applies_sinks. Unknown dtype returns true (never refuse unchecked);
// the resolver falls back to FP16 KV so an unserved head_dim (throws, SETTLED S-22) never reaches
// a launcher.
bool paged_attention_serves_head_dim(QType kv_dtype, int head_dim);

// True only if FP8 KV gets the FAST paged decode kernels at this head_dim; otherwise
// LAUNCH_FP8_FALLBACK runs silently. Fast kernels (fp8_multitok[_gqa].cu) are head_dim=128
// only: 4 FP8 bytes/lane is what makes one uint32 load/lane work.
constexpr bool paged_fp8_decode_has_fast_kernel(int head_dim) { return head_dim == 128; }

// Terminal for a launcher that has no template for `head_dim`. One function so
// the 17 sites are one line each and the message is written once.
[[noreturn]] void paged_attention_unsupported_head_dim(const char* fn, int head_dim);

// Launch the split-K reduce kernel (shared across FP16/FP8/INT8).
void paged_attention_launch_reduce(float* partial, half* O, int batch_size, int n_heads, int head_dim,
                                   int num_splits, cudaStream_t stream,
                                   const half* attn_sinks = nullptr);

}  // namespace imp
