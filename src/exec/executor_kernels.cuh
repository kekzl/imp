#pragma once

// Executor kernel declarations; include from the .cu that launches or takes the address of one.
// Split from executor_kernels.h so host-only executor TUs compile as C++ (#2209).

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// CUDA kernels used by the executor
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256) void broadcast_add_bias_fp16_kernel(half* __restrict__ out,
                                                                      const half* __restrict__ bias, int rows,
                                                                      int cols);

__global__ __launch_bounds__(256) void scale_fp16_kernel(half* __restrict__ data, half scale, int64_t n);

__global__ __launch_bounds__(256) void elementwise_add_fp16_kernel(half* __restrict__ a,
                                                                   const half* __restrict__ b, int64_t n);

__global__ __launch_bounds__(256) void elementwise_add_store_fp16_kernel(const half* __restrict__ a,
                                                                         const half* __restrict__ b,
                                                                         half* __restrict__ out, int64_t n);

__global__ __launch_bounds__(256) void fp32_to_fp16_rowscale_kernel(const float* __restrict__ in,
                                                                    half* __restrict__ out, int rows,
                                                                    int cols);

__global__ __launch_bounds__(512) void rmsnorm_fp32_accum_to_fp16_kernel(
    const half* __restrict__ input, const half* __restrict__ norm_w, float* __restrict__ fp32_accum,
    half* __restrict__ output, int d_model, float eps, float weight_offset);

__global__ __launch_bounds__(256) void fp16_to_fp32_kernel(const half* __restrict__ in,
                                                           float* __restrict__ out, int64_t n);

__global__ __launch_bounds__(256) void elementwise_add_fp32_kernel(float* __restrict__ a,
                                                                   const float* __restrict__ b, int64_t n);

__global__ __launch_bounds__(256) void write_kv_cache_fused_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, half* __restrict__ k_cache_base, half* __restrict__ v_cache_base,
    int block_stride, int row_elems, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_fp8_fused_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, __nv_fp8_e4m3* __restrict__ k_cache_base,
    __nv_fp8_e4m3* __restrict__ v_cache_base, float inv_scale, int block_stride, int row_elems,
    int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_int8_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, int8_t* __restrict__ k_cache_base,
    int8_t* __restrict__ v_cache_base, half* __restrict__ k_scale_base, half* __restrict__ v_scale_base,
    int block_stride, int scale_block_stride, int n_kv_heads, int head_dim, int block_size, int n_tokens,
    int max_blocks_per_seq, int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_int4_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, uint8_t* __restrict__ k_cache_base,
    uint8_t* __restrict__ v_cache_base, half* __restrict__ k_scale_base, half* __restrict__ v_scale_base,
    int block_stride, int scale_block_stride, int n_kv_heads, int head_dim, int block_size, int n_tokens,
    int max_blocks_per_seq, int n_sequences);

// NVFP4 KV cache write: per-token-head-group_of_16 absmax → UE4M3 scale, FP4 E2M1
// nibbles packed 2/byte. Layout matches paged_attention_decode_nvfp4 reader.
__global__ __launch_bounds__(256) void write_kv_cache_nvfp4_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ k_cache_base,        // [block, slot, head, head_dim/2] packed FP4
    uint8_t* __restrict__ v_cache_base,        // same shape
    uint8_t* __restrict__ k_scale_base,        // [block, slot, head, head_dim/16] UE4M3
    uint8_t* __restrict__ v_scale_base,        // same shape
    int block_stride,                          // kKVBlockSize * n_kv_heads * head_dim / 2 (bytes)
    int scale_block_stride,                    // kKVBlockSize * n_kv_heads * (head_dim / 16) (bytes)
    int n_kv_heads, int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences);

// MXFP4-KV write: identical layout to NVFP4 but encodes scales as UE8M0 bytes
// (pure-exponent, 2^(bits-127)) instead of E4M3. Matches paged_attention_decode_mxfp4_kv reader.
__global__ __launch_bounds__(256) void write_kv_cache_mxfp4_kv_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ k_cache_base,        // [block, slot, head, head_dim/2] packed FP4
    uint8_t* __restrict__ v_cache_base,        // same shape
    uint8_t* __restrict__ k_scale_base,        // [block, slot, head, head_dim/16] UE8M0
    uint8_t* __restrict__ v_scale_base,        // same shape
    int block_stride,                          // kKVBlockSize * n_kv_heads * head_dim / 2 (bytes)
    int scale_block_stride,                    // kKVBlockSize * n_kv_heads * (head_dim / 16) (bytes)
    int n_kv_heads, int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences);

// BitDecoding Phase 3c residual write: copies one (K,V) FP16 row pair per
// token into the per-(seq,layer) residual ring slot, replacing a pair of
// per-layer cudaMemcpyAsync calls (the D2D copy engine serialized small
// transfers and dominated decode tg/s, -3x on Qwen3-4B Q8 NVFP4-KV @4K).
// Graph-capture-safe form: resolves the destination ring slot at kernel
// execution time from the device-resident `widx` + the persistent slot
// index. Caller passes the per-(seq_slot,layer) K/V base pointer, not the ring slot (computed inside the
// kernel).
__global__ void residual_kv_write_indirect_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* __restrict__ residual_k_layer_seq_base,    // (slot, layer, K=0) base
    half* __restrict__ residual_v_layer_seq_base,    // (slot, layer, V=1) base
    const int* __restrict__ d_residual_widx_ptr,     // [max_seqs] device array
    int seq_slot,                                     // index into d_residual_widx_ptr
    int slot_elems);

// Graph-capture-safe MULTI-seq variant (#1708): the per-batch-index
// destination is computed on device from the layer base, per-seq stride
// and ring index, so nothing is frozen at capture time. Replaces a form
// that built a host device-pointer array with cudaMallocAsync/Free per
// call inside the captured region, where a replay wrote through freed memory.
__global__ void residual_kv_write_multi_indirect_kernel(
    const half* __restrict__ k_in,                // [n_tokens, slot_elems]
    const half* __restrict__ v_in,                // [n_tokens, slot_elems]
    half* __restrict__ residual_k_layer_base,     // slot 0 of this layer, K
    half* __restrict__ residual_v_layer_base,     // slot 0 of this layer, V
    int64_t seq_stride_elems,                     // half-elements between seq slots
    const int* __restrict__ d_seq_slots,          // [n_tokens] residual slot per batch idx
    const int* __restrict__ d_residual_widx_ptr,  // [max_seqs] device array
    int slot_elems);

// Advance the residual ring state for one slot. Single-thread kernel:
//   d_widx[slot] = (d_widx[slot] + 1) % residual_n_tokens
//   d_fc[slot]   = min(d_fc[slot] + 1, residual_n_tokens)
__global__ void advance_residual_state_kernel(
    int* __restrict__ d_widx,
    int* __restrict__ d_fc,
    int slot,
    int residual_n_tokens);

// Same, for every sequence in a multi-seq decode step (#1708). The
// single-slot form above is reached only when state.kv_seq_id>=0 (N==1
// only), so a multi-seq step advanced the ring on the HOST, which a graph replay never ran.
__global__ void advance_residual_state_multi_kernel(int* __restrict__ d_widx, int* __restrict__ d_fc,
                                                    const int* __restrict__ d_seq_slots, int n_seqs,
                                                    int residual_n_tokens);

__global__ __launch_bounds__(256) void write_kv_cache_rope_fused_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, half* __restrict__ k_cache_base, half* __restrict__ v_cache_base,
    int block_stride, int row_elems, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences,
    int n_kv_heads, int head_dim, float theta, float inv_scaling, int rope_pairs, bool neox,
    const float* __restrict__ longrope_inv_freqs);

__global__ __launch_bounds__(256) void rope_q_only_fp16_kernel(half* __restrict__ Q,
                                                               const int* __restrict__ positions, int n_heads,
                                                               int head_dim, float theta, float inv_scaling,
                                                               int rope_pairs, bool neox,
                                                               const float* __restrict__ longrope_inv_freqs);

__global__ __launch_bounds__(256) void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n);

__global__ __launch_bounds__(256) void logit_softcap_fp32_kernel(float* __restrict__ data, float softcap,
                                                                 float inv_softcap, int64_t n);

__global__ __launch_bounds__(256) void fp32_to_fp16_kernel(const float* __restrict__ in,
                                                           half* __restrict__ out, int64_t n);

// GDN/Qwen3.5/3.6 attention output-gate split (interleaved layout
// [Q_h0(hd)|Gate_h0(hd)|...]): splits per head into two contiguous
// [n,nh*hd] buffers in one launch, replacing an nh*2 cudaMemcpy2DAsync
// loop. Instantiated for half and __nv_bfloat16.
template <typename T>
__global__ __launch_bounds__(256) void attn_gate_split_interleaved_kernel(
    const T* __restrict__ src, T* __restrict__ q_dst, T* __restrict__ gate_dst, int n_tokens, int nh,
    int hd, int q_out_dim);

}  // namespace imp
