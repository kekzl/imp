#pragma once

#include "core/tensor.h"
#include "model/model_config.h"  // QType (Q6_K, Q4_0, etc.)
#include "compute/gemm.h"        // block_q8_1

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <unordered_map>

namespace imp {

// Forward declarations for legacy gemm_dispatch parameters
struct FP8CacheEntry;
struct NvFP4QuantResult;
struct CutlassNvFP4Weight;
struct CutlassMxFP4Weight;

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

__global__ __launch_bounds__(256) void fp32_accum_add_fp16_kernel(float* __restrict__ accum,
                                                                  const half* __restrict__ branch, int64_t n);

__global__ __launch_bounds__(256) void fp32_to_fp16_rowscale_kernel(const float* __restrict__ in,
                                                                    half* __restrict__ out, int rows,
                                                                    int cols);

__global__ __launch_bounds__(512) void rmsnorm_fp32_accum_to_fp16_kernel(
    const half* __restrict__ input, const half* __restrict__ norm_w, float* __restrict__ fp32_accum,
    half* __restrict__ output, int d_model, float eps, float weight_offset);

// FP32-input variant of rmsnorm_fp32_accum_to_fp16_kernel — used when the
// upstream op (e.g. attention output projection) keeps its result in FP32 to
// avoid cuBLAS's internal FP16 output truncation.
__global__ __launch_bounds__(512) void rmsnorm_fp32in_fp32_accum_to_fp16_kernel(
    const float* __restrict__ input, const half* __restrict__ norm_w, float* __restrict__ fp32_accum,
    half* __restrict__ output, int d_model, float eps, float weight_offset);

__global__ __launch_bounds__(256) void fp16_to_fp32_kernel(const half* __restrict__ in,
                                                           float* __restrict__ out, int64_t n);

__global__ __launch_bounds__(256) void elementwise_add_fp32_kernel(float* __restrict__ a,
                                                                   const float* __restrict__ b, int64_t n);

__global__ __launch_bounds__(256) void write_kv_cache_kernel(const half* __restrict__ data_in,
                                                             const int* __restrict__ positions,
                                                             const int* __restrict__ block_tables,
                                                             half* __restrict__ cache_base, int block_stride,
                                                             int row_elems, int block_size, int n_tokens,
                                                             int max_blocks_per_seq, int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_fused_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, half* __restrict__ k_cache_base, half* __restrict__ v_cache_base,
    int block_stride, int row_elems, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_fp8_kernel(
    const half* __restrict__ data_in, const int* __restrict__ positions, const int* __restrict__ block_tables,
    __nv_fp8_e4m3* __restrict__ cache_base, float inv_scale, int block_stride, int row_elems, int block_size,
    int n_tokens, int max_blocks_per_seq, int n_sequences);

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

// BitDecoding Phase 3c residual write — copies one (K, V) FP16 row pair per
// token into the per-(seq, layer) residual ring slot. Replaces a pair of
// `cudaMemcpyAsync` we used to launch per layer; the device-to-device copy
// engine path serialized small transfers and dominated decode tg/s
// (-3× regression on Qwen3-4B Q8 NVFP4-KV bench at 4K ctx).
//
// Multi-seq form: pass n_tokens > 1 with the device pointer arrays.
__global__ void residual_kv_write_multi_kernel(
    const half* __restrict__ k_in,                  // [n_tokens, slot_elems]
    const half* __restrict__ v_in,                  // [n_tokens, slot_elems]
    half* const* __restrict__ residual_k_dst_ptrs,  // [n_tokens] device array of dst pointers
    half* const* __restrict__ residual_v_dst_ptrs,  // [n_tokens] device array of dst pointers
    int slot_elems);

// Graph-capture-safe variant. Resolves the destination ring slot at kernel
// execution time by reading the device-resident `widx` (one int per seq slot)
// pointer + the persistent slot index. Caller passes the per-(seq_slot, layer)
// K/V base pointer (NOT the ring slot — that's computed inside the kernel).
__global__ void residual_kv_write_indirect_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* __restrict__ residual_k_layer_seq_base,    // (slot, layer, K=0) base
    half* __restrict__ residual_v_layer_seq_base,    // (slot, layer, V=1) base
    const int* __restrict__ d_residual_widx_ptr,     // [max_seqs] device array
    int seq_slot,                                     // index into d_residual_widx_ptr
    int slot_elems);

// Advance the residual ring state for one slot. Single-thread kernel:
//   d_widx[slot] = (d_widx[slot] + 1) % residual_n_tokens
//   d_fc[slot]   = min(d_fc[slot] + 1, residual_n_tokens)
__global__ void advance_residual_state_kernel(
    int* __restrict__ d_widx,
    int* __restrict__ d_fc,
    int slot,
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

__global__ __launch_bounds__(256) void add_fp16_bias_to_fp32_kernel(float* __restrict__ data,
                                                                    const half* __restrict__ bias,
                                                                    int n_tokens, int n_cols);

__global__ __launch_bounds__(256) void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n);

__global__ __launch_bounds__(256) void logit_softcap_fp32_kernel(float* __restrict__ data, float softcap,
                                                                 float inv_softcap, int64_t n);

__global__ __launch_bounds__(256) void fp32_to_fp16_kernel(const float* __restrict__ in,
                                                           half* __restrict__ out, int64_t n);

// ---------------------------------------------------------------------------
// GDN/Qwen3.5/3.6 attention output-gate split (interleaved layout):
// Source row layout: [Q_h0(hd) | Gate_h0(hd) | Q_h1(hd) | Gate_h1(hd) | ...]
// Splits per head into two contiguous [n, nh*hd] buffers in one launch,
// replacing the nh × 2 cudaMemcpy2DAsync loop in executor_attention.cu
// (~656 D2D copies per decode step on Qwen3.5 GDN — Finding 2).
// Element type is templated; instantiated for half (FP16) and __nv_bfloat16.
template <typename T>
__global__ __launch_bounds__(256) void attn_gate_split_interleaved_kernel(
    const T* __restrict__ src, T* __restrict__ q_dst, T* __restrict__ gate_dst, int n_tokens, int nh,
    int hd, int q_out_dim);

// Host launcher: dispatches the right T based on dtype size (2 = half/bf16).
void attn_gate_split_interleaved(const void* src, void* q_dst, void* gate_dst, int n_tokens, int nh, int hd,
                                 int q_out_dim, int element_bytes, cudaStream_t stream);

// ---------------------------------------------------------------------------
// dp4a GEMV helpers (shared by executor_forward.cu and executor_kernels.cu)
// ---------------------------------------------------------------------------

// Returns true if the quant type supports dp4a (Q8_1-input) GEMV kernels.
inline bool is_dp4a_qtype(QType qt) {
    return qt == QType::Q6_K || qt == QType::Q8_0 || qt == QType::Q4_0 || qt == QType::Q4_K ||
           qt == QType::Q5_K || qt == QType::Q2_K || qt == QType::Q3_K;
}

// Dispatch dp4a GEMV by quant type: y = W @ q8_1 (FP16 output).
void dispatch_dp4a_gemv(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M,
                        int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Host-side helper functions
// ---------------------------------------------------------------------------

void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream);

// Device-to-device copy as a kernel launch (stream-async, ~10 us host cost)
// instead of cudaMemcpyAsync's WDDM DMA submission (~165 us blocked host
// time per call on this WSL2 host). Use for per-layer copies on the decode
// hot path; falls back to cudaMemcpyAsync for unaligned buffers.
void device_copy_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);

// Pipelined batched-decode chain advance (see decode_pipeline_advance.cu):
// token_ids[i] = slot i's sampled token, positions[i]++, context_lens[i]++,
// per-row history append (penalty rows), plus n_patches block-table scatter
// writes (offsets are flat indices into the pool block-table region; patch/
// pos arrays must be device-readable — the engine passes mapped pinned
// memory). One tiny single-block launch.
void decode_pipeline_advance(int n_rows, const int32_t* slot_tokens, size_t slot_stride_bytes,
                             int32_t* d_token_ids, int* d_positions, int* d_context_lens,
                             int* d_block_tables, int n_patches, const int* d_patch_offsets,
                             const int* d_patch_values, int32_t* d_hist_base, int hist_stride,
                             const int* d_hist_pos, cudaStream_t stream);

void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream);

// Fused 3-way bias add: out_a += bias_a, out_b += bias_b, out_c += bias_c in one launch.
// Skips any output where bias.data == nullptr.
void add_bias_3way(Tensor& out_a, const Tensor& bias_a, Tensor& out_b, const Tensor& bias_b, Tensor& out_c,
                   const Tensor& bias_c, cudaStream_t stream);

// Fused residual add + RMSNorm: hidden += residual; output = rmsnorm(hidden, weight).
// Saves 1 kernel launch + 1 DRAM round-trip vs separate add + norm.
void residual_add_rmsnorm(Tensor& hidden, const Tensor& residual, const Tensor& weight, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset = 0.0f);

// Fused add-store + RMSNorm: hidden = a + b; hidden = rmsnorm(hidden, weight).
// Replaces: elementwise_add_store(a, b, h) + rmsnorm(h, w, no) + memcpy(h, no).
// 3 ops → 1 kernel. Used by sandwich-norm post-attention and post-FFN paths.
void add_rmsnorm_inplace(const Tensor& a, const Tensor& b, Tensor& hidden, const Tensor& weight, float eps,
                         cudaStream_t stream, float weight_offset = 0.0f);

// Fused RMSNorm + residual add: output = rmsnorm(input, weight) + residual.
// Replaces: rmsnorm(in, w, out) + elementwise_add(out, r).
// 2 ops → 1 kernel. Used by sandwich-norm post-FFN path.
void rmsnorm_add_residual(const Tensor& input, const Tensor& weight, const Tensor& residual, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset = 0.0f);

Tensor slice_rows(const Tensor& buf, int n_tokens);

// GemmContext forward decl — defined in gemm_context.h.
// The legacy gemm_dispatch free function is now a file-local uncached
// fallback (gemm_dispatch_uncached_fallback in executor_kernels.cu).
struct GemmContext;

// MMVQ scratch buffer prewarm + hot-path getter live in gemm_scratch.h since
// R5 Slice 8.6 (TU hoist).

}  // namespace imp
