#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace imp {

// Gather contiguous K (or V) of past tokens [0, n_past) from the paged FP16
// cache into a flat buffer.
//
// dst:         FP16 [n_past, nkv, hd] contiguous (caller-allocated)
// src:         FP16 paged base pointer (KVCache::k_ptr / v_ptr)
//              Layout: [num_blocks, block_size, nkv, hd]
// block_table: device pointer, [ceil(n_past/block_size)] int32
// n_past:      number of tokens to gather (positions 0..n_past-1)
// block_size:  KV cache block_size (e.g. 16)
// nkv:         number of KV heads
// hd:          head_dim
// d_n_past:    optional DEVICE int overriding n_past as the gather bound.
//              When set, the host n_past only sizes the grid (an upper
//              capacity) and the kernel reads the real token count from
//              device — required for CUDA-graph replay across context growth
//              (#847 graph-captured verify), where the graph bakes the grid
//              but the context keeps growing between replays.
void paged_kv_gather_fp16(half* dst, const half* src, const int* block_table,
                          int n_past, int block_size, int nkv, int hd, cudaStream_t stream,
                          const int* d_n_past = nullptr);

// FP8 E4M3 paged → FP16 flat with per-tensor scalar dequant: dst[i] = (half)((float)src[i] * kv_scale).
// Same indexing as paged_kv_gather_fp16; src is FP8 E4M3 (1 byte / elem).
void paged_kv_gather_fp8_to_fp16(half* dst, const __nv_fp8_e4m3* src, const int* block_table,
                                 float kv_scale, int n_past, int block_size, int nkv, int hd,
                                 cudaStream_t stream, const int* d_n_past = nullptr);

// NVFP4 paged → FP16 flat with per-group-of-16 UE4M3 scale dequant.
// src_packed layout: [num_blocks, block_size, nkv, hd/2] uint8 (2 FP4 nibbles/byte).
// src_scales layout: [num_blocks, block_size, nkv, hd/16] uint8 (UE4M3 bytes).
// Matches `write_kv_cache_nvfp4_kernel` writes / `paged_attention_decode_nvfp4` reads.
// Used by chunked prefill to materialize past-chunk K/V into FP16 for cuBLAS attention.
void paged_kv_gather_nvfp4_to_fp16(half* dst, const uint8_t* src_packed,
                                   const uint8_t* src_scales, const int* block_table,
                                   int n_past, int block_size, int nkv, int hd,
                                   cudaStream_t stream, const int* d_n_past = nullptr);

// MXFP4-KV paged → FP16 flat with per-group-of-16 UE8M0 scale dequant.
// Layout identical to NVFP4; only the scale byte semantics differ (UE8M0 vs UE4M3).
// Matches `write_kv_cache_mxfp4_kv_kernel` / `paged_attention_decode_mxfp4_kv`.
void paged_kv_gather_mxfp4_kv_to_fp16(half* dst, const uint8_t* src_packed,
                                       const uint8_t* src_scales, const int* block_table,
                                       int n_past, int block_size, int nkv, int hd,
                                       cudaStream_t stream, const int* d_n_past = nullptr);

// INT4 paged → FP16 flat with per-head FP16 scale dequant (symmetric, range [-8,7]).
// src_packed layout: [num_blocks, block_size, nkv, hd/2] uint8 (low nibble=even d, high=odd d).
// src_scales layout: [num_blocks, block_size, nkv]      FP16 (one scale per head per token).
// Matches `write_kv_cache_int4_kernel`. Used by chunked prefill for INT4 KV.
void paged_kv_gather_int4_to_fp16(half* dst, const uint8_t* src_packed,
                                  const half* src_scales, const int* block_table,
                                  int n_past, int block_size, int nkv, int hd,
                                  cudaStream_t stream, const int* d_n_past = nullptr);

// INT8 paged → FP16 flat with per-head FP16 scale dequant (symmetric, range [-127,127]).
// src layout:        [num_blocks, block_size, nkv, hd] int8.
// src_scales layout: [num_blocks, block_size, nkv]     FP16 (one scale per head per token).
// Matches `write_kv_cache_int8_kernel`. Used by chunked prefill for INT8 KV.
void paged_kv_gather_int8_to_fp16(half* dst, const int8_t* src, const half* src_scales,
                                  const int* block_table, int n_past, int block_size, int nkv, int hd,
                                  cudaStream_t stream, const int* d_n_past = nullptr);

// Append the current chunk's contiguous FP16 K (or V) rows behind the gathered
// past rows, at a DEVICE-computed row offset: dst[*d_past_len + i] = src[i]
// for i in [0, n). Replaces the host-offset cudaMemcpyAsync in the chunked
// continuation path for graph-captured verify (#847) — the destination offset
// must track the growing context between graph replays.
void kv_chunk_append_fp16(half* dst, const half* src, const int* d_past_len, int n,
                          int row_elems, cudaStream_t stream);

}  // namespace imp
