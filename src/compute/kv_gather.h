#pragma once

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
void paged_kv_gather_fp16(half* dst, const half* src, const int* block_table,
                          int n_past, int block_size, int nkv, int hd, cudaStream_t stream);

// FP8 E4M3 paged → FP16 flat with per-tensor scalar dequant: dst[i] = (half)((float)src[i] * kv_scale).
// Same indexing as paged_kv_gather_fp16; src is FP8 E4M3 (1 byte / elem).
void paged_kv_gather_fp8_to_fp16(half* dst, const __nv_fp8_e4m3* src, const int* block_table,
                                 float kv_scale, int n_past, int block_size, int nkv, int hd,
                                 cudaStream_t stream);

}  // namespace imp
