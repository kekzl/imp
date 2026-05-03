#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Naive reference attention for debugging.
// Q/K/V/O are [seq_len, heads * head_dim] row-major with interleaved heads.
void naive_attention_prefill(const half* Q, const half* K, const half* V, half* O, int seq_len, int n_heads,
                             int n_kv_heads, int head_dim, float scale, float softcap, cudaStream_t stream,
                             int sliding_window = 0);

}  // namespace imp
