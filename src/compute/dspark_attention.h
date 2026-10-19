#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace imp {

// Non-causal attention of `rows` query rows (<= kDSparkMaxRows) over kv_len keys, head_dim 128, GQA
// (n_heads a multiple of n_kv_heads, group <= 8): Q [rows, n_heads * 128], K/V [kv_len, n_kv_heads * 128],
// O [rows, n_heads * 128], all FP16 row-major. Split-K over keys with an online softmax, then one combine
// launch; no host sync. `work` holds dspark_attention_work_floats(...) floats.
constexpr int kDSparkMaxRows = 8;
size_t dspark_attention_work_floats(int rows, int n_heads, int max_kv_len);
cudaError_t dspark_attention(const half* q, const half* k, const half* v, half* o, float* work, int rows,
                             int n_heads, int n_kv_heads, int kv_len, float scale, cudaStream_t stream);
// Graph-replayable: kv_len = *d_ctx + rows read on the device, <= max_kv_len (sizes the grid).
cudaError_t dspark_attention_dev(const half* q, const half* k, const half* v, half* o, float* work, int rows,
                                 int n_heads, int n_kv_heads, const int* d_ctx, int max_kv_len, float scale,
                                 cudaStream_t stream);

}  // namespace imp
