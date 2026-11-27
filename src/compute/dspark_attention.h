#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace imp {

// Non-causal attention of `rows` query rows (<= kDSparkMaxRows) over kv_len keys, head_dim 128, GQA
// (n_heads a multiple of n_kv_heads, group <= 8 at rows <= 8, group <= 4 at 9..16 rows): Q [rows, n_heads *
// 128], K/V [kv_len, n_kv_heads * 128], O [rows, n_heads * 128], all FP16 row-major. Split-K over keys with
// an online softmax, then one combine launch; no host sync. `work` holds dspark_attention_work_floats(...)
// floats.
constexpr int kDSparkMaxRows = 16;
size_t dspark_attention_work_floats(int rows, int n_heads, int max_kv_len);
cudaError_t dspark_attention(const half* q, const half* k, const half* v, half* o, float* work, int rows,
                             int n_heads, int n_kv_heads, int kv_len, float scale, cudaStream_t stream);
// Graph-replayable: kv_len = *d_ctx + rows read on the device, <= max_kv_len (sizes the grid).
// d_first (optional): the launch's key 0 is row *d_first of K / V (a window that moves with the position).
// blocks (optional, device): key k at row blocks[k / 16] * 16 + k % 16 of K / V (a paged KV layer).
// Arguments in the old parameter order (audit D-11); d_first / blocks keep the old nullptr defaults.
struct DsparkAttentionDevArgs {
    const half* q = nullptr;
    const half* k = nullptr;
    const half* v = nullptr;
    half* o = nullptr;
    float* work = nullptr;
    int rows = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    const int* d_ctx = nullptr;
    int max_kv_len = 0;
    float scale = 0.0f;
    cudaStream_t stream = nullptr;
    const int* d_first = nullptr;
    const int* blocks = nullptr;
};
cudaError_t dspark_attention_dev(const DsparkAttentionDevArgs& a);

}  // namespace imp
