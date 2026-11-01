#pragma once

// KV eviction proof harness (kv_cache.evict_keep_ratio, docs/plans/2026-11-01-kv-eviction.md).
// TriAttention key scoring (arXiv 2604.04921 Eqs. 6-13) on the paged cache plus the decode-time
// block-table compaction that hides the freed blocks (-1 holes) from every paged decode kernel.

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// Executor scratch (T2 arena, GraphExecutor::allocate_kv_evict_scratch_). enabled = every buffer
// is live; decode attention then always reads the compacted block table.
struct KvEvictScratch {
    bool enabled = false;
    int n_kv_layers = 0;
    int cap_blocks = 0;  // row stride capacity of compact_bt and bt
    int cap_rows = 0;
    float* qstats = nullptr;       // [n_kv_layers][kv_evict_qstats_floats]
    float* uw = nullptr;           // [n_kv_layers][n_heads][head_dim + head_dim/2]
    double* head_stats = nullptr;  // [n_heads][2]
    unsigned* tok_max = nullptr;   // [cap_blocks * block_size]
    float* tok_acc = nullptr;      // [cap_blocks * block_size]
    int* bt = nullptr;             // [cap_blocks]
    int* compact_bt = nullptr;     // [cap_rows][cap_blocks]
    int* compact_ctx = nullptr;    // [cap_rows]
};

// Per-layer prompt query statistics: [n_heads][pairs][3] floats (sum of pre-RoPE pair re,
// im, |pair|) followed by one count float, padded to 4.
inline int64_t kv_evict_qstats_floats(int n_heads, int head_dim) {
    return (static_cast<int64_t>(n_heads) * (head_dim / 2) * 3) + 4;
}

// Accumulates the pre-RoPE statistics of n post-RoPE query rows q [n, n_heads*head_dim] F16 by
// undoing linear RoPE (theta, inv_scaling, rope_dim = head_dim). False when the shape does not
// fit the kernel's shared-memory accumulator (n_heads * head_dim/2 > 4096).
[[nodiscard]] bool kv_evict_qstats_accumulate(const half* q, const int* positions, int n, int n_heads,
                                              int head_dim, bool neox, float theta, float inv_scaling,
                                              float* stats_layer, cudaStream_t stream);

// Cache view of one KV layer's keys. Strides are bytes between consecutive physical blocks.
struct KvEvictKeyView {
    QType dtype = QType::F16;  // F16, FP8_E4M3 or NVFP4
    const void* k_base = nullptr;
    int64_t block_stride = 0;
    const uint8_t* scale_base = nullptr;  // NVFP4 UE4M3 group scales
    int64_t scale_block_stride = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int block_size = 16;
};

// Two passes over tokens [tok_begin, tok_end) of one sequence (logical block table bt, device).
// u: [n_heads, head_dim] effective query (mean over future offsets of the rotated query centre),
// w: [n_heads, head_dim/2] norm-term weight. Pass 0 adds per-head sum / sum of squares of the
// score into head_stats [n_heads][2] (double). Pass 1 z-normalizes with them, takes the max over
// all heads and writes it into tok_max (ordered-float encoding, 0 = unset).
void kv_evict_score_pass(int pass, const KvEvictKeyView& kv, const float* u, const float* w, const int* bt,
                         int tok_begin, int tok_end, int n_heads, double* head_stats, unsigned* tok_max,
                         cudaStream_t stream);

// tok_acc[t] += decoded tok_max[t] (skipped when unset), tok_max[t] = 0, t in [tok_begin, tok_end).
void kv_evict_accumulate_layer(unsigned* tok_max, float* tok_acc, int tok_begin, int tok_end,
                               cudaStream_t stream);

// Per row: copies the live entries (>= 0) of bt[row] among its first ceil(ctx/bs) blocks to
// out_bt[row] in order, out_ctx[row] = ctx - holes * bs. Identity on rows without holes.
// Holes are always full blocks (eviction never touches the recent window).
void kv_evict_compact_tables(const int* bt, const int* ctx_lens, int n_rows, int stride, int block_size,
                             int* out_bt, int* out_ctx, cudaStream_t stream);

}  // namespace imp
