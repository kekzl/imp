#pragma once

// Sparse decode attention (attention.sparse_topk_tokens): Quest-class top-k
// page selection. Per-block key min/max metadata bounds every query dot
// product; the top-scoring blocks are compacted into a per-step block table
// the unmodified paged decode kernels consume. All device-side, graph-safe.
// Design: docs/plans/2026-08-28-sparse-decode-attention.md.

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Update per-block key min/max metadata after a paged KV write. Reads the
// just-written K rows back from the cache (dtype-exact, post-RoPE). FP8
// metadata stores the raw scale-1 dequant: the per-layer KV scale is one
// positive factor per score and cannot change the block ranking.
// Supported cache dtypes: F16, FP8_E4M3 (callers gate at init).
// Parameter contract mirrors the write_kv_cache_* kernels (same
// positions/block_tables/max_blocks_per_seq/n_sequences semantics).
void sparse_update_key_minmax(QType cache_dtype, const void* k_cache_base, void* minmax_base,
                              const int* positions, const int* block_tables, int n_kv_heads, int head_dim,
                              int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences,
                              cudaStream_t stream);

// Score every context block of every sequence against the current queries and
// build a compacted block table (ascending block order, at most budget_blocks
// entries per sequence; sink/recent blocks always kept) plus the matching
// context lengths. n_blocks <= budget_blocks degenerates to an identity copy
// of the table row - attention over it is bit-identical to dense.
// Requires budget_blocks > sink_blocks + recent_blocks (host-clamped).
// q: [n_seq, n_heads, head_dim] half (post-RoPE decode queries).
// scores_scratch: [n_seq, max_blocks_per_seq] float.
// sparse_block_tables: [n_seq, budget_blocks] int. sparse_context_lens: [n_seq].
void sparse_select_blocks(const half* q, const void* minmax_base, const int* block_tables,
                          const int* context_lens, int n_seq, int n_heads, int n_kv_heads, int head_dim,
                          int block_size, int max_blocks_per_seq, int budget_blocks, int sink_blocks,
                          int recent_blocks, float* scores_scratch, int* sparse_block_tables,
                          int* sparse_context_lens, cudaStream_t stream);

}  // namespace imp
