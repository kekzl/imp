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

// Batched form: ONE launch updates every KV layer (grid.y) at the end of a
// forward - decode steps, spec verify chunks AND prefill chunks (the inline
// per-layer form cost the multi-stream serving prefill ~12%, 2026-08-29).
// Selection only runs at decode time, so prefill metadata is complete before
// the first decode step; decode lag is covered by the forced recent blocks.
// seq_offsets: the ragged prefill's [n_seq+1] device offsets (nullptr
// otherwise) - it is the token->table-row mapping ragged forwards need.
// Layer strides in BYTES between consecutive layers' block-0 pointers
// (uniform scalar geometry only - the init gate guarantees it).
// k_scale_base/sc_layer_stride_bytes carry the NVFP4 UE4M3 group scales
// (nullptr/0 for F16 and FP8) - the packed nibbles alone do not define a key
// value, so the metadata bound needs both regions.
void sparse_update_key_minmax_all_layers(QType cache_dtype, const void* k_base, int64_t k_layer_stride_bytes,
                                         const void* k_scale_base, int64_t sc_layer_stride_bytes,
                                         void* minmax_base, int64_t mm_layer_stride_bytes,
                                         const int* positions, const int* block_tables,
                                         const int* seq_offsets, int n_layers, int n_kv_heads, int head_dim,
                                         int block_size, int n_tokens, int max_blocks_per_seq,
                                         int n_sequences, cudaStream_t stream);

// Score every context block of every sequence against the current queries and
// build a compacted block table (ascending block order, at most budget_blocks
// entries per sequence; sink/recent blocks always kept) plus the matching
// context lengths. n_blocks <= engage_blocks degenerates to an identity copy
// of the table row - attention over it is bit-identical to dense
// (engage_blocks >= budget_blocks encodes attention.sparse_min_ctx).
// Requires budget_blocks > sink_blocks + recent_blocks (host-clamped) and
// table row capacity table_blocks >= engage_blocks.
// q: [n_seq, n_heads, head_dim] half (post-RoPE decode queries).
// scores_scratch: [n_seq, max_blocks_per_seq] float.
// sparse_block_tables: [n_seq, table_blocks] int. sparse_context_lens: [n_seq].
void sparse_select_blocks(const half* q, const void* minmax_base, const int* block_tables,
                          const int* context_lens, int n_seq, int n_heads, int n_kv_heads, int head_dim,
                          int block_size, int max_blocks_per_seq, int budget_blocks, int sink_blocks,
                          int recent_blocks, int engage_blocks, int table_blocks, float* scores_scratch,
                          int* sparse_block_tables, int* sparse_context_lens, cudaStream_t stream);

}  // namespace imp
