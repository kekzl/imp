#pragma once

// Sparse decode attention (attention.sparse_topk_tokens): Quest-class top-k page
// selection. Per-block key mean/std metadata scores every query dot product; top-scoring blocks
// compact into a per-step block table the unmodified paged decode kernels consume. All
// device-side, graph-safe.

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Batched form: ONE launch updates every KV layer (grid.y) at the end of a forward
// (decode steps, spec verify chunks, prefill chunks); the inline per-layer form cost
// multi-stream serving prefill measurably more. Selection only runs at decode time, so
// prefill metadata is complete before the first decode step; decode lag is covered by
// the forced recent blocks.
// seq_offsets: the ragged prefill's [n_seq+1] device offsets (nullptr otherwise), the
// token->table-row mapping ragged forwards need. Layer strides are in BYTES between
// consecutive layers' block-0 pointers (uniform scalar geometry only). k_scale_base/
// sc_layer_stride_bytes carry the NVFP4 UE4M3 group scales (nullptr/0 for F16 and FP8):
// packed nibbles alone don't define a key value, so the metadata needs both regions.
// Arguments in the old parameter order (audit D-11); all required except seq_offsets and the scale pair.
struct SparseKeyMinmaxArgs {
    QType cache_dtype = QType::F16;
    const void* k_base = nullptr;
    int64_t k_layer_stride_bytes = 0;
    const void* k_scale_base = nullptr;
    int64_t sc_layer_stride_bytes = 0;
    void* minmax_base = nullptr;
    int64_t mm_layer_stride_bytes = 0;
    const int* positions = nullptr;
    const int* block_tables = nullptr;
    const int* seq_offsets = nullptr;
    int n_layers = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int block_size = 0;
    int n_tokens = 0;
    int max_blocks_per_seq = 0;
    int n_sequences = 0;
    cudaStream_t stream = nullptr;
};
void sparse_update_key_minmax_all_layers(const SparseKeyMinmaxArgs& a);

// Scores every context block of every sequence against current queries and builds a
// compacted block table (ascending order, at most budget_blocks entries per sequence;
// sink/recent blocks always kept) plus matching context lengths. n_blocks <=
// engage_blocks degenerates to an identity copy (bit-identical to dense attention).
// Requires budget_blocks > sink_blocks + recent_blocks (host-clamped) and table row
// capacity table_blocks >= engage_blocks.
// q: [n_seq, n_heads, head_dim] half (post-RoPE decode queries).
// scores_scratch: [n_seq, max_blocks_per_seq] float.
// sparse_block_tables: [n_seq, table_blocks] int. sparse_context_lens: [n_seq].
// Arguments in the old parameter order (audit D-11); all required.
struct SparseSelectBlocksArgs {
    const half* q = nullptr;
    const void* minmax_base = nullptr;
    const int* block_tables = nullptr;
    const int* context_lens = nullptr;
    int n_seq = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int block_size = 0;
    int max_blocks_per_seq = 0;
    int budget_blocks = 0;
    int sink_blocks = 0;
    int recent_blocks = 0;
    int engage_blocks = 0;
    int table_blocks = 0;
    float* scores_scratch = nullptr;
    int* sparse_block_tables = nullptr;
    int* sparse_context_lens = nullptr;
    float std_coef = 0.0f;
    cudaStream_t stream = nullptr;
};
void sparse_select_blocks(const SparseSelectBlocksArgs& a);
// True when both launch sites score this shape with the kv-head-split kernel (32 / n_kv_heads in {4, 8, 16}
// lanes per kv head, head_dim a multiple of 4 x lanes); false = the per-4-dims kernel.
[[nodiscard]] bool sparse_score_uses_split(int n_kv_heads, int head_dim);

// Sparse prefill (attention.sparse_prefill_topk_tokens): past tokens a continuation chunk attends
// to under a budget_blocks page budget; the partial tail block always stays. past_len when
// ceil(past_len/block_size) <= budget_blocks (identity).
int sparse_prefill_past_tokens(int past_len, int block_size, int budget_blocks);

// One page selection for a whole prefill chunk: scores past blocks against sample_rows rows of
// q [n_rows, n_heads, head_dim] (evenly spaced, last row included), merges the rows by gap to
// each row's best block, writes the compacted ascending past table (sink/recent forced). Within
// the budget it copies the table (dense equivalent). Returns the past token count to attend, or
// -1 when the table exceeds capacity_blocks. Scratch: scores [64, capacity], agg [capacity],
// out_block_table [capacity], out_ctx_scratch [2] (device).
// Arguments in the old parameter order (audit D-11); all required.
struct SparsePrefillSelectArgs {
    const half* q = nullptr;
    int n_rows = 0;
    int sample_rows = 0;
    const void* minmax_layer = nullptr;
    const int* past_block_table = nullptr;
    int past_len = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int block_size = 0;
    int capacity_blocks = 0;
    int budget_blocks = 0;
    int sink_blocks = 0;
    int recent_blocks = 0;
    float std_coef = 0.0f;
    float* scores_scratch = nullptr;
    float* agg_scratch = nullptr;
    int* out_block_table = nullptr;
    int* out_ctx_scratch = nullptr;
    cudaStream_t stream = nullptr;
};
int sparse_prefill_select_past(const SparsePrefillSelectArgs& a);

}  // namespace imp
