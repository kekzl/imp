#pragma once

// Qwen4Exp QSA indexer (modeling_qwen4_exp.py Qwen4ExpTextQSAIndexer): per query, the
// visible tokens are cut into blocks of `ratio` (4); each block key is the mean of the raw
// index keys -> k_layernorm (1+w) -> RoPE at the block's first position; the query heads are
// q_layernorm (1+w) -> RoPE; score = sum_h relu(q_h . blk) / sqrt(dim); the top
// `budget / ratio` (512) blocks plus the incomplete tail block stay visible. Below
// 512 complete blocks every block is selected and the selection equals dense attention.
//
// Kernels here take the position from the device (state.positions) so a captured decode
// step replays correctly; the host decides only shapes that are stable per graph.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

inline constexpr int kQsaDim = 128;  // indexer head dim (config indexer_head_dim)

struct QsaGeom {
    int n_heads;   // indexer query heads (4)
    int ratio;     // tokens per block (4)
    int budget;    // token budget (2048): block_topk = budget / ratio
    int rope_dim;  // rotated dims of each 128-dim head (64)
    float theta;
    float eps;
};

// qk: [rows, (n_heads + 1) * 128] FP16 from index_qk_proj. Writes q_out [rows, n_heads, 128]
// (norm (1+w_q) -> RoPE at positions[row]) and raw_keys[positions[row]] = the key part.
void qsa_prep_queries(const half* qk, const int* positions, const half* w_q, half* q_out,
                      half* raw_keys, int rows, const QsaGeom& g, cudaStream_t stream);

// Block keys for blocks [b0, b0 + nb): mean of `ratio` raw keys -> (1+w_k) norm -> RoPE at
// ratio * b. With `positions` non-null the block is the last complete one before
// positions[0] (decode, device-derived; nb = 1) and b0 is ignored.
void qsa_pool_blocks(const half* raw_keys, const half* w_k, half* block_keys, int b0, int nb,
                     const int* positions, const QsaGeom& g, cudaStream_t stream);

// Per query row r (position p = positions[r]): scores every complete block (< (p+1)/ratio),
// selects the top block_topk (ties: lowest block index), writes the visible token list
// sel_tokens[r][0..sel_count[r]) ascending: selected blocks' tokens, then the tail
// (ratio * n_complete .. p). Capacity per row = budget + ratio - 1.
// scores: scratch [rows, max_blocks] FP32.
void qsa_select(const half* q, const int* positions, const half* block_keys, float* scores,
                int max_blocks, int32_t* sel_tokens, int32_t* sel_count, int rows,
                const QsaGeom& g, cudaStream_t stream);

// Copies the selected tokens' K/V rows from the paged cache (layout
// [block][block_size][n_kv][hd], one sequence's block table `bt`) into a scratch paged
// cache where row r owns blocks [r * blocks_per_row, (r+1) * blocks_per_row); writes
// scratch_ctx[r] = sel_count[r]. scratch_bt is filled by qsa_init_scratch_bt.
void qsa_gather_kv(const half* k_cache, const half* v_cache, const int* bt, int block_size,
                   int n_kv, int hd, const int32_t* sel_tokens, const int32_t* sel_count,
                   int cap, half* k_scratch, half* v_scratch, int blocks_per_row,
                   int32_t* scratch_ctx, int rows, cudaStream_t stream);
void qsa_init_scratch_bt(int32_t* scratch_bt, int rows, int blocks_per_row, cudaStream_t stream);

}  // namespace imp
