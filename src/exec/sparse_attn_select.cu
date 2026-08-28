// Sparse decode attention: Quest-class top-k page selection.
// Three kernels: per-block key min/max maintenance, block scoring against the
// current queries, and top-k selection into a compacted block table the
// unmodified paged decode kernels consume. Everything runs device-side from
// device inputs (block tables, context lens), so the whole path is
// CUDA-graph-safe while the context grows during replay.
// Design + gates: docs/plans/2026-08-28-sparse-decode-attention.md.

#include "exec/sparse_attn_select.h"
#include "exec/executor_kernels_internal.cuh"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include <cuda_fp8.h>
#include <float.h>

namespace imp {

namespace {

__device__ __forceinline__ float cache_val_to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float cache_val_to_float(__nv_fp8_e4m3 v) { return static_cast<float>(v); }

// ---------------------------------------------------------------------------
// Metadata maintenance. Owner-CTA scheme, race-free without atomics: CTA i is
// active iff token i's physical block differs from token i-1's; the owner
// covers every same-block token after it in this launch (same-block tokens
// are adjacent in all call shapes: prefill contiguous positions, ragged
// row-range per sequence, spec verify chunk contiguous, multi-seq decode one
// token per sequence with exclusive blocks). slot 0 initializes, otherwise
// merge with the stored metadata - block reuse is covered because a fresh
// block's first write is always slot 0.
// Metadata layout per (layer, block): row_elems half2, (min, max) per
// (kv_head, dim) element, element index e = kv_head * head_dim + d.
// ---------------------------------------------------------------------------
template <typename CacheT>
__global__ void sparse_update_key_minmax_kernel(const CacheT* __restrict__ k_cache_base,
                                                __half2* __restrict__ minmax_base,
                                                const int* __restrict__ positions,
                                                const int* __restrict__ block_tables, int row_elems,
                                                int block_size, int n_tokens, int max_blocks_per_seq,
                                                int n_sequences) {
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;
    const int pos = positions[token_idx];
    int slot;
    const int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq,
                                         n_sequences, slot);
    if (block_id < 0)
        return;
    if (token_idx > 0) {
        int prev_slot;
        const int prev_block = kv_resolve_slot(block_tables, positions[token_idx - 1], block_size,
                                               token_idx - 1, max_blocks_per_seq, n_sequences, prev_slot);
        if (prev_block == block_id)
            return;  // not the owner of this block's span
    }
    // Forward span of same-block tokens in this launch (at most block_size).
    int span = 1;
    while (token_idx + span < n_tokens && span < block_size) {
        int s2;
        const int b2 = kv_resolve_slot(block_tables, positions[token_idx + span], block_size,
                                       token_idx + span, max_blocks_per_seq, n_sequences, s2);
        if (b2 != block_id)
            break;
        span++;
    }
    const CacheT* blk = k_cache_base + (int64_t)block_id * block_size * row_elems;
    __half2* mm = minmax_base + (int64_t)block_id * row_elems;
    const bool init = (slot == 0);
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        float mn, mx;
        if (init) {
            mn = FLT_MAX;
            mx = -FLT_MAX;
        } else {
            const __half2 cur = mm[e];
            mn = __low2float(cur);
            mx = __high2float(cur);
        }
        for (int j = 0; j < span; j++) {
            const float v = cache_val_to_float(blk[(int64_t)(slot + j) * row_elems + e]);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        mm[e] = __floats2half2_rn(mn, mx);
    }
}

// ---------------------------------------------------------------------------
// Block scoring. One warp per block, grid-stride over blocks (grid shape is
// context-independent - capture-safe while ctx grows during replay).
// score(b) = max over q heads h of sum_d max(q_h[d]*min[d], q_h[d]*max[d])
// over h's kv head metadata - an upper bound on any softmax logit the block
// can produce for the current query (Quest).
// ---------------------------------------------------------------------------
constexpr int kScoreThreads = 256;
constexpr int kScoreWarps = kScoreThreads / kWarpSize;
constexpr int kMaxGroup = 16;  // n_heads / n_kv_heads ceiling (host-gated)

__global__ void sparse_score_blocks_kernel(const half* __restrict__ q,
                                           const __half2* __restrict__ minmax_base,
                                           const int* __restrict__ block_tables,
                                           const int* __restrict__ context_lens,
                                           float* __restrict__ scores, int n_heads, int n_kv_heads,
                                           int head_dim, int block_size, int max_blocks_per_seq,
                                           int scores_stride) {
    const int seq = blockIdx.y;
    const int ctx_len = context_lens[seq];
    const int n_blocks = (ctx_len + block_size - 1) / block_size;
    const int q_elems = n_heads * head_dim;
    extern __shared__ half q_smem[];
    const half* q_seq = q + (int64_t)seq * q_elems;
    for (int i = threadIdx.x; i < q_elems; i += blockDim.x)
        q_smem[i] = q_seq[i];
    __syncthreads();

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int g = n_heads / n_kv_heads;
    const int row_elems = n_kv_heads * head_dim;
    const int* bt = block_tables + (int64_t)seq * max_blocks_per_seq;
    float* sc = scores + (int64_t)seq * scores_stride;

    for (int b = blockIdx.x * kScoreWarps + warp; b < n_blocks; b += gridDim.x * kScoreWarps) {
        const int block_id = bt[b];
        if (block_id < 0) {
            if (lane == 0)
                sc[b] = -FLT_MAX;
            continue;
        }
        const __half2* mm = minmax_base + (int64_t)block_id * row_elems;
        float best = -FLT_MAX;
        for (int kvh = 0; kvh < n_kv_heads; kvh++) {
            float part[kMaxGroup];
#pragma unroll
            for (int h = 0; h < kMaxGroup; h++)
                part[h] = 0.0f;
            for (int d = lane; d < head_dim; d += kWarpSize) {
                const __half2 v = mm[kvh * head_dim + d];
                const float mn = __low2float(v);
                const float mx = __high2float(v);
                // Full unroll with a guard keeps part[] register-resident (a
                // runtime trip count would spill it to a local frame).
#pragma unroll
                for (int h = 0; h < kMaxGroup; h++) {
                    if (h >= g)
                        break;
                    const float qv = __half2float(q_smem[(kvh * g + h) * head_dim + d]);
                    part[h] += fmaxf(qv * mn, qv * mx);
                }
            }
#pragma unroll
            for (int h = 0; h < kMaxGroup; h++) {
                if (h >= g)
                    break;
                const float s = warp_reduce_sum(part[h]);
                best = fmaxf(best, s);
            }
        }
        if (lane == 0)
            sc[b] = best;
    }
}

// ---------------------------------------------------------------------------
// Top-k selection + compacted table build. One CTA per sequence.
//
// The selection key is 64-bit: (monotone score bits << 32) | ~block_index -
// unique per block, so the radix threshold is exact, ties resolve to the
// LOWER block index, and the result is deterministic. 8 x 8-bit MSB radix
// passes narrow the k-th largest key among the middle blocks; sink and
// recent blocks are forced. A bitmap + ballot compaction emits the selected
// physical block ids in ascending block order, which preserves the dense
// kernel's softmax accumulation order.
// ---------------------------------------------------------------------------
constexpr int kSelectThreads = 256;

__device__ __forceinline__ uint32_t score_key(float s) {
    uint32_t u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void sparse_select_topk_kernel(const float* __restrict__ scores,
                                          const int* __restrict__ block_tables,
                                          const int* __restrict__ context_lens,
                                          int* __restrict__ sparse_bt, int* __restrict__ sparse_ctx,
                                          int block_size, int max_blocks_per_seq, int scores_stride,
                                          int budget_blocks, int sink_blocks, int recent_blocks) {
    const int seq = blockIdx.x;
    const int ctx_len = context_lens[seq];
    const int n_blocks = (ctx_len + block_size - 1) / block_size;
    const int* bt = block_tables + (int64_t)seq * max_blocks_per_seq;
    int* out = sparse_bt + (int64_t)seq * budget_blocks;

    if (n_blocks <= budget_blocks) {
        // Identity: attention over this table is bit-identical to dense.
        for (int i = threadIdx.x; i < n_blocks; i += blockDim.x)
            out[i] = bt[i];
        if (threadIdx.x == 0)
            sparse_ctx[seq] = ctx_len;
        return;
    }

    const int mid_lo = sink_blocks;
    const int mid_hi = n_blocks - recent_blocks;  // > mid_lo (host-clamped budget)
    const int k = budget_blocks - sink_blocks - recent_blocks;
    const float* sc = scores + (int64_t)seq * scores_stride;

    // Dynamic smem: hist[256] | bitmap words | word ranks
    extern __shared__ uint32_t sel_smem[];
    uint32_t* hist = sel_smem;                       // 256
    const int n_words = (max_blocks_per_seq + 31) / 32;
    uint32_t* bitmap = hist + 256;                   // n_words
    uint32_t* word_rank = bitmap + n_words;          // n_words
    __shared__ uint64_t s_prefix;
    __shared__ int s_k_rem;

    if (threadIdx.x == 0) {
        s_prefix = 0;
        s_k_rem = k;
    }

    // 8 MSB-first radix passes over the middle blocks' 64-bit keys.
    for (int level = 7; level >= 0; level--) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x)
            hist[i] = 0;
        __syncthreads();
        const uint64_t prefix = s_prefix;
        const int shift = level * 8;
        for (int b = mid_lo + threadIdx.x; b < mid_hi; b += blockDim.x) {
            const uint64_t key = ((uint64_t)score_key(sc[b]) << 32) | (uint32_t)~b;
            if (level == 7 || (key >> (shift + 8)) == prefix)
                atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int rem = s_k_rem;
            int bin = 255;
            for (; bin >= 0; bin--) {
                const int c = (int)hist[bin];
                if (rem <= c)
                    break;
                rem -= c;
            }
            // bin >= 0 always: k middle keys exist below the threshold search.
            s_prefix = (s_prefix << 8) | (uint32_t)bin;
            s_k_rem = rem;
        }
        __syncthreads();
    }
    // Keys are unique, so after 8 passes s_prefix IS the k-th largest key:
    // select every middle key >= it, exactly k of them.
    const uint64_t threshold = s_prefix;

    for (int i = threadIdx.x; i < n_words; i += blockDim.x)
        bitmap[i] = 0;
    __syncthreads();
    // Forced sinks + recents.
    for (int b = threadIdx.x; b < mid_lo; b += blockDim.x)
        atomicOr(&bitmap[b >> 5], 1u << (b & 31));
    for (int b = mid_hi + threadIdx.x; b < n_blocks; b += blockDim.x)
        atomicOr(&bitmap[b >> 5], 1u << (b & 31));
    // Selected middles.
    for (int b = mid_lo + threadIdx.x; b < mid_hi; b += blockDim.x) {
        const uint64_t key = ((uint64_t)score_key(sc[b]) << 32) | (uint32_t)~b;
        if (key >= threshold)
            atomicOr(&bitmap[b >> 5], 1u << (b & 31));
    }
    __syncthreads();
    // Exclusive scan of per-word popcounts (word count <= 256: one serial
    // pass by thread 0 is a few hundred iterations).
    if (threadIdx.x == 0) {
        uint32_t run = 0;
        for (int w = 0; w < n_words; w++) {
            word_rank[w] = run;
            run += __popc(bitmap[w]);
        }
    }
    __syncthreads();
    // Emit ascending: physical ids gathered from the dense table.
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        uint32_t bits = bitmap[w];
        uint32_t rank = word_rank[w];
        while (bits) {
            const int bit = __ffs(bits) - 1;
            out[rank++] = bt[w * 32 + bit];
            bits &= bits - 1;
        }
    }
    if (threadIdx.x == 0) {
        const int tail_fill = ctx_len - (n_blocks - 1) * block_size;
        sparse_ctx[seq] = (budget_blocks - 1) * block_size + tail_fill;
    }
}

}  // namespace

void sparse_update_key_minmax(QType cache_dtype, const void* k_cache_base, void* minmax_base,
                              const int* positions, const int* block_tables, int n_kv_heads, int head_dim,
                              int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences,
                              cudaStream_t stream) {
    if (n_tokens <= 0)
        return;
    const int row_elems = n_kv_heads * head_dim;
    const int threads = 128;
    if (cache_dtype == QType::FP8_E4M3) {
        sparse_update_key_minmax_kernel<__nv_fp8_e4m3><<<n_tokens, threads, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3*>(k_cache_base), static_cast<__half2*>(minmax_base), positions,
            block_tables, row_elems, block_size, n_tokens, max_blocks_per_seq, n_sequences);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        sparse_update_key_minmax_kernel<half><<<n_tokens, threads, 0, stream>>>(
            static_cast<const half*>(k_cache_base), static_cast<__half2*>(minmax_base), positions,
            block_tables, row_elems, block_size, n_tokens, max_blocks_per_seq, n_sequences);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

void sparse_select_blocks(const half* q, const void* minmax_base, const int* block_tables,
                          const int* context_lens, int n_seq, int n_heads, int n_kv_heads, int head_dim,
                          int block_size, int max_blocks_per_seq, int budget_blocks, int sink_blocks,
                          int recent_blocks, float* scores_scratch, int* sparse_block_tables,
                          int* sparse_context_lens, cudaStream_t stream) {
    // Fixed grid.x: work distribution adapts device-side via grid-stride, so a
    // captured graph stays correct while the context grows during replay.
    dim3 score_grid(32, n_seq);
    const size_t q_smem = (size_t)n_heads * head_dim * sizeof(half);
    sparse_score_blocks_kernel<<<score_grid, kScoreThreads, q_smem, stream>>>(
        q, static_cast<const __half2*>(minmax_base), block_tables, context_lens, scores_scratch, n_heads,
        n_kv_heads, head_dim, block_size, max_blocks_per_seq, max_blocks_per_seq);
    IMP_CUDA_CHECK_LAUNCH();

    const int n_words = (max_blocks_per_seq + 31) / 32;
    const size_t sel_smem = (256 + 2 * (size_t)n_words) * sizeof(uint32_t);
    sparse_select_topk_kernel<<<n_seq, kSelectThreads, sel_smem, stream>>>(
        scores_scratch, block_tables, context_lens, sparse_block_tables, sparse_context_lens, block_size,
        max_blocks_per_seq, max_blocks_per_seq, budget_blocks, sink_blocks, recent_blocks);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
