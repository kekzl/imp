// Sparse decode attention: Quest-class top-k page selection. Three kernels: per-block
// key mean/std maintenance, block scoring against the current queries, top-k selection
// into a compacted block table the unmodified paged decode kernels consume. All
// device-side from device inputs, so the whole path is CUDA-graph-safe while context
// grows during replay. Design: docs/plans/2026-08-28-sparse-decode-attention.md.

#include "exec/sparse_attn_select.h"
#include "exec/executor_kernels_internal.cuh"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include "quant/turboquant_fp4.cuh"
#include <cuda_fp8.h>
#include <float.h>
#include <algorithm>

namespace imp {

namespace {

__device__ __forceinline__ float cache_val_to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float cache_val_to_float(__nv_fp8_e4m3 v) { return static_cast<float>(v); }

// Key element access per KV dtype. The owner-CTA/span logic below is identical
// for every dtype; only "read key element e of slot s in this block" differs,
// so it lives here and the race-relevant part stays single-sourced.
template <typename CacheT>
struct KeyReaderPlain {
    const CacheT* __restrict__ blk;
    int row_elems;
    __device__ __forceinline__ float get(int s, int e) const {
        return cache_val_to_float(blk[((int64_t)s * row_elems) + e]);
    }
};

// NVFP4 key reader: two elements per byte (low nibble = even), UE4M3 group scale per 16
// contiguous elements in a parallel array. head_dim is a multiple of 16 on this path and
// heads are contiguous within a row, so the flat group index over the row is exactly
// e/16, no head_dim needed.
struct KeyReaderNvfp4 {
    const uint8_t* __restrict__ blk;
    const uint8_t* __restrict__ sc;
    int row_bytes;  // row_elems / 2
    int sc_row;     // row_elems / 16
    __device__ __forceinline__ float get(int s, int e) const {
        const uint8_t byte = blk[((int64_t)s * row_bytes) + (e >> 1)];
        const uint32_t code = (e & 1) ? (byte >> 4) : (byte & 0xF);
        const float mag = kTQFP4DequantLUT[code & 0x7];
        const float v = (code & 0x8) ? -mag : mag;
        __nv_fp8_e4m3 sbits;
        const uint8_t raw = sc[((int64_t)s * sc_row) + (e >> 4)];
        memcpy(&sbits, &raw, 1);
        return v * static_cast<float>(sbits);
    }
};

// Metadata maintenance: owner-CTA scheme, race-free without atomics. CTA i is active iff
// token i's physical block differs from token i-1's; the owner covers every same-block
// token after it (same-block tokens are adjacent in every call shape: prefill
// contiguous, ragged row-range per sequence, spec verify chunk contiguous, multi-seq
// decode one token per sequence with exclusive blocks). The owner rebuilds the pair from
// slots [0, slot + span): a rolled-back draft row leaves no trace, and neither does a
// reused block's old content.
// Metadata layout per (layer, block): row_elems half2 per (kv_head, dim) element, index
// e = kv_head * head_dim + d. The pair carries (mean, std) over the block's written rows.
// Batched decode variant: one launch covers every KV layer (grid.y), legal because
// decode selection force-includes recent blocks, so metadata may lag by one step without
// affecting the bound.
// Ragged token->block-table-row mapping: with seq_offsets ([n_seq+1], the device twin of
// h_seq_offsets), token i belongs to the seq whose offset range contains i, and that is
// the block-table row; nullptr falls back to plain kv_resolve_slot semantics.
__device__ __forceinline__ int sparse_resolve_block(const int* block_tables, const int* positions,
                                                    const int* seq_offsets, int token_idx, int block_size,
                                                    int max_blocks_per_seq, int n_sequences, int& slot) {
    if (seq_offsets != nullptr && max_blocks_per_seq > 0) {
        int seq = 0;
        while (seq + 1 < n_sequences && token_idx >= seq_offsets[seq + 1])
            seq++;
        const int pos = positions[token_idx];
        slot = pos % block_size;
        return block_tables[((int64_t)seq * max_blocks_per_seq) + (pos / block_size)];
    }
    return kv_resolve_slot(block_tables, positions[token_idx], block_size, token_idx, max_blocks_per_seq,
                           n_sequences, slot);
}

// Owner resolution: returns the physical block this CTA owns, or -1 when
// another CTA owns it (or there is nothing to do). Sets slot and span.
__device__ __forceinline__ int sparse_owner_block(const int* block_tables, const int* positions,
                                                  const int* seq_offsets, int token_idx, int block_size,
                                                  int n_tokens, int max_blocks_per_seq, int n_sequences,
                                                  int& slot, int& span) {
    const int block_id = sparse_resolve_block(block_tables, positions, seq_offsets, token_idx, block_size,
                                              max_blocks_per_seq, n_sequences, slot);
    if (block_id < 0)
        return -1;
    if (token_idx > 0) {
        int prev_slot;
        const int prev_block = sparse_resolve_block(block_tables, positions, seq_offsets, token_idx - 1,
                                                    block_size, max_blocks_per_seq, n_sequences, prev_slot);
        if (prev_block == block_id)
            return -1;
    }
    span = 1;
    while (token_idx + span < n_tokens && span < block_size) {
        int s2;
        const int b2 = sparse_resolve_block(block_tables, positions, seq_offsets, token_idx + span,
                                            block_size, max_blocks_per_seq, n_sequences, s2);
        if (b2 != block_id || s2 != slot + span)
            break;  // consecutive slots only - the read below walks slot+j
        span++;
    }
    return block_id;
}

// The pair holds (mean, std) over the block's rows. One Welford pass per element rather
// than sum/sumsq: E[x^2]-E[x]^2 cancels most of the mantissa once |mean| is a few
// multiples of the spread.
template <typename Reader>
__device__ __forceinline__ void sparse_rebuild_block(Reader rd, __half2* __restrict__ mm, int rows,
                                                     int row_elems) {
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        float mean = 0.0f, m2 = 0.0f;
        for (int j = 0; j < rows; j++) {
            const float v = rd.get(j, e);
            const float d = v - mean;
            mean += d / (float)(j + 1);
            m2 += d * (v - mean);
        }
        mm[e] = __floats2half2_rn(mean, sqrtf(fmaxf(m2 / (float)rows, 0.0f)));
    }
}

template <typename CacheT>
__global__ void sparse_update_key_minmax_layers_kernel(
    const CacheT* __restrict__ k_base, int64_t k_layer_stride,  // elems
    __half2* __restrict__ mm_base, int64_t mm_layer_stride,     // half2 elems
    const int* __restrict__ positions, const int* __restrict__ block_tables,
    const int* __restrict__ seq_offsets, int row_elems, int block_size, int n_tokens, int max_blocks_per_seq,
    int n_sequences) {
    const int token_idx = blockIdx.x;
    const int layer = blockIdx.y;
    if (token_idx >= n_tokens)
        return;
    int slot, span;
    const int block_id = sparse_owner_block(block_tables, positions, seq_offsets, token_idx, block_size,
                                            n_tokens, max_blocks_per_seq, n_sequences, slot, span);
    if (block_id < 0)
        return;
    KeyReaderPlain<CacheT> rd{k_base + layer * k_layer_stride + (int64_t)block_id * block_size * row_elems,
                              row_elems};
    sparse_rebuild_block(rd, mm_base + (layer * mm_layer_stride) + ((int64_t)block_id * row_elems),
                         slot + span, row_elems);
}

// NVFP4 twin: same owner scheme, two base pointers (packed nibbles + UE4M3
// group scales). Strides are in bytes because both regions are byte arrays.
__global__ void sparse_update_key_minmax_layers_nvfp4_kernel(
    const uint8_t* __restrict__ k_base, int64_t k_layer_stride_bytes,
    const uint8_t* __restrict__ sc_base, int64_t sc_layer_stride_bytes,
    __half2* __restrict__ mm_base, int64_t mm_layer_stride, const int* __restrict__ positions,
    const int* __restrict__ block_tables, const int* __restrict__ seq_offsets, int row_elems,
    int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    const int token_idx = blockIdx.x;
    const int layer = blockIdx.y;
    if (token_idx >= n_tokens)
        return;
    int slot, span;
    const int block_id = sparse_owner_block(block_tables, positions, seq_offsets, token_idx, block_size,
                                            n_tokens, max_blocks_per_seq, n_sequences, slot, span);
    if (block_id < 0)
        return;
    const int row_bytes = row_elems / 2;
    const int sc_row = row_elems / 16;
    KeyReaderNvfp4 rd{k_base + (layer * k_layer_stride_bytes) + ((int64_t)block_id * block_size * row_bytes),
                      sc_base + (layer * sc_layer_stride_bytes) + ((int64_t)block_id * block_size * sc_row),
                      row_bytes, sc_row};
    sparse_rebuild_block(rd, mm_base + (layer * mm_layer_stride) + ((int64_t)block_id * row_elems),
                         slot + span, row_elems);
}

// Block scoring: one warp per block, grid-stride over blocks (context-independent grid
// shape, capture-safe while ctx grows during replay).
//   score(b) = max over q heads h of sum_d (q_h[d]*mean[d] + |q_h[d]|*coef*std[d])
// Mean/std (arXiv 2605.27740): the page's own centre and spread, not a bound but an
// estimate. It replaced the Quest min/max corner bound, which one extreme token per dimension
// drives (NIAH 2/10 vs 10/10 at a 4096 budget, Qwen3.8-27B, 2026-09-12).
struct ScoreMeanStd {
    __device__ __forceinline__ static float term(float q, float mean, float sd, float coef) {
        return (q * mean) + (fabsf(q) * coef * sd);
    }
};

constexpr int kScoreThreads = 256;
constexpr int kScoreWarps = kScoreThreads / kWarpSize;
constexpr int kMaxGroup = 16;  // n_heads / n_kv_heads ceiling (host-gated)
constexpr int kQPad = 8;       // split score kernel: halves of padding per staged q row

template <typename Score>
__global__ void sparse_score_blocks_kernel(const half* __restrict__ q,
                                           const __half2* __restrict__ minmax_base,
                                           const int* __restrict__ block_tables,
                                           const int* __restrict__ context_lens, float* __restrict__ scores,
                                           int n_heads, int n_kv_heads, int head_dim, int block_size,
                                           int max_blocks_per_seq, int scores_stride, int engage_blocks,
                                           float std_coef, int q_stride, int bt_stride, int ctx_const) {
    const int seq = blockIdx.y;
    const int ctx_len = context_lens ? context_lens[seq] : ctx_const;
    const int n_blocks = (ctx_len + block_size - 1) / block_size;
    // Identity regime (selection copies the table verbatim) or a CTA past the
    // work: exit before the q smem staging - at short contexts the staging
    // dominated the launch (256 CTAs x 8 KiB of dead q traffic).
    if (n_blocks <= engage_blocks || blockIdx.x * kScoreWarps >= n_blocks)
        return;
    const int q_elems = n_heads * head_dim;
    extern __shared__ half q_smem[];
    const half* q_seq = q + ((int64_t)seq * q_stride);
    for (int i = threadIdx.x; i < q_elems; i += blockDim.x)
        q_smem[i] = q_seq[i];
    __syncthreads();

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int g = n_heads / n_kv_heads;
    const int row_elems = n_kv_heads * head_dim;
    const int* bt = block_tables + ((int64_t)seq * bt_stride);
    float* sc = scores + ((int64_t)seq * scores_stride);

    for (int b = (blockIdx.x * kScoreWarps) + warp; b < n_blocks; b += gridDim.x * kScoreWarps) {
        const int block_id = bt[b];
        if (block_id < 0) {
            if (lane == 0)
                sc[b] = -FLT_MAX;
            continue;
        }
        const __half2* mm = minmax_base + ((int64_t)block_id * row_elems);
        float best = -FLT_MAX;
        // 16-byte metadata loads (4 (min,max) pairs per lane per step) with a one-ahead prefetch
        // across kv heads: a scalar 8-byte dependent-load form is latency-bound. head_dim is a
        // multiple of 16 (kNVFP4Group), so head_dim*4B is 16B-aligned and lane*4 stays in-bounds
        // per 128-dim sweep.
        const int d_first = lane * 4;
        const bool lane_live = d_first < head_dim;
        float4 raw_next{};
        if (lane_live)
            raw_next = *reinterpret_cast<const float4*>(&mm[d_first]);
        for (int kvh = 0; kvh < n_kv_heads; kvh++) {
            float part[kMaxGroup];
#pragma unroll
            for (int h = 0; h < kMaxGroup; h++)
                part[h] = 0.0f;
            for (int d0 = d_first; d0 < head_dim; d0 += 4 * kWarpSize) {
                const float4 raw = (d0 == d_first)
                                       ? raw_next
                                       : *reinterpret_cast<const float4*>(&mm[(kvh * head_dim) + d0]);
                if (d0 == d_first && kvh + 1 < n_kv_heads && lane_live)
                    raw_next = *reinterpret_cast<const float4*>(&mm[((kvh + 1) * head_dim) + d_first]);
                const __half2 m01[2] = {*reinterpret_cast<const __half2*>(&raw.x),
                                        *reinterpret_cast<const __half2*>(&raw.y)};
                const __half2 m23[2] = {*reinterpret_cast<const __half2*>(&raw.z),
                                        *reinterpret_cast<const __half2*>(&raw.w)};
                // Full unroll with a guard keeps part[] register-resident (a
                // runtime trip count would spill it to a local frame).
#pragma unroll
                for (int h = 0; h < kMaxGroup; h++) {
                    if (h >= g)
                        break;
                    const half* qp = &q_smem[(((kvh * g) + h) * head_dim) + d0];
                    const __half2 q01 = *reinterpret_cast<const __half2*>(qp);
                    const __half2 q23 = *reinterpret_cast<const __half2*>(qp + 2);
                    float acc = 0.0f;
                    const float q0 = __low2float(q01), q1 = __high2float(q01);
                    const float q2 = __low2float(q23), q3 = __high2float(q23);
                    acc += Score::term(q0, __low2float(m01[0]), __high2float(m01[0]), std_coef);
                    acc += Score::term(q1, __low2float(m01[1]), __high2float(m01[1]), std_coef);
                    acc += Score::term(q2, __low2float(m23[0]), __high2float(m23[0]), std_coef);
                    acc += Score::term(q3, __low2float(m23[1]), __high2float(m23[1]), std_coef);
                    part[h] += acc;
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

// Same score with the warp split over (kv head, dim span): kLanesPerKv = 32 / n_kv_heads lanes per kv head,
// head_dim / kLanesPerKv dims per lane, all of a lane's 16-byte metadata loads issued before its sums, then a
// log2(kLanesPerKv)-step reduce per q head. Qwen3.8-27B at 50k context: the one-lane-per-4-dims form ran 8
// rows in 60.1 us (nsys), 120 shuffles per block per warp.
template <typename Score, int kLanesPerKv>
__global__ void sparse_score_blocks_split_kernel(
    const half* __restrict__ q, const __half2* __restrict__ minmax_base, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, float* __restrict__ scores, int n_heads, int n_kv_heads,
    int head_dim, int block_size, int scores_stride, int engage_blocks, float std_coef, int q_stride,
    int bt_stride, int ctx_const) {
    const int seq = blockIdx.y;
    const int ctx_len = context_lens ? context_lens[seq] : ctx_const;
    const int n_blocks = (ctx_len + block_size - 1) / block_size;
    if (n_blocks <= engage_blocks || blockIdx.x * kScoreWarps >= n_blocks)
        return;
    const int q_elems = n_heads * head_dim;
    extern __shared__ half q_smem[];
    const half* q_seq = q + ((int64_t)seq * q_stride);
    // Rows of head_dim + kQPad halves: the kLanesPerKv-lane groups of different kv heads hit different banks.
    for (int i = threadIdx.x; i < q_elems; i += blockDim.x)
        q_smem[((i / head_dim) * (head_dim + kQPad)) + (i % head_dim)] = q_seq[i];
    __syncthreads();

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int g = n_heads / n_kv_heads;
    const int row_elems = n_kv_heads * head_dim;
    const int kvh = lane / kLanesPerKv;
    const int d_lane = (lane % kLanesPerKv) * 4;  // a lane reads dims d_lane + 4 * kLanesPerKv * k
    const int* bt = block_tables + ((int64_t)seq * bt_stride);
    float* sc = scores + ((int64_t)seq * scores_stride);

    for (int b = (blockIdx.x * kScoreWarps) + warp; b < n_blocks; b += gridDim.x * kScoreWarps) {
        const int block_id = bt[b];
        if (block_id < 0) {
            if (lane == 0)
                sc[b] = -FLT_MAX;
            continue;
        }
        const __half2* mm = minmax_base + ((int64_t)block_id * row_elems) +
                            (static_cast<int64_t>(kvh) * head_dim) + d_lane;
        float part[kMaxGroup];
#pragma unroll
        for (int h = 0; h < kMaxGroup; h++)
            part[h] = 0.0f;
        for (int d = 0; d < head_dim; d += 4 * kLanesPerKv) {
            const float4 raw = *reinterpret_cast<const float4*>(&mm[d]);
            const __half2 m01[2] = {*reinterpret_cast<const __half2*>(&raw.x),
                                    *reinterpret_cast<const __half2*>(&raw.y)};
            const __half2 m23[2] = {*reinterpret_cast<const __half2*>(&raw.z),
                                    *reinterpret_cast<const __half2*>(&raw.w)};
#pragma unroll
            for (int h = 0; h < kMaxGroup; h++) {
                if (h >= g)
                    break;
                const half* qp = &q_smem[(((kvh * g) + h) * (head_dim + kQPad)) + d_lane + d];
                const __half2 q01 = *reinterpret_cast<const __half2*>(qp);
                const __half2 q23 = *reinterpret_cast<const __half2*>(qp + 2);
                float acc = 0.0f;
                acc += Score::term(__low2float(q01), __low2float(m01[0]), __high2float(m01[0]), std_coef);
                acc += Score::term(__high2float(q01), __low2float(m01[1]), __high2float(m01[1]), std_coef);
                acc += Score::term(__low2float(q23), __low2float(m23[0]), __high2float(m23[0]), std_coef);
                acc += Score::term(__high2float(q23), __low2float(m23[1]), __high2float(m23[1]), std_coef);
                part[h] += acc;
            }
        }
        float best = -FLT_MAX;
#pragma unroll
        for (int h = 0; h < kMaxGroup; h++) {
            if (h >= g)
                break;
            float s = part[h];
#pragma unroll
            for (int off = kLanesPerKv / 2; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
            best = fmaxf(best, s);
        }
#pragma unroll
        for (int off = kLanesPerKv; off < kWarpSize; off <<= 1)
            best = fmaxf(best, __shfl_xor_sync(0xFFFFFFFFu, best, off));
        if (lane == 0)
            sc[b] = best;
    }
}

// The split score kernel for this shape, nullptr = the per-4-dims kernel serves it.
template <typename Score>
using SplitScoreFn = void (*)(const half*, const __half2*, const int*, const int*, float*, int, int, int, int,
                              int, int, float, int, int, int);
template <typename Score>
SplitScoreFn<Score> split_score_kernel(int n_kv_heads, int head_dim) {
    if (n_kv_heads <= 0 || kWarpSize % n_kv_heads != 0)
        return nullptr;
    const int lanes = kWarpSize / n_kv_heads;
    if (head_dim % (4 * lanes) != 0)
        return nullptr;
    switch (lanes) {
        case 4:
            return sparse_score_blocks_split_kernel<Score, 4>;
        case 8:
            return sparse_score_blocks_split_kernel<Score, 8>;
        case 16:
            return sparse_score_blocks_split_kernel<Score, 16>;
        default:
            return nullptr;
    }
}

template <typename Score>
void launch_score(dim3 grid, size_t q_smem, cudaStream_t stream, const half* q, const __half2* mm,
                  const int* block_tables, const int* context_lens, float* scores, int n_heads,
                  int n_kv_heads, int head_dim, int block_size, int max_blocks_per_seq, int scores_stride,
                  int engage_blocks, float std_coef, int q_stride, int bt_stride, int ctx_const) {
    if (auto* split = split_score_kernel<Score>(n_kv_heads, head_dim)) {
        split<<<grid, kScoreThreads, static_cast<size_t>(n_heads) * (head_dim + kQPad) * sizeof(half),
                stream>>>(q, mm, block_tables, context_lens, scores, n_heads, n_kv_heads, head_dim,
                          block_size, scores_stride, engage_blocks, std_coef, q_stride, bt_stride, ctx_const);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    sparse_score_blocks_kernel<Score>
        <<<grid, kScoreThreads, q_smem, stream>>>(q, mm, block_tables, context_lens, scores, n_heads,
                                                  n_kv_heads, head_dim, block_size, max_blocks_per_seq,
                                                  scores_stride, engage_blocks, std_coef, q_stride, bt_stride,
                                                  ctx_const);
    IMP_CUDA_CHECK_LAUNCH();
}

// Sparse prefill: one past-page selection per query chunk from sampled rows. A block's
// chunk score is max over rows of (score_r(b) - max_b score_r(b)): the gap to each row's best
// block, so a high-norm row does not outrank the others. One CTA, rows <= kMaxPrefillRows.
constexpr int kMergeThreads = 1024;
constexpr int kMaxPrefillRows = 64;

__global__ void sparse_prefill_merge_rows_kernel(const float* __restrict__ scores, float* __restrict__ agg,
                                                 int rows, int n_blocks, int stride) {
    __shared__ float row_max[kMaxPrefillRows];
    __shared__ float red[kMergeThreads / kWarpSize];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    for (int r = 0; r < rows; r++) {
        float m = -FLT_MAX;
        for (int b = threadIdx.x; b < n_blocks; b += blockDim.x)
            m = fmaxf(m, scores[((int64_t)r * stride) + b]);
        m = warp_reduce_max(m);
        if (lane == 0)
            red[warp] = m;
        __syncthreads();
        if (warp == 0) {
            m = lane < (int)(blockDim.x / kWarpSize) ? red[lane] : -FLT_MAX;
            m = warp_reduce_max(m);
            if (lane == 0)
                row_max[r] = m;
        }
        __syncthreads();
    }
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float best = -FLT_MAX;
        for (int r = 0; r < rows; r++)
            best = fmaxf(best, scores[((int64_t)r * stride) + b] - row_max[r]);
        agg[b] = best;
    }
}

__global__ void sparse_prefill_set_len_kernel(int* __restrict__ len, int value) { *len = value; }

// Top-k selection + compacted table build, one CTA per sequence. Selection key is
// 64-bit: (monotone score bits << 32) | ~block_index, unique per block, so the radix
// threshold is exact, ties resolve to the LOWER block index, result is deterministic.
// 8x 8-bit MSB radix passes narrow the k-th largest key among middle blocks; sink and
// recent blocks are forced. Bitmap + ballot compaction emits selected physical block ids
// in ascending order, preserving the dense kernel's softmax accumulation order.
constexpr int kSelectThreads = 256;

__device__ __forceinline__ uint32_t score_key(float s) {
    uint32_t u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void sparse_select_topk_kernel(const float* __restrict__ scores,
                                          const int* __restrict__ block_tables,
                                          const int* __restrict__ context_lens, int* __restrict__ sparse_bt,
                                          int* __restrict__ sparse_ctx, int block_size,
                                          int max_blocks_per_seq, int scores_stride, int budget_blocks,
                                          int sink_blocks, int recent_blocks, int engage_blocks,
                                          int table_blocks) {
    const int seq = blockIdx.x;
    const int ctx_len = context_lens[seq];
    const int n_blocks = (ctx_len + block_size - 1) / block_size;
    const int* bt = block_tables + ((int64_t)seq * max_blocks_per_seq);
    int* out = sparse_bt + ((int64_t)seq * table_blocks);

    if (n_blocks <= engage_blocks) {
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
    const float* sc = scores + ((int64_t)seq * scores_stride);

    // Dynamic smem: hist[256] | bitmap words | word ranks | cached score keys. The radix
    // passes iterate the middle keys 8 times; reading float scores from global every pass
    // makes the single-CTA kernel latency-bound, so read once into smem and run every pass
    // over smem.
    extern __shared__ uint32_t sel_smem[];
    uint32_t* hist = sel_smem;  // 256
    const int n_words = (max_blocks_per_seq + 31) / 32;
    uint32_t* bitmap = hist + 256;           // n_words
    uint32_t* word_rank = bitmap + n_words;  // n_words
    uint32_t* tie_bm = word_rank + n_words;  // n_words
    uint32_t* tie_rank = tie_bm + n_words;   // n_words
    uint32_t* keys = tie_rank + n_words;     // max_blocks_per_seq
    __shared__ uint32_t s_prefix;
    __shared__ int s_k_rem;
    __shared__ int s_bin;

    if (threadIdx.x == 0) {
        s_prefix = 0;
        s_k_rem = k;
    }
    for (int b = mid_lo + threadIdx.x; b < mid_hi; b += blockDim.x)
        keys[b] = score_key(sc[b]);

    // 4 MSB-first radix passes over the 32-bit score keys (multiplicity kept; ties resolve
    // by ascending block index in the bitmap phase below). The per-pass threshold bin comes
    // from a parallel suffix scan of the histogram: a serial thread-0 scan over 8 passes
    // makes this single-CTA kernel latency-bound.
    for (int level = 3; level >= 0; level--) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x)
            hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        const int shift = level * 8;
        for (int b = mid_lo + threadIdx.x; b < mid_hi; b += blockDim.x) {
            const uint32_t key = keys[b];
            if (level == 3 || (key >> (shift + 8)) == prefix)
                atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        // Inclusive suffix scan of hist in place (Hillis-Steele).
        const int t = threadIdx.x;
        for (int off = 1; off < 256; off <<= 1) {
            uint32_t add = 0;
            if (t < 256 && t + off < 256)
                add = hist[t + off];
            __syncthreads();
            if (t < 256)
                hist[t] += add;
            __syncthreads();
        }
        // hist[t] = candidates with byte >= t. The threshold byte is the
        // unique t with hist[t+1] < k_rem <= hist[t].
        const int k_rem = s_k_rem;
        if (t < 256) {
            const uint32_t at = hist[t];
            const uint32_t above = (t < 255) ? hist[t + 1] : 0;
            if ((int)above < k_rem && k_rem <= (int)at)
                s_bin = t;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const uint32_t above = (s_bin < 255) ? hist[s_bin + 1] : 0;
            s_prefix = (s_prefix << 8) | (uint32_t)s_bin;
            s_k_rem = k_rem - (int)above;
        }
        __syncthreads();
    }
    // T32 = the k-th largest score key (with multiplicity); k_rem = how many
    // keys EQUAL to it still belong in the top k, taken by ascending index.
    const uint32_t threshold = s_prefix;
    const int tie_take = s_k_rem;

    for (int i = threadIdx.x; i < n_words; i += blockDim.x) {
        bitmap[i] = 0;
        tie_bm[i] = 0;
    }
    __syncthreads();
    // Forced sinks + recents.
    for (int b = threadIdx.x; b < mid_lo; b += blockDim.x)
        atomicOr(&bitmap[b >> 5], 1u << (b & 31));
    for (int b = mid_hi + threadIdx.x; b < n_blocks; b += blockDim.x)
        atomicOr(&bitmap[b >> 5], 1u << (b & 31));
    // Middles: strictly-above selected outright, ties marked for rank-capped
    // resolution.
    for (int b = mid_lo + threadIdx.x; b < mid_hi; b += blockDim.x) {
        const uint32_t key = keys[b];
        if (key > threshold)
            atomicOr(&bitmap[b >> 5], 1u << (b & 31));
        else if (key == threshold)
            atomicOr(&tie_bm[b >> 5], 1u << (b & 31));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t run = 0;
        for (int w = 0; w < n_words; w++) {
            tie_rank[w] = run;
            run += __popc(tie_bm[w]);
        }
    }
    __syncthreads();
    // Take the tie_take lowest-index ties into the selection.
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        uint32_t bits = tie_bm[w];
        uint32_t rank = tie_rank[w];
        while (bits) {
            const int bit = __ffs(bits) - 1;
            if ((int)rank < tie_take)
                atomicOr(&bitmap[w], 1u << bit);
            rank++;
            bits &= bits - 1;
        }
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
            out[rank++] = bt[(w * 32) + bit];
            bits &= bits - 1;
        }
    }
    if (threadIdx.x == 0) {
        const int tail_fill = ctx_len - ((n_blocks - 1) * block_size);
        sparse_ctx[seq] = ((budget_blocks - 1) * block_size) + tail_fill;
    }
}

}  // namespace

bool sparse_score_uses_split(int n_kv_heads, int head_dim) {
    return split_score_kernel<ScoreMeanStd>(n_kv_heads, head_dim) != nullptr;
}

void sparse_update_key_minmax_all_layers(QType cache_dtype, const void* k_base, int64_t k_layer_stride_bytes,
                                         const void* k_scale_base, int64_t sc_layer_stride_bytes,
                                         void* minmax_base, int64_t mm_layer_stride_bytes,
                                         const int* positions, const int* block_tables,
                                         const int* seq_offsets, int n_layers, int n_kv_heads, int head_dim,
                                         int block_size, int n_tokens, int max_blocks_per_seq,
                                         int n_sequences, cudaStream_t stream) {
    if (n_tokens <= 0 || n_layers <= 0)
        return;
    const int row_elems = n_kv_heads * head_dim;
    const int threads = 128;
    const dim3 grid(n_tokens, n_layers);
    const int64_t mm_stride = mm_layer_stride_bytes / (int64_t)sizeof(__half2);
    if (cache_dtype == QType::NVFP4) {
        // The init gate refuses NVFP4 without a scale pool, so a null here
        // would be a wiring bug, not a configuration: fail loud rather than
        // write metadata from garbage.
        if (k_scale_base == nullptr) {
            IMP_LOG_ERROR("sparse_update_key_minmax: NVFP4 cache without a scale pool");
            return;
        }
        sparse_update_key_minmax_layers_nvfp4_kernel<<<grid, threads, 0, stream>>>(
            static_cast<const uint8_t*>(k_base), k_layer_stride_bytes,
            static_cast<const uint8_t*>(k_scale_base), sc_layer_stride_bytes,
            static_cast<__half2*>(minmax_base), mm_stride, positions, block_tables, seq_offsets, row_elems,
            block_size, n_tokens, max_blocks_per_seq, n_sequences);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (cache_dtype == QType::FP8_E4M3) {
        sparse_update_key_minmax_layers_kernel<__nv_fp8_e4m3>
            <<<grid, threads, 0, stream>>>(static_cast<const __nv_fp8_e4m3*>(k_base), k_layer_stride_bytes,
                                           static_cast<__half2*>(minmax_base), mm_stride, positions,
                                           block_tables, seq_offsets, row_elems, block_size, n_tokens,
                                           max_blocks_per_seq, n_sequences);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        sparse_update_key_minmax_layers_kernel<half>
            <<<grid, threads, 0, stream>>>(static_cast<const half*>(k_base),
                                           k_layer_stride_bytes / (int64_t)sizeof(half),
                                           static_cast<__half2*>(minmax_base), mm_stride, positions,
                                           block_tables, seq_offsets, row_elems, block_size, n_tokens,
                                           max_blocks_per_seq, n_sequences);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

namespace {

void launch_select_topk(int n_seq, const float* scores, const int* block_tables, const int* context_lens,
                        int* sparse_block_tables, int* sparse_context_lens, int block_size,
                        int max_blocks_per_seq, int budget_blocks, int sink_blocks, int recent_blocks,
                        int engage_blocks, int table_blocks, cudaStream_t stream) {
    const int n_words = (max_blocks_per_seq + 31) / 32;
    const size_t sel_smem = (256 + (4 * (size_t)n_words) + (size_t)max_blocks_per_seq) * sizeof(uint32_t);
    // 128k-context tables need ~35 KiB; opt in past the 48 KiB default ONLY
    // when actually exceeded - the attribute sticks to the function and can
    // shift the L1/SMEM carveout the driver picks around it.
    constexpr size_t kSmemDefault = static_cast<int64_t>(48) * 1024;
    static size_t sel_smem_granted = kSmemDefault;
    if (sel_smem > sel_smem_granted) {
        // Failed opt-in: grant not recorded (retried next call); the launch below fails and is reported.
        if (const cudaError_t err = cudaFuncSetAttribute(sparse_select_topk_kernel,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         (int)sel_smem);
            err == cudaSuccess)
            sel_smem_granted = sel_smem;
        else
            IMP_LOG_ERROR("sparse_select_topk: smem opt-in %zu B failed: %s", sel_smem,
                          cudaGetErrorString(err));
    }
    sparse_select_topk_kernel<<<n_seq, kSelectThreads, sel_smem, stream>>>(
        scores, block_tables, context_lens, sparse_block_tables, sparse_context_lens, block_size,
        max_blocks_per_seq, max_blocks_per_seq, budget_blocks, sink_blocks, recent_blocks, engage_blocks,
        table_blocks);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace

void sparse_select_blocks(const half* q, const void* minmax_base, const int* block_tables,
                          const int* context_lens, int n_seq, int n_heads, int n_kv_heads, int head_dim,
                          int block_size, int max_blocks_per_seq, int budget_blocks, int sink_blocks,
                          int recent_blocks, int engage_blocks, int table_blocks, float* scores_scratch,
                          int* sparse_block_tables, int* sparse_context_lens, float std_coef,
                          cudaStream_t stream) {
    // Fixed grid.x: work distribution adapts device-side via grid-stride, so a captured
    // graph stays correct while the context grows during replay. 256 CTAs: fewer leaves the
    // kernel latency-bound at low SM occupancy; the grid-stride loop makes the extra CTAs
    // free at short ctx.
    dim3 score_grid(256, n_seq);
    const size_t q_smem = (size_t)n_heads * head_dim * sizeof(half);
    const auto* mm = static_cast<const __half2*>(minmax_base);
    launch_score<ScoreMeanStd>(score_grid, q_smem, stream, q, mm, block_tables, context_lens, scores_scratch,
                               n_heads, n_kv_heads, head_dim, block_size, max_blocks_per_seq,
                               max_blocks_per_seq, engage_blocks, std_coef, n_heads * head_dim,
                               max_blocks_per_seq, 0);

    launch_select_topk(n_seq, scores_scratch, block_tables, context_lens, sparse_block_tables,
                       sparse_context_lens, block_size, max_blocks_per_seq, budget_blocks, sink_blocks,
                       recent_blocks, engage_blocks, table_blocks, stream);
}

int sparse_prefill_past_tokens(int past_len, int block_size, int budget_blocks) {
    const int n_blocks = (past_len + block_size - 1) / block_size;
    if (n_blocks <= budget_blocks)
        return past_len;
    return ((budget_blocks - 1) * block_size) + (past_len - ((n_blocks - 1) * block_size));
}

int sparse_prefill_select_past(const half* q, int n_rows, int sample_rows, const void* minmax_layer,
                               const int* past_block_table, int past_len, int n_heads, int n_kv_heads,
                               int head_dim, int block_size, int capacity_blocks, int budget_blocks,
                               int sink_blocks, int recent_blocks, float std_coef, float* scores_scratch,
                               float* agg_scratch, int* out_block_table, int* out_ctx_scratch,
                               cudaStream_t stream) {
    const int n_blocks = (past_len + block_size - 1) / block_size;
    if (n_rows <= 0 || past_len <= 0 || n_blocks > capacity_blocks)
        return -1;
    const int rows = std::max(1, std::min({sample_rows, n_rows, kMaxPrefillRows}));
    // Rows (i+1)*n/rows - 1: evenly spread over the chunk, the last row always in.
    const int step = n_rows / rows;
    const int first = n_rows - 1 - ((rows - 1) * step);
    const int q_elems = n_heads * head_dim;
    if (n_blocks > budget_blocks) {
        dim3 score_grid(256, rows);
        const size_t q_smem = (size_t)q_elems * sizeof(half);
        const half* q_first = q + ((int64_t)first * q_elems);
        const auto* mm = static_cast<const __half2*>(minmax_layer);
        launch_score<ScoreMeanStd>(score_grid, q_smem, stream, q_first, mm, past_block_table, nullptr,
                                   scores_scratch, n_heads, n_kv_heads, head_dim, block_size, capacity_blocks,
                                   capacity_blocks, /*engage_blocks=*/0, std_coef, step * q_elems,
                                   /*bt_stride=*/0, past_len);
        sparse_prefill_merge_rows_kernel<<<1, kMergeThreads, 0, stream>>>(scores_scratch, agg_scratch, rows,
                                                                          n_blocks, capacity_blocks);
        IMP_CUDA_CHECK_LAUNCH();
    }
    sparse_prefill_set_len_kernel<<<1, 1, 0, stream>>>(out_ctx_scratch, past_len);
    IMP_CUDA_CHECK_LAUNCH();
    // n_blocks <= budget: the select kernel's identity branch copies the table (dense equivalent).
    launch_select_topk(1, agg_scratch, past_block_table, out_ctx_scratch, out_block_table,
                       out_ctx_scratch + 1, block_size, capacity_blocks, budget_blocks, sink_blocks,
                       recent_blocks,
                       /*engage_blocks=*/budget_blocks, capacity_blocks, stream);
    return sparse_prefill_past_tokens(past_len, block_size, budget_blocks);
}

}  // namespace imp
