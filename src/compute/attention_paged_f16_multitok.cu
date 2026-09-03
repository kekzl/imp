// F16 paged decode attention, TOK tokens per warp iteration, HPC Q heads per
// CTA (attention.paged_f16_multitok, 2026-09-03).
//
// The cooperative GQA kernel (attention_paged.cu) stages each KV block into
// shared memory with one 2-byte load per element and a runtime head_dim
// division per element, then walks one token per warp iteration. ncu at
// 32 x 1100 (32/8 heads, HD=128): 316 us, DRAM 22%, warps active 51%, top
// stall long_scoreboard 6.2, 213M instructions for 144 MB of KV. Here a lane
// holds HEAD_DIM/32 contiguous elements and loads its 8 or 16 bytes of a K
// row straight from global, TOK rows before any reduction; the K row is
// converted once and dotted against HPC Q heads, so a CTA of NUM_WARPS warps
// reads the KV group once for HPC heads and the grid has
// n_kv_heads x (n_q_per_kv / HPC) CTAs per sequence. The softmax state is
// the unnormalised (m, l, o) form per head, normalised once at the end so
// the shared cross-warp merge is unchanged. HD=128/256; sliding window via
// the effective start and the StreamingLLM sentinel like the FP8 kernel;
// sink-token (n_sinks) geometry stays on the cooperative kernel.
//
// The split-K instance (batch x heads below the CTA target, i.e. single
// stream at long context) walks its share of the blocks and writes one
// (m, l, o) partial per head for the shared reduce kernel. The per-head
// split-K pipeline kernel re-reads the KV group once per Q head through L2:
// batch 1 x 32k on 32/8 measured 178.9 us (750 GB/s) against 91.1 us
// (1474 GB/s) on the 8/8 MHA shape with the same bytes (2026-09-03); the
// HPC sharing removes that factor.
#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace imp {

namespace {

template <int ELEMS>
struct LaneVec;
template <>
struct LaneVec<4> {
    using type = uint2;
};
template <>
struct LaneVec<8> {
    using type = uint4;
};

template <int ELEMS>
__device__ __forceinline__ void lane_vec_to_float(const typename LaneVec<ELEMS>::type& v, float* out) {
    const half2* h2 = reinterpret_cast<const half2*>(&v);
#pragma unroll
    for (int i = 0; i < ELEMS / 2; i++) {
        const float2 f = __half22float2(h2[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

// Per-warp softmax state for HPC heads.
template <int HEAD_DIM, int HPC>
struct HeadState {
    static constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    float m[HPC];
    float l[HPC];
    float o[HPC][ELEMS];
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int h = 0; h < HPC; h++) {
            m[h] = -FLT_MAX;
            l[h] = 0.0f;
#pragma unroll
            for (int e = 0; e < ELEMS; e++)
                o[h][e] = 0.0f;
        }
    }
    // Divide o by l once so the shared merges (which expect per-warp o
    // already normalised, weight = exp(m - gmax) * l) are unchanged.
    __device__ __forceinline__ void normalise() {
#pragma unroll
        for (int h = 0; h < HPC; h++) {
            if (l[h] > 0.0f) {
                const float inv_l = 1.0f / l[h];
#pragma unroll
                for (int e = 0; e < ELEMS; e++)
                    o[h][e] *= inv_l;
            }
        }
    }
};

// One KV block for one warp: rows [first_tok, n_tok) in groups of TOK, each K
// row loaded as one lane vector, converted once and dotted against HPC heads.
template <int HEAD_DIM, int TOK, int HPC>
__device__ __forceinline__ void f16_block_multitok(const half* __restrict__ K_block,
                                                   const half* __restrict__ V_block, int first_tok, int n_tok,
                                                   int kv_slot_stride,
                                                   const float (&q_reg)[HPC][HEAD_DIM / WARP_SIZE],
                                                   float scale, float softcap, HeadState<HEAD_DIM, HPC>& st) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    using Vec = typename LaneVec<ELEMS>::type;
    for (int t = first_tok; t < n_tok; t += TOK) {
        // K rows of TOK tokens in flight before any reduction; a partial
        // group clamps to a valid row and is masked to p = 0 below.
        Vec kp[TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const int ti = (t + i < n_tok) ? (t + i) : (n_tok - 1);
            kp[i] = __ldcs(reinterpret_cast<const Vec*>(K_block + ti * kv_slot_stride));
        }
        float dot[HPC][TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            float kf[ELEMS];
            lane_vec_to_float<ELEMS>(kp[i], kf);
#pragma unroll
            for (int h = 0; h < HPC; h++) {
                float d = 0.0f;
#pragma unroll
                for (int e = 0; e < ELEMS; e++)
                    d = fmaf(q_reg[h][e], kf[e], d);
                dot[h][i] = d;
            }
        }
        // HPC x TOK independent warp reductions; the shuffle chains interleave.
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
#pragma unroll
            for (int h = 0; h < HPC; h++)
#pragma unroll
                for (int i = 0; i < TOK; i++)
                    dot[h][i] += __shfl_xor_sync(0xffffffffu, dot[h][i], off);
        }
        float p[HPC][TOK];
        float alpha[HPC];
#pragma unroll
        for (int h = 0; h < HPC; h++) {
            float m_new = st.m[h];
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                dot[h][i] = apply_softcap(dot[h][i] * scale, softcap);
                if (t + i >= n_tok)
                    dot[h][i] = -FLT_MAX;
                m_new = fmaxf(m_new, dot[h][i]);
            }
            alpha[h] = expf(st.m[h] - m_new);
            float p_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                p[h][i] = (t + i < n_tok) ? expf(dot[h][i] - m_new) : 0.0f;
                p_sum += p[h][i];
            }
            st.l[h] = alpha[h] * st.l[h] + p_sum;
            st.m[h] = m_new;
        }
        // V rows of the group in flight together; masked tokens carry p = 0.
        Vec vp[TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const int ti = (t + i < n_tok) ? (t + i) : (n_tok - 1);
            vp[i] = __ldcs(reinterpret_cast<const Vec*>(V_block + ti * kv_slot_stride));
        }
#pragma unroll
        for (int h = 0; h < HPC; h++)
#pragma unroll
            for (int e = 0; e < ELEMS; e++)
                st.o[h][e] *= alpha[h];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            float vf[ELEMS];
            lane_vec_to_float<ELEMS>(vp[i], vf);
#pragma unroll
            for (int h = 0; h < HPC; h++)
#pragma unroll
                for (int e = 0; e < ELEMS; e++)
                    st.o[h][e] = fmaf(p[h][i], vf[e], st.o[h][e]);
        }
    }
}

template <int HEAD_DIM, int HPC>
__device__ __forceinline__ void load_q_heads(const half* __restrict__ Q, int batch_idx, int n_heads,
                                             int head0, int lane_offset,
                                             float (&q_reg)[HPC][HEAD_DIM / WARP_SIZE]) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    using Vec = typename LaneVec<ELEMS>::type;
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        const Vec qv = *reinterpret_cast<const Vec*>(Q + (int64_t)batch_idx * n_heads * HEAD_DIM +
                                                     (int64_t)(head0 + h) * HEAD_DIM + lane_offset);
        lane_vec_to_float<ELEMS>(qv, q_reg[h]);
    }
}

template <int HEAD_DIM, int TOK, int HPC>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_decode_f16_multitok_kernel(
    const half* __restrict__ Q, const half* __restrict__ K_cache, const half* __restrict__ V_cache,
    half* __restrict__ O, const int* __restrict__ block_tables, const int* __restrict__ context_lens,
    int n_heads, int n_kv_heads, int n_q_per_kv, int block_size, float scale, int max_num_blocks,
    int sliding_window, float softcap, const half* __restrict__ attn_sinks) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    static_assert(ELEMS == 4 || ELEMS == 8, "HD=128 (uint2 per lane) or HD=256 (uint4 per lane)");

    const int batch_idx = blockIdx.x;
    const int groups_per_kv = n_q_per_kv / HPC;
    const int kv_head = blockIdx.y / groups_per_kv;
    const int head0 = kv_head * n_q_per_kv + (blockIdx.y % groups_per_kv) * HPC;
    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    float q_reg[HPC][ELEMS];
    load_q_heads<HEAD_DIM, HPC>(Q, batch_idx, n_heads, head0, lane_offset, q_reg);

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;

    HeadState<HEAD_DIM, HPC> st;
    st.init();

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        const int phys_block = bt[blk];
        if (phys_block < 0)
            continue;  // StreamingLLM sentinel, same guard as the plain kernels
        const int tok_start = blk * block_size;
        int n_tok = block_size;
        if (tok_start + n_tok > ctx_len)
            n_tok = ctx_len - tok_start;
        const int first_tok = (tok_start < effective_start) ? (effective_start - tok_start) : 0;
        f16_block_multitok<HEAD_DIM, TOK, HPC>(K_cache + (int64_t)phys_block * kv_block_stride +
                                                   kv_head * HEAD_DIM + lane_offset,
                                               V_cache + (int64_t)phys_block * kv_block_stride +
                                                   kv_head * HEAD_DIM + lane_offset,
                                               first_tok, n_tok, kv_slot_stride, q_reg, scale, softcap, st);
    }

    st.normalise();
    extern __shared__ char smem_f16_mt[];
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        __syncthreads();
        crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_f16_mt), st.m[h], st.l[h], st.o[h],
                                             warp_id, lane_id, lane_offset, O, batch_idx, n_heads, head0 + h,
                                             attn_sinks);
    }
}

// Split-K instance: grid (batch, n_kv_heads x groups, num_splits); each CTA
// walks its share of the blocks for HPC heads and writes one partial per head
// in the layout the shared reduce kernel expects.
template <int HEAD_DIM, int TOK, int HPC>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_splitk_f16_multitok_kernel(
    const half* __restrict__ Q, const half* __restrict__ K_cache, const half* __restrict__ V_cache,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int n_heads, int n_kv_heads, int n_q_per_kv, int block_size,
    float scale, int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    static_assert(ELEMS == 4 || ELEMS == 8, "HD=128 (uint2 per lane) or HD=256 (uint4 per lane)");

    const int batch_idx = blockIdx.x;
    const int groups_per_kv = n_q_per_kv / HPC;
    const int kv_head = blockIdx.y / groups_per_kv;
    const int head0 = kv_head * n_q_per_kv + (blockIdx.y % groups_per_kv) * HPC;
    const int split_idx = blockIdx.z;
    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    const int total_blocks = num_ctx_blocks - first_block;
    const int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    const int split_start = first_block + split_idx * blocks_per_split;
    int split_end = split_start + blocks_per_split;
    if (split_end > num_ctx_blocks)
        split_end = num_ctx_blocks;
    if (split_start >= split_end) {
#pragma unroll
        for (int h = 0; h < HPC; h++)
            write_empty_split_sentinel<HEAD_DIM>(partial_out, batch_idx, n_heads, head0 + h, num_splits,
                                                 split_idx, lane_offset);
        return;
    }

    float q_reg[HPC][ELEMS];
    load_q_heads<HEAD_DIM, HPC>(Q, batch_idx, n_heads, head0, lane_offset, q_reg);

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;

    HeadState<HEAD_DIM, HPC> st;
    st.init();

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        const int phys_block = bt[blk];
        if (phys_block < 0)
            continue;
        const int tok_start = blk * block_size;
        int n_tok = block_size;
        if (tok_start + n_tok > ctx_len)
            n_tok = ctx_len - tok_start;
        const int first_tok = (tok_start < effective_start) ? (effective_start - tok_start) : 0;
        f16_block_multitok<HEAD_DIM, TOK, HPC>(K_cache + (int64_t)phys_block * kv_block_stride +
                                                   kv_head * HEAD_DIM + lane_offset,
                                               V_cache + (int64_t)phys_block * kv_block_stride +
                                                   kv_head * HEAD_DIM + lane_offset,
                                               first_tok, n_tok, kv_slot_stride, q_reg, scale, softcap, st);
    }

    st.normalise();
    extern __shared__ char smem_f16_sk_mt[];
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        __syncthreads();
        crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_f16_sk_mt), st.m[h], st.l[h], st.o[h],
                                          warp_id, lane_id, lane_offset, partial_out, batch_idx, n_heads,
                                          head0 + h, num_splits, split_idx);
    }
}

template <int HEAD_DIM, int HPC>
void launch_f16_multitok(const half* Q, const half* K_cache, const half* V_cache, half* O,
                         const int* block_tables, const int* context_lens, int batch_size, int n_heads,
                         int n_kv_heads, int n_q_per_kv, int block_size, float scale, int max_num_blocks,
                         int sliding_window, float softcap, const half* attn_sinks, cudaStream_t stream) {
    const size_t smem_bytes = NUM_WARPS * sizeof(float) * 2 + NUM_WARPS * HEAD_DIM * sizeof(float);
    dim3 grid(batch_size, n_kv_heads * (n_q_per_kv / HPC));
    dim3 block(BLOCK_THREADS);
    paged_attention_decode_f16_multitok_kernel<HEAD_DIM, 4, HPC>
        <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, O, block_tables, context_lens, n_heads,
                                              n_kv_heads, n_q_per_kv, block_size, scale, max_num_blocks,
                                              sliding_window, softcap, attn_sinks);
    IMP_CUDA_CHECK_LAUNCH();
}

template <int HEAD_DIM, int HPC>
void launch_f16_splitk_multitok(const half* Q, const half* K_cache, const half* V_cache, float* partial,
                                const int* block_tables, const int* context_lens, int batch_size, int n_heads,
                                int n_kv_heads, int n_q_per_kv, int block_size, float scale,
                                int max_num_blocks, int num_splits, int sliding_window, float softcap,
                                cudaStream_t stream) {
    const size_t smem_bytes = NUM_WARPS * sizeof(float) * 2 + NUM_WARPS * HEAD_DIM * sizeof(float);
    dim3 grid(batch_size, n_kv_heads * (n_q_per_kv / HPC), num_splits);
    dim3 block(BLOCK_THREADS);
    paged_attention_splitk_f16_multitok_kernel<HEAD_DIM, 4, HPC>
        <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, partial, block_tables, context_lens,
                                              n_heads, n_kv_heads, n_q_per_kv, block_size, scale,
                                              max_num_blocks, num_splits, sliding_window, softcap);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace

int paged_attention_f16_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested) {
    if (head_dim != 128 && head_dim != 256)
        return 0;
    if (n_q_per_kv < 1 || n_q_per_kv > 8)
        return 0;
    // Heads per CTA: the largest of 4 / 2 / 1 that divides the GQA ratio, or
    // the caller's choice when it divides. HD=256 caps at 2 (register budget).
    int hpc = requested;
    if (hpc != 1 && hpc != 2 && hpc != 4)
        hpc = 0;
    if (hpc == 0 || n_q_per_kv % hpc != 0) {
        hpc = (n_q_per_kv % 4 == 0) ? 4 : (n_q_per_kv % 2 == 0) ? 2 : 1;
        if (head_dim == 256 && hpc > 2)
            hpc = 2;
    }
    return hpc;
}

bool paged_attention_decode_f16_multitok_launch(const half* Q, const half* K_cache, const half* V_cache,
                                                half* O, const int* block_tables, const int* context_lens,
                                                int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                                int block_size, float scale, int max_num_blocks,
                                                int sliding_window, float softcap, const half* attn_sinks,
                                                int heads_per_cta, cudaStream_t stream) {
    const int n_q_per_kv = n_heads / n_kv_heads;
    const int hpc = paged_attention_f16_multitok_heads_per_cta(head_dim, n_q_per_kv, heads_per_cta);
    if (hpc == 0)
        return false;
#define LAUNCH_F16_MT(HD, HPCV)                                                                              \
    launch_f16_multitok<HD, HPCV>(Q, K_cache, V_cache, O, block_tables, context_lens, batch_size, n_heads,   \
                                  n_kv_heads, n_q_per_kv, block_size, scale, max_num_blocks, sliding_window, \
                                  softcap, attn_sinks, stream)
    if (head_dim == 128) {
        if (hpc == 4)
            LAUNCH_F16_MT(128, 4);
        else if (hpc == 2)
            LAUNCH_F16_MT(128, 2);
        else
            LAUNCH_F16_MT(128, 1);
    } else {
        if (hpc == 4)
            LAUNCH_F16_MT(256, 4);
        else if (hpc == 2)
            LAUNCH_F16_MT(256, 2);
        else
            LAUNCH_F16_MT(256, 1);
    }
#undef LAUNCH_F16_MT
    return true;
}

bool paged_attention_splitk_f16_multitok_launch(const half* Q, const half* K_cache, const half* V_cache,
                                                float* partial, const int* block_tables,
                                                const int* context_lens, int batch_size, int n_heads,
                                                int n_kv_heads, int head_dim, int block_size, float scale,
                                                int max_num_blocks, int num_splits, int sliding_window,
                                                float softcap, int heads_per_cta, cudaStream_t stream) {
    const int n_q_per_kv = n_heads / n_kv_heads;
    const int hpc = paged_attention_f16_multitok_heads_per_cta(head_dim, n_q_per_kv, heads_per_cta);
    if (hpc == 0 || num_splits < 1)
        return false;
#define LAUNCH_F16_SK_MT(HD, HPCV)                                                                       \
    launch_f16_splitk_multitok<HD, HPCV>(Q, K_cache, V_cache, partial, block_tables, context_lens,       \
                                         batch_size, n_heads, n_kv_heads, n_q_per_kv, block_size, scale, \
                                         max_num_blocks, num_splits, sliding_window, softcap, stream)
    if (head_dim == 128) {
        if (hpc == 4)
            LAUNCH_F16_SK_MT(128, 4);
        else if (hpc == 2)
            LAUNCH_F16_SK_MT(128, 2);
        else
            LAUNCH_F16_SK_MT(128, 1);
    } else {
        if (hpc == 4)
            LAUNCH_F16_SK_MT(256, 4);
        else if (hpc == 2)
            LAUNCH_F16_SK_MT(256, 2);
        else
            LAUNCH_F16_SK_MT(256, 1);
    }
#undef LAUNCH_F16_SK_MT
    return true;
}

}  // namespace imp
