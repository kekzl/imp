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

template <int HEAD_DIM, int TOK, int HPC>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_decode_f16_multitok_kernel(
    const half* __restrict__ Q, const half* __restrict__ K_cache, const half* __restrict__ V_cache,
    half* __restrict__ O, const int* __restrict__ block_tables, const int* __restrict__ context_lens,
    int n_heads, int n_kv_heads, int n_q_per_kv, int block_size, float scale, int max_num_blocks,
    int sliding_window, float softcap, const half* __restrict__ attn_sinks) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    static_assert(ELEMS == 4 || ELEMS == 8, "HD=128 (uint2 per lane) or HD=256 (uint4 per lane)");
    using Vec = typename LaneVec<ELEMS>::type;

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
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        const Vec qv = *reinterpret_cast<const Vec*>(Q + (int64_t)batch_idx * n_heads * HEAD_DIM +
                                                     (int64_t)(head0 + h) * HEAD_DIM + lane_offset);
        lane_vec_to_float<ELEMS>(qv, q_reg[h]);
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;

    float m_w[HPC], l_w[HPC], o_reg[HPC][ELEMS];
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        m_w[h] = -FLT_MAX;
        l_w[h] = 0.0f;
#pragma unroll
        for (int e = 0; e < ELEMS; e++)
            o_reg[h][e] = 0.0f;
    }

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        const int phys_block = bt[blk];
        if (phys_block < 0)
            continue;  // StreamingLLM sentinel, same guard as the plain kernels
        const half* K_block = K_cache + (int64_t)phys_block * kv_block_stride + kv_head * HEAD_DIM +
                              lane_offset;
        const half* V_block = V_cache + (int64_t)phys_block * kv_block_stride + kv_head * HEAD_DIM +
                              lane_offset;
        const int tok_start = blk * block_size;
        int n_tok = block_size;
        if (tok_start + n_tok > ctx_len)
            n_tok = ctx_len - tok_start;
        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

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
                float m_new = m_w[h];
#pragma unroll
                for (int i = 0; i < TOK; i++) {
                    dot[h][i] = apply_softcap(dot[h][i] * scale, softcap);
                    if (t + i >= n_tok)
                        dot[h][i] = -FLT_MAX;
                    m_new = fmaxf(m_new, dot[h][i]);
                }
                alpha[h] = expf(m_w[h] - m_new);
                float p_sum = 0.0f;
#pragma unroll
                for (int i = 0; i < TOK; i++) {
                    p[h][i] = (t + i < n_tok) ? expf(dot[h][i] - m_new) : 0.0f;
                    p_sum += p[h][i];
                }
                l_w[h] = alpha[h] * l_w[h] + p_sum;
                m_w[h] = m_new;
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
                    o_reg[h][e] *= alpha[h];
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                float vf[ELEMS];
                lane_vec_to_float<ELEMS>(vp[i], vf);
#pragma unroll
                for (int h = 0; h < HPC; h++)
#pragma unroll
                    for (int e = 0; e < ELEMS; e++)
                        o_reg[h][e] = fmaf(p[h][i], vf[e], o_reg[h][e]);
            }
        }
    }

    // Normalise once per head so the shared merge (per-warp o already divided
    // by l, weight = exp(m - gmax) * l) is unchanged; one merge per head.
    extern __shared__ char smem_f16_mt[];
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        if (l_w[h] > 0.0f) {
            const float inv_l = 1.0f / l_w[h];
#pragma unroll
            for (int e = 0; e < ELEMS; e++)
                o_reg[h][e] *= inv_l;
        }
        __syncthreads();
        crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_f16_mt), m_w[h], l_w[h], o_reg[h],
                                             warp_id, lane_id, lane_offset, O, batch_idx, n_heads, head0 + h,
                                             attn_sinks);
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

}  // namespace

bool paged_attention_decode_f16_multitok_launch(const half* Q, const half* K_cache, const half* V_cache,
                                                half* O, const int* block_tables, const int* context_lens,
                                                int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                                int block_size, float scale, int max_num_blocks,
                                                int sliding_window, float softcap, const half* attn_sinks,
                                                int heads_per_cta, cudaStream_t stream) {
    if (head_dim != 128 && head_dim != 256)
        return false;
    const int n_q_per_kv = n_heads / n_kv_heads;
    if (n_q_per_kv < 1 || n_q_per_kv > 8)
        return false;
    // Heads per CTA: the largest of 4 / 2 / 1 that divides the GQA ratio, or
    // the caller's choice when it divides. HD=256 caps at 2 (register budget).
    int hpc = heads_per_cta;
    if (hpc != 1 && hpc != 2 && hpc != 4)
        hpc = 0;
    if (hpc == 0 || n_q_per_kv % hpc != 0) {
        hpc = (n_q_per_kv % 4 == 0) ? 4 : (n_q_per_kv % 2 == 0) ? 2 : 1;
        if (head_dim == 256 && hpc > 2)
            hpc = 2;
    }
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

}  // namespace imp
