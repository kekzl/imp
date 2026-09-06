// NVFP4 paged decode attention, four tokens per warp iteration, HPC Q heads
// per CTA sharing each dequantised K/V row (2026-09-03).
//
// The multitok kernels (attention_paged_nvfp4_multitok.cu) run one CTA per
// (batch, Q head[, split]); every Q head of a KV group re-reads and
// re-dequantises the same nibbles. Same KV bytes, batch 1 with the split-K
// scratch registered: 24/4 heads HD=256 at 77k 207.8 us (427 GB/s) against
// 62.4 us (1423 GB/s) on the 4/4 MHA shape; 32/8 HD=128 at 32k 86.3 us
// against 30.1 us on 8/8. Here a CTA holds HPC Q heads of one KV head:
// the K row is loaded and converted once, dotted against HPC q vectors, the
// V row converted once and accumulated into HPC outputs; the grid is
// (batch, n_kv_heads x groups[, splits]). Unnormalised (m, l, o) per head
// with one division at the end so the shared merges are unchanged.
#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/pdl_device.cuh"
#include "compute/attention_paged_nvfp4_multitok.cuh"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace imp {

namespace {

using nvfp4_mt::fp4_pair_to_half2;
using nvfp4_mt::load_packed;
using nvfp4_mt::load_q_half2;
using nvfp4_mt::ue4m3_scale_to_float;

template <int HEAD_DIM, int HPC>
struct GqaState {
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

// Tokens [first_tok, n_tok) of one block for this warp, TOK at a time, for
// HPC heads. K/V nibbles and scales are loaded and converted once per token.
template <int HEAD_DIM, int TOK, int HPC>
__device__ __forceinline__ void nvfp4_block_multitok_gqa(
    const uint8_t* __restrict__ K_block, const uint8_t* __restrict__ V_block,
    const uint8_t* __restrict__ K_sc_block, const uint8_t* __restrict__ V_sc_block, int first_tok, int n_tok,
    int kv_slot_stride, int sc_slot_stride, int kv_head_bytes, int sc_groups, int kv_head, int lane_offset,
    int lane_group, const half2 (&q_h2)[HPC][HEAD_DIM / WARP_SIZE / 2], float scale, float softcap,
    GqaState<HEAD_DIM, HPC>& st) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int PACK = ELEMS / 2;
    const uint8_t* K_lane = K_block + kv_head * kv_head_bytes + lane_offset / 2;
    const uint8_t* V_lane = V_block + kv_head * kv_head_bytes + lane_offset / 2;
    const uint8_t* K_sc_lane = K_sc_block + kv_head * sc_groups + lane_group;
    const uint8_t* V_sc_lane = V_sc_block + kv_head * sc_groups + lane_group;
    for (int t = first_tok; t < n_tok; t += TOK) {
        uint32_t kw[TOK];
        uint8_t ks[TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const int ti = (t + i < n_tok) ? (t + i) : (n_tok - 1);  // clamped, masked below
            kw[i] = load_packed<PACK>(K_lane + ti * kv_slot_stride);
            ks[i] = __ldg(K_sc_lane + ti * sc_slot_stride);
        }
        float dot[HPC][TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            // The lane's PACK pairs share one UE4M3 group scale: dot the raw
            // E2M1 values (exact in fp16) and apply the scale once per head.
            const float ksc = ue4m3_scale_to_float(ks[i]);
            half2 kh2[PACK];
#pragma unroll
            for (int b = 0; b < PACK; b++)
                kh2[b] = fp4_pair_to_half2((kw[i] >> (8 * b)) & 0xFF);
#pragma unroll
            for (int h = 0; h < HPC; h++) {
                half2 acc = __hmul2(q_h2[h][0], kh2[0]);
#pragma unroll
                for (int b = 1; b < PACK; b++)
                    acc = __hfma2(q_h2[h][b], kh2[b], acc);
                const float2 pr = __half22float2(acc);
                dot[h][i] = (pr.x + pr.y) * ksc;
            }
        }
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
        uint32_t vw[TOK];
        uint8_t vs[TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const int ti = (t + i < n_tok) ? (t + i) : (n_tok - 1);
            vw[i] = load_packed<PACK>(V_lane + ti * kv_slot_stride);
            vs[i] = __ldg(V_sc_lane + ti * sc_slot_stride);
        }
#pragma unroll
        for (int h = 0; h < HPC; h++)
#pragma unroll
            for (int e = 0; e < ELEMS; e++)
                st.o[h][e] *= alpha[h];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const float vsc = ue4m3_scale_to_float(vs[i]);
            float vf[ELEMS];
#pragma unroll
            for (int b = 0; b < PACK; b++) {
                const float2 f = __half22float2(fp4_pair_to_half2((vw[i] >> (8 * b)) & 0xFF));
                vf[2 * b] = f.x;
                vf[2 * b + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < HPC; h++) {
                const float w = p[h][i] * vsc;
#pragma unroll
                for (int e = 0; e < ELEMS; e++)
                    st.o[h][e] = fmaf(w, vf[e], st.o[h][e]);
            }
        }
    }
}

template <int HEAD_DIM, int HPC>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_decode_nvfp4_multitok_gqa_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales, half* __restrict__ O,
    const int* __restrict__ block_tables, const int* __restrict__ context_lens, int n_heads, int n_kv_heads,
    int n_q_per_kv, int block_size, float scale, int max_num_blocks, int sliding_window, float softcap,
    const half* __restrict__ attn_sinks) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
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
    const int lane_group = lane_offset / 16;

    half2 q_h2[HPC][ELEMS / 2];
#pragma unroll
    for (int h = 0; h < HPC; h++)
        load_q_half2<HEAD_DIM>(Q, batch_idx, head0 + h, n_heads, lane_offset, q_h2[h]);

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int kv_block_stride = block_size * kv_slot_stride;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_slot_stride = n_kv_heads * sc_groups;
    const int sc_block_stride = block_size * sc_slot_stride;

    GqaState<HEAD_DIM, HPC> st;
    st.init();

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        const int phys = bt[blk];
        if (phys < 0)
            continue;
        const int tok_start = blk * block_size;
        int n_tok = block_size;
        if (tok_start + n_tok > ctx_len)
            n_tok = ctx_len - tok_start;
        const int first_tok = (tok_start < effective_start) ? (effective_start - tok_start) : 0;
        nvfp4_block_multitok_gqa<HEAD_DIM, 4, HPC>(K_cache + (int64_t)phys * kv_block_stride,
                                                   V_cache + (int64_t)phys * kv_block_stride,
                                                   K_scales + (int64_t)phys * sc_block_stride,
                                                   V_scales + (int64_t)phys * sc_block_stride, first_tok,
                                                   n_tok, kv_slot_stride, sc_slot_stride, kv_head_bytes,
                                                   sc_groups, kv_head, lane_offset, lane_group, q_h2, scale,
                                                   softcap, st);
    }

    pdl_trigger();  // KV walk done; the dependent o_proj may be scheduled during the reduce + O store
    st.normalise();
    extern __shared__ char smem_nvfp4_gqa[];
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        __syncthreads();
        crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_nvfp4_gqa), st.m[h], st.l[h],
                                             st.o[h], warp_id, lane_id, lane_offset, O, batch_idx, n_heads,
                                             head0 + h, attn_sinks);
    }
}

template <int HEAD_DIM, int HPC>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_splitk_nvfp4_multitok_gqa_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int n_heads, int n_kv_heads, int n_q_per_kv, int block_size,
    float scale, int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
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
    const int lane_group = lane_offset / 16;

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

    half2 q_h2[HPC][ELEMS / 2];
#pragma unroll
    for (int h = 0; h < HPC; h++)
        load_q_half2<HEAD_DIM>(Q, batch_idx, head0 + h, n_heads, lane_offset, q_h2[h]);

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int kv_block_stride = block_size * kv_slot_stride;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_slot_stride = n_kv_heads * sc_groups;
    const int sc_block_stride = block_size * sc_slot_stride;

    GqaState<HEAD_DIM, HPC> st;
    st.init();

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        const int phys = bt[blk];
        if (phys < 0)
            continue;
        const int tok_start = blk * block_size;
        int n_tok = block_size;
        if (tok_start + n_tok > ctx_len)
            n_tok = ctx_len - tok_start;
        const int first_tok = (tok_start < effective_start) ? (effective_start - tok_start) : 0;
        nvfp4_block_multitok_gqa<HEAD_DIM, 4, HPC>(K_cache + (int64_t)phys * kv_block_stride,
                                                   V_cache + (int64_t)phys * kv_block_stride,
                                                   K_scales + (int64_t)phys * sc_block_stride,
                                                   V_scales + (int64_t)phys * sc_block_stride, first_tok,
                                                   n_tok, kv_slot_stride, sc_slot_stride, kv_head_bytes,
                                                   sc_groups, kv_head, lane_offset, lane_group, q_h2, scale,
                                                   softcap, st);
    }

    pdl_trigger();  // KV walk done; the dependent may be scheduled during the reduce + partial store
    st.normalise();
    extern __shared__ char smem_nvfp4_sk_gqa[];
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        __syncthreads();
        crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_nvfp4_sk_gqa), st.m[h], st.l[h],
                                          st.o[h], warp_id, lane_id, lane_offset, partial_out, batch_idx,
                                          n_heads, head0 + h, num_splits, split_idx);
    }
}

template <int HEAD_DIM, int HPC>
void launch_gqa(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache, const uint8_t* K_scales,
                const uint8_t* V_scales, half* O, float* partial, const int* block_tables,
                const int* context_lens, int batch_size, int n_heads, int n_kv_heads, int n_q_per_kv,
                int block_size, float scale, int max_num_blocks, int num_splits, int sliding_window,
                float softcap, const half* attn_sinks, cudaStream_t stream) {
    const size_t smem_bytes = NUM_WARPS * sizeof(float) * 2 + NUM_WARPS * HEAD_DIM * sizeof(float);
    dim3 block(BLOCK_THREADS);
    const int groups = n_kv_heads * (n_q_per_kv / HPC);
    if (num_splits > 1) {
        dim3 grid(batch_size, groups, num_splits);
        paged_attention_splitk_nvfp4_multitok_gqa_kernel<HEAD_DIM, HPC>
            <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, partial,
                                                  block_tables, context_lens, n_heads, n_kv_heads, n_q_per_kv,
                                                  block_size, scale, max_num_blocks, num_splits,
                                                  sliding_window, softcap);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        dim3 grid(batch_size, groups);
        paged_attention_decode_nvfp4_multitok_gqa_kernel<HEAD_DIM, HPC>
            <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, O, block_tables,
                                                  context_lens, n_heads, n_kv_heads, n_q_per_kv, block_size,
                                                  scale, max_num_blocks, sliding_window, softcap, attn_sinks);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace

int paged_attention_nvfp4_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested) {
    if (head_dim != 128 && head_dim != 256)
        return 0;
    if (n_q_per_kv < 2 || n_q_per_kv > 16)
        return 0;
    // The largest of 4 / 3 / 2 dividing the GQA ratio (Qwen3.8 24/4 = 6 -> 3,
    // Qwen3-14B 40/8 = 5 -> 0: no grouping), or the caller's choice when it
    // divides. 1 = the per-head kernels.
    int hpc = requested;
    if (hpc != 2 && hpc != 3 && hpc != 4)
        hpc = 0;
    if (hpc == 0 || n_q_per_kv % hpc != 0) {
        hpc = (n_q_per_kv % 4 == 0) ? 4 : (n_q_per_kv % 3 == 0) ? 3 : (n_q_per_kv % 2 == 0) ? 2 : 0;
    }
    return hpc;
}

bool paged_attention_nvfp4_multitok_gqa_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                               const uint8_t* K_scales, const uint8_t* V_scales, half* O,
                                               float* partial, const int* block_tables,
                                               const int* context_lens, int batch_size, int n_heads,
                                               int n_kv_heads, int head_dim, int block_size, float scale,
                                               int max_num_blocks, int num_splits, int sliding_window,
                                               float softcap, const half* attn_sinks, int heads_per_cta,
                                               cudaStream_t stream) {
    const int n_q_per_kv = (n_kv_heads > 0) ? n_heads / n_kv_heads : 0;
    const int hpc = paged_attention_nvfp4_multitok_heads_per_cta(head_dim, n_q_per_kv, heads_per_cta);
    if (hpc == 0)
        return false;
#define LAUNCH_GQA(HD, HPCV)                                                                              \
    launch_gqa<HD, HPCV>(Q, K_cache, V_cache, K_scales, V_scales, O, partial, block_tables, context_lens, \
                         batch_size, n_heads, n_kv_heads, n_q_per_kv, block_size, scale, max_num_blocks,  \
                         num_splits, sliding_window, softcap, attn_sinks, stream)
    if (head_dim == 256) {
        if (hpc == 4)
            LAUNCH_GQA(256, 4);
        else if (hpc == 3)
            LAUNCH_GQA(256, 3);
        else
            LAUNCH_GQA(256, 2);
    } else {
        if (hpc == 4)
            LAUNCH_GQA(128, 4);
        else if (hpc == 3)
            LAUNCH_GQA(128, 3);
        else
            LAUNCH_GQA(128, 2);
    }
#undef LAUNCH_GQA
    return true;
}

}  // namespace imp
