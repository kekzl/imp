// NVFP4 (E2M1 pairs + UE4M3 group scales) paged decode attention, four tokens
// per warp iteration (attention.paged_nvfp4_multitok, 2026-09-03).
//
// The scalar kernels in attention_paged_nvfp4.cu walk one token per warp
// iteration: one K word, one scale byte, a 5-shuffle reduction, the
// online-softmax update, one V word, all dependent. Measured on the hybrid's
// geometry (24/4 heads, HD=256) with the split-K scratch registered: 1 x 77k
// context 299 us per launch = 296 GB/s (~18% of DRAM), 32 x 1100 123 us. The
// FP8 twin of this structure (attention_paged_fp8_multitok.cu) read 2x. Here
// a warp issues the K words and scale bytes of four tokens before reducing
// any of them, reduces four independent dots, takes one max/rescale per
// group, issues the four V words together; unnormalised (m, l, o) with one
// division at the end so the shared merges (plain and split-K) are unchanged.
// HD=128 (4 elems per lane, 2 packed bytes) and HD=256 (8 elems, one 4-byte
// word) instances, E4M3 scales; other shapes stay on the scalar kernels.
#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstring>

namespace imp {
namespace {

__device__ __forceinline__ float ue4m3_scale_to_float(uint8_t bits) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &bits, 1);
    return static_cast<float>(v);
}

// One packed FP4 byte (low nibble = .x, high nibble = .y) -> half2 via
// cvt.rn.f16x2.e2m1x2 (sm_120, CUDA 13.2+).
__device__ __forceinline__ half2 fp4_pair_to_half2(uint32_t byte_val) {
    uint32_t fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(fp16x2) : "r"(byte_val));
    return *reinterpret_cast<half2*>(&fp16x2);
}

// The lane's PACK packed bytes as one load (PACK = 4: one word, PACK = 2: one
// ushort). __ldg, not .cs: cross-head re-reads are L2 hits worth keeping (#1785).
template <int PACK>
__device__ __forceinline__ uint32_t load_packed(const uint8_t* __restrict__ p) {
    if constexpr (PACK == 4)
        return __ldg(reinterpret_cast<const uint32_t*>(p));
    else
        return static_cast<uint32_t>(__ldg(reinterpret_cast<const unsigned short*>(p)));
}

// Walk the tokens [first_tok, n_tok) of one block for this warp, TOK at a time.
// Updates the unnormalised (m_w, l_w, o_reg).
template <int HEAD_DIM, int TOK>
__device__ __forceinline__ void nvfp4_block_multitok(
    const uint8_t* __restrict__ K_block, const uint8_t* __restrict__ V_block,
    const uint8_t* __restrict__ K_sc_block, const uint8_t* __restrict__ V_sc_block, int first_tok, int n_tok,
    int kv_slot_stride, int sc_slot_stride, int kv_head_bytes, int sc_groups, int kv_head, int lane_offset,
    int lane_group, const half2* q_h2, float scale, float softcap, float& m_w, float& l_w, float* o_reg) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int PACK = ELEMS / 2;  // packed bytes per lane per token
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
        float dot[TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const half2 ksc = __float2half2_rn(ue4m3_scale_to_float(ks[i]));
            float d = 0.0f;
#pragma unroll
            for (int b = 0; b < PACK; b++) {
                half2 kh2 = __hmul2(fp4_pair_to_half2((kw[i] >> (8 * b)) & 0xFF), ksc);
                const float2 pr = __half22float2(__hmul2(q_h2[b], kh2));
                d += pr.x + pr.y;
            }
            dot[i] = d;
        }
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
#pragma unroll
            for (int i = 0; i < TOK; i++)
                dot[i] += __shfl_xor_sync(0xffffffffu, dot[i], off);
        }
        float m_new = m_w;
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            dot[i] = apply_softcap(dot[i] * scale, softcap);
            if (t + i >= n_tok)
                dot[i] = -FLT_MAX;
            m_new = fmaxf(m_new, dot[i]);
        }
        const float alpha = expf(m_w - m_new);
        float p[TOK];
        float p_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            p[i] = (t + i < n_tok) ? expf(dot[i] - m_new) : 0.0f;
            p_sum += p[i];
        }
        l_w = alpha * l_w + p_sum;
        m_w = m_new;
        uint32_t vw[TOK];
        uint8_t vs[TOK];
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const int ti = (t + i < n_tok) ? (t + i) : (n_tok - 1);
            vw[i] = load_packed<PACK>(V_lane + ti * kv_slot_stride);
            vs[i] = __ldg(V_sc_lane + ti * sc_slot_stride);
        }
#pragma unroll
        for (int e = 0; e < ELEMS; e++)
            o_reg[e] *= alpha;
#pragma unroll
        for (int i = 0; i < TOK; i++) {
            const float w = p[i] * ue4m3_scale_to_float(vs[i]);
#pragma unroll
            for (int b = 0; b < PACK; b++) {
                const float2 vf = __half22float2(fp4_pair_to_half2((vw[i] >> (8 * b)) & 0xFF));
                o_reg[2 * b] += w * vf.x;
                o_reg[2 * b + 1] += w * vf.y;
            }
        }
    }
}

template <int HEAD_DIM>
__device__ __forceinline__ void load_q_half2(const half* __restrict__ Q, int batch_idx, int head_idx,
                                             int n_heads, int lane_offset, half2* q_h2) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q + (int64_t)batch_idx * n_heads * HEAD_DIM +
                                                         (int64_t)head_idx * HEAD_DIM + lane_offset);
#pragma unroll
    for (int i = 0; i < ELEMS / 2; i++)
        q_h2[i] = Q_ptr2[i];
}

template <int HEAD_DIM>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_decode_nvfp4_multitok_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales, half* __restrict__ O,
    const int* __restrict__ block_tables, const int* __restrict__ context_lens, int n_heads, int n_kv_heads,
    int block_size, float scale, int max_num_blocks, int sliding_window, float softcap,
    const half* __restrict__ attn_sinks) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head = head_idx / (n_heads / n_kv_heads);
    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;
    const int lane_group = lane_offset / 16;
    half2 q_h2[ELEMS / 2];
    load_q_half2<HEAD_DIM>(Q, batch_idx, head_idx, n_heads, lane_offset, q_h2);

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int kv_block_stride = block_size * kv_slot_stride;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_slot_stride = n_kv_heads * sc_groups;
    const int sc_block_stride = block_size * sc_slot_stride;

    float m_w = -FLT_MAX, l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;
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
        nvfp4_block_multitok<HEAD_DIM, 4>(K_cache + (int64_t)phys * kv_block_stride,
                                          V_cache + (int64_t)phys * kv_block_stride,
                                          K_scales + (int64_t)phys * sc_block_stride,
                                          V_scales + (int64_t)phys * sc_block_stride, first_tok, n_tok,
                                          kv_slot_stride, sc_slot_stride, kv_head_bytes, sc_groups, kv_head,
                                          lane_offset, lane_group, q_h2, scale, softcap, m_w, l_w, o_reg);
    }
    if (l_w > 0.0f) {
        const float inv_l = 1.0f / l_w;
#pragma unroll
        for (int e = 0; e < ELEMS; e++)
            o_reg[e] *= inv_l;
    }
    extern __shared__ char smem_nvfp4_mt[];
    __syncthreads();
    crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_nvfp4_mt), m_w, l_w, o_reg, warp_id,
                                         lane_id, lane_offset, O, batch_idx, n_heads, head_idx, attn_sinks);
}

template <int HEAD_DIM>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_splitk_nvfp4_multitok_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int n_heads, int n_kv_heads, int block_size, float scale,
    int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head = head_idx / (n_heads / n_kv_heads);
    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;
    const int lane_group = lane_offset / 16;
    half2 q_h2[ELEMS / 2];
    load_q_half2<HEAD_DIM>(Q, batch_idx, head_idx, n_heads, lane_offset, q_h2);

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
        write_empty_split_sentinel<HEAD_DIM>(partial_out, batch_idx, n_heads, head_idx, num_splits, split_idx,
                                             lane_offset);
        return;
    }
    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int kv_block_stride = block_size * kv_slot_stride;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_slot_stride = n_kv_heads * sc_groups;
    const int sc_block_stride = block_size * sc_slot_stride;

    float m_w = -FLT_MAX, l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        const int phys = bt[blk];
        if (phys < 0)
            continue;
        const int tok_start = blk * block_size;
        int n_tok = block_size;
        if (tok_start + n_tok > ctx_len)
            n_tok = ctx_len - tok_start;
        const int first_tok = (tok_start < effective_start) ? (effective_start - tok_start) : 0;
        nvfp4_block_multitok<HEAD_DIM, 4>(K_cache + (int64_t)phys * kv_block_stride,
                                          V_cache + (int64_t)phys * kv_block_stride,
                                          K_scales + (int64_t)phys * sc_block_stride,
                                          V_scales + (int64_t)phys * sc_block_stride, first_tok, n_tok,
                                          kv_slot_stride, sc_slot_stride, kv_head_bytes, sc_groups, kv_head,
                                          lane_offset, lane_group, q_h2, scale, softcap, m_w, l_w, o_reg);
    }
    if (l_w > 0.0f) {
        const float inv_l = 1.0f / l_w;
#pragma unroll
        for (int e = 0; e < ELEMS; e++)
            o_reg[e] *= inv_l;
    }
    extern __shared__ char smem_sk_nvfp4_mt[];
    __syncthreads();
    crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_sk_nvfp4_mt), m_w, l_w, o_reg, warp_id,
                                      lane_id, lane_offset, partial_out, batch_idx, n_heads, head_idx,
                                      num_splits, split_idx);
}

}  // namespace

// Launch for head_dim 128 / 256 (E4M3 scales). Returns false for other shapes
// so the caller falls back to the scalar kernels. num_splits > 1 launches the
// split-K kernel into `partial` (the caller runs the reduce).
bool paged_attention_decode_nvfp4_multitok_launch(const half* Q, const uint8_t* K_cache,
                                                  const uint8_t* V_cache, const uint8_t* K_scales,
                                                  const uint8_t* V_scales, half* O, float* partial,
                                                  const int* block_tables, const int* context_lens,
                                                  int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                                  int block_size, float scale, int max_num_blocks,
                                                  int num_splits, int sliding_window, float softcap,
                                                  const half* attn_sinks, cudaStream_t stream) {
    if (head_dim != 128 && head_dim != 256)
        return false;
    const size_t smem_bytes = NUM_WARPS * sizeof(float) * 2 + NUM_WARPS * head_dim * sizeof(float);
    dim3 block(BLOCK_THREADS);
    if (num_splits > 1) {
        dim3 grid(batch_size, n_heads, num_splits);
        if (head_dim == 256)
            paged_attention_splitk_nvfp4_multitok_kernel<256>
                <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, partial,
                                                      block_tables, context_lens, n_heads, n_kv_heads,
                                                      block_size, scale, max_num_blocks, num_splits,
                                                      sliding_window, softcap);
        else
            paged_attention_splitk_nvfp4_multitok_kernel<128>
                <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, partial,
                                                      block_tables, context_lens, n_heads, n_kv_heads,
                                                      block_size, scale, max_num_blocks, num_splits,
                                                      sliding_window, softcap);
    } else {
        dim3 grid(batch_size, n_heads);
        if (head_dim == 256)
            paged_attention_decode_nvfp4_multitok_kernel<256>
                <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, O,
                                                      block_tables, context_lens, n_heads, n_kv_heads,
                                                      block_size, scale, max_num_blocks, sliding_window,
                                                      softcap, attn_sinks);
        else
            paged_attention_decode_nvfp4_multitok_kernel<128>
                <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, O,
                                                      block_tables, context_lens, n_heads, n_kv_heads,
                                                      block_size, scale, max_num_blocks, sliding_window,
                                                      softcap, attn_sinks);
    }
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
