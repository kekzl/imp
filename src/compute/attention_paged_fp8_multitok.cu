// FP8 E4M3 paged decode attention, TOK tokens per warp iteration
// (attention.paged_fp8_multitok, 2026-09-03).
//
// The plain kernel (attention_paged_fp8.cu) walks one token per warp
// iteration: one 128-byte K row, a 5-shuffle reduction, the online-softmax
// update, one 128-byte V row, and the next token waits on all of it. On a
// dense 40-layer model at 32 streams x ~1.1k context that kernel was 33% of
// the serving kernel time at ~25% of DRAM bandwidth (Qwen3-14B-NVFP4,
// 2026-09-03). Here a warp issues the K rows of TOK tokens before reducing
// any of them, reduces TOK independent dots, takes one max / rescale per
// group, and issues the TOK V rows together, so TOK x 2 loads are in flight
// per warp instead of two. The softmax state is the unnormalised (m, l, o)
// form; o is divided by l once at the end so the shared cross-warp merge
// (normalised o, weight = exp(m - gmax) * l) is unchanged. HD=128 only;
// every other shape stays on the plain kernel.
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

__device__ __forceinline__ float fp8_bits_to_float(uint8_t bits) {
    __nv_fp8_e4m3 val;
    memcpy(&val, &bits, 1);
    return static_cast<float>(val);
}

// Four packed e4m3 bytes -> two half2 with two cvt instructions (e4m3x2 ->
// f16x2) instead of four byte extractions and four scalar conversions. ncu on
// the byte-wise form at 32 x 1100: SM throughput 70%, DRAM 33%, stall
// not_selected 2.25, i.e. issue-bound rather than memory-bound.
__device__ __forceinline__ void fp8x4_to_half2x2(uint32_t packed, half2& lo, half2& hi) {
    __half2_raw r0 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xFFFFu),
                                                __NV_E4M3);
    __half2_raw r1 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
    lo = half2(r0);
    hi = half2(r1);
}

template <int HEAD_DIM, int TOK>
__global__ void __launch_bounds__(BLOCK_THREADS) paged_attention_decode_fp8_multitok_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    half* __restrict__ O, const int* __restrict__ block_tables, const int* __restrict__ context_lens,
    int n_heads, int n_kv_heads, int block_size, float scale, float kv_scale, int max_num_blocks,
    int sliding_window, float softcap, const half* __restrict__ attn_sinks) {
    static_assert(HEAD_DIM == 128, "multitok kernel is the HD=128 instance (4 FP8 bytes per lane)");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;  // 4
    static_assert(ELEMS == 4, "one uint32 per lane");

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head = head_idx / (n_heads / n_kv_heads);
    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    half2 q_h2[ELEMS / 2];  // the lane's four q elements as two half2
    {
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q + (int64_t)batch_idx * n_heads * HEAD_DIM +
                                                             (int64_t)head_idx * HEAD_DIM + lane_offset);
#pragma unroll
        for (int i = 0; i < ELEMS / 2; i++)
            q_h2[i] = Q_ptr2[i];
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;
    const float fused_scale = scale * kv_scale;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
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
        const int phys_block = bt[blk];
        if (phys_block < 0)
            continue;  // StreamingLLM sentinel, same guard as the plain kernel
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride + kv_head * HEAD_DIM +
                                 lane_offset;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride + kv_head * HEAD_DIM +
                                 lane_offset;
        const int tok_start = blk * block_size;
        int n_tok = block_size;
        if (tok_start + n_tok > ctx_len)
            n_tok = ctx_len - tok_start;
        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

        for (int t = first_tok; t < n_tok; t += TOK) {
            // K rows of TOK tokens in flight before any reduction. Out-of-range
            // tokens of a partial group load a valid row (clamped) and are
            // masked to p = 0 below.
            uint32_t kp[TOK];
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                const int ti = (t + i < n_tok) ? (t + i) : (n_tok - 1);
                kp[i] = __ldcs(reinterpret_cast<const uint32_t*>(K_block + ti * kv_slot_stride));
            }
            float dot[TOK];
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                half2 k_lo, k_hi;
                fp8x4_to_half2x2(kp[i], k_lo, k_hi);
                const float2 p0 = __half22float2(__hmul2(q_h2[0], k_lo));
                const float2 p1 = __half22float2(__hmul2(q_h2[1], k_hi));
                dot[i] = (p0.x + p0.y) + (p1.x + p1.y);
            }
            // TOK independent warp reductions (the shuffle chains interleave).
#pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
#pragma unroll
                for (int i = 0; i < TOK; i++)
                    dot[i] += __shfl_xor_sync(0xffffffffu, dot[i], off);
            }
            float m_new = m_w;
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                dot[i] = apply_softcap(dot[i] * fused_scale, softcap);
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
            // V rows of the group in flight together; the invalid ones carry p = 0.
            uint32_t vp[TOK];
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                const int ti = (t + i < n_tok) ? (t + i) : (n_tok - 1);
                vp[i] = __ldcs(reinterpret_cast<const uint32_t*>(V_block + ti * kv_slot_stride));
            }
#pragma unroll
            for (int e = 0; e < ELEMS; e++)
                o_reg[e] *= alpha;
#pragma unroll
            for (int i = 0; i < TOK; i++) {
                const float w = p[i] * kv_scale;
                half2 v_lo, v_hi;
                fp8x4_to_half2x2(vp[i], v_lo, v_hi);
                const float2 f0 = __half22float2(v_lo);
                const float2 f1 = __half22float2(v_hi);
                o_reg[0] += w * f0.x;
                o_reg[1] += w * f0.y;
                o_reg[2] += w * f1.x;
                o_reg[3] += w * f1.y;
            }
        }
    }

    // Normalise once so the shared merge (which expects per-warp o already
    // divided by l, weight = exp(m - gmax) * l) is unchanged.
    if (l_w > 0.0f) {
        const float inv_l = 1.0f / l_w;
#pragma unroll
        for (int e = 0; e < ELEMS; e++)
            o_reg[e] *= inv_l;
    }
    extern __shared__ char smem_fp8_mt[];
    __syncthreads();
    crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_fp8_mt), m_w, l_w, o_reg, warp_id,
                                         lane_id, lane_offset, O, batch_idx, n_heads, head_idx, attn_sinks);
}

}  // namespace

void paged_attention_decode_fp8_multitok_hd128(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                               half* O, const int* block_tables, const int* context_lens,
                                               int batch_size, int n_heads, int n_kv_heads, int block_size,
                                               float scale, float kv_scale, int max_num_blocks,
                                               int sliding_window, float softcap, const half* attn_sinks,
                                               cudaStream_t stream) {
    const size_t smem_bytes = NUM_WARPS * sizeof(float) * 2 + NUM_WARPS * 128 * sizeof(float);
    dim3 grid(batch_size, n_heads);
    dim3 block(BLOCK_THREADS);
    // TOK=4. An 8-token instance measured 94.2 vs 91.7 us at 32 x 1100 and
    // 327.9 vs 328.8 at 4096 (2026-09-03): the extra registers cost what the
    // extra loads in flight bought.
    paged_attention_decode_fp8_multitok_kernel<128, 4>
        <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, O, block_tables, context_lens, n_heads,
                                              n_kv_heads, block_size, scale, kv_scale, max_num_blocks,
                                              sliding_window, softcap, attn_sinks);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
