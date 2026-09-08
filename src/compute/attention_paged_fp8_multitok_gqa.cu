// FP8 E4M3 paged decode attention, HD=128, with LPR lanes per KV row, TOK
// rows per lane group per iteration and HPC Q heads per CTA
// (attention.paged_fp8_multitok with Q-head grouping, 2026-09-08).
//
// The four-token kernel (attention_paged_fp8_multitok.cu) spreads one
// 128-byte K row over the 32 lanes of a warp: 4 bytes per lane, a 5-shuffle
// reduction per head and token, and each of the ratio Q heads of a KV group
// is a CTA of its own that re-reads the group through L2 and re-converts
// every byte. ncu at 32 x 1100 after the paired cvt (2026-09-03): issue
// 65%, DRAM ~35%, ~36 warps/SM, i.e. instruction-bound. Here LPR lanes hold
// one row (128 / LPR bytes per lane, one 16- or 8-byte load), a warp holds
// 32 / LPR rows per load instruction, a dot reduces over log2(LPR) shuffles,
// and a CTA carries HPC Q heads of one KV head: a row is loaded and
// converted once and dotted against HPC q slices read from shared memory
// (kept out of registers so two CTAs fit an SM). The warp walks its blocks
// through a cursor and keeps the K and V rows of the NEXT row group in
// flight while it reduces the current one, across block boundaries; the
// block table entry is read one block ahead. Each lane group runs its own
// online softmax over the rows it saw; the groups of a warp merge over
// shuffles at the end, then the shared cross-warp merge runs per head after
// a shared-memory relayout to its 4-dims-per-lane form. Grid
// (batch, n_kv_heads x n_q_per_kv / HPC); sliding window via the effective
// start, the StreamingLLM sentinel, softcap and learned sinks as in the
// four-token kernel.
//
// Shipped instance <LPR 16, TOK 2, HPC 1..5> (microbench 32 x 1100, 40/8,
// 2026-09-08, ptxas registers in brackets): the four-token kernel 95.3 us;
// 8 lanes x 16 bytes, TOK 2: 54.8 [217, one CTA per SM]; 16 lanes, TOK 4:
// 63.9 [148]; 16 lanes, TOK 2: 54.7 [128, two CTAs per SM]. 32 x 4096:
// 334.6 -> 186.2 us (1442 GB/s). Reading q from registers instead of shared
// memory cost 17 registers and the second CTA (66.4 us); issuing V after the
// softmax instead of with K read 63.6.
#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/pdl_device.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>

namespace imp {
namespace {

constexpr int HD = 128;

template <int BYTES>
struct LaneRow;
template <>
struct LaneRow<16> {
    using Vec = uint4;
    static __device__ __forceinline__ void words(const uint4& v, uint32_t (&w)[4]) {
        w[0] = v.x;
        w[1] = v.y;
        w[2] = v.z;
        w[3] = v.w;
    }
};
template <>
struct LaneRow<8> {
    using Vec = uint2;
    static __device__ __forceinline__ void words(const uint2& v, uint32_t (&w)[2]) {
        w[0] = v.x;
        w[1] = v.y;
    }
};

// Packed e4m3 words -> half2 pairs, two cvt per word (the paired form of
// the four-token kernel).
template <int NW>
__device__ __forceinline__ void fp8_words_to_half2(const uint32_t (&w)[NW], half2 (&out)[2 * NW]) {
#pragma unroll
    for (int i = 0; i < NW; i++) {
        const __half2_raw r0 =
            __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] & 0xFFFFu), __NV_E4M3);
        const __half2_raw r1 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] >> 16), __NV_E4M3);
        out[2 * i] = half2(r0);
        out[2 * i + 1] = half2(r1);
    }
}

// The lane's q slice, re-read from shared memory per iteration: the volatile
// asm keeps ptxas from hoisting it into HPC x NH2 registers (20 at 16 lanes
// x 5 heads, 40 at 8), which is what buys the second CTA per SM.
template <int NH2>
__device__ __forceinline__ void lds_q_slice(uint32_t saddr, half2 (&q)[NH2]) {
    static_assert(NH2 == 4 || NH2 == 8, "one or two 16-byte shared loads");
    uint32_t w[NH2];
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "r"(saddr));
    if constexpr (NH2 == 8)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
                     : "r"(saddr + 16));
#pragma unroll
    for (int i = 0; i < NH2; i++)
        q[i] = *reinterpret_cast<const half2*>(&w[i]);
}

// The lane's slice of q . k: fp16 products, pairs accumulated in fp16, the
// pair sums added in fp32 (the four-token kernel multiplies in fp16 and
// adds every product in fp32; the oracle envelope covers both).
template <int NH2>
__device__ __forceinline__ float dot_slice(const half2 (&q)[NH2], const half2 (&k)[NH2]) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NH2; i += 2) {
        const float2 f = __half22float2(__hfma2(q[i + 1], k[i + 1], __hmul2(q[i], k[i])));
        s += f.x + f.y;
    }
    return s;
}

// The warp's position in its share of the KV blocks: block, row group start
// t, valid rows of the block, the lane's K and V row-0 pointers, and the
// physical block of the warp's NEXT block (read one block ahead).
struct Cursor {
    int blk;
    int t;
    int n_tok;
    int next_phys;
    const uint8_t* K_block;
    const uint8_t* V_block;
};

struct WalkGeom {
    const int* bt;
    const uint8_t* K_cache;
    const uint8_t* V_cache;
    int num_ctx_blocks;
    int ctx_len;
    int block_size;
    int effective_start;
    int64_t kv_block_stride;
    int kv_head_off;  // kv_head * HD + dim0
};

__device__ __forceinline__ bool cursor_enter_block(Cursor& c, const WalkGeom& g, int phys) {
    c.K_block = g.K_cache + (int64_t)phys * g.kv_block_stride + g.kv_head_off;
    c.V_block = g.V_cache + (int64_t)phys * g.kv_block_stride + g.kv_head_off;
    const int tok_start = c.blk * g.block_size;
    c.n_tok = (tok_start + g.block_size > g.ctx_len) ? (g.ctx_len - tok_start) : g.block_size;
    c.t = (tok_start < g.effective_start) ? (g.effective_start - tok_start) : 0;
    return c.t < c.n_tok;
}

// First block of the warp with rows to read (sentinel and empty blocks
// skipped); false when there is none.
__device__ __forceinline__ bool cursor_init(Cursor& c, const WalkGeom& g, int first_blk) {
    c.blk = first_blk;
    while (c.blk < g.num_ctx_blocks) {
        const int phys = g.bt[c.blk];
        const int nb = c.blk + NUM_WARPS;
        c.next_phys = (nb < g.num_ctx_blocks) ? g.bt[nb] : -1;
        if (phys >= 0 && cursor_enter_block(c, g, phys))
            return true;
        c.blk = nb;
    }
    return false;
}

// Next row group: within the block, else the warp's next block with rows.
template <int ROWS>
__device__ __forceinline__ bool cursor_advance(Cursor& c, const WalkGeom& g) {
    c.t += ROWS;
    if (c.t < c.n_tok)
        return true;
    while (true) {
        c.blk += NUM_WARPS;
        if (c.blk >= g.num_ctx_blocks)
            return false;
        const int phys = c.next_phys;
        const int nb = c.blk + NUM_WARPS;
        c.next_phys = (nb < g.num_ctx_blocks) ? g.bt[nb] : -1;
        if (phys >= 0 && cursor_enter_block(c, g, phys))
            return true;
    }
}

// The min-blocks term pins the two CTAs per SM the HPC=5 instance sits on
// at exactly 128 registers; measured neutral against the bare bound (55.1 vs
// 54.7 us at HPC 5, 56.0 vs 56.2 at HPC 4, 2026-09-08), no spill.
template <int LPR, int TOK, int HPC>
__global__ void __launch_bounds__(BLOCK_THREADS, 2) paged_attention_decode_fp8_gqa_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    half* __restrict__ O, const int* __restrict__ block_tables, const int* __restrict__ context_lens,
    int n_heads, int n_kv_heads, int n_q_per_kv, int block_size, float scale, float kv_scale,
    int max_num_blocks, int sliding_window, float softcap, const half* __restrict__ attn_sinks) {
    static_assert(LPR == 8 || LPR == 16, "8 lanes x 16 bytes or 16 lanes x 8 bytes per 128-byte row");
    constexpr int BYTES = HD / LPR;       // e4m3 bytes = dims per lane
    constexpr int NW = BYTES / 4;         // 32-bit words per lane
    constexpr int NH2 = BYTES / 2;        // half2 per lane
    constexpr int RPW = WARP_SIZE / LPR;  // rows per warp per load instruction
    constexpr int ROWS = TOK * RPW;       // rows per warp iteration
    using Vec = typename LaneRow<BYTES>::Vec;

    const int batch_idx = blockIdx.x;
    const int groups_per_kv = n_q_per_kv / HPC;
    const int kv_head = blockIdx.y / groups_per_kv;
    const int head0 = kv_head * n_q_per_kv + (blockIdx.y % groups_per_kv) * HPC;
    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row_slot = lane_id / LPR;         // which of the RPW rows of a load this lane holds
    const int dim0 = (lane_id % LPR) * BYTES;  // the lane's first dim of the row

    // Shared memory: the cross-warp merge's region, the relayout rows, then
    // the CTA's HPC q heads (contiguous in Q) for the per-iteration slice reads.
    extern __shared__ char smem_fp8_gqa[];
    float* merge_smem = reinterpret_cast<float*>(smem_fp8_gqa);
    float* relayout = merge_smem + 2 * NUM_WARPS + NUM_WARPS * HD;  // NUM_WARPS x HD
    half2* q_smem = reinterpret_cast<half2*>(relayout + NUM_WARPS * HD);  // HPC x HD / 2
    {
        const half2* q_src = reinterpret_cast<const half2*>(Q + ((int64_t)batch_idx * n_heads + head0) * HD);
        for (int i = threadIdx.x; i < HPC * HD / 2; i += BLOCK_THREADS)
            q_smem[i] = q_src[i];
    }
    __syncthreads();
    const uint32_t q_saddr = static_cast<uint32_t>(__cvta_generic_to_shared(q_smem)) + dim0 * sizeof(half);

    WalkGeom g;
    g.bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    g.K_cache = K_cache;
    g.V_cache = V_cache;
    g.num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    g.ctx_len = ctx_len;
    g.block_size = block_size;
    g.effective_start = (sliding_window > 0 && ctx_len > sliding_window) ? (ctx_len - sliding_window) : 0;
    g.kv_block_stride = (int64_t)block_size * n_kv_heads * HD;
    g.kv_head_off = kv_head * HD + dim0;
    const int kv_slot_stride = n_kv_heads * HD;
    const float fused_scale = scale * kv_scale;

    float m[HPC], l[HPC], o[HPC][BYTES];
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        m[h] = -FLT_MAX;
        l[h] = 0.0f;
#pragma unroll
        for (int e = 0; e < BYTES; e++)
            o[h][e] = 0.0f;
    }

    // The lane's K and V rows of a row group: row t + j * RPW + row_slot,
    // clamped to a valid row past n_tok (masked to p = 0 in the reduce).
    auto load_group = [&](const Cursor& c, Vec (&kq)[TOK], Vec (&vq)[TOK]) {
#pragma unroll
        for (int j = 0; j < TOK; j++) {
            const int r = c.t + j * RPW + row_slot;
            const int rc = (r < c.n_tok) ? r : (c.n_tok - 1);
            kq[j] = __ldcs(reinterpret_cast<const Vec*>(c.K_block + rc * kv_slot_stride));
            vq[j] = __ldcs(reinterpret_cast<const Vec*>(c.V_block + rc * kv_slot_stride));
        }
    };

    Cursor cur;
    bool valid = cursor_init(cur, g, g.effective_start / block_size + warp_id);
    Vec kq[TOK], vq[TOK];
    if (valid)
        load_group(cur, kq, vq);
    while (valid) {
        // The next group's rows go in flight before this group is reduced
        // (warp-uniform control flow: the cursor is per warp).
        Cursor nxt = cur;
        const bool nvalid = cursor_advance<ROWS>(nxt, g);
        Vec kq_n[TOK], vq_n[TOK];
        if (nvalid)
            load_group(nxt, kq_n, vq_n);

        half2 kh[TOK][NH2];
#pragma unroll
        for (int j = 0; j < TOK; j++) {
            uint32_t w[NW];
            LaneRow<BYTES>::words(kq[j], w);
            fp8_words_to_half2<NW>(w, kh[j]);
        }
        float s[HPC][TOK];
#pragma unroll
        for (int h = 0; h < HPC; h++) {
            half2 qh[NH2];
            lds_q_slice<NH2>(q_saddr + h * HD * sizeof(half), qh);
#pragma unroll
            for (int j = 0; j < TOK; j++)
                s[h][j] = dot_slice<NH2>(qh, kh[j]);
        }
        // Reduce within the LPR-lane row group; HPC x TOK chains interleave.
#pragma unroll
        for (int off = LPR / 2; off > 0; off >>= 1) {
#pragma unroll
            for (int h = 0; h < HPC; h++)
#pragma unroll
                for (int j = 0; j < TOK; j++)
                    s[h][j] += __shfl_xor_sync(0xffffffffu, s[h][j], off);
        }
        float p[HPC][TOK];
        float alpha[HPC];
#pragma unroll
        for (int h = 0; h < HPC; h++) {
            float m_new = m[h];
#pragma unroll
            for (int j = 0; j < TOK; j++) {
                const bool row_ok = (cur.t + j * RPW + row_slot) < cur.n_tok;
                s[h][j] = row_ok ? apply_softcap(s[h][j] * fused_scale, softcap) : -FLT_MAX;
                m_new = fmaxf(m_new, s[h][j]);
            }
            alpha[h] = expf(m[h] - m_new);
            float p_sum = 0.0f;
#pragma unroll
            for (int j = 0; j < TOK; j++) {
                const bool row_ok = (cur.t + j * RPW + row_slot) < cur.n_tok;
                p[h][j] = row_ok ? expf(s[h][j] - m_new) : 0.0f;
                p_sum += p[h][j];
            }
            l[h] = alpha[h] * l[h] + p_sum;
            m[h] = m_new;
        }
        // P . V; masked rows carry p = 0.
#pragma unroll
        for (int h = 0; h < HPC; h++)
#pragma unroll
            for (int e = 0; e < BYTES; e++)
                o[h][e] *= alpha[h];
#pragma unroll
        for (int j = 0; j < TOK; j++) {
            uint32_t w[NW];
            LaneRow<BYTES>::words(vq[j], w);
            half2 vh[NH2];
            fp8_words_to_half2<NW>(w, vh);
            float vf[BYTES];
#pragma unroll
            for (int i = 0; i < NH2; i++) {
                const float2 f = __half22float2(vh[i]);
                vf[2 * i] = f.x;
                vf[2 * i + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < HPC; h++) {
                const float wgt = p[h][j] * kv_scale;
#pragma unroll
                for (int e = 0; e < BYTES; e++)
                    o[h][e] = fmaf(wgt, vf[e], o[h][e]);
            }
        }

        cur = nxt;
        valid = nvalid;
#pragma unroll
        for (int j = 0; j < TOK; j++) {
            kq[j] = kq_n[j];
            vq[j] = vq_n[j];
        }
    }

    // Merge the RPW row groups of the warp (xor LPR, 2 LPR, ...): every lane
    // of the warp then holds the warp's (m, l, o) for its dims.
#pragma unroll
    for (int h = 0; h < HPC; h++) {
#pragma unroll
        for (int off = LPR; off < WARP_SIZE; off <<= 1) {
            const float m_o = __shfl_xor_sync(0xffffffffu, m[h], off);
            const float l_o = __shfl_xor_sync(0xffffffffu, l[h], off);
            const float m_new = fmaxf(m[h], m_o);
            const float a = expf(m[h] - m_new);
            const float b = expf(m_o - m_new);
            l[h] = a * l[h] + b * l_o;
#pragma unroll
            for (int e = 0; e < BYTES; e++)
                o[h][e] = a * o[h][e] + b * __shfl_xor_sync(0xffffffffu, o[h][e], off);
            m[h] = m_new;
        }
        // Normalise once so the shared merge (per-warp o already divided by
        // l, weight = exp(m - gmax) * l) is unchanged.
        if (l[h] > 0.0f) {
            const float inv_l = 1.0f / l[h];
#pragma unroll
            for (int e = 0; e < BYTES; e++)
                o[h][e] *= inv_l;
        }
    }

    pdl_trigger();  // KV walk done; the dependent o_proj may be scheduled during the reduce + O store
    constexpr int ELEMS = HD / WARP_SIZE;  // the merge's 4 dims per lane
    const int lane_offset = lane_id * ELEMS;
#pragma unroll
    for (int h = 0; h < HPC; h++) {
        __syncwarp();
        if (row_slot == 0) {
#pragma unroll
            for (int e = 0; e < BYTES; e++)
                relayout[warp_id * HD + dim0 + e] = o[h][e];
        }
        __syncwarp();
        float o4[ELEMS];
#pragma unroll
        for (int i = 0; i < ELEMS; i++)
            o4[i] = relayout[warp_id * HD + lane_offset + i];
        __syncthreads();
        crosswarp_reduce_and_write<HD>(merge_smem, m[h], l[h], o4, warp_id, lane_id, lane_offset, O, batch_idx,
                                       n_heads, head0 + h, attn_sinks);
    }
}

template <int LPR, int TOK, int HPC>
void launch_gqa(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache, half* O, const int* block_tables,
                const int* context_lens, int batch_size, int n_heads, int n_kv_heads, int n_q_per_kv,
                int block_size, float scale, float kv_scale, int max_num_blocks, int sliding_window,
                float softcap, const half* attn_sinks, cudaStream_t stream) {
    const size_t smem_bytes = (2 * NUM_WARPS + 2 * NUM_WARPS * HD) * sizeof(float) + HPC * HD * sizeof(half);
    dim3 grid(batch_size, n_kv_heads * (n_q_per_kv / HPC));
    dim3 block(BLOCK_THREADS);
    paged_attention_decode_fp8_gqa_kernel<LPR, TOK, HPC>
        <<<grid, block, smem_bytes, stream>>>(Q, K_cache, V_cache, O, block_tables, context_lens, n_heads,
                                              n_kv_heads, n_q_per_kv, block_size, scale, kv_scale,
                                              max_num_blocks, sliding_window, softcap, attn_sinks);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace

int paged_attention_fp8_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested) {
    if (head_dim != 128)
        return 0;
    if (n_q_per_kv < 1 || n_q_per_kv > 16)
        return 0;
    // The largest of 5 / 4 / 3 / 2 / 1 dividing the GQA ratio (Qwen3-14B
    // 40/8 = 5, Qwen3-8B and Llama 32/8 = 4, Qwen3-32B 64/8 -> 4), or the
    // caller's choice when it divides.
    int hpc = requested;
    if (hpc < 1 || hpc > 5)
        hpc = 0;
    if (hpc == 0 || n_q_per_kv % hpc != 0)
        hpc = (n_q_per_kv % 5 == 0) ? 5
              : (n_q_per_kv % 4 == 0) ? 4
              : (n_q_per_kv % 3 == 0) ? 3
              : (n_q_per_kv % 2 == 0) ? 2
                                      : 1;
    return hpc;
}

bool paged_attention_fp8_multitok_gqa_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                             half* O, const int* block_tables, const int* context_lens,
                                             int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                             int block_size, float scale, float kv_scale, int max_num_blocks,
                                             int sliding_window, float softcap, const half* attn_sinks,
                                             int heads_per_cta, cudaStream_t stream) {
    const int n_q_per_kv = (n_kv_heads > 0) ? n_heads / n_kv_heads : 0;
    const int hpc = paged_attention_fp8_multitok_heads_per_cta(head_dim, n_q_per_kv, heads_per_cta);
    if (hpc == 0)
        return false;
#define LAUNCH_FP8_GQA(HPCV)                                                                            \
    launch_gqa<16, 2, HPCV>(Q, K_cache, V_cache, O, block_tables, context_lens, batch_size, n_heads,     \
                            n_kv_heads, n_q_per_kv, block_size, scale, kv_scale, max_num_blocks,         \
                            sliding_window, softcap, attn_sinks, stream)
    switch (hpc) {
        case 5:
            LAUNCH_FP8_GQA(5);
            break;
        case 4:
            LAUNCH_FP8_GQA(4);
            break;
        case 3:
            LAUNCH_FP8_GQA(3);
            break;
        case 2:
            LAUNCH_FP8_GQA(2);
            break;
        default:
            LAUNCH_FP8_GQA(1);
            break;
    }
#undef LAUNCH_FP8_GQA
    return true;
}

}  // namespace imp
