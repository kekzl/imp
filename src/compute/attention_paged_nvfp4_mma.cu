// NVFP4 paged split-K decode on m16n8k16 tensor cores: one warp = one 16-token paged block x all
// (<= 8) Q heads of one KV head. S^T = Q K^T (A = Q, rows 8..15 zero), O^T += V^T P^T (P reuses the
// S^T accumulator layout). Head dims are permuted per lane to match the packed FP4 bytes; the
// UE4M3 group scale is folded into the converted K/V fragments (E2M1 x UE4M3 is exact in fp16).
#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/pdl_device.cuh"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>

namespace imp {

namespace {

constexpr int kMmaWarps = 4;
constexpr int kMmaThreads = kMmaWarps * WARP_SIZE;
constexpr int kMmaHeads = 8;   // n of m16n8k16
constexpr int kMmaBlock = 16;  // tokens per paged block = one k16 PV step

// Four packed FP4 bytes -> four half2 (low nibble = .x).
__device__ __forceinline__ void e2m1x8_to_h2(uint32_t w, uint32_t (&h)[4]) {
    asm("{ .reg .b8 b0, b1, b2, b3;\n"
        "  mov.b32 {b0, b1, b2, b3}, %4;\n"
        "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %3, b3; }"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w));
}

// Two UE4M3 bytes (low 16 bits of `pair`) -> half2.
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t pair) {
    uint32_t h;
    asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(h) : "r"(pair));
    return h;
}

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
    half2 r = __hmul2(*reinterpret_cast<half2*>(&a), *reinterpret_cast<half2*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    half2 r = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int N>
__device__ __forceinline__ void ldg_words(const uint8_t* p, uint32_t (&w)[N]) {
    static_assert(N % 2 == 0, "8-byte granules");
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int i = 0; i < N / 4; i++) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(p) + i);
            w[4 * i] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < N / 2; i++) {
            const uint2 v = __ldg(reinterpret_cast<const uint2*>(p) + i);
            w[2 * i] = v.x;
            w[2 * i + 1] = v.y;
        }
    }
}

// Lane (g = lane / 4, t = lane % 4) layout, HD = head dim, RB = HD / 2 packed bytes per row:
//   QK k-step s pairs byte t*RB/4 + 2s (a0/b0) and + 1 (a2/b1): lane t walks a contiguous RB/4.
//   PV m-tile mt: row g = dim 2j, row g + 8 = dim 2j + 1, j = g * HD/16 + mt: lane g walks
//   HD/16 contiguous bytes of tokens 2t, 2t+1, 2t+8, 2t+9.
template <int HD>
struct MmaWarpState {
    static constexpr int KS = HD / 16;  // QK k-steps = PV m-tiles
    float o[KS][4];
    float m;  // running max of head g
    float l;  // this lane's share of head g's denominator
};

template <int HD>
__device__ __forceinline__ void mma_block(
    const uint8_t* __restrict__ K_blk, const uint8_t* __restrict__ V_blk, const uint8_t* __restrict__ Ks_blk,
    const uint8_t* __restrict__ Vs_blk, int first_tok, int n_tok, int slot_stride, int sc_slot_stride,
    const uint32_t (&qa)[HD / 16][2], float scale, float softcap, MmaWarpState<HD>& st, int g, int t) {
    constexpr int KS = HD / 16;
    constexpr int RB = HD / 2;
    constexpr int KW = RB / 16;  // words of one K row per lane (RB/4 bytes)
    constexpr int VW = KS / 4;   // words of one V row per lane (HD/16 bytes)
    auto clamp_tok = [&](int r) { return r < first_tok ? first_tok : (r >= n_tok ? n_tok - 1 : r); };

    // K rows g and g + 8, V rows 2t, 2t+1, 2t+8, 2t+9 (clamped: masked rows still read finite bytes).
    uint32_t kw[2][KW], vw[4][VW];
    uint32_t ksc[2], vsc[4];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = clamp_tok(g + 8 * r);
        ldg_words<KW>(K_blk + static_cast<int64_t>(row) * slot_stride + static_cast<ptrdiff_t>(t) * (RB / 4),
                      kw[r]);
        const uint8_t* sp = Ks_blk + static_cast<int64_t>(row) * sc_slot_stride +
                            static_cast<ptrdiff_t>(t) * (KS / 4);
        ksc[r] = (KS / 4 == 4) ? __ldg(reinterpret_cast<const uint32_t*>(sp))
                               : __ldg(reinterpret_cast<const unsigned short*>(sp));
    }
    constexpr int kVRow[4] = {0, 1, 8, 9};
#pragma unroll
    for (int r = 0; r < 4; r++) {
        const int row = clamp_tok(2 * t + kVRow[r]);
        ldg_words<VW>(V_blk + static_cast<int64_t>(row) * slot_stride + static_cast<ptrdiff_t>(g) * (HD / 16),
                      vw[r]);
        const uint8_t* sp = Vs_blk + static_cast<int64_t>(row) * sc_slot_stride +
                            static_cast<ptrdiff_t>(g) * (KS / 8);
        vsc[r] = (KS / 8 == 2) ? __ldg(reinterpret_cast<const unsigned short*>(sp)) : __ldg(sp);
    }

    // S^T[head g][token]: n-tile 0 = tokens 0..7, 1 = 8..15.
    float s[2][4] = {};
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
#pragma unroll
        for (int w = 0; w < KW; w++) {
            uint32_t kh[4];
            e2m1x8_to_h2(kw[nt][w], kh);
            // word w = k-steps 2w, 2w + 1; scale group of k-step s = s / 4
#pragma unroll
            for (int half_w = 0; half_w < 2; half_w++) {
                const int ks = 2 * w + half_w;
                const int k0 = 2 * half_w;
                const uint32_t sel = static_cast<uint32_t>(ks / 4);
                const uint32_t sc = e4m3x2_to_h2(__byte_perm(ksc[nt], 0, sel | (sel << 4)));
                mma_16816(s[nt], qa[ks][0], 0u, qa[ks][1], 0u, hmul2_u32(kh[k0], sc),
                          hmul2_u32(kh[k0 + 1], sc));
            }
        }
    }

    // Online softmax for head g over this lane's tokens 2t, 2t+1, 8+2t, 8+2t+1.
    float x[4] = {s[0][0], s[0][1], s[1][0], s[1][1]};
    float mx = st.m;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int tok = (i >> 1) * 8 + 2 * t + (i & 1);
        x[i] = apply_softcap(x[i] * scale, softcap);
        if (tok < first_tok || tok >= n_tok)
            x[i] = -FLT_MAX;
        mx = fmaxf(mx, x[i]);
    }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float alpha = expf(st.m - mx);
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int tok = (i >> 1) * 8 + 2 * t + (i & 1);
        p[i] = (tok < first_tok || tok >= n_tok) ? 0.0f : expf(x[i] - mx);
    }
    st.l = alpha * st.l + ((p[0] + p[1]) + (p[2] + p[3]));
    st.m = mx;
    // O^T columns of this lane are heads 2t, 2t + 1; alpha lives on lanes 4 * head.
    if (!__all_sync(0xffffffffu, alpha == 1.0f)) {
        const float a0 = __shfl_sync(0xffffffffu, alpha, 8 * t);
        const float a1 = __shfl_sync(0xffffffffu, alpha, 8 * t + 4);
#pragma unroll
        for (int mt = 0; mt < KS; mt++) {
            st.o[mt][0] *= a0;
            st.o[mt][1] *= a1;
            st.o[mt][2] *= a0;
            st.o[mt][3] *= a1;
        }
    }
    const uint32_t pb0 = pack_h2(p[0], p[1]);
    const uint32_t pb1 = pack_h2(p[2], p[3]);

    // V scale pairs (token 2t, 2t+1) and (2t+8, 2t+9) per scale group of this lane's bytes.
    constexpr int VG = KS / 8;  // groups per lane row: HD/16 bytes / 8
    uint32_t vs01[VG], vs89[VG];
#pragma unroll
    for (int gi = 0; gi < VG; gi++) {
        vs01[gi] = e4m3x2_to_h2(__byte_perm(vsc[0], vsc[1], gi | ((4 + gi) << 4)));
        vs89[gi] = e4m3x2_to_h2(__byte_perm(vsc[2], vsc[3], gi | ((4 + gi) << 4)));
    }
#pragma unroll
    for (int w = 0; w < VW; w++) {
        uint32_t vh[4][4];
#pragma unroll
        for (int r = 0; r < 4; r++)
            e2m1x8_to_h2(vw[r][w], vh[r]);
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const int mt = 4 * w + b;
            const uint32_t sa = vs01[mt / 8], sb = vs89[mt / 8];
            // lows = dim 2j, highs = dim 2j + 1
            const uint32_t a0 = hmul2_u32(__byte_perm(vh[0][b], vh[1][b], 0x5410), sa);
            const uint32_t a1 = hmul2_u32(__byte_perm(vh[0][b], vh[1][b], 0x7632), sa);
            const uint32_t a2 = hmul2_u32(__byte_perm(vh[2][b], vh[3][b], 0x5410), sb);
            const uint32_t a3 = hmul2_u32(__byte_perm(vh[2][b], vh[3][b], 0x7632), sb);
            mma_16816(st.o[mt], a0, a1, a2, a3, pb0, pb1);
        }
    }
}

template <int HD>
__global__ void __launch_bounds__(kMmaThreads) paged_attention_splitk_nvfp4_mma_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int n_heads, int n_kv_heads, float scale, int max_num_blocks,
    int num_splits, int sliding_window, float softcap) {
    constexpr int KS = HD / 16;
    const int batch_idx = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int n_q = n_heads / n_kv_heads;
    const int head0 = kv_head * n_q;
    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int g = lane >> 2, t = lane & 3;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / kMmaBlock;
    const int num_ctx_blocks = (ctx_len + kMmaBlock - 1) / kMmaBlock;
    const int blocks_per_split = (num_ctx_blocks - first_block + num_splits - 1) / num_splits;
    const int split_start = first_block + split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_ctx_blocks);
    constexpr int partial_stride = 2 + HD;
    if (split_start >= split_end) {
        for (int i = threadIdx.x; i < n_q * partial_stride; i += kMmaThreads) {
            const int h = i / partial_stride, e = i % partial_stride;
            float* out = partial_out +
                         static_cast<int64_t>((batch_idx * n_heads + head0 + h) * num_splits + split_idx) *
                             partial_stride;
            out[e] = (e == 0) ? -FLT_MAX : 0.0f;
        }
        return;
    }

    // Q A-fragments of head g: k-step s = dims t*HD/4 + 4s .. + 3 (a0 = first pair, a2 = second).
    uint32_t qa[KS][2];
    {
        const bool live = g < n_q;
        const uint4* qp = reinterpret_cast<const uint4*>(
            Q + (static_cast<int64_t>(batch_idx) * n_heads + head0 + (live ? g : 0)) * HD +
            static_cast<ptrdiff_t>(t) * (HD / 4));
#pragma unroll
        for (int i = 0; i < KS / 2; i++) {
            const uint4 v = live ? __ldg(qp + i) : make_uint4(0, 0, 0, 0);
            qa[2 * i][0] = v.x;
            qa[2 * i][1] = v.y;
            qa[2 * i + 1][0] = v.z;
            qa[2 * i + 1][1] = v.w;
        }
    }

    const int slot_stride = n_kv_heads * (HD / 2);
    const int sc_slot_stride = n_kv_heads * (HD / 16);
    const int64_t kv_block_stride = static_cast<int64_t>(kMmaBlock) * slot_stride;
    const int64_t sc_block_stride = static_cast<int64_t>(kMmaBlock) * sc_slot_stride;
    const int* bt = block_tables + static_cast<int64_t>(batch_idx) * max_num_blocks;

    MmaWarpState<HD> st;
    st.m = -FLT_MAX;
    st.l = 0.0f;
#pragma unroll
    for (int mt = 0; mt < KS; mt++)
#pragma unroll
        for (int i = 0; i < 4; i++)
            st.o[mt][i] = 0.0f;

    for (int blk = split_start + warp_id; blk < split_end; blk += kMmaWarps) {
        const int phys = bt[blk];
        if (phys < 0)
            continue;
        const int tok_start = blk * kMmaBlock;
        const int n_tok = min(kMmaBlock, ctx_len - tok_start);
        const int first_tok = (tok_start < effective_start) ? (effective_start - tok_start) : 0;
        mma_block<HD>(K_cache + phys * kv_block_stride + static_cast<ptrdiff_t>(kv_head) * (HD / 2),
                      V_cache + phys * kv_block_stride + static_cast<ptrdiff_t>(kv_head) * (HD / 2),
                      K_scales + phys * sc_block_stride + static_cast<ptrdiff_t>(kv_head) * (HD / 16),
                      V_scales + phys * sc_block_stride + static_cast<ptrdiff_t>(kv_head) * (HD / 16),
                      first_tok, n_tok, slot_stride, sc_slot_stride, qa, scale, softcap, st, g, t);
    }
    pdl_trigger();

    // Cross-warp merge: per-warp (m, l) per head and O^T to smem, then a fixed-order sum.
    extern __shared__ float smem_mma[];
    float* s_m = smem_mma;                                             // [warps][8]
    float* s_l = s_m + static_cast<ptrdiff_t>(kMmaWarps) * kMmaHeads;  // [warps][8]
    float* s_o = s_l + static_cast<ptrdiff_t>(kMmaWarps) * kMmaHeads;  // [warps][8][HD]
    float l = st.l;
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (t == 0) {
        s_m[warp_id * kMmaHeads + g] = st.m;
        s_l[warp_id * kMmaHeads + g] = l;
    }
    float* wo = s_o + static_cast<ptrdiff_t>(warp_id) * kMmaHeads * HD;
#pragma unroll
    for (int mt = 0; mt < KS; mt++) {
        const int d = 2 * (g * KS + mt);
        wo[(2 * t) * HD + d] = st.o[mt][0];
        wo[(2 * t + 1) * HD + d] = st.o[mt][1];
        wo[(2 * t) * HD + d + 1] = st.o[mt][2];
        wo[(2 * t + 1) * HD + d + 1] = st.o[mt][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_q * HD; i += kMmaThreads) {
        const int h = i / HD, d = i % HD;
        float gm = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < kMmaWarps; w++)
            gm = fmaxf(gm, s_m[w * kMmaHeads + h]);
        float gl = 0.0f, o = 0.0f;
#pragma unroll
        for (int w = 0; w < kMmaWarps; w++) {
            const float f = expf(s_m[w * kMmaHeads + h] - gm);
            gl += f * s_l[w * kMmaHeads + h];
            o += f * s_o[(w * kMmaHeads + h) * HD + d];
        }
        float* out = partial_out +
                     static_cast<int64_t>((batch_idx * n_heads + head0 + h) * num_splits + split_idx) *
                         partial_stride;
        if (d == 0) {
            out[0] = gm;
            out[1] = gl;
        }
        out[2 + d] = o;
    }
}

}  // namespace

bool paged_attention_nvfp4_mma_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                      const uint8_t* K_scales, const uint8_t* V_scales, float* partial,
                                      const int* block_tables, const int* context_lens, int batch_size,
                                      int n_heads, int n_kv_heads, int head_dim, int block_size, float scale,
                                      int max_num_blocks, int num_splits, int sliding_window, float softcap,
                                      cudaStream_t stream) {
    if (partial == nullptr || num_splits < 2 || block_size != kMmaBlock || n_kv_heads <= 0)
        return false;
    if (head_dim != 128 && head_dim != 256)
        return false;
    const int n_q = n_heads / n_kv_heads;
    if (n_q < 2 || n_q > kMmaHeads || n_heads % n_kv_heads != 0)
        return false;
    const size_t smem = (static_cast<size_t>(2 * kMmaWarps * kMmaHeads) +
                         static_cast<size_t>(kMmaWarps) * kMmaHeads * head_dim) *
                        sizeof(float);
    const dim3 grid(batch_size, n_kv_heads, num_splits);
    if (head_dim == 256) {
        static const bool attr = [] {
            return cudaFuncSetAttribute(paged_attention_splitk_nvfp4_mma_kernel<256>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (2 * kMmaWarps * kMmaHeads + kMmaWarps * kMmaHeads * 256) *
                                            static_cast<int>(sizeof(float))) == cudaSuccess;
        }();
        if (!attr)
            return false;
        paged_attention_splitk_nvfp4_mma_kernel<256>
            <<<grid, kMmaThreads, smem, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, partial,
                                                  block_tables, context_lens, n_heads, n_kv_heads, scale,
                                                  max_num_blocks, num_splits, sliding_window, softcap);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        paged_attention_splitk_nvfp4_mma_kernel<128>
            <<<grid, kMmaThreads, smem, stream>>>(Q, K_cache, V_cache, K_scales, V_scales, partial,
                                                  block_tables, context_lens, n_heads, n_kv_heads, scale,
                                                  max_num_blocks, num_splits, sliding_window, softcap);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}

}  // namespace imp
