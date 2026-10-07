// Chunked SSD form of the Mamba2 prefill scan (gdn.ssd_scan): 64-token chunks as fp16
// tensor-core GEMMs with fp32 accumulate; only the 128 x 32 state slice crosses chunks.
// Not bit-identical to ssm_scan_reg (no per-token state rounding); grid (heads, head_dim / 32).
#include "compute/gdn_scan_chunkpar.cuh"
#include "compute/ssm_scan_ssd.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {
namespace {

using chunkpar::mma_f16_16x8x16;
using chunkpar::split_f16x2;

constexpr int kL = 64;         // tokens per chunk
constexpr int kN = 128;        // state size
constexpr int kDW = 16;        // head_dim columns per CTA
constexpr int kJ = kDW / 8;    // n8 tiles per CTA column slice
constexpr int kThreads = 128;  // 4 warps: 16 chunk rows / 32 state rows each
constexpr int kBS = kN + 8;    // B/C tile row stride (halves), conflict-free fragment loads
constexpr int kXS = kL + 8;    // x tiles [d][t]
constexpr int kHS = kN + 8;    // state tiles [d][n]

struct Smem {
    half c[kL * kBS];
    half b[kL * kBS];
    half xt[kDW * kXS];    // raw x, transposed
    half h_hi[kDW * kHS];  // chunk-start state * 2^-eh, hi/lo split
    half h_lo[kDW * kHS];
    float cs[kL];   // inclusive cumsum of dt * A within the chunk
    float dtv[kL];  // softplus(dt + bias), 0 past real_n
    float w[kL];    // exp(cs_last - cs) * dtv * 2^-ew
    float hmax[kThreads / 32];
    float cl;
    int ew;
};

__device__ __forceinline__ uint32_t ld32(const half* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ uint32_t pack2(half lo, half hi) {
    return static_cast<uint32_t>(__half_as_ushort(lo)) | (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

__device__ __forceinline__ float pow2(int e) {
    e = max(-126, min(127, e));
    return __int_as_float((e + 127) << 23);
}

// e with |m| * 2^-e < 2^15: fp16 hi/lo operands stay finite, power-of-2 scaling is exact.
__device__ __forceinline__ int fp16_exp(float m) {
    if (!(m > 0.0f) || !isfinite(m))
        return 0;
    int ex;
    frexpf(m, &ex);
    return ex - 15;
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_incl_scan(float v, int lane) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o)
            v += u;
    }
    return v;
}

// State fragment (mt, j, e) of warp w: n = 32w + 16mt + g + 8*(e>=2), d = 8j + 2q + (e&1).
__device__ __forceinline__ void publish_state(Smem& s, const float (&hacc)[2][kJ][4], int warp, int g, int q,
                                              int eh) {
    const float sc = pow2(-eh);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < kJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; e += 2) {
                const int n = 32 * warp + 16 * mt + g + (e ? 8 : 0);
                const int d = 8 * j + 2 * q;
                uint32_t hi, lo;
                split_f16x2(hacc[mt][j][e] * sc, hacc[mt][j][e + 1] * sc, hi, lo);
                s.h_hi[d * kHS + n] = __ushort_as_half(static_cast<unsigned short>(hi & 0xffffu));
                s.h_hi[(d + 1) * kHS + n] = __ushort_as_half(static_cast<unsigned short>(hi >> 16));
                s.h_lo[d * kHS + n] = __ushort_as_half(static_cast<unsigned short>(lo & 0xffffu));
                s.h_lo[(d + 1) * kHS + n] = __ushort_as_half(static_cast<unsigned short>(lo >> 16));
            }
}

__device__ __forceinline__ float state_absmax(const float (&hacc)[2][kJ][4]) {
    float m = 0.0f;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < kJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                m = fmaxf(m, fabsf(hacc[mt][j][e]));
    return warp_max(m);
}

// Thread coordinates: warp w owns chunk rows [16w, 16w + 16) and state rows [32w, 32w + 32).
struct Lane {
    int warp, lane, g, q;
};

template <bool H_FP16>
__device__ __forceinline__ void state_io(void* h_state, int64_t h_base, int head_dim, const Lane& l,
                                         float (&hacc)[2][kJ][4], bool store) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < kJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = (32 * l.warp) + (16 * mt) + l.g + (e >= 2 ? 8 : 0);
                const int dc = (8 * j) + (2 * l.q) + (e & 1);
                const int64_t idx = h_base + (static_cast<int64_t>(n) * head_dim) + dc;
                if constexpr (H_FP16) {
                    half* p = static_cast<half*>(h_state);
                    if (store)
                        p[idx] = __float2half(hacc[mt][j][e]);
                    else
                        hacc[mt][j][e] = __half2float(p[idx]);
                } else {
                    float* p = static_cast<float*>(h_state);
                    if (store)
                        p[idx] = hacc[mt][j][e];
                    else
                        hacc[mt][j][e] = p[idx];
                }
            }
}

// Block-wide state absmax -> scale exponent, then the hi/lo state tiles for the next chunk.
// The barrier inside also retires every read of the previous chunk's tiles.
__device__ __forceinline__ int sync_publish_state(Smem& s, const float (&hacc)[2][kJ][4], const Lane& l) {
    const float m = state_absmax(hacc);
    if (l.lane == 0)
        s.hmax[l.warp] = m;
    __syncthreads();
    const int eh = fp16_exp(fmaxf(fmaxf(s.hmax[0], s.hmax[1]), fmaxf(s.hmax[2], s.hmax[3])));
    publish_state(s, hacc, l.warp, l.g, l.q, eh);
    return eh;
}

constexpr int kBcPer = kL * (kN / 8) / kThreads;  // uint4 of B (and of C) per thread and chunk
constexpr int kXPer = kL * kDW / kThreads;        // x halves per thread and chunk

// Global operands of one CTA.
struct Src {
    const half* __restrict__ B;
    const half* __restrict__ C;
    const half* __restrict__ x;
    const half* __restrict__ dt;
    int64_t bc, inner, col0;
    int grp, n_tokens, real_n, n_heads, h;
};

// Next chunk's operands, held in registers while the current chunk computes.
struct Staged {
    uint4 b[kBcPer], c[kBcPer];
    half x[kXPer];
    half dt[2];  // warp 0 only
};

// Loads chunk c0 into registers; rows past n_tokens (dt: past real_n) read as zero.
__device__ __forceinline__ void fetch(Staged& p, const Src& g, int c0, const Lane& l, int tid) {
#pragma unroll
    for (int k = 0; k < kBcPer; ++k) {
        const int i = tid + (k * kThreads);
        const int r = i >> 4, v = i & 15, t = c0 + r;
        p.b[k] = make_uint4(0, 0, 0, 0);
        p.c[k] = p.b[k];
        if (t < g.n_tokens) {
            const int col = (g.grp * kN) + (8 * v);
            const int64_t off = (static_cast<int64_t>(t) * g.bc) + col;
            p.b[k] = __ldg(reinterpret_cast<const uint4*>(g.B + off));
            p.c[k] = __ldg(reinterpret_cast<const uint4*>(g.C + off));
        }
    }
#pragma unroll
    for (int k = 0; k < kXPer; ++k) {
        const int i = tid + (k * kThreads);
        const int r = i / kDW, d = i % kDW, t = c0 + r;
        p.x[k] = t < g.n_tokens ? g.x[(static_cast<int64_t>(t) * g.inner) + g.col0 + d] : __float2half(0.0f);
    }
    if (l.warp == 0) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int t = c0 + l.lane + (32 * k);
            p.dt[k] = t < g.real_n ? g.dt[(static_cast<int64_t>(t) * g.n_heads) + g.h] : __float2half(0.0f);
        }
    }
}

// Staged registers -> B/C tiles and the transposed x tile [d][t].
__device__ __forceinline__ void store(Smem& s, const Staged& p, int tid) {
#pragma unroll
    for (int k = 0; k < kBcPer; ++k) {
        const int i = tid + (k * kThreads);
        const int r = i >> 4, v = i & 15;
        *reinterpret_cast<uint4*>(&s.b[(r * kBS) + (8 * v)]) = p.b[k];
        *reinterpret_cast<uint4*>(&s.c[(r * kBS) + (8 * v)]) = p.c[k];
    }
#pragma unroll
    for (int k = 0; k < kXPer; ++k) {
        const int i = tid + (k * kThreads);
        s.xt[((i % kDW) * kXS) + (i / kDW)] = p.x[k];
    }
}

// Warp 0: dt -> softplus, cumsum of dt * A, state-update weights. dt = 0 past real_n.
__device__ __forceinline__ void chunk_decay(Smem& s, const Staged& p, int c0, int real_n, float dt_b,
                                            float a_log_h, int lane) {
    float v[2], lg[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        float dv = 0.0f;
        if (c0 + lane + (32 * k) < real_n) {
            dv = __half2float(p.dt[k]) + dt_b;
            dv = (dv > 20.0f) ? dv : logf(1.0f + expf(dv));
        }
        v[k] = dv;
        lg[k] = dv * a_log_h;
    }
    const float cs0 = warp_incl_scan(lg[0], lane);
    const float cs1 = warp_incl_scan(lg[1], lane) + __shfl_sync(0xffffffffu, cs0, 31);
    const float cl = __shfl_sync(0xffffffffu, cs1, 31);
    const float w0 = expf(cl - cs0) * v[0], w1 = expf(cl - cs1) * v[1];
    const int ew = fp16_exp(warp_max(fmaxf(w0, w1))) + 15;  // w * 2^-ew <= 1
    const float wsc = pow2(-ew);
    s.cs[lane] = cs0;
    s.cs[lane + 32] = cs1;
    s.dtv[lane] = v[0];
    s.dtv[lane + 32] = v[1];
    s.w[lane] = w0 * wsc;
    s.w[lane + 32] = w1 * wsc;
    if (lane == 0) {
        s.cl = cl;
        s.ew = ew;
    }
}

// G = C B^T (causal tiles only) and Y_inter = C H on the same C fragments.
__device__ __forceinline__ void gemm_g_inter(const Smem& s, const Lane& l, float (&acc)[8][4],
                                             float (&yh)[kJ][4]) {
    const int r0 = 16 * l.warp;
#pragma unroll
    for (int kk = 0; kk < kN / 16; ++kk) {
        const int k = (16 * kk) + (2 * l.q);
        uint32_t a[4];
        a[0] = ld32(&s.c[((r0 + l.g) * kBS) + k]);
        a[1] = ld32(&s.c[((r0 + l.g + 8) * kBS) + k]);
        a[2] = ld32(&s.c[((r0 + l.g) * kBS) + k + 8]);
        a[3] = ld32(&s.c[((r0 + l.g + 8) * kBS) + k + 8]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j <= (2 * l.warp) + 1) {
                uint32_t b[2];
                b[0] = ld32(&s.b[(((8 * j) + l.g) * kBS) + k]);
                b[1] = ld32(&s.b[(((8 * j) + l.g) * kBS) + k + 8]);
                mma_f16_16x8x16(acc[j], a, b);
            }
        }
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int row = ((8 * j) + l.g) * kHS;
            const uint32_t bh[2] = {ld32(&s.h_hi[row + k]), ld32(&s.h_hi[row + k + 8])};
            const uint32_t bl[2] = {ld32(&s.h_lo[row + k]), ld32(&s.h_lo[row + k + 8])};
            mma_f16_16x8x16(yh[j], a, bh);
            mma_f16_16x8x16(yh[j], a, bl);
        }
    }
}

// M = G * exp(cs_t - cs_s) * dt_s on s <= t, each row scaled by 2^-e into fp16 range.
__device__ __forceinline__ void mask_rows(const Smem& s, const Lane& l, float (&acc)[8][4], int& ea,
                                          int& eb) {
    const int ta = (16 * l.warp) + l.g, tb = ta + 8;
    const float cs_a = s.cs[ta], cs_b = s.cs[tb];
    float ma = 0.0f, mb = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int sc = (8 * j) + (2 * l.q) + (e & 1);
            const bool upper = e >= 2;
            const int row = upper ? tb : ta;
            const float m = sc <= row ? acc[j][e] * expf((upper ? cs_b : cs_a) - s.cs[sc]) * s.dtv[sc] : 0.0f;
            acc[j][e] = m;
            if (upper)
                mb = fmaxf(mb, fabsf(m));
            else
                ma = fmaxf(ma, fabsf(m));
        }
    ea = fp16_exp(quad_max(ma));
    eb = fp16_exp(quad_max(mb));
    const float sa = pow2(-ea), sb = pow2(-eb);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j][0] *= sa;
        acc[j][1] *= sa;
        acc[j][2] *= sb;
        acc[j][3] *= sb;
    }
}

// Y_intra = M X: the masked G accumulators are the A fragments (hi/lo), x raw fp16.
__device__ __forceinline__ void gemm_intra(const Smem& s, const Lane& l, const float (&acc)[8][4],
                                           float (&yi)[kJ][4]) {
#pragma unroll
    for (int kk = 0; kk < kL / 16; ++kk) {
        if (kk > l.warp)
            continue;
        uint32_t ahi[4], alo[4];
        const int k0 = 2 * kk;
        split_f16x2(acc[k0][0], acc[k0][1], ahi[0], alo[0]);
        split_f16x2(acc[k0][2], acc[k0][3], ahi[1], alo[1]);
        split_f16x2(acc[k0 + 1][0], acc[k0 + 1][1], ahi[2], alo[2]);
        split_f16x2(acc[k0 + 1][2], acc[k0 + 1][3], ahi[3], alo[3]);
        const int k = (16 * kk) + (2 * l.q);
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int row = ((8 * j) + l.g) * kXS;
            const uint32_t b[2] = {ld32(&s.xt[row + k]), ld32(&s.xt[row + k + 8])};
            mma_f16_16x8x16(yi[j], ahi, b);
            mma_f16_16x8x16(yi[j], alo, b);
        }
    }
}

// y = Y_intra * 2^e + exp(cs_t) 2^eh Y_inter + D x, gated, for rows < n_tokens.
template <bool FUSE_GATE>
__device__ __forceinline__ void write_y(const Smem& s, const Lane& l, const float (&yi)[kJ][4],
                                        const float (&yh)[kJ][4], int ea, int eb, int eh, float d_val,
                                        half* __restrict__ y, const half* __restrict__ z, int c0,
                                        int n_tokens, int64_t inner, int64_t col0) {
    const int ta = (16 * l.warp) + l.g;
    const float hs = pow2(eh);
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
        const int row = ta + (e ? 8 : 0);
        const int t = c0 + row;
        if (t >= n_tokens)
            continue;
        const float u = pow2(e ? eb : ea), v = expf(s.cs[row]) * hs;
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int d = (8 * j) + (2 * l.q);
            float y0 = (yi[j][e] * u) + (yh[j][e] * v) + (d_val * __half2float(s.xt[(d * kXS) + row]));
            float y1 = (yi[j][e + 1] * u) + (yh[j][e + 1] * v) +
                       (d_val * __half2float(s.xt[((d + 1) * kXS) + row]));
            const int64_t off = (static_cast<int64_t>(t) * inner) + col0 + d;
            if constexpr (FUSE_GATE) {
                const float2 zv = __half22float2(*reinterpret_cast<const half2*>(z + off));
                y0 *= zv.x / (1.0f + expf(-zv.x));
                y1 *= zv.y / (1.0f + expf(-zv.y));
            }
            *reinterpret_cast<half2*>(y + off) = __floats2half2_rn(y0, y1);
        }
    }
}

__device__ __forceinline__ void scale_state(float (&hacc)[2][kJ][4], float f) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < kJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                hacc[mt][j][e] *= f;
}

// B fragment pair (t = k, k + 1) of w * x as fp16 hi/lo, from the raw x tile.
__device__ __forceinline__ void wx_frag(const Smem& s, int row, int k, uint32_t& hi, uint32_t& lo) {
    const float2 xv = __half22float2(*reinterpret_cast<const half2*>(&s.xt[row + k]));
    split_f16x2(xv.x * s.w[k], xv.y * s.w[k + 1], hi, lo);
}

// H = exp(cs_last) H + B^T (w x); w carries 2^-ew, undone around the mma.
__device__ __forceinline__ void update_state(const Smem& s, const Lane& l, float (&hacc)[2][kJ][4]) {
    const int ew = s.ew;
    scale_state(hacc, expf(s.cl) * pow2(-ew));
#pragma unroll
    for (int kk = 0; kk < kL / 16; ++kk) {
        const int k = (16 * kk) + (2 * l.q);
        uint32_t a[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int n = (32 * l.warp) + (16 * mt) + l.g;
            a[mt][0] = pack2(s.b[(k * kBS) + n], s.b[((k + 1) * kBS) + n]);
            a[mt][1] = pack2(s.b[(k * kBS) + n + 8], s.b[((k + 1) * kBS) + n + 8]);
            a[mt][2] = pack2(s.b[((k + 8) * kBS) + n], s.b[((k + 9) * kBS) + n]);
            a[mt][3] = pack2(s.b[((k + 8) * kBS) + n + 8], s.b[((k + 9) * kBS) + n + 8]);
        }
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int row = ((8 * j) + l.g) * kXS;
            uint32_t bh[2], bl[2];
            wx_frag(s, row, k, bh[0], bl[0]);
            wx_frag(s, row, k + 8, bh[1], bl[1]);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                mma_f16_16x8x16(hacc[mt][j], a[mt], bh);
                mma_f16_16x8x16(hacc[mt][j], a[mt], bl);
            }
        }
    }
    scale_state(hacc, pow2(ew));
}

template <int R, int C>
__device__ __forceinline__ void zero(float (&a)[R][C]) {
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
        for (int k = 0; k < C; ++k)
            a[i][k] = 0.0f;
}

template <bool H_FP16, bool FUSE_GATE>
__global__ void __launch_bounds__(kThreads) ssm_scan_ssd_kernel(
    const half* __restrict__ x, const half* __restrict__ B_in, const half* __restrict__ C_in,
    const half* __restrict__ dt_raw, const float* __restrict__ A_log, const float* __restrict__ D_skip,
    const float* __restrict__ dt_bias, void* __restrict__ h_state, half* __restrict__ y,
    const half* __restrict__ z, int n_tokens, int n_heads, int head_dim, int n_groups,
    const int* __restrict__ d_real_n) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& s = *reinterpret_cast<Smem*>(smem_raw);
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const Lane l{tid >> 5, tid & 31, (tid & 31) >> 2, tid & 3};
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    const int grp = h / (n_heads / n_groups);
    const float a_log_h = A_log[h], d_val = D_skip[h], dt_b = dt_bias[h];
    const int64_t inner = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t bc = static_cast<int64_t>(n_groups) * kN;
    const int dcol = static_cast<int>(blockIdx.y) * kDW;
    const int64_t col0 = (static_cast<int64_t>(h) * head_dim) + dcol;  // x/y/z column of d = 0
    const int64_t h_base = (static_cast<int64_t>(h) * kN * head_dim) + dcol;

    const Src src{B_in, C_in, x, dt_raw, bc, inner, col0, grp, n_tokens, real_n, n_heads, h};
    Staged p;
    fetch(p, src, 0, l, tid);

    float hacc[2][kJ][4];
    state_io<H_FP16>(h_state, h_base, head_dim, l, hacc, false);
    int eh = sync_publish_state(s, hacc, l);

    for (int c0 = 0; c0 < n_tokens; c0 += kL) {
        store(s, p, tid);
        if (l.warp == 0)
            chunk_decay(s, p, c0, real_n, dt_b, a_log_h, l.lane);
        __syncthreads();
        if (c0 + kL < n_tokens)
            fetch(p, src, c0 + kL, l, tid);  // in flight across this chunk's math

        float acc[8][4], yh[kJ][4], yi[kJ][4];
        zero(acc);
        zero(yh);
        zero(yi);
        gemm_g_inter(s, l, acc, yh);
        int ea, eb;
        mask_rows(s, l, acc, ea, eb);
        gemm_intra(s, l, acc, yi);
        write_y<FUSE_GATE>(s, l, yi, yh, ea, eb, eh, d_val, y, z, c0, n_tokens, inner, col0);
        update_state(s, l, hacc);
        eh = sync_publish_state(s, hacc, l);
    }
    if (real_n > 0)
        state_io<H_FP16>(h_state, h_base, head_dim, l, hacc, true);
    if (real_n > 0)
        state_io<H_FP16>(h_state, h_base, head_dim, l, hacc, true);
}

template <bool H_FP16, bool FUSE_GATE>
void launch_ssd(const SsmScanArgs& a) {
    auto* k = ssm_scan_ssd_kernel<H_FP16, FUSE_GATE>;
    static const cudaError_t attr = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         static_cast<int>(sizeof(Smem)));
    (void)attr;
    const dim3 grid(a.n_heads, a.head_dim_ssm / kDW);
    k<<<grid, kThreads, sizeof(Smem), a.stream>>>(a.x, a.B, a.C, a.dt, a.A_log, a.D, a.dt_bias, a.h_state,
                                                  a.y, a.z, a.n_tokens, a.n_heads, a.head_dim_ssm, a.n_groups,
                                                  a.d_real_n);
    IMP_CUDA_CHECK_LAUNCH();
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }
bool aligned4(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 3u) == 0; }

}  // namespace

bool ssm_scan_ssd_launch(const SsmScanArgs& a, bool fp16) {
    if (a.state_size != kN || a.head_dim_ssm % kDW != 0 || a.n_groups <= 0 || a.n_heads % a.n_groups != 0 ||
        a.h_snap != nullptr || a.n_tokens < kSsdMinTokens || !aligned16(a.B) || !aligned16(a.C) ||
        !aligned4(a.y) || (a.z && !aligned4(a.z)))
        return false;
    const bool fused = a.z != nullptr;
    if (fp16 && fused)
        launch_ssd<true, true>(a);
    else if (fp16)
        launch_ssd<true, false>(a);
    else if (fused)
        launch_ssd<false, true>(a);
    else
        launch_ssd<false, false>(a);
    return true;
}

}  // namespace imp
