// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// Vendored from kekzl/apa include/apa/apa_pass2.cuh (v0.8.2, abe6994); change there first, then copy.
// APA pass 2: exact FP16 attention over the hot tiles of pass 1, merged with its cold-tile partials.
// CTA = same packed rows as pass 1 (Launch<D>::NW warps); it streams the union of its warps' hot tiles, each
// warp computes only its own. QK^T and PV on mma.sync m16n8k16 (f16 x f16 -> f16 accumulate), online softmax
// in fp32 (log2 domain).
#pragma once
#include <type_traits>

#include "apa_attn.cuh"
#include "apa_prep.cuh"

namespace apa {

// bench: per item (start, waits done, end) in globaltimer ns, (type 0 quant / 1 pass 1 / 2 pass 2, k, r, SM)
#ifdef APA_TRACE
struct TraceRec {
    unsigned long long t0, t1, t2, info;
};
__device__ TraceRec* g_apa_trace;
__device__ unsigned g_apa_trace_n, g_apa_trace_cap;
__device__ uint64_t g_apa_tw[1024];  // per CTA: pass-2 wait done
__device__ __forceinline__ uint64_t gtime() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void trace(uint64_t t0, uint64_t t1, int type, int k, int r) {
    if (threadIdx.x != 0 || g_apa_trace == nullptr)
        return;
    unsigned sm;
    asm("mov.u32 %0, %%smid;" : "=r"(sm));
    const unsigned i = atomicAdd(&g_apa_trace_n, 1u);
    if (i < g_apa_trace_cap)
        g_apa_trace[i] = TraceRec{t0, t1, gtime(),
                                  ((uint64_t)type << 56) | ((uint64_t)sm << 40) | ((uint64_t)k << 20) | r};
}
#endif

// f16 x f16 -> f16 accumulate (full-rate HMMA on sm_120, as imp's FA2 default): d/c are two half2 registers.
__device__ __forceinline__ void mma_f16h(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
template <typename T>
__device__ __forceinline__ uint32_t load_h2(const T* p) {  // two consecutive elements -> half2 bits
    if constexpr (std::is_same_v<T, __half>) {
        return *reinterpret_cast<const uint32_t*>(p);
    } else {
        return pack_h2(to_f(p[0]), to_f(p[1]));
    }
}

// Launch config per head dim: warps per CTA, pipeline stages (1 CTA per SM).
template <int D>
struct Launch;
template <>
struct Launch<64> {
#ifndef APA_NW64
#define APA_NW64 16
#define APA_ST64 6
#endif
    static constexpr int NW = APA_NW64, ST = APA_ST64;
};
#ifndef APA_NW128
#define APA_NW128 12
#define APA_ST128 6
#endif
template <>
struct Launch<128> {
    static constexpr int NW = APA_NW128, ST = APA_ST128;
};
template <>
struct Launch<256> {
    static constexpr int NW = 8, ST = 4;
};

// Pass-2 step: BT keys of K and V in smem, [BT rows][D / 8 chunks of 16 B], chunk c stored at c ^ (row & 7).
// hd 256: half tiles (SUB 2 steps per KV tile), 3 stages = 96 KiB as at hd 128.
template <int D>
struct P2Cfg {
    static constexpr int BT = D == 128 ? BKV : BKV / 2;
    static constexpr int SUB = BKV / BT;
    static constexpr int NT = BT / 8;  // n8 tiles of S
    static constexpr int CH = D / 8;   // 16 B chunks per row
    static constexpr int ROWB = D * 2;
    static constexpr int SMEM = 2 * BT * ROWB;
};

// O accumulator, DT n8 tiles x (row lo cols T0*2,+1 | row hi). APA_P2_F32O 0: half2, f16 across tiles
// (full-rate HMMA). 1: fp32 across tiles, f16 per 16-token k-step; all-exact lc_122880_1 cos min 0.988660 ->
// 0.999997, attn +14 to 25 % at eps 5e-3 (register spills).
#ifndef APA_P2_F32O
#define APA_P2_F32O 0
#endif
template <int DT>
struct OAcc {
#if APA_P2_F32O
    alignas(16) float v[DT][4];
    __device__ __forceinline__ void zero() {
#pragma unroll
        for (int i = 0; i < DT; ++i)
            v[i][0] = v[i][1] = v[i][2] = v[i][3] = 0.f;
    }
    __device__ __forceinline__ void scale(float a_lo, float a_hi) {
#pragma unroll
        for (int i = 0; i < DT; ++i)
            v[i][0] *= a_lo, v[i][1] *= a_lo, v[i][2] *= a_hi, v[i][3] *= a_hi;
    }
    __device__ __forceinline__ float2 get(int dt, int half) const {
        return make_float2(v[dt][2 * half], v[dt][2 * half + 1]);
    }
    // n8 tiles 2dp, 2dp+1 += P V (one k-step)
    __device__ __forceinline__ void mma2(int dp, const uint32_t* pa, const uint32_t* vb) {
        uint32_t t[2][2] = {{0u, 0u}, {0u, 0u}};
        mma_f16h(t[0], pa, vb[0], vb[1]);
        mma_f16h(t[1], pa, vb[2], vb[3]);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&t[j][0]));
            const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&t[j][1]));
            v[2 * dp + j][0] += lo.x, v[2 * dp + j][1] += lo.y, v[2 * dp + j][2] += hi.x,
                v[2 * dp + j][3] += hi.y;
        }
    }
#else
    alignas(16) uint32_t v[DT][2];
    __device__ __forceinline__ void zero() {
#pragma unroll
        for (int i = 0; i < DT; ++i)
            v[i][0] = v[i][1] = 0u;
    }
    __device__ __forceinline__ void scale(float a_lo, float a_hi) {
        const __half2 al = __float2half2_rn(a_lo), ah = __float2half2_rn(a_hi);
#pragma unroll
        for (int i = 0; i < DT; ++i) {
            *reinterpret_cast<__half2*>(&v[i][0]) = __hmul2(*reinterpret_cast<const __half2*>(&v[i][0]), al);
            *reinterpret_cast<__half2*>(&v[i][1]) = __hmul2(*reinterpret_cast<const __half2*>(&v[i][1]), ah);
        }
    }
    __device__ __forceinline__ float2 get(int dt, int half) const {
        return __half22float2(*reinterpret_cast<const __half2*>(&v[dt][half]));
    }
    __device__ __forceinline__ void mma2(int dp, const uint32_t* pa, const uint32_t* vb) {
        mma_f16h(v[2 * dp], pa, vb[0], vb[1]);
        mma_f16h(v[2 * dp + 1], pa, vb[2], vb[3]);
    }
#endif
};

// Per-warp exact state: O, max and sum (log2).
template <int D>
struct Rows2 {
    OAcc<D / 8> o;
    float m_lo, m_hi, l_lo, l_hi;
};

// Q A-fragments (D / 16 k-steps of 16): a0 row lo k T0*2, a1 row hi, a2 row lo k+8, a3 row hi k+8.
template <int D, typename T>
__device__ __forceinline__ void load_q16(uint32_t (&qa)[D / 16][4], const T* ql, const T* qh) {
#pragma unroll
    for (int k = 0; k < D / 16; ++k) {
        qa[k][0] = load_h2(ql + k * 16);
        qa[k][1] = load_h2(qh + k * 16);
        qa[k][2] = load_h2(ql + k * 16 + 8);
        qa[k][3] = load_h2(qh + k * 16 + 8);
    }
}

// Pass 1 scores are q.(k - kmean): row shift q.kmean * qmul from the lane's 32 Q elements + quad sum.
template <int D>
__device__ __forceinline__ void row_shift(const uint32_t (&qa)[D / 16][4], const float* kmean, float qmul,
                                          int T0, float& sh_lo, float& sh_hi) {
    sh_lo = sh_hi = 0.f;
#pragma unroll
    for (int k = 0; k < D / 16; ++k)
#pragma unroll
        for (int hlf = 0; hlf < 2; ++hlf) {
            const int c = k * 16 + hlf * 8 + T0 * 2;
            const float k0 = kmean[c], k1 = kmean[c + 1];
            const __half2 a = *reinterpret_cast<const __half2*>(&qa[k][hlf * 2]);
            const __half2 bq = *reinterpret_cast<const __half2*>(&qa[k][hlf * 2 + 1]);
            sh_lo += __low2float(a) * k0 + __high2float(a) * k1;
            sh_hi += __low2float(bq) * k0 + __high2float(bq) * k1;
        }
    sh_lo += __shfl_xor_sync(~0u, sh_lo, 1);
    sh_lo += __shfl_xor_sync(~0u, sh_lo, 2);
    sh_hi += __shfl_xor_sync(~0u, sh_hi, 1);
    sh_hi += __shfl_xor_sync(~0u, sh_hi, 2);
    sh_lo *= qmul;
    sh_hi *= qmul;
}

// Pass-2 tile range of one item: sink tile 0 (if sink), then tiles [lo, hi) (lo >= 1).
struct P2Range {
    int lo, hi;
    bool sink;
};

// Highest tile in [lo, below) of the union of NW warps' masks (W words each); -1: none.
template <int NW>
__device__ __forceinline__ int prev_tile(const uint32_t* whot0, int W, int below, int lo) {
    if (below <= lo)
        return -1;
    for (int w = (below - 1) >> 5; w >= (lo >> 5); --w) {
        uint32_t bits = 0u;
#pragma unroll
        for (int i = 0; i < NW; ++i)
            bits |= whot0[i * W + w];
        if (w == (below - 1) >> 5 && (below & 31))
            bits &= (1u << (below & 31)) - 1u;
        if (w == (lo >> 5))
            bits &= ~0u << (lo & 31);
        if (bits)
            return w * 32 + 31 - __clz(bits);
    }
    return -1;
}

// Next step after st (SUB steps of P2Cfg::BT keys per KV tile, -1: start) in the union: the sink tile, then
// tiles hi - 1 .. lo (pass-1 order: segments of the KV window chain to one pass over every tile); -1 at the
// end.
template <int NW, int SUB>
__device__ __forceinline__ int next_step(const uint32_t* whot0, int W, int st, const P2Range& rg) {
    if (SUB > 1 && st >= 0 && st % SUB != SUB - 1)
        return st + 1;
    const int t = st < 0 ? -1 : st / SUB;
    uint32_t b0 = 0u;
    if (t < 0 && rg.sink) {
#pragma unroll
        for (int i = 0; i < NW; ++i)
            b0 |= whot0[i * W];
    }
    const int n = (b0 & 1u) ? 0 : prev_tile<NW>(whot0, W, t <= 0 ? rg.hi : t, rg.lo);
    return n < 0 ? -1 : n * SUB;
}

// S = Q K^T (f16 accumulate, as imp's FA2 fa2_f16acc): B from K rows (n = token), x4 = 2 k-steps.
template <int D>
__device__ __forceinline__ void qk16(float (&s)[P2Cfg<D>::NT][4], const uint32_t (&qa)[D / 16][4],
                                     uint32_t kbase, int lm_i, int lm_r) {
#pragma unroll
    for (int nt = 0; nt < P2Cfg<D>::NT; ++nt) {
        uint32_t sh2[2] = {0u, 0u};
#pragma unroll
        for (int q2 = 0; q2 < D / 32; ++q2) {
            const int r = nt * 8 + lm_r, ch = 4 * q2 + lm_i;
            uint32_t kb[4];
            ldsm_x4(kb, kbase + r * P2Cfg<D>::ROWB + ((ch ^ (r & 7)) << 4));
            mma_f16h(sh2, qa[2 * q2], kb[0], kb[1]);
            mma_f16h(sh2, qa[2 * q2 + 1], kb[2], kb[3]);
        }
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&sh2[0]));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&sh2[1]));
        s[nt][0] = lo.x, s[nt][1] = lo.y, s[nt][2] = hi.x, s[nt][3] = hi.y;
    }
}

// Scale, shift, mask (col > lim or col < lb), new row max over the quad.
template <int NT>
__device__ __forceinline__ void scale_mask16(float (&s)[NT][4], float qmul, float sh_lo, float sh_hi,
                                             int col0, int lim_lo, int lim_hi, int lb_lo, int lb_hi,
                                             float& mx_lo, float& mx_hi) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const bool hi = e >> 1;
            const int c = col0 + nt * 8 + (e & 1);
            float x = fmaf(s[nt][e], qmul, -(hi ? sh_hi : sh_lo));
            if (c > (hi ? lim_hi : lim_lo) || c < (hi ? lb_hi : lb_lo))
                x = -INFINITY;
            s[nt][e] = x;
            if (hi)
                mx_hi = fmaxf(mx_hi, x);
            else
                mx_lo = fmaxf(mx_lo, x);
        }
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 1));
    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 2));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 1));
    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 2));
}

// Exact online softmax step: rescale O and l on a moved max, s -> p, l += row sums.
template <int D, int NT>
__device__ __forceinline__ void softmax16(Rows2<D>& w, float (&s)[NT][4], float mx_lo, float mx_hi) {
    const float a_lo = (mx_lo == w.m_lo) ? 1.f : ex2(w.m_lo - mx_lo);
    const float a_hi = (mx_hi == w.m_hi) ? 1.f : ex2(w.m_hi - mx_hi);
    w.m_lo = mx_lo;
    w.m_hi = mx_hi;
    const float ms_lo = (w.m_lo == -INFINITY) ? 0.f : w.m_lo, ms_hi = (w.m_hi == -INFINITY) ? 0.f : w.m_hi;
    w.l_lo *= a_lo;
    w.l_hi *= a_hi;
    if (a_lo != 1.f || a_hi != 1.f) {
        w.o.scale(a_lo, a_hi);
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        s[nt][0] = ex2(s[nt][0] - ms_lo);
        s[nt][1] = ex2(s[nt][1] - ms_lo);
        s[nt][2] = ex2(s[nt][2] - ms_hi);
        s[nt][3] = ex2(s[nt][3] - ms_hi);
        w.l_lo += s[nt][0] + s[nt][1];
        w.l_hi += s[nt][2] + s[nt][3];
    }
}

// O += P V: P A-fragment per 16-token k-step from the S C-fragments, V via ldmatrix.trans.
template <int D>
__device__ __forceinline__ void pv16(Rows2<D>& w, const float (&s)[P2Cfg<D>::NT][4], uint32_t vbase, int lm_i,
                                     int lm_r) {
#pragma unroll
    for (int kk = 0; kk < P2Cfg<D>::BT / 16; ++kk) {
        const uint32_t pa[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
        for (int dp = 0; dp < D / 16; ++dp) {
            const int r = kk * 16 + (lm_i & 1) * 8 + lm_r, ch = 2 * dp + (lm_i >> 1);
            uint32_t vb[4];
            ldsm_x4_t(vb, vbase + r * P2Cfg<D>::ROWB + ((ch ^ (r & 7)) << 4));
            w.o.mma2(dp, pa, vb);
        }
    }
}

// Log-sum-exp merge of packed row r with its pass-1 cold partial (m1, l1, part); writes O.
template <int D, typename T>
__device__ __forceinline__ void merge_row(const Rows2<D>& w, int half, const Pass1Out& p1, T* O,
                                          const Dims& dm, int b, int hk, int r, int T0) {
    const size_t qrow = (size_t)(b * dm.Hkv + hk) * p1.rp + r;
    const float m2 = half ? w.m_hi : w.m_lo, l2 = half ? w.l_hi : w.l_lo;
    const float2 ml1 = p1.ml[qrow];
    const float M = fmaxf(ml1.x, m2);
    const float w1 = (ml1.y > 0.f) ? ex2(ml1.x - M) : 0.f, w2 = ex2(m2 - M);
    const float inv = 1.f / (w1 * ml1.y + w2 * l2);
    const float* part = p1.part + qrow * D + T0 * 2;
    T* dst = O + (((size_t)b * dm.Sq + r / dm.G) * dm.H + hk * dm.G + r % dm.G) * D + T0 * 2;
#pragma unroll
    for (int dt = 0; dt < D / 8; ++dt) {
        const float2 a = *reinterpret_cast<const float2*>(part + dt * 8);
        const float2 ov = w.o.get(dt, half);
        store2(dst + dt * 8, (w1 * a.x + w2 * ov.x) * inv, (w1 * a.y + w2 * ov.y) * inv);
    }
}

// Pass 2 per q block with all NW warps: the union of their hot tiles streams through P2S_STG K/V stages; any
// warp that finds the next stage free claims and loads it; every warp passes every step, computing only its
// own tiles. vs 3 groups of 4 warps with bar.sync (<= 0.3.0): 1.2 to 3.1 % less attention time at eps 0.005 /
// 0.01.
constexpr int P2S_STG = 3;
constexpr uint32_t P2S_DONE = ~0u;
struct P2Stream {
    uint64_t full[P2S_STG];   // count 32: the claimant warp's cp.async.mbarrier.arrive.noinc
    uint64_t empty[P2S_STG];  // count NW * 32: every lane after each union step
    unsigned long long
        claim;  // low: next union step to load (P2S_DONE: none), high: its predecessor step + 1
};

// Claim and load union step k (stage k % P2S_STG) if that stage's previous use is consumed. Warp-collective.
template <int D, int NW, typename RS>
__device__ __forceinline__ void p2s_refill(P2Stream& ps, uint8_t* smem, const RS& rs, const uint32_t* whot0,
                                           int W, const P2Range& rg, const Dims& dm, int b, int hk,
                                           int lane) {
    unsigned long long c = 0;
    int tn = -1;
    bool ok = false;
    if (lane == 0) {
        c = *reinterpret_cast<volatile unsigned long long*>(&ps.claim);
        const uint32_t k = (uint32_t)c;
        if (k != P2S_DONE) {
            const uint32_t s = k % P2S_STG, use = k / P2S_STG;
            if (use == 0 || mbar_test((uint32_t)__cvta_generic_to_shared(&ps.empty[s]), (use - 1) & 1)) {
                tn = next_step<NW, P2Cfg<D>::SUB>(whot0, W, (int)(c >> 32) - 1, rg);
                const unsigned long long nc = tn < 0 ? (unsigned long long)P2S_DONE
                                                     : ((unsigned long long)(tn + 1) << 32) | (k + 1);
                ok = tn >= 0 && atomicCAS(&ps.claim, c, nc) == c;
                if (tn < 0)
                    atomicCAS(&ps.claim, c, nc);
            }
        }
    }
    if (!__shfl_sync(~0u, ok, 0))
        return;
    const uint32_t k = __shfl_sync(~0u, (uint32_t)c, 0), s = k % P2S_STG;
    tn = __shfl_sync(~0u, tn, 0);
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem) + s * P2Cfg<D>::SMEM;
    // BT rows x CH chunks of K and V by cp.async (zeros past Skv / holes); lane: chunk lane % CH of rows
    // lane / CH + RPI * i. Row offsets first (lane: rows lane, lane + 32; paged: independent block-table
    // loads), then shuffled.
    using P = P2Cfg<D>;
    constexpr int RPI = 32 / P::CH;  // rows per iteration
    const int ch = lane & (P::CH - 1), j0 = tn * P::BT;
    const uint32_t ro0 = j0 + lane < dm.Skv ? rs.row_off(b, j0 + lane, hk, dm, D) : ~0u;
    const uint32_t ro1 = P::BT > 32 && j0 + lane + 32 < dm.Skv ? rs.row_off(b, j0 + lane + 32, hk, dm, D)
                                                               : ~0u;
#pragma unroll 4
    for (int i = 0; i < P::BT / RPI; ++i) {
        const int r = lane / P::CH + RPI * i;
        const uint32_t ro = __shfl_sync(~0u, i < 32 / RPI ? ro0 : ro1, r & 31);
        const uint32_t dst = base + r * P::ROWB + ((ch ^ (r & 7)) << 4);
        const bool in = ro != ~0u;
        cp_async16(dst, rs.at(false, in ? ro : 0u) + ch * 8, in);
        cp_async16(dst + P::BT * P::ROWB, rs.at(true, in ? ro : 0u) + ch * 8, in);
    }
    const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&ps.full[s]);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

// Pass 2 of q block qb with all NW warps (pass-1 warp w = warp w here), pass-2 segment k of nk: tiles of rg0
// after pass-1 segment k (nk 1: every tile after all of pass 1). k > 0 continues the state of segment k - 1,
// k < nk - 1 stores it (registers as they are: bitwise the same as one pass), k = nk - 1 merges. u: union
// steps of this CTA so far (ring position, same in every thread), advanced by this item's union length.
template <int D, int NW, bool CAUSAL, bool SW, typename T, typename RS>
__device__ __forceinline__ void pass2_cta(const T* __restrict__ Q, const RS rs, T* __restrict__ O,
                                          const Dims& dm, float qmul, const Pass1Out& p1, int b, int hk,
                                          int nqb, int qb, P2Stream& ps, uint8_t* smem, uint32_t& u,
                                          const P2Range& rg0, int k, int nk) {
    const int bhk = b * dm.Hkv + hk, R = dm.R, G = dm.G, crow0 = qb * NW * 16, tid = threadIdx.x;
    if (crow0 >= R)
        return;
    const int warp = tid >> 5, lane = tid & 31, T0 = lane & 3, row0 = crow0 + warp * 16;
    const uint32_t* whot0 = p1.warp_hot + (size_t)(bhk * nqb + qb) * NW * p1.W;
    const uint32_t* whot = whot0 + warp * p1.W;
    const bool valid = row0 < R, last = k == nk - 1;
    const int p_wmax = dm.q_offset + min(row0 + 15, R - 1) / G;
    const int p_cmax = dm.q_offset + (min(crow0 + NW * 16, R) - 1) / G;
    const int r_lo = row0 + (lane >> 2), r_hi = r_lo + 8;
    const int lim_lo = CAUSAL ? min(dm.q_offset + min(r_lo, R - 1) / G, dm.Skv - 1) : dm.Skv - 1;
    const int lim_hi = CAUSAL ? min(dm.q_offset + min(r_hi, R - 1) / G, dm.Skv - 1) : dm.Skv - 1;
    const int sw = SW ? dm.swin : 0;  // sliding window: first visible key per row (INT_MIN: none)
    const int lb_lo = sw > 0 ? dm.q_offset + min(r_lo, R - 1) / G - sw + 1 : INT_MIN;
    const int lb_hi = sw > 0 ? dm.q_offset + min(r_hi, R - 1) / G - sw + 1 : INT_MIN;
    const int lb_wmin = sw > 0 ? dm.q_offset + min(row0, R - 1) / G - sw + 1 : INT_MIN;
    auto q_row = [&](int r) {
        const int rr = min(r, R - 1);
        return Q + (((size_t)b * dm.Sq + rr / G) * dm.H + hk * G + rr % G) * D + T0 * 2;
    };
    uint32_t qa[D / 16][4];
    load_q16<D>(qa, q_row(r_lo), q_row(r_hi));
    float sh_lo, sh_hi;
    row_shift<D>(qa, p1.ksum + (size_t)bhk * D, qmul, T0, sh_lo, sh_hi);
    Rows2<D> w;
    w.o.zero();
    w.m_lo = w.m_hi = -INFINITY;
    w.l_lo = w.l_hi = 0.f;
    bool mine_any = false, got = k == 0, step_any = false;  // warp-uniform
    // state of segments < k: (m, l) per lane, O registers only if the warp had a hot tile (m finite)
    constexpr int OW = sizeof(w.o.v) / 16;
    const size_t wt = (size_t)(bhk * nqb + qb) * NW + warp;
    auto load_state = [&]() {
        const float4 a = p1.p2ml[wt * 32 + lane];
        w.m_lo = a.x, w.m_hi = a.y, w.l_lo = a.z, w.l_hi = a.w;
        mine_any = __any_sync(~0u, a.x != -INFINITY || a.y != -INFINITY);
        if (mine_any) {
#pragma unroll
            for (int i = 0; i < OW; ++i)
                reinterpret_cast<uint4*>(&w.o.v)[i] = p1.p2o[(wt * OW + i) * 32 + lane];
        }
        got = true;
    };

    if (tid == 0) {  // fused launch: pass-1 items (and earlier pass-2 segments) may still run
        wait_geq(p1.ready + bhk * nqb + qb, last ? p1.nseg : k + 1);
        if (k > 0)
            wait_geq(p1.p2ready + bhk * nqb + qb, k);
    }
#ifdef APA_TRACE
    if (tid == 0)
        g_apa_tw[blockIdx.x & 1023] = gtime();
#endif
    __syncthreads();
    const int lm_i = lane >> 3, lm_r = lane & 7;
    const P2Range rg{rg0.lo, min(rg0.hi, min(dm.ntkv, CAUSAL ? p_cmax / BKV + 1 : dm.ntkv)), rg0.sink};
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    using P = P2Cfg<D>;
    for (int t = next_step<NW, P::SUB>(whot0, p1.W, -1, rg); t >= 0;
         t = next_step<NW, P::SUB>(whot0, p1.W, t, rg), ++u) {  // t: step (KV tile at SUB 1)
        const uint32_t s = u % P2S_STG, ph = (u / P2S_STG) & 1;
        const uint32_t fb = (uint32_t)__cvta_generic_to_shared(&ps.full[s]);
        for (;;) {  // warp-uniform: lane 0 tests, the warp loads stages meanwhile
            if (__shfl_sync(~0u, lane == 0 ? (int)mbar_test(fb, ph) : 0, 0))
                break;
            p2s_refill<D, NW>(ps, smem, rs, whot0, p1.W, rg, dm, b, hk, lane);
        }
        mbar_wait(fb, ph);  // acquire in every lane (phase already complete)
        const int j0 = t * P::BT, tt = t / P::SUB;
        if (valid && (!CAUSAL || j0 <= p_wmax) && j0 + P::BT > lb_wmin &&
            ((whot[tt >> 5] >> (tt & 31)) & 1u)) {
            if (!got)
                load_state();
            mine_any = step_any = true;
            float sc[P::NT][4];
            qk16<D>(sc, qa, sbase + s * P::SMEM, lm_i, lm_r);
            float mx_lo = w.m_lo, mx_hi = w.m_hi;
            scale_mask16(sc, qmul, sh_lo, sh_hi, j0 + T0 * 2, lim_lo, lim_hi, lb_lo, lb_hi, mx_lo, mx_hi);
            softmax16(w, sc, mx_lo, mx_hi);
            pv16<D>(w, sc, sbase + s * P::SMEM + P::BT * P::ROWB, lm_i, lm_r);
        }
        mbar_arrive((uint32_t)__cvta_generic_to_shared(&ps.empty[s]));
        p2s_refill<D, NW>(ps, smem, rs, whot0, p1.W, rg, dm, b, hk, lane);
    }
    if (!valid)
        return;
    if (!last) {  // segment 0 always stores (the next segment reads it); later ones only a changed state
        if (k == 0 || step_any) {
            p1.p2ml[wt * 32 + lane] = make_float4(w.m_lo, w.m_hi, w.l_lo, w.l_hi);
            if (step_any) {
#pragma unroll
                for (int i = 0; i < OW; ++i)
                    p1.p2o[(wt * OW + i) * 32 + lane] = reinterpret_cast<const uint4*>(&w.o.v)[i];
            }
        }
        return;
    }
    if (!got)
        load_state();
    if (!mine_any)
        return;
    w.l_lo += __shfl_xor_sync(~0u, w.l_lo, 1);
    w.l_lo += __shfl_xor_sync(~0u, w.l_lo, 2);
    w.l_hi += __shfl_xor_sync(~0u, w.l_hi, 1);
    w.l_hi += __shfl_xor_sync(~0u, w.l_hi, 2);
    if (r_lo < R)
        merge_row<D>(w, 0, p1, O, dm, b, hk, r_lo, T0);
    if (r_hi < R)
        merge_row<D>(w, 1, p1, O, dm, b, hk, r_hi, T0);
}

// KV window: quant items fill WIN_SLOTS ring slots of S tiles per (b, kv head) ahead of the pass-1 segments
// that read them. kv == nullptr: KV holds every tile (stride kvcap), one segment, no quant items.
// Ring slots: segment k + 1 reuses the slot of k + 1 - WIN_SLOTS. 3: its quant items go before pass 1 of
// segment k (k - 2 is done by then); 2: after half of them (k - 1 done by then).
#ifndef APA_WIN_SLOTS
#define APA_WIN_SLOTS 2
#endif
constexpr int WIN_SLOTS = APA_WIN_SLOTS;
// 1: pass 2 of segment k runs after pass 1 of segment k + 1 (state carried, bitwise as one pass); 0: all at
// the end.
#ifndef APA_P2_SEG
#define APA_P2_SEG 1
#endif
#ifndef APA_P2_EARLY  // segments 0 .. APA_P2_EARLY - 1 get early pass-2 items, the rest goes to the last one
#define APA_P2_EARLY 1000000
#endif
constexpr int WIN_QT = 9;  // tiles per quant item (3 passes of 3 tiles)
struct Win {
    uint8_t* kv;       // sink tile per (b, kv head), then WIN_SLOTS slots of [bhk][S] tiles
    uint32_t* qdone;   // [nseg][bhk] quant items done
    uint32_t* pdone;   // [nseg][bhk] pass-1 CTAs done
    int* ovf;          // quant overflow flag (sampled stats too narrow), nullptr: not checked
    int S, nseg, qpb;  // tiles per segment, segments, quant items per (segment, b, kv head)
};
// Segment k: tiles [lo, hi), hi = ntkv - k * S (tile 0: sink, quantized with segment 0).
__host__ __device__ inline int seg_hi(int k, int ntkv, int S) { return ntkv - k * S; }
__host__ __device__ inline int seg_lo(int k, int ntkv, int S) {
    const int lo = ntkv - (k + 1) * S;
    return lo > 1 ? lo : 1;
}

// Quant item smem: QB tiles per quant_tile pass (vt QB * 64 * (D + 2) halves) + K mean. hd 128: 3 tiles (1536
// groups on 384 threads), hd 256: 2 (2048 groups on 256 threads).
template <int D>
struct QuantCfg {
    static constexpr int QB = D == 128 ? 3 : 2;
    static constexpr int SMEM = D * 4 + QB * BKV * (D + 2) * 2;
};

// Quant item r of segment k: WIN_QT tiles of one (b, kv head) into slot k % WIN_SLOTS (item 0 of k = 0: +
// sink).
template <int D, int NW, typename RS>
__device__ __forceinline__ void quant_item(const RS& rs, const Dims& dm, const float* __restrict__ ksum,
                                           const HeadScale* __restrict__ hs, const Win& wn, int k, int r,
                                           int nqb, uint8_t* smem) {
    constexpr int TILE = Cfg<D>::TILE, NT = NW * 32, QB = QuantCfg<D>::QB;
    const int bhkn = dm.B * dm.Hkv, bhk = r / wn.qpb, j = r % wn.qpb, b = bhk / dm.Hkv, hk = bhk % dm.Hkv;
    const int tid = threadIdx.x, lo = seg_lo(k, dm.ntkv, wn.S), hi = seg_hi(k, dm.ntkv, wn.S);
#ifdef APA_TRACE
    const uint64_t tr0 = gtime();
#endif
    if (tid == 0 && k >= WIN_SLOTS)
        wait_geq(wn.pdone + (size_t)(k - WIN_SLOTS) * bhkn + bhk, nqb);
#ifdef APA_TRACE
    const uint64_t tr1 = gtime();
#endif
    float* kmean = reinterpret_cast<float*>(smem);
    auto vt = reinterpret_cast<__half(*)[D + 2]>(smem + D * 4);
    for (int d = tid; d < D; d += NT)
        kmean[d] = ksum[(size_t)bhk * D + d];
    __syncthreads();
    const HeadScale sc = hs[bhk];
    uint8_t* slot = wn.kv +
                    ((size_t)bhkn + (size_t)(k % WIN_SLOTS) * bhkn * wn.S + (size_t)bhk * wn.S) * TILE;
    const int t0 = lo + j * WIN_QT, t1 = min(hi, t0 + WIN_QT);
    float amx = 0.f;
    if (k == 0 && j == 0) {  // the sink tile 0
        amx = quant_tile<D>(rs, dm, kmean, sc.k, sc.v, wn.kv + (size_t)bhk * TILE, b, hk, 0, 1, tid, NT, vt);
        __syncthreads();  // vt read by every thread
    }
    for (int t = t0; t < t1; t += QB) {
        const int nb = min(QB, t1 - t);
        uint8_t* blob = slot + (size_t)(t - lo) * TILE;
        amx = fmaxf(amx, quant_tile<D>(rs, dm, kmean, sc.k, sc.v, blob, b, hk, t, nb, tid, NT, vt));
        __syncthreads();
    }
    if (wn.ovf != nullptr && __syncthreads_or(!(amx <= P_SCALE)) && tid == 0)
        atomicOr(wn.ovf, 1);  // NaN / inf too
    fence_proxy_async();      // the tiles go to bulk copies
    __threadfence();
    __syncthreads();
    if (tid == 0)
        add_release(wn.qdone + (size_t)k * bhkn + bhk, 1);
#ifdef APA_TRACE
    trace(tr0, tr1, 0, k, r);
#endif
}

// Both passes in one persistent launch (Launch<D> warps, grid <= SMs). A CTA takes tickets in order: per key
// segment k, the quant items of segment k + 1, then pass 1 of segment k on every q block (segment 0 after the
// quant items of segment 0); past the last pass-1 ticket it runs pass 2 on whole q blocks (pass2_cta). Every
// wait (quant of k, segment k - 1 of the q block, slot reuse, pass 1 for pass 2) is on an earlier ticket,
// held by a running CTA: no deadlock. Pass 2 fills the SMs of pass 1's last wave. gate: run only if *gate
// (redo).
template <int D, bool CAUSAL, bool SW, bool P2S, typename T, typename RS>
__global__ void __launch_bounds__(Launch<D>::NW * 32, 1) apa_kernel(
    const uint8_t* __restrict__ Qq, const uint8_t* __restrict__ Qs, const float* __restrict__ Qr,
    const uint8_t* __restrict__ KV, const HeadScale* __restrict__ hs, const T* __restrict__ Q, RS rs,
    T* __restrict__ O, Dims dm, float qmul, Pass1Out p1, int n1, int nqb, uint32_t* ticket, Win wn,
    const int* gate) {
    constexpr int NW = Launch<D>::NW, TILE = Cfg<D>::TILE;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ int item_s;
    __shared__ P2Stream ps;
    if (gate != nullptr && *gate == 0)
        return;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < P2S_STG; ++s) {
            mbar_init((uint32_t)__cvta_generic_to_shared(&ps.full[s]), 32);
            mbar_init((uint32_t)__cvta_generic_to_shared(&ps.empty[s]), NW * 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    uint32_t u = 0;  // pass-2 union steps of this CTA (ring position)
    const int bhkn = dm.B * dm.Hkv, nq = (wn.kv != nullptr && n1 > 0) ? wn.qpb * bhkn : 0;
    const int n2e = P2S && nq > 0 && wn.nseg > 1 ? n1 : 0;  // pass-2 items per segment in the ticket stream
    const int ne = n2e > 0 ? min(wn.nseg - 1, APA_P2_EARLY) : 0;  // segments with early pass-2 items
    const int n_items = wn.nseg * (nq + n1) + ne * n2e;
    for (;;) {
        __syncthreads();  // previous item done in every thread: item_s and smem are free
        if (threadIdx.x == 0)
            item_s = (int)atomicAdd(ticket, 1u);
        __syncthreads();
        const int x = item_s;
        if (x >= n_items)
            break;
        // ticket -> item: Q(0) | P(0)[0, n1a) Q(1) P(0)[n1a, n1) | P(1).. Q(2) .. P2(0) | .. | P(nseg - 1)
        // P2(nseg - 2); P2(k): pass 2 of segment k (n2e > 0), after pass 1 of segment k + 1
        const int n1a = WIN_SLOTS >= 3 ? 0 : n1 / 2;
        int k = 0, r = x;
        bool quant = x < nq;
        if (!quant) {
            const int y = x - nq;
            if (y < nq + n1) {
                k = 0, r = y;
            } else {
                const int y1 = y - (nq + n1), bs = nq + n1 + n2e;
                if (y1 < ne * bs) {
                    k = 1 + y1 / bs;
                    r = y1 - (k - 1) * bs;
                } else {
                    const int y2 = y1 - ne * bs;
                    k = 1 + ne + y2 / (nq + n1);
                    r = y2 - (k - 1 - ne) * (nq + n1);
                }
            }
            const int n1k = n1 + (k < wn.nseg - 1 ? nq : 0);
            if (k <= ne && r >= n1k) {  // P2(k - 1)
                const int j = r - n1k, jx = j % nqb, jhk = (j / nqb) % dm.Hkv, jb = j / (nqb * dm.Hkv);
                const int jq = CAUSAL ? nqb - 1 - jx : jx, kk = k - 1;
#ifdef APA_TRACE
                const uint64_t t0 = gtime();
#endif
                if (threadIdx.x == 0)
                    ps.claim = u;  // next union step u, predecessor -1 (read after pass2_cta's sync)
                const P2Range rg{seg_lo(kk, dm.ntkv, wn.S), seg_hi(kk, dm.ntkv, wn.S), kk == 0};
                pass2_cta<D, NW, CAUSAL, SW, T>(Q, rs, O, dm, qmul, p1, jb, jhk, nqb, jq, ps, smem, u, rg, kk,
                                                wn.nseg);
                // publish the carried state: every thread fences, then a release add
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0)
                    add_release(p1.p2ready + (jb * dm.Hkv + jhk) * nqb + jq, 1);
#ifdef APA_TRACE
                trace(t0, g_apa_tw[blockIdx.x & 1023], 2, kk, j);
#endif
                continue;
            }
            quant = k < wn.nseg - 1 && r >= n1a && r < n1a + nq;
            if (quant)
                ++k, r -= n1a;
            else if (k < wn.nseg - 1 && r >= n1a)
                r -= nq;
        }
        if (quant) {
            quant_item<D, NW>(rs, dm, p1.ksum, hs, wn, k, r, nqb, smem);
            continue;
        }
        const int qx = r % nqb, hk = (r / nqb) % dm.Hkv, b = r / (nqb * dm.Hkv), bhk = b * dm.Hkv + hk;
        const int qb = CAUSAL ? nqb - 1 - qx : qx;  // heavy causal blocks first
        Seg sg;
#ifdef APA_TRACE
        const uint64_t t0 = gtime();
        uint64_t t1 = t0;
#endif
        if (wn.kv != nullptr) {
            if (threadIdx.x == 0) {
                if (k > 0)
                    wait_geq(p1.ready + bhk * nqb + qb, k);  // segment k - 1 of this q block
                wait_geq(wn.qdone + (size_t)k * bhkn + bhk, wn.qpb);
                fence_proxy_async();
            }
            __syncthreads();
#ifdef APA_TRACE
            t1 = gtime();
#endif
            sg.sink = wn.kv + (size_t)bhk * TILE;
            sg.lo = seg_lo(k, dm.ntkv, wn.S);
            sg.hi = seg_hi(k, dm.ntkv, wn.S);
            sg.seg = wn.kv +
                     ((size_t)bhkn + (size_t)(k % WIN_SLOTS) * bhkn * wn.S + (size_t)bhk * wn.S) * TILE;
        } else {
            sg.sink = KV + (size_t)bhk * dm.kvcap * TILE;
            sg.seg = sg.sink + TILE;
            sg.lo = 1;
            sg.hi = dm.ntkv;
        }
        sg.first = k == 0;
        sg.last = k == wn.nseg - 1;
        pass1_cta<D, NW, Launch<D>::ST, CAUSAL, SW, T>(Qq, Qs, Qr, sg, hs, O, dm, p1, b, hk, qb, nqb, smem);
        // publish: every thread fences its writes (O, part, ml, carry, masks), then release adds
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            add_release(p1.ready + bhk * nqb + qb, 1);
            if (wn.pdone != nullptr)
                add_release(wn.pdone + (size_t)k * bhkn + bhk, 1);
        }
#ifdef APA_TRACE
        trace(t0, t1, 1, k, r);
#endif
    }
    // pass-2 worker: whole q blocks from ticket[1], all NW warps on one tile stream; the last pass-2 segment
    // (n2e > 0) or every tile
    const int k2 = ne, nk2 = ne + 1;  // segments ne .. nseg - 1
    const P2Range rg2 = ne > 0 ? P2Range{seg_lo(wn.nseg - 1, dm.ntkv, wn.S), seg_hi(ne, dm.ntkv, wn.S), false}
                               : P2Range{1, dm.ntkv, true};
    for (const int n2 = nqb * dm.Hkv * dm.B;;) {
        __syncthreads();  // previous q block done in every warp: claim and item_s are free
        if (threadIdx.x == 0) {
            item_s = (int)atomicAdd(ticket + 1, 1u);
            ps.claim = u;  // next union step u, predecessor tile -1
        }
        __syncthreads();
        const int j = item_s;
        if (j >= n2)
            return;
#ifdef APA_TRACE
        const uint64_t t0 = gtime();
#endif
        const int jx = j % nqb, jhk = (j / nqb) % dm.Hkv, jb = j / (nqb * dm.Hkv);
        pass2_cta<D, NW, CAUSAL, SW, T>(Q, rs, O, dm, qmul, p1, jb, jhk, nqb, CAUSAL ? nqb - 1 - jx : jx, ps,
                                        smem, u, rg2, k2, nk2);
#ifdef APA_TRACE
        trace(t0, g_apa_tw[blockIdx.x & 1023], 2, k2, j);
#endif
    }
}

}  // namespace apa
