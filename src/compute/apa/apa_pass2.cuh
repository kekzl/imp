// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// Vendored from kekzl/apa include/apa/apa_pass2.cuh (v0.6.1, 015301f); change there first, then copy.
// APA pass 2: exact FP16 attention over the hot tiles of pass 1, merged with its cold-tile partials.
// CTA = same packed rows as pass 1 (12 warps); it streams the union of its warps' hot tiles, each warp
// computes only its own. QK^T and PV on mma.sync m16n8k16 (f16 x f16 -> f16 accumulate), online softmax in
// fp32 (log2 domain).
#pragma once
#include <type_traits>

#include "apa_attn.cuh"

namespace apa {

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

// D = 128. K/V tile in smem: [64 rows][16 chunks of 16 B], chunk c stored at c ^ (row & 7).
constexpr int P2_SMEM = 2 * BKV * 128 * 2;  // K + V tile, D = 128 FP16
constexpr int P2_ROWB = 128 * 2;

// O accumulator, 16 n8 tiles x (row lo cols T0*2,+1 | row hi). APA_P2_F32O 0: half2, f16 across tiles
// (full-rate HMMA). 1: fp32 across tiles, f16 per 16-token k-step; all-exact lc_122880_1 cos min 0.988660 ->
// 0.999997, attn +14 to 25 % at eps 5e-3 (register spills).
#ifndef APA_P2_F32O
#define APA_P2_F32O 0
#endif
struct OAcc {
#if APA_P2_F32O
    float v[16][4];
    __device__ __forceinline__ void zero() {
#pragma unroll
        for (int i = 0; i < 16; ++i)
            v[i][0] = v[i][1] = v[i][2] = v[i][3] = 0.f;
    }
    __device__ __forceinline__ void scale(float a_lo, float a_hi) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
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
    uint32_t v[16][2];
    __device__ __forceinline__ void zero() {
#pragma unroll
        for (int i = 0; i < 16; ++i)
            v[i][0] = v[i][1] = 0u;
    }
    __device__ __forceinline__ void scale(float a_lo, float a_hi) {
        const __half2 al = __float2half2_rn(a_lo), ah = __float2half2_rn(a_hi);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
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
struct Rows2 {
    OAcc o;
    float m_lo, m_hi, l_lo, l_hi;
};

// Q A-fragments (8 k-steps of 16): a0 row lo k T0*2, a1 row hi, a2 row lo k+8, a3 row hi k+8.
template <typename T>
__device__ __forceinline__ void load_q16(uint32_t (&qa)[8][4], const T* ql, const T* qh) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        qa[k][0] = load_h2(ql + k * 16);
        qa[k][1] = load_h2(qh + k * 16);
        qa[k][2] = load_h2(ql + k * 16 + 8);
        qa[k][3] = load_h2(qh + k * 16 + 8);
    }
}

// Pass 1 scores are q.(k - kmean): row shift q.kmean * qmul from the lane's 32 Q elements + quad sum.
__device__ __forceinline__ void row_shift(const uint32_t (&qa)[8][4], const float* kmean, float qmul, int T0,
                                          float& sh_lo, float& sh_hi) {
    sh_lo = sh_hi = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
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

// Spin until pass-1 CTA of this q block published (fused launch: pass-1 items may still run).
__device__ __forceinline__ void wait_ready(const uint32_t* rdy) {
    uint32_t v;
    for (;;) {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(rdy) : "memory");
        if (v)
            return;
        __nanosleep(500);
    }
}

// Next tile after t in the union of NW warps' masks (W words each); -1 at t_end.
template <int NW>
__device__ __forceinline__ int next_tile(const uint32_t* whot0, int W, int t, int t_end) {
    for (int w = (t + 1) >> 5; w * 32 < t_end; ++w) {
        uint32_t bits = 0u;
#pragma unroll
        for (int i = 0; i < NW; ++i)
            bits |= whot0[i * W + w];
        if (w == (t + 1) >> 5)
            bits &= ~0u << ((t + 1) & 31);
        if (bits) {
            const int n = w * 32 + __ffs(bits) - 1;
            return n < t_end ? n : -1;
        }
    }
    return -1;
}

// S = Q K^T (f16 accumulate, as imp's FA2 fa2_f16acc): B from K rows (n = token), x4 = 2 k-steps.
__device__ __forceinline__ void qk16(float (&s)[8][4], const uint32_t (&qa)[8][4], uint32_t kbase, int lm_i,
                                     int lm_r) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        uint32_t sh2[2] = {0u, 0u};
#pragma unroll
        for (int q2 = 0; q2 < 4; ++q2) {
            const int r = nt * 8 + lm_r, ch = 4 * q2 + lm_i;
            uint32_t kb[4];
            ldsm_x4(kb, kbase + r * P2_ROWB + ((ch ^ (r & 7)) << 4));
            mma_f16h(sh2, qa[2 * q2], kb[0], kb[1]);
            mma_f16h(sh2, qa[2 * q2 + 1], kb[2], kb[3]);
        }
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&sh2[0]));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&sh2[1]));
        s[nt][0] = lo.x, s[nt][1] = lo.y, s[nt][2] = hi.x, s[nt][3] = hi.y;
    }
}

// Scale, shift, mask (col > lim), new row max over the quad.
__device__ __forceinline__ void scale_mask16(float (&s)[8][4], float qmul, float sh_lo, float sh_hi, int col0,
                                             int lim_lo, int lim_hi, float& mx_lo, float& mx_hi) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const bool hi = e >> 1;
            float x = fmaf(s[nt][e], qmul, -(hi ? sh_hi : sh_lo));
            if (col0 + nt * 8 + (e & 1) > (hi ? lim_hi : lim_lo))
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
__device__ __forceinline__ void softmax16(Rows2& w, float (&s)[8][4], float mx_lo, float mx_hi) {
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
    for (int nt = 0; nt < 8; ++nt) {
        s[nt][0] = ex2(s[nt][0] - ms_lo);
        s[nt][1] = ex2(s[nt][1] - ms_lo);
        s[nt][2] = ex2(s[nt][2] - ms_hi);
        s[nt][3] = ex2(s[nt][3] - ms_hi);
        w.l_lo += s[nt][0] + s[nt][1];
        w.l_hi += s[nt][2] + s[nt][3];
    }
}

// O += P V: P A-fragment per 16-token k-step from the S C-fragments, V via ldmatrix.trans.
__device__ __forceinline__ void pv16(Rows2& w, const float (&s)[8][4], uint32_t vbase, int lm_i, int lm_r) {
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const uint32_t pa[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
            const int r = kk * 16 + (lm_i & 1) * 8 + lm_r, ch = 2 * dp + (lm_i >> 1);
            uint32_t vb[4];
            ldsm_x4_t(vb, vbase + r * P2_ROWB + ((ch ^ (r & 7)) << 4));
            w.o.mma2(dp, pa, vb);
        }
    }
}

// Log-sum-exp merge of packed row r with its pass-1 cold partial (m1, l1, part); writes O.
template <typename T>
__device__ __forceinline__ void merge_row(const Rows2& w, int half, const Pass1Out& p1, T* O, const Dims& dm,
                                          int b, int hk, int r, int T0) {
    const size_t qrow = (size_t)(b * dm.Hkv + hk) * dm.R + r;
    const float m2 = half ? w.m_hi : w.m_lo, l2 = half ? w.l_hi : w.l_lo;
    const float2 ml1 = p1.ml[qrow];
    const float M = fmaxf(ml1.x, m2);
    const float w1 = (ml1.y > 0.f) ? ex2(ml1.x - M) : 0.f, w2 = ex2(m2 - M);
    const float inv = 1.f / (w1 * ml1.y + w2 * l2);
    const float* part = p1.part + qrow * 128 + T0 * 2;
    T* dst = O + (((size_t)b * dm.Sq + r / dm.G) * dm.H + hk * dm.G + r % dm.G) * 128 + T0 * 2;
#pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
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
        claim;  // low: next union step to load (P2S_DONE: none), high: its predecessor tile + 1
};

// Claim and load union step k (stage k % P2S_STG) if that stage's previous use is consumed. Warp-collective.
template <int NW, typename RS>
__device__ __forceinline__ void p2s_refill(P2Stream& ps, uint8_t* smem, const RS& rs, const uint32_t* whot0,
                                           int W, int t_end, const Dims& dm, int b, int hk, int lane) {
    unsigned long long c = 0;
    int tn = -1;
    bool ok = false;
    if (lane == 0) {
        c = *reinterpret_cast<volatile unsigned long long*>(&ps.claim);
        const uint32_t k = (uint32_t)c;
        if (k != P2S_DONE) {
            const uint32_t s = k % P2S_STG, use = k / P2S_STG;
            if (use == 0 || mbar_test((uint32_t)__cvta_generic_to_shared(&ps.empty[s]), (use - 1) & 1)) {
                tn = next_tile<NW>(whot0, W, (int)(c >> 32) - 1, t_end);
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
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem) + s * P2_SMEM;
    // 64 rows x 16 chunks of K and V by cp.async (zeros past Skv / holes); lane: chunk lane & 15 of rows lane
    // / 16 + 2i row offsets first (lane: rows lane, lane + 32; paged: two independent block-table loads),
    // then shuffled
    const int ch = lane & 15, j0 = tn * BKV;
    const uint32_t ro0 = j0 + lane < dm.Skv ? rs.row_off(b, j0 + lane, hk, dm, 128) : ~0u;
    const uint32_t ro1 = j0 + lane + 32 < dm.Skv ? rs.row_off(b, j0 + lane + 32, hk, dm, 128) : ~0u;
#pragma unroll 4
    for (int i = 0; i < BKV / 2; ++i) {
        const int r = (lane >> 4) + 2 * i;
        const uint32_t ro = __shfl_sync(~0u, i < BKV / 4 ? ro0 : ro1, r & 31);
        const uint32_t dst = base + r * P2_ROWB + ((ch ^ (r & 7)) << 4);
        const bool in = ro != ~0u;
        cp_async16(dst, rs.at(false, in ? ro : 0u) + ch * 8, in);
        cp_async16(dst + BKV * P2_ROWB, rs.at(true, in ? ro : 0u) + ch * 8, in);
    }
    const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&ps.full[s]);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

// Pass 2 of q block qb with all NW warps (pass-1 warp w = warp w here). u: union steps of this CTA so far
// (ring position, same in every thread), advanced by this q block's union length.
template <int NW, bool CAUSAL, typename T, typename RS>
__device__ __forceinline__ void pass2_cta(const T* __restrict__ Q, const RS rs, T* __restrict__ O,
                                          const Dims& dm, float qmul, const Pass1Out& p1, int b, int hk,
                                          int nqb, int qb, P2Stream& ps, uint8_t* smem, uint32_t& u) {
    const int bhk = b * dm.Hkv + hk, R = dm.R, G = dm.G, crow0 = qb * NW * 16, tid = threadIdx.x;
    if (crow0 >= R)
        return;
    const int warp = tid >> 5, lane = tid & 31, T0 = lane & 3, row0 = crow0 + warp * 16;
    const uint32_t* whot0 = p1.warp_hot + (size_t)(bhk * nqb + qb) * NW * p1.W;
    const uint32_t* whot = whot0 + warp * p1.W;
    const bool valid = row0 < R;
    const int p_wmax = dm.q_offset + min(row0 + 15, R - 1) / G;
    const int p_cmax = dm.q_offset + (min(crow0 + NW * 16, R) - 1) / G;
    const int r_lo = row0 + (lane >> 2), r_hi = r_lo + 8;
    const int lim_lo = CAUSAL ? min(dm.q_offset + min(r_lo, R - 1) / G, dm.Skv - 1) : dm.Skv - 1;
    const int lim_hi = CAUSAL ? min(dm.q_offset + min(r_hi, R - 1) / G, dm.Skv - 1) : dm.Skv - 1;
    auto q_row = [&](int r) {
        const int rr = min(r, R - 1);
        return Q + (((size_t)b * dm.Sq + rr / G) * dm.H + hk * G + rr % G) * 128 + T0 * 2;
    };
    uint32_t qa[8][4];
    load_q16(qa, q_row(r_lo), q_row(r_hi));
    float sh_lo, sh_hi;
    row_shift(qa, p1.ksum + (size_t)bhk * 128, qmul, T0, sh_lo, sh_hi);
    Rows2 w;
    w.o.zero();
    w.m_lo = w.m_hi = -INFINITY;
    w.l_lo = w.l_hi = 0.f;
    bool mine_any = false;

    if (tid == 0)
        wait_ready(p1.ready + bhk * nqb + qb);
    __syncthreads();
    const int lm_i = lane >> 3, lm_r = lane & 7;
    const int t_end = min(dm.ntkv, CAUSAL ? p_cmax / BKV + 1 : dm.ntkv);
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    for (int t = next_tile<NW>(whot0, p1.W, -1, t_end); t >= 0;
         t = next_tile<NW>(whot0, p1.W, t, t_end), ++u) {
        const uint32_t s = u % P2S_STG, ph = (u / P2S_STG) & 1;
        const uint32_t fb = (uint32_t)__cvta_generic_to_shared(&ps.full[s]);
        for (;;) {  // warp-uniform: lane 0 tests, the warp loads stages meanwhile
            if (__shfl_sync(~0u, lane == 0 ? (int)mbar_test(fb, ph) : 0, 0))
                break;
            p2s_refill<NW>(ps, smem, rs, whot0, p1.W, t_end, dm, b, hk, lane);
        }
        mbar_wait(fb, ph);  // acquire in every lane (phase already complete)
        const int j0 = t * BKV;
        if (valid && (!CAUSAL || j0 <= p_wmax) && ((whot[t >> 5] >> (t & 31)) & 1u)) {
            mine_any = true;
            float sc[8][4];
            qk16(sc, qa, sbase + s * P2_SMEM, lm_i, lm_r);
            float mx_lo = w.m_lo, mx_hi = w.m_hi;
            scale_mask16(sc, qmul, sh_lo, sh_hi, j0 + T0 * 2, lim_lo, lim_hi, mx_lo, mx_hi);
            softmax16(w, sc, mx_lo, mx_hi);
            pv16(w, sc, sbase + s * P2_SMEM + BKV * P2_ROWB, lm_i, lm_r);
        }
        mbar_arrive((uint32_t)__cvta_generic_to_shared(&ps.empty[s]));
        p2s_refill<NW>(ps, smem, rs, whot0, p1.W, t_end, dm, b, hk, lane);
    }
    if (!valid || !mine_any)
        return;
    w.l_lo += __shfl_xor_sync(~0u, w.l_lo, 1);
    w.l_lo += __shfl_xor_sync(~0u, w.l_lo, 2);
    w.l_hi += __shfl_xor_sync(~0u, w.l_hi, 1);
    w.l_hi += __shfl_xor_sync(~0u, w.l_hi, 2);
    if (r_lo < R)
        merge_row(w, 0, p1, O, dm, b, hk, r_lo, T0);
    if (r_hi < R)
        merge_row(w, 1, p1, O, dm, b, hk, r_hi, T0);
}

// Both passes in one launch (hd 128, 12 warps). Each CTA takes a ticket: tickets < n1 run pass 1 on a q
// block, the rest run pass 2 on whole pass-1 q blocks (pass2_cta). Tickets go to running CTAs in order, so
// the pass-1 item a pass-2 CTA waits on is resident (no deadlock); pass 2 fills the SMs of pass 1's last
// wave.
template <bool CAUSAL, typename T, typename RS>
__global__ void __launch_bounds__(384, 1) apa_kernel(
    const uint8_t* __restrict__ Qq, const uint8_t* __restrict__ Qs, const float* __restrict__ Qr,
    const uint8_t* __restrict__ KV, const HeadScale* __restrict__ hs, const T* __restrict__ Q, RS rs,
    T* __restrict__ O, Dims dm, float qmul, Pass1Out p1, int n1, int nqb, uint32_t* ticket) {
    constexpr int NW = 12;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ int item_s;
    if (threadIdx.x == 0)
        item_s = (int)atomicAdd(ticket, 1u);
    __syncthreads();
    const int item = item_s, i = item < n1 ? item : item - n1;
    const int qx = i % nqb, hk = (i / nqb) % dm.Hkv, b = i / (nqb * dm.Hkv);
    const int qb = CAUSAL ? nqb - 1 - qx : qx;  // heavy causal blocks first
    if (item < n1) {
        pass1_cta<128, NW, 6, CAUSAL, T>(Qq, Qs, Qr, KV, hs, O, dm, p1, b, hk, qb, nqb, smem);
        return;
    }
    // pass-2 worker: whole q blocks from ticket[1], all 12 warps on one tile stream
    __shared__ P2Stream ps;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < P2S_STG; ++s) {
            mbar_init((uint32_t)__cvta_generic_to_shared(&ps.full[s]), 32);
            mbar_init((uint32_t)__cvta_generic_to_shared(&ps.empty[s]), NW * 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    uint32_t u = 0;
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
        const int jx = j % nqb, jhk = (j / nqb) % dm.Hkv, jb = j / (nqb * dm.Hkv);
        pass2_cta<NW, CAUSAL, T>(Q, rs, O, dm, qmul, p1, jb, jhk, nqb, CAUSAL ? nqb - 1 - jx : jx, ps, smem,
                                 u);
    }
}

}  // namespace apa
