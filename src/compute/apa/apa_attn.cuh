// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// Vendored from kekzl/apa include/apa/apa_attn.cuh (v0.6.1, 015301f); change there first, then copy.
// APA pass 1: all-FP4 flash attention that diverts hot tiles (mass share > eps of the running row sum) to
// pass 2 (exact FP16). Cold tiles finish here; warps with hot tiles export (o, m, l) for the merge.
#pragma once
#include "apa_common.cuh"

namespace apa {

// Pass-1 outputs. Masks: one bit per KV tile, W words per warp (warp id = (bhk * nqb + qb) * NW + warp)
// (zeroed before pass 1). part/ml: packed rows (bhk * R + r).
struct Pass1Out {
    float eps;
    int W;
    uint32_t* warp_hot;
    uint32_t* ready;  // [bhk * nqb + qb] != 0: pass-1 CTA finished (release); pass 2 acquires it
    float* part;      // [rows][D]: sum over cold tiles of p * v, p relative to m
    float2* ml;       // [rows]: (m, sum over cold tiles of p), log2 domain
    const float*
        ksum;  // [bhk][D] K column mean (finalized): pass 1 scores are q.(k - kmean), pass 2 must match
#ifdef APA_DBG
    float4* dbg;  // [rows]: (m, lambda, l_cold, max tile share at test time), lambda / l_cold in ml units
#endif
};

// Lazy row max: m may lag the true row max by up to TAU (-DAPA_EXACT_MAX: exact).
#ifndef APA_EXACT_MAX
constexpr float TAU = 4.f;
#else
constexpr float TAU = 0.f;
#endif
// P' = P * 1536 / 2^k per 16-group, 2^k >= 256 * groupmax(P): P' <= 6, 2^k in UE4M3.
constexpr float LOG2_PS = 10.584962500721156f;                // log2(1536)
constexpr uint32_t ONES = 0x22222222u, SF_ONE = 0x38383838u;  // E2M1 1.0 x8, UE4M3 1.0 x4

// Per-warp running state of its two lane rows (lo = T1, hi = T1 + 8).
template <int DT>
struct Rows {
    float o[DT][4];
    float ol[4];         // P.1 over cold tiles (row sums via a ones tile)
    float m_lo, m_hi;    // running max, log2 domain
    float la_lo, la_hi;  // running row sum over ALL tiles (hot ones too)
#ifdef APA_DBG
    float ms_lo, ms_hi;  // max tile share l_t / (lambda + l_t) at test time
#endif
};

template <int D>
__device__ __forceinline__ void load_q(uint32_t (&qa)[Cfg<D>::KSTEPS][4], uint32_t (&qsf)[Cfg<D>::KSTEPS],
                                       const uint8_t* Qq, const uint8_t* Qs, size_t q_lo, size_t q_hi,
                                       int T0) {
#pragma unroll
    for (int ks = 0; ks < Cfg<D>::KSTEPS; ++ks) {
        const uint8_t* lo = Qq + q_lo * (D / 2) + ks * 32 + T0 * 4;
        const uint8_t* hi = Qq + q_hi * (D / 2) + ks * 32 + T0 * 4;
        qa[ks][0] = *reinterpret_cast<const uint32_t*>(lo);
        qa[ks][1] = *reinterpret_cast<const uint32_t*>(hi);
        qa[ks][2] = *reinterpret_cast<const uint32_t*>(lo + 16);
        qa[ks][3] = *reinterpret_cast<const uint32_t*>(hi + 16);
        qsf[ks] = *reinterpret_cast<const uint32_t*>(Qs + ((T0 & 1) ? q_hi : q_lo) * (D / 16) + ks * 4);
    }
}

// S = Q K^T (raw; log2 score = s * c), all K fragments + scales of the tile preloaded (hd 128). Q enters as
// two E2M1 terms (qa, qa2: the residual of the first in the same row scale), one accumulator: FP4 rounding of
// Q decided the worst rows (AUDIT.md, Phase 6; lc_65536_0 min cos 0.773 -> 0.985). K scales: lane T1 reads
// D/2 contiguous bytes = rows nt*8+T1 (D/16 B each); word nt*KST + ks.
template <int D>
__device__ __forceinline__ void qk_tile(float (&s)[8][4], const uint32_t (&qa)[Cfg<D>::KSTEPS][4],
                                        const uint32_t (&qsf)[Cfg<D>::KSTEPS], uint32_t stage,
                                        const uint8_t* sgen, const uint32_t (&k_lane)[Cfg<D>::CPR / 4],
                                        int T1, const uint32_t (&qa2)[Cfg<D>::KSTEPS][4],
                                        const uint32_t (&qsf2)[Cfg<D>::KSTEPS]) {
    using C = Cfg<D>;
    constexpr int KST = C::KSTEPS;
    const uint8_t* ksb = sgen + C::K_BYTES + C::V_BYTES + T1 * (D / 2);
    uint32_t kf[8][2 * KST], ksw[8 * KST];
#pragma unroll
    for (int i = 0; i < 8 * KST / 4; ++i)
        *reinterpret_cast<uint4*>(ksw + 4 * i) = *reinterpret_cast<const uint4*>(ksb + 16 * i);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int gq = 0; gq < C::CPR / 4; ++gq)
            ldsm_x4(kf[nt] + 4 * gq, stage + k_lane[gq] + nt * 8 * C::RB);
    }
#pragma unroll
    for (int ks = 0; ks < KST; ++ks)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            mma_fp4(s[nt], qa[ks], kf[nt][2 * ks], kf[nt][2 * ks + 1], qsf[ks], ksw[nt * KST + ks]);
#pragma unroll
    for (int ks = 0; ks < KST; ++ks)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            mma_fp4(s[nt], qa2[ks], kf[nt][2 * ks], kf[nt][2 * ks + 1], qsf2[ks], ksw[nt * KST + ks]);
}

// Causal diagonal / KV tail: col > lim -> -inf.
__device__ __forceinline__ void mask_tile(float (&s)[8][4], int col0, int lim_lo, int lim_hi) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            if (col0 + nt * 8 + (e & 1) > ((e >> 1) ? lim_hi : lim_lo))
                s[nt][e] = -INFINITY;
}

// Raw group maxima: gm[0]=lo h0, [1]=hi h0, [2]=lo h1, [3]=hi h1; pair-reduced = 16-col groups.
__device__ __forceinline__ void group_max(const float (&s)[8][4], float (&gm)[4]) {
    gm[0] = gm[1] = gm[2] = gm[3] = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int h = (nt >> 2) * 2;
        gm[h] = fmaxf(gm[h], fmaxf(s[nt][0], s[nt][1]));
        gm[h + 1] = fmaxf(gm[h + 1], fmaxf(s[nt][2], s[nt][3]));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        gm[i] = fmaxf(gm[i], __shfl_xor_sync(~0u, gm[i], 1));
}

// Hot tiles lift the cold frame m to at most (tile max - HOT_DROP), log2: a hot sink does not push cold P
// below the UE4M3 scale floor (E2M1 zero). 0 = frame always at the running max (APA 0.2.0).
#ifndef APA_HOT_DROP
#define APA_HOT_DROP 32
#endif
constexpr float HOT_DROP = APA_HOT_DROP;

// Candidate row max: full reduction only when a lane exceeds m + TAU; no rescale.
template <int DT>
__device__ __forceinline__ void cand_max(const Rows<DT>& w, float mx_lo, float mx_hi, float& c_lo,
                                         float& c_hi) {
    c_lo = w.m_lo;
    c_hi = w.m_hi;
    if (TAU > 0.f && !__any_sync(~0u, mx_lo > w.m_lo + TAU || mx_hi > w.m_hi + TAU))
        return;
    c_lo = fmaxf(w.m_lo, fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 2)));
    c_hi = fmaxf(w.m_hi, fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 2)));
}

// Move o, ol, la to frame mx >= m (scale 2^(m - mx)).
template <int DT>
__device__ __forceinline__ void update_max(Rows<DT>& w, float mx_lo, float mx_hi) {
    if (!__any_sync(~0u, mx_lo != w.m_lo || mx_hi != w.m_hi))
        return;
    const float a_lo = (mx_lo == w.m_lo) ? 1.f : ex2(w.m_lo - mx_lo);
    const float a_hi = (mx_hi == w.m_hi) ? 1.f : ex2(w.m_hi - mx_hi);
#pragma unroll
    for (int i = 0; i < DT; ++i) {
        w.o[i][0] *= a_lo;
        w.o[i][1] *= a_lo;
        w.o[i][2] *= a_hi;
        w.o[i][3] *= a_hi;
    }
    w.ol[0] *= a_lo;
    w.ol[1] *= a_lo;
    w.ol[2] *= a_hi;
    w.la_lo *= a_lo;
    w.la_hi *= a_hi;
    w.ol[3] *= a_hi;
    w.m_lo = mx_lo;
    w.m_hi = mx_hi;
}

// Per-group power-of-two P scale folded into the exponent. Clamp u to [-6, 8]: 8 because fmaf skips the
// rounding of m = gm*c (row-max group reads 8+eps); -6 keeps 2^k a normal UE4M3 ((k+7)<<3).
__device__ __forceinline__ void p_scales(const float (&gm)[4], float m_lo, float m_hi, float c_lo, float c_hi,
                                         float (&bias)[4], uint32_t (&kb)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float m = (i & 1) ? m_hi : m_lo;
        const float u = fminf(fmaxf(fmaf(gm[i], (i & 1) ? c_hi : c_lo, (8.f - TAU) - m), -6.f), 8.f);
        // scale 2^floor(u) * (1 + mi / 8) >= 2^u, mi = ceil((2^frac - 1) * 8) in 0..8 (8 carries into the
        // exponent code): the group maximum lands near E2M1 6, not anywhere in (3, 6] as with a power of two
        // (AUDIT.md, Phase 7)
        const float vd = __fadd_rd(u, 12582919.f);  // floor(u) + 7 in the low mantissa bits
        const float fl = vd - 12582919.f;
        const float mm = __fadd_ru((ex2(u - fl) - 1.f) * 8.f, 12582912.f);
        kb[i] = ((__float_as_uint(vd) - 0x4B400000u) << 3) + (__float_as_uint(mm) - 0x4B400000u);
        bias[i] = (LOG2_PS - TAU - m) - (fl + lg2(1.f + (mm - 12582912.f) * 0.125f));
    }
}

__device__ __forceinline__ void exp_scores(float (&s)[8][4], float c_lo, float c_hi, const float (&bias)[4]) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int h = (nt >> 2) * 2;
        s[nt][0] = ex2(fmaf(s[nt][0], c_lo, bias[h]));
        s[nt][1] = ex2(fmaf(s[nt][1], c_lo, bias[h]));
        s[nt][2] = ex2(fmaf(s[nt][2], c_hi, bias[h + 1]));
        s[nt][3] = ex2(fmaf(s[nt][3], c_hi, bias[h + 1]));
    }
}

// P into the A fragment (slot j of a0 = s[j/2][j&1], see slot_token).
__device__ __forceinline__ void pack_p(const float (&s)[8][4], uint32_t (&pa)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        float f[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            f[j] = s[(q >> 1) * 4 + (j >> 1)][(q & 1) * 2 + (j & 1)];
        pa[q] = pack8_e2m1(f);
    }
}

// Scale word: lanes with T0 even provide row lo, odd row hi; bytes = groups 0..3.
__device__ __forceinline__ uint32_t p_scale_word(const uint32_t (&kb)[4], int T0) {
    const uint32_t mine = (T0 & 1) ? (kb[1] | (kb[3] << 8)) : (kb[0] | (kb[2] << 8));
    const uint32_t other = __shfl_xor_sync(~0u, mine, 2);
    const uint32_t a = (T0 & 2) ? other : mine, b = (T0 & 2) ? mine : other;
    return __byte_perm(a, b, 0x5140);  // [a0, b0, a1, b1] = groups 0..3
}

// O += P V. V scales: lane T1 reads D/2 contiguous bytes = channel rows dt*8+T1 (4 B each).
template <int D>
__device__ __forceinline__ void pv_tile(float (&o)[Cfg<D>::DT][4], const uint32_t (&pa)[4], uint32_t psf,
                                        uint32_t stage, const uint8_t* sgen, uint32_t v_lane, int T1) {
    using C = Cfg<D>;
    uint32_t vs[C::DT];
#pragma unroll
    for (int i = 0; i < C::DT / 4; ++i)
        *reinterpret_cast<uint4*>(vs + 4 * i) = *reinterpret_cast<const uint4*>(
            sgen + C::K_BYTES + C::V_BYTES + C::KS_BYTES + T1 * (D / 2) + i * 16);
#pragma unroll
    for (int dp = 0; dp < C::DT / 2; ++dp) {
        uint32_t vb[4];
        ldsm_x4(vb, stage + v_lane + dp * 512);
        mma_fp4(o[dp * 2], pa, vb[0], vb[1], psf, vs[dp * 2]);
        mma_fp4(o[dp * 2 + 1], pa, vb[2], vb[3], psf, vs[dp * 2 + 1]);
    }
}

// Per-warp tile geometry and Q (constant over the KV loop).
template <int D>
struct WarpQ {
    uint32_t qa[Cfg<D>::KSTEPS][4], qsf[Cfg<D>::KSTEPS];
    uint32_t qa2[Cfg<D>::KSTEPS][4], qsf2[Cfg<D>::KSTEPS];  // residual Q term
    uint32_t k_lane[Cfg<D>::CPR / 4], v_lane;
    float c_lo, c_hi;  // log2 score = s * c (per row)
    int T0, T1, pos_lo, pos_hi, p_wmin;
};

// One active tile of one warp: QK, mask, online max, P, hot test (tile share of the running row sum la
// over all tiles; la only grows, so a cold verdict stays valid); cold tiles add P V. Returns hot.
template <int D, bool CAUSAL>
__device__ __forceinline__ bool tile_step(Rows<Cfg<D>::DT>& w, const WarpQ<D>& q, uint32_t stage,
                                          const uint8_t* sgen, int j0, const Dims& dm, float eps) {
    float s[8][4];
    qk_tile<D>(s, q.qa, q.qsf, stage, sgen, q.k_lane, q.T1, q.qa2, q.qsf2);
    if ((CAUSAL && j0 + BKV - 1 > q.p_wmin) || j0 + BKV > dm.Skv) {
        const int lim_lo = CAUSAL ? min(q.pos_lo, dm.Skv - 1) : dm.Skv - 1;
        const int lim_hi = CAUSAL ? min(q.pos_hi, dm.Skv - 1) : dm.Skv - 1;
        mask_tile(s, j0 + q.T0 * 2, lim_lo, lim_hi);
    }
    float gm[4], c_lo, c_hi;
    group_max(s, gm);
    cand_max(w, fmaxf(gm[0], gm[2]) * q.c_lo, fmaxf(gm[1], gm[3]) * q.c_hi, c_lo, c_hi);
    float bias[4];
    uint32_t kb[4], pa[4];
    p_scales(gm, c_lo, c_hi, q.c_lo, q.c_hi, bias, kb);
    exp_scores(s, q.c_lo, q.c_hi, bias);
    pack_p(s, pa);
    const uint32_t psf = p_scale_word(kb, q.T0);
    float tl[4] = {0.f, 0.f, 0.f, 0.f};  // tile row sums from P.1 (frame c): every lane holds its rows' sums
    mma_fp4(tl, pa, ONES, ONES, psf, SF_ONE);
    const float a_lo = (c_lo == w.m_lo) ? 1.f : ex2(w.m_lo - c_lo),
                a_hi = (c_hi == w.m_hi) ? 1.f : ex2(w.m_hi - c_hi);
#ifdef APA_DBG
    w.ms_lo = fmaxf(w.ms_lo, tl[0] / (w.la_lo * a_lo + tl[0]));  // 0/0 (masked row) is NaN: fmaxf keeps ms
    w.ms_hi = fmaxf(w.ms_hi, tl[2] / (w.la_hi * a_hi + tl[2]));
#endif
    if (__any_sync(~0u, tl[0] > eps * (w.la_lo * a_lo + tl[0]) || tl[2] > eps * (w.la_hi * a_hi + tl[2]))) {
        const float f_lo = fmaxf(w.m_lo, c_lo - HOT_DROP), f_hi = fmaxf(w.m_hi, c_hi - HOT_DROP);
        const float b_lo = (f_lo == c_lo) ? 1.f : ex2(c_lo - f_lo),
                    b_hi = (f_hi == c_hi) ? 1.f : ex2(c_hi - f_hi);
        update_max(w, f_lo, f_hi);
        w.la_lo += tl[0] * b_lo;  // up to 2^HOT_DROP x tile sum: fp32 range
        w.la_hi += tl[2] * b_hi;
        return true;
    }
    update_max(w, c_lo, c_hi);
    w.la_lo += tl[0];
    w.la_hi += tl[2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        w.ol[i] += tl[i];
    pv_tile<D>(w.o, pa, psf, stage, sgen, q.v_lane, q.T1);
    return false;
}

// Epilogue of a valid warp: with hot tiles export the cold partials (p units; o and ol carry P_SCALE 1536
// and the lazy-max offset 2^-TAU, undone here), else normalize and store O.
template <int D, typename OutT>
__device__ __forceinline__ void p1_epilogue(const Rows<Cfg<D>::DT>& w, bool any_hot, const Pass1Out& out,
                                            OutT* O, const Dims& dm, int b, int hk, float scv, int r_lo,
                                            int T0) {
    constexpr int DT = Cfg<D>::DT;
    const int R = dm.R, G = dm.G, r_hi = r_lo + 8;
    const bool ok_lo = r_lo < R, ok_hi = r_hi < R;
    const size_t qb_row = (size_t)(b * dm.Hkv + hk) * R;
#ifdef APA_DBG
    if (T0 == 0 && out.dbg != nullptr) {
        constexpr float u = (TAU > 0.f ? 16.f : 1.f) / 1536.f;
        if (ok_lo)
            out.dbg[qb_row + r_lo] = make_float4(w.m_lo, w.la_lo * u, w.ol[0] * u, w.ms_lo);
        if (ok_hi)
            out.dbg[qb_row + r_hi] = make_float4(w.m_hi, w.la_hi * u, w.ol[2] * u, w.ms_hi);
    }
#endif
    if (any_hot) {
        constexpr float kUnscale = (TAU > 0.f ? 16.f : 1.f) / 1536.f;
        const float f = scv * kUnscale;
        float* p_lo = out.part + (qb_row + min(r_lo, R - 1)) * D + T0 * 2;
        float* p_hi = out.part + (qb_row + min(r_hi, R - 1)) * D + T0 * 2;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            if (ok_lo)
                *reinterpret_cast<float2*>(p_lo + dt * 8) = make_float2(w.o[dt][0] * f, w.o[dt][1] * f);
            if (ok_hi)
                *reinterpret_cast<float2*>(p_hi + dt * 8) = make_float2(w.o[dt][2] * f, w.o[dt][3] * f);
        }
        if (T0 == 0 && ok_lo)
            out.ml[qb_row + r_lo] = make_float2(w.m_lo, w.ol[0] * kUnscale);
        if (T0 == 0 && ok_hi)
            out.ml[qb_row + r_hi] = make_float2(w.m_hi, w.ol[2] * kUnscale);
        return;
    }
    const float f_lo = scv / w.ol[0], f_hi = scv / w.ol[2];
    // packed row r -> O[b][r / G][hk*G + r % G][:]
    auto out_row = [&](int r) {
        return O + (((size_t)b * dm.Sq + r / G) * dm.H + hk * G + r % G) * D + T0 * 2;
    };
    OutT* o_lo = out_row(ok_lo ? r_lo : 0);
    OutT* o_hi = out_row(ok_hi ? r_hi : 0);
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        if (ok_lo)
            store2(o_lo + dt * 8, w.o[dt][0] * f_lo, w.o[dt][1] * f_lo);
        if (ok_hi)
            store2(o_hi + dt * 8, w.o[dt][2] * f_hi, w.o[dt][3] * f_hi);
    }
}

// Pass-1 tile order (APA_ORDER 1): sink tile 0, then the diagonal backwards, so the heavy tiles enter the
// running row sum first and the hot test judges middle tiles against a near-final sum. 0 = ascending.
#ifndef APA_ORDER
#define APA_ORDER 1
#endif
__device__ __forceinline__ int tile_at(int i, int ntiles) {
    return (APA_ORDER == 0 || i == 0) ? i : ntiles - i;
}

// Bulk-copy pipeline (thread 0 = producer): STAGES slots, full/empty mbarriers.
template <int NW, int STAGES, int TILE>
__device__ __forceinline__ void pipe_init(uint32_t full0, uint32_t empty0, uint32_t sbase, const uint8_t* kv,
                                          int ntiles) {
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(full0 + s * 8, 1);
            mbar_init(empty0 + s * 8, NW * 32);  // every lane arrives: no warp sync / lane branch
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES && s < ntiles; ++s) {
            mbar_expect_tx(full0 + s * 8, TILE);
            bulk_g2s(sbase + s * TILE, kv + (size_t)tile_at(s, ntiles) * TILE, TILE, full0 + s * 8);
        }
    }
}

template <int D>
__device__ __forceinline__ void warp_q_init(WarpQ<D>& q, const uint8_t* Qq, const uint8_t* Qs,
                                            const float* Qr, float sck, size_t q_lo, size_t q_hi, int lane) {
    using C = Cfg<D>;
    q.T0 = lane & 3;
    q.T1 = lane >> 2;
    load_q<D>(q.qa, q.qsf, Qq, Qs, q_lo, q_hi, q.T0);
    q.c_lo = Qr[q_lo] * sck;
    q.c_hi = Qr[q_hi] * sck;
    // ldmatrix lane addressing: the swizzle term depends only on lane bits, not on the n-tile index.
    // K: x4 group gq covers chunks 4gq..4gq+3 of a row (2 k-steps).
    const int lm_i = lane >> 3, lm_r = lane & 7;
#pragma unroll
    for (int gq = 0; gq < C::CPR / 4; ++gq)
        q.k_lane[gq] = lm_r * C::RB + k_swz<D>(lm_r, 4 * gq + lm_i) * 16;
    q.v_lane = C::K_BYTES + ((lm_i >> 1) * 8 + lm_r) * 32 + v_swz(lm_r, lm_i & 1) * 16;  // + dp * 512
}

// One CTA: BQ packed rows (q block qb) of one (b, kv head); K/V tiles shared by its G q heads.
template <int D, int NW, int STAGES, bool CAUSAL, typename OutT>
__device__ __forceinline__ void pass1_cta(const uint8_t* __restrict__ Qq, const uint8_t* __restrict__ Qs,
                                          const float* __restrict__ Qr, const uint8_t* __restrict__ KV,
                                          const HeadScale* __restrict__ hs, OutT* __restrict__ O,
                                          const Dims& dm, const Pass1Out& out, int b, int hk, int qb, int nqb,
                                          uint8_t* smem) {
    static_assert(D == 128, "APA pass 1: hd 128 (K fragments preloaded, x4 ldmatrix)");
    using C = Cfg<D>;
    constexpr int BQ = NW * 16, TILE = C::TILE, DT = C::DT;
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];

    const int bhk = b * dm.Hkv + hk, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int R = dm.R, G = dm.G;
    const int row0 = qb * BQ + warp * 16;  // packed row of this warp's first m16 row
    const bool valid = row0 < R;           // warps past the end only release stages
    const int p_wmax = dm.q_offset + min(row0 + 15, R - 1) / G;
    const int p_cmax = dm.q_offset + (min((qb + 1) * BQ, R) - 1) / G;  // last position in the CTA
    const int ntiles = CAUSAL ? min((p_cmax + BKV) / BKV, dm.ntkv) : dm.ntkv;
    const uint8_t* kv = KV + (size_t)bhk * dm.kvcap * TILE;
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(full);
    const uint32_t empty0 = (uint32_t)__cvta_generic_to_shared(empty);
    pipe_init<NW, STAGES, TILE>(full0, empty0, sbase, kv, ntiles);

    const int r_lo = row0 + (lane >> 2);  // this lane's packed rows: r_lo, r_lo + 8
    const size_t qb_row = (size_t)bhk * R;
    const size_t q_lo = qb_row + min(r_lo, R - 1), q_hi = qb_row + min(r_lo + 8, R - 1);  // clamped for loads
    const HeadScale sc = hs[bhk];
    WarpQ<D> q;
    warp_q_init<D>(q, Qq, Qs, Qr, sc.k, q_lo, q_hi, lane);
    const size_t qrows = (size_t)dm.B * dm.Hkv * dm.R;  // residual Q term after the first in Qq / Qs
    load_q<D>(q.qa2, q.qsf2, Qq + qrows * (D / 2), Qs + qrows * (D / 16), q_lo, q_hi, q.T0);
    q.pos_lo = dm.q_offset + r_lo / G;
    q.pos_hi = dm.q_offset + (r_lo + 8) / G;
    q.p_wmin = dm.q_offset + min(row0, R - 1) / G;

    Rows<DT> w;
#pragma unroll
    for (int i = 0; i < DT; ++i)
        w.o[i][0] = w.o[i][1] = w.o[i][2] = w.o[i][3] = 0.f;
    w.ol[0] = w.ol[1] = w.ol[2] = w.ol[3] = 0.f;
    w.m_lo = w.m_hi = -INFINITY;
    w.la_lo = w.la_hi = 0.f;
#ifdef APA_DBG
    w.ms_lo = w.ms_hi = 0.f;
#endif
    bool any_hot = false;  // warp-uniform
    uint32_t* whot = out.warp_hot + ((size_t)(bhk * nqb + qb) * NW + warp) * out.W;

    for (int i = 0; i < ntiles; ++i) {
        const int st = i % STAGES, t = tile_at(i, ntiles);
        const uint32_t stage = sbase + st * TILE;
        mbar_wait(full0 + st * 8, (i / STAGES) & 1);
        if (valid && (!CAUSAL || t * BKV <= p_wmax) &&  // warp-uniform
            tile_step<D, CAUSAL>(w, q, stage, smem + st * TILE, t * BKV, dm, out.eps)) {
            if (lane == 0)
                atomicOr(whot + (t >> 5), 1u << (t & 31));
            any_hot = true;
        }
        mbar_arrive(empty0 + st * 8);
        if (threadIdx.x == 0 && i + STAGES < ntiles) {
            mbar_wait(empty0 + st * 8, (i / STAGES) & 1);
            mbar_expect_tx(full0 + st * 8, TILE);
            bulk_g2s(stage, kv + (size_t)tile_at(i + STAGES, ntiles) * TILE, TILE, full0 + st * 8);
        }
    }

    if (valid)
        p1_epilogue<D, OutT>(w, any_hot, out, O, dm, b, hk, sc.v, r_lo, q.T0);
    // publish this CTA: every thread fences its writes (O, part, ml, masks), then one release store
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        asm volatile("st.release.gpu.global.u32 [%0], 1;" ::"l"(out.ready + bhk * nqb + qb) : "memory");
}

}  // namespace apa
