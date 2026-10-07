// RA2 attention kernel: packed GQA rows, causal with q_offset, any Sq/Skv, output [B][Sq][H][D].
// Vendored from the ra2 standalone repo (kekzl/ra2); change there first, then copy.
// Per KV tile: S = Q K^T on int8 MMA, online softmax with per-16 power-of-two P scales, O += P (V4 + R4).
#pragma once
#include "ra2_common.cuh"

namespace ra2 {

#ifndef RA2_EXACT_MAX
constexpr float kTau = 4.f;  // row max may lag by up to 4 (log2): reduction + rescale only past that
#else
constexpr float kTau = 0.f;
#endif
constexpr float kLog2Ps = 10.584962500721156f;  // log2(1536): P' = P * 1536 / 2^k <= 6

// Q int8 rows (D bytes): k32 step ks = bytes ks*32 + T0*4 (a0/a1) and +16 (a2/a3), as the FP4 layout.
template <int D>
__device__ __forceinline__ void load_q(uint32_t (&qa)[Cfg<D>::KSTEPS][4], const uint8_t* lo,
                                       const uint8_t* hi) {
#pragma unroll
    for (int ks = 0; ks < Cfg<D>::KSTEPS; ++ks) {
        qa[ks][0] = *reinterpret_cast<const uint32_t*>(lo + ks * 32);
        qa[ks][1] = *reinterpret_cast<const uint32_t*>(hi + ks * 32);
        qa[ks][2] = *reinterpret_cast<const uint32_t*>(lo + ks * 32 + 16);
        qa[ks][3] = *reinterpret_cast<const uint32_t*>(hi + ks * 32 + 16);
    }
}

// S = Q K^T for one tile in int32. K fragment of n-tile nt: x4 group gq covers chunks 4gq..4gq+3.
template <int D>
__device__ __forceinline__ void qk_tile(int (&s)[8][4], const uint32_t (&qa)[Cfg<D>::KSTEPS][4],
                                        uint32_t stage, const uint32_t* k_lane) {
    using C = Cfg<D>;
    constexpr int KST = C::KSTEPS, NG = C::CPR / 4;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
        s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0;
    if constexpr (D <= 64) {  // all K fragments first: 8 independent accumulator chains per k-step
        uint32_t kf[8][2 * KST];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int gq = 0; gq < NG; ++gq)
                ldsm_x4(kf[nt] + 4 * gq, stage + k_lane[gq] + nt * 8 * C::RB);
#pragma unroll
        for (int ks = 0; ks < KST; ++ks)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt)
                mma_s8(s[nt], qa[ks], kf[nt][2 * ks], kf[nt][2 * ks + 1]);
    } else {  // register budget: one n-tile at a time
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            uint32_t kf[2 * KST];
#pragma unroll
            for (int gq = 0; gq < NG; ++gq)
                ldsm_x4(kf + 4 * gq, stage + k_lane[gq] + nt * 8 * C::RB);
#pragma unroll
            for (int ks = 0; ks < KST; ++ks)
                mma_s8(s[nt], qa[ks], kf[2 * ks], kf[2 * ks + 1]);
        }
    }
}

// Element (nt, e) of this lane sits at column col0 + nt*8 + (e&1) (col0 = j0 + T0*2), row lo (e<2) or hi.
__device__ __forceinline__ bool is_masked(int col0, int nt, int e, int lim_lo, int lim_hi) {
    return col0 + nt * 8 + (e & 1) > ((e >> 1) ? lim_hi : lim_lo);
}
__device__ __forceinline__ void mask_scores(int (&s)[8][4], int col0, int lim_lo, int lim_hi) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            if (is_masked(col0, nt, e, lim_lo, lim_hi))
                s[nt][e] = MASKED;
}
__device__ __forceinline__ void mask_probs(float (&p)[8][4], int col0, int lim_lo, int lim_hi) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            if (is_masked(col0, nt, e, lim_lo, lim_hi))
                p[nt][e] = 0.f;
}

// Group maxima of the raw scores: gm[0]=lo h0, [1]=hi h0, [2]=lo h1, [3]=hi h1 (pair-reduced = 16 columns).
__device__ __forceinline__ void group_max(const int (&s)[8][4], float (&gm)[4]) {
    int gi[4] = {MASKED, MASKED, MASKED, MASKED};
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int h = (nt >> 2) * 2;
        gi[h] = max(gi[h], max(s[nt][0], s[nt][1]));
        gi[h + 1] = max(gi[h + 1], max(s[nt][2], s[nt][3]));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        gm[i] = i2f(max(gi[i], __shfl_xor_sync(~0u, gi[i], 1)));
}

// Lazy online-softmax max: reduce and rescale O only when a lane's partial row max passes m + kTau.
template <int DT>
__device__ __forceinline__ void update_max(float mx_lo, float mx_hi, float& m_lo, float& m_hi,
                                           float (&o)[DT][4], float (&ol)[4]) {
    if (kTau > 0.f && !__any_sync(~0u, mx_lo > m_lo + kTau || mx_hi > m_hi + kTau))
        return;
    mx_lo = fmaxf(m_lo, fmaxf(mx_lo, __shfl_xor_sync(~0u, mx_lo, 2)));
    mx_hi = fmaxf(m_hi, fmaxf(mx_hi, __shfl_xor_sync(~0u, mx_hi, 2)));
    if (!__any_sync(~0u, mx_lo != m_lo || mx_hi != m_hi))
        return;
    const float a_lo = (mx_lo == m_lo) ? 1.f : ex2(m_lo - mx_lo);  // -inf - finite -> 0
    const float a_hi = (mx_hi == m_hi) ? 1.f : ex2(m_hi - mx_hi);
#pragma unroll
    for (int i = 0; i < DT; ++i) {
        o[i][0] *= a_lo;
        o[i][1] *= a_lo;
        o[i][2] *= a_hi;
        o[i][3] *= a_hi;
    }
    ol[0] *= a_lo;
    ol[1] *= a_lo;
    ol[2] *= a_hi;
    ol[3] *= a_hi;
    m_lo = mx_lo;
    m_hi = mx_hi;
}

// Per 16-group power-of-two P scale 2^k folded into the exponent; kb = UE4M3 bits (k+7)<<3.
// u clamps to [-6, 8]: 8 because fmaf skips the rounding of m = gm*c; -6 keeps 2^k normal (groups below
// 2^-14 of the row max quantize coarser, still <= 6).
__device__ __forceinline__ void p_scales(const float (&gm)[4], float m_lo, float m_hi, float c_lo, float c_hi,
                                         float (&bias)[4], uint32_t (&kb)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float m = (i & 1) ? m_hi : m_lo;
        const float u = fminf(fmaxf(fmaf(gm[i], (i & 1) ? c_hi : c_lo, (8.f - kTau) - m), -6.f), 8.f);
        const float v = __fadd_ru(u, 12582919.f);  // 1.5*2^23 + 7: ceil(u) + 7 in the low mantissa bits
        kb[i] = (__float_as_uint(v) - 0x4B400000u) << 3;
        bias[i] = (kLog2Ps - kTau - m) - (v - 12582919.f);
    }
}

__device__ __forceinline__ void exp_scores(const int (&s)[8][4], float c_lo, float c_hi,
                                           const float (&bias)[4], float (&p)[8][4]) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int h = (nt >> 2) * 2;
        p[nt][0] = ex2(fmaf(i2f(s[nt][0]), c_lo, bias[h]));
        p[nt][1] = ex2(fmaf(i2f(s[nt][1]), c_lo, bias[h]));
        p[nt][2] = ex2(fmaf(i2f(s[nt][2]), c_hi, bias[h + 1]));
        p[nt][3] = ex2(fmaf(i2f(s[nt][3]), c_hi, bias[h + 1]));
    }
}

// P into the PV A fragment: slot j of a0 = p[j/2][j&1] (row lo, n-tiles 0..3), see slot_token.
__device__ __forceinline__ void pack_p(const float (&p)[8][4], uint32_t (&pa)[4]) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const int r = w & 1, nb = (w >> 1) * 4;
        float f[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            f[j] = p[nb + (j >> 1)][2 * r + (j & 1)];
        pa[w] = pack8_e2m1(f);
    }
}

// Scale word: lanes with T0 even provide row lo, odd row hi; bytes = groups 0..3.
__device__ __forceinline__ uint32_t p_scale_word(const uint32_t (&kb)[4], int T0) {
    const uint32_t mine = (T0 & 1) ? (kb[1] | (kb[3] << 8)) : (kb[0] | (kb[2] << 8));
    const uint32_t other = __shfl_xor_sync(~0u, mine, 2);
    const uint32_t a = (T0 & 2) ? other : mine, b = (T0 & 2) ? mine : other;
    return __byte_perm(a, b, 0x5140);  // [a0, b0, a1, b1]
}

// O += P (V4 + R4), ol += P 1. V scales: lane T1 reads D/2 contiguous bytes = channel rows dt*8+T1.
template <int D>
__device__ __forceinline__ void pv_tile(float (&o)[Cfg<D>::DT][4], float (&ol)[4], const uint32_t (&pa)[4],
                                        uint32_t psf, uint32_t stage, const uint8_t* sgen, uint32_t v_lane,
                                        int T1) {
    using C = Cfg<D>;
    constexpr int DT = C::DT;
#pragma unroll
    for (int term = 0; term < 2; ++term) {
        const int voff = term ? C::OFF_VR - C::OFF_V : 0, soff = term ? C::OFF_VSR : C::OFF_VS;
        uint32_t vs[DT];
#pragma unroll
        for (int i = 0; i < DT / 4; ++i)
            *reinterpret_cast<uint4*>(vs + 4 * i) = *reinterpret_cast<const uint4*>(sgen + soff +
                                                                                    T1 * (D / 2) + i * 16);
#pragma unroll
        for (int dp = 0; dp < DT / 2; ++dp) {
            uint32_t vb[4];
            ldsm_x4(vb, stage + v_lane + voff + dp * 512);
            mma_fp4(o[dp * 2], pa, vb[0], vb[1], psf, vs[dp * 2]);
            mma_fp4(o[dp * 2 + 1], pa, vb[2], vb[3], psf, vs[dp * 2 + 1]);
        }
    }
    constexpr uint32_t kOnes = 0x22222222u, kSfOne = 0x38383838u;  // E2M1 1.0 x8, UE4M3 1.0 x4
    mma_fp4(ol, pa, kOnes, kOnes, psf, kSfOne);
}

// Packed row r -> O[b][r / G][hk*G + r % G][:]; rows past R are skipped.
template <int D, typename OutT>
__device__ __forceinline__ void store_rows(OutT* O, const Dims& dm, int b, int hk, int r_lo, int T0,
                                           const float (&o)[Cfg<D>::DT][4], float f_lo, float f_hi) {
    const int G = dm.G, r_hi = r_lo + 8;
    auto out_row = [&](int r) {
        return O + (((size_t)b * dm.Sq + r / G) * dm.H + hk * G + r % G) * D + T0 * 2;
    };
    const bool ok_lo = r_lo < dm.R, ok_hi = r_hi < dm.R;
    OutT* o_lo = out_row(ok_lo ? r_lo : 0);
    OutT* o_hi = out_row(ok_hi ? r_hi : 0);
#pragma unroll
    for (int dt = 0; dt < Cfg<D>::DT; ++dt) {
        if (ok_lo)
            store2(o_lo + dt * 8, o[dt][0] * f_lo, o[dt][1] * f_lo);
        if (ok_hi)
            store2(o_hi + dt * 8, o[dt][2] * f_hi, o[dt][3] * f_hi);
    }
}

// grid (ceil(R/BQ), Hkv, B). Each CTA: BQ packed rows of one (b, kv head); K/V tiles shared by its G q heads.
template <int D, int NW, int STAGES, bool CAUSAL, typename OutT, int MB = 1>
__global__ void __launch_bounds__(NW * 32, MB) attn_kernel(const uint8_t* __restrict__ Qq,
                                                           const float* __restrict__ Qr,
                                                           const uint8_t* __restrict__ KV,
                                                           const HeadScale* __restrict__ hs,
                                                           OutT* __restrict__ O, Dims dm) {
    using C = Cfg<D>;
    constexpr int BQ = NW * 16, TILE = C::TILE, DT = C::DT, NG = C::CPR / 4;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];

    const int b = blockIdx.z, hk = blockIdx.y, bhk = b * dm.Hkv + hk;
    const int qb = CAUSAL ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;  // heavy causal blocks first
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int T0 = lane & 3, T1 = lane >> 2;
    const int R = dm.R, G = dm.G;
    const int row0 = qb * BQ + warp * 16;                   // packed row of this warp's first m16 row
    const bool valid = row0 < R;                            // warps past the end only release stages
    const int p_wmin = dm.q_offset + min(row0, R - 1) / G;  // positions covered by this warp
    const int p_wmax = dm.q_offset + min(row0 + 15, R - 1) / G;
    const int p_cmax = dm.q_offset + (min((qb + 1) * BQ, R) - 1) / G;  // last position in the CTA
    const int ntiles = CAUSAL ? min((p_cmax + BKV) / BKV, dm.ntkv) : dm.ntkv;
    const uint8_t* kv = KV + (size_t)bhk * dm.ntkv * TILE;
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(full);
    const uint32_t empty0 = (uint32_t)__cvta_generic_to_shared(empty);

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(full0 + s * 8, 1);
            mbar_init(empty0 + s * 8, NW * 32);  // every lane arrives: no warp sync / lane branch
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int s = 0; s < min(STAGES, ntiles); ++s) {
            mbar_expect_tx(full0 + s * 8, TILE);
            bulk_g2s(sbase + s * TILE, kv + (size_t)s * TILE, TILE, full0 + s * 8);
        }
    }

    const int r_lo = row0 + T1, r_hi = r_lo + 8;  // this lane's packed rows
    const size_t q_lo = (size_t)bhk * R + min(r_lo, R - 1), q_hi = (size_t)bhk * R + min(r_hi, R - 1);
    const int lim_lo = CAUSAL ? min(dm.q_offset + r_lo / G, dm.Skv - 1) : dm.Skv - 1;  // last visible column
    const int lim_hi = CAUSAL ? min(dm.q_offset + r_hi / G, dm.Skv - 1) : dm.Skv - 1;
    uint32_t qa[C::KSTEPS][4];
    load_q<D>(qa, Qq + q_lo * D + T0 * 4, Qq + q_hi * D + T0 * 4);
    const float qr_lo = Qr[q_lo], qr_hi = Qr[q_hi];  // log2 score = s * qr * (K scale of the tile)

    float o[DT][4], ol[4] = {0.f, 0.f, 0.f, 0.f};  // ol: P.1 (row sums via a ones tile)
#pragma unroll
    for (int i = 0; i < DT; ++i)
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m_lo = -INFINITY, m_hi = -INFINITY;  // running max, log2 domain

    // ldmatrix lane addressing: the swizzle term depends only on lane bits, not on the n-tile index.
    const int lm_i = lane >> 3, lm_r = lane & 7;
    uint32_t k_lane[NG];
#pragma unroll
    for (int gq = 0; gq < NG; ++gq)
        k_lane[gq] = lm_r * C::RB + k_swz<D>(lm_r, 4 * gq + lm_i) * 16;  // + nt*8*RB
    const uint32_t v_lane = C::OFF_V + ((lm_i >> 1) * 8 + lm_r) * 32 +
                            v_swz(lm_r, lm_i & 1) * 16;  // + dp * 512

    for (int t = 0; t < ntiles; ++t) {
        const int st = t % STAGES;
        const uint32_t stage = sbase + st * TILE;
        const uint8_t* sgen = smem + st * TILE;
        mbar_wait(full0 + st * 8, (t / STAGES) & 1);
        const int j0 = t * BKV;
        if (valid && (!CAUSAL || j0 <= p_wmax)) {  // warp-uniform
            const float ksc = *reinterpret_cast<const float*>(sgen + C::OFF_KSC);
            const float c_lo = qr_lo * ksc, c_hi = qr_hi * ksc;
            int s[8][4];
            qk_tile<D>(s, qa, stage, k_lane);
            // causal diagonal and KV tail tiles: MASKED drives the maxima, masked P is zeroed after ex2
            // (a near-zero row scale would lift MASKED * c to ~0).
            const bool need_mask = (CAUSAL && j0 + BKV - 1 > p_wmin) || j0 + BKV > dm.Skv;
            if (need_mask)
                mask_scores(s, j0 + T0 * 2, lim_lo, lim_hi);
            float gm[4];
            group_max(s, gm);
            update_max<DT>(fmaxf(gm[0], gm[2]) * c_lo, fmaxf(gm[1], gm[3]) * c_hi, m_lo, m_hi, o, ol);
            float bias[4];
            uint32_t kb[4], pa[4];
            p_scales(gm, m_lo, m_hi, c_lo, c_hi, bias, kb);
            float p[8][4];
            exp_scores(s, c_lo, c_hi, bias, p);
            if (need_mask)
                mask_probs(p, j0 + T0 * 2, lim_lo, lim_hi);
            pack_p(p, pa);
            pv_tile<D>(o, ol, pa, p_scale_word(kb, T0), stage, sgen, v_lane, T1);
        }
        mbar_arrive(empty0 + st * 8);
        if (threadIdx.x == 0 && t + STAGES < ntiles) {
            mbar_wait(empty0 + st * 8, (t / STAGES) & 1);
            mbar_expect_tx(full0 + st * 8, TILE);
            bulk_g2s(stage, kv + (size_t)(t + STAGES) * TILE, TILE, full0 + st * 8);
        }
    }

    // ---- epilogue: ol holds full row sums (the MMA reduced over all 64 slots)
    if (!valid)
        return;
    const float vg = hs[bhk].v;
    store_rows<D>(O, dm, b, hk, r_lo, T0, o, vg / ol[0], vg / ol[2]);
}

}  // namespace ra2
