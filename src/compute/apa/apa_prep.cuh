// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// Vendored from kekzl/apa include/apa/apa_prep.cuh (v0.2.0, 502705a); change there first, then copy.
// APA prep: KV readers (flat / paged FP16 / paged NVFP4), K/V stats, quantization into tile blobs, Q packing.
#pragma once
#include "apa_common.cuh"

namespace apa {

// ---------------------------------------------------------------- KV readers: 16 consecutive channels ->
// float Flat K/V: [B][Skv][Hkv][D], T = __half or __nv_bfloat16.
template <typename T>
struct FlatKV {
    const T* k;
    const T* v;
    // Row s of K or V (D contiguous elements); pass 2 copies rows as 16 B chunks.
    __device__ __forceinline__ const T* row(bool isv, int b, int s, int hk, const Dims& dm, int D) const {
        return (isv ? v : k) + (((size_t)b * dm.Skv + s) * dm.Hkv + hk) * D;
    }
    // Pass-2 tile loads: 32-bit element offset of row s (same for K and V), resolved by at().
    __device__ __forceinline__ uint32_t row_off(int b, int s, int hk, const Dims& dm, int D) const {
        return (uint32_t)(((b * dm.Skv + s) * dm.Hkv + hk) * D);
    }
    __device__ __forceinline__ const T* at(bool isv, uint32_t off) const { return (isv ? v : k) + off; }
    __device__ __forceinline__ void load16(bool isv, int b, int s, int hk, int d0, const Dims& dm, int D,
                                           float* x) const {
        const T* p = row(isv, b, s, hk, dm, D) + d0;
        const uint4 u0 = *reinterpret_cast<const uint4*>(p), u1 = *reinterpret_cast<const uint4*>(p + 8);
        const T* e0 = reinterpret_cast<const T*>(&u0);
        const T* e1 = reinterpret_cast<const T*>(&u1);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            x[i] = to_f(e0[i]), x[8 + i] = to_f(e1[i]);
    }
};

// Paged FP16 (imp layout): block [bs][Hkv][D], block_table [B][max_blocks], -1 = hole (reads zeros).
struct PagedF16KV {
    const __half* k;
    const __half* v;
    const int* block_table;
    int bs, max_blocks;
    // Tokens s >= tail come from flat kt/vt [B][Skv - tail][Hkv][D] (imp: the chunk not yet in the cache).
    const __half* kt = nullptr;
    const __half* vt = nullptr;
    int tail = 0x7fffffff;
    int bs_shift = -1;  // log2(bs) when bs is a power of two (no integer division per row)
    // Row s of K or V (D contiguous halves), nullptr for a hole.
    __device__ __forceinline__ const __half* row(bool isv, int b, int s, int hk, const Dims& dm,
                                                 int D) const {
        if (s >= tail)
            return (isv ? vt : kt) + (((size_t)b * (dm.Skv - tail) + s - tail) * dm.Hkv + hk) * D;
        const int bi = bs_shift >= 0 ? s >> bs_shift : s / bs, si = bs_shift >= 0 ? s & (bs - 1) : s % bs;
        const int blk = block_table[(size_t)b * max_blocks + bi];
        return blk < 0 ? nullptr : (isv ? v : k) + (((size_t)blk * bs + si) * dm.Hkv + hk) * D;
    }
    // Pass-2 tile loads: 31-bit element offset of row s, bit 31 = tail buffer, ~0u = hole; resolved by at().
    __device__ __forceinline__ uint32_t row_off(int b, int s, int hk, const Dims& dm, int D) const {
        if (s >= tail)
            return 0x80000000u | (uint32_t)(((b * (dm.Skv - tail) + s - tail) * dm.Hkv + hk) * D);
        const int bi = bs_shift >= 0 ? s >> bs_shift : s / bs, si = bs_shift >= 0 ? s & (bs - 1) : s % bs;
        const int blk = __ldg(block_table + b * max_blocks + bi);
        return blk < 0 ? ~0u : (uint32_t)(((blk * bs + si) * dm.Hkv + hk) * D);
    }
    __device__ __forceinline__ const __half* at(bool isv, uint32_t off) const {
        return ((off >> 31) ? (isv ? vt : kt) : (isv ? v : k)) + (off & 0x7fffffffu);
    }
    __device__ __forceinline__ void load16(bool isv, int b, int s, int hk, int d0, const Dims& dm, int D,
                                           float* x) const {
        const __half* r = row(isv, b, s, hk, dm, D);
        if (r == nullptr) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] = 0.f;
            return;
        }
        const __half* p = r + d0;
        const uint4 u0 = *reinterpret_cast<const uint4*>(p), u1 = *reinterpret_cast<const uint4*>(p + 8);
        const __half* e0 = reinterpret_cast<const __half*>(&u0);
        const __half* e1 = reinterpret_cast<const __half*>(&u1);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            x[i] = __half2float(e0[i]), x[8 + i] = __half2float(e1[i]);
    }
};

// Paged NVFP4 (imp layout): data [block][slot][Hkv][D/2] (even d = low nibble), scales
// [block][slot][Hkv][D/16] UE4M3 per 16 channels, no global scale.
struct PagedNvfp4KV {
    const uint8_t* k;
    const uint8_t* v;
    const uint8_t* k_sc;
    const uint8_t* v_sc;
    const int* block_table;
    int bs, max_blocks;
    __device__ __forceinline__ void load16(bool isv, int b, int s, int hk, int d0, const Dims& dm, int D,
                                           float* x) const {
        const int blk = block_table[(size_t)b * max_blocks + s / bs];
        if (blk < 0) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] = 0.f;
            return;
        }
        const size_t slot = ((size_t)blk * bs + s % bs) * dm.Hkv + hk;
        const uint2 w = *reinterpret_cast<const uint2*>((isv ? v : k) + slot * (D / 2) + d0 / 2);
        const float sc = e4m3_dec((isv ? v_sc : k_sc)[slot * (D / 16) + d0 / 16]);
        constexpr float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t nib = ((i < 8 ? w.x : w.y) >> (4 * (i & 7))) & 0xF;
            x[i] = (nib & 8 ? -kE2M1[nib & 7] : kE2M1[nib & 7]) * sc;
        }
    }
};

// ---------------------------------------------------------------- stats: K mean, K/V amax per (b, kv head)
// grid (ceil(ntkv / STATS_TILES), Hkv, B), 256 threads. amax[bhk*3 + {1:k, 2:v}] as uint bits (atomicMax),
// kpart[bhk][chunk][D] K column sums of the chunk.
template <int D, typename Reader>
__global__ void __launch_bounds__(256) stats_kernel(Reader rd, Dims dm, unsigned* amax, float* kpart,
                                                    const int* redo = nullptr) {
    if (redo != nullptr && *redo == 0)
        return;  // KvState still valid: stats stay frozen
    constexpr int G16 = D / 16, RSTEP = 256 / G16, NWARP = 8;
    __shared__ float red[NWARP][D];
    const int b = blockIdx.z, hk = blockIdx.y, bhk = b * dm.Hkv + hk, tid = threadIdx.x;
    const int g = tid % G16, s0 = blockIdx.x * STATS_TILES * BKV, s1 = min(s0 + STATS_TILES * BKV, dm.Skv);
    // Column sums in a fixed order (deterministic): registers over this thread's rows, then lanes of the same
    // channel group (xor G16 .. 16), then the 8 warps in order. Maxima are order-free (atomicMax).
    float acc[16], x[16], ak = 0.f, av = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] = 0.f;
    for (int s = s0 + tid / G16; s < s1; s += RSTEP) {
        rd.load16(false, b, s, hk, g * 16, dm, D, x);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ak = fmaxf(ak, fabsf(x[i]));
            acc[i] += x[i];
        }
        rd.load16(true, b, s, hk, g * 16, dm, D, x);
#pragma unroll
        for (int i = 0; i < 16; ++i)
            av = fmaxf(av, fabsf(x[i]));
    }
#pragma unroll
    for (int o = G16; o < 32; o <<= 1)
#pragma unroll
        for (int i = 0; i < 16; ++i)
            acc[i] += __shfl_xor_sync(~0u, acc[i], o);
    if ((tid & 31) < G16)
#pragma unroll
        for (int i = 0; i < 16; ++i)
            red[tid >> 5][g * 16 + i] = acc[i];
    for (int o = 16; o; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(~0u, ak, o));
        av = fmaxf(av, __shfl_xor_sync(~0u, av, o));
    }
    if ((tid & 31) == 0) {
        atomicMax(&amax[bhk * 3 + 1], __float_as_uint(ak));
        atomicMax(&amax[bhk * 3 + 2], __float_as_uint(av));
    }
    __syncthreads();
    for (int d = tid; d < D; d += 256) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < NWARP; ++w)
            sum += red[w][d];
        kpart[((size_t)bhk * gridDim.x + blockIdx.x) * D + d] = sum;
    }
}

// ---------------------------------------------------------------- KV quantization into tile blobs
// grid (B * Hkv), D threads: K mean = sum of the kpart chunks in order / n, head scales kg / vg from the
// maxima, widened by headroom (> 1 when later keys reuse the scales: persistent KvState, values up to
// headroom x).
template <int D>
__global__ void __launch_bounds__(D) finalize_stats_kernel(const unsigned* __restrict__ amax,
                                                           const float* __restrict__ kpart, int nchunk,
                                                           float* __restrict__ kmean,
                                                           HeadScale* __restrict__ hs, int n, float headroom,
                                                           const int* redo = nullptr) {
    if (redo != nullptr && *redo == 0)
        return;
    __shared__ float red[D];
    const int bhk = blockIdx.x, d = threadIdx.x;
    float sum = 0.f;  // stats_kernel chunks in order: deterministic
    for (int c = 0; c < nchunk; ++c)
        sum += kpart[((size_t)bhk * nchunk + c) * D + d];
    const float m = sum / n;
    kmean[bhk * D + d] = m;
    red[d] = fabsf(m);
    __syncthreads();
    for (int o = D / 2; o; o >>= 1) {
        if (d < o)
            red[d] = fmaxf(red[d], red[d + o]);
        __syncthreads();
    }
    if (d == 0) {
        const float kg = headroom * fmaxf(__uint_as_float(amax[bhk * 3 + 1]) + red[0], 1e-20f) / P_SCALE;
        const float vg = headroom * fmaxf(__uint_as_float(amax[bhk * 3 + 2]), 1e-20f) / P_SCALE;
        hs[bhk] = {1.f, kg, vg};
    }
}

// grid (ntkv - t0, Hkv, B): tiles t0.. into KV (stride kvcap tiles per (b, kv head)) with the finalized
// K mean (mean-centred K, softmax-invariant) and head scales; tokens >= Skv are zero (masked in attention).
template <int D, typename Reader>
__global__ void __launch_bounds__(256) quant_kv_kernel(Reader rd, Dims dm, const float* __restrict__ kmean_g,
                                                       const HeadScale* __restrict__ hs,
                                                       uint8_t* __restrict__ KV, int t0, int t_keep = 0,
                                                       const int* redo = nullptr) {
    using C = Cfg<D>;
    __shared__ __half vt[BKV][D + 2];  // V / vg (|x| <= 2688 fits FP16)
    __shared__ float kmean[D];
    const int b = blockIdx.z, hk = blockIdx.y, bhk = b * dm.Hkv + hk, tile = t0 + blockIdx.x,
              tid = threadIdx.x;
    if (tile < t_keep && redo != nullptr && *redo == 0)
        return;  // cached tile of a still-valid KvState
    for (int d = tid; d < D; d += 256)
        kmean[d] = kmean_g[bhk * D + d];
    __syncthreads();
    const float kg = hs[bhk].k, vg = hs[bhk].v;
    uint8_t* blob = KV + ((size_t)bhk * dm.kvcap + tile) * C::TILE;

    for (int p = tid; p < BKV * C::G16; p += 256) {
        const int r = p / C::G16, g = p % C::G16, s = tile * BKV + r;
        float x[16];
        uint2 w;
        // ---- V row to smem (transposed gather below)
        if (s < dm.Skv) {
            rd.load16(true, b, s, hk, g * 16, dm, D, x);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
            vt[r][g * 16 + i] = __float2half(x[i] / vg);
        // ---- K (token row r, swizzled 16 B chunk g>>1, half g&1)
        if (s < dm.Skv) {
            rd.load16(false, b, s, hk, g * 16, dm, D, x);
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] = (x[i] - kmean[g * 16 + i]) / kg;
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] = 0.f;
        }
        const uint8_t sb = quant16(x, w);
        *reinterpret_cast<uint2*>(blob + r * C::RB + k_swz<D>(r, g >> 1) * 16 + (g & 1) * 8) = w;
        blob[C::K_BYTES + C::V_BYTES + (r & 7) * (D / 2) + (r >> 3) * (D / 16) + g] = sb;  // [T1][nt][D/16]
    }
    __syncthreads();
    for (int p = tid; p < D * (BKV / 16); p += 256) {
        // ---- Vt (channel row d, slot group g of 4)
        const int d = p >> 2, g = p & 3;
        float x[16];
        uint2 w;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            x[i] = __half2float(vt[slot_token(g * 16 + i)][d]);
        const uint8_t sb = quant16(x, w);
        *reinterpret_cast<uint2*>(blob + C::K_BYTES + d * 32 + v_swz(d, g >> 1) * 16 + (g & 1) * 8) = w;
        blob[C::K_BYTES + C::V_BYTES + C::KS_BYTES + (d & 7) * (D / 2) + (d >> 3) * 4 + g] =
            sb;  // [T1][dt][4]
    }
}

// ---------------------------------------------------------------- KvState validity (incremental prep)
// grid (B * Hkv), 32 threads: zero the K / V maxima when redo is set.
template <int D>  // template: header-only, one instance per TU
__global__ void __launch_bounds__(32) reset_stats_kernel(unsigned* amax, const int* redo) {
    if (*redo == 0)
        return;
    if (threadIdx.x < 3)
        amax[blockIdx.x * 3 + threadIdx.x] = 0u;
}

// One warp. Fingerprint = first 16 K channels of (b 0, kv head 0) at keys 0, len / 2, len - 1 (48 floats).
// check: redo = (stored fingerprint of len != current keys); write: store the fingerprint of len.
template <int D, typename Reader>
__global__ void __launch_bounds__(32) kv_fingerprint_kernel(Reader rd, Dims dm, float* fp, int len, int* redo,
                                                            bool check) {
    const int lane = threadIdx.x, pos[3] = {0, len / 2, len - 1};
    float x[16];
    bool diff = false;
    if (lane < 3) {
        rd.load16(false, 0, pos[lane], 0, 0, dm, D, x);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (check)
                diff |= fp[lane * 16 + i] != x[i];
            else
                fp[lane * 16 + i] = x[i];
        }
    }
    diff = __any_sync(~0u, diff);
    if (check && lane == 0)
        *redo = diff ? 1 : 0;
}

// ---------------------------------------------------------------- Q packing + quantization
// grid (ceil(R/64), Hkv, B), 256 threads. Q [B][Sq][H][D] -> packed rows per (b, kv head):
// Qq [bhk][R][D/2], Qs [bhk][R][D/16], Qr [bhk][R]; qmul = softmax scale * log2(e).
template <int D, typename T>
__global__ void __launch_bounds__(256) quant_q_kernel(const T* __restrict__ Q, Dims dm, float qmul,
                                                      uint8_t* __restrict__ Qq, uint8_t* __restrict__ Qs,
                                                      float* __restrict__ Qr) {
    constexpr int G16 = D / 16;
    const int b = blockIdx.z, hk = blockIdx.y, bhk = b * dm.Hkv + hk, tid = threadIdx.x;
    for (int p = tid; p < BKV * G16; p += 256) {  // uniform trip count: shuffles below stay warp-complete
        const int r = blockIdx.x * BKV + p / G16, g = p % G16;
        const bool ok = r < dm.R;
        const int rr = ok ? r : dm.R - 1;
        const int s = rr / dm.G, h = hk * dm.G + rr % dm.G;
        const T* src = Q + (((size_t)b * dm.Sq + s) * dm.H + h) * D + g * 16;
        const uint4 u0 = *reinterpret_cast<const uint4*>(src), u1 = *reinterpret_cast<const uint4*>(src + 8);
        const T* e0 = reinterpret_cast<const T*>(&u0);
        const T* e1 = reinterpret_cast<const T*>(&u1);
        float x[16], am = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = to_f(e0[i]) * qmul;
            x[8 + i] = to_f(e1[i]) * qmul;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
            am = fmaxf(am, fabsf(x[i]));
#pragma unroll
        for (int o = 1; o < G16; o <<= 1)
            am = fmaxf(am, __shfl_xor_sync(~0u, am, o));
        const float qr = fmaxf(am, 1e-20f) / P_SCALE;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            x[i] /= qr;
        uint2 w;
        const uint8_t sb = quant16(x, w);
        if (!ok)
            continue;
        const size_t row = (size_t)bhk * dm.R + r;
        if (g == 0)
            Qr[row] = qr;
        *reinterpret_cast<uint2*>(Qq + row * (D / 2) + g * 8) = w;
        Qs[row * (D / 16) + g] = sb;
    }
}

}  // namespace apa
