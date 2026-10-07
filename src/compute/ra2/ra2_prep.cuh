// RA2 prep: KV readers (flat / paged FP16 / paged NVFP4), stats, K int8 + V two-term FP4 tile blobs, Q int8.
// Vendored from the ra2 standalone repo (kekzl/ra2); change there first, then copy.
#pragma once
#include "ra2_common.cuh"

namespace ra2 {

// ---------------------------------------------------------------- KV readers: 16 consecutive channels ->
// float Flat K/V: [B][Skv][Hkv][D], T = __half or __nv_bfloat16.
template <typename T>
struct FlatKV {
    const T* k;
    const T* v;
    __device__ __forceinline__ void load16(bool isv, int b, int s, int hk, int d0, const Dims& dm, int D,
                                           float* x) const {
        const T* p = (isv ? v : k) + (((size_t)b * dm.Skv + s) * dm.Hkv + hk) * D + d0;
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
    __device__ __forceinline__ void load16(bool isv, int b, int s, int hk, int d0, const Dims& dm, int D,
                                           float* x) const {
        const int blk = block_table[(size_t)b * max_blocks + s / bs];
        if (blk < 0) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] = 0.f;
            return;
        }
        const __half* p = (isv ? v : k) + (((size_t)blk * bs + s % bs) * dm.Hkv + hk) * D + d0;
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
        dec8_e2m1(w.x, x);
        dec8_e2m1(w.y, x + 8);
#pragma unroll
        for (int i = 0; i < 16; ++i)
            x[i] *= sc;
    }
};

// ---------------------------------------------------------------- stats: K mean, V amax per (b, kv head)
// grid (ntkv, Hkv, B), 256 threads. amax[bhk*3 + 2] (V) as uint bits, ksum[bhk*D + d].
template <int D, typename Reader>
__global__ void __launch_bounds__(256) stats_kernel(Reader rd, Dims dm, unsigned* amax, float* ksum) {
    constexpr int G16 = D / 16;
    __shared__ float ks_s[D];
    const int b = blockIdx.z, hk = blockIdx.y, bhk = b * dm.Hkv + hk, tid = threadIdx.x;
    for (int d = tid; d < D; d += 256)
        ks_s[d] = 0.f;
    __syncthreads();
    float av = 0.f;
    for (int p = tid; p < BKV * G16; p += 256) {
        const int s = blockIdx.x * BKV + p / G16, g = p % G16;
        if (s >= dm.Skv)
            continue;
        float x[16];
        rd.load16(false, b, s, hk, g * 16, dm, D, x);
#pragma unroll
        for (int i = 0; i < 16; ++i)
            atomicAdd(&ks_s[g * 16 + i], x[i]);
        rd.load16(true, b, s, hk, g * 16, dm, D, x);
#pragma unroll
        for (int i = 0; i < 16; ++i)
            av = fmaxf(av, fabsf(x[i]));
    }
    for (int o = 16; o; o >>= 1)
        av = fmaxf(av, __shfl_xor_sync(~0u, av, o));
    if ((tid & 31) == 0)
        atomicMax(&amax[bhk * 3 + 2], __float_as_uint(av));
    __syncthreads();
    for (int d = tid; d < D; d += 256)
        atomicAdd(&ksum[bhk * D + d], ks_s[d]);
}

// ---------------------------------------------------------------- KV quantization into tile blobs
// grid (ntkv, Hkv, B), 256 threads. Tokens >= Skv are zero (masked in attention).
// K: mean-centred per channel (softmax-invariant), int8 with one scale per tile (OFF_KSC).
// V: two-term FP4 (V4 + residual) per 16 token slots, global scale vg per (b, kv head).
template <int D, typename Reader>
__global__ void __launch_bounds__(256) quant_kv_kernel(Reader rd, Dims dm, const unsigned* __restrict__ amax,
                                                       const float* __restrict__ ksum,
                                                       uint8_t* __restrict__ KV, HeadScale* __restrict__ hs) {
    using C = Cfg<D>;
    __shared__ __half vt[BKV][D + 2];  // V / vg (|x| <= 2688 fits FP16)
    __shared__ float kmean[D];
    __shared__ float red[8];
    const int b = blockIdx.z, hk = blockIdx.y, bhk = b * dm.Hkv + hk, tile = blockIdx.x, tid = threadIdx.x;
    for (int d = tid; d < D; d += 256)
        kmean[d] = ksum[bhk * D + d] / dm.Skv;
    const float vg = fmaxf(__uint_as_float(amax[bhk * 3 + 2]), 1e-20f) / P_SCALE;
    if (tile == 0 && tid == 0)
        hs[bhk] = {1.f, 1.f, vg};
    uint8_t* blob = KV + ((size_t)bhk * dm.ntkv + tile) * C::TILE;
    __syncthreads();

    // ---- pass 1: V rows to smem, K tile amax after centring
    float ak = 0.f;
    for (int p = tid; p < BKV * C::G16; p += 256) {
        const int r = p / C::G16, g = p % C::G16, s = tile * BKV + r;
        float x[16];
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
        if (s < dm.Skv) {
            rd.load16(false, b, s, hk, g * 16, dm, D, x);
#pragma unroll
            for (int i = 0; i < 16; ++i)
                ak = fmaxf(ak, fabsf(x[i] - kmean[g * 16 + i]));
        }
    }
    for (int o = 16; o; o >>= 1)
        ak = fmaxf(ak, __shfl_xor_sync(~0u, ak, o));
    if ((tid & 31) == 0)
        red[tid >> 5] = ak;
    __syncthreads();
    ak = 0.f;
    for (int i = 0; i < 8; ++i)
        ak = fmaxf(ak, red[i]);
    const float ksc = fmaxf(ak, 1e-20f) / 127.f, kinv = 1.f / ksc;
    if (tid == 0)
        *reinterpret_cast<float*>(blob + C::OFF_KSC) = ksc;

    // ---- pass 2: K int8, one 16 B chunk per (row, group), swizzled
    for (int p = tid; p < BKV * C::G16; p += 256) {
        const int r = p / C::G16, g = p % C::G16, s = tile * BKV + r;
        float x[16];
        if (s < dm.Skv) {
            rd.load16(false, b, s, hk, g * 16, dm, D, x);
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] -= kmean[g * 16 + i];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                x[i] = 0.f;
        }
        *reinterpret_cast<uint4*>(blob + r * C::RB + k_swz<D>(r, g) * 16) = pack16_s8(x, kinv);
    }
    // ---- Vt two-term FP4 (channel row d, slot group g of 4)
    for (int p = tid; p < D * (BKV / 16); p += 256) {
        const int d = p >> 2, g = p & 3;
        float x[16];
        uint2 w0, w1;
        uint8_t sb0, sb1;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            x[i] = __half2float(vt[slot_token(g * 16 + i)][d]);
        quant16_res(x, w0, sb0, w1, sb1);
        const int off = d * 32 + v_swz(d, g >> 1) * 16 + (g & 1) * 8;
        *reinterpret_cast<uint2*>(blob + C::OFF_V + off) = w0;
        *reinterpret_cast<uint2*>(blob + C::OFF_VR + off) = w1;
        const int so = (d & 7) * (D / 2) + (d >> 3) * 4 + g;  // [T1][dt][4]
        blob[C::OFF_VS + so] = sb0;
        blob[C::OFF_VSR + so] = sb1;
    }
}

// ---------------------------------------------------------------- Q packing + quantization
// grid (ceil(R/64), Hkv, B), 256 threads. Q [B][Sq][H][D] -> packed int8 rows per (b, kv head):
// Qq [bhk][R][D], Qr [bhk][R] = row amax * qmul / 127; qmul = softmax scale * log2(e).
template <int D, typename T>
__global__ void __launch_bounds__(256) quant_q_kernel(const T* __restrict__ Q, Dims dm, float qmul,
                                                      uint8_t* __restrict__ Qq, float* __restrict__ Qr) {
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
        const float qr = fmaxf(am, 1e-20f) / 127.f;
        const uint4 w = pack16_s8(x, 1.f / qr);
        if (!ok)
            continue;
        const size_t row = (size_t)bhk * dm.R + r;
        if (g == 0)
            Qr[row] = qr;
        *reinterpret_cast<uint4*>(Qq + row * D + g * 16) = w;
    }
}

}  // namespace ra2
