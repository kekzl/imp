// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// Vendored from kekzl/apa include/apa/apa_common.cuh (v0.2.0, 502705a); change there first, then copy.
// APA common (all-FP4 base from ra2 1d72da2): Cfg<D>, layouts, PTX helpers (sm_120a).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

namespace apa {

constexpr int BKV = 64;                 // tokens per KV tile
constexpr int STATS_TILES = 16;         // KV tiles per stats_kernel block (one K column-sum chunk)
constexpr float P_SCALE = 448.f * 6.f;  // global scales map |x| max to E2M1 6 * UE4M3 448
constexpr float LOG2E = 1.4426950408889634f;

// Per head dim: tile blob = K [64][D/2] | Vt [D][32] | Ks [8][8][D/16] | Vs [8][D/8][4].
template <int D>
struct Cfg {
    static_assert(D == 64 || D == 128 || D == 256, "head dim");
    static constexpr int K_BYTES = BKV * D / 2;
    static constexpr int V_BYTES = D * BKV / 2;
    static constexpr int KS_BYTES = BKV * D / 16;
    static constexpr int VS_BYTES = D * BKV / 16;
    static constexpr int TILE = K_BYTES + V_BYTES + KS_BYTES + VS_BYTES;
    static constexpr int RB = D / 2;    // K row bytes
    static constexpr int CPR = D / 32;  // 16 B chunks per K row
    static constexpr int KSH = CPR == 2 ? 2 : CPR == 4 ? 1 : 0;
    static constexpr int KSTEPS = D / 64;  // k64 steps of Q.K^T
    static constexpr int DT = D / 8;       // n8 tiles of P.V
    static constexpr int G16 = D / 16;     // 16-element groups per row
};

struct HeadScale {
    float q, k, v;
};

// Slot k (0..63) of the PV A-operand holds token tok(k): matches the QK C-fragment so P needs no shuffle.
__host__ __device__ constexpr int slot_token(int k) {
    return ((k >> 5) * 4 + ((k & 7) >> 1)) * 8 + ((k >> 3) & 3) * 2 + (k & 1);
}
// 8 consecutive rows land in 8 distinct 16 B slots of a 128 B line (ldmatrix conflict-free).
template <int D>
__host__ __device__ constexpr int k_swz(int row, int c) {
    return c ^ ((row >> Cfg<D>::KSH) & (Cfg<D>::CPR - 1));
}
__host__ __device__ constexpr int v_swz(int row, int c) { return c ^ ((row >> 2) & 1); }  // Vt rows: 32 B

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ uint32_t pack8_e2m1(const float* v) {
    uint32_t out;
    asm("{\n .reg .b8 b0, b1, b2, b3;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        " mov.b32 %0, {b0, b1, b2, b3};\n}"
        : "=r"(out)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return out;
}

__device__ __forceinline__ uint8_t e4m3_enc(float x) {
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float e4m3_dec(uint8_t b) {
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(__half(h));
}

// 16 values -> 8 E2M1 bytes + UE4M3 block scale (x already divided by the global scale).
__device__ __forceinline__ uint8_t quant16(const float* x, uint2& w) {
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        am = fmaxf(am, fabsf(x[i]));
    const uint8_t sb = e4m3_enc(am / 6.f);
    const float inv = 1.f / fmaxf(e4m3_dec(sb), 1.f / 512);
    float q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        q[i] = x[i] * inv;
    w = make_uint2(pack8_e2m1(q), pack8_e2m1(q + 8));
    return sb;
}

__device__ __forceinline__ void mma_fp4(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sfa,
                                        uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
          "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));  // byte/thread selectors 0
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n .reg .pred p;\n"
        "W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra W;\n}" ::"r"(bar),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
        "l"(src), "r"(bytes), "r"(bar)
        : "memory");
}

// Problem geometry shared by prep and attention. Q/O: [B][Sq][H][D]; GQA group G = H / Hkv.
// Packed Q rows per (b, kv head): row r = position r / G, q head hk*G + r % G (R = Sq * G rows).
struct Dims {
    int B, Sq, Skv, H, Hkv, G, R, ntkv, q_offset;
    int kvcap;  // KV blob stride in tiles per (b, kv head): ntkv, or a persistent KvState capacity
};

__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void store2(__half* p, float a, float b) {
    *reinterpret_cast<__half2*>(p) = __floats2half2_rn(a, b);
}
__device__ __forceinline__ void store2(__nv_bfloat16* p, float a, float b) {
    *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

}  // namespace apa
