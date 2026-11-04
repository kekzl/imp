// Shared device helpers of the per-row-scale FP8 GEMMs (gemv_fp8_rowscale_fp32.cu: LM head + split-K,
// gemv_fp8_rowscale_f16.cu: streamed GDN / drafter projections). Moved verbatim: every kernel keeps the
// same per-chunk MMA chain, so the two TUs produce the same output bits.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace imp::fp8_rowscale {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kRowsPerPass = 32;  // activation rows per launch (2 m16 tiles)

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two_codes) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two_codes), __NV_E4M3);
    uint32_t r;
    memcpy(&r, &h, sizeof(r));
    return r;
}

__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Activation fragments of one 64-wide K chunk: lane t holds halves 16t..16t+15 of rows g and g + 8.
template <int MT>
__device__ __forceinline__ void load_x_chunk(uint4 (&xr)[MT][2][2], const uint4* const (&xp)[MT][2],
                                             const bool (&x_ok)[MT][2], int k0, int t) {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4* p = xp[mt][h] + (k0 >> 3) + 2 * t;
            xr[mt][h][0] = x_ok[mt][h] ? __ldg(p) : zero;
            xr[mt][h][1] = x_ok[mt][h] ? __ldg(p + 1) : zero;
        }
}

// One 64-wide K chunk: 4 MMA k-steps over kNT weight tiles and MT activation tiles.
template <int MT, int kNT>
__device__ __forceinline__ void mma_chunk_x(float (&acc)[MT][kNT][4], const uint4 (&wr)[kNT],
                                            const uint4 (&xr)[MT][2][2]) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        uint32_t b0[kNT], b1[kNT];
#pragma unroll
        for (int j = 0; j < kNT; ++j) {
            const uint32_t wv = word(wr[j], s);
            b0[j] = e4m3x2_to_f16x2(wv & 0xffffu);
            b1[j] = e4m3x2_to_f16x2(wv >> 16);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            // halves 4s..4s+3 of the lane's 16: words 2s, 2s+1 across the two uint4
            const uint4& lo0 = xr[mt][0][s >> 1];
            const uint4& lo1 = xr[mt][1][s >> 1];
            const int w0 = (2 * s) & 3;
            const uint32_t a0 = word(lo0, w0), a2 = word(lo0, w0 + 1);
            const uint32_t a1 = word(lo1, w0), a3 = word(lo1, w0 + 1);
#pragma unroll
            for (int j = 0; j < kNT; ++j)
                mma_16816(acc[mt][j], a0, a1, a2, a3, b0[j], b1[j]);
        }
    }
}

template <int MT, int kNT>
__device__ __forceinline__ void mma_chunk(float (&acc)[MT][kNT][4], const uint4 (&wr)[kNT],
                                          const uint4* const (&xp)[MT][2], const bool (&x_ok)[MT][2], int k0,
                                          int t) {
    uint4 xr[MT][2][2];
    load_x_chunk<MT>(xr, xp, x_ok, k0, t);
    mma_chunk_x<MT, kNT>(acc, wr, xr);
}

// Warps 1..3 park their split-K partials; warp 0 adds them in fixed order w = 0, 1, 2.
template <int MT, int kNT>
__device__ __forceinline__ void stash_partials(const float (&acc)[MT][kNT][4], float (*s_red)[32], int lane) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < kNT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                s_red[(((mt * kNT) + j) * 4) + e][lane] = acc[mt][j][e];
}

template <int MT, int kNT>
__device__ __forceinline__ void add_partials(float (&acc)[MT][kNT][4], const float (*s_red)[MT * kNT * 4][32],
                                             int lane) {
#pragma unroll
    for (int w = 0; w < kWarps - 1; ++w)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int j = 0; j < kNT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[mt][j][e] += s_red[w][(((mt * kNT) + j) * 4) + e][lane];
}

template <int MT, int kNT, typename OutT>
__device__ __forceinline__ void store_rows(const float (&acc)[MT][kNT][4],
                                           const float* __restrict__ row_scales, OutT* __restrict__ y, int M,
                                           int n_rows, int n_base, int g, int t) {
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = n_base + (8 * j) + (2 * t) + e;
            if (col >= M)
                continue;
            const float sc = row_scales[col];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r0 = (mt * 16) + g;
                if (r0 < n_rows)
                    y[(static_cast<int64_t>(r0) * M) + col] = static_cast<OutT>(acc[mt][j][e] * sc);
                if (r0 + 8 < n_rows)
                    y[(static_cast<int64_t>(r0 + 8) * M) + col] = static_cast<OutT>(acc[mt][j][2 + e] * sc);
            }
        }
    }
}

constexpr int kStageBytes = 128;             // K bytes per row per stage: 2 MMA chunks
constexpr int kRowPitch = kStageBytes + 64;  // 192: rows g, g+1 land in opposite bank halves

__device__ __forceinline__ void cp_async16(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

template <int kNT, int kS>
constexpr int cpasync_smem_bytes() {
    return kWarps * kS * kNT * 8 * kRowPitch;
}

// Split-K FP16 dispatch (gemv_fp8_rowscale_fp32.cu): the reference twin of the streamed projection kernel.
[[nodiscard]] bool split_k_f16(const void* W_fp8, const float* d_row_scales, const half* x, half* y, int M,
                               int K, int n_rows, cudaStream_t stream);

}  // namespace imp::fp8_rowscale
