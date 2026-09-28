// Per-row-scale FP8 E4M3 LM head GEMM with FP32 output (gemm.nvfp4_lm_head=fp8, #2156):
// y[r][row] = scale[row] * dot(W[row], x[r]). mma.sync m16n8k16 (E4M3 -> FP16 is exact, FP32
// accumulate); one weight pass serves up to 32 activation rows.

#include "compute/gemm.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace imp {

namespace {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kNT = 4;  // n8 tiles per block (every warp): 32 vocab rows
constexpr int kRowsPerBlock = kNT * 8;
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

// One 64-wide K chunk: 4 MMA k-steps over kNT weight tiles and MT activation tiles.
template <int MT>
__device__ __forceinline__ void mma_chunk(float (&acc)[MT][kNT][4], const uint4 (&wr)[kNT],
                                          const uint4* const (&xp)[MT][2], const bool (&x_ok)[MT][2], int k0,
                                          int t) {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint4 xr[MT][2][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4* p = xp[mt][h] + (k0 >> 3) + 2 * t;
            xr[mt][h][0] = x_ok[mt][h] ? __ldg(p) : zero;
            xr[mt][h][1] = x_ok[mt][h] ? __ldg(p + 1) : zero;
        }
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

// One block = 32 vocab rows; its 4 warps each take a quarter of K (split-K 4 regardless of n,
// so a wave holds ~4.6x more work units than one-warp-per-32-rows: 1.13 -> target 1.6 TB/s).
// K is walked in 64-wide chunks; lane (g = lane/4, t = lane%4) holds physical k 16t..16t+15 of
// its weight row (16 B) and activation rows. MMA step s maps logical k {2t,2t+1 | 2t+8,2t+9} to
// physical 16t+4s+{0,1 | 2,3} for A and B alike, so the dot product is unchanged.
// Row r's MMA chain and the fixed w0+w1+w2+w3 reduction never depend on n_rows (#2152 rule).
template <int MT>
__global__ void __launch_bounds__(kThreads) fp8_head_mma_kernel(const uint8_t* __restrict__ W,
                                                                const float* __restrict__ row_scales,
                                                                const half* __restrict__ x,
                                                                float* __restrict__ y, int M, int K,
                                                                int n_rows) {
    constexpr int kAcc = MT * kNT * 4;
    __shared__ float s_red[kWarps - 1][kAcc][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_base = blockIdx.x * kRowsPerBlock;
    const int k_len = K / kWarps;
    const int k_begin = warp * k_len;

    const uint4* wp[kNT];
    bool w_ok[kNT];
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
        const int r = n_base + 8 * j + g;
        w_ok[j] = r < M;
        wp[j] = reinterpret_cast<const uint4*>(W + static_cast<int64_t>(w_ok[j] ? r : 0) * K);
    }
    const uint4* xp[MT][2];
    bool x_ok[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mt * 16 + h * 8 + g;
            x_ok[mt][h] = r < n_rows;
            xp[mt][h] = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(x_ok[mt][h] ? r : 0) * K);
        }

    float acc[MT][kNT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < kNT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[mt][j][e] = 0.0f;

    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    // kU chunks of weights in flight per warp; the tail runs one chunk at a time. Same per-chunk
    // math and chunk order either way.
    constexpr int kU = 4;
    const int k_end = k_begin + k_len;
    int k0 = k_begin;
    for (; k0 + kU * 64 <= k_end; k0 += kU * 64) {
        uint4 wr[kU][kNT];
#pragma unroll
        for (int u = 0; u < kU; ++u)
#pragma unroll
            for (int j = 0; j < kNT; ++j)
                wr[u][j] = w_ok[j] ? __ldcs(wp[j] + ((k0 + 64 * u) >> 4) + t) : zero;
#pragma unroll
        for (int u = 0; u < kU; ++u)
            mma_chunk<MT>(acc, wr[u], xp, x_ok, k0 + 64 * u, t);
    }
    for (; k0 < k_end; k0 += 64) {
        uint4 wr[kNT];
#pragma unroll
        for (int j = 0; j < kNT; ++j)
            wr[j] = w_ok[j] ? __ldcs(wp[j] + (k0 >> 4) + t) : zero;
        mma_chunk<MT>(acc, wr, xp, x_ok, k0, t);
    }

    if (warp > 0) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int j = 0; j < kNT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    s_red[warp - 1][(mt * kNT + j) * 4 + e][lane] = acc[mt][j][e];
    }
    __syncthreads();
    if (warp != 0)
        return;
#pragma unroll
    for (int w = 0; w < kWarps - 1; ++w)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int j = 0; j < kNT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[mt][j][e] += s_red[w][(mt * kNT + j) * 4 + e][lane];

#pragma unroll
    for (int j = 0; j < kNT; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = n_base + 8 * j + 2 * t + e;
            if (col >= M)
                continue;
            const float sc = row_scales[col];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r0 = mt * 16 + g;
                if (r0 < n_rows)
                    y[static_cast<int64_t>(r0) * M + col] = acc[mt][j][e] * sc;
                if (r0 + 8 < n_rows)
                    y[static_cast<int64_t>(r0 + 8) * M + col] = acc[mt][j][2 + e] * sc;
            }
        }
    }
}

}  // namespace

bool gemv_fp8_rowscale_fp32(const void* W_fp8, const float* d_row_scales, const half* x, float* y, int M,
                            int K, int n_rows, cudaStream_t stream) {
    if (!W_fp8 || !d_row_scales || !x || !y || M <= 0 || K <= 0 || (K % (64 * kWarps)) != 0 || n_rows <= 0)
        return false;
    const auto* W = static_cast<const uint8_t*>(W_fp8);
    const int blocks = (M + kRowsPerBlock - 1) / kRowsPerBlock;
    for (int r0 = 0; r0 < n_rows; r0 += kRowsPerPass) {
        const int nb = std::min(kRowsPerPass, n_rows - r0);
        const half* xr = x + static_cast<int64_t>(r0) * K;
        float* yr = y + static_cast<int64_t>(r0) * M;
        if (nb <= 16)
            fp8_head_mma_kernel<1><<<blocks, kThreads, 0, stream>>>(W, d_row_scales, xr, yr, M, K, nb);
        else
            fp8_head_mma_kernel<2><<<blocks, kThreads, 0, stream>>>(W, d_row_scales, xr, yr, M, K, nb);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}

}  // namespace imp
