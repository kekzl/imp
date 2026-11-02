// Small-M FP8 E4M3 x E4M3 GEMM (batched decode, M 1..32): weight-streaming, no smem pipeline.
//   CTA = 16 weight rows x the whole K; warp w owns the 64-byte k-steps ks % kWarps == w, so a round of
//   kWarps * kU k-steps reads one contiguous span per row; weights go global -> registers (LDG.128,
//   L1 no-allocate), each lane holds rows g and g+8 (g = lane / 4) at byte t4 * 16 of the k-step.
//   mma.sync m16n8k32 e4m3 with W as the A operand (16 rows) and the activations as B (8 tokens per
//   n-tile, kNT tiles): the k permutation phys = t4 * 16 + j is applied to both operands alike.
//   fp32 accumulate, warp partials summed in warp order through smem, per-tensor scales + beta in fp32.
// A row's result depends on (N, K, kWarps) only: tokens never share an MMA column, the K order is fixed.

#include "quant/fp8_gemm_smallm.h"

#include "core/logging.h"
#include "core/pdl_device.cuh"
#include "core/pdl_launch.cuh"

#include <cstdint>

namespace imp {
namespace {

constexpr int kRows = 16;          // weight rows per CTA (MMA m)
constexpr int kKStep = 64;         // bytes per k-step: two m16n8k32 MMAs
constexpr int kDefaultWarps = 8;   // fixes the K order: never chosen by M
constexpr int kDefaultUnroll = 2;  // Qwen3-8B cold L2, M 1..32: gate/down 37.4-42.0 us (cuBLASLt 58.8-72.8)

__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void mma_e4m3(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// One CTA: rows n0..n0+15 of W, all M tokens, the whole K. Shared by the single and multi kernels.
template <int kNT, int kWarps, int kU>
__device__ __forceinline__ void fp8_smallm_cta(const uint8_t* __restrict__ a, const uint8_t* __restrict__ w,
                                               const float* __restrict__ a_scale,
                                               const float* __restrict__ w_scale, float beta,
                                               half* __restrict__ y, int M, int N, int K, int n0) {
    __shared__ float red[kWarps][kNT * 8][kRows];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int ks_total = K / kKStep;
    constexpr int kPerRound = kWarps * kU;
    const int rounds = (ks_total + kPerRound - 1) / kPerRound;

    pdl_trigger();
    // #2218 bounded: n0 + g + 8 < N (N % 16 == 0, grid N / 16); ks * 64 + t4 * 16 + 16 <= K.
    const uint8_t* w0 = w + (static_cast<int64_t>(n0 + g) * K) + (static_cast<int64_t>(t4) * 16);
    const uint8_t* w1 = w0 + (static_cast<int64_t>(8) * K);
    const uint8_t* xr[kNT];
    bool xv[kNT];
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
        const int tok = (nt * 8) + g;
        xv[nt] = tok < M;
        xr[nt] = a + (static_cast<int64_t>(xv[nt] ? tok : 0) * K) + (static_cast<int64_t>(t4) * 16);
    }

    float acc[kNT][4];
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
        for (int r = 0; r < 4; ++r)
            acc[nt][r] = 0.0f;

    // Two register buffers: round r+1's loads are in flight while round r's MMAs run.
    uint4 wa0[kU], wb0[kU], x0[kU][kNT];
    uint4 wa1[kU], wb1[kU], x1[kU][kNT];
    auto load_w = [&](uint4(&da)[kU], uint4(&db)[kU], int round) {
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int ks = (round * kPerRound) + (u * kWarps) + warp;
            if (ks < ks_total) {
                da[u] = ldg_stream(w0 + (static_cast<int64_t>(ks) * kKStep));
                db[u] = ldg_stream(w1 + (static_cast<int64_t>(ks) * kKStep));
            } else {
                da[u] = make_uint4(0, 0, 0, 0);
                db[u] = make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto load_x = [&](uint4(&dx)[kU][kNT], int round) {
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int ks = (round * kPerRound) + (u * kWarps) + warp;
#pragma unroll
            for (int nt = 0; nt < kNT; ++nt)
                dx[u][nt] = (xv[nt] && ks < ks_total) ? *reinterpret_cast<const uint4*>(
                                                            xr[nt] + (static_cast<int64_t>(ks) * kKStep))
                                                      : make_uint4(0, 0, 0, 0);
        }
    };
    auto mma_round = [&](const uint4(&da)[kU], const uint4(&db)[kU], const uint4(&dx)[kU][kNT]) {
#pragma unroll
        for (int u = 0; u < kU; ++u) {
#pragma unroll
            for (int nt = 0; nt < kNT; ++nt) {
                mma_e4m3(acc[nt], da[u].x, db[u].x, da[u].y, db[u].y, dx[u][nt].x, dx[u][nt].y);
                mma_e4m3(acc[nt], da[u].z, db[u].z, da[u].w, db[u].w, dx[u][nt].z, dx[u][nt].w);
            }
        }
    };
    // Weights are immutable: round 0 (and 1) stream in before the grid dependency (pdl_device.cuh).
    load_w(wa0, wb0, 0);
    if (rounds > 1)
        load_w(wa1, wb1, 1);
    pdl_wait();
    load_x(x0, 0);
    if (rounds > 1)
        load_x(x1, 1);
    for (int round = 0; round < rounds; round += 2) {
        mma_round(wa0, wb0, x0);
        if (round + 2 < rounds) {
            load_w(wa0, wb0, round + 2);
            load_x(x0, round + 2);
        }
        if (round + 1 >= rounds)
            break;
        mma_round(wa1, wb1, x1);
        if (round + 3 < rounds) {
            load_w(wa1, wb1, round + 3);
            load_x(x1, round + 3);
        }
    }

    // C fragment: c0/c1 row g, tokens t4*2, t4*2+1; c2/c3 row g+8.
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
        const int tok = (nt * 8) + (t4 * 2);
        red[warp][tok][g] = acc[nt][0];
        red[warp][tok + 1][g] = acc[nt][1];
        red[warp][tok][g + 8] = acc[nt][2];
        red[warp][tok + 1][g + 8] = acc[nt][3];
    }
    __syncthreads();
    const float scale = *a_scale * *w_scale;
    for (int i = threadIdx.x; i < kNT * 8 * kRows; i += kWarps * 32) {
        const int tok = i / kRows;
        const int r = i % kRows;
        if (tok >= M)
            continue;
        float s = 0.0f;
#pragma unroll
        for (int ww = 0; ww < kWarps; ++ww)
            s += red[ww][tok][r];
        half* o = y + (static_cast<int64_t>(tok) * N) + n0 + r;
        float v = s * scale;
        if (beta != 0.0f)
            v += beta * __half2float(*o);
        *o = __float2half(v);
    }
}

template <int kNT, int kWarps, int kU>
__global__ void __launch_bounds__(kWarps * 32) fp8_smallm_kernel(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ w, const float* __restrict__ a_scale,
    const float* __restrict__ w_scale, float beta, half* __restrict__ y, int M, int N, int K) {
    fp8_smallm_cta<kNT, kWarps, kU>(a, w, a_scale, w_scale, beta, y, M, N, K, blockIdx.x * kRows);
}

// Up to kFp8SmallMMaxSiblings weights on one activation (q|k|v, gate|up), beta 0: grid.x walks their
// 16-row tiles back to back; constant-index selects keep the argument struct out of local memory.
struct MultiArgs {
    const uint8_t* w[kFp8SmallMMaxSiblings];
    const float* ws[kFp8SmallMMaxSiblings];
    half* y[kFp8SmallMMaxSiblings];
    int N[kFp8SmallMMaxSiblings];
    int tiles_end[kFp8SmallMMaxSiblings];  // inclusive prefix sums of 16-row tiles
};

template <int kNT, int kWarps, int kU>
__global__ void __launch_bounds__(kWarps * 32) fp8_smallm_multi_kernel(MultiArgs p,
                                                                       const uint8_t* __restrict__ a,
                                                                       const float* __restrict__ a_scale,
                                                                       int M, int K) {
    const int t = blockIdx.x;
    const bool t1 = t >= p.tiles_end[0];
    const bool t2 = t >= p.tiles_end[1];
    const int first = t2 ? p.tiles_end[1] : (t1 ? p.tiles_end[0] : 0);
    const uint8_t* w = t2 ? p.w[2] : (t1 ? p.w[1] : p.w[0]);
    const float* ws = t2 ? p.ws[2] : (t1 ? p.ws[1] : p.ws[0]);
    half* y = t2 ? p.y[2] : (t1 ? p.y[1] : p.y[0]);
    const int N = t2 ? p.N[2] : (t1 ? p.N[1] : p.N[0]);
    fp8_smallm_cta<kNT, kWarps, kU>(a, w, a_scale, ws, 0.0f, y, M, N, K, (t - first) * kRows);
}

template <int kNT, int kWarps, int kU>
void launch_multi(const MultiArgs& p, int tiles, const void* a, const float* as, int M, int K,
                  cudaStream_t stream) {
    auto* k = fp8_smallm_multi_kernel<kNT, kWarps, kU>;
    pdl::enable_kernel(k);
    pdl::launch(k, dim3(tiles), dim3(kWarps * 32), size_t(0), stream, p, static_cast<const uint8_t*>(a), as,
                M, K);
    IMP_CUDA_CHECK_LAUNCH();
}

template <int kNT, int kWarps, int kU>
void launch(const void* a, const void* w, const float* as, const float* ws, float beta, half* y, int M, int N,
            int K, cudaStream_t stream) {
    auto* k = fp8_smallm_kernel<kNT, kWarps, kU>;
    pdl::enable_kernel(k);
    pdl::launch(k, dim3(N / kRows), dim3(kWarps * 32), size_t(0), stream, static_cast<const uint8_t*>(a),
                static_cast<const uint8_t*>(w), as, ws, beta, y, M, N, K);
    IMP_CUDA_CHECK_LAUNCH();
}

template <int kWarps, int kU>
void launch_nt(const void* a, const void* w, const float* as, const float* ws, float beta, half* y, int M,
               int N, int K, cudaStream_t stream) {
    if (M <= 8)
        launch<1, kWarps, kU>(a, w, as, ws, beta, y, M, N, K, stream);
    else if (M <= 16)
        launch<2, kWarps, kU>(a, w, as, ws, beta, y, M, N, K, stream);
    else
        launch<4, kWarps, kU>(a, w, as, ws, beta, y, M, N, K, stream);
}

bool args_ok(const void* a, const void* w, const float* as, const float* ws, const half* y, int M, int N,
             int K) {
    if (M < 1 || M > kFp8SmallMMaxM || N < kRows || (N % kRows) != 0 || K < kKStep || (K % kKStep) != 0)
        return false;
    if (!a || !w || !as || !ws || !y)
        return false;
    return (reinterpret_cast<uintptr_t>(a) % 16) == 0 && (reinterpret_cast<uintptr_t>(w) % 16) == 0;
}

}  // namespace

bool gemm_fp8_smallm_tuned(const void* a_fp8, const void* w_fp8, const float* a_scale, const float* w_scale,
                           float beta, half* y, int M, int N, int K, cudaStream_t stream, int warps,
                           int unroll) {
    if (!args_ok(a_fp8, w_fp8, a_scale, w_scale, y, M, N, K))
        return false;
    if (warps == 8 && unroll == 1)
        launch_nt<8, 1>(a_fp8, w_fp8, a_scale, w_scale, beta, y, M, N, K, stream);
    else if (warps == 8 && unroll == 2)
        launch_nt<8, 2>(a_fp8, w_fp8, a_scale, w_scale, beta, y, M, N, K, stream);
    else if (warps == 8 && unroll == 4)
        launch_nt<8, 4>(a_fp8, w_fp8, a_scale, w_scale, beta, y, M, N, K, stream);
    else
        return false;
    return true;
}

bool gemm_fp8_smallm(const void* a_fp8, const void* w_fp8, const float* a_scale, const float* w_scale,
                     float beta, half* y, int M, int N, int K, cudaStream_t stream) {
    return gemm_fp8_smallm_tuned(a_fp8, w_fp8, a_scale, w_scale, beta, y, M, N, K, stream, kDefaultWarps,
                                 kDefaultUnroll);
}

bool gemm_fp8_smallm_multi(const Fp8SmallMSibling* t, int count, const void* a_fp8, const float* a_scale,
                           int M, int K, cudaStream_t stream) {
    if (count < 2 || count > kFp8SmallMMaxSiblings)
        return false;
    MultiArgs p{};
    int tiles = 0;
    for (int i = 0; i < count; ++i) {
        if (!args_ok(a_fp8, t[i].w, a_scale, t[i].w_scale, t[i].y, M, t[i].N, K))
            return false;
        p.w[i] = static_cast<const uint8_t*>(t[i].w);
        p.ws[i] = t[i].w_scale;
        p.y[i] = t[i].y;
        p.N[i] = t[i].N;
        tiles += t[i].N / kRows;
        p.tiles_end[i] = tiles;
    }
    for (int i = count; i < kFp8SmallMMaxSiblings; ++i)
        p.tiles_end[i] = tiles;
    if (M <= 8)
        launch_multi<1, kDefaultWarps, kDefaultUnroll>(p, tiles, a_fp8, a_scale, M, K, stream);
    else if (M <= 16)
        launch_multi<2, kDefaultWarps, kDefaultUnroll>(p, tiles, a_fp8, a_scale, M, K, stream);
    else
        launch_multi<4, kDefaultWarps, kDefaultUnroll>(p, tiles, a_fp8, a_scale, M, K, stream);
    return true;
}

}  // namespace imp
