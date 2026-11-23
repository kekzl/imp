// I8B32 projection for calls of up to 16 rows (gemv_i8b32.h): the streamed split-K scheme of
// gemv_fp8_rowscale_f16.cu (warp w = K quarter w, cp.async ring of whole weight lines, activation
// fragments one stage ahead) with int8 weights scaled per 32-k block before the MMA.

#include "compute/gemv_i8b32.h"

#include "compute/gemv_fp8_rowscale_mma.cuh"
#include "core/logging.h"

#include <algorithm>

namespace imp {

namespace {

using namespace fp8_rowscale;  // NOLINT(google-build-using-namespace): the shared MMA and cp.async helpers

__global__ void quantize_i8b32_k(const half* w, int8_t* q, half* scales, size_t blocks) {
    const size_t b = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (b >= blocks)
        return;
    const half* p = w + (b * 32);
    float mx = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i)
        mx = fmaxf(mx, fabsf(__half2float(p[i])));
    const half sh = __float2half(mx / 127.0f);
    const float sc = __half2float(sh);
    const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    scales[b] = sh;
#pragma unroll
    for (int i = 0; i < 32; ++i)
        q[(b * 32) + i] = static_cast<int8_t>(fminf(fmaxf(rintf(__half2float(p[i]) * inv), -127.0f), 127.0f));
}

// Two int8 (low 16 bits) -> half2, times the block scale.
__device__ __forceinline__ uint32_t i8x2_scaled(uint32_t two, __half2 s2) {
    const float lo = static_cast<float>(static_cast<int8_t>(two & 0xffu));
    const float hi = static_cast<float>(static_cast<int8_t>((two >> 8) & 0xffu));
    const __half2 h = __hmul2(__floats2half2_rn(lo, hi), s2);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// One 64-wide K chunk. Lane t holds weights 16 t .. 16 t + 15 of the chunk: the first 32-k block for
// t < 2, the second otherwise (sw = the two block scales of the lane's weight row, per tile).
template <int MT, int kNT>
__device__ __forceinline__ void mma_chunk_i8(float (&acc)[MT][kNT][4], const uint4 (&wr)[kNT],
                                             const uint32_t (&sw)[kNT], const uint4 (&xr)[MT][2][2], int t) {
    __half2 s2[kNT];
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
        const uint32_t one = t < 2 ? (sw[j] & 0xffffu) : (sw[j] >> 16);
        const uint32_t both = one | (one << 16);
        s2[j] = *reinterpret_cast<const __half2*>(&both);
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        uint32_t b0[kNT], b1[kNT];
#pragma unroll
        for (int j = 0; j < kNT; ++j) {
            const uint32_t wv = word(wr[j], s);
            b0[j] = i8x2_scaled(wv & 0xffffu, s2[j]);
            b1[j] = i8x2_scaled(wv >> 16, s2[j]);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
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

// Accumulators -> fp16 rows (no row scale: the block scales went into the weights).
template <int MT, int kNT>
__device__ __forceinline__ void store_plain(const float (&acc)[MT][kNT][4], half* __restrict__ y, int M,
                                            int n_rows, int n_base, int g, int t) {
#pragma unroll
    for (int j = 0; j < kNT; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = n_base + (8 * j) + (2 * t) + e;
            if (col >= M)
                continue;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r0 = (mt * 16) + g;
                if (r0 < n_rows)
                    y[(static_cast<int64_t>(r0) * M) + col] = __float2half(acc[mt][j][e]);
                if (r0 + 8 < n_rows)
                    y[(static_cast<int64_t>(r0 + 8) * M) + col] = __float2half(acc[mt][j][2 + e]);
            }
        }
}

// S: weight bytes per row per stage (64 or 128): K / 4 is a whole number of stages.
template <int MT, int kNT, int kS, int S>
__global__ void __launch_bounds__(kThreads) i8_proj_kernel(const uint8_t* __restrict__ W,
                                                           const half* __restrict__ scales,
                                                           const half* __restrict__ x, half* __restrict__ y,
                                                           int M, int K, int n_rows) {
    constexpr int kC = S / 64;  // MMA chunks per stage
    constexpr int kR = kNT * 8;
    constexpr int kLanes = S / 16;  // lanes per weight row of a load
    constexpr int kRowsPerLoad = 32 / kLanes;
    constexpr int kLoads = kR / kRowsPerLoad;
    constexpr int kPitch = S + (S / 2);  // rows g, g + 1 in opposite bank halves
    constexpr int kSlot = kR * kPitch;
    __shared__ float s_red[kWarps - 1][MT * kNT * 4][32];
    extern __shared__ __align__(16) uint8_t ring_smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_base = blockIdx.x * kR;
    const int k_begin = warp * (K / kWarps);
    const int n_stages = K / (kWarps * S);
    const uint32_t ring = static_cast<uint32_t>(__cvta_generic_to_shared(ring_smem)) + (warp * kS * kSlot);

    const uint8_t* src[kLoads];
    uint32_t dst[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int row = (i * kRowsPerLoad) + (lane / kLanes);
        const int r = min(n_base + row, M - 1);  // clamped rows are never stored
        src[i] = W + (static_cast<int64_t>(r) * K) + k_begin + static_cast<int64_t>((lane % kLanes) * 16);
        dst[i] = (row * kPitch) + ((lane % kLanes) * 16);
    }
    auto issue = [&](int st) {
        if (st < n_stages) {
            const uint32_t slot = ring + ((st % kS) * kSlot);
#pragma unroll
            for (int i = 0; i < kLoads; ++i)
                cp_async16(slot + dst[i], src[i] + (static_cast<int64_t>(st) * S));
        }
        cp_async_commit();
    };
    const uint4* xp[MT][2];
    bool x_ok[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = (mt * 16) + (h * 8) + g;
            x_ok[mt][h] = r < n_rows;
            xp[mt][h] = reinterpret_cast<const uint4*>(x + (static_cast<int64_t>(x_ok[mt][h] ? r : 0) * K));
        }
    // The two block scales of a 64-k chunk, per tile: this lane's weight row.
    const uint32_t* sp[kNT];
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
        const int r = min(n_base + (8 * j) + g, M - 1);
        sp[j] = reinterpret_cast<const uint32_t*>(scales + (static_cast<int64_t>(r) * (K / 32)) +
                                                  (k_begin / 32));
    }
#pragma unroll
    for (int s = 0; s < kS - 1; ++s)
        issue(s);

    float acc[MT][kNT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < kNT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[mt][j][e] = 0.0f;
    uint4 xn[kC][MT][2][2];
    uint32_t sn[kC][kNT];
    auto load_stage = [&](int st) {
#pragma unroll
        for (int cc = 0; cc < kC; ++cc) {
            load_x_chunk<MT>(xn[cc], xp, x_ok, k_begin + (st * S) + (cc * 64), t);
#pragma unroll
            for (int j = 0; j < kNT; ++j)
                sn[cc][j] = __ldg(sp[j] + (static_cast<int64_t>(st) * kC) + cc);
        }
    };
    load_stage(0);
    for (int st = 0; st < n_stages; ++st) {
        uint4 xc[kC][MT][2][2];
        uint32_t sc[kC][kNT];
#pragma unroll
        for (int cc = 0; cc < kC; ++cc) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    xc[cc][mt][h][0] = xn[cc][mt][h][0];
                    xc[cc][mt][h][1] = xn[cc][mt][h][1];
                }
#pragma unroll
            for (int j = 0; j < kNT; ++j)
                sc[cc][j] = sn[cc][j];
        }
        if (st + 1 < n_stages)
            load_stage(st + 1);
        cp_async_wait<kS - 2>();
        __syncwarp();
        issue(st + kS - 1);  // refills the slot consumed by stage st - 1
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): smem offsets < 25 KB
        const uint8_t* slot = ring_smem + (warp * kS * kSlot) + ((st % kS) * kSlot);
#pragma unroll
        for (int cc = 0; cc < kC; ++cc) {
            uint4 wr[kNT];
#pragma unroll
            for (int j = 0; j < kNT; ++j)
                // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): smem offsets < 25 KB
                wr[j] = *reinterpret_cast<const uint4*>(slot + (((8 * j) + g) * kPitch) + (cc * 64) +
                                                        (16 * t));
            mma_chunk_i8<MT, kNT>(acc, wr, sc[cc], xc[cc], t);
        }
        __syncwarp();
    }
    cp_async_wait<0>();

    if (warp > 0)
        stash_partials<MT, kNT>(acc, s_red[warp - 1], lane);
    __syncthreads();
    if (warp != 0)
        return;
    add_partials<MT, kNT>(acc, s_red, lane);
    store_plain<MT, kNT>(acc, y, M, n_rows, n_base, g, t);
}

constexpr int kI8Stages = 2;

template <int kNT, int S>
constexpr int i8_smem_bytes() {
    return kWarps * kI8Stages * kNT * 8 * (S + (S / 2));
}

template <int kNT, int S>
bool launch_i8(const uint8_t* W, const half* scales, const half* x, half* y, int M, int K, int n_rows,
               cudaStream_t stream) {
    static const bool ok = cudaFuncSetAttribute(i8_proj_kernel<1, kNT, kI8Stages, S>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                i8_smem_bytes<kNT, S>()) == cudaSuccess;
    if (!ok)
        return false;
    i8_proj_kernel<1, kNT, kI8Stages, S>
        <<<(M + (kNT * 8) - 1) / (kNT * 8), kThreads, i8_smem_bytes<kNT, S>(), stream>>>(W, scales, x, y, M,
                                                                                         K, n_rows);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

void quantize_i8b32(const half* w, int8_t* q, half* scales, size_t rows, int K, cudaStream_t stream) {
    const size_t blocks = rows * static_cast<size_t>(K / 32);
    quantize_i8b32_k<<<static_cast<unsigned>((blocks + 255) / 256), 256, 0, stream>>>(w, q, scales, blocks);
    IMP_CUDA_CHECK_LAUNCH();
}

bool gemv_i8b32_f16(const int8_t* q, const half* scales, const half* x, half* y, int M, int K, int n_rows,
                    cudaStream_t stream) {
    if (!q || !scales || !x || !y || M <= 0 || K <= 0 || K % 256 != 0 || n_rows < 1 || n_rows > 16 ||
        reinterpret_cast<uintptr_t>(q) % 16 != 0 || reinterpret_cast<uintptr_t>(scales) % 4 != 0)
        return false;
    const auto* W = reinterpret_cast<const uint8_t*>(q);
    const bool wide = K % 512 == 0;  // whole 128-byte stages per K quarter
    if (n_rows <= 8)
        return wide ? launch_i8<1, 128>(W, scales, x, y, M, K, n_rows, stream)
                    : launch_i8<1, 64>(W, scales, x, y, M, K, n_rows, stream);
    return wide ? launch_i8<2, 128>(W, scales, x, y, M, K, n_rows, stream)
                : launch_i8<2, 64>(W, scales, x, y, M, K, n_rows, stream);
}

}  // namespace imp
