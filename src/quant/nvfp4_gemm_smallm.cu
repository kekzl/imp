// nvfp4_gemm_smallm.cu — small-M NVFP4 GEMM: y[m, n] = W_nvfp4[n, :] @ x[m, :]
// for M <= 32 activation rows, W in the PLAIN layout (packed nibbles + linear
// FP8 micro-scales), x in FP16. W4A16, the Marlin recipe: dequantize the
// weight tile to FP16 in shared memory (scales folded in during dequant) and
// run plain FP16 tensor-core MMAs — no block-scaled MMA, so no 128-row SF
// atom constraint on the tile shape.
//
// Why it exists (docs/plans/2026-08-24-qwen38-port.md, 2026-08-25): batched
// decode at n_seq<=32 runs its projections through the CUTLASS 128x128x128
// block-scaled tile — 40 CTAs on the N=5120 shapes, measured 41.4 us for a
// 14 MB weight read (19% of the bandwidth floor). The cross-engine profile
// showed vLLM's Marlin doing the same class of work at ~24% less kernel
// time on this very card. Prior iterations of this file (v1-v5, preserved
// in /mnt/data/imp-scratch/) established that a SIMT FMA inner loop caps at
// ~68 us (smem+ALU bound); this version replaces the inner loop with
// wmma 16x16x16 HMMA so compute stops being the wall and the kernel becomes
// what it should be: a weight stream.
//
// Block layout: kThreads = 128 (4 warps), block tile M(32) x kNR(32) outputs,
// each warp owns one 16x16 sub-tile. Per K-tile (kKT elements) the block
// stages the x tile [32, kKT] and the DEQUANTIZED weight tile [kNR, kKT]
// (FP16, scale applied) in shared memory, then each warp runs kKT/16 MMAs.
// Weight bytes are read from DRAM exactly once per GEMM; the x tile re-reads
// from L2 per block. Global loads for tile t+1 prefetch into registers over
// the MMA stream of tile t (v4's double buffering, kept).

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "quant/fp8_utils.cuh"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <mma.h>

namespace imp {
namespace {

using namespace nvcuda;

// x is re-read by every block from L2 (the whole [M, K] tile is ~327 KiB of
// a 96 MiB L2); mark its loads evict-last so the once-only weight stream
// cannot push it out. The complementary half of the __ldcs on the weights.
// Blackwell's evict_last needs 256-bit vectors (v8.b32 / v4.b64).
__device__ __forceinline__ void ldg_evict_last_256(const void* p, uint4& a, uint4& b) {
    asm volatile("ld.global.L2::evict_last.v4.b64 {%0,%1,%2,%3}, [%4];"
                 : "=l"(*reinterpret_cast<unsigned long long*>(&a.x)),
                   "=l"(*reinterpret_cast<unsigned long long*>(&a.z)),
                   "=l"(*reinterpret_cast<unsigned long long*>(&b.x)),
                   "=l"(*reinterpret_cast<unsigned long long*>(&b.z))
                 : "l"(p));
}

constexpr int kSmM = 32;        // activation rows per launch (M tile)
constexpr int kNR = 32;         // weight rows per block (N tile)
constexpr int kKT = 128;        // K elements per tile (8 micro-blocks/row)
constexpr int kThreads = 128;   // 4 warps = 4 16x16 output sub-tiles
constexpr int kMbPerTile = kKT / kMicroBlockSize;  // 16
constexpr int kXPad = 8;        // pads keep uint4 stores aligned; wmma ldm
constexpr int kWPad = 8;        // takes the stride either way
constexpr int kSplitK = 3;      // grid.y: K-range splits. 160 blocks starved
                                // 170 SMs at 8.4% occupancy (ncu 2026-08-25);
                                // 640 fill them. Deterministic two-kernel
                                // reduction, not atomics.

// smem: x tile 32*136*2 = 8.5 KiB, w tile 32*136*2 = 8.5 KiB,
// output staging 32*32*2 = 2 KiB. ~19 KiB — 2+ blocks per SM.
__global__ void __launch_bounds__(kThreads) gemm_nvfp4_smallm_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, float* __restrict__ ws_partials, int M, int N_out, int K) {
    const int n_base = blockIdx.x * kNR;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / 32;

    __shared__ half s_x[kSmM][kKT + kXPad];
    __shared__ half s_w[kNR][kKT + kWPad];
    __shared__ half s_out[kSmM][kNR];

    // Warp -> 16x16 output sub-tile.
    const int warp_m = warp / (kNR / 16);
    const int warp_n = warp % (kNR / 16);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int n_tiles = K / kKT;
    const int per_split = (n_tiles + kSplitK - 1) / kSplitK;
    const int kt0 = split * per_split;
    const int kt1 = min(n_tiles, kt0 + per_split);
    // Register prefetch for the NEXT tile (v4 double buffering).
    constexpr int kXVecs = (kSmM * kKT / 8) / kThreads;   // uint4 x-loads per thread
    constexpr int kWMbs = (kNR * kMbPerTile) / kThreads;  // micro-blocks per thread
    uint4 px[kXVecs];
    uint2 pw[kWMbs];
    float pcs[kWMbs];
    auto fetch = [&](int t) {
        const int k_base = t * kKT;
#pragma unroll
        for (int v = 0; v < kXVecs; v += 2) {
            // 32 B per thread (evict_last's minimum on this arch): thread
            // tid owns vectors 2*tid and 2*tid+1 of round v/2 — adjacent, so
            // one 256-bit load covers both.
            const int i = tid * 2 + (v / 2) * kThreads * 2;
            const int m = i / (kKT / 8);
            const int kv = (i % (kKT / 8)) * 8;
            if (m < M) {
                ldg_evict_last_256(x + (int64_t)m * K + k_base + kv, px[v], px[v + 1]);
            } else {
                px[v] = uint4{0, 0, 0, 0};
                px[v + 1] = uint4{0, 0, 0, 0};
            }
        }
#pragma unroll
        for (int v = 0; v < kWMbs; ++v) {
            const int wi = tid + v * kThreads;
            const int n = n_base + wi / kMbPerTile;
            const int mi = (k_base / kMicroBlockSize) + (wi % kMbPerTile);
            if (n < N_out) {
                // Streaming load (evict-first): the weight bytes are read
                // exactly once, and letting them age normally in L2 evicts
                // the x tile that every block re-reads — measured as a
                // bimodal 23/59 us kernel until the hint went in.
                pw[v] = __ldcs(reinterpret_cast<const uint2*>(packed_data + (int64_t)n * (K / 2) +
                                                              (int64_t)mi * 8));
                pcs[v] = tensor_scale *
                         fp8_e4m3_to_float_fast(micro_scales[(int64_t)n * (K / kMicroBlockSize) + mi]);
            } else {
                pw[v] = uint2{0, 0};
                pcs[v] = 0.0f;
            }
        }
    };
    auto stage = [&]() {
#pragma unroll
        for (int v = 0; v < kXVecs; v += 2) {
            const int i = tid * 2 + (v / 2) * kThreads * 2;
            const int m = i / (kKT / 8);
            const int kv = (i % (kKT / 8)) * 8;
            *reinterpret_cast<uint4*>(&s_x[m][kv]) = px[v];
            *reinterpret_cast<uint4*>(&s_x[m][kv + 8]) = px[v + 1];
        }
#pragma unroll
        for (int v = 0; v < kWMbs; ++v) {
            const int wi = tid + v * kThreads;
            const int row = wi / kMbPerTile;
            const int mb = wi % kMbPerTile;
            const uint8_t* pb = reinterpret_cast<const uint8_t*>(&pw[v]);
            const half2 cs2 = __float2half2_rn(pcs[v]);
            half2* dst = reinterpret_cast<half2*>(&s_w[row][mb * kMicroBlockSize]);
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                uint32_t w_fp16x2;
                asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
                    : "=r"(w_fp16x2)
                    : "r"(static_cast<uint32_t>(pb[b])));
                dst[b] = __hmul2(*reinterpret_cast<const half2*>(&w_fp16x2), cs2);
            }
        }
    };

    fetch(kt0);
    for (int t = kt0; t < kt1; ++t) {
        stage();
        __syncthreads();
        if (t + 1 < kt1)
            fetch(t + 1);  // globals for t+1 fly over the MMAs below
        // y[m, n] = sum_k x[m, k] * w[n, k]: A = s_x row_major; s_w's
        // row-major [n, k] IS B[k][n] col_major with ld = the row stride.
#pragma unroll
        for (int k0 = 0; k0 < kKT; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::load_matrix_sync(a, &s_x[warp_m * 16][k0], kKT + kXPad);
            wmma::load_matrix_sync(b, &s_w[warp_n * 16][k0], kKT + kWPad);
            wmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    // FP32 partials, one [kSmM, N_out] plane per split: y-tile exclusive per
    // (block.x, split), so the reduction is deterministic — no atomics.
    {
        __shared__ float s_acc[kSmM][kNR];
        wmma::store_matrix_sync(&s_acc[warp_m * 16][warp_n * 16], acc, kNR, wmma::mem_row_major);
        __syncthreads();
        float* plane = ws_partials + (size_t)split * kSmM * N_out;
        for (int i = tid; i < kSmM * kNR; i += kThreads) {
            const int m = i / kNR;
            const int n = i % kNR;
            const int gn = n_base + n;
            if (gn < N_out)
                plane[(int64_t)m * N_out + gn] = s_acc[m][n];
        }
    }
    (void)s_out;
}

// Reduce the kSplitK partial planes into the FP16 output. kAcc adds onto
// the existing y (the o_proj/down residual-add call sites use beta=1).
template <bool kAcc>
__global__ void smallm_splitk_reduce_kernel(const float* __restrict__ ws_partials,
                                            half* __restrict__ y, int M, int N_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N_out)
        return;
    const int m = i / N_out, n = i % N_out;
    float acc = kAcc ? __half2float(y[(int64_t)m * N_out + n]) : 0.0f;
#pragma unroll
    for (int sp = 0; sp < kSplitK; ++sp)
        acc += ws_partials[(size_t)sp * kSmM * N_out + (int64_t)m * N_out + n];
    y[(int64_t)m * N_out + n] = __float2half(acc);
}

}  // namespace

size_t gemm_nvfp4_smallm_workspace_bytes(int N_out) {
    return (size_t)kSplitK * kSmM * N_out * sizeof(float);
}

// y[m, n] = W[n, :] @ x[m, :], W plain NVFP4, x FP16 [M, K] row-major,
// y FP16 [M, N] row-major. M <= 32; K must be a multiple of 128.
// d_workspace: gemm_nvfp4_smallm_workspace_bytes(N_out) of device scratch.
bool gemm_nvfp4_smallm(const NvFP4QuantResult& W, const half* x, half* y, int M, int N_out, int K,
                       void* d_workspace, cudaStream_t stream, bool accumulate) {
    if (M <= 0 || M > kSmM || (K % kKT) != 0)
        return false;
    if (W.packed_data == nullptr || W.micro_scales == nullptr || d_workspace == nullptr)
        return false;
    const dim3 grid((N_out + kNR - 1) / kNR, kSplitK);
    gemm_nvfp4_smallm_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W.packed_data), reinterpret_cast<const uint8_t*>(W.micro_scales),
        W.tensor_scale, x, static_cast<float*>(d_workspace), M, N_out, K);
    IMP_CUDA_CHECK_LAUNCH();
    const int total = M * N_out;
    if (accumulate)
        smallm_splitk_reduce_kernel<true><<<(total + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(d_workspace), y, M, N_out);
    else
        smallm_splitk_reduce_kernel<false><<<(total + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(d_workspace), y, M, N_out);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
