// Per-row-scale FP8 E4M3 GEMM: FP32 output for the LM head (gemm.nvfp4_lm_head=fp8, #2156), FP16 for
// small-M GDN projections. y[r][row] = scale[row] * dot(W[row], x[r]). mma.sync m16n8k16 (E4M3 -> FP16 is
// exact, FP32 accumulate); one weight pass serves up to 32 activation rows. LM-head passes of <= 16 rows
// stream the weight through shared memory (fp8_head_cpasync_kernel), bit-identical to the split-K kernel.
// The FP16 entry points live in gemv_fp8_rowscale_f16.cu; this TU keeps their split-K twin.

#include "compute/gemm.h"
#include "compute/gemv_fp8_rowscale_mma.cuh"
#include "core/logging.h"
#include "core/device_info.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace imp {

namespace {

using namespace fp8_rowscale;  // NOLINT(google-build-using-namespace): the TU's own helpers

constexpr int kNTDefault = 8;  // n8 tiles per block: 64 vocab rows (32: 637.5 -> 578.0 us, #2680)

// One block = kNT * 8 vocab rows and re-reads all of x from L2; its 4 warps each take a quarter
// of K (split-K 4 regardless of n, so a wave holds ~4.6x more work units than one-warp-per-32-rows: 1.13 ->
// target 1.6 TB/s). K is walked in 64-wide chunks; lane (g = lane/4, t = lane%4) holds physical k 16t..16t+15
// of its weight row (16 B) and activation rows. MMA step s maps logical k {2t,2t+1 | 2t+8,2t+9} to physical
// 16t+4s+{0,1 | 2,3} for A and B alike, so the dot product is unchanged. Row r's MMA chain and the fixed
// w0+w1+w2+w3 reduction never depend on n_rows (#2152 rule).
template <int MT, int kNT, int kU, typename OutT>
__global__ void __launch_bounds__(kThreads) fp8_head_mma_kernel(const uint8_t* __restrict__ W,
                                                                const float* __restrict__ row_scales,
                                                                const half* __restrict__ x,
                                                                OutT* __restrict__ y, int M, int K,
                                                                int n_rows) {
    constexpr int kAcc = MT * kNT * 4;
    __shared__ float s_red[kWarps - 1][kAcc][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_base = blockIdx.x * kNT * 8;
    const int k_len = K / kWarps;
    const int k_begin = warp * k_len;

    const uint4* wp[kNT];
    bool w_ok[kNT];
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
        const int r = n_base + (8 * j) + g;
        w_ok[j] = r < M;
        wp[j] = reinterpret_cast<const uint4*>(W + (static_cast<int64_t>(w_ok[j] ? r : 0) * K));
    }
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
    const int k_end = k_begin + k_len;
    int k0 = k_begin;
    for (; k0 + (kU * 64) <= k_end; k0 += kU * 64) {
        uint4 wr[kU][kNT];
#pragma unroll
        for (int u = 0; u < kU; ++u)
#pragma unroll
            for (int j = 0; j < kNT; ++j)
                wr[u][j] = w_ok[j] ? __ldcs(wp[j] + ((k0 + (64 * u)) >> 4) + t) : zero;
#pragma unroll
        for (int u = 0; u < kU; ++u)
            mma_chunk<MT, kNT>(acc, wr[u], xp, x_ok, k0 + (64 * u), t);
    }
    for (; k0 < k_end; k0 += 64) {
        uint4 wr[kNT];
#pragma unroll
        for (int j = 0; j < kNT; ++j)
            wr[j] = w_ok[j] ? __ldcs(wp[j] + (k0 >> 4) + t) : zero;
        mma_chunk<MT, kNT>(acc, wr, xp, x_ok, k0, t);
    }

    if (warp > 0)
        stash_partials<MT, kNT>(acc, s_red[warp - 1], lane);
    __syncthreads();
    if (warp != 0)
        return;
    add_partials<MT, kNT>(acc, s_red, lane);
    store_rows<MT, kNT>(acc, row_scales, y, M, n_rows, n_base, g, t);
}

// Streamed twin: each warp owns kNT * 8 vocab rows over all of K. One warp instruction fetches 4 rows x
// 128 B (full lines) into a per-warp kS-stage cp.async ring; lanes read their 16 B fragments from shared
// memory. The split-K kernel's direct fragment load is 8 rows x 64 B (half lines): ~1390 GB/s vs ~1670.
// The four K quarters run in sequence, one chain each, folded q0 + q1 + q2 + q3 like add_partials, with
// the same per-chunk MMAs in the same order: every output bit equals fp8_head_mma_kernel's.

template <int MT, int kNT, int kS, typename OutT>
__global__ void __launch_bounds__(kThreads) fp8_head_cpasync_kernel(const uint8_t* __restrict__ W,
                                                                    const float* __restrict__ row_scales,
                                                                    const half* __restrict__ x,
                                                                    OutT* __restrict__ y, int M, int K,
                                                                    int n_rows) {
    constexpr int kR = kNT * 8;            // rows per warp
    constexpr int kLoads = kR / 4;         // cp.async per lane per stage
    constexpr int kSlot = kR * kRowPitch;  // bytes per stage
    extern __shared__ __align__(16) uint8_t smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_base = ((blockIdx.x * kWarps) + warp) * kR;
    if (n_base >= M)
        return;
    const uint32_t ring = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + (warp * kS * kSlot);
    const int n_stages = K / kStageBytes;
    const int stages_per_q = n_stages / kWarps;

    // Load lane mapping: row i*4 + lane/8, 16 B piece lane%8 of the stage's 128 B.
    const uint8_t* src[kLoads];
    uint32_t dst[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int row = (i * 4) + (lane >> 3);
        const int r = min(n_base + row, M - 1);  // clamped rows are never stored
        src[i] = W + (static_cast<int64_t>(r) * K) + static_cast<int64_t>((lane & 7) * 16);
        dst[i] = (row * kRowPitch) + ((lane & 7) * 16);
    }
    auto issue = [&](int st) {
        if (st < n_stages) {
            const uint32_t slot = ring + ((st % kS) * kSlot);
#pragma unroll
            for (int i = 0; i < kLoads; ++i)
                cp_async16(slot + dst[i], src[i] + (static_cast<int64_t>(st) * kStageBytes));
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

#pragma unroll
    for (int s = 0; s < kS - 1; ++s)
        issue(s);

    float acc[MT][kNT][4];
    float tot[MT][kNT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < kNT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[mt][j][e] = 0.0f;
    for (int st = 0; st < n_stages; ++st) {
        cp_async_wait<kS - 2>();
        __syncwarp();
        issue(st + kS - 1);  // refills the slot consumed by stage st - 1
        const int slot_off = (warp * kS * kSlot) + ((st % kS) * kSlot);
        const uint8_t* slot = smem + slot_off;
#pragma unroll
        for (int cc = 0; cc < kStageBytes / 64; ++cc) {
            uint4 wr[kNT];
#pragma unroll
            for (int j = 0; j < kNT; ++j) {
                const int off = (((8 * j) + g) * kRowPitch) + (cc * 64) + (16 * t);
                wr[j] = *reinterpret_cast<const uint4*>(slot + off);
            }
            mma_chunk<MT, kNT>(acc, wr, xp, x_ok, (st * kStageBytes) + (cc * 64), t);
        }
        __syncwarp();
        if ((st + 1) % stages_per_q == 0) {
            const bool first = st + 1 == stages_per_q;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int j = 0; j < kNT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[mt][j][e] = first ? acc[mt][j][e] : tot[mt][j][e] + acc[mt][j][e];
                        acc[mt][j][e] = 0.0f;
                    }
        }
    }
    cp_async_wait<0>();
    store_rows<MT, kNT>(tot, row_scales, y, M, n_rows, n_base, g, t);
}

// Streamed head: 32 rows per warp, 3 stages (72 KB per block). Qwen3.8 head [248320 x 5120] at 1 / 4 /
// 16 rows: 774 / 777 / 792 us vs 927 / 932 / 959 split-K; above 16 rows (two m16 tiles) split-K wins.
constexpr int kStreamNT = 4;
constexpr int kStreamStages = 3;
constexpr int kStreamMaxRows = 16;

// FP32 (LM head) only: the FP16 twin serves narrow GDN projections, where 128 rows per block underfill the
// grid.
bool stream_head_ready() {
    static const bool ok = cudaFuncSetAttribute(fp8_head_cpasync_kernel<1, kStreamNT, kStreamStages, float>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                cpasync_smem_bytes<kStreamNT, kStreamStages>()) ==
                           cudaSuccess;
    return ok;
}

// One pass of <= 16 activation rows through the streamed kernel.
void launch_stream_pass(const uint8_t* W, const float* d_row_scales, const half* x, float* y, int M, int K,
                        int n_rows, cudaStream_t stream) {
    constexpr int kRowsPerBlock = kWarps * kStreamNT * 8;
    const int blocks = (M + kRowsPerBlock - 1) / kRowsPerBlock;
    fp8_head_cpasync_kernel<1, kStreamNT, kStreamStages, float>
        <<<blocks, kThreads, cpasync_smem_bytes<kStreamNT, kStreamStages>(), stream>>>(W, d_row_scales, x, y,
                                                                                       M, K, n_rows);
    IMP_CUDA_CHECK_LAUNCH();
}

// stream_small (FP32 only): passes of <= 16 rows take the streamed kernel when K % 512 == 0; same bits.
template <int kNT, int kU, typename OutT>
bool launch_fp8_head(const void* W_fp8, const float* d_row_scales, const half* x, OutT* y, int M, int K,
                     int n_rows, cudaStream_t stream, bool stream_small = false) {
    if (!W_fp8 || !d_row_scales || !x || !y || M <= 0 || K <= 0 || (K % (64 * kWarps)) != 0 || n_rows <= 0)
        return false;
    const auto* W = static_cast<const uint8_t*>(W_fp8);
    bool streamed = false;
    if constexpr (std::is_same_v<OutT, float>)
        streamed = stream_small && (K % (kStageBytes * kWarps)) == 0 && stream_head_ready();
    const int blocks = (M + (kNT * 8) - 1) / (kNT * 8);
    for (int r0 = 0; r0 < n_rows; r0 += kRowsPerPass) {
        const int nb = std::min(kRowsPerPass, n_rows - r0);
        const half* xr = x + (static_cast<int64_t>(r0) * K);
        OutT* yr = y + static_cast<int64_t>(r0) * M;
        if constexpr (std::is_same_v<OutT, float>) {
            if (streamed && nb <= kStreamMaxRows) {
                launch_stream_pass(W, d_row_scales, xr, yr, M, K, nb, stream);
                continue;
            }
        }
        if (nb <= 16)
            fp8_head_mma_kernel<1, kNT, kU, OutT>
                <<<blocks, kThreads, 0, stream>>>(W, d_row_scales, xr, yr, M, K, nb);
        else
            fp8_head_mma_kernel<2, kNT, kU, OutT>
                <<<blocks, kThreads, 0, stream>>>(W, d_row_scales, xr, yr, M, K, nb);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}

}  // namespace

bool gemv_fp8_rowscale_fp32(const void* W_fp8, const float* d_row_scales, const half* x, float* y, int M,
                            int K, int n_rows, cudaStream_t stream) {
    return launch_fp8_head<kNTDefault, 2>(W_fp8, d_row_scales, x, y, M, K, n_rows, stream,
                                          /*stream_small=*/true);
}

// Narrow weights (GDN out: 2560 rows = 40 blocks at kNT 8) take fewer rows per block so a wave
// still fills every SM twice; the per-row MMA chain is the same for every kNT.
bool fp8_rowscale::split_k_f16(const void* W_fp8, const float* d_row_scales, const half* x, half* y, int M,
                               int K, int n_rows, cudaStream_t stream) {
    const int min_blocks = 2 * device_sm_count();  // 340 on RTX 5090 (170 SMs)
    if ((M + 63) / 64 >= min_blocks)
        return launch_fp8_head<8, 2>(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
    if ((M + 31) / 32 >= min_blocks)
        return launch_fp8_head<4, 4>(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
    return launch_fp8_head<2, 4>(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
}

// Tuning hook (tests only): nt 4 / 8 = split-K kernel only (kNT * 8 vocab rows per block); 0 = the
// default dispatch. Every variant keeps the per-row MMA chain and quarter order: bit-identical outputs.
bool gemv_fp8_rowscale_fp32_tuned(const void* W_fp8, const float* d_row_scales, const half* x, float* y,
                                  int M, int K, int n_rows, int nt, cudaStream_t stream) {
    switch (nt) {
        case 0:
            return gemv_fp8_rowscale_fp32(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
        case 4:
            return launch_fp8_head<4, 4>(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
        case 8:
            return launch_fp8_head<8, 2>(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
        default:
            return false;
    }
}

}  // namespace imp
