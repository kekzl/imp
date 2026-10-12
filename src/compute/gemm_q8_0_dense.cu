// Dense Q8_0 x Q8_1 dp4a GEMM for small M: GGUF 34-B blocks read in place, weight read once per
// TILE_M rows. Replaces whole-weight dequant + cuBLAS on uncached Q8_0 decode rows (#2688).

#include "compute/gemm.h"
#include "compute/gemm_q8_0_dense.h"
#include "core/logging.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace imp {

namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kBlockBytes = 34;  // fp16 d + 32 x int8
constexpr size_t kSmemLimit = 101376;

size_t smem_bytes(int tile_m, int q8_per_row) {
    return static_cast<size_t>(tile_m) * q8_per_row * (32 + sizeof(float));
}

// One warp per output column; lane l walks blocks l, l+32, ... of that weight row.
template <int TILE_M, bool ACCUM>
__global__ void __launch_bounds__(kThreads, 1) gemm_q8_0_dp4a_dense_kernel(const uint8_t* __restrict__ w,
                                                                           const block_q8_1* __restrict__ q8,
                                                                           const float* __restrict__ d8,
                                                                           half* __restrict__ out, int M,
                                                                           int N, int q8_per_row) {
    extern __shared__ char smem_raw[];
    int8_t* smem_qs = reinterpret_cast<int8_t*>(smem_raw);
    float* smem_d8 = reinterpret_cast<float*>(smem_raw + static_cast<ptrdiff_t>(TILE_M) * q8_per_row * 32);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n = blockIdx.x * kWarps + warp;
    const bool valid = n < N;  // no early return: every thread loads its share of the tile
    const uint8_t* w_row = w + static_cast<size_t>(valid ? n : 0) * q8_per_row * kBlockBytes;

    for (int m_base = 0; m_base < M; m_base += TILE_M) {
        const int m_count = min(TILE_M, M - m_base);
        for (int i = threadIdx.x; i < m_count * q8_per_row; i += kThreads) {
            const int mi = i / q8_per_row;
            const int qi = i % q8_per_row;
            const block_q8_1& src = q8[static_cast<size_t>(m_base + mi) * q8_per_row + qi];
            int4* dst = reinterpret_cast<int4*>(smem_qs + static_cast<ptrdiff_t>(i) * 32);
            dst[0] = *reinterpret_cast<const int4*>(src.qs);
            dst[1] = *reinterpret_cast<const int4*>(src.qs + 16);
            smem_d8[i] = d8[static_cast<size_t>(m_base + mi) * q8_per_row + qi];
        }
        __syncthreads();

        float acc[TILE_M];
#pragma unroll
        for (int mi = 0; mi < TILE_M; ++mi)
            acc[mi] = 0.0f;

        if (valid) {
            for (int qi = lane; qi < q8_per_row; qi += 32) {
                // 34-B blocks: 2-B aligned only, so 16-bit loads.
                const uint16_t* bp = reinterpret_cast<const uint16_t*>(w_row +
                                                                       static_cast<size_t>(qi) * kBlockBytes);
                const float dw = __half2float(__ushort_as_half(bp[0]));
                int wq[8];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    wq[j] = static_cast<int>(static_cast<uint32_t>(bp[1 + 2 * j]) |
                                             (static_cast<uint32_t>(bp[2 + 2 * j]) << 16));
#pragma unroll
                for (int mi = 0; mi < TILE_M; ++mi) {
                    if (mi >= m_count)
                        break;
                    const int idx = mi * q8_per_row + qi;
                    const int4* xv = reinterpret_cast<const int4*>(smem_qs +
                                                                   static_cast<ptrdiff_t>(idx) * 32);
                    const int4 x0 = xv[0];
                    const int4 x1 = xv[1];
                    int s = 0;
                    s = __dp4a(wq[0], x0.x, s);
                    s = __dp4a(wq[1], x0.y, s);
                    s = __dp4a(wq[2], x0.z, s);
                    s = __dp4a(wq[3], x0.w, s);
                    s = __dp4a(wq[4], x1.x, s);
                    s = __dp4a(wq[5], x1.y, s);
                    s = __dp4a(wq[6], x1.z, s);
                    s = __dp4a(wq[7], x1.w, s);
                    acc[mi] += dw * smem_d8[idx] * static_cast<float>(s);
                }
            }
#pragma unroll
            for (int mi = 0; mi < TILE_M; ++mi)
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc[mi] += __shfl_down_sync(0xFFFFFFFFu, acc[mi], off);
            if (lane == 0) {
                for (int mi = 0; mi < m_count; ++mi) {
                    half* o = out + static_cast<size_t>(m_base + mi) * N + n;
                    *o = __float2half(ACCUM ? __half2float(*o) + acc[mi] : acc[mi]);
                }
            }
        }
        __syncthreads();
    }
}

template <int TILE_M, bool ACCUM>
void launch(const uint8_t* w, const block_q8_1* q8, const float* d8, half* out, int M, int N, int q8_per_row,
            cudaStream_t stream) {
    const size_t smem = smem_bytes(TILE_M, q8_per_row);
    static size_t configured = 0;  // per instantiation
    if (smem > configured && smem > size_t{48} * 1024) {
        IMP_CUDA_CHECK_LOG(cudaFuncSetAttribute(gemm_q8_0_dp4a_dense_kernel<TILE_M, ACCUM>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(smem)));
        configured = smem;
    }
    const dim3 grid((N + kWarps - 1) / kWarps);
    gemm_q8_0_dp4a_dense_kernel<TILE_M, ACCUM>
        <<<grid, kThreads, smem, stream>>>(w, q8, d8, out, M, N, q8_per_row);
    IMP_CUDA_CHECK_LAUNCH();
}

template <bool ACCUM>
void launch_tiled(int tile_m, const uint8_t* w, const block_q8_1* q8, const float* d8, half* out, int M,
                  int N, int q8_per_row, cudaStream_t stream) {
    if (tile_m == 16)
        launch<16, ACCUM>(w, q8, d8, out, M, N, q8_per_row, stream);
    else if (tile_m == 8)
        launch<8, ACCUM>(w, q8, d8, out, M, N, q8_per_row, stream);
    else
        launch<4, ACCUM>(w, q8, d8, out, M, N, q8_per_row, stream);
}

}  // namespace

int gemm_q8_0_dp4a_dense_tile_m(int K) {
    for (int t : {16, 8, 4})
        if (smem_bytes(t, K / 32) <= kSmemLimit)
            return t;
    return 0;
}

bool gemm_q8_0_dp4a_dense(const void* w_q8_0, const half* x, half* out, void* q8_scratch, float* d8_scratch,
                          int M, int N, int K, float beta, cudaStream_t stream) {
    const int tile_m = gemm_q8_0_dp4a_dense_tile_m(K);
    if (M <= 0 || N <= 0 || K % 32 != 0 || tile_m == 0 || (beta != 0.0f && beta != 1.0f))
        return false;
    auto* q8 = reinterpret_cast<block_q8_1*>(q8_scratch);
    quantize_fp16_to_q8_1(x, q8, d8_scratch, M * K, stream);
    const auto* w = static_cast<const uint8_t*>(w_q8_0);
    if (beta == 1.0f)
        launch_tiled<true>(tile_m, w, q8, d8_scratch, out, M, N, K / 32, stream);
    else
        launch_tiled<false>(tile_m, w, q8, d8_scratch, out, M, N, K / 32, stream);
    return true;
}

}  // namespace imp
