// Multi-row dp4a GEMV (1..4 activation rows) on GGUF blocks, generic over DequantTraits: the weight is read
// once per call; each row's dot product re-decodes the block (dp4a_block), cheap next to the DRAM read.

#include "compute/gemv_dp4a_rows.h"

#include "compute/gemv_dp4a_iq4_traits.cuh"
#include "compute/gemv_dp4a_traits.cuh"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kRowsMax = 4;

template <typename QT, bool ADD>
__global__ void __launch_bounds__(128) gemv_dp4a_rows_kernel(const uint8_t* __restrict__ W,
                                                             const block_q8_1* __restrict__ q8_1,
                                                             const float* __restrict__ d8, half* y, int N,
                                                             int K, int rows) {
    static_assert(!QT::kNeedsQ8Sum, "multi-row GEMV passes no q8 sum");
    constexpr int kWarps = 4;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x;
    if (row >= N)
        return;
    const int total_q8 = K / 32;
    const uint8_t* row_w = W + (static_cast<size_t>(row) * (K / QT::kBlockElems) * QT::kBlockBytes);

    float sum[kRowsMax] = {};
    for (int b = (warp_id * 32) + lane; b < total_q8; b += kWarps * 32) {
        const uint8_t* bp = row_w + (static_cast<size_t>(b / QT::kQ8PerWeight) * QT::kBlockBytes);
        const int sub = b % QT::kQ8PerWeight;
#pragma unroll
        for (int r = 0; r < kRowsMax; ++r) {
            if (r < rows) {
                const size_t a = (static_cast<size_t>(r) * total_q8) + b;
                int xi[8];
                load_q8_1_qs(xi, q8_1 + a);
                sum[r] += QT::dp4a_block(bp, sub, xi, d8[a], 0.0f);
            }
        }
    }
    __shared__ float partial[kWarps][kRowsMax];
#pragma unroll
    for (int r = 0; r < kRowsMax; ++r) {
        float s = sum[r];
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(0xFFFFFFFFu, s, off);
        if (lane == 0)
            partial[warp_id][r] = s;
    }
    __syncthreads();
    if (threadIdx.x < rows) {
        const int r = static_cast<int>(threadIdx.x);
        float total = partial[0][r] + partial[1][r] + partial[2][r] + partial[3][r];
        half* o = y + (static_cast<size_t>(r) * N) + row;
        if constexpr (ADD)
            total += __half2float(*o);
        *o = __float2half(total);
    }
}

template <typename QT>
bool launch_rows(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int N, int K, int rows,
                 bool add, cudaStream_t stream) {
    if (K % QT::kBlockElems != 0)
        return false;
    const auto* w = static_cast<const uint8_t*>(W);
    if (add)
        gemv_dp4a_rows_kernel<QT, true><<<N, 128, 0, stream>>>(w, q8_1, d8, y, N, K, rows);
    else
        gemv_dp4a_rows_kernel<QT, false><<<N, 128, 0, stream>>>(w, q8_1, d8, y, N, K, rows);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool gemv_dp4a_rows(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8, half* y, int N,
                    int K, int rows, bool add, cudaStream_t stream) {
    if (rows < 1 || rows > kRowsMax || N < 1)
        return false;
    switch (qtype) {
        case QType::Q8_0:
            return launch_rows<Q8_0_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q4_0:
            return launch_rows<Q4_0_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q2_K:
            return launch_rows<Q2_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q3_K:
            return launch_rows<Q3_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q4_K:
            return launch_rows<Q4_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q5_K:
            return launch_rows<Q5_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::Q6_K:
            return launch_rows<Q6_K_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::IQ4_NL:
            return launch_rows<IQ4_NL_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        case QType::IQ4_XS:
            return launch_rows<IQ4_XS_Traits>(W, q8_1, d8, y, N, K, rows, add, stream);
        default:
            return false;
    }
}

}  // namespace imp
