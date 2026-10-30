// Mixed-type multi-weight dp4a GEMV (decode, M=1): several GGUF weights that read the same activation in
// one launch and one q8_1 quantize. UD GGUF GDN layers mix types (in_proj Q5_K / Q4_K / IQ4_XS, alpha and
// beta Q8_0), so the load-time packed input weight (one qtype) is refused and four GEMVs + four quantizes
// ran per layer: alpha/beta (48 rows each) took 4.6 us per launch on Qwen3.8-27B-UD-Q4_K_M.

#include "compute/gemv_dp4a_multi.h"

#include "compute/gemm.h"
#include "compute/gemv_dp4a_iq3s_traits.cuh"
#include "compute/gemv_dp4a_iq4_traits.cuh"
#include "compute/gemv_dp4a_traits.cuh"
#include "core/logging.h"

#include <cuda_fp16.h>

namespace imp {
namespace {

struct MultiArgs {
    const uint8_t* w[kDp4aMultiMaxSegs];
    half* y[kDp4aMultiMaxSegs];
    int rows[kDp4aMultiMaxSegs];
    int qtype[kDp4aMultiMaxSegs];
    int n;
    int K;
};

constexpr int kWarps = 4;

// Thread partial of one row: the gemv_dp4a_kpar_kernel loop.
template <typename QT>
__device__ __forceinline__ float kpar_row(const uint8_t* __restrict__ W, int row,
                                          const block_q8_1* __restrict__ q8_1, const float* __restrict__ d8,
                                          int K) {
    const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;
    const size_t row_bytes = static_cast<size_t>(K / QT::kBlockElems) * QT::kBlockBytes;
    const uint8_t* row_w = W + (static_cast<size_t>(row) * row_bytes);
    float sum = 0.0f;
    constexpr int kStride = kWarps * 32;
    for (int b = static_cast<int>(threadIdx.x); b < total_q8; b += kStride) {
        int xi[8];
        load_q8_1_qs(xi, q8_1 + b);
        const float dq = d8[b];
        float q8_sum = 0.0f;
        if constexpr (QT::kNeedsQ8Sum)
            q8_sum = __half2float(q8_1[b].s);
        const int wb = b / QT::kQ8PerWeight;
        const int sub = b % QT::kQ8PerWeight;
        sum += QT::dp4a_block(row_w + (static_cast<size_t>(wb) * QT::kBlockBytes), sub, xi, dq, q8_sum);
    }
    return sum;
}

__device__ __forceinline__ float row_dispatch(int qtype, const uint8_t* W, int row, const block_q8_1* q8_1,
                                              const float* d8, int K) {
    switch (static_cast<QType>(qtype)) {
        case QType::Q4_0:
            return kpar_row<Q4_0_Traits>(W, row, q8_1, d8, K);
        case QType::Q4_K:
            return kpar_row<Q4_K_Traits>(W, row, q8_1, d8, K);
        case QType::Q5_K:
            return kpar_row<Q5_K_Traits>(W, row, q8_1, d8, K);
        case QType::Q6_K:
            return kpar_row<Q6_K_Traits>(W, row, q8_1, d8, K);
        case QType::Q2_K:
            return kpar_row<Q2_K_Traits>(W, row, q8_1, d8, K);
        case QType::Q3_K:
            return kpar_row<Q3_K_Traits>(W, row, q8_1, d8, K);
        case QType::IQ4_NL:
            return kpar_row<IQ4_NL_Traits>(W, row, q8_1, d8, K);
        case QType::IQ4_XS:
            return kpar_row<IQ4_XS_Traits>(W, row, q8_1, d8, K);
        case QType::IQ3_S:
            return kpar_row<IQ3_S_Traits>(W, row, q8_1, d8, K);
        default:
            return kpar_row<Q8_0_Traits>(W, row, q8_1, d8, K);
    }
}

// One block per row over the concatenated segments; segment and type are block-uniform.
__global__ void __launch_bounds__(kWarps * 32) gemv_dp4a_multi_kernel(const MultiArgs a,
                                                                      const block_q8_1* __restrict__ q8_1,
                                                                      const float* __restrict__ d8) {
    // Constant-index select: a runtime a.w[s] copied the whole parameter block to a 128 B local frame
    // per thread (73.0 vs 79.7 tok/s e2e).
    int row = static_cast<int>(blockIdx.x);
    const uint8_t* w = a.w[0];
    half* y = a.y[0];
    int qtype = a.qtype[0];
    int base = 0;
    int start = a.rows[0];
#pragma unroll
    for (int i = 1; i < kDp4aMultiMaxSegs; ++i) {
        if (i < a.n && row >= start) {
            w = a.w[i];
            y = a.y[i];
            qtype = a.qtype[i];
            base = start;
        }
        start += a.rows[i];
    }
    row -= base;
    float sum = row_dispatch(qtype, w, row, q8_1, d8, a.K);
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    __shared__ float partial[kWarps];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0)
        y[row] = __float2half(partial[0] + partial[1] + partial[2] + partial[3]);
}

int block_elems(QType q) {
    switch (q) {
        case QType::Q8_0:
        case QType::Q4_0:
        case QType::IQ4_NL:
            return 32;
        case QType::Q4_K:
        case QType::Q5_K:
        case QType::Q6_K:
        case QType::Q2_K:
        case QType::Q3_K:
        case QType::IQ4_XS:
        case QType::IQ3_S:
            return 256;
        default:
            return 0;
    }
}

}  // namespace

bool gemv_dp4a_multi_supports(QType qtype, int K) {
    const int be = block_elems(qtype);
    return be > 0 && K > 0 && K % be == 0;
}

bool gemv_dp4a_multi(const Dp4aGemvSeg* segs, int n_segs, const block_q8_1* q8_1, const float* d8, int K,
                     cudaStream_t stream) {
    if (n_segs < 1 || n_segs > kDp4aMultiMaxSegs || q8_1 == nullptr || d8 == nullptr || K % 32 != 0)
        return false;
    MultiArgs a{};
    int total = 0;
    for (int i = 0; i < n_segs; ++i) {
        const Dp4aGemvSeg& sg = segs[i];
        if (sg.w == nullptr || sg.y == nullptr || sg.rows <= 0 || !gemv_dp4a_multi_supports(sg.qtype, K))
            return false;
        a.w[i] = static_cast<const uint8_t*>(sg.w);
        a.y[i] = sg.y;
        a.rows[i] = sg.rows;
        a.qtype[i] = static_cast<int>(sg.qtype);
        total += sg.rows;
    }
    a.n = n_segs;
    a.K = K;
    gemv_dp4a_multi_kernel<<<total, kWarps * 32, 0, stream>>>(a, q8_1, d8);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
