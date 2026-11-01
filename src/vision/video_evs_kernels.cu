#include "vision/video_evs_kernels.h"

#include "compute/warp_reduce.cuh"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kEvsWarpsPerBlock = 8;
constexpr float kEvsFirstPairScore = 255.0f;  // vision/video_evs.h kEvsFirstGroupScore
constexpr float kEvsNormEps = 1e-8f;          // torch cosine_similarity eps

__global__ void evs_dissimilarity_kernel(const half* __restrict__ emb, int n_rows, int tpg, int dim,
                                         float* __restrict__ dissim) {
    const int row = static_cast<int>(blockIdx.x) * kEvsWarpsPerBlock + static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (row >= n_rows)
        return;
    if (row < tpg) {
        if (lane == 0)
            dissim[row] = kEvsFirstPairScore;
        return;
    }
    const half* x = emb + (static_cast<size_t>(row) * dim);
    const half* y = x - (static_cast<size_t>(tpg) * dim);
    float xy = 0.0f, xx = 0.0f, yy = 0.0f;
    if ((dim & 1) == 0) {
        const auto* x2 = reinterpret_cast<const half2*>(x);
        const auto* y2 = reinterpret_cast<const half2*>(y);
        for (int k = lane; k < dim / 2; k += 32) {
            const float2 a = __half22float2(x2[k]);
            const float2 b = __half22float2(y2[k]);
            xy += (a.x * b.x) + (a.y * b.y);
            xx += (a.x * a.x) + (a.y * a.y);
            yy += (b.x * b.x) + (b.y * b.y);
        }
    } else {
        for (int k = lane; k < dim; k += 32) {
            const float a = __half2float(x[k]);
            const float b = __half2float(y[k]);
            xy += a * b;
            xx += a * a;
            yy += b * b;
        }
    }
    xy = warp_reduce_sum(xy);
    xx = warp_reduce_sum(xx);
    yy = warp_reduce_sum(yy);
    if (lane == 0)
        dissim[row] = 1.0f - (xy / (fmaxf(sqrtf(xx), kEvsNormEps) * fmaxf(sqrtf(yy), kEvsNormEps)));
}

__global__ void gather_rows_kernel(const half* __restrict__ in, const int32_t* __restrict__ rows, int dim,
                                   half* __restrict__ out) {
    const size_t r = blockIdx.x;
    const half* src = in + (static_cast<size_t>(rows[r]) * dim);
    half* dst = out + (r * dim);
    if ((dim & 7) == 0) {
        const auto* s = reinterpret_cast<const uint4*>(src);
        auto* d = reinterpret_cast<uint4*>(dst);
        for (int k = static_cast<int>(threadIdx.x); k < dim / 8; k += static_cast<int>(blockDim.x))
            d[k] = s[k];
        return;
    }
    for (int k = static_cast<int>(threadIdx.x); k < dim; k += static_cast<int>(blockDim.x))
        dst[k] = src[k];
}

}  // namespace

void launch_evs_dissimilarity(const half* emb, int groups, int tokens_per_group, int dim, float* dissim,
                              cudaStream_t stream) {
    const int n_rows = groups * tokens_per_group;
    if (n_rows <= 0 || dim <= 0)
        return;
    const int blocks = (n_rows + kEvsWarpsPerBlock - 1) / kEvsWarpsPerBlock;
    evs_dissimilarity_kernel<<<blocks, kEvsWarpsPerBlock * 32, 0, stream>>>(emb, n_rows, tokens_per_group,
                                                                            dim, dissim);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_gather_rows(const half* in, const int32_t* rows, int n_out, int dim, half* out,
                        cudaStream_t stream) {
    if (n_out <= 0 || dim <= 0)
        return;
    gather_rows_kernel<<<n_out, 128, 0, stream>>>(in, rows, dim, out);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
