#include "compute/head_norm_rope.h"

#include "core/logging.h"

namespace imp {

namespace {

constexpr int kHeadDim = kHeadNormRopeDim;

// Per (row, head): RMSNorm over head_dim (skipped for w == nullptr), then NeoX RoPE at
// positions[row], in place. Moved from runtime/dspark_drafter.cu, arithmetic unchanged.
__global__ void head_norm_rope_kernel(half* __restrict__ x, int row_stride, int heads,
                                      const half* __restrict__ w, const int* __restrict__ pos,
                                      float log2_theta, float eps) {
    const int row = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    half* v = x + (static_cast<size_t>(row) * row_stride) + (static_cast<size_t>(head) * kHeadDim);
    const int i = threadIdx.x;  // pair (i, i + 64)
    float a = __half2float(v[i]);
    float b = __half2float(v[i + (kHeadDim / 2)]);
    if (w != nullptr) {
        float ss = (a * a) + (b * b);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            ss += __shfl_xor_sync(0xffffffffu, ss, o);
        __shared__ float part[2];
        if ((i & 31) == 0)
            part[i >> 5] = ss;
        __syncthreads();
        const float inv = rsqrtf(((part[0] + part[1]) / kHeadDim) + eps);
        a *= inv * __half2float(w[i]);
        b *= inv * __half2float(w[i + (kHeadDim / 2)]);
    }
    const float freq = exp2f(-log2_theta * (2.0f * i) / kHeadDim);
    float s = 0.0f;
    float c = 0.0f;
    sincosf(static_cast<float>(pos[row]) * freq, &s, &c);
    v[i] = __float2half((a * c) - (b * s));
    v[i + (kHeadDim / 2)] = __float2half((b * c) + (a * s));
}

}  // namespace

void head_norm_rope_fp16(half* x, int rows, int row_stride, int heads, const half* w, const int* d_pos,
                         float log2_theta, float eps, cudaStream_t stream) {
    if (rows <= 0 || heads <= 0)
        return;
    head_norm_rope_kernel<<<rows * heads, kHeadDim / 2, 0, stream>>>(x, row_stride, heads, w, d_pos,
                                                                     log2_theta, eps);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
