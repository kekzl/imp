#include "compute/rms_norm_rows.h"

#include "core/logging.h"

namespace imp {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float weight_f(half v) { return __half2float(v); }
__device__ __forceinline__ float weight_f(float v) { return v; }

// One block per row. The sum: each thread's strided squares, an xor-shuffle per warp, the warp sums in
// warp order; the arithmetic of inkling_ops.cu rms_norm_k and dspark_rmsnorm_f32_in_kernel, which it
// replaces.
template <typename W>
__global__ void rms_norm_rows_k(const float* x, const W* w, float* of, half* oh, int H, float eps) {
    __shared__ float part[kThreads / 32];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        ss += xr[i] * xr[i];
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0)
        part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < static_cast<int>(blockDim.x >> 5); ++k)
        tot += part[k];
    const float r = rsqrtf(tot / static_cast<float>(H) + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = xr[i] * r * (w != nullptr ? weight_f(w[i]) : 1.0f);
        if (of != nullptr)
            of[static_cast<size_t>(blockIdx.x) * H + i] = v;
        if (oh != nullptr)
            oh[static_cast<size_t>(blockIdx.x) * H + i] = __float2half(v);
    }
}

template <typename W>
void launch(const float* x, const W* w, float* out_f, half* out_h, int rows, int H, float eps,
            cudaStream_t s) {
    if (rows <= 0 || H <= 0)
        return;
    rms_norm_rows_k<W><<<rows, kThreads, 0, s>>>(x, w, out_f, out_h, H, eps);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace

void rms_norm_rows(const float* x, const float* w, float* out_f, half* out_h, int rows, int H, float eps,
                   cudaStream_t stream) {
    launch(x, w, out_f, out_h, rows, H, eps, stream);
}

void rms_norm_rows(const float* x, const half* w, float* out_f, half* out_h, int rows, int H, float eps,
                   cudaStream_t stream) {
    launch(x, w, out_f, out_h, rows, H, eps, stream);
}

}  // namespace imp
