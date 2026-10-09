// Qwen4Exp gated residual kernels. See gated_residual.h for the math.

#include "compute/gated_residual.h"

#include <cuda_fp16.h>

#include "compute/gemm_internal_device.cuh"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxHc = 8;  // hc_read_decode_kernel stream registers

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// One block per (row, stream): RMS over d elements, then x / rms * (1 + w).
__global__ void hc_grouped_rmsnorm_kernel(const half* __restrict__ x, const half* __restrict__ w,
                                          half* __restrict__ out, int hc, int d, float eps) {
    const int row = blockIdx.x / hc;
    const int s = blockIdx.x % hc;
    const size_t base = (static_cast<size_t>(row) * hc + s) * d;
    const half* xr = x + base;
    const half* wr = w + static_cast<size_t>(s) * d;
    half* orow = out + base;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        const float v = __half2float(xr[j]);
        sum += v * v;
    }
    __shared__ float red[kThreads / 32];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0)
            red[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / static_cast<float>(d) + eps);
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        orow[j] = __float2half(__half2float(xr[j]) * inv * (1.0f + __half2float(wr[j])));
}

// Decode (n = 1) hc_read front end: grouped RMSNorm + [down; inject] GEMV + silu / 2*sigmoid.
// grid ceil((lowrank + n_inj) / 8), 256 threads, smem = K halves. Same op order as
// hc_grouped_rmsnorm_kernel + gemv_fp16_kernel (sm_120 path) + the epilogue kernels: bit-identical.
__global__ void hc_read_decode_kernel(const half* __restrict__ x, const half* __restrict__ w,
                                      const half* __restrict__ down, const half* __restrict__ inject,
                                      half* __restrict__ normed, half* __restrict__ low,
                                      half* __restrict__ inj, int hc, int d, int lowrank, int n_inj,
                                      float eps, float inv_hc) {
    extern __shared__ __align__(16) half s_x[];
    __shared__ float red[kMaxHc][kThreads / 32];
    __shared__ float inv[kMaxHc];
    const int K = hc * d;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    float part[kMaxHc];
#pragma unroll
    for (int s = 0; s < kMaxHc; ++s) {
        float sum = 0.0f;
        if (s < hc)
            for (int j = threadIdx.x; j < d; j += blockDim.x) {
                const float v = __half2float(x[static_cast<size_t>(s) * d + j]);
                sum += v * v;
            }
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        part[s] = sum;
    }
    if (lane == 0)
        for (int s = 0; s < hc; ++s)
            red[s][warp] = part[s];
    __syncthreads();
    if (warp == 0)
        for (int s = 0; s < hc; ++s) {
            float v = (lane < blockDim.x / 32) ? red[s][lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0)
                inv[s] = rsqrtf(v / static_cast<float>(d) + eps);
        }
    __syncthreads();
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        const int s = j / d;
        const half v = __float2half(__half2float(x[j]) * inv[s] * (1.0f + __half2float(w[j])));
        s_x[j] = v;
        if (blockIdx.x == 0)
            normed[j] = v;
    }
    __syncthreads();

    const int row = blockIdx.x * (kThreads / 32) + warp;
    if (row >= lowrank + n_inj)
        return;
    const half* A_row = row < lowrank ? down + static_cast<size_t>(row) * K
                                      : inject + static_cast<size_t>(row - lowrank) * K;
    const float4* A_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(s_x);
    float sum = 0.0f;
    for (int i = lane; i < K / 16; i += 32) {
        const size_t i2 = 2 * static_cast<size_t>(i);
        const float4 a0 = A_v[i2], a1 = A_v[i2 + 1];
        const float4 x0 = x_v[i2], x1 = x_v[i2 + 1];
        const half2* a0h = reinterpret_cast<const half2*>(&a0);
        const half2* x0h = reinterpret_cast<const half2*>(&x0);
        const half2* a1h = reinterpret_cast<const half2*>(&a1);
        const half2* x1h = reinterpret_cast<const half2*>(&x1);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const half2 p = __hmul2(a0h[q], x0h[q]);
            sum += __half2float(p.x) + __half2float(p.y);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const half2 p = __hmul2(a1h[q], x1h[q]);
            sum += __half2float(p.x) + __half2float(p.y);
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane != 0)
        return;
    const float y = __half2float(__float2half(sum)) * inv_hc;
    if (row < lowrank)
        low[row] = __float2half(y * sigmoidf_(y));
    else
        inj[row - lowrank] = __float2half(2.0f * sigmoidf_(y));
}

// hc_sub + hc_inject_add in one pass: hidden[t, s*d + j] += half(h - mixed)[t, j] * inj[t, s].
__global__ void hc_write_kernel(half* __restrict__ hidden, const half* __restrict__ h,
                                const half* __restrict__ mixed, const half* __restrict__ inj, int hc, int d,
                                int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int64_t t = i / (static_cast<int64_t>(hc) * d);
    const int64_t rem = i - t * hc * d;
    const int s = static_cast<int>(rem / d);
    const int j = static_cast<int>(rem - static_cast<int64_t>(s) * d);
    const int64_t k = t * d + j;
    const half out = __float2half(__half2float(h[k]) - __half2float(mixed[k]));
    const float g = __half2float(inj[t * hc + s]);
    const float o = __half2float(out);
    hidden[i] = __float2half(__half2float(hidden[i]) + o * g);
}

__global__ void hc_silu_div_kernel(half* __restrict__ x, float inv_hc, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __half2float(x[i]) * inv_hc;
        x[i] = __float2half(v * sigmoidf_(v));
    }
}

// Grid over n*d; loop over the hc streams.
__global__ void hc_mix_kernel(const half* __restrict__ mixw, const half* __restrict__ normed,
                              half* __restrict__ out, half* __restrict__ out2, int hc, int d, int64_t nd) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nd)
        return;
    const int64_t t = i / d;
    const int j = static_cast<int>(i - t * d);
    const size_t base = static_cast<size_t>(t) * hc * d + j;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s) {
        const size_t k = base + static_cast<size_t>(s) * d;
        acc += sigmoidf_(__half2float(mixw[k])) * __half2float(normed[k]);
    }
    const half r = __float2half(acc / static_cast<float>(hc));
    out[i] = r;
    if (out2)
        out2[i] = r;
}

__global__ void hc_inject_weights_kernel(half* __restrict__ inj, float inv_hc, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        inj[i] = __float2half(2.0f * sigmoidf_(__half2float(inj[i]) * inv_hc));
}

// Grid over n*hc*d.
__global__ void hc_inject_add_kernel(half* __restrict__ hidden, const half* __restrict__ out,
                                     const half* __restrict__ inj, int hc, int d, int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int64_t t = i / (static_cast<int64_t>(hc) * d);
    const int64_t rem = i - t * hc * d;
    const int s = static_cast<int>(rem / d);
    const int j = static_cast<int>(rem - static_cast<int64_t>(s) * d);
    const float g = __half2float(inj[t * hc + s]);
    const float o = __half2float(out[t * d + j]);
    hidden[i] = __float2half(__half2float(hidden[i]) + o * g);
}

__global__ void hc_sub_kernel(const half* __restrict__ a, const half* __restrict__ b, half* __restrict__ out,
                              int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(__half2float(a[i]) - __half2float(b[i]));
}

__global__ void hc_repeat_kernel(const half* __restrict__ src, half* __restrict__ dst, int hc, int d,
                                 int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int64_t t = i / (static_cast<int64_t>(hc) * d);
    const int j = static_cast<int>((i - t * hc * d) % d);
    dst[i] = src[t * d + j];
}

inline int blocks_for(int64_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

}  // namespace

void hc_grouped_rmsnorm(const Tensor& x, const Tensor& w, Tensor& out, int hc, int d, float eps,
                        cudaStream_t stream) {
    const int rows = static_cast<int>(x.shape[0]);
    if (rows == 0)
        return;
    hc_grouped_rmsnorm_kernel<<<rows * hc, kThreads, 0, stream>>>(
        static_cast<const half*>(x.data), static_cast<const half*>(w.data), static_cast<half*>(out.data), hc, d,
        eps);
    IMP_CUDA_CHECK_LAUNCH();
}

void hc_silu_div(Tensor& x, int hc, cudaStream_t stream) {
    const int64_t n = x.numel();
    if (n == 0)
        return;
    hc_silu_div_kernel<<<blocks_for(n), kThreads, 0, stream>>>(static_cast<half*>(x.data),
                                                               1.0f / static_cast<float>(hc), n);
    IMP_CUDA_CHECK_LAUNCH();
}

void hc_mix(const Tensor& mixw, const Tensor& normed, Tensor& out, int hc, int d, cudaStream_t stream,
            Tensor* out2) {
    const int64_t nd = out.numel();
    if (nd == 0)
        return;
    hc_mix_kernel<<<blocks_for(nd), kThreads, 0, stream>>>(static_cast<const half*>(mixw.data),
                                                           static_cast<const half*>(normed.data),
                                                           static_cast<half*>(out.data),
                                                           out2 ? static_cast<half*>(out2->data) : nullptr,
                                                           hc, d, nd);
    IMP_CUDA_CHECK_LAUNCH();
}

bool hc_read_decode(const Tensor& x, const Tensor& w, const Tensor& down, const Tensor* inject,
                    Tensor& normed, Tensor& low, Tensor& inj, int hc, int d, float eps, cudaStream_t stream) {
    const int K = hc * d;
    const int lowrank = static_cast<int>(down.shape[0]);
    const int n_inj = inject ? static_cast<int>(inject->shape[0]) : 0;
    const size_t smem = static_cast<size_t>(K) * sizeof(half);
    if (x.shape[0] != 1 || hc > kMaxHc || K % 16 != 0 || smem > size_t{48} * 1024 ||
        down.qtype != QType::F16 || (inject && inject->qtype != QType::F16) || x.qtype != QType::F16 ||
        down.shape[1] != K || (inject && inject->shape[1] != K))
        return false;
    const int blocks = (lowrank + n_inj + kThreads / 32 - 1) / (kThreads / 32);
    hc_read_decode_kernel<<<blocks, kThreads, smem, stream>>>(
        static_cast<const half*>(x.data), static_cast<const half*>(w.data),
        static_cast<const half*>(down.data), inject ? static_cast<const half*>(inject->data) : nullptr,
        static_cast<half*>(normed.data), static_cast<half*>(low.data),
        inject ? static_cast<half*>(inj.data) : nullptr, hc, d, lowrank, n_inj, eps,
        1.0f / static_cast<float>(hc));
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

void hc_write(Tensor& hidden, const Tensor& h, const Tensor& mixed, const Tensor& inj, int hc, int d,
              cudaStream_t stream) {
    const int64_t total = hidden.numel();
    if (total == 0)
        return;
    hc_write_kernel<<<blocks_for(total), kThreads, 0, stream>>>(static_cast<half*>(hidden.data),
                                                                static_cast<const half*>(h.data),
                                                                static_cast<const half*>(mixed.data),
                                                                static_cast<const half*>(inj.data), hc, d,
                                                                total);
    IMP_CUDA_CHECK_LAUNCH();
}

void hc_inject_weights(Tensor& inj, int hc, cudaStream_t stream) {
    const int64_t n = inj.numel();
    if (n == 0)
        return;
    hc_inject_weights_kernel<<<blocks_for(n), kThreads, 0, stream>>>(static_cast<half*>(inj.data),
                                                                     1.0f / static_cast<float>(hc), n);
    IMP_CUDA_CHECK_LAUNCH();
}

void hc_inject_add(Tensor& hidden, const Tensor& out, const Tensor& inj, int hc, int d, cudaStream_t stream) {
    const int64_t total = hidden.numel();
    if (total == 0)
        return;
    hc_inject_add_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
        static_cast<half*>(hidden.data), static_cast<const half*>(out.data), static_cast<const half*>(inj.data),
        hc, d, total);
    IMP_CUDA_CHECK_LAUNCH();
}

void hc_sub(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    const int64_t n = out.numel();
    if (n == 0)
        return;
    hc_sub_kernel<<<blocks_for(n), kThreads, 0, stream>>>(static_cast<const half*>(a.data),
                                                          static_cast<const half*>(b.data),
                                                          static_cast<half*>(out.data), n);
    IMP_CUDA_CHECK_LAUNCH();
}

void hc_repeat(const Tensor& src, Tensor& dst, int hc, int d, cudaStream_t stream) {
    const int64_t total = dst.numel();
    if (total == 0)
        return;
    hc_repeat_kernel<<<blocks_for(total), kThreads, 0, stream>>>(static_cast<const half*>(src.data),
                                                                 static_cast<half*>(dst.data), hc, d, total);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
