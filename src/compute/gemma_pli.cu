#include "compute/gemma_pli.h"
#include "core/logging.h"

#include <cuda_fp16.h>

namespace imp {

namespace {

constexpr int kPliThreads = 256;

__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0)
        red[warp] = v;
    __syncthreads();
    float s = 0.0f;
    for (int w = 0; w < (kPliThreads >> 5); ++w)
        s += red[w];
    return s;
}

// grid (n_layers, n), kPliThreads threads; each thread holds <= 4 of the dim values.
__global__ void pli_project_combine_kernel(half* __restrict__ proj, const half* __restrict__ table,
                                           const int32_t* __restrict__ ids, const half* __restrict__ norm_w,
                                           int n_layers, int dim, int vocab, int32_t mm_token_id,
                                           int32_t pad_id, float eps, float tok_scale, float out_scale) {
    __shared__ float red[kPliThreads / 32];
    const int l = static_cast<int>(blockIdx.x);
    const int t = static_cast<int>(blockIdx.y);
    const size_t row = static_cast<size_t>(n_layers) * dim;
    half* x = proj + (static_cast<size_t>(t) * row) + (static_cast<size_t>(l) * dim);
    int32_t id = ids[t];
    if (id == mm_token_id || id < 0 || id >= vocab)
        id = pad_id;
    const half* e = table + (static_cast<size_t>(id) * row) + (static_cast<size_t>(l) * dim);

    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = static_cast<int>(threadIdx.x) + (k * kPliThreads);
        v[k] = j < dim ? __half2float(x[j]) : 0.0f;
        ss += v[k] * v[k];
    }
    const float inv = rsqrtf((block_sum(ss, red) / static_cast<float>(dim)) + eps);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = static_cast<int>(threadIdx.x) + (k * kPliThreads);
        if (j < dim) {
            const float y = (v[k] * inv * __half2float(norm_w[j])) + (__half2float(e[j]) * tok_scale);
            x[j] = __float2half(y * out_scale);
        }
    }
}

__global__ void pli_gate_mul_kernel(half* __restrict__ gate, const half* __restrict__ pli, int64_t total,
                                    int dim, int row_stride, int col0) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= total)
        return;
    const int64_t t = i / dim;
    const int j = static_cast<int>(i - (t * dim));
    const float g = __half2float(gate[i]);
    // gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    const float a = 0.7978845608028654f * (g + (0.044715f * g * g * g));
    const float act = 0.5f * g * (1.0f + tanhf(a));
    gate[i] = __float2half(act * __half2float(pli[(t * row_stride) + col0 + j]));
}

}  // namespace

void pli_project_combine(void* proj, const void* table, const int32_t* ids, const void* norm_w, int n,
                         int n_layers, int dim, int vocab, int32_t mm_token_id, int32_t pad_id, float eps,
                         float tok_scale, float out_scale, cudaStream_t stream) {
    if (n <= 0 || n_layers <= 0 || dim <= 0)
        return;
    if (dim > 4 * kPliThreads) {
        IMP_LOG_ERROR("pli_project_combine: dim %d > %d", dim, 4 * kPliThreads);
        return;
    }
    const dim3 grid(static_cast<unsigned>(n_layers), static_cast<unsigned>(n));
    pli_project_combine_kernel<<<grid, kPliThreads, 0, stream>>>(static_cast<half*>(proj),
                                                                 static_cast<const half*>(table), ids,
                                                                 static_cast<const half*>(norm_w), n_layers,
                                                                 dim, vocab, mm_token_id, pad_id, eps,
                                                                 tok_scale, out_scale);
    IMP_CUDA_CHECK_LAUNCH();
}

void pli_gate_mul(void* gate, const void* pli, int n, int n_layers, int dim, int layer, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(n) * dim;
    if (total <= 0)
        return;
    const int threads = 256;
    const auto blocks = static_cast<unsigned>((total + threads - 1) / threads);
    pli_gate_mul_kernel<<<blocks, threads, 0, stream>>>(static_cast<half*>(gate),
                                                        static_cast<const half*>(pli), total, dim,
                                                        n_layers * dim, layer * dim);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
