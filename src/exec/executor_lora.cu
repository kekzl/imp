// LoRA runtime delta — issue #522.
//
// y += scale * (x · A^T) · B^T   with A [r,K], B [N,r], x [n,K], y [n,N].
//
// Two regimes:
//  - n == 1 (decode): two purpose-built GEMV kernels. cuBLAS is NOT safe
//    here — decode runs under CUDA-graph capture and cublasLt fails with
//    status 14 inside capture (same class as the IQ4 finding, #556). The
//    skinny shapes (r <= 64) make custom kernels trivially sufficient.
//  - n > 1 (prefill): two cuBLAS gemm() calls (prefill is never captured).
//
// The rank intermediate lives in a small persistent scratch sized at
// set_lora() time (max_tokens × max_rank); adapters are swapped by pointer,
// the engine invalidates decode graphs on swap so captures never hold stale
// adapter pointers.

#include "exec/executor.h"
#include "compute/gemm.h"
#include "core/logging.h"
#include "lora/lora_adapter.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// t[i] = dot(A[i, :], x)  — one block per rank row, fp32 accumulate.
__global__ void lora_gemv_a_kernel(const half* __restrict__ A, const half* __restrict__ x,
                                   float* __restrict__ t, int r, int K) {
    int row = blockIdx.x;
    if (row >= r)
        return;
    const half* arow = A + static_cast<int64_t>(row) * K;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += __half2float(arow[k]) * __half2float(x[k]);
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        t[row] = red[0];
}

// y[i] += scale * dot(B[i, :], t)  — r is tiny, one thread per output row.
__global__ void lora_gemv_b_kernel(const half* __restrict__ B, const float* __restrict__ t,
                                   half* __restrict__ y, float scale, int N, int r) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const half* brow = B + static_cast<int64_t>(row) * r;
    float acc = 0.0f;
    for (int k = 0; k < r; k++)
        acc += __half2float(brow[k]) * t[k];
    y[row] = __float2half(__half2float(y[row]) + scale * acc);
}

void GraphExecutor::set_lora(const LoraAdapter* adapter) {
    lora_ = adapter;
    if (!adapter)
        return;
    // Rank-intermediate scratch: fp32 [max_rank] for decode + fp16
    // [max_tokens × max_rank] for prefill. Grow-only.
    size_t need = sizeof(float) * static_cast<size_t>(adapter->max_rank()) +
                  sizeof(half) * static_cast<size_t>(max_tokens_) * adapter->max_rank();
    if (need > lora_scratch_sz_) {
        if (lora_scratch_)
            IMP_CUDA_CHECK_LOG(cudaFree(lora_scratch_));
        lora_scratch_ = nullptr;
        lora_scratch_sz_ = 0;
        if (cudaMalloc(&lora_scratch_, need) != cudaSuccess) {
            IMP_LOG_ERROR("LoRA: scratch alloc failed (%zu B) — adapter disabled", need);
            lora_ = nullptr;
            return;
        }
        lora_scratch_sz_ = need;
    }
}

// Apply one projection's delta. x must be the projection INPUT in F16
// ([n, K] row-major, contiguous), y the projection OUTPUT ([n, N] F16);
// the delta accumulates into y.
void GraphExecutor::lora_delta_(const LoraWeights& w, const void* x, void* y, int n,
                                cudaStream_t stream) {
    const float s = lora_->scale();
    if (n == 1) {
        float* t = static_cast<float*>(lora_scratch_);
        lora_gemv_a_kernel<<<w.r, 256, 0, stream>>>(static_cast<const half*>(w.A),
                                                    static_cast<const half*>(x), t, w.r, w.K);
        IMP_CUDA_CHECK_LAUNCH();
        int threads = 256;
        int blocks = (w.N + threads - 1) / threads;
        lora_gemv_b_kernel<<<blocks, threads, 0, stream>>>(static_cast<const half*>(w.B), t,
                                                           static_cast<half*>(y), s, w.N, w.r);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    // Prefill: t = x · A^T  (gemm computes C = alpha · A_in @ B_w^T + beta · C
    // with B_w row-major [N, K] — exactly A's [r, K] layout).
    half* t_buf = reinterpret_cast<half*>(static_cast<float*>(lora_scratch_) + lora_->max_rank());
    int64_t x_shape[2] = {n, w.K};
    int64_t a_shape[2] = {w.r, w.K};
    int64_t t_shape[2] = {n, w.r};
    int64_t b_shape[2] = {w.N, w.r};
    int64_t y_shape[2] = {n, w.N};
    Tensor xt(const_cast<void*>(x), QType::F16, 2, x_shape, true);
    Tensor at(w.A, QType::F16, 2, a_shape, true);
    Tensor tt(t_buf, QType::F16, 2, t_shape, true);
    Tensor bt(w.B, QType::F16, 2, b_shape, true);
    Tensor yt(y, QType::F16, 2, y_shape, true);
    gemm(xt, at, tt, 1.0f, 0.0f, stream);
    gemm(tt, bt, yt, s, 1.0f, stream);
}

}  // namespace imp
