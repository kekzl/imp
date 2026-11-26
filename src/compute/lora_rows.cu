// LoRA delta over row-strided fp16 buffers: t = x . A^T, y += scale * t . B^T (compute/lora_rows.h).
// The decode GEMV of executor_lora.cu is the rows = 1 case of these kernels.

#include "compute/lora_rows.h"

#include "core/logging.h"

#include <cstdint>

namespace imp {

namespace {

// t[row][j] = dot(A[j, :], x[row, :]), fp32: one block per (rank row j, row).
__global__ void lora_rows_a_k(const half* __restrict__ A, const half* __restrict__ x, int ldx,
                              float* __restrict__ t, int r, int K) {
    const int j = blockIdx.x;
    if (j >= r)
        return;
    const half* arow = A + (static_cast<int64_t>(j) * K);
    const half* xrow = x + (static_cast<int64_t>(blockIdx.y) * ldx);
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += __half2float(arow[k]) * __half2float(xrow[k]);
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        t[(static_cast<int64_t>(blockIdx.y) * r) + j] = red[0];
}

// y[row][i] += scale * dot(B[i, :], t[row, :]): r is tiny, one thread per output column.
__global__ void lora_rows_b_k(const half* __restrict__ B, const float* __restrict__ t, half* __restrict__ y,
                              int ldy, float scale, int N, int r) {
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N)
        return;
    const half* brow = B + (static_cast<int64_t>(i) * r);
    const float* trow = t + (static_cast<int64_t>(blockIdx.y) * r);
    half* yrow = y + (static_cast<int64_t>(blockIdx.y) * ldy);
    float acc = 0.0f;
    for (int k = 0; k < r; k++)
        acc += __half2float(brow[k]) * trow[k];
    yrow[i] = __float2half(__half2float(yrow[i]) + (scale * acc));
}

}  // namespace

void lora_delta_rows(const LoraRowsArgs& a, cudaStream_t s) {
    if (a.rows <= 0 || a.r <= 0)
        return;
    constexpr int kThreads = 256;
    lora_rows_a_k<<<dim3(a.r, a.rows), kThreads, 0, s>>>(a.A, a.x, a.ldx, a.t, a.r, a.K);
    IMP_CUDA_CHECK_LAUNCH();
    lora_rows_b_k<<<dim3((a.N + kThreads - 1) / kThreads, a.rows), kThreads, 0, s>>>(a.B, a.t, a.y, a.ldy,
                                                                                     a.scale, a.N, a.r);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
