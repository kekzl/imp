#include "quant/dequant_awq.h"
#include "core/logging.h"

namespace imp {

// One thread per output element; x walks K so the [N, K] store is coalesced.
__global__ void dequant_awq4_kernel(half* __restrict__ out, const int32_t* __restrict__ qweight,
                                    const int32_t* __restrict__ qzeros, const half* __restrict__ scales, int N,
                                    int K, int group_size) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= K || n >= N)
        return;
    out[static_cast<int64_t>(n) * K + k] = awq::dequant_elem(qweight, qzeros, scales, N, group_size, k, n);
}

void dequant_awq4(half* out, const int32_t* qweight, const int32_t* qzeros, const half* scales, int N, int K,
                  int group_size, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((K + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    dequant_awq4_kernel<<<grid, block, 0, stream>>>(out, qweight, qzeros, scales, N, K, group_size);
    IMP_CUDA_CHECK_LAUNCH();
}

void dequant_packed4(bool awq_gemm, half* out, const int32_t* qweight, const int32_t* qzeros, const half* scales,
                     const int32_t* g_idx, int N, int K, int group_size, gptq::ZeroFormat gptq_fmt,
                     cudaStream_t stream) {
    if (awq_gemm)
        dequant_awq4(out, qweight, qzeros, scales, N, K, group_size, stream);
    else
        dequant_gptq4(out, qweight, qzeros, scales, g_idx, N, K, group_size, gptq_fmt, stream);
}

}  // namespace imp
