// Classifier-free guidance combine (roadmap row 117): one elementwise pass over a vocab row.

#include "compute/cfg_combine.h"
#include "core/logging.h"

namespace imp {

namespace {

__global__ void cfg_combine_kernel(float* __restrict__ cond, const float* __restrict__ uncond, float scale,
                                   int n) {
    const int i = (static_cast<int>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < n)
        cond[i] += scale * (cond[i] - uncond[i]);
}

}  // namespace

void cfg_combine(float* cond, const float* uncond, float scale, int n, cudaStream_t stream) {
    if (cond == nullptr || uncond == nullptr || n <= 0)
        return;
    constexpr int kThreads = 256;
    cfg_combine_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(cond, uncond, scale, n);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
