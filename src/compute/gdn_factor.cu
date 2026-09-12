// Clear for the factored spare's recurrent rows (compute/gdn_factor.cuh).
//
// The scan writes a row for EVERY group of a verify chunk, but only an
// accepted draft may be applied by the next step. Rejected groups, finished
// requests and reassigned slots therefore have to be cleared, and clearing is
// writing the 0 sentinel into g. Only g: the rest of the row is then dead and
// costs nothing to leave behind.
//
// Its own translation unit rather than a member of ssm_conv_tap.cu, which
// carries the conv half; gdn.cu is at the 600-code-LOC kernel ceiling.

#include "compute/gdn_factor.cuh"
#include "core/logging.h"

namespace imp {

namespace {

__global__ void gdn_factor_clear_kernel(float* __restrict__ fac, int64_t layer_stride, int n_layers,
                                        const int* __restrict__ slots, int n_heads, int fac_stride) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= n_heads)
        return;
    const int layer = blockIdx.y;
    const int slot = slots[blockIdx.z];
    fac[layer * layer_stride + (static_cast<int64_t>(slot) * n_heads + head) * fac_stride] = 0.0f;
}

}  // namespace

void gdn_factor_clear(float* fac, int64_t layer_stride, int n_layers, const int* slots, int n_slots,
                      int n_heads, int fac_stride, cudaStream_t stream) {
    if (!fac || !slots || n_slots <= 0 || n_layers <= 0 || n_heads <= 0 || fac_stride <= 0)
        return;
    constexpr int kThreads = 128;
    dim3 grid((n_heads + kThreads - 1) / kThreads, n_layers, n_slots);
    gdn_factor_clear_kernel<<<grid, kThreads, 0, stream>>>(fac, layer_stride, n_layers, slots, n_heads,
                                                           fac_stride);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
