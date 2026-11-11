// Clears the factored spare's recurrent rows (compute/gdn_factor.cuh). Scan writes a row
// per verify-chunk group; rejected/finished/reassigned rows are cleared by zeroing g only
// (rest of the row is dead). Own TU vs ssm_conv_tap.cu: gdn.cu is at the 600-LOC kernel ceiling.

#include "compute/gdn_factor.cuh"
#include "core/logging.h"
#include <cuda_bf16.h>

namespace imp {

namespace {

__global__ void gdn_factor_clear_kernel(float* __restrict__ fac, int64_t layer_stride, int n_layers,
                                        const int* __restrict__ slots, int n_heads, int fac_stride) {
    const int head = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (head >= n_heads)
        return;
    const int layer = blockIdx.y;
    const int slot = slots[blockIdx.z];
    fac[(layer * layer_stride) + (((static_cast<int64_t>(slot) * n_heads) + head) * fac_stride)] = 0.0f;
}

// One CTA per (head, layer, slot pair), thread d owns state column d: the scan's register layout
// (h_state[h][s][d]), so a row of s is one coalesced line.
template <typename StateT>
__global__ void gdn_factor_fold_kernel(char* __restrict__ h_pool, int64_t slot_stride, int64_t layer_stride,
                                       const float* __restrict__ fac, int64_t fac_layer_stride,
                                       const int* __restrict__ dst_slots, const int* __restrict__ fac_slots,
                                       int n_heads, int state_size, int head_dim, int fac_stride) {
    const int d = threadIdx.x;
    const int h = blockIdx.x;
    const int layer = blockIdx.y;
    const float* __restrict__ row = fac + (layer * fac_layer_stride) +
                                    (((static_cast<int64_t>(fac_slots[blockIdx.z]) * n_heads) + h) *
                                     fac_stride);
    const float g = row[0];
    if (g == 0.0f || d >= head_dim)
        return;
    const float* __restrict__ k = row + kGdnFactorPad;
    const float delta_d = k[state_size + d];
    StateT* H_col = reinterpret_cast<StateT*>(h_pool + (dst_slots[blockIdx.z] * slot_stride) +
                                              (layer * layer_stride)) +
                    (static_cast<size_t>(h) * state_size * head_dim) + d;
    for (int s = 0; s < state_size; s++) {
        StateT* p = H_col + (static_cast<ptrdiff_t>(s) * head_dim);
        *p = static_cast<StateT>((g * static_cast<float>(*p)) + (k[s] * delta_d));
    }
}

}  // namespace

void gdn_factor_fold(void* h_pool, bool bf16, int64_t slot_stride_bytes, int64_t layer_stride_bytes,
                     const float* fac, int64_t fac_layer_stride, int n_layers, const int* dst_slots,
                     const int* fac_slots, int n, int n_heads, int state_size, int head_dim, int fac_stride,
                     cudaStream_t stream) {
    if (!h_pool || !fac || !dst_slots || !fac_slots || n <= 0 || n_layers <= 0 || n_heads <= 0 ||
        state_size <= 0 || head_dim <= 0 || head_dim > 1024 || fac_stride <= 0)
        return;
    const dim3 grid(n_heads, n_layers, n);
    auto* pool = static_cast<char*>(h_pool);
    if (bf16) {
        gdn_factor_fold_kernel<__nv_bfloat16>
            <<<grid, head_dim, 0, stream>>>(pool, slot_stride_bytes, layer_stride_bytes, fac,
                                            fac_layer_stride, dst_slots, fac_slots, n_heads, state_size,
                                            head_dim, fac_stride);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    gdn_factor_fold_kernel<float><<<grid, head_dim, 0, stream>>>(pool, slot_stride_bytes, layer_stride_bytes,
                                                                 fac, fac_layer_stride, dst_slots, fac_slots,
                                                                 n_heads, state_size, head_dim, fac_stride);
    IMP_CUDA_CHECK_LAUNCH();
}

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
