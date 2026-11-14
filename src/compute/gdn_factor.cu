// Fold of the bound verify's factor rows into the live slot (compute/gdn_factor.cuh). Own TU vs
// ssm_conv_tap.cu: gdn.cu is at the 600-LOC kernel ceiling.

#include "compute/gdn_factor.cuh"
#include "core/logging.h"
#include <cuda_bf16.h>

namespace imp {

namespace {

// One CTA per (head, layer, slot pair), thread d owns state column d: the scan's register layout
// (h_state[h][s][d]), so a row of s is one coalesced line.
template <typename StateT>
__global__ void gdn_factor_fold_kernel(char* __restrict__ h_pool, int64_t slot_stride, int64_t layer_stride,
                                       const float* __restrict__ fac, int64_t fac_layer_stride,
                                       const int* __restrict__ dst_slots, const int* __restrict__ fac_slots,
                                       int n_heads, int state_size, int head_dim, int fac_stride,
                                       const int* __restrict__ n_rows) {
    const int d = threadIdx.x;
    const int h = blockIdx.x;
    const int layer = blockIdx.y;
    const int n = n_rows ? min(n_rows[blockIdx.z], kGdnFactorFoldMaxRows) : 1;
    if (d >= head_dim || n <= 0)
        return;
    const float* __restrict__ row = fac + (layer * fac_layer_stride) +
                                    (((static_cast<int64_t>(fac_slots[blockIdx.z]) * n_heads) + h) *
                                     fac_stride);
    // Rows of one head are n_heads * fac_stride floats apart (the next level of the same group).
    float g[kGdnFactorFoldMaxRows], delta_d[kGdnFactorFoldMaxRows];
    const float* k[kGdnFactorFoldMaxRows];
    bool any = false;
    for (int r = 0; r < kGdnFactorFoldMaxRows; r++) {
        const float* __restrict__ row_r = row + (static_cast<int64_t>(r < n ? r : 0) * n_heads * fac_stride);
        g[r] = r < n ? row_r[0] : 0.0f;
        k[r] = row_r + kGdnFactorPad;
        delta_d[r] = k[r][state_size + d];
        any = any || g[r] != 0.0f;
    }
    if (!any)
        return;
    StateT* H_col = reinterpret_cast<StateT*>(h_pool + (dst_slots[blockIdx.z] * slot_stride) +
                                              (layer * layer_stride)) +
                    (static_cast<size_t>(h) * state_size * head_dim) + d;
    for (int s = 0; s < state_size; s++) {
        StateT* p = H_col + (static_cast<ptrdiff_t>(s) * head_dim);
        float v = static_cast<float>(*p);
        for (int r = 0; r < kGdnFactorFoldMaxRows; r++)
            if (g[r] != 0.0f)
                v = (g[r] * v) + (k[r][s] * delta_d[r]);
        *p = static_cast<StateT>(v);
    }
}

}  // namespace

void gdn_factor_fold(void* h_pool, bool bf16, int64_t slot_stride_bytes, int64_t layer_stride_bytes,
                     const float* fac, int64_t fac_layer_stride, int n_layers, const int* dst_slots,
                     const int* fac_slots, int n, int n_heads, int state_size, int head_dim, int fac_stride,
                     cudaStream_t stream, const int* n_rows) {
    if (!h_pool || !fac || !dst_slots || !fac_slots || n <= 0 || n_layers <= 0 || n_heads <= 0 ||
        state_size <= 0 || head_dim <= 0 || head_dim > 1024 || fac_stride <= 0)
        return;
    const dim3 grid(n_heads, n_layers, n);
    auto* pool = static_cast<char*>(h_pool);
    if (bf16) {
        gdn_factor_fold_kernel<__nv_bfloat16>
            <<<grid, head_dim, 0, stream>>>(pool, slot_stride_bytes, layer_stride_bytes, fac,
                                            fac_layer_stride, dst_slots, fac_slots, n_heads, state_size,
                                            head_dim, fac_stride, n_rows);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    gdn_factor_fold_kernel<float><<<grid, head_dim, 0, stream>>>(pool, slot_stride_bytes, layer_stride_bytes,
                                                                 fac, fac_layer_stride, dst_slots, fac_slots,
                                                                 n_heads, state_size, head_dim, fac_stride,
                                                                 n_rows);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
