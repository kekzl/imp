#pragma once

// M=1 FP8 GEMVs inside the RMSNorm fold (quant/nvfp4_gemm.h), for checkpoints that mix NVFP4 and FP8
// projections. Weight scale: per tensor (s_*), or per row when the device pointer rs_* is set.

#include "quant/nvfp4_gemm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// GDN input projections of one layer, all reading x[K]: FP8 in [n_in, K], FP8 gate [n_gate, K],
// FP16 alpha and beta [n_ab, K] each. y = (W @ x) * norm_fold_scale(fold); fold off = plain GEMV.
struct Fp8GdnInputArgs {
    const uint8_t* w_in = nullptr;
    const uint8_t* w_gate = nullptr;
    const half* w_alpha = nullptr;
    const half* w_beta = nullptr;
    const float* rs_in = nullptr;
    const float* rs_gate = nullptr;
    float s_in = 1.0f;
    float s_gate = 1.0f;
    const half* x = nullptr;
    half* y_in = nullptr;
    half* y_gate = nullptr;
    half* y_alpha = nullptr;
    half* y_beta = nullptr;
    int n_in = 0;
    int n_gate = 0;
    int n_ab = 0;
    int K = 0;
    NvFP4NormFoldIn fold;
};
void gemv_fp8_gdn_input(const Fp8GdnInputArgs& a, cudaStream_t stream);

// h[M] = (A_fp8[M,K] * scale) @ x[K] + h[M], then the fold emit for the norm that follows (x != h).
void gemv_fp8_residual(const void* A, float scale, const float* d_row_scales, const half* x, half* h, int M,
                       int K, const NvFP4NormFoldOut& fold, cudaStream_t stream);

}  // namespace imp
