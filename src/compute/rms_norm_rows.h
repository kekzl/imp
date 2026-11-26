#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Rows of fp32 x [rows, H]: v = x * rsqrt(mean(x^2) + eps) * w (w == nullptr: 1), one 256-thread block
// per row, into out_f (fp32) and / or out_h (fp16); either may be null, either may alias x. One kernel for
// the drafters' fp32-in norms (audit D-14): Inkling (fp32 weights) and DSpark's fc output (fp16 weights).
void rms_norm_rows(const float* x, const float* w, float* out_f, half* out_h, int rows, int H, float eps,
                   cudaStream_t stream);
void rms_norm_rows(const float* x, const half* w, float* out_f, half* out_h, int rows, int H, float eps,
                   cudaStream_t stream);

}  // namespace imp
