#pragma once
// Inkling2-Loop per-site kernels: what inkling_fused.cu (any shape) and inkling_fixed.cu (the model's
// shape at compile time) share.

#include "compute/inkling_ops.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp::inkling::detail {

constexpr int kGateMaxRank = 128;

constexpr int kMixSlice = 128;  // elements per block
constexpr int kMixVals = 45;    // 5 x 5 Gram matrix of the states, then 4 x 5 query dots

// The conv site of channel c (MixPre, row 0): conv_residual_k / moe_tail_k for one row, expression for
// expression. Returns out; r = the residual it read.
__device__ __forceinline__ float mix_pre(const MixPre& p, int c, int C, float& r) {
    const float* wc = p.w + (size_t)c * 4;
    const float s0 = p.state[c], s1 = p.state[(size_t)C + c], s2 = p.state[(size_t)2 * C + c];
    float v = 0.0f;
    if (p.kind == 1)
        v = __half2float(p.x[c]);
    else
        for (int k = 0; k < p.top_k; ++k)
            v += p.weights[k] * __half2float(p.x[(size_t)k * C + c]);
    const float sum = wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * v + v;
    r = p.resid[c];
    if (p.copy)
        p.copy[c] = r;
    // moe_tail_k adds its residual in the multiply's FMA, conv_residual_k in a rounded step of its own.
    float o = __fmaf_rn(sum, 0.4472135954999579f, r);
    if (p.kind == 1) {
        const float a = __fmul_rn(sum, 0.4472135954999579f);
        o = p.add ? __fadd_rn(r, a) : a;
    }
    p.out[c] = o;
    if (p.log)
        p.log[c] = v;
    p.state[c] = s1, p.state[(size_t)C + c] = s2, p.state[(size_t)2 * C + c] = v;
    return o;
}

// The launches of inkling_fixed.cu; false = this shape (or set_fixed_shape_kernels(false)) takes the
// runtime-bounded kernel. Same bits either way.
[[nodiscard]] bool fixed_shape_kernels();
[[nodiscard]] bool mix_partial_fixed(const MixArgs& a, int T, int H, int slices, cudaStream_t s);
[[nodiscard]] bool mix_out_fixed(const MixArgs& a, int T, int H, int slices, cudaStream_t s);
[[nodiscard]] bool gate_down_fixed(const GateNormArgs& a, float* gd, int T, int H, cudaStream_t s);
[[nodiscard]] bool gate_mix_fixed(const GateNormArgs& a, const float* gd, const GateOut& o, int T, int H,
                                  cudaStream_t s);
[[nodiscard]] bool gate_rms_fixed(const GateNormArgs& a, const GateOut& o, int T, int H, cudaStream_t s);

}  // namespace imp::inkling::detail
