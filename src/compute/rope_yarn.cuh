#pragma once
// Shared YaRN RoPE device helpers: single source of truth for the frequency ramp +
// blending math used by both the main forward (rope.cu) and the MTP mrope kernel
// (mtp_forward.cu), so the two paths cannot drift (#897, #880).

#include <cuda_runtime.h>

namespace imp {

// Accurate sin/cos for RoPE (#1316): __sinf/__cosf argument reduction is only accurate for
// |x|<48039; at the lowest-frequency pair the angle is the token position, drifting
// measurably before that bound (max|gpu-cpu| ~3.0e-6 @40, 2.3e-4 @2000, 1.0e-2 @131071).
// --use_fast_math maps sinf/cosf onto the intrinsics, so reduction must happen before the call.
// Lives here (not rope.cu) because the YaRN branch needs it too (#1316 missed it, #1630).
__device__ __forceinline__ void rope_sincos(double angle_exact, float* s, float* c) {
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    constexpr double kInvTwoPi = 0.15915494309189533576888376337251;
    // Multiply by the reciprocal rather than divide: FP64 division is the
    // expensive part on sm_120 (1/64 rate), and the reduction does not need
    // the extra accuracy a true divide would buy.
    double reduced = fma(-kTwoPi, floor(angle_exact * kInvTwoPi), angle_exact);
    *s = __sinf(static_cast<float>(reduced));
    *c = __cosf(static_cast<float>(reduced));
}

// Linear ramp: 1.0 when i0/2 <= low, 0.0 when i0/2 >= high, linear blend between.
static __device__ __forceinline__ float rope_yarn_ramp(float low, float high, int i0) {
    float y = (i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// YaRN frequency blend: interpolated (freq_scale*theta_extrap) vs extrapolated
// (theta_extrap), per the correction-dim ramp; ext_factor==0 reduces to pure linear scaling.
// theta_extrap is DOUBLE: the angle is position*frequency and does not survive float at
// the context limit; reducing a float angle in double fixes the intrinsic's reduction but
// not the angle itself (#1630, was 1.3e-3 off at position 131071 in float).
static __device__ __forceinline__ void rope_yarn(double theta_extrap, float freq_scale, float corr_dim_0,
                                                 float corr_dim_1, int i0, float ext_factor, float mscale,
                                                 float& cos_theta, float& sin_theta) {
    double theta_interp = static_cast<double>(freq_scale) * theta_extrap;
    double theta = theta_interp;

    if (ext_factor != 0.0f) {
        double ramp_mix = static_cast<double>(rope_yarn_ramp(corr_dim_0, corr_dim_1, i0)) *
                          static_cast<double>(ext_factor);
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    // Reduced before the intrinsic, like the other two branches. mscale is
    // applied after, so the scaling is unchanged.
    float c, sn;
    rope_sincos(theta, &sn, &c);
    cos_theta = c * mscale;
    sin_theta = sn * mscale;
}

}  // namespace imp
