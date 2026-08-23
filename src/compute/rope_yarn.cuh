#pragma once
// Shared YaRN RoPE device helpers. Single source of truth for the frequency
// ramp + blending math used by both the main forward's rope kernel (rope.cu)
// and the MTP draft head's mrope kernel (mtp_forward.cu). Keeping one copy
// prevents the two paths from drifting (see issue #897 / the #880 YaRN class).

#include <cuda_runtime.h>

namespace imp {

// Accurate sin/cos for RoPE (#1316).
//
// __sinf/__cosf are the fast intrinsics; NVIDIA specifies their argument
// reduction as accurate only for |x| < 48039. At the lowest-frequency rotary
// pair the angle IS the token position, and the drift is measurable long
// before that bound: against a CPU reference, max|gpu-cpu| goes 3.0e-6 at
// position 40, 2.3e-4 at 2000, 1.0e-2 at 131071. The build is compiled with
// --use_fast_math, which maps sinf/cosf straight back onto the intrinsics, so
// the reduction has to happen before the call.
//
// It lives here rather than in rope.cu because the YaRN branch needs it too:
// #1316 reduced two of rope_forward's three branches and left the YaRN one
// calling the raw intrinsics on an unreduced angle (#1630).
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

// YaRN frequency blending: blends between interpolated (freq_scale * theta_extrap)
// and extrapolated (theta_extrap) based on the correction dimension ramp.
// When ext_factor == 0, reduces to pure linear scaling (theta = freq_scale * theta_extrap).
// theta_extrap is a DOUBLE because the angle it carries is the token position
// scaled by a frequency, and at the context limit that number does not survive
// float: reducing a float angle in double fixes the intrinsic's argument
// reduction but not the angle it was handed. The linear branch has formed this
// product in double since #1316; this one had it in float, which left it 1.3e-3
// out at position 131071 against double truth after the reduction alone (#1630).
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
