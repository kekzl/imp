#pragma once
// Shared YaRN RoPE device helpers. Single source of truth for the frequency
// ramp + blending math used by both the main forward's rope kernel (rope.cu)
// and the MTP draft head's mrope kernel (mtp_forward.cu). Keeping one copy
// prevents the two paths from drifting (see issue #897 / the #880 YaRN class).

#include <cuda_runtime.h>

namespace imp {

// Linear ramp: 1.0 when i0/2 <= low, 0.0 when i0/2 >= high, linear blend between.
static __device__ __forceinline__ float rope_yarn_ramp(float low, float high, int i0) {
    float y = (i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// YaRN frequency blending: blends between interpolated (freq_scale * theta_extrap)
// and extrapolated (theta_extrap) based on the correction dimension ramp.
// When ext_factor == 0, reduces to pure linear scaling (theta = freq_scale * theta_extrap).
static __device__ __forceinline__ void rope_yarn(float theta_extrap, float freq_scale, float corr_dim_0,
                                                 float corr_dim_1, int i0, float ext_factor, float mscale,
                                                 float& cos_theta, float& sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dim_0, corr_dim_1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    cos_theta = __cosf(theta) * mscale;
    sin_theta = __sinf(theta) * mscale;
}

}  // namespace imp
