#pragma once

namespace imp {

// gpt-oss clamped GLU (#547): gate_c = min(gate, 7), up_c = clamp(up, -7, 7).
// out = (up_c + 1) * gate_c * sigmoid(1.702 * gate_c). One definition: the fused MoE quantize
// (#2466) must round exactly like gpt_oss_glu_fp16_kernel.
__device__ __forceinline__ float gpt_oss_glu_elem(float g, float u) {
    constexpr float kLimit = 7.0f;
    constexpr float kAlpha = 1.702f;
    g = fminf(g, kLimit);
    u = fminf(fmaxf(u, -kLimit), kLimit);
    float glu = g / (1.0f + __expf(-kAlpha * g));
    return (u + 1.0f) * glu;
}

}  // namespace imp
