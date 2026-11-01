#pragma once
// Kernels of the Gemma-4 audio tower that no shared primitive covers: the subsampling
// Conv2d + channel LayerNorm, the relative-position chunked local attention, the causal
// depthwise conv, and the FP32-residual row norms. GEMMs go through cuBLASLt (encoder.cpp).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {

// Conv2d k3 s2 p1, no bias. in [c_in, rows, cols] FP32 -> out [c_out, (rows-1)/2+1, (cols-1)/2+1].
void launch_g4a_conv2d_s2(const float* in, const float* w, float* out, int c_in, int c_out, int rows,
                          int cols, cudaStream_t stream);

// LayerNorm over channels (no bias) + ReLU at each (row, col) of x [c, rows, cols]. Then either
// out_chw (same layout, times row_mask[row]) or out_rc16 ([rows, cols * c], index col * c + ch).
void launch_g4a_chan_ln_relu(const float* x, const float* w, int c, int rows, int cols, float eps,
                             const uint8_t* row_mask, float* out_chw, half* out_rc16, cudaStream_t stream);

// Row RMSNorm: v = clamp(in, +-clip); y = v * rsqrt(mean(v^2) + eps) * (w ? w : 1) * scale
// (+ residual). in is in32 or in16; writes out16 and/or out32 (out32 may alias residual).
struct G4aNormArgs {
    const float* in32 = nullptr;
    const half* in16 = nullptr;
    const float* w = nullptr;
    const float* residual = nullptr;
    half* out16 = nullptr;
    float* out32 = nullptr;
    float clip = 3.0e38f;
    float eps = 1e-6f;
    float scale = 1.0f;
};
void launch_g4a_rmsnorm(const G4aNormArgs& a, int rows, int dim, cudaStream_t stream);

void launch_g4a_clamp16(const half* in, half* out, int64_t n, float lo, float hi, cudaStream_t stream);
void launch_g4a_silu16(half* x, int64_t n, cudaStream_t stream);
// out[r, :] = in[r, :d] * sigmoid(in[r, d:]), in [rows, 2d].
void launch_g4a_glu16(const half* in, half* out, int rows, int d, cudaStream_t stream);
void launch_g4a_add16_to32(float* x, const half* y, int64_t n, cudaStream_t stream);
void launch_g4a_f32_to_16(const float* x, half* y, int64_t n, cudaStream_t stream);
void launch_g4a_add_bias16(half* x, const float* bias, int rows, int dim, cudaStream_t stream);

// Causal depthwise conv (kernel k, left pad k-1) over rows of in [rows, d], then
// clamp(+-clip) -> RMSNorm * w -> SiLU -> out [rows, d] FP16. w_conv [d, k], w_norm [d].
void launch_g4a_dwconv_norm_silu(const half* in, const float* w_conv, const float* w_norm, half* out,
                                 int rows, int d, int k, float clip, float eps, cudaStream_t stream);

// HF Gemma4AudioAttention: key j attends from query i iff 0 <= i-j < window and kv_valid[j];
// logit = cap * tanh((q*q_scale) . (k*k_scale + rel_k[P-1-(i-j)]) / cap), softmax, @ v.
// q, k, v, out [rows, heads*hd] FP16; rel_k [P, heads*hd] FP32; q_scale [hd] FP32.
void launch_g4a_local_attention(const half* q, const half* k, const half* v, const float* rel_k,
                                const float* q_scale, const uint8_t* kv_valid, half* out, int rows, int heads,
                                int hd, int window, int n_rel, float k_scale, float cap, cudaStream_t stream);

}  // namespace imp
