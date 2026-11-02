#pragma once

// FP32-residual sandwich-norm chain fused per row (Gemma-3/4 `fp32_hidden_`, decode).
// Every mode is bit-identical to the unfused launches it replaces (tests/test_fp32_residual_norm.cu).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

enum class Fp32ResidMode : int {
    // no = rmsnorm_fp32_to_fp16(fp32, w_norm) [+ NVFP4 quantize of no].
    kNorm = 0,
    // rmsnorm_fp32_accum_to_fp16(in_a, w_accum, fp32 -> h); r = h; then kNorm.
    kPostAttn = 1,
    // a = rmsnorm(sanitize(in_a), w_a); b = rmsnorm(sanitize(in_b), w_b); h = a + b;
    // rmsnorm_fp32_accum_to_fp16(h, w_accum, fp32 -> h); h, fp32 *= layer_scale; then kNorm if w_norm.
    kTail = 2,
};

struct Fp32ResidArgs {
    int d_model = 0;
    float eps = 0.0f;
    float w_off = 0.0f;
    const __half* in_a = nullptr;         // kPostAttn: attention output; kTail: MoE branch
    const __half* w_a = nullptr;          // kTail: MoE branch post-norm
    const __half* in_b = nullptr;         // kTail: dense branch
    const __half* w_b = nullptr;          // kTail: dense branch post-norm
    const __half* w_accum = nullptr;      // post_attn_norm / post_ffn_norm
    float* fp32 = nullptr;                // [rows, d] FP32 residual (read-modify-write)
    __half* h_out = nullptr;              // [rows, d] FP16 view of the residual
    __half* r_out = nullptr;              // kPostAttn: residual save (nullable)
    const __half* layer_scale = nullptr;  // kTail: per-layer scalar (nullable)
    const __half* w_norm = nullptr;       // the norm of the next consumer (nullable in kTail)
    __half* no_out = nullptr;             // [rows, d] FP16 normed output
    uint8_t* xq_packed = nullptr;         // small-M NVFP4 activation scratch (nullable)
    uint8_t* xq_scales = nullptr;
    // Second norm of the same FP32 row (Gemma-4 router input): no2 = rmsnorm(fp32, w_norm2, w_off2)
    // rounded to FP16, then * half(no2_scale) in FP16 (scale_fp16_kernel). Nullable; needs w_norm.
    const __half* w_norm2 = nullptr;
    float w_off2 = 0.0f;
    __half* no2_out = nullptr;
    float no2_scale = 1.0f;
};

// One CTA per row, 512 threads. Envelope: d_model % 256 == 0, d_model <= 4096 (the accumulate
// kernel's two-chunk thread mapping), kTail also d_model >= 2048 (rmsnorm()'s row-block range).
// Returns false outside it; nothing is launched then.
[[nodiscard]] bool fp32_residual_norm(Fp32ResidMode mode, const Fp32ResidArgs& a, int rows,
                                      cudaStream_t stream);
// The d_model half of that envelope, for callers that must decide before skipping unfused launches.
[[nodiscard]] bool fp32_residual_norm_fits(Fp32ResidMode mode, int d_model);

}  // namespace imp
