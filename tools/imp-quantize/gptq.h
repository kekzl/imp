#pragma once

// GPTQ for NVFP4 (roadmap row 101): quantizes W [N, K] column by column against H = E[x^T x] of its
// input and spreads each column's error over the columns not yet quantized (Frantar et al. 2022).
// The micro-scale of a 16-column group is fixed from the error-updated weights when the group
// starts; codes and scales are written directly, a round-to-nearest pass over the result would pick
// other scales. tensor_scale is the caller's (RTN's absmax rule or a fused group's).

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace imp::gptq {

struct Result {
    std::vector<unsigned char> packed;  // [N, K/2]
    std::vector<unsigned char> micro;   // [N, K/16] FP8 E4M3
    double loss_rtn = 0.0;              // sum over rows of dW H dW^T, round-to-nearest
    double loss_gptq = 0.0;             // the same for the GPTQ codes
};

// H is consumed (damped in place). damp: fraction of the mean diagonal added to it. Qwen3-1.7B
// AWQ+GPTQ PPL: 19.58 at 0.01, 18.07-18.46 over 5-30, AWQ alone 18.23 (docs/QUANTIZATION.md).
inline constexpr float kDefaultDamp = 10.0f;
// GPTAQ (Li et al., arXiv 2504.02692): H from the quantized model's inputs xq, dxx column-major
// [K, K] with dxx[j * K + i] = E[(x_i - xq_i) xq_j] (x = full-precision input, same token); every
// update adds W P, P = alpha * triu1(dxx U^T) U. Empty dxx or alpha 0 = GPTQ. 0.25 = reference code.
inline constexpr float kDefaultGptaqAlpha = 0.25f;
[[nodiscard]] std::expected<Result, std::string> quantize_nvfp4(const uint16_t* w_fp16, int64_t N, int64_t K,
                                                                std::vector<float> H, float tensor_scale,
                                                                float damp = kDefaultDamp,
                                                                std::vector<float> dxx = {},
                                                                float alpha = 0.0f);

// In place on a column-major K x K device matrix: H (symmetric positive definite) -> U, the upper
// Cholesky factor of H^-1 (H^-1 = U^T U). Only the upper triangle holds U on return.
[[nodiscard]] std::expected<void, std::string> upper_cholesky_of_inverse(float* d_a, int64_t K);

// sum over rows of e_r H e_r^T for a host error matrix e [N, K] (row-major) and H [K, K].
[[nodiscard]] std::expected<double, std::string> output_loss(const std::vector<float>& e,
                                                             const std::vector<float>& H, int64_t N,
                                                             int64_t K);

// Layer index of a checkpoint weight name (".layers.N."), -1 when absent.
int layer_of(const std::string& name);

}  // namespace imp::gptq
