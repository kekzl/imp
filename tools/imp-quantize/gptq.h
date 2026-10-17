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

// H is consumed (damped in place). damp: fraction of the mean diagonal added to it.
[[nodiscard]] std::expected<Result, std::string> quantize_nvfp4(const uint16_t* w_fp16, int64_t N, int64_t K,
                                                                std::vector<float> H, float tensor_scale,
                                                                float damp = 0.01f);

// Layer index of a checkpoint weight name (".layers.N."), -1 when absent.
int layer_of(const std::string& name);

}  // namespace imp::gptq
