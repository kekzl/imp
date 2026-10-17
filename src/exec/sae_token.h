#pragma once

// Per-token SAE summary (roadmap row 116), shared by the engine, Request and the server.

#include <cstdint>
#include <utility>
#include <vector>

namespace imp {

inline constexpr int kSaeMaxTopN = 32;
inline constexpr int kSaeMaxClamps = 16;

// Top-n active features of one row by activation (ties: lower id first), its L0 and the
// relative reconstruction error of the unclamped residual.
struct SaeTokenInfo {
    int l0 = 0;
    float rel_err = 0.0f;
    std::vector<std::pair<int32_t, float>> top;
};
SaeTokenInfo sae_token_info(const float* acts, int width, int top_n, float rel_err);

}  // namespace imp
