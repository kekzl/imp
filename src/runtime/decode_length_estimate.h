#pragma once

// KV admission by expected decode length (#2486): p90 of the last kWindow finished outputs,
// kInitialTokens until kMinSamples arrived, floored at kFloorTokens. Overrun is safe because
// a dry pool swaps the request to host instead of cancelling it (engine_kv_swap.cpp).

#include <algorithm>
#include <array>
#include <cstddef>

namespace imp {

class DecodeLengthEstimate {
public:
    static constexpr int kWindow = 64;
    static constexpr int kMinSamples = 8;
    static constexpr int kInitialTokens = 1024;
    static constexpr int kFloorTokens = 256;

    void record(int output_tokens) {
        ring_[next_] = std::max(0, output_tokens);
        next_ = (next_ + 1) % kWindow;
        n_ = std::min(n_ + 1, kWindow);
    }

    // Tokens to reserve per request before max_tokens caps it.
    [[nodiscard]] int tokens() const {
        if (n_ < kMinSamples)
            return kInitialTokens;
        std::array<int, kWindow> s{};
        std::copy_n(ring_.begin(), n_, s.begin());
        const auto k = static_cast<std::ptrdiff_t>((n_ * 9) / 10);
        std::nth_element(s.begin(), s.begin() + k, s.begin() + n_);
        return std::max(kFloorTokens, s[static_cast<size_t>(k)]);
    }

    [[nodiscard]] int samples() const { return n_; }

private:
    std::array<int, kWindow> ring_{};
    int next_ = 0;
    int n_ = 0;
};

// Decode tokens admission reserves for one request. mode: -1 = max_tokens (pre-#2486),
// 0 = estimate, > 0 = fixed token count. Never above max_tokens.
[[nodiscard]] inline int admission_decode_tokens(int mode, int max_tokens, const DecodeLengthEstimate& est) {
    if (mode < 0)
        return max_tokens;
    const int want = mode > 0 ? mode : est.tokens();
    return std::min(max_tokens, want);
}

}  // namespace imp
