#pragma once

// FP8 E4M3 KV cache scale per KV layer (kv_cache.fp8_scale). Pure host logic, CPU-tested
// (tests/test_kv_fp8_scale.cpp). Stored byte = e4m3(x / scale), read back as float(byte) * scale.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>

namespace imp {

// legacy: first real prefill's absmax / 448, floored at 1.0 (91 of 92 Qwen3 layers sit on the floor).
// checkpoint: k_scale / v_scale tensors, static from load. calibrate: fixed-text prefill at warmup.
enum class Fp8KvScaleMode { Legacy, Checkpoint, Calibrate };

[[nodiscard]] inline std::optional<Fp8KvScaleMode> parse_fp8_kv_scale_mode(std::string_view s) {
    if (s == "legacy")
        return Fp8KvScaleMode::Legacy;
    if (s == "checkpoint")
        return Fp8KvScaleMode::Checkpoint;
    if (s == "calibrate")
        return Fp8KvScaleMode::Calibrate;
    return std::nullopt;
}

inline constexpr float kFp8KvE4M3Max = 448.0f;
// 2^-15: smallest E4M3 subnormal (2^-9) times scale stays >= 2^-24, exact in the FP16 gather output.
// 2^15: inv_scale stays a normal FP16 (the write kernel multiplies in half).
inline constexpr int kFp8KvScaleMinExp = -15;
inline constexpr int kFp8KvScaleMaxExp = 15;

// Smallest power of two >= x, clamped to [2^-15, 2^15]; x <= 0 or non-finite gives 1.0.
// A power of two keeps the E4M3 normal-range grid of scale 1 (same relative rounding, bit-equal
// mantissas) and makes inv_scale exact in FP16: only the subnormal and saturation edges move.
[[nodiscard]] inline float fp8_kv_pow2_ceil(float x) {
    if (!(x > 0.0f) || !std::isfinite(x))
        return 1.0f;
    int e = 0;
    const float m = std::frexp(x, &e);  // x = m * 2^e, m in [0.5, 1)
    if (m == 0.5f)
        --e;  // x is a power of two already
    return std::ldexp(1.0f, std::clamp(e, kFp8KvScaleMinExp, kFp8KvScaleMaxExp));
}

// Checkpoint k_scale / v_scale (dequant multipliers, vLLM convention), shared: the larger one, rounded
// up. Both absent (<= 0) gives 1.0 (vLLM default).
[[nodiscard]] inline float fp8_kv_scale_from_checkpoint(float k_scale, float v_scale) {
    const float s = std::max(k_scale, v_scale);
    return s > 0.0f ? fp8_kv_pow2_ceil(s) : 1.0f;
}

// Calibrated: max(K, V) absmax of the calibration prefill times headroom (>= 1) over 448, rounded up
// to a power of two (effective headroom [h, 2h)), no 1.0 floor. No amax (all-zero layer) gives 1.0.
[[nodiscard]] inline float fp8_kv_scale_from_amax(float k_amax, float v_amax, float headroom) {
    const float amax = std::max(k_amax, v_amax);
    if (!(amax > 0.0f) || !std::isfinite(amax))
        return 1.0f;
    const float h = (headroom >= 1.0f && std::isfinite(headroom)) ? headroom : 1.0f;
    return fp8_kv_pow2_ceil(amax * h / kFp8KvE4M3Max);
}

// One KV layer's K and V scales. Equal unless kv_cache.fp8_scale_split (legacy is always shared).
struct Fp8KvScalePair {
    float k = 1.0f;
    float v = 1.0f;
    bool operator==(const Fp8KvScalePair&) const = default;
};

// Static scales per layer: split = K and V each from their own tensor / amax, else one shared value.
[[nodiscard]] inline Fp8KvScalePair fp8_kv_scales_from_checkpoint(float k_scale, float v_scale, bool split) {
    if (!split) {
        const float s = fp8_kv_scale_from_checkpoint(k_scale, v_scale);
        return {s, s};
    }
    return {fp8_kv_scale_from_checkpoint(k_scale, 0.0f), fp8_kv_scale_from_checkpoint(0.0f, v_scale)};
}

[[nodiscard]] inline Fp8KvScalePair fp8_kv_scales_from_amax(float k_amax, float v_amax, float headroom,
                                                            bool split) {
    if (!split) {
        const float s = fp8_kv_scale_from_amax(k_amax, v_amax, headroom);
        return {s, s};
    }
    return {fp8_kv_scale_from_amax(k_amax, 0.0f, headroom), fp8_kv_scale_from_amax(0.0f, v_amax, headroom)};
}

// The FP8 paged decode kernels take (softmax_scale, kv_scale): logits use softmax_scale * kv_scale,
// V uses kv_scale. Separate K/V scales fold into them: {softmax_scale * k / v, v}, exact for powers of two.
struct Fp8KvDecodeScales {
    float softmax_scale;
    float kv_scale;
};

[[nodiscard]] inline Fp8KvDecodeScales fp8_kv_decode_scales(float softmax_scale, Fp8KvScalePair s) {
    if (s.k == s.v)
        return {softmax_scale, s.v};  // shared (legacy): the unsplit call, bit for bit
    return {softmax_scale * (s.k / s.v), s.v};
}

// Exponent histogram of FP16 K or V values: bin 0 = zero, 1..40 = floor(log2|x|) -24..15, 41 = inf/NaN.
inline constexpr int kFp8KvHistBins = 42;

[[nodiscard]] inline int fp8_kv_hist_bin(uint16_t half_bits) {
    const int e = (half_bits >> 10) & 0x1F;
    const int m = half_bits & 0x3FF;
    if (e == 0x1F)
        return kFp8KvHistBins - 1;
    if (e == 0) {
        if (m == 0)
            return 0;
        int top = 9;  // FP16 subnormal: m * 2^-24, exponent -24 + floor(log2 m)
        while ((m >> top) == 0)
            --top;
        return 1 + top;
    }
    return e - 15 + 25;
}

// E4M3 edge counts of a histogram at a power-of-two scale s (byte = e4m3(x / s)): flushed |x/s| < 2^-10
// (rounds to 0), subnormal [2^-10, 2^-6) (absolute step 2^-9), saturated |x/s| >= 512 or inf (clips
// to 448; [448, 512) clips too and is not counted). nonzero excludes exact zeros.
struct Fp8KvEdgeCounts {
    uint64_t nonzero = 0;
    uint64_t flushed = 0;
    uint64_t subnormal = 0;
    uint64_t saturated = 0;
};

[[nodiscard]] inline Fp8KvEdgeCounts fp8_kv_edge_counts(const uint32_t* hist, float pow2_scale) {
    Fp8KvEdgeCounts c;
    const int j = std::ilogb(pow2_scale);
    for (int b = 1; b < kFp8KvHistBins; ++b) {
        const uint64_t n = hist[b];
        c.nonzero += n;
        if (b == kFp8KvHistBins - 1) {
            c.saturated += n;
            continue;
        }
        const int r = (b - 25) - j;  // floor(log2 |x / s|)
        if (r <= -11)
            c.flushed += n;
        else if (r <= -7)
            c.subnormal += n;
        else if (r >= 9)
            c.saturated += n;
    }
    return c;
}

}  // namespace imp
