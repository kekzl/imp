#pragma once

// FP8 E4M3 KV cache scale per KV layer (kv_cache.fp8_scale). Pure host logic, CPU-tested
// (tests/test_kv_fp8_scale.cpp). Stored byte = e4m3(x / scale), read back as float(byte) * scale.

#include <algorithm>
#include <cmath>
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

// Checkpoint k_scale / v_scale (dequant multipliers, vLLM convention). The KV kernels take one scale
// per layer for K and V: the larger one, rounded up. Both absent (<= 0) gives 1.0 (vLLM default).
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

}  // namespace imp
