#pragma once

#include "core/qtype.h"

#include <string>

namespace imp {

// gemm.nvfp4_lm_head values. Auto = per-row FP8 E4M3 head from a 16-bit source, the checkpoint
// head for an 8-bit quantized source, else the #982 NVFP4 net rule (#2166, #2224); Nvfp4 = forced
// NVFP4 cache (speed opt-in); Source = checkpoint precision; Fp8 = FP8 head, source path if
// unsupported. Legacy bools: true/1 -> Nvfp4, false/0 -> Source.
enum class LmHeadMode { Auto, Nvfp4, Source, Fp8 };

inline LmHeadMode lm_head_mode(const std::string& v) {
    if (v == "on" || v == "true" || v == "1")
        return LmHeadMode::Nvfp4;
    if (v == "off" || v == "false" || v == "0")
        return LmHeadMode::Source;
    if (v == "fp8")
        return LmHeadMode::Fp8;
    return LmHeadMode::Auto;
}

// Per-row FP8 head on disk (imp-quantize, #2479): lm_head.weight F8_E4M3 [V, D] plus this F32 [V]
// tensor, value = code * scale. Same layout quantize_fp8_rows_async builds at load.
inline constexpr const char* kLmHeadRowScaleTensor = "lm_head.weight_row_scale";

// LM-head source width. E4M3 (3 mantissa bits) is a loss against an 8-bit quantized head:
// Qwen3-8B-Q8_0 first-token top-1 flips (#2224).
enum class LmHeadSource { Float16Plus, Quant8, QuantNarrow };

inline LmHeadSource lm_head_source(QType q) {
    switch (q) {
        case QType::F32:
        case QType::F16:
        case QType::BF16:
            return LmHeadSource::Float16Plus;
        case QType::Q8_0:
        case QType::Q8_1:
        case QType::Q8_K:
        case QType::FP8_E4M3:
        case QType::FP8_E5M2:
            return LmHeadSource::Quant8;
        default:
            return LmHeadSource::QuantNarrow;
    }
}

// This mode builds the FP8 head for a head stored as `src`: fp8 always, auto from a 16-bit source.
[[nodiscard]] inline bool lm_head_mode_fp8(LmHeadMode m, QType src) {
    return m == LmHeadMode::Fp8 ||
           (m == LmHeadMode::Auto && lm_head_source(src) == LmHeadSource::Float16Plus);
}

// auto serves an 8-bit quantized head at checkpoint precision: no FP8, no NVFP4 (#2224).
[[nodiscard]] inline bool lm_head_auto_keeps_source(LmHeadMode m, QType src) {
    return m == LmHeadMode::Auto && lm_head_source(src) == LmHeadSource::Quant8;
}

// Budget view: a per-row FP8 checkpoint head (#2479) is served as is in every mode, nothing built.
[[nodiscard]] inline bool lm_head_builds_fp8(LmHeadMode m, QType src, bool checkpoint_fp8_rows) {
    return !checkpoint_fp8_rows && lm_head_mode_fp8(m, src);
}
[[nodiscard]] inline bool lm_head_serves_source(LmHeadMode m, QType src, bool checkpoint_fp8_rows) {
    return checkpoint_fp8_rows || lm_head_auto_keeps_source(m, src);
}

}  // namespace imp
