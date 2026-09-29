#pragma once

#include <string>

namespace imp {

// gemm.nvfp4_lm_head values. Auto = per-row FP8 E4M3 head where the head supports it, else the #982
// NVFP4 net rule (#2166); Nvfp4 = forced NVFP4 cache (speed opt-in); Source = checkpoint precision;
// Fp8 = FP8 head, source path if unsupported. Legacy bools: true/1 -> Nvfp4, false/0 -> Source.
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

// Modes that build the FP8 head when the head supports it.
inline bool lm_head_mode_fp8(LmHeadMode m) { return m == LmHeadMode::Fp8 || m == LmHeadMode::Auto; }

}  // namespace imp
