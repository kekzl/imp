#pragma once

#include <string>

namespace imp {

// gemm.nvfp4_lm_head values. Auto = #982 net rule, Nvfp4 = forced NVFP4 cache, Source = checkpoint
// precision, Fp8 = per-row FP8 E4M3 head (#2156). Legacy bools: true/1 -> Nvfp4, false/0 -> Source.
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

}  // namespace imp
