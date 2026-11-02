#pragma once

// Checkpoint FP8 KV scale tensors: name -> K/V slot and host scalar read. Pure, CPU-tested
// (tests/test_kv_fp8_scale.cpp). Consumer: kv_cache.fp8_scale=checkpoint (src/core/kv_fp8_scale.h).

#include "core/logging.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace imp {

enum class KvScaleSlot { K, V, KV };

// parts = canonical name split on '.', [model, layers, N, self_attn, ...]. Accepted:
// ModelOpt self_attn.{k_proj.k_scale, v_proj.v_scale}, compressed-tensors self_attn.{k_scale,
// v_scale}, vLLM self_attn.attn.{k_scale,v_scale}, legacy single self_attn.{,*.}kv_scale.
[[nodiscard]] inline std::optional<KvScaleSlot> parse_kv_scale_tensor_name(
    const std::vector<std::string>& parts) {
    if (parts.size() < 5 || parts.size() > 6 || parts[0] != "model" || parts[1] != "layers" ||
        parts[3] != "self_attn")
        return std::nullopt;
    const std::string& leaf = parts.back();
    if (parts.size() == 6) {
        const std::string& mid = parts[4];
        if (mid != "k_proj" && mid != "v_proj" && mid != "attn" && mid != "qkv_proj")
            return std::nullopt;
    }
    if (leaf == "k_scale")
        return KvScaleSlot::K;
    if (leaf == "v_scale")
        return KvScaleSlot::V;
    if (leaf == "kv_scale")
        return KvScaleSlot::KV;
    return std::nullopt;
}

// One host-resident F32 / BF16 / F16 element as float, shape [] or [1] (the SafeTensors loader keeps
// shape [] as ndim 0, numel() 0); nullopt otherwise or when not finite / <= 0.
[[nodiscard]] inline std::optional<float> read_host_positive_scalar(const Tensor& t) {
    if (t.data == nullptr || t.on_device || (t.ndim != 0 && t.numel() != 1))
        return std::nullopt;
    float v = 0.0f;
    if (t.qtype == QType::F32) {
        std::memcpy(&v, t.data, sizeof(float));
    } else if (t.qtype == QType::BF16) {
        uint16_t b = 0;
        std::memcpy(&b, t.data, sizeof(b));
        const uint32_t u = static_cast<uint32_t>(b) << 16;
        std::memcpy(&v, &u, sizeof(v));
    } else if (t.qtype == QType::F16) {
        uint16_t h = 0;
        std::memcpy(&h, t.data, sizeof(h));
        const int e = (h >> 10) & 0x1f;
        const int m = h & 0x3ff;
        if (e == 0x1f)
            return std::nullopt;
        v = e == 0 ? std::ldexp(static_cast<float>(m), -24)
                   : std::ldexp(static_cast<float>(m | 0x400), e - 25);
        if (h & 0x8000)
            v = -v;
    } else {
        return std::nullopt;
    }
    if (!std::isfinite(v) || !(v > 0.0f))
        return std::nullopt;
    return v;
}

// WeightMap step: `matched` unchanged when already true or the name is no KV scale; else the value
// lands in k / v (an unreadable one is warned and dropped, 0 = absent stays) and the result is true.
[[nodiscard]] inline bool kv_scale_matched(bool matched, const std::vector<std::string>& parts,
                                           const Tensor& t, const std::string& name, float& k, float& v) {
    if (matched)
        return true;
    const auto slot = parse_kv_scale_tensor_name(parts);
    if (!slot)
        return false;
    const auto val = read_host_positive_scalar(t);
    if (!val) {
        IMP_LOG_WARN("WeightMap: FP8 KV scale %s is not one positive host F32/BF16/F16 value, ignored",
                     name.c_str());
        return true;
    }
    if (*slot != KvScaleSlot::V)
        k = *val;
    if (*slot != KvScaleSlot::K)
        v = *val;
    return true;
}

}  // namespace imp
