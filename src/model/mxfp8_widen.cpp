#include "model/mxfp8_widen.h"

#include "core/logging.h"
#include "quant/fp8_row_quant.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace imp {

namespace {

float e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0xF && m == 0x7)
        return NAN;
    const float mag = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                             : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
    return (b & 0x80) ? -mag : mag;
}

}  // namespace

uint16_t mxfp8_to_bf16(uint8_t e4m3, uint8_t e8m0) {
    const float v = std::ldexp(e4m3_value(e4m3), static_cast<int>(e8m0) - 127);
    return static_cast<uint16_t>(std::bit_cast<uint32_t>(v) >> 16);
}

void mxfp8_quantize(const uint16_t* fp16, int64_t n, int64_t k, uint8_t* q, uint8_t* scale) {
    const int64_t nb = k / kMxfp8Block;
    std::vector<float> v(kMxfp8Block);
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < kMxfp8Block; ++i) {
                __half_raw hr;
                hr.x = fp16[r * k + b * kMxfp8Block + i];
                v[i] = __half2float(__half(hr));
                amax = std::fmax(amax, std::fabs(v[i]));
            }
            int e = 0;
            if (amax > 0.0f && std::isfinite(amax)) {
                e = static_cast<int>(std::ceil(std::log2(amax / fp8_rows::kE4M3Max)));
                while (e < 127 && std::ldexp(amax, -e) > fp8_rows::kE4M3Max)
                    ++e;
                while (e > -127 && std::ldexp(amax, -(e - 1)) <= fp8_rows::kE4M3Max)
                    --e;
                e = std::clamp(e, -127, 127);
            }
            scale[r * nb + b] = static_cast<uint8_t>(e + 127);
            const float inv = std::ldexp(1.0f, -e);
            for (int i = 0; i < kMxfp8Block; ++i)
                q[r * k + b * kMxfp8Block + i] = fp8_rows::encode(v[i], inv);
        }
    }
}

bool is_mxfp8_pair(const Tensor& weight, const Tensor& scale) {
    return weight.qtype == QType::FP8_E4M3 && weight.ndim == 2 && scale.qtype == QType::INT8 &&
           scale.ndim == 2 && weight.shape[1] % kMxfp8Block == 0 && scale.shape[0] == weight.shape[0] &&
           scale.shape[1] == weight.shape[1] / kMxfp8Block && !weight.on_device && !scale.on_device;
}

int widen_mxfp8_pairs(std::unordered_map<std::string, Tensor>& map, std::vector<void*>& owned) {
    static const std::string kW = ".weight";
    std::vector<std::string> scales;
    for (auto& [name, w] : map) {
        if (name.size() <= kW.size() || name.compare(name.size() - kW.size(), kW.size(), kW) != 0)
            continue;
        const auto it = map.find(name + "_scale");
        if (it == map.end() || !is_mxfp8_pair(w, it->second))
            continue;
        const int64_t n = w.shape[0];
        const int64_t k = w.shape[1];
        auto* dst = static_cast<uint16_t*>(std::malloc(static_cast<size_t>(n * k) * sizeof(uint16_t)));
        if (dst == nullptr)
            throw std::bad_alloc();
        owned.push_back(dst);
        const auto* q = static_cast<const uint8_t*>(w.data);
        const auto* s = static_cast<const uint8_t*>(it->second.data);
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t b = 0; b < k / kMxfp8Block; ++b) {
                const uint8_t e = s[r * (k / kMxfp8Block) + b];
                if (e == 0xFF)
                    throw std::runtime_error("MXFP8 " + name + ": E8M0 scale is NaN (255)");
                for (int64_t i = b * kMxfp8Block; i < (b + 1) * kMxfp8Block; ++i)
                    dst[r * k + i] = mxfp8_to_bf16(q[r * k + i], e);
            }
        }
        const int64_t shape[2] = {n, k};
        w = Tensor(dst, QType::BF16, 2, shape, /*on_device=*/false);
        scales.push_back(it->first);
    }
    for (const auto& s : scales)
        map.erase(s);
    if (!scales.empty())
        IMP_LOG_INFO("MXFP8: %zu weights widened to BF16 at load", scales.size());
    return static_cast<int>(scales.size());
}

}  // namespace imp
