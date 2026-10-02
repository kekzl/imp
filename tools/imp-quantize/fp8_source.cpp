#include "fp8_source.h"

#include "core/fp_bits.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace imp::quantize {

namespace {

int64_t ceil_div(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }

// Every scale value as float, in whichever precision it was stored.
std::expected<std::vector<float>, std::string> read_scales(const RawTensor& scale) {
    std::vector<float> scales(static_cast<size_t>(scale.numel()));
    const size_t elem = scale.dtype == "F32" ? 4 : 2;
    if (scale.nbytes != 0 && scale.nbytes != scales.size() * elem)
        return std::unexpected("scale " + scale.name + " holds " + std::to_string(scale.nbytes) +
                               " bytes for " + std::to_string(scales.size()) + " values");
    if (scale.dtype == "F32") {
        std::memcpy(scales.data(), scale.data, scales.size() * sizeof(float));
    } else if (scale.dtype == "BF16") {
        const auto* s16 = static_cast<const uint16_t*>(scale.data);
        for (size_t i = 0; i < scales.size(); i++)
            scales[i] = bf16_to_float(s16[i]);
    } else if (scale.dtype == "F16") {
        // Not mis-read as BF16 (off by a factor); half_to_float renormalises subnormals (a hand-widened
        // version was up to 1025x too large).
        const auto* s16 = static_cast<const uint16_t*>(scale.data);
        for (size_t i = 0; i < scales.size(); i++)
            scales[i] = half_to_float(s16[i]);
    } else {
        return std::unexpected("unsupported scale dtype " + scale.dtype);
    }
    return scales;
}

}  // namespace

bool is_fp8_e4m3_dtype(const std::string& dtype) {
    // safetensors has spelled this both ways across exporter versions.
    return dtype == "F8_E4M3" || dtype == "FP8_E4M3" || dtype == "F8_E4M3FN";
}

float e4m3_to_float(uint8_t bits) {
    const uint32_t s = (bits >> 7) & 0x1u;
    const uint32_t e = (bits >> 3) & 0xFu;
    const uint32_t m = bits & 0x7u;
    float v;
    if (e == 0) {
        // Subnormal: no implicit leading one, fixed 2^-6 scale.
        v = std::ldexp(static_cast<float>(m) / 8.0f, -6);
    } else if (e == 0xFu && m == 0x7u) {
        // E4M3 spends the would-be infinity encoding on NaN; there is no Inf.
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        v = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
    }
    return s ? -v : v;
}

int derive_block_edge(int64_t n, int64_t k, int64_t scale_rows, int64_t scale_cols) {
    if (n <= 0 || k <= 0 || scale_rows <= 0 || scale_cols <= 0)
        return 0;
    auto explains = [&](int64_t b) {
        return b > 0 && ceil_div(n, b) == scale_rows && ceil_div(k, b) == scale_cols;
    };
    // Dividing the rows gives the edge exactly when the dimension is a multiple
    // of it, which is the case in every released checkpoint seen.
    const int64_t from_rows = ceil_div(n, scale_rows);
    if (explains(from_rows))
        return static_cast<int>(from_rows);
    // When a dimension isn't a multiple, the block size is underdetermined by the shapes alone
    // (300 rows in 3 scale rows fits any b from 100-150); trying only the edges real quantizers use
    // resolves it. Anything else returns 0, since a wrong stride here is silent.
    for (int64_t b : {128, 64, 256, 32, 512}) {
        if (explains(b))
            return static_cast<int>(b);
    }
    return 0;
}

std::expected<std::vector<uint16_t>, std::string> fp8_block_scaled_to_fp16(const RawTensor& weight,
                                                                           const RawTensor& scale_inv) {
    if (!is_fp8_e4m3_dtype(weight.dtype))
        return std::unexpected("weight dtype " + weight.dtype + " is not E4M3");
    if (weight.shape.size() != 2 || scale_inv.shape.size() != 2)
        return std::unexpected("block-scaled FP8 needs a 2-D weight and a 2-D scale grid");
    const int64_t n = weight.shape[0], k = weight.shape[1];
    const int block = derive_block_edge(n, k, scale_inv.shape[0], scale_inv.shape[1]);
    if (block == 0)
        return std::unexpected("no block size explains weight [" + std::to_string(n) + "," +
                               std::to_string(k) + "] against scale [" + std::to_string(scale_inv.shape[0]) +
                               "," + std::to_string(scale_inv.shape[1]) + "]");

    const int64_t sc = scale_inv.shape[1];
    auto read = read_scales(scale_inv);
    if (!read)
        return std::unexpected(read.error());
    const std::vector<float>& scales = *read;

    const auto* w = static_cast<const uint8_t*>(weight.data);
    std::vector<uint16_t> tmp(static_cast<size_t>(n * k));
    for (int64_t i = 0; i < n; i++) {
        const int64_t br = i / block;
        for (int64_t j = 0; j < k; j++) {
            const float s = scales[static_cast<size_t>(br * sc + j / block)];
            const float v = e4m3_to_float(w[static_cast<size_t>(i * k + j)]) * s;
            tmp[static_cast<size_t>(i * k + j)] = float_to_half(v);
        }
    }
    return tmp;
}

bool is_per_tensor_scale(const RawTensor& scale) {
    return scale.shape.empty() || (scale.shape.size() == 1 && scale.shape[0] == 1);
}

std::expected<std::vector<uint16_t>, std::string> fp8_tensor_scaled_to_fp16(const RawTensor& weight,
                                                                            const RawTensor& scale) {
    if (!is_fp8_e4m3_dtype(weight.dtype))
        return std::unexpected("weight dtype " + weight.dtype + " is not E4M3");
    if (weight.shape.size() != 2)
        return std::unexpected("per-tensor FP8 needs a 2-D weight");
    if (!is_per_tensor_scale(scale))
        return std::unexpected("per-tensor FP8 needs a scalar scale, got rank " +
                               std::to_string(scale.shape.size()));
    auto read = read_scales(scale);
    if (!read)
        return std::unexpected(read.error());
    const float s = (*read)[0];
    const auto* w = static_cast<const uint8_t*>(weight.data);
    std::vector<uint16_t> tmp(static_cast<size_t>(weight.numel()));
    for (size_t i = 0; i < tmp.size(); i++)
        tmp[i] = float_to_half(e4m3_to_float(w[i]) * s);
    return tmp;
}

Fp8Pairing pair_fp8_scales(const std::map<std::string, const RawTensor*>& by_name) {
    static const std::string kW = ".weight";
    Fp8Pairing p;
    for (const auto& [name, t] : by_name) {
        if (!is_fp8_e4m3_dtype(t->dtype) || name.size() <= kW.size() ||
            name.compare(name.size() - kW.size(), kW.size(), kW) != 0)
            continue;
        const std::string base = name.substr(0, name.size() - kW.size());
        if (const auto it = by_name.find(base + ".weight_scale_inv"); it != by_name.end()) {
            p.scale_of[name] = it->second;
            p.consumed.insert(it->first);
            p.n_block++;
        } else if (const auto ts = by_name.find(base + ".weight_scale");
                   ts != by_name.end() && is_per_tensor_scale(*ts->second)) {
            p.scale_of[name] = ts->second;
            p.consumed.insert(ts->first);
            if (by_name.count(base + ".input_scale"))
                p.consumed.insert(base + ".input_scale");
            p.n_tensor++;
        } else {
            p.unpaired.push_back(name);
        }
    }
    return p;
}

std::expected<std::vector<uint16_t>, std::string> fp8_scaled_to_fp16(const RawTensor& weight,
                                                                     const RawTensor& scale) {
    return is_per_tensor_scale(scale) ? fp8_tensor_scaled_to_fp16(weight, scale)
                                      : fp8_block_scaled_to_fp16(weight, scale);
}

}  // namespace imp::quantize
