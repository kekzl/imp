#include "fp8_source.h"

#include "core/fp_bits.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace imp::quantize {

namespace {

int64_t ceil_div(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }

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
    // When a dimension is NOT a multiple, the edge is underdetermined by the
    // shapes: 300 rows in 3 scale rows is explained by any b from 100 to 150,
    // and the division picks 100 rather than the 128 that produced the file.
    // Trying the edges quantizers actually use resolves it; anything else still
    // returns 0, because a wrong stride here is silent.
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

    // Read the scale grid once, in whichever precision it was stored.
    const int64_t sr = scale_inv.shape[0], sc = scale_inv.shape[1];
    std::vector<float> scales(static_cast<size_t>(sr * sc));
    if (scale_inv.dtype == "F32") {
        std::memcpy(scales.data(), scale_inv.data, scales.size() * sizeof(float));
    } else if (scale_inv.dtype == "BF16") {
        const auto* s16 = static_cast<const uint16_t*>(scale_inv.data);
        for (size_t i = 0; i < scales.size(); i++)
            scales[i] = bf16_to_float(s16[i]);
    } else if (scale_inv.dtype == "F16") {
        // Not seen in a released checkpoint, but cheap to accept correctly
        // rather than to mis-read as BF16, which would be off by a factor.
        // This used to widen the half by hand, and got subnormals wrong: it
        // pasted the subnormal mantissa under a normal exponent instead of
        // renormalising, so all 2046 subnormal patterns came out up to 1025x
        // too large (0x0001 read as 6.1e-05 instead of 5.96e-08). Every scale
        // in that range therefore scaled its whole weight block wrong.
        const auto* s16 = static_cast<const uint16_t*>(scale_inv.data);
        for (size_t i = 0; i < scales.size(); i++)
            scales[i] = half_to_float(s16[i]);
    } else {
        return std::unexpected("unsupported scale dtype " + scale_inv.dtype);
    }

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

}  // namespace imp::quantize
