#pragma once

// Host IEEE-754 half / bfloat16 <-> float conversion.
//
// Used by the gpt-oss GGUF residual-stream 2^-4 rescale (gguf_loader.cpp). The
// rescale must multiply fp16/bf16 scales by 0.0625 in the FLOAT domain — the old
// exponent-bit-subtract trick was wrong for small scales (biased exponent 0 ->
// left unscaled, 16x too large; exponents 1..4 -> flushed to zero), which
// corrupted Q8_0 gpt-oss weights and produced garbage output (PR #808). These
// helpers are exact for normals and correct for denormals/underflow.

#include <cmath>
#include <cstdint>
#include <cstring>

namespace imp {

inline float gguf_half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float v;
    if (e == 0)
        v = std::ldexp(static_cast<float>(m), -24);  // (m/1024) * 2^-14
    else if (e == 0x1F)
        v = m ? std::nanf("") : HUGE_VALF;
    else
        v = std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, static_cast<int>(e) - 15);
    return s ? -v : v;
}

inline uint16_t gguf_float_to_half(float x) {
    uint32_t b;
    std::memcpy(&b, &x, sizeof(b));
    uint32_t sign = (b >> 16) & 0x8000u;
    uint32_t ue = (b >> 23) & 0xFFu;
    uint32_t mant = b & 0x7FFFFFu;
    if (ue == 0xFF)
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));  // inf/nan
    int32_t e = static_cast<int32_t>(ue) - 127 + 15;
    if (e >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
    if (e <= 0) {                                       // denormal / underflow
        if (e < -10)
            return static_cast<uint16_t>(sign);  // -> +/-0
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - e);
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u)
            h++;  // round to nearest
        return static_cast<uint16_t>(sign | h);
    }
    uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13));
    if (mant & 0x1000u) {  // round to nearest even
        if ((mant & 0x1FFFu) != 0x1000u || (h & 1u))
            h++;
    }
    return h;
}

inline float gguf_bf16_to_float(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t gguf_float_to_bf16(float x) {
    uint32_t b;
    std::memcpy(&b, &x, sizeof(b));
    uint32_t r = (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;  // round to nearest even
    return static_cast<uint16_t>(r);
}

}  // namespace imp
