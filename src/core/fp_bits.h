#pragma once

// Host IEEE-754 half/bfloat16 <-> float conversion, one copy, constexpr:
// merges ten hand-written copies across the tree (loaders, quant, lora,
// vision, imp-quantize, imp-server, imp-bench) that existed to keep the
// conversion host-only (linkable into the CPU test lane without cuda_fp16.h).
// Bit-exact vs every prior caller except imp-bench, which truncated instead
// of rounding on synthetic random weights.
// float_to_half rounds a subnormal tie half-up; CUDA's __float2half rounds
// half-to-even, disagreeing on exactly 1024 float patterns at the subnormal
// tie boundary (first: 0x33000000 = 2^-25). Left as-is: this header merges
// without moving numbers; changing a rounding mode is a separate change.
// Bit moves via std::bit_cast (same codegen as memcpy, but a constant
// expression, so identities are compiler-checked, not test-remembered).

#include <bit>
#include <cmath>
#include <cstdint>

namespace imp {

// IEEE-754 binary16 -> float. Exact for normals, denormals and infinities.
constexpr float half_to_float(uint16_t h) {
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

// float -> IEEE-754 binary16, round-to-nearest-even on normals, half-up on the
// subnormal tie (see THE 1024 above).
constexpr uint16_t float_to_half(float x) {
    uint32_t b = std::bit_cast<uint32_t>(x);
    uint32_t sign = (b >> 16) & 0x8000u;
    uint32_t ue = (b >> 23) & 0xFFu;
    uint32_t mant = b & 0x7FFFFFu;
    if (ue == 0xFF)
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));  // inf/nan
    int32_t e = static_cast<int32_t>(ue) - 127 + 15;
    if (e >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
    if (e <= 0) {                                      // denormal / underflow
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

// bfloat16 is the top 16 bits of the float pattern, so widening is a shift.
constexpr float bf16_to_float(uint16_t b) { return std::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

// float -> bfloat16, round-to-nearest-even on the 16 discarded bits.
constexpr uint16_t float_to_bf16(float x) {
    uint32_t b = std::bit_cast<uint32_t>(x);
    uint32_t r = (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
    return static_cast<uint16_t>(r);
}

// Checked at compile time. std::ldexp is constexpr in C++23, so half_to_float
// is too, and these identities cost nothing at runtime.
static_assert(half_to_float(0x3C00) == 1.0f);
static_assert(half_to_float(0x0000) == 0.0f);
static_assert(half_to_float(0xBC00) == -1.0f);
static_assert(float_to_half(1.0f) == 0x3C00);
static_assert(float_to_half(0.0625f) == 0x2C00);  // the gpt-oss 2^-4 rescale factor
static_assert(float_to_half(2.0f) == 0x4000);
static_assert(float_to_half(0.5f) == 0x3800);
static_assert(bf16_to_float(0x3F80) == 1.0f);
static_assert(float_to_bf16(1.0f) == 0x3F80);
static_assert(bf16_to_float(float_to_bf16(-3.5f)) == -3.5f);
// Denormal half: exponent 0, mantissa 1 -> 2^-24. The exponent-bit-subtract
// trick this replaced (PR #808) flushed this to zero and corrupted gpt-oss.
static_assert(half_to_float(0x0001) == 5.9604644775390625e-08f);
static_assert(float_to_half(5.9604644775390625e-08f) == 0x0001);

}  // namespace imp
