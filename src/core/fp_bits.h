#pragma once

// Host IEEE-754 half/bfloat16 <-> float conversion and FP8 E4M3 decode, one copy, constexpr:
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
#include <cstdint>

namespace imp {

// IEEE-754 binary16 -> float. Exact for normals, denormals and infinities.
constexpr float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float v;
    if (e == 0)
        v = static_cast<float>(m) * 0x1p-24f;  // (m/1024) * 2^-14, exact for m < 2^10
    else if (e == 0x1F)
        v = std::bit_cast<float>(m ? 0x7FC00000u : 0x7F800000u);  // qNaN / inf
    else
        v = std::bit_cast<float>(((e + 112u) << 23) | (m << 13));  // rebias exponent 15 -> 127
    return s ? -v : v;
}

// Rounding of a binary16 subnormal result (|x| < 2^-14); normals always round to nearest even.
// kHalfUp: float_to_half. kNearestEven: ngram tables (= __float2half). kTruncate: wupload staging.
enum class HalfSubnormal : uint8_t { kHalfUp, kNearestEven, kTruncate };

// float -> IEEE-754 binary16, round-to-nearest-even on normals, subnormals per R.
template <HalfSubnormal R>
constexpr uint16_t float_to_half_as(float x) {
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
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if constexpr (R == HalfSubnormal::kHalfUp) {
            if (rem >= halfway)
                h++;
        } else if constexpr (R == HalfSubnormal::kNearestEven) {
            if (rem > halfway || (rem == halfway && (h & 1u)))
                h++;
        }
        return static_cast<uint16_t>(sign | h);
    }
    uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13));
    if (mant & 0x1000u) {  // round to nearest even
        if ((mant & 0x1FFFu) != 0x1000u || (h & 1u))
            h++;
    }
    return h;
}

// float -> binary16 with the subnormal half-up tie (see THE 1024 above).
constexpr uint16_t float_to_half(float x) { return float_to_half_as<HalfSubnormal::kHalfUp>(x); }

// Sign of the E4M3 NaN decode: kPositive = 0x7FC00000 for 0x7F and 0xFF (mxfp8, imp-quantize);
// kSigned = 0xFFC00000 for 0xFF (ngram tables).
enum class E4m3Nan : uint8_t { kPositive, kSigned };

// OCP FP8 E4M3 (bias 7, no Inf, S.1111.111 = NaN) -> float, exact.
template <E4m3Nan N = E4m3Nan::kPositive>
constexpr float e4m3_to_float(uint8_t b) {
    const uint32_t e = (b >> 3) & 0xFu, m = b & 0x7u;
    const bool neg = (b & 0x80u) != 0;
    if (e == 0xFu && m == 0x7u)
        return std::bit_cast<float>((N == E4m3Nan::kSigned && neg) ? 0xFFC00000u : 0x7FC00000u);
    const float v = e == 0 ? static_cast<float>(m) * 0x1p-9f                        // (m/8) * 2^-6
                           : std::bit_cast<float>(((e + 120u) << 23) | (m << 20));  // rebias 7 -> 127
    return neg ? -v : v;
}

// bfloat16 is the top 16 bits of the float pattern, so widening is a shift.
constexpr float bf16_to_float(uint16_t b) { return std::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

// float -> bfloat16, round-to-nearest-even on the 16 discarded bits.
constexpr uint16_t float_to_bf16(float x) {
    uint32_t b = std::bit_cast<uint32_t>(x);
    uint32_t r = (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
    return static_cast<uint16_t>(r);
}

// Checked at compile time by g++, nvcc and clang: bit_cast and float multiply only,
// no std::ldexp (clang does not constant-fold __builtin_ldexpf, #2285).
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
// 2^-25: the subnormal tie between 0x0000 and 0x0001, one result per mode.
static_assert(float_to_half_as<HalfSubnormal::kHalfUp>(0x1p-25f) == 0x0001);
static_assert(float_to_half_as<HalfSubnormal::kNearestEven>(0x1p-25f) == 0x0000);
static_assert(float_to_half_as<HalfSubnormal::kTruncate>(0x1.FFFFFEp-25f) == 0x0000);
static_assert(e4m3_to_float(0x38) == 1.0f);
static_assert(e4m3_to_float(0x7E) == 448.0f);
static_assert(e4m3_to_float(0x01) == 0x1p-9f);
static_assert(e4m3_to_float(0xB8) == -1.0f);
static_assert(std::bit_cast<uint32_t>(e4m3_to_float(0xFF)) == 0x7FC00000u);
static_assert(std::bit_cast<uint32_t>(e4m3_to_float<E4m3Nan::kSigned>(0xFF)) == 0xFFC00000u);

}  // namespace imp
