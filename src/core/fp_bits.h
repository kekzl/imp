#pragma once

// Host IEEE-754 half / bfloat16 <-> float conversion: one copy, constexpr.
//
// WHY THIS FILE EXISTS. These conversions were written out by hand in TEN
// files: model/gguf_half.h, quant/awq_transform.cpp, lora/lora_adapter.cpp,
// vision/vision_loader.cpp (twice), vision/qwen3vl_vision_upload.cpp,
// imp-quantize (fp8_source.cpp, awq_plan.cpp, checkpoint_out.cpp),
// imp-server/handlers_misc.cpp and imp-bench/bench_e2e.cpp. Several of them
// carry the same comment explaining that the code is written out rather than
// taken from cuda_fp16.h so the translation unit stays host-only and the CPU
// test lane can link it. That reason is real; needing ten copies of it was
// not. A header with no CUDA dependency satisfies it once.
//
// MEASURED before merging them (all 2^16 half patterns, all 2^32 float
// patterns, nvcc 13.3 / GCC 15.2, 2026-08-21):
//
//   fp16 -> float, gguf vs handlers_misc vs checkpoint_out vs __half2float
//                                                    0 of 65536 differ
//   float -> fp16, gguf vs imp-quantize fp8_source   0 of 4294967296 differ
//   float -> fp16, gguf vs CUDA __float2half      1024 of 4294967296 differ
//   float -> fp16, gguf vs imp-bench        352290816 of 4294967296 differ
//
// So the merge below is bit-exact for every caller except imp-bench, which
// truncated where everyone else rounds (its inputs are synthetic random
// weights, so the last mantissa bit is all that moves).
//
// THE 1024. float_to_half rounds a subnormal tie half-UP; CUDA's __float2half
// rounds it half-to-EVEN. The disagreement is exactly the 1024 float patterns
// that land on an exact tie at the subnormal boundary, the first being
// 0x33000000 = 2^-25, half of the smallest subnormal. Host and device
// therefore narrow those 1024 values differently. Left as-is deliberately:
// this header merges copies without moving numbers, and changing a rounding
// mode is a separate change with its own evidence.
//
// The bit moves are std::bit_cast, not memcpy: same codegen, but a constant
// expression, so the identities at the bottom are checked by the compiler
// instead of by a test somebody has to remember to run.

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
