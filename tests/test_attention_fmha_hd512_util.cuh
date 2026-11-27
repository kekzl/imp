#pragma once
// Shared by tests/test_attention_fmha_hd512.cu and tests/bench/test_attention_fmha_hd512_bench.cu.

#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace imp {
namespace {

// Integer LCG → f32 with multiply-only transforms (no libm), cubed to make a
// heavy tail, then rounded to f16 — the QK-normed-activation regime that broke
// benign-fill parity tests historically (#493/#512).
void lcg_fill(std::vector<half>& out, uint32_t seed, float amp) {
    uint32_t s = seed * 2654435761u + 1013904223u;
    for (auto& h : out) {
        s = s * 1664525u + 1013904223u;
        float u = ((s >> 8) & 0xFFFFFF) * (1.0f / 16777216.0f);  // [0,1)
        float c = (2.0f * u - 1.0f);
        c = c * c * c;  // cubed: heavy tail
        float v = c * amp;
        if (((s >> 4) & 0xFF) == 0)
            v *= 4.0f;  // ~1/256 outliers
        h = __float2half(v);
    }
}

}  // namespace
}  // namespace imp
