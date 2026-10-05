#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// MXFP8 weight (#2475): F8_E4M3 `<m>.weight` [N,K] beside a U8 E8M0 `<m>.weight_scale` [N,K/32].
inline constexpr int kMxfp8Block = 32;

// One element: e4m3 * 2^(e8m0 - 127) as BF16 bits; exact (3 mantissa bits fit in 7).
[[nodiscard]] uint16_t mxfp8_to_bf16(uint8_t e4m3, uint8_t e8m0);

// Writer (imp-quantize): FP16 bits [n,k] -> E4M3 q [n,k] + E8M0 scale [n,k/32]. Per block the
// smallest power of two with amax / 2^e <= 448, E4M3 round-to-nearest-even.
void mxfp8_quantize(const uint16_t* fp16, int64_t n, int64_t k, uint8_t* q, uint8_t* scale);

// True when `scale` is the E8M0 grid of `weight` (dtype and shape), not an NVFP4/FP8 scale.
[[nodiscard]] bool is_mxfp8_pair(const Tensor& weight, const Tensor& scale);

// Widens every MXFP8 pair in `map` to a host BF16 weight (malloc'd, appended to `owned`), erases
// the scale. Returns the pair count; throws on an E8M0 NaN (255).
int widen_mxfp8_pairs(std::unordered_map<std::string, Tensor>& map, std::vector<void*>& owned);

}  // namespace imp
