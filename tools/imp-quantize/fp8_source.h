#pragma once

// Reading an FP8 checkpoint as a quantization SOURCE.
//
// A growing share of large models is published only in FP8 (DeepSeek-V3,
// Qwen3.8's FP8 line), so refusing the dtype refuses whole release families
// rather than an edge case. The tool's own accepted set was BF16/F16, which is
// what this widens.
//
// Both conventions seen in the wild store the same thing: an E4M3 weight
// [N, K] beside a `<prefix>.weight_scale_inv` block-scale grid [N/B, K/B],
// where a scale entry multiplies its whole BxB tile. Only the scale's dtype
// differs (DeepSeek-V3 F32, Qwen3.8-27B-FP8 BF16), and B is 128 in both.
//
// B is DERIVED from the two shapes rather than assumed, because a checkpoint
// that used another tile would otherwise be read with a silently wrong stride:
// the result still loads, still generates, and is simply wrong.
//
// Note this is a different layout from the scalar per-tensor `weight_scale`
// imp's runtime already handles for Modelopt exports (pre_dequant_phase0).
// `weight_scale_inv` appears nowhere else in the tree.

#include "model/safetensors_raw.h"

#include <cstdint>
#include <string>
#include <expected>
#include <vector>

namespace imp::quantize {

// True for the FP8 dtypes safetensors spells for E4M3. E5M2 is deliberately
// absent: no checkpoint seen stores weights in it, and guessing would decode
// a different exponent bias into plausible-looking garbage.
bool is_fp8_e4m3_dtype(const std::string& dtype);

// One E4M3 byte as a float. Exposed for the test: the bit layout is the part
// that cannot be checked by looking at a converted checkpoint.
float e4m3_to_float(uint8_t bits);

// The square block edge B implied by a weight [N, K] and a scale grid
// [ceil(N/B), ceil(K/B)]. Returns 0 when no single B explains both dimensions,
// which is the caller's signal to refuse rather than to pick one.
int derive_block_edge(int64_t n, int64_t k, int64_t scale_rows, int64_t scale_cols);

// Widen an FP8 weight to FP16 by multiplying each value with its block scale.
// Returns N*K FP16 values, or the error text on any shape or dtype mismatch.
// A partially converted buffer is not a value this can return.
[[nodiscard]] std::expected<std::vector<uint16_t>, std::string> fp8_block_scaled_to_fp16(
    const RawTensor& weight, const RawTensor& scale_inv);

}  // namespace imp::quantize
