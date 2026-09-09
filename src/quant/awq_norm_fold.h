#pragma once

// Folding an AWQ divisor into an RMSNorm weight, for BOTH conventions imp
// serves, plus the storage hazard that only one of them has.
//
// imp applies a norm weight in one of two ways (src/compute/layernorm.h):
//
//   plain        out = norm(x) * g
//   unit offset  out = norm(x) * (1 + g)      Qwen3.5/3.6/3.8, folded at load
//                                             in src/model/weight_upload.cu
//
// An AWQ divisor s on input channel j is legal because the producer divides
// that channel by s. For a plain norm that is g/s. For a unit-offset norm it
// is NOT: dividing the stored delta divides g, not the gain the kernel uses.
// The value that reproduces the divided gain is
//
//     g' = (1 + g)/s - 1      so that (1 + g') = (1 + g)/s
//
// exact in exact arithmetic, and the whole reason the family was refused
// before. What is NOT exact is storing g' back in the tensor's own dtype:
// the stored value sits near -1 while the gain (1 + g) can be tiny, so an
// absolute half-ulp near 1 becomes an unbounded RELATIVE error on the gain.
// Measured on Qwen3.8-27B layers 0-3 (40960 channels of both block norms):
// 4 channels have |1 + g| < 0.05, the smallest gain is 0.00390625, and in
// BF16 that channel loses 25 % at s = 1.25, 50 % at s = 1.5 and 100 % at
// s >= 2, where the stored delta saturates to exactly -1 and the channel is
// deleted from the layer while its weights were multiplied UP by s.
//
// So the divisor is not applied blind: awq_clamp_norm_divisors moves a channel's
// s towards 1 until the gain it can actually reproduce is inside a relative
// bound. s = 1 always satisfies it (the source value is already representable),
// so the worst case is a channel that keeps its weights, never one that loses
// them. Widening the norm to F32 is not an alternative: the loader applies the
// +1 on BF16-source paths only, so an F32 norm would load without the offset.

#include <cstddef>
#include <string>
#include <vector>

namespace imp {

// Which of the two conventions a norm site follows.
enum class NormOffset { Plain, Unit };

// The gain the kernel multiplies by for a stored delta g.
inline float awq_norm_gain(float g, NormOffset off) { return off == NormOffset::Unit ? 1.0f + g : g; }

// v after a round trip through `dtype` ("F32" / "F16" / "BF16"). An unknown
// dtype returns v: callers refuse those before folding.
float awq_round_to_dtype(float v, const std::string& dtype);

// What the producer stores so its gain is divided by s.
float awq_fold_norm_value(float g, float s, NormOffset off);

// Relative error of the gain the STORED bytes reproduce against the gain the
// fold intended, in `dtype`. 0 when the intended gain is 0: a channel whose
// gain is already zero has nothing to lose. `g` is first taken to `dtype`, so
// this is what the FOLD costs, never what the source value already cost.
float awq_fold_gain_error(float g, float s, NormOffset off, const std::string& dtype);

// The divisor closest to `s` whose folded value stays inside `tol`, searched
// between s and 1. Returns s unchanged when the bound already holds.
float awq_clamp_norm_divisor(float g, float s, NormOffset off, const std::string& dtype, float tol);

struct NormFoldReport {
    size_t channels = 0;            // divisors inspected
    size_t clamped = 0;             // divisors the bound moved
    float worst_rel_err = 0.0f;     // after clamping
    float worst_clamp_from = 0.0f;  // the divisor the bound rejected hardest
    float worst_clamp_to = 0.0f;    // and what it became
};

// Clamps every divisor in `div` against the source values `g` (n of them, read
// out of the producer tensor). In place, so the CONSUMER's column scale is the
// same vector the producer will be folded with.
void awq_clamp_norm_divisors(const float* g, size_t n, NormOffset off, const std::string& dtype, float tol,
                             std::vector<float>& div, NormFoldReport& report);

}  // namespace imp
