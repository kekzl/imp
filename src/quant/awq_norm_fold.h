#pragma once

// AWQ divisor s on channel j folds into a norm weight in two conventions (layernorm.h):
// plain: out=norm(x)*g -> g'=g/s. Unit-offset (Qwen3.5/3.6/3.8, folded at load): out=norm(x)*
// (1+g) -> g'=(1+g)/s-1, exact in real arithmetic but not in the tensor's stored dtype: g'
// sits near -1 while the gain (1+g) can be tiny, so a half-ulp error near -1 becomes an
// unbounded RELATIVE error on the gain (BF16: 25% loss at s=1.25, 100% at s>=2, delta
// saturates to -1 while weights were scaled UP by s). awq_clamp_norm_divisors moves s
// toward 1 until the reproducible gain is inside a relative bound; s=1 always satisfies it,
// so the worst case is a channel that keeps its weights, never one that loses them.

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

// Relative error of the gain the STORED bytes reproduce vs the gain the fold intended, in
// `dtype`. 0 when the intended gain is 0. `g` is first rounded to `dtype`, so this is what
// the FOLD costs, not what the source value already cost.
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
