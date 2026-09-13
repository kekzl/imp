#include "quant/awq_norm_fold.h"

#include "core/fp_bits.h"

#include <algorithm>
#include <cmath>

namespace imp {

float awq_round_to_dtype(float v, const std::string& dtype) {
    if (dtype == "BF16")
        return bf16_to_float(float_to_bf16(v));
    if (dtype == "F16")
        return half_to_float(float_to_half(v));
    return v;  // F32 and anything the caller already refused
}

float awq_fold_norm_value(float g, float s, NormOffset off) {
    if (!(s > 0.0f) || !std::isfinite(s))
        return g;
    return off == NormOffset::Unit ? (1.0f + g) / s - 1.0f : g / s;
}

float awq_fold_gain_error(float g, float s, NormOffset off, const std::string& dtype) {
    // Measured against what the tensor ALREADY holds, not the caller's float: the fold's
    // source is a stored value, and a bound charging the fold for the source's own
    // representation would have no s where it's satisfied. This makes s=1 exact by
    // construction, letting the clamp below terminate on a divisor rather than on zero.
    g = awq_round_to_dtype(g, dtype);
    const float want = awq_norm_gain(g, off) / s;
    if (want == 0.0f || !std::isfinite(want))
        return 0.0f;  // a gain of zero cannot be lost
    const float stored = awq_round_to_dtype(awq_fold_norm_value(g, s, off), dtype);
    return std::fabs(awq_norm_gain(stored, off) - want) / std::fabs(want);
}

float awq_clamp_norm_divisor(float g, float s, NormOffset off, const std::string& dtype, float tol) {
    if (!(s > 0.0f) || !std::isfinite(s))
        return 1.0f;
    if (awq_fold_gain_error(g, s, off, dtype) <= tol)
        return s;
    // s=1 always holds (the source value is already in the tensor's dtype, storing it back is
    // exact), so a channel is never deleted, only kept as-is in the worst case. Bisects
    // geometrically since s is multiplicative and must search both directions (AWQ can
    // normalise a group either way); `lo` always satisfies the bound, `hi` never does.
    float lo = 1.0f, hi = s;
    for (int i = 0; i < 40; i++) {
        const float mid = std::sqrt(lo * hi);
        if (!std::isfinite(mid) || mid <= 0.0f)
            break;
        if (awq_fold_gain_error(g, mid, off, dtype) <= tol)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

void awq_clamp_norm_divisors(const float* g, size_t n, NormOffset off, const std::string& dtype, float tol,
                             std::vector<float>& div, NormFoldReport& report) {
    report = NormFoldReport{};
    if (!g || div.size() != n)
        return;
    report.channels = n;
    float worst_move = 0.0f;
    for (size_t j = 0; j < n; j++) {
        const float s = div[j];
        const float kept = awq_clamp_norm_divisor(g[j], s, off, dtype, tol);
        if (kept != s) {
            report.clamped++;
            // "Hardest moved" is a ratio, not a difference: a divisor pulled
            // from 4 to 1 and one pulled from 0.25 to 1 lost the same factor.
            const float move = (s > 0.0f && kept > 0.0f) ? std::fabs(std::log(s / kept)) : 0.0f;
            if (move >= worst_move) {
                worst_move = move;
                report.worst_clamp_from = s;
                report.worst_clamp_to = kept;
            }
            div[j] = kept;
        }
        report.worst_rel_err = std::max(report.worst_rel_err, awq_fold_gain_error(g[j], div[j], off, dtype));
    }
}

}  // namespace imp
