#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct FFN {
    // SwiGLU/GeGLU sparsity probe, instrumentation only (no skipping): counts,
    // per layer, intermediate-dim rows below 5 hardcoded thresholds {0.005,
    // 0.01, 0.02, 0.05, 0.1}. Flushed via flush_ffn_sparsity_probe_log(). Default off.
    bool sparsity_probe = false;

    // Phase 2 FFN row-skipping in down_proj: per Q8_0 block (K=32) of
    // |silu(gate)*up|, skip the 34-byte weight block if amax < threshold.
    // 0.0 = disabled, bit-identical. Only active for Q8_0 down_proj decode (n=1).
    float sparsity_threshold = 0.0f;
};
}  // namespace imp::cfg
