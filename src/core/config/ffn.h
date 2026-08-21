#pragma once

// FFN configuration, one of the nine sections split out of
// core/dispatch_policy.h on 2026-08-21.
//
// WHY. dispatch_policy.h aggregates all nine and is included by 23 translation
// units, of which 21 touch two sections or fewer. Adding one field to it costs
// 137.1 s of incremental rebuild, against 9.1 s for a small .cpp and 14.6 s for
// the largest .cu the file-size gate polices. A TU that needs only this section
// can include only this header and stop rebuilding when the others change.
//
// This is F-10 one level down, and dispatch_policy.h's own preamble records the
// original: config.h was included by 22 files, 85 TUs transitively, and changed
// 130 times in six months - "the highest build cost in the repo". Lifting nine
// sections into an aggregate fixed that, and gave the aggregate the same
// property for the same reason.
//
// Pure move: the contents below are byte-identical to their previous form, and
// dispatch_policy.h includes every one of these, so no existing include breaks.

#include <cstdint>
#include <string>
#include <vector>

namespace imp {
namespace cfg {

struct FFN {
    // SwiGLU/GeGLU sparsity probe (instrumentation-only — no skipping).
    // When enabled, every dense-FFN decode step runs a reduce kernel
    // that counts, for each of 5 hard-coded thresholds {0.005, 0.01,
    // 0.02, 0.05, 0.1}, the number of intermediate-dim rows i with
    // |silu(gate[i]) * up[i]| < t. Per-layer counters accumulate
    // across all generations of the process and are flushed via
    // imp::flush_ffn_sparsity_probe_log() (engine destruction or
    // explicit call). Purpose: measure the upside of contextual FFN
    // sparsity on this model class before writing a single gather
    // kernel. ~1 µs overhead per layer per token when on; zero when
    // off. Default off.
    bool sparsity_probe = false;

    // Phase 2 — actual FFN row-skipping in down_proj via per-block mask.
    // For each Q8_0-block of K (=32 elements) compute amax of
    // |silu(gate)*up|; if amax < threshold the whole 34-byte Q8_0
    // weight block is skipped (no HBM load) in the down_proj GEMV.
    // 0.0 = disabled = bit-identical to baseline. Recommended range
    // 0.005..0.05; per-layer sparsity see ffn_sparsity_probe data.
    // Only active for Q8_0 down_proj decode (n=1) today; other dtypes
    // fall through to the unmasked dispatch automatically.
    float sparsity_threshold = 0.0f;
};
}  // namespace cfg
}  // namespace imp
