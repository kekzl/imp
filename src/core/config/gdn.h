#pragma once

// GDN configuration, one of the nine sections split out of
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

namespace imp::cfg {

struct GDN {
    bool fp32_scan = false;
    bool fp32_out = false;
    float norm_eps_override = 0.0f;  // 0 = use model default
    bool ref_kernel = false;
    bool vhead_reorder = false;
    // GDN chunkwise SSD scan refactor — Phase 1b.1 structural prototype.
    // When true,
    // the executor dispatches GDN scan through
    // `gdn_scan_chunkwise_{f32,fp32out}` (chunk-cached K/Q in shared
    // memory) instead of the per-token-loop `gdn_scan_fused_{f32,fp32out}`.
    // Bit-near-equivalent output (FP16 1e-3 / FP32 1e-5 tolerances per
    // Phase 1a); microbench shows +16.7 % on the GDN scan kernel alone
    // at n_tok=4096 (1.567 → 1.343 µs/tok on RTX 5090). Phase 4
    // cold-median A/B on Qwen3.6-35B-A3B Q4_K_M showed the end-to-end
    // wall delta is within the cuBLAS variance band (±0.5 % across
    // pp512 / pp2048 / tg128), so flipping the default on is wall-neutral
    // for the hero MoE model and unlocks the kernel-level win for
    // workloads where the GDN scan is a larger share of wall (longer
    // contexts, pure-GDN models like Qwen3.5-4B-GDN / Qwen3.5-9B-GDN
    // when bench data becomes available). Opt out via
    // `--set gdn.chunkwise_scan=false` if a model regresses.
    // After the Phase 2 ladder (2a / 2b / 2c, all shipped) was
    // exhaustively benched, Phase 1b.1 remains the fastest chunkwise
    // path on sm_120 — the WY-rep + TC-MMA variants all stay behind it.
    bool chunkwise_scan = true;
    // Override gated-DeltaNet weight layout.
    std::string layout_override;
};
}  // namespace imp::cfg
