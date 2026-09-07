#pragma once

// Why CUDA graphs are off for this engine (architecture audit, finding F-14).
//
// `config_.use_cuda_graphs` is a mutable bool that eight sites across four
// translation units can turn off. All eight logged, so the demotion was
// observable in a transcript — but "is this model graph-eligible, and why not"
// had no single answer site, and the reason was gone by the time anything
// downstream (the resolved-dispatch summary, a bug report, a benchmark) wanted
// it.
//
// The audit proposed hoisting the seven init-time demotions into one resolver.
// That is not implementable as stated: two of the seven inputs do not exist at
// resolver time — the expert-offload decision is made during weight upload, and
// the pinned-buffer demotion is the result of an allocation that failed at
// warmup. Hoisting the other five would leave three answer sites instead of one,
// which is the condition the finding is about.
//
// So the decision sites stay where their inputs are, and what is centralised is
// the *answer*: every site calls Engine::demote_graphs_(reason), which records
// the first reason, logs uniformly, and makes the reason readable afterwards.
//
// Deliberately header-only and dependency-free (no CUDA, no RuntimeConfig) so
// the CPU test lane can cover the enum↔name binding without a GPU.

#include <cstdint>

namespace imp {

enum class GraphDemotionReason {
    None,  // graphs still enabled

    // ── init-time ────────────────────────────────────────────────────
    ConfigNever,                 // runtime.cuda_graphs = "never"
    DebugRaw,                    // runtime.debug_raw = true (naked FP16 path)
    CalibrationActive,           // the collector allocates on first sight of a weight
    StreamingKvConfigured,       // block table mutates per decode step
    ExpertsOnHost,               // MoE host-offload; captured decode would replay stale pointers
    PinnedSampleBufUnavailable,  // pinned host buffer allocation failed at warmup

    // ── mid-run ──────────────────────────────────────────────────────
    // A demotion taken while requests are in flight, not an init property of
    // the model. Kept distinct precisely so it cannot be confused with a model
    // that was never eligible, and because it is the one reason that can be
    // reversed: Engine::promote_graphs_ lifts it once the pressure clears
    // (kv_pressure_repromotes_graphs below).
    StreamingKvKvPressure,  // KV cache >90% full → StreamingLLM auto-enabled
};

// Every enumerator has a name; a new reason that forgets one prints "?" and the
// unit test fails rather than the summary line quietly degrading.
const char* graph_demotion_reason_name(GraphDemotionReason r);

// True for the reasons that are decided while requests are already running.
bool graph_demotion_is_mid_run(GraphDemotionReason r);

// The mid-run trigger, as arithmetic the CPU lane can pin (AUDIT_arch_2026
// C-4): the pool counts as exhausted when the free list plus the reclaimable
// prefix-cache blocks fall under a tenth of it. Counting the free list alone
// fired on a pool that was one third cached-but-reclaimable, and the demotion
// then cost 2387 -> 1443 tok/s for the rest of the process (#1879).
inline bool kv_pressure_demotes_graphs(int free_blocks, int reclaimable_blocks, int pool_total) {
    return pool_total > 0 && free_blocks + reclaimable_blocks < pool_total / 10;
}

// The way back (C-3). The trigger is transient, so the latch is lifted when a
// fifth of the pool is free again (hysteresis against flapping at the 10 %
// line), and only while StreamingLLM has evicted nothing: an evicted window
// leaves -1 sentinels in block tables that a replayed graph would read (#948
// class), and the KV it dropped cannot be re-materialised. A pool that spiked
// from many short sequences never evicts (eviction needs a sequence past
// sinks + window), which is the case worth recovering.
inline bool kv_pressure_repromotes_graphs(int free_blocks, int reclaimable_blocks, int pool_total,
                                          uint64_t evicted_blocks) {
    return evicted_blocks == 0 && pool_total > 0 && free_blocks + reclaimable_blocks >= pool_total / 5;
}

}  // namespace imp
