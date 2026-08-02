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

namespace imp {

enum class GraphDemotionReason {
    None,  // graphs still enabled

    // ── init-time ────────────────────────────────────────────────────
    ConfigNever,                 // runtime.cuda_graphs = "never"
    Gemma4NoGraphs,              // [gemma4] no_graphs = true (regression bisect knob)
    PureSsmLayers,               // Mamba2 recurrent state is not graph-safe yet
    CalibrationActive,           // the collector allocates on first sight of a weight
    StreamingKvConfigured,       // block table mutates per decode step
    ExpertsOnHost,               // MoE host-offload; captured decode would replay stale pointers
    PinnedSampleBufUnavailable,  // pinned host buffer allocation failed at warmup

    // ── mid-run ──────────────────────────────────────────────────────
    // A one-way demotion taken while requests are in flight, not an init
    // property of the model. Kept distinct precisely so it cannot be confused
    // with a model that was never eligible.
    StreamingKvKvPressure,  // KV cache >90% full → StreamingLLM auto-enabled
};

// Every enumerator has a name; a new reason that forgets one prints "?" and the
// unit test fails rather than the summary line quietly degrading.
const char* graph_demotion_reason_name(GraphDemotionReason r);

// True for the reasons that are decided while requests are already running.
bool graph_demotion_is_mid_run(GraphDemotionReason r);

}  // namespace imp
