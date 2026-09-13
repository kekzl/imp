#pragma once

// Why CUDA graphs are off for this engine (architecture audit, finding F-14).
//
// `config_.use_cuda_graphs` is a mutable bool that eight sites across four
// translation units can turn off; the decision sites stay where their inputs
// are (two of the seven init-time ones do not exist at a common resolver
// time: expert-offload is decided during weight upload, the pinned-buffer
// demotion follows a warmup allocation failure). What is centralised is the
// *answer*: every site calls Engine::demote_graphs_(reason), which records
// the first reason, logs uniformly, and makes it readable afterwards.
//
// Header-only and dependency-free (no CUDA, no RuntimeConfig) so the CPU test
// lane can cover the enum-name binding without a GPU.

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
    // A mid-run demotion, not an init property: kept distinct so it can't be
    // confused with a model that was never eligible, and because it is the
    // one reason Engine::promote_graphs_ can reverse (kv_pressure_repromotes_graphs).
    StreamingKvKvPressure,  // KV cache >90% full → StreamingLLM auto-enabled
};

// Every enumerator has a name; a new reason that forgets one prints "?" and the
// unit test fails rather than the summary line quietly degrading.
const char* graph_demotion_reason_name(GraphDemotionReason r);

// True for the reasons that are decided while requests are already running.
bool graph_demotion_is_mid_run(GraphDemotionReason r);

// The mid-run trigger, as arithmetic the CPU lane can pin (AUDIT_arch_2026
// C-4): counts the pool exhausted when free + reclaimable prefix-cache
// blocks fall under a tenth of it. Free-list-alone false-fired at a third
// reclaimable, costing 2387 -> 1443 tok/s (#1879).
inline bool kv_pressure_demotes_graphs(int free_blocks, int reclaimable_blocks, int pool_total) {
    return pool_total > 0 && free_blocks + reclaimable_blocks < pool_total / 10;
}

// The way back (C-3): lifted when a fifth of the pool is free again
// (hysteresis against flapping at the 10 % line), only while StreamingLLM has
// evicted nothing (an evicted window leaves -1 sentinels a replayed graph
// would read, #948 class, and dropped KV can't be re-materialised). Recovers
// the case of a pool that spiked from many short sequences (never evicts).
inline bool kv_pressure_repromotes_graphs(int free_blocks, int reclaimable_blocks, int pool_total,
                                          uint64_t evicted_blocks) {
    return evicted_blocks == 0 && pool_total > 0 && free_blocks + reclaimable_blocks >= pool_total / 5;
}

// The valve above is F16-only: StreamingLLM's -1 sentinels have no
// equivalent in quantized KV layouts. On FP8/NVFP4 KV (the default for
// Qwen3.8-27B and every hybrid) the same 90 % pressure has no valve, so this
// logs which pool state fired, once per process, instead of the operator
// only seeing the eventual hard cancel in decode_prepare_kv_.
inline bool kv_pressure_warns_no_streaming_valve(int free_blocks, int reclaimable_blocks,
                                                 int pool_total, bool kv_is_f16) {
    return !kv_is_f16 && kv_pressure_demotes_graphs(free_blocks, reclaimable_blocks, pool_total);
}

// Caps a bounded async-loop burst's step count against the tokens it has
// (`prepare_graph_loop`: max_tokens left, capped by reserved KV blocks). A
// wider cap made the parked runner refuse rearm on a request's last burst
// (ceiling = initial_context + max_tokens, e.g. 76 + 8 > 83) or let a
// KV-starved burst decode past its reserved blocks. 0 = unbounded (the loop
// runs to max_steps).
inline int burst_launch_step_limit(int step_limit, int remaining) {
    if (step_limit <= 0 || remaining <= 0)
        return step_limit;
    return step_limit < remaining ? step_limit : remaining;
}

}  // namespace imp
