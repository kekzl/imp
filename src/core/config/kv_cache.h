#pragma once

// KVCache configuration, one of the nine sections split out of
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

#include "core/config/swa_sizing_mode.h"

namespace imp::cfg {

struct KVCache {
    // "auto" (default) keeps FP16 but upgrades to FP8 E4M3 for models whose
    // author declares kv_cache_quant_algo=FP8 AND whose arch family is
    // verified safe for long-context FP8 KV (see kv_fp8_hint_default_safe).
    // "fp16" forces FP16 (opt out of the hint). fp8|int8|int4|nvfp4|mxfp4
    // force that dtype regardless of the hint.
    std::string dtype = "auto";
    bool allow_nondeterministic_fp8 = false;
    // Legacy unconditional FP8 auto-upgrade: force FP8 E4M3 whenever the
    // dtype resolves to FP16 (pre-hint behavior). imp.conf key only — the
    // old IMP_KV_FP8_AUTO env var is no longer read.
    bool fp8_auto_legacy = false;
    // BitDecoding Phase 3: residual FP16 cache for newest N tokens.
    // 0 = disabled (keeps Phase 1+2 behavior). Typical: 4..32.
    // Only meaningful with kv_cache.dtype = "nvfp4" + kv_cache.bitdecoding_qk.
    int bitdecoding_residual_tokens = 0;
    // BitDecoding TC path for NVFP4 paged attention QK. Default off, and
    // since 2026-08-26 that is a measured verdict, not just caution: on the
    // 32-stream Qwen3.8-27B-NVFP4 burst (3 alternating trials/arm) the TC
    // path reads 954-997 tok/s aggregate against the scalar kernel's
    // 1009-1050 (~-5%). See docs/plans/2026-08-24-qwen38-port.md, "NVFP4
    // decode attention" - the same section records the refuted GQA-tile
    // variant (branch perf/nvfp4-gqa-decode).
    bool bitdecoding_qk = false;
    // Growable KV pool: reserve address space for the pool the configuration
    // asked for, commit physical memory for what the card can spare right now,
    // and commit more as it frees up.
    //
    // What it fixes is a pool sized once, at the moment the free-VRAM reading
    // is least trustworthy. A server started while another process still holds
    // the card lands on the rescue floor and stays there for its whole life,
    // cancelling every prompt past a few hundred tokens while reporting a
    // successful load. With this, that server heals instead.
    //
    // Second use case (2026-08-27): long-context concurrency. The shadow plan
    // commits conservatively (it charges the library-reserve constant and
    // leaves forward scratch unmodelled), and the difference to the live-pass
    // sizing becomes growth headroom the scheduler commits under aggregate
    // admission pressure. Measured on Qwen3.8-27B-NVFP4, 32 concurrent
    // 8k-prompt/512-token requests: wall 86.0 -> 65.2 s median (-24%), pool
    // 2046 -> 6483 blocks. Prefill-bound bursts see no change.
    //
    // Needs CUDA virtual memory management on the device; where that is absent
    // the pool is fixed and everything behaves exactly as before. Growth costs
    // one driver mapping call per layer (measured 1.18 ms per 256 MiB) and
    // happens at most once per growth event, not per step.
    bool growable = false;
    // Percent of the planned pool to COMMIT at startup when growable. 100 keeps
    // today's behaviour: commit whatever the residual clamp allowed, and grow
    // only if that was less than planned.
    //
    // Lower is the point of the whole mechanism. A successful cudaMalloc proves
    // nothing about free VRAM on WSL2: measured on this box, a second server
    // started against a card already holding 31.4 GiB took its full 10.2 GiB of
    // KV anyway, which means it spilled into host memory and will decode at a
    // fraction of the bandwidth with nothing reporting an error. Committing a
    // fraction up front and growing into demand is the version of that decision
    // that cannot silently overshoot.
    int growable_initial_pct = 100;
    // SWA-aware KV sizing: sliding-window layers (gpt-oss window=128 on
    // every other layer, gemma-3 5:1 pattern) allocate only the trailing
    // window in a small dedicated block group instead of full-length KV
    // (~2x more KV tokens on gpt-oss, ~5-6x on gemma-3). Auto-disabled
    // (logged) for models without SWA layers, INT8/INT4 KV, hybrids,
    // MLA, StreamingLLM, green contexts, and deterministic mode.
    // Numerically exact: PPL bit-parity vs full-length KV on gemma-3-12b
    // and gpt-oss-20b (deterministic_gemm A/B, 2026-07-24).
    // Tri-state: "auto" enables the savings only when prefix caching is
    // off (one-shot imp-cli runs), so serving keeps warm-prefix TTFT;
    // "on" forces sizing and disables prefix caching (freed window
    // blocks cannot back prefix reuse — snapshot-based reuse is a
    // follow-up); "off" disables. Legacy bools map to on/off.
    std::string swa_sizing = "auto";

    // SWA window snapshots: device budget (MiB) for packed windowed-layer
    // KV snapshots — what makes prefix caching valid under SWA sizing
    // (freed window blocks cannot back reuse; the snapshot restores the
    // trailing window at the reuse boundary, like the recurrent-state
    // snapshots do for hybrids). One snapshot per prefill end, LRU.
    // 0 = off: swa_sizing=auto then yields to prefix caching, and
    // swa_sizing=on force-disables it.
    int swa_snapshot_mb = 0;

    // Pin the KV pool to exactly this many blocks; 0 = size it from the
    // VRAM budget as usual. An operator sharing a card wants the pool to
    // be a declared quantity rather than "whatever was left", and it is
    // what makes the admission guardrail (I6) reachable from a config
    // rather than only through the C API.
    int max_blocks = 0;

    SwaSizingMode swa_sizing_mode() const {
        if (swa_sizing == "auto")
            return SwaSizingMode::Auto;
        if (swa_sizing == "on" || swa_sizing == "true" || swa_sizing == "True" || swa_sizing == "1" ||
            swa_sizing == "yes")
            return SwaSizingMode::On;
        return SwaSizingMode::Off;
    }
};
}  // namespace imp::cfg
