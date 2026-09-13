#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

#include "core/config/swa_sizing_mode.h"

namespace imp::cfg {

struct KVCache {
    // "auto" (default): keep FP16, upgrade to FP8 E4M3 when the model declares
    // kv_cache_quant_algo=FP8 AND the arch family is hint-verified safe for
    // long-context FP8 KV. "fp16" opts out; fp8|int8|int4|nvfp4|mxfp4 force it.
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
    // BitDecoding TC path for NVFP4 paged attention QK. Default off: measured
    // slower than the scalar kernel on 32-stream decode. See
    // docs/plans/2026-08-24-qwen38-port.md.
    bool bitdecoding_qk = false;
    // Growable KV pool: reserves the configured address space, commits
    // physical memory for what's free now, grows into more as it frees up.
    // Fixes pools sized once at the least-trustworthy free-VRAM moment. Needs
    // CUDA VMM; falls back to a fixed pool where absent. Growth is capped by
    // KVCache::try_grow_to so it cannot overshoot into a WDDM spill. Default on.
    bool growable = true;
    // Percent of the planned pool to commit at startup when growable. 100
    // commits whatever the residual clamp allowed, growing only if short; a
    // lower value (default 25, since vram.lazy_commit) commits a fraction and
    // grows at admission: a successful cudaMalloc proves nothing about free
    // VRAM on WSL2 (a full-size commit can silently spill to host memory).
    int growable_initial_pct = 25;
    // SWA-aware KV sizing: sliding-window layers allocate only the trailing
    // window instead of full-length KV. Auto-disabled for models without SWA,
    // INT8/INT4 KV, hybrids, MLA, StreamingLLM, deterministic mode. Numerically
    // exact (PPL bit-parity vs full KV). Tri-state: "auto" only when prefix
    // caching is off, "on" forces sizing and disables prefix caching, "off" disables.
    std::string swa_sizing = "auto";

    // SWA window snapshots: device budget (MiB) for packed windowed-layer KV
    // snapshots, what makes prefix caching valid under SWA sizing (freed window
    // blocks can't back reuse otherwise). One snapshot per prefill end, LRU. 0 = off.
    int swa_snapshot_mb = 0;

    // Pin the KV pool to exactly this many blocks; 0 = size from the VRAM
    // budget. Lets an operator sharing a card declare pool size explicitly and
    // reach the admission guardrail (I6) from config, not only the C API.
    int max_blocks = 0;

    // Tokens per KV block. 0 = auto = 16 for every model. An explicit value
    // must be a multiple of 16 in [16,256]: 16 is the FP8 TC decode tile and
    // the NVFP4 WMMA tile, and the prefix cache's reuse granularity. Refused
    // at load when outside that set, not silently rounded (AUDIT_arch_2026 B-5).
    int block_size = 0;

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
