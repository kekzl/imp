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
    // FP8 E4M3 KV scale per layer (src/core/kv_fp8_scale.h). "legacy" (default): first real prefill's
    // absmax / 448, floored at 1.0. "checkpoint": k_scale / v_scale tensors, static from load, 1.0
    // where absent. "calibrate": fixed-text prefill at warmup, amax * kKvFp8ScaleHeadroom / 448.
    // Static modes round up to a power of two and never move after load (prefix cache, row invariance).
    std::string fp8_scale = "legacy";
    // checkpoint / calibrate only: K and V each get their own power-of-two scale per layer (Gemma-4 K amax
    // ~1.6, V ~16); false = one shared scale, max of the two. Legacy is always shared.
    bool fp8_scale_split = false;
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

    // Host spill tier (#2203): pinned host RAM (MiB) for prefix-cache blocks reclaimed under pool
    // pressure; a later prefix hit restores them by H2D copy instead of re-prefill. 0 = off.
    // Needs prefix caching; refused on SWA layers. A spilled block carries its sparse key min/max pages;
    // a GDN hybrid restores it through recurrent snapshot chains (8 chats x 31k, 8000 blocks: TTFT -90 %).
    int host_spill_mb = 0;

    // Pin the KV pool to exactly this many blocks; 0 = size from the VRAM
    // budget. Lets an operator sharing a card declare pool size explicitly and
    // reach the admission guardrail (I6) from config, not only the C API.
    int max_blocks = 0;

    // Tokens per KV block. 0 = auto = 16 for every model. An explicit value
    // must be a multiple of 16 in [16,256]: 16 is the FP8 TC decode tile and
    // the NVFP4 WMMA tile, and the prefix cache's reuse granularity. Refused
    // at load when outside that set, not silently rounded (AUDIT_ARCH_2026 B-5).
    int block_size = 0;

    SwaSizingMode swa_sizing_mode() const {
        if (swa_sizing == "auto")
            return SwaSizingMode::Auto;
        if (swa_sizing == "on" || swa_sizing == "true" || swa_sizing == "True" || swa_sizing == "1" ||
            swa_sizing == "yes")
            return SwaSizingMode::On;
        return SwaSizingMode::Off;
    }

    // Mode gate shared by the max_seq_len resolver and init_kv_cache: auto yields to prefix
    // caching unless a snapshot budget lets the two coexist.
    [[nodiscard]] bool swa_sizing_wanted(bool prefix_caching) const {
        const SwaSizingMode m = swa_sizing_mode();
        return m == SwaSizingMode::On ||
               (m == SwaSizingMode::Auto && (!prefix_caching || swa_snapshot_mb > 0));
    }
};

// fp8_scale = "calibrate": margin over the calibration amax before E4M3 saturates (448), never swept.
inline constexpr float kKvFp8ScaleHeadroom = 2.0f;
// Growable pool capped by free VRAM: after a refused growth, skip the free-VRAM query for this
// many ms (default since the grow-ahead worker, never swept).
inline constexpr int kKvGrowRetryMs = 100;
}  // namespace imp::cfg
