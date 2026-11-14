#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct MoE {
    int force_host_experts = 0;  // last N layers forced to host (0 = none)
    bool skip = false;
    bool no_expert_cache = false;
    // Share of free VRAM the expert LRU cache may claim, percent. Sets how many
    // tokens of routing history the cache holds (a 30B-A3B's 73 slots/layer is
    // ~3 tokens). Exposed so the trade is measurable, not hardcoded.
    // Share of free VRAM the expert LRU cache may take. 0 = automatic: free VRAM minus
    // the allocator headroom and a state/KV floor (executor_workspace_buffers.cu). A flat
    // 15 % starved a 56 GiB host-resident model - 45 % of free was the hand-set value that
    // made it usable, and the automatic split lands above it.
    int expert_cache_budget_pct = 0;
    // Copy host-resident NVFP4 experts into pinned host memory at load, so
    // per-expert H2D becomes real DMA (on WSL2 an mmap can't be page-locked in
    // place, so this copies rather than registers). A TRADE not a win: big
    // prefill gain, much slower load, VRAM cost of the pinned copies. Off by
    // default; weight_upload pins NVFP4 prequant experts anyway when host RAM allows,
    // and whole-layer staging follows the pinned slabs, not this flag.
    bool pin_host_experts = false;
    // Phase 2: assert device-side mirror == host-side LRU state after every
    // cache mutation. D2H readback per update (~120 KiB): never enable in perf
    // runs, only CI/regression diagnosis.
    bool expert_cache_debug_parity = false;
    // Per-process MoE workspace reserve override (MiB). 0 = use computed
    // default.
    int reserve_mib = 0;
    // Host expert tier for unpinned host-resident NVFP4 experts (#2621): pinned LRU of expert
    // units filled by pread from the checkpoint. 0 = host RAM available - max(6 GiB, 10 % RAM), N = cap MiB
    // (below the experts: they are not pinned, the tier serves), -1 = off.
    int host_expert_pool_mib = 0;
};

// Former MoE keys, values measured once and never moved (keys removed 2026-11-14).
// Host-layer prefill staging blocks (1 = whole layer); staged Flash-Next pp3692 203 -> 1143 tok/s.
inline constexpr int kMoeStageExpertChunks = 4;
// Device-args MoE decode GEMM (16-row grouped, gate + up + act fused) up to this many tokens.
inline constexpr int kMoeNvfp4DecodeRows = 64;
// Next-layer expert prefetch keys (max 32): Flash-Next server 0 / 2 / 4 / 6 / 7 keys =
// 91.9-92.2 / 96.4-96.8 / 99.8-100.2 / 102.6-102.8 / 101.8-103.0 tok/s.
inline constexpr int kMoeExpertPrefetch = 6;
// Copy blocks per prefetched key: 1 / 2 blocks = 102.6-103.0 / 99.8-100.3 tok/s (Flash-Next, 6 keys).
inline constexpr int kMoeExpertPrefetchBlocks = 1;
}  // namespace imp::cfg
