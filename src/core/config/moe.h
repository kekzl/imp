#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct MoE {
    int expert_overhead_pct = 10;
    int force_host_experts = 0;  // last N layers forced to host (0 = none)
    bool skip = false;
    bool force_fp16_sync = false;
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
    // Dispatch a staged host-resident MoE layer through the CUTLASS grouped
    // NVFP4 prefill instead of the per-expert dequant fallback. Needs pinned slabs.
    // Qwen3.8-Flash-Next-NVFP4: pp3692 203 -> 1143 tok/s; Qwen3-30B-A3B-NVFP4, 48 host layers:
    // pp4096 2631-2748 -> 7067-7093. Staged in moe.stage_expert_chunks blocks (below).
    bool staged_cutlass_prefill = true;
    // Stage only the experts the routing touched, with a gather kernel from the mapped
    // pinned slabs, instead of memcpying all n_experts per projection. A prompt that
    // touches every expert moves the same bytes; a short one moves a fraction.
    // Needs pin_host_experts + device_expert_cache (it reads the cache's device views).
    bool stage_touched_only = true;
    // Stage a host layer in this many expert blocks (gate+up per block, then down per block):
    // the staging buffer holds 2 projections of n_experts/chunks experts instead of all 3 of
    // n_experts, the rest goes to the expert cache. 1 = whole-layer staging.
    int stage_expert_chunks = 4;
    // Phase 2: assert device-side mirror == host-side LRU state after every
    // cache mutation. D2H readback per update (~120 KiB): never enable in perf
    // runs, only CI/regression diagnosis.
    bool expert_cache_debug_parity = false;
    // Phase 4 async prefetch: at layer L, issue async H2D for up to this many
    // of layer L+1's most-recent (proj, expert) pairs not currently cached.
    // 0 = disabled (default). Sensible values 3..16.
    int prefetch_top_k = 0;
    // Escape hatch to drop the "experts on host -> graphs off" guard. Currently
    // a no-op: every host-resident-expert MoE path reads routing on the host,
    // so moe_host_args_capture_guard always throws under capture. Needs routing
    // AND expert residency resolved device-side (host-issued H2D on a miss). See docs/roadmap.md.
    bool allow_graphs_under_offload = false;
    bool zero_workspace = false;
    bool no_shared_mlp = false;
    bool no_shexp_gate = false;
    bool no_cutlass3x = false;
    // Per-process MoE workspace reserve override (MiB). 0 = use computed
    // default.
    int reserve_mib = 0;
    // CUTLASS 3.x device-args full path for NVFP4 MoE prefill. Default ON
    // since 2026-05-14 (+11-39% pp512 on 4-model A/B).
    bool nvfp4_device_args = true;
    // Opt-in smallM kernel branch for NVFP4 MoE prefill.
    bool nvfp4_smallM = false;
    // Decode on host-resident NVFP4 experts: resolve routing to cache slots and gather the
    // misses on the device (no per-layer D2H + host LRU). Needs pin_host_experts (the
    // slabs are mapped); otherwise the host LRU path serves. 2026-09-20: see
    // src/exec/expert_cache_device.h.
    bool device_expert_cache = true;
    // Threshold M for smallM kernel (clamped to [0,128]).
    int nvfp4_smallM_threshold = 64;
    // Rows-per-block (NR) for multi-row NVFP4 MoE decode kernels: one warp
    // computes one row (threads/block = NR*32). Valid: 4, 8 (default), 16, 32;
    // others fall back to 8. Only reaches single-sequence decode
    // (can_decode_fast refuses n!=1); output is bit-identical across values.
    // NR=16/32 measured worse and are settled; NR=4 is NOT settled (noise-level effect).
    int mr_nr = 8;
};
}  // namespace imp::cfg
