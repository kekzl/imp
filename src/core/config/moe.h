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
    // Dispatch a staged host-resident MoE layer through the own grouped
    // NVFP4 prefill GEMM instead of the per-expert dequant fallback. Needs pinned slabs.
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
    // AND expert residency resolved device-side (host-issued H2D on a miss). See docs/ROADMAP.md.
    bool allow_graphs_under_offload = false;
    bool zero_workspace = false;
    bool no_shared_mlp = false;
    bool no_shexp_gate = false;
    bool no_cutlass3x = false;
    // Per-process MoE workspace reserve override (MiB). 0 = use computed
    // default.
    int reserve_mib = 0;
    // Device-args full path for NVFP4 MoE prefill (gemm_grouped_nvfp4_dispatch). Default ON
    // since 2026-05-14 (+11-39% pp512 on 4-model A/B, then on CUTLASS 3.x).
    bool nvfp4_device_args = true;
    // Opt-in smallM kernel branch for NVFP4 MoE prefill.
    bool nvfp4_smallM = false;
    // Decode on host-resident NVFP4 or GGUF experts: resolve routing to cache slots and gather the
    // misses on the device (no per-layer D2H + host LRU). Needs pin_host_experts (the
    // slabs are mapped); otherwise the host LRU path serves. 2026-09-20: see
    // src/exec/expert_cache_device.h. GGUF (packed experts in a mapped WSL2 slab): Qwen3-30B-A3B-Q4_K_M
    // 48 host layers tg128 73.6 -> 236 tok/s, CUDA graphs stay on.
    bool device_expert_cache = true;
    // Decode (n = 1) on the device expert cache: the shared expert FFN runs on a side stream
    // while the layer's misses are gathered, into its own scratch; phase 8 joins and adds, so
    // the add order and the output bits are unchanged.
    bool shared_expert_overlap = true;
    // Decode on the device expert cache: the miss gather runs on a side stream while the routed
    // experts that hit run their GEMVs; the misses follow once the gather lands. Output bits unchanged.
    bool expert_gather_overlap = true;
    // Router of rows n >= 2 (prefill, batched decode, verify) as one launch: logits GEMM + softmax /
    // sigmoid + top-k + renorm (moe_router_fused), instead of GEMM + split-K reduce + fp16 -> fp32 +
    // top-k. F16 router, <= 128 experts, no group limit; off under runtime.deterministic.
    bool router_fused = true;
    // Host expert tier for unpinned host-resident NVFP4 experts (#2621): pinned LRU of expert
    // units filled by pread from the checkpoint. 0 = host RAM available - max(6 GiB, 10 % RAM), N = cap MiB
    // (below the experts: they are not pinned, the tier serves), -1 = off.
    int host_expert_pool_mib = 0;
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
