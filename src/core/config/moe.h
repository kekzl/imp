#pragma once

// MoE configuration, one of the nine sections split out of
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

struct MoE {
    int expert_overhead_pct = 10;
    int force_host_experts = 0;  // last N layers forced to host (0 = none)
    bool skip = false;
    bool force_fp16_sync = false;
    bool no_expert_cache = false;
    // Share of free VRAM the expert LRU cache may claim, in percent. The pool
    // depth this yields is what decides how many tokens of routing history the
    // cache can hold — 73 slots/layer on a 30B-A3B is ~3 tokens, which catches
    // the ~45% next-token reuse but not the ~80%-within-8 band. Exposed so that
    // trade is measurable rather than hardcoded; 15 is the long-standing value.
    int expert_cache_budget_pct = 15;
    // Copy host-resident NVFP4 experts into pinned host memory at load, so the
    // per-expert H2D transfers become real DMAs instead of driver-staged
    // copies. On WSL2 an mmap cannot be page-locked in place, which is why this
    // is a copy rather than a registration (the GGUF packed path does the same
    // thing at weight_upload.cu's Path A1).
    //
    // It is a TRADE, not a win, which is why it is off by default. Measured on
    // Qwen3-30B-A3B-NVFP4 with all 48 MoE layers host-resident, SIX alternating
    // paired rounds (the switch exists partly so the arms can alternate without
    // a rebuild):
    //   prefill pp512  276.6 -> 790.8 tok/s   2.9x
    //   decode  tg256  no effect              3 pairs up, 3 down
    //   model load     5.1 s -> 22.6 s        4.4x, for ~14 GiB of pinned copies
    //
    // The prefill figure is 2.9x rather than the +14.8 % first measured because
    // this flag also gates whole-layer staging: the per-projection slabs it
    // builds are what make one memcpy per projection possible, and a pageable
    // source is driver-staged whatever its size. With this off, layer staging
    // measures 252-286 tok/s, i.e. nothing, so its 324 MiB is not allocated.
    // Decode is unaffected because its cache hits 96-98 % and barely transfers;
    // prefill touches every expert and is what the transfers cost.
    //
    // Three paired rounds read decode as -33 % and that was noise: this path's
    // own decode spread is wider than the effect (the off arm alone measured
    // 34.7 to 66.1 tok/s across six runs). Do not re-derive this from fewer
    // than six pairs.
    bool pin_host_experts = false;
    // Dispatch a STAGED host-resident layer through the CUTLASS grouped NVFP4
    // prefill instead of the per-expert dequant fallback. Requires
    // pin_host_experts (which is what makes staging possible at all).
    //
    // Measured on Qwen3-30B-A3B-NVFP4, all 48 MoE layers host-resident, six
    // alternating paired rounds:
    //   prefill pp512  663.2 -> 1563.9 tok/s   +136 %, 6/6 pairs, spread <1 %
    //   decode  tg256   59.4 ->   37.7 tok/s   -36 %,  6/6 pairs
    //
    // Opt-in because that decode figure is real but NOT understood, and it
    // reverses with context: at pp8 instead of pp512 the same arms measure
    // 25.5 -> 30.6 tok/s, i.e. the staged path is FASTER. This code only runs
    // at n > 1, so it cannot slow the decode kernels directly; what differs is
    // the expert cache's state when decode inherits it (hit rate 91 % vs
    // 92.9-98.4 % after a long prefill, 84.8 % vs 80.6 % after a short one).
    // Until that is explained, a 2.4x prefill win does not get to impose an
    // unexplained decode cost by default.
    bool staged_cutlass_prefill = false;
    // Phase 2 (MoE host-offload Graphs design): assert device-side mirror
    // == host-side LRU state after every cache mutation. Off by default;
    // turn on via `moe.expert_cache_debug_parity = true` in imp.conf for
    // CI / regression diagnosis. Has a meaningful cost (D2H readback of
    // ~120 KiB per cache update) — never enable in perf runs.
    bool expert_cache_debug_parity = false;
    // Phase 4 (async prefetch): at the start of layer L, issue async
    // H2D for up to this many of layer L+1's most-recent (proj, expert)
    // pairs that aren't currently cached. 0 disables the prefetcher
    // (default — safety first, Phase 4 perf gains depend on workload
    // and need per-model measurement). Sensible values: 3..16.
    int prefetch_top_k = 0;
    // Drop the "experts on host → graphs off" guard. Kept as an escape
    // hatch, but measured 2026-08-11 it currently buys NOTHING: every MoE
    // path serving host-resident experts reads routing on the host, so
    // moe_host_args_capture_guard throws under capture and the runner
    // aborts to per-step decode on every attempt. The older note here —
    // "correct only when prefetch coverage matches router selection" —
    // oversold it; capture never reaches the point where that would be the
    // question. Making this real needs routing AND expert residency
    // resolved device-side, and residency needs a host-issued H2D on a
    // miss. See docs/roadmap.md.
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
    // Threshold M for smallM kernel (clamped to [0,128]).
    int nvfp4_smallM_threshold = 64;
    // Rows-per-block (NR) for multi-row NVFP4 MoE decode kernels
    // (gemv_nvfp4_moe_{gate_up,decode}_mr<NR>). One warp computes one
    // row, so threads-per-block = NR * 32. Higher NR amortizes block
    // launch overhead at the cost of fewer concurrent CTAs. Valid
    // values: 4, 8 (default), 16, 32. Other values fall back to 8.
    int mr_nr = 8;
};
}  // namespace imp::cfg
