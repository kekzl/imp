# imp — Architecture Review Master Report

**Branch:** `review/architecture-2026-05-16`
**Anchor commit:** `f58eb9e`
**Target:** sm_120a / RTX 5090 / GB202 only — nothing sacred
**Date:** 2026-05-16

This document is the **entry point**. Each phase report is linked below with
a one-line abstract; the full roadmap, sequenced refactors and 30/60/90 plan
live in [`phase5_synthesis.md`](./phase5_synthesis.md).

---

## Brecher-Statement

> imp is architecturally clean and at-the-roof on **decode**, but
> **NVFP4 MoE prefill is 14.7× slower than vLLM on the identical CUTLASS
> template**, and `src/graph/` is a **15.8 KLOC clay tablet** where every
> quant format, every model arch and every dispatch decision lives as
> inline `if`-ladders. The codebase is small enough to refactor, mature
> enough to merit it, and broken in exactly two places — perf (around
> the kernel, not in it) and extensibility (per-arch behavior is
> `if (cfg.arch == ModelArch::GEMMA4)` sprinkled in 40 sites across 4
> files).

---

## Phase Reports

| # | Subagent | Report | LOC | Focus |
|---:|---|---|---:|---|
| 1 | cartographer | [`phase1_inventory.md`](./phase1_inventory.md) | 751 | Static map: subsystems, public ABI, hot-path TUs, build flags, **ballast bilanz: ~5 390 LOC** |
| 2 | perfhawk | [`phase2_perf.md`](./phase2_perf.md) | 1 159 | Roofline tables, GB202 feature audit, CUDA-graphs landscape, **deep-dive on Qwen3-Coder NVFP4 prefill 14.7× gap** |
| 3 | codereaper | [`phase3_maint.md`](./phase3_maint.md) | 1 283 | Coupling hotspots, 10 danger zones, template wildwuchs, **kahlschlag total: ~5 467 LOC** |
| 4 | integrator | [`phase4_ext.md`](./phase4_ext.md) | 1 044 | 3 simulated integrations (Qwen3.5-A3B / Gemma 5 / Mamba2-hybrid), `ArchPlugin` proposal, **"<500 LOC = new model" target** |
| 5 | orchestrator | [`phase5_synthesis.md`](./phase5_synthesis.md) | 653 | Roadmaps (perf/maint/ext), cross-axis ranking, risks, **30/60/90 day plan** |

**Total analysis:** 4 890 LOC of MD across 5 reports, ~277 KB. No source files modified.

---

## Top-3 Wins (already brecher)

1. **Decode is at-the-roof bandwidth-bound.** `gemv_nvfp4_moe_decode_kernel` hits
   ~261 tok/s vs ~270 tok/s ceiling on Qwen3-Coder NVFP4
   [P2 §1 — `src/quant/nvfp4_gemm.cu:855`].
2. **Public C ABI is clean.** 24 functions, 4 headers, zero internal-header
   leakage in `include/imp/` [P1 §2 — `include/imp/imp.h:1-142`].
3. **Hot-loop allocator hygiene is honored.** No `cudaMalloc/cudaFree` in true
   per-token loops [P2 §2.6, §5.6].

## Top-3 Brüche (must change)

1. **NVFP4 MoE prefill 14.7× slower than vLLM** on the same CUTLASS template.
   Gap is 100 % around-the-kernel: activation-quant fusion missing, per-layer
   launch overhead, scheduler maturity
   [P2 §6 — `src/compute/gemm_cutlass_grouped_3x.cu:30-86`,
   `src/graph/executor_forward_moe.cu:559-563`].
2. **`graph/` is a 15.8 KLOC god-layer.** 21-param `gemm_dispatch_impl`,
   6-map `WeightCaches` god-struct, duplicate per-qtype dispatch table
   [P3 §1.2, §2 #1, §5.2 #4 — `src/graph/executor_kernels.cu:2003-2269`,
   `src/graph/executor.h:286`, `src/compute/weight_dispatch.cu:73-125`].
3. **Per-arch behavior is `if (cfg.arch == ModelArch::GEMMA4)` × 40** across
   `executor_attention.cu` (14), `executor_forward_moe.cu` (19),
   `executor_forward.cu` (2), `engine.cpp` (5+)
   [P3 §1.5, P4 §0, §2.4].

---

## Key Numbers

| Metric | Value | Source |
|---|---|---|
| Total src LOC | ~90 000 | P1 §3 |
| Ballast quantified | ~5 390 LOC (P1) / ~5 467 LOC (P3) → converged **~5 460 LOC = 6.1 %** | P1 §7, P3 §10 |
| Single largest streichkandidat | `src/compute/mmq_q4k_v2.cu` — 1 667 LOC | P1 §7, P3 §10 |
| Worst danger zone | `src/graph/executor_kernels.cu` — 2 327 LOC, god-dispatch | P3 §2 #1 |
| Today's "new MoE arch" cost | ~280-650 LOC across 25-27 files, 2-4 days | P4 §1 |
| Target "<500 LOC = new model" | ~120-200 LOC in a single plugin file | P4 §5 |
| Top quick win | Fuse SwiGLU + activation-quant in MoE down-phase, 2-3 days | P5 §2.2 M1 |
| Top big bet | Close NVFP4 MoE prefill 14.7× gap, 6-12 weeks → ~3× | P5 §2.3 B1 |
| First domino | R0: env-var + error-handling sweep (16 `IMP_*` getenvs → `RuntimeConfig`, 12 `throw`s → `IMP_LOG_ERROR`) | P5 §5 |
| Cross-axis champion | R5: `GemmKernel` registry — -1 000 LOC, wins perf+maint+ext | P5 §5 |

---

## 30 / 60 / 90 Day Plan (one-liners — full plan in P5 §8)

- **Day 0-30 (foundation):** Quick Wins QW1-QW8 + R0 (env-var/error-handling sweep) + retire `mmq_q4k_v2.cu` + relocate `src/compute/bench_*` to `tests/bench/`. Net: ~3 700 LOC out, zero perf regression, branch ready for R5.
- **Day 31-60 (cross-axis):** R5 (`GemmKernel` registry) lands, killing `WeightCaches` + 21-param dispatch + duplicate `weight_dispatch.cu` table. M1 (SwiGLU+act-quant fusion) ships, first slice of the 14.7× gap. M2-M4 (D2H-sync removal, graph pool sizing) land. Net: -1 000 LOC, +5-10 % pp on Qwen3-Coder NVFP4.
- **Day 61-90 (ext + perf big bet):** R6 (`ArchPlugin` interface) lands; first migration (Qwen3.5-A3B) proves "<500 LOC = new arch". M3 (Phase-4 MoE-prefill graphs) lands. B1 prefill gap shrinks from 14.7× to ~5-7× (M1+M3 combined). Net: new arch onboarding 25 files → 1 file; pp on Qwen3-Coder NVFP4 from 1 258 → ~4 000-5 000 tok/s.

---

## Open Questions for the Maintainer

Distilled from P5 §9. Each phrased as a closed question with the recommended default in brackets — pick or override.

1. Delete `src/compute/mmq_q4k_v2.cu` (1 667 LOC) now, before R5? [**recommend: delete now**; restore from git if a dense-Q4_K_M model without fp16_cache ever materializes — but the memo has been "pending real-world win" for 3 weeks]
2. Promote `IMP_*` env vars to `RuntimeConfig` fields with public C API setters, or keep as undocumented env knobs? [**recommend: promote** — all 16 are read on hot paths; env-read-per-call is a perf + maint sin]
3. Pin CUTLASS to v4.5.0 in **both** `CMakeLists.txt:74` and `Dockerfile:27` (today they disagree), or stay on v4.4.2 in containers? [**recommend: v4.5.0 everywhere** — v4.5.0 fixed the NVFP4 non-determinism per memo `cutlass_nvfp4_sm120_nondeterministic_2026_05_05.md`]
4. Move `src/compute/bench_*` TUs (~1 772 LOC) to `tests/bench/` outside of the library archive? [**recommend: yes** — they ship in `libimp.a` today, bloating the binary for zero runtime value]
5. Retire WMMA paths entirely now that `mma.sync` covers all sm_120a use cases, or keep as debug reference? [**recommend: retire** — flagged in P1 §7 as ballast, listed in P3 §3 as template wildwuchs]
6. Make the `ArchPlugin` migration internal-only first (Qwen3.5-A3B + Gemma-4 only) before declaring it stable? [**recommend: yes** — 2 in-tree migrations before promoting to a public extension point]
7. The Big Bet (B1, NVFP4 prefill gap) needs CUTLASS scheduler-tuning work upstream — file an issue with NVIDIA CUTLASS or fork? [**recommend: file upstream issue + carry local patch** — fork-only loses future CUTLASS fixes]
8. Phase 4's "Mamba2-hybrid" simulation says Nemotron-H needed `RecurrentState` polymorphism that doesn't exist today. Add it pre-emptively, or wait for a real hybrid model request? [**recommend: wait** — premature abstraction; the cost is documented in P4 §3 so the next attempt has the blueprint]

---

## Risks (P5 §6 summary)

- **R5 / GemmKernel registry:** highest-blast-radius refactor; every model touches it. Mitigation: ship behind `IMP_DISPATCH_V2=1` for one release cycle, gate verify on parity vs. main, ratchet to default in cycle N+1.
- **mmq_q4k_v2 deletion:** marginal regression risk on dense Q4_K_M-without-fp16_cache — none of imp's currently supported models match that profile, but a Llama-3.3-70B Q4_K_M release would resurface it. Mitigation: keep the commit reachable via git log; restore-cost is low.
- **NVFP4 prefill big-bet (B1):** scheduler maturity is upstream-bound; some of the gap may not be closeable on imp's side. Mitigation: instrument before/after with `ncu` traces, set a 5× ceiling as success criterion (not 1×).
- **`ArchPlugin` v1 over-engineering:** premature seams hurt more than missing seams. Mitigation: derive the interface from 2 in-tree migrations (Qwen3.5-A3B + Gemma-4), not from a hypothetical Mamba2.

---

## Cross-Phase Conflicts

Surfaced in P5 §7 — the cases where two phase reports landed on different sides of a decision:

- **`mmq_q4k_v2.cu` — dead or dormant?** P1 §7 lists it as the single largest streichkandidat; P2 §6 leaves the door open ("maybe a future dense-Q4_K_M release"); P3 §10 puts it back on the delete list. **Synthesis verdict (P5 §7):** delete now. The "future dense Q4_K_M model" has been pending 3 weeks per `mmq_q4k_v2_phase2_shipped_2026_05_16.md`; restoration from git is cheap.
- **WMMA retirement risk.** P1 calls it ballast, P3 calls it template wildwuchs, P2 doesn't take a firm stance. **Synthesis verdict:** retire. All sm_120a use cases now covered by `mma.sync`.

---

## How to Use This Report

1. Skim this file for the brecher-statement and key numbers.
2. Read [`phase5_synthesis.md`](./phase5_synthesis.md) for the full roadmap and 30/60/90 plan.
3. Drill into [`phase1_inventory.md`](./phase1_inventory.md) → [`phase4_ext.md`](./phase4_ext.md) for evidence behind any specific claim — each citation in this doc points to a phase report section, and each phase report section cites `file:line` in the source tree.
4. The branch `review/architecture-2026-05-16` holds 6 commits (one per phase + this master report). No source files were modified. No merge, no PR — the branch is a parking spot for the analysis.

— Generated by Claude Code, multi-subagent orchestration, 2026-05-16.
