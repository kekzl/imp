# Roadmap Inventory — 2026-05-08

Purpose: discover and classify every roadmap-like artifact in the repo so the master plan only includes items that can plausibly meet the Quality Gate in this run.

## Sources scanned

- `docs/roadmap.md` — primary roadmap (7 known limitations + 3 perf items + research interest)
- `docs/sm120-real-perf-plan.md` — five-lever sm_120a perf plan
- `docs/audit/safetensors_audit.md` — Phase 1 + Phase 2 outcomes, list of remaining-unresolved items
- `git log --oneline -200` — recent merges + memory pointers
- `gh issue list --state open --limit 50` — none open
- `gh pr list --state open --limit 30` — none open
- `rg -i "TODO|FIXME|XXX" src/` — no priority/owner-tagged markers in src

## Quality-Gate baseline

A roadmap item is **FEASIBLE** only if it can be landed in this run without:
- adding any new third-party dependency,
- skipping/disabling tests to ship green,
- shipping a partial kernel without a numerical reference,
- multi-week kernel rewrites or new compute architecture,
- shortcut/symptom-mask fixes that don't address root cause.

Borderline calls go to UNCERTAIN, not FEASIBLE.

## Items

### A. `docs/roadmap.md` — Known limitations

| # | Item | Severity | Feasibility | Reasoning |
|---|---|---|---|---|
| L1 | FP8 KV cache: Gemma-4 carve-out | P2 | INFEASIBLE | Per-layer head_dim awareness in KV write/read kernels — multi-day kernel work + multi-model regression sweep |
| L2 | Chunked prefill: paged-prefill kernel pending | P2 | INFEASIBLE | New paged-prefill attention kernel + multi-context regression suite. Mitigation already shipped in PR #114 (single-chunk default) |
| L3 | NVFP4 SmoothQuant `input_scale` (Mistral-3.2) | P2 | INFEASIBLE | Refuted as scalar-alpha; real fix needs per-channel SmoothQuant scaling vector applied during activation quant — multi-week, only one test model |
| L4 | Qwen3.5-27B MXFP4 fails at load | P3 | INFEASIBLE | Needs host-dequant path + StoragePlanner integration — large change, single-model unlock |
| L5 | Gemma-4 Q4_K_M code-gen drift | P3 | INFEASIBLE | FP16 accumulator drift across 30 layers; "use Q5/Q8" workaround documented |
| L6 | MoE expert offload disables CUDA Graphs | P2 | INFEASIBLE | Needs device-side LRU prefetch + async pipeline — significant runtime kernel work |

### B. `docs/roadmap.md` — Performance work

| # | Item | Severity | Feasibility | Reasoning |
|---|---|---|---|---|
| P1 | Closing TurboQuant–FP8 gap | P3 | INFEASIBLE | "Algorithm-inherent" per roadmap — needs MXFP4 K-direction redesign |
| P2 | pp=512 on large dense models | P3 | INFEASIBLE | cuBLAS autotune variance, "not gating any user" per roadmap |
| P3 | Speculative decoding | P3 | OBSOLETE | Investigated and shelved per roadmap; CLI flags already removed in `7380ea8` |

### C. `docs/roadmap.md` — Research interest

| # | Item | Severity | Feasibility | Reasoning |
|---|---|---|---|---|
| R1 | `cudaMemcpyWithAttributesAsync` L2 hints | P3 | INFEASIBLE | Narrow benefit (single-transfer L2 persist), no benchmark scaffolding |
| R2 | `add.f32x2` native PTX | P3 | UNCERTAIN→INFEASIBLE | Like SFU-exp2 lever (`memory/lever5_sfu_exp2_neutral_2026_05_06.md`): mathematically tempting, likely net-zero on imp's HMMA-pipe-bound paths. Without bench harness change, can't prove benefit |
| R3 | `cp.async.bulk` `.ignore_oob` | P3 | INFEASIBLE | Paged-attention kernel rewrite with TMA descriptors |
| R4 | `st.async.b128` 16B async stores | P3 | INFEASIBLE | KV writeback rewrite |
| R5 | `cvt .bf16x2 ↔ narrow (.e2m1x2, .e4m3x2)` | P2 | OBSOLETE | Shipped in PR #125 (commit `3eb7ef5`, vectorized FP4 dequant in paged KV decode +25.6%) |
| R6 | `.scale_vec::4X` with `.ue8m0` for MXFP4 MMA | P3 | INFEASIBLE | MXFP4 SafeTensors path does not exist on imp — no test fleet |
| R7 | FP4 PV (Phase-3 P×V in attention) | P2 | INFEASIBLE | "Quality-risky"; prereq is PV-only A/B test harness which doesn't exist; SageAttention3-style two-level accumulator is multi-week |
| R8a | K2 MLA (DeepSeek latent KV) | P2 | INFEASIBLE | Multi-week — separate attention path |
| R8b | K5 H2O token eviction | P3 | INFEASIBLE | Score-based eviction kernel + scheduler change |
| R8c | K8 CPU offload async prefetch | P3 | INFEASIBLE | Significant runtime + memory-tier work |
| R9 | BitDecoding (MXFP4 KV) | P3 | INFEASIBLE | Builds on MXFP4 KV which is research-grade |
| R10 | DeltaKV residual KV compression | P3 | INFEASIBLE | New KV compression algorithm + kernel |

### D. `docs/sm120-real-perf-plan.md` — Five-lever plan

| # | Lever | Status (per memory) | Feasibility now |
|---|---|---|---|
| Lv1 | SSM-Layer in `cutlass_nvfp4_cache` | OBSOLETE | DONE — `fb92be9` registers `ssm_in/ssm_out` at `executor_pre_dequant.cu:466-467` |
| Lv2 | NVFP4 KV-cache with HW absmax | OBSOLETE | DONE 2026-05-07 (PR #108) + vectorized PTX cvt 2026-05-08 (PR #125) |
| Lv3 | CLC-persistent for continuous batching | INFEASIBLE | "Multi-user only, +10-20%"; single-author single-target experiment, no multi-tenant validation harness |
| Lv4 | Tile-tuning for 99-KiB SMEM (correction from 228) | OBSOLETE | A/B'd 2026-05-06: baseline `<128,128,128>` wins all variants per `memory/lever4_tile_tuning_baseline_wins_2026_05_06.md` |
| Lv5 | Online-softmax SFU-exp2 micro-opt | OBSOLETE | Shipped+reverted 2026-05-06 (`memory/lever5_sfu_exp2_neutral_2026_05_06.md`). Net-zero |

### E. `docs/audit/safetensors_audit.md` — items "truly unresolved"

| # | Item | Severity | Feasibility | Reasoning |
|---|---|---|---|---|
| AU1 | GLM architecture mapping | P2 | INFEASIBLE | Needs new `GLM` enum + dedicated forward path (multi-week per audit note) |
| AU2 | Native SentencePiece (`.model`) parser | P2 | UNCERTAIN | "~few hundred LoC" but full byte-fallback/Unigram protobuf decode + tokenization-roundtrip golden against actual checkpoints is a significant testing burden. Defer in favor of correctness-hardening work this run |
| AU3 | AWQ INT4 dequant kernel | P2 | INFEASIBLE | "Multi-week" per audit, requires column-packed + interleave-permutation kernel |
| AU4 | DeepSeek MLA attention | P1 | INFEASIBLE | "Multi-week effort" per audit |
| AU5 | Multimodal SafeTensors loaders | P2 | INFEASIBLE | Per-family work for Qwen-VL / Llava / Pixtral / Gemma-3 vision |
| AU6 | Tiktoken parser | P3 | INFEASIBLE | Out of scope per audit ("uncommon in supported families; ignored") |

## Summary

**FEASIBLE: 0** — none entered the master plan
**UNCERTAIN: 1** (AU2 — SentencePiece native parser; deferred this run in favor of correctness-hardening) — `Status: deferred (see followups.md)`
**INFEASIBLE: 21** — all in `docs/audit/followups.md`
**OBSOLETE: 5** (P3 spec-decode, R5 cvt narrow PTX, Lv1 SSM cache, Lv2 NVFP4 KV, Lv4 tile-tuning, Lv5 SFU-exp2)

This is consistent with imp's mature state: most listed items are either explicit dead-ends (already investigated, documented, dropped) or multi-week kernel/architecture work that no autonomous single-run can land at full Quality Gate.

**Decision:** Per the mission's conditional model, this run focuses entirely on Objective 1 — SafeTensors + NVFP4 hardening. Every roadmap entry is captured in `docs/audit/followups.md` with reasoning. AU2 stays UNCERTAIN and is re-evaluated only if Objective-1 work completes with runway.

## Open commits cross-reference

`git log` since the audit branch landed (`b6c2b9c`):
- `3eb7ef5` perf(nvfp4): vectorized FP4 dequant in paged KV decode (+25.6%) — closes R5
- `7539949` chore(skills): commit imp-specific Claude Code skills
- `8c39b8f` ci: auto-enable squash auto-merge
- `79051f4` ci: ccache + base image bump
- `454ca58` fix(nvfp4): graph-safe gemm_nvfp4 dequant fallback — Lever 2 stage 1

No `roadmap:` / `next:` / `followup:` markers in the last 200 commits beyond historical references already absorbed by `docs/roadmap.md`.
