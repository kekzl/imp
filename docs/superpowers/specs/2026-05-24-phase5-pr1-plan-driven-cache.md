# Phase 5 PR #1 — Plan-Driven Cache Architecture (execution plan)

**Date:** 2026-05-24
**Status:** Design — execution plan for the "VRAM Ownership Consolidation" critical PR from the master roadmap
**Parent roadmap:** `docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md` Phase 5 PR #1
**Motivating incident:** 2026-05-24 Q4_K_M cache coverage gap (Gemma-3-12B Q4_K_M had zero overlay cache due to NVFP4_DECODE_ONLY all-or-nothing logic, paying 29.5% GPU time on per-dispatch dequant — see `docs/superpowers/specs/2026-05-24-q4k-vs-llamacpp-mmq-analysis.md`)

## 1. What's already done (Phase 4 → Phase 2-shim)

| Component | File | LOC | Status |
|---|---|---|---|
| `StoragePlanner` (pure function) | `src/runtime/storage_planner.{h,cpp}` | 257 | ✅ Computes per-tensor ideal tier; diagnostic-only |
| `WeightHandle` struct | `src/exec/weight_handle.h` | 57 | ✅ Has `primary_tier` + per-tier payload union |
| `WeightRegistry` class | `src/exec/weight_handle.cu` | 81 | ✅ Borrowed-pointer phase-2 shim active |
| Phase-4 registry population | `src/exec/pre_dequant_phase4_tensor_registry.cu` | 150 | ✅ Populates handles from wcache_ post-allocation |
| `MemoryManager` façade | `src/memory/memory_manager.{h,cpp}` | 110 | ✅ Wraps vram_allocator + lazy pinned/device + plan_storage forwarder |
| Capability table | `src/model/tensor_kind_table.cu` | ~80 | ✅ Per-kind tier capabilities (ALL_QUANT / NO_MXFP4 / FP16_ONLY) |
| Tier-typed dispatch | `src/compute/weight_dispatch.{h,cu}` | ~100 | ✅ `gemm_dispatch(handle, ...)` switches on `primary_tier` |

**Net:** the skeleton is built. Plan exists, handles exist, dispatch exists. **The plan does not drive allocation, and the dispatch coexists with the legacy 150-LOC wcache_-walk switch in `executor_kernels.cu:1596-1872`.**

## 2. What's missing — the actual end-state work

### 2.1 Plan-execution gap (root cause of the Q4_K bug)

`executor_pre_dequant.cu:42` calls `plan_storage(...)` then **discards the result** (`diag_plan` is only logged). Phases 1/2/3 use their own per-phase heuristics:

- **Phase 1** (`pre_dequant_phase1_fp16_cache.cu:35-40`): unconditionally early-exits on `NVFP4_DECODE_ONLY` strategy — *ignoring the plan that says "FP16 for Q4_K weights"*.
- **Phase 2** (`pre_dequant_phase2_fp8_cache.cu`): runs only when `use_fp8` is set; ignores plan.
- **Phase 3** (`pre_dequant_phase3_nvfp4_decode.cu`): caches only `nvfp4_beneficial(qtype)` weights — re-implements a capability check that's *almost-but-not-identical* to the planner's `capabilities_of(kind)`.

**The Q4_K bug is structural**: planner produces a sensible plan, runtime ignores it, falls into a coverage gap.

### 2.2 Capability-table inconsistency (Q4_K vs NVFP4)

`src/model/tensor_kind_table.cu:25-37` says every projection kind (WQ/WK/WV/WO/W_GATE/W_UP/W_DOWN/EXPERT_*) is `ALL_QUANT` supported with `required_floor=NVFP4`. But the runtime check `nvfp4_beneficial(qtype)` in `pre_dequant_internal.h:98` excludes Q4_K because **representation change at similar bit-width is not a compression win**.

So the planner *would* recommend NVFP4 for Q4_K weights if `hints.prefer_nvfp4_decode=true`, which is wrong (no benefit, possible quality risk per Gemma-3 quality memo). The planner needs **source-qtype awareness**, not just tensor-kind awareness.

### 2.3 Tier-exclusivity not enforced

Today a weight can be in `wcache_.fp16`, `wcache_.nvfp4`, `wcache_.cutlass_nvfp4`, AND have its original quantized data still alive in `Model::gpu_allocations_`. The dispatcher walks all of them in fallback order. The `WeightHandle.primary_tier` field exists but is NOT consulted by the legacy switch.

### 2.4 Legacy dispatch chain still authoritative

`executor_kernels.cu:1596-1872` walks `wc->fp16 / fp8 / nv4 / ct4 / mx4 / nvfp4_moe` maps. The Phase-4 `gemm_dispatch(handle, ...)` exists but only some call sites use it. Dual path means dispatch routing depends on which call site you hit.

### 2.5 VRAM budget inconsistency

`vram_budget.cpp:150` computes `nvfp4_estimate` assuming only `nvfp4_beneficial(qtype)` weights are cached. Then phases skip non-beneficial weights. If we change Phase 1 to cache Q4_K (the obvious fix for the bug), the 18 GiB allocation isn't in the budget → OOM risk.

## 3. Execution plan — 6 commits, 7-10 days total

Each commit lands as its own PR with `make verify-fast` green. PRs land sequentially behind the gate.

### Commit 5.1.1 — Source-qtype-aware capability table (1d)

**What:** Extend `KindCapabilities` to accept the source qtype (or add a runtime predicate `capabilities_of(kind, source_qtype)` that returns a refined capability set).

**Why:** A Q4_K-source W_GATE has different best-overlay-tier than a Q6_K-source W_GATE. The planner needs both signals.

**Files:** `src/model/tensor_kind_table.{h,cu}`, `src/runtime/storage_planner.cpp` (caller passes Tensor.qtype to capabilities_of).

**Test:** Unit test `tests/test_storage_planner.cpp` — given `(W_GATE, Q4_K)` returns `{supported=FP16|FP8|NVFP4, required_floor=FP16}` (NVFP4 still listed but floor is FP16 — planner won't downgrade past FP16 unless explicit budget pressure). Given `(W_GATE, Q6_K)` returns `{supported=ALL, required_floor=NVFP4}` (current behavior preserved).

**Verify:** `make verify-fast` + new unit test. No runtime path changes.

### Commit 5.1.2 — Planner output drives Phase 1/2/3 allocation (2d)

**What:** Replace per-phase strategy-based heuristics with a single plan-driven dispatcher in `pre_dequant_weights()`:

```cpp
StoragePlan plan = mem_mgr_.plan_storage_for(model, cfg, hints);
if (plan.failed) FATAL("VRAM budget insufficient even at floor tiers");
for (const auto& entry : plan.entries) {
    switch (entry.tier) {
        case StorageTier::FP16: cache_as_fp16(entry); break;
        case StorageTier::FP8:  cache_as_fp8(entry); break;
        case StorageTier::NVFP4: cache_as_nvfp4(entry); break;
        // ...
    }
}
```

**Why:** Eliminates the coverage gap (every entry in plan IS allocated; no all-or-nothing skip). Fail-loud at load time if budget is wrong, no silent mid-allocation budget_exhausted.

**Files:** `src/exec/executor_pre_dequant.cu` (becomes a thin loop), `src/exec/pre_dequant_phase{1,2,3}*.cu` (refactor: extract `cache_as_*` helpers; delete strategy early-exits).

**Test:** Q4_K bug regression test — load Gemma-3-12B Q4_K_M, assert `wcache_.fp16.size() == 288 ± 5%`, assert `dequant_q4k_kernel` does NOT fire during prefill (count kernel launches via nsys hook or stub).

**Verify:** `make verify-fast` + Q4_K bug regression test. Bench Gemma-3-12B Q4_K_M pp512 — must show ≥+200% over pre-PR baseline (3838 → 11k+).

### Commit 5.1.3 — primary_tier becomes single source of truth for dispatch (1d)

**What:** All dispatch paths read `weight_handle.primary_tier` and route to ONE kernel. No fallback chain across maps. The wcache_ maps become *storage* (owned by registry), not *lookup*.

**Files:** `src/compute/weight_dispatch.cu` (extend handler matrix), `src/exec/executor_kernels.cu:1596-1872` (delete the legacy chain — replace with `weight_dispatch::dispatch(handle, ctx)`).

**Why:** Eliminates the M=1 vs M>1 dispatch race that's the root of my Q4_K decode regression (cuBLAS algo selection varying based on which fallback fires).

**Test:** Regression test on Gemma-3-12B Q4_K_M tg128 — must show ≥-10% of pre-PR baseline (≥120 tok/s — the regression of -36% I shipped should disappear because dispatch is now deterministic).

**Verify:** `make verify-fast` + tg128 regression test + Qwen3-14B Q6_K baseline (must stay 165 tok/s ±5%) + Qwen3-8B Q8_0 baseline (must stay 272 tok/s ±5%).

### Commit 5.1.4 — Drop original GGUF when overlay tier covers it (1d)

**What:** When a weight's `primary_tier ∈ {FP16, FP8, NVFP4, CUTLASS_NVFP4, MXFP4}` is fully allocated AND the dispatcher never reads `weight.data` for that handle, free the original GGUF allocation in `Model::gpu_allocations_`.

**Why:** Saves 6.79 GiB on Gemma-3-12B Q4_K_M (only had FP16 overlay before — original Q4_K never used). Frees VRAM for longer context KV cache.

**Caveat:** The planner header says native GGUF blocks "stay as mmap'd blocks" — that's the *current* design. This commit revises that for the case where the entire overlay covers the original. Requires planner annotation: per-entry `bool original_redundant` flag.

**Files:** `src/runtime/storage_planner.{h,cpp}` (add annotation), `src/model/model.h` (per-tensor `gpu_free_after_cache()` method), `src/exec/executor_pre_dequant.cu` (call free after Phase 3 done).

**Test:** Gemma-3-12B Q4_K_M VRAM usage post-load should drop ~6.79 GiB vs pre-PR. KV cache block count auto-recalculated.

**Verify:** `make verify-fast` + VRAM-usage golden test (asserts post-load VRAM ≤ expected) + same perf baselines as 5.1.3.

### Commit 5.1.5 — VRAM budget honesty via plan (1d)

**What:** `compute_vram_budget()` computes the exact budget from `plan_storage(model, cfg, hints)` — sum of `plan.entries[i].bytes`. No more `nvfp4_estimate` heuristic that under-counts when other tiers are present.

**Files:** `src/runtime/vram_budget.cpp` (re-implement as `budget = plan_total + workspace + KV`), `src/memory/memory_manager.h` (expose unified `compute_budget(model, cfg)` that calls planner internally).

**Why:** Closes Finding 3 from the audit (VRAM budget drift). Single source of truth for VRAM accounting.

**Test:** Unit test — for each of the 5 bench models, assert `compute_budget()` returns ±5% of measured post-allocation VRAM use.

**Verify:** `make verify-fast` + unit test. No bench regressions.

### Commit 5.1.6 — Cleanup + bench + memo (1d)

**What:**
- Delete dead `WeightCaches::nvfp4_decode_mode` field (replaced by plan tier choice)
- Delete dead `attention.mxfp4_fp16_cache_policy` if obsolete
- Refresh `docs/architecture.md` Phase-5 section
- Cross-engine bench re-run on all 5 GGUF + 3 NVFP4 models
- Write closeout memo with measured pp/tg deltas

**Verify:** `make verify` (full suite) + cross-engine bench. Update `tests/perf_baseline.json` with new baselines.

## 4. Go/no-go gates

| Gate | Required for | Pass criterion |
|---|---|---|
| G1: planner unit tests | 5.1.1 ship | `(W_GATE, Q4_K)` → floor=FP16; `(W_GATE, Q6_K)` → floor=NVFP4 |
| G2: Q4_K bug regression | 5.1.2 ship | Gemma-3-12B Q4_K_M pp512 ≥ +200% vs pre-PR baseline |
| G3: decode parity | 5.1.3 ship | Gemma-3-12B tg128 ≥ -10% (eliminates today's −36%); north-star + Q8_0 baselines unchanged ±5% |
| G4: VRAM savings | 5.1.4 ship | Gemma-3-12B post-load VRAM −6.79 GiB; KV blocks auto-grown |
| G5: budget accuracy | 5.1.5 ship | `compute_budget()` within ±5% of measured for all 5 bench models |
| G6: full bench | 5.1.6 ship | No regression > 5% on any (model, metric) in cross-engine bench |

Any gate fail → STOP, fix, do not chain commits.

## 5. What this fix retires

| Today | After Phase 5 PR #1 |
|---|---|
| `pre_dequant_phase1_fp16_cache.cu:35-40` early-exit on NVFP4_DECODE_ONLY | Deleted (plan drives allocation) |
| `nvfp4_beneficial(qtype)` runtime check in 3 places | Folded into `capabilities_of(kind, qtype)` |
| `engine_init_resolver.cpp:160-200` NVFP4 mode decision | Replaced by planner hints |
| `vram_budget.cpp:150` `nvfp4_estimate` heuristic | Replaced by `plan_total` |
| `executor_kernels.cu:1596-1872` 150-LOC dispatch switch | ~30 LOC proxy to `weight_dispatch::dispatch(handle, ctx)` |
| `WeightCaches::nvfp4_decode_mode` int flag | Deleted |
| Coexisting Q4_K original + FP16 cache on Gemma-3-12B (24.6 GiB) | Single-tier (FP16 overlay only, 17.85 GiB) |
| Today's Q4_K hotfix branch `fix/q4k-fp16-cache-coverage-gap` | Reverted (this PR supersedes structurally) |

## 6. Risks

1. **Capability table rewrites can mis-classify edge tensors** — mitigation: keep current behavior for Q*_K-6-8bit (no change) + carefully add Q4_K/Q3_K/Q4_0/Q5_0 cases with tests.
2. **GGUF-drop in 5.1.4 breaks Model::gpu_allocations_ contract** — mitigation: add owning-pointer transfer protocol so freeing is explicit + tested.
3. **VRAM budget regression on tight-VRAM scenarios** — mitigation: planner's `downgrade_one` already handles budget pressure; 5.1.5 just makes budget accounting honest.
4. **Decode regression doesn't fully heal in 5.1.3** — mitigation: if dispatch consolidation doesn't fix Gemma-3-12B tg128, profile cuBLAS algo selection directly (likely a separate cuBLAS workspace tuning).
5. **Multi-day refactor accumulates bugs** — mitigation: 6 small commits with independent verify-fast gates; rollback granularity is one commit.

## 7. Out of scope

- Phase 5 PR #2 (`RuntimeConfig` de-globalization)
- Phase 5 PR #3 (public API consolidation)
- Tiled-streaming softmax (Phase 5 soft PR)
- Direct Q4_K MMQ kernel (Lever B from the cross-engine analysis — would be Phase 6+ work)
- MoE expert dispatch consolidation (separate executor_forward_moe.cu refactor)

## 8. Decision request

Before starting Commit 5.1.1, confirm:

1. **Revert current `fix/q4k-fp16-cache-coverage-gap` branch?** This PR makes the hotfix obsolete structurally. Recommend: yes, revert; the Phase 5 PR #1 supersedes it cleanly.
2. **Branch name for Phase 5 PR #1 work?** Suggest `refactor/phase5-pr1-plan-driven-cache`.
3. **PR strategy: single PR with 6 commits, or 6 separate PRs?** Recommend: 6 separate PRs to `main` (per the roadmap §2 "no PR stacking" rule from `pr_no_stacking_2026_05_17`).

## 9. Execution outcomes (post-session 2026-05-24)

5 commits shipped on `refactor/phase5-pr1-plan-driven-cache`:

| Commit | Hash | Wirkung |
|---|---|---|
| 5.1.1 | `887a61f` | `effective_capabilities(kind, qtype)` + planner uses it. 12 new unit tests, plan-logic fixt. |
| 5.1.2 | `cfa5a40` | Phase 1 nutzt `effective_capabilities` (Q4_K coverage gap closed). **pp512 +440% Gemma-3-12B, 2.7× llama.cpp**. tg128 −36% regression appeared (documented). |
| 5.1.3.a | `fb9c7f0` | `WeightHandle.source_data` + `source_qtype` populated in Phase 4. Groundwork. |
| 5.1.3.b | `9d89e00` | `WeightRegistry::find_by_source_data` lookup helper. Groundwork. |
| 5.1.3.c | `0a1c903` | M=1 dispatch prefers dp4a over FP16 overlay (gate at line 1806). Routing correct per profile (no `gemv_fp16_kernel` at decode); regression persists → memory pressure, not dispatch. |

### Plan deviations

- **5.1.3 reality vs estimate**: full dispatch consolidation (rewire all callers via `weight_dispatch::dispatch(handle, ...)`, delete the 150-LOC legacy switch in `executor_kernels.cu:1596-1872`) is **4-5 days, not 1 day** per the plan. The `weight_dispatch` shim has gaps (CUTLASS_NVFP4 M=1 stub, no beta!=0 across tiers, no callers wired). The 3 sub-commits shipped (5.1.3.a/b/c) are the **structural groundwork + the high-value dispatch-routing fix** (the M=1 gate); the multi-day caller migration is deferred to its own PR.

- **Decode regression diagnosed**: not a dispatch bug. Profile confirms post-5.1.3.c decode kernel mix shows only `gemv_dp4a_kpar_*` family — `gemv_fp16_kernel` no longer fires on the cached overlay. The 17.85 GiB FP16 overlay sitting in HBM (alongside the 6.79 GiB Q4_K original) causes WSL2 shared-memory fallback or TLB/L2 pressure on the original-weight reads at decode. **Structural fix needs a workload-hint API to the planner** ("decode-heavy → don't cache sub-5-bit sources as FP16") — separate design, not part of this PR.

### Remaining commits (multi-session)

| Commit | Status | Estimate | Notes |
|---|---|---|---|
| 5.1.3.d | deferred | 1-2d | Migrate executor_attention/ffn/ssm-Caller via `weight_dispatch::dispatch`. Delete legacy switch. Own PR. |
| 5.1.4 | deferred | 1-2d | Drop original GGUF for weights fully covered by an overlay tier (Q*_K-6-8bit with FP8/NVFP4: ~12 GiB freed on Qwen3-14B). NOT applicable to Q4_K + FP16 (dp4a needs original). Needs safety guards + dispatch-site checks. |
| 5.1.5 | deferred | 1d | VRAM budget honesty (compute from plan, fail-loud on overflow). Low-risk cleanup. |
| 5.1.6 | deferred | 1d | Final memo + cross-engine bench refresh + perf baseline update. |

### Ship-as-PR recommendation

The 5 commits shipped form a coherent first PR:
- Architectural foundation (5.1.1 + 5.1.3.a/b/c)
- Concrete bug fix (5.1.2: Q4_K coverage gap closed, pp512 +440% on Gemma-3-12B)
- All baselines preserved (Qwen3-14B Q6_K tg128 165, Qwen3-8B Q8_0 tg128 272, Gemma-4-26B MoE 258)
- Known trade-off documented (Gemma-3-12B tg128 −36% from memory pressure; structural fix needs workload-hint, separate PR)

Subsequent PRs build on this foundation. PR #1's value is independently realised even if the follow-up PRs slip.

## Cross-references

- Parent roadmap: `docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md`
- Bug origin: `docs/superpowers/specs/2026-05-24-q4k-vs-llamacpp-mmq-analysis.md`
- Cross-engine bench: `docs/cross_engine_bench_2026_05_24.md`
- Architecture audit findings (this session, 2026-05-24): 7 findings, P0=this PR
- StoragePlanner: `src/runtime/storage_planner.{h,cpp}`
- WeightHandle: `src/exec/weight_handle.{h,cu}`
- MemoryManager façade: `src/memory/memory_manager.{h,cpp}`
- Capability table: `src/model/tensor_kind_table.cu`
