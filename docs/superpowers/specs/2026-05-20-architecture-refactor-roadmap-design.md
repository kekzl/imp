# imp — Architecture Refactor Roadmap

**Status:** Design / Spec
**Date:** 2026-05-20
**Author:** Raphael Friedmann (with Claude Opus 4.7)
**Type:** Master roadmap. Each phase will get its own implementation plan via the `writing-plans` skill.

---

## 1. Motivation

A ruthless review of the current architecture (vs. the canonical pipeline diagram on the `docs/arch-diagram` branch) surfaced ten structural problems. They cluster naturally into three risk tiers: dead/misleading structures, internal coupling, and architectural layering. This document captures the agreed phasing.

The diagram itself admits several of the problems verbatim — "1 GiB S-matrix!" with an exclamation mark, "Attention Dispatcher (3 layers, 10 paths)", and the cuBLAS-vs-FMHA gate label "TRUE (typical Qwen3/Gemma-4)" that quietly disables ~5000 LOC of FMHA code in the default path. The roadmap below treats these as work items to resolve, not constants to preserve.

## 2. Ground Rules

- **Order:** risk-ascending. Phase 1 = pure cleanup with no perf surface. Phase 5 = invasive layer redrawing.
- **Phase gate:** the **critical PRs** of phase N must be merged before the critical PRs of phase N+1 land. **Soft PRs** within a phase may slip into the next phase or be dropped entirely.
- **Perf invariant:** no hard CI gate during refactor PRs. `tests/perf_baseline.json` keeps running, but the 3% decode / 5% prefill thresholds are advisory, not blocking, for refactor work. Author judges case-by-case; a regression demands a written reason, not an automatic veto.
- **Correctness invariant:** `make verify-fast` must stay green on every PR. `make verify` must stay green at every phase boundary.
- **Solo work:** the author is the only committer. No coordination overhead; the roadmap exists for self-discipline and future Claude sessions, not team alignment.
- **No third-party dependencies added.** The CLAUDE.md rule stands. Refactors are within the existing dependency set.

## 3. The Five Phases

### Phase 1 — Lügen entfernen
**Status (2026-05-20):** Closed. Landed: #287 (delete dead Graph IR), #286 + #289 (architecture diagram + README link), #291 (narrative companion docs/architecture.md), #290 (soft PR — rename src/graph/ → src/exec/), #292 (closeout: roadmap.md path sweep + this status line). Deferred follow-ups: stale `src/graph/` paths in `docs/plans/*_2026_05_17.md` historical design memos and `review/phase3_maint.md` audit snapshot — these are point-in-time records.

**Goal:** What the repo claims must be true.

**Critical PRs**
1. Delete the dead `Graph` IR.
   - Files: `src/graph/op.h`, `src/graph/op.cpp`, `src/graph/graph.h`, `src/graph/graph.cpp` (~250 LOC).
   - Evidence of deadness: `grep -rln 'OpType::' src/` returns only those four files. `Graph::to_dot()` is never called from a live execution path. No executor consumes `OpNode`.
   - Verify: `make verify-fast` green after deletion.
2. Merge the architecture diagram to `main`.
   - Source: commit `163bcf7` on `docs/arch-diagram`. Brings `docs/architecture.dot`, `docs/architecture.svg`, `docs/architecture.png`.
   - Link from `README.md` (top-level "Architecture" section) and from `CLAUDE.md`'s "Where to look" table.
3. Write `docs/architecture.md` as the canonical narrative companion to the SVG. Describes the four phases (load → init → prefill → decode) in text, links to the section headers in source.

**Soft PRs**
- Rename `src/graph/` → `src/exec/`. Mechanical: include-path updates + CMake target renames. Cannot land until critical PR 1 is merged.
- Move `Graph::to_dot()`-style debug visualization to a single `tools/imp-arch-dump` binary if there is appetite for runtime arch inspection. Otherwise drop.

**Expected outcome:** ~300 LOC deleted, one canonical architecture artifact, one directory that no longer lies about its contents.

---

### Phase 2 — Attention-Dispatcher entrümpeln
**Goal:** From "3 layers, 10 paths" to "Default cuBLAS / Sliding-Fallback / Decode-paged" plus one FMHA variant for non-cuBLAS cases.

**Critical PRs**
1. Archive `attention_fmha_sm120_cluster.cu` (1102 LOC). Refuted by author in memo `fmha_tma_lever_refuted_2026_05_14.md` — cp.async beats TMA bulk on SM120. Move source to `docs/archive/fmha_sm120_cluster/` with a resurrection memo describing the build flag that would re-enable it.
2. Archive `attention_fmha_mxf4nvf4_sm120.cu`. Path was never default-on for any production NVFP4 model in current benchmarks. Same archive treatment.
3. Archive `attention_naive.cu`. Only the Gemma-4 hd=512 fallback referenced it, and the chunked-prefill path (`gemma4_chunked_prefill_2026_05_15.md`) replaced that need.
4. Simplify `executor_attention.cu:847` gate from the current 4-clause predicate to a 2-level switch:
   - **Prefill:** default cuBLAS+softmax → FMHA fallback (one variant only) → sliding-window-special-case.
   - **Decode:** keep the existing `switch(cache_dtype)` at `:996` but flatten the wrapper paths.
   - Target: file goes from 1300 LOC to ≤700 LOC.

**Soft PRs**
- Write `docs/attention-dispatch.md` documenting the final matrix (dtype × prefill/decode × paged × sliding) and which kernel handles each cell.
- Add a compile-only resurrection test per archived FMHA variant so future ports detect bitrot before runtime.
- Move `attention_paged_common.cuh` includes into the per-dtype paged files (eliminate the umbrella header).

**Out of scope here:** The "1 GiB S-matrix" wound. That belongs to Phase 5 because it requires a rewrite of the cuBLAS-attention prefill path, not just a deletion.

**Expected outcome:** ≈5000 LOC of compiled-but-rarely-fired FMHA code leaves the hot path; build time drops; the dispatch logic becomes legible enough to fit in a doc page.

---

### Phase 3 — Pre-Dequant + Quant-Zoo aufräumen
**Goal:** `executor_pre_dequant.cu` (2693 LOC) becomes a thin dispatcher over a format registry, with one source file per quant family.

**Critical PRs**
1. Define the registry surface in `src/graph/pre_dequant_registry.h`:
   ```cpp
   using DequantFn = void(*)(const WeightHandle&, void* dst, /* ... */);
   void register_pre_dequant(QType src, QType dst, DequantFn fn);
   DequantFn get_pre_dequant(QType src, QType dst);
   ```
2. Split `executor_pre_dequant.cu` by quant family:
   - `pre_dequant_q8.cu` (Q8_0 → FP16, Q8_0 → FP8)
   - `pre_dequant_q4k.cu` (Q4_K_M → FP16, Q4_K_M → INT8 via Phase 3 IMMA path)
   - `pre_dequant_mxfp4.cu`
   - `pre_dequant_nvfp4.cu`
   - `pre_dequant_fp16_cache.cu` (the FP16-on-device cache promotion)
3. Reduce remaining `executor_pre_dequant.cu` to ≤200 LOC — pure dispatcher + the registry seed.

**Soft PRs**
- Apply the same registry pattern to `src/graph/gemm_kernel_*.cu` (8 files: `cutlass_nvfp4`, `fp8`, `generic_dequant`, `gguf`, `mxfp4`, `nvfp4_gemm`, `nvfp4_gemv`, `q4k_imma`). Currently dispatched via `gemm_kernel_registry.cu` — confirm or refactor.
- Reconcile the `src/quant/dequant_*.cu` files (in-place dequant for hot paths) with `src/graph/pre_dequant_*.cu` (one-shot dequant at init). Currently both exist; document or merge.

**Expected outcome:** Adding a new quant format becomes one file plus one `register_pre_dequant` call. The 2693-LOC monolith is gone.

---

### Phase 4 — Engine.cpp zerteilen
**Goal:** `src/runtime/engine.cpp` shrinks from 3112 LOC to ≤800 LOC. Each subsystem becomes a named owner with constructor injection.

**Critical PRs** (one subsystem per PR, ordered by independence)
1. **`InitResolver`** — owns `init_resolve_kv_dtype_policy_`, `init_resolve_ssm_dtype_`, `init_resolve_fp8_prefill_`, `init_resolve_quant_flags_`, `init_compute_max_seq_len_`, `init_apply_debug_raw_overrides_`. Pure function-style; takes `RuntimeConfig` + `Model` in, returns a resolved `EngineRuntime` struct.
2. **`WeightUploadOrchestrator`** — owns `init_weights` (and indirectly calls `upload_weight` / `upload_expert_weights` from `model/weight_upload.cu`). Includes the FP8/NVFP4 pre-dequant trigger.
3. **`KVCacheInitializer`** — owns `init_kv_cache` and the paged-block geometry decisions. Outputs a configured `KVCacheManager`.
4. **`WorkspaceBuilder`** — owns `init_features` plus the three `executor_workspace_*.cu` files (they already exist as separate files; this PR ties the ownership together).
5. **`Warmup`** — owns `warmup()` and the CUDA graph capture orchestration. Wraps `cuda_graph.cu`.
6. **`Scheduler`** — owns `step`, `step_async_graph_resume`, `step_schedule`, `step_prefill`, `prefill_allocate_kv_blocks_`. The actual hot-loop driver.
7. **`SamplingHelper` + `StopController`** — owns `fill_sampling_params`, `upload_penalties`, `should_stop`, `track_think_state`, `build_banned_token_list`. Separate PRs if they grow; combined here because they share state.
8. **`Engine` becomes a façade** — owns construction wiring, exposes the same `Engine::*` public methods, delegates each one to the appropriate subsystem. Target: ≤800 LOC.

**Soft PRs**
- Move `src/runtime/mtp_forward.cu` → `src/compute/mtp_forward.cu`. It is a kernel file, not an orchestrator.
- Move `src/runtime/vision_pipeline.{cpp,h}` → `src/vision/vision_pipeline.{cpp,h}`. Aligns with the other `vision_*` files.
- Promote `MTPSubsystem` and `VisionSubsystem` from inline Engine members to named classes (currently they exist as embedded `vision_` member + `mtp_*_` fields). Same pattern as Scheduler.

**Expected outcome:** Engine.cpp diff per future feature drops dramatically. New subsystems can be added by writing one file + wiring it in `Engine::init()`. Test boundaries become tractable.

---

### Phase 5 — Schichten und APIs (höchstes Risiko)
**Goal:** Structural honesty. The fuzzy boundaries become sharp.

**Critical PRs**
1. **VRAM ownership consolidation.** Today: `src/memory/vram_allocator.cu`, `src/memory/device_allocator.cu`, `src/memory/pinned_allocator.cpp`, `src/runtime/vram_budget.cpp`, `src/runtime/storage_planner.cpp`. Consolidate into one `MemoryManager` (target location: `src/memory/memory_manager.{cu,h}`) that owns:
   - Device VRAM pool (existing `vram_allocator` logic).
   - Pinned host allocator.
   - Budget tracking (free VRAM, planned reservations).
   - Storage tier planning (which tensor lives on device, which on host).
   - The other files either become thin internal helpers or are deleted.
2. **`RuntimeConfig` de-globalization.** Today: `RuntimeConfig::current()` is a global singleton accessed from any `.cu`/`.cpp` file. Change:
   - `RuntimeConfig` becomes per-`Engine`. Passed by const reference into every subsystem constructor (Phase 4 sets this up).
   - `RuntimeConfig::current()` is kept only at the C-API boundary (`src/api/imp_api.cpp`) — translating C-side config into the per-Engine instance.
   - The `gemma4` section moves to `ModelConfig::arch_overrides` (or a new `ModelOverrides` struct). Model-specific config belongs to the model, not to the runtime.
3. **Public API consolidation.** Today: `imp_generate` / `imp_generate_streaming` are parallel implementations alongside `imp_prefill_with_params` + `imp_decode_step`. Change: `imp_generate*` becomes a thin wrapper that loops `imp_prefill_with_params` then `imp_decode_step` until stop. Single source of truth for each generation feature.

**Soft PRs**
- **Tiled streaming softmax for cuBLAS attention prefill.** The "1 GiB S-matrix" allocation in `init_features` (`executor_workspace_buffers.cu`). Two paths to consider:
  - (a) Implement tiled softmax in the cuBLAS path — keep the cuBLAS QK^T and PV calls but stream S in tiles, freeing the ~1 GiB.
  - (b) After Phase 2 simplification, evaluate switching the default to the single remaining FMHA variant if it's perf-competitive on the typical Qwen3/Gemma-4 configs that motivated the cuBLAS gate originally.
  - Either path is multi-week and perf-sensitive; lands as soft because the structural Phase 5 critical PRs unblock the work, not gate-block on it.
- Rename `src/graph/` → `src/exec/` if it didn't land as a Phase 1 soft PR.

**Expected outcome:** Hidden singletons are gone. Adding a new model architecture means writing one `ModelOverrides` block, not adding a section to `RuntimeConfig`. The public API has one canonical entry point per operation.

---

## 4. Transition Logic

A phase is **closed** when its critical PRs are merged to `main` and `make verify` is green. Soft PRs may stay open in branches or be dropped.

Phase N+1's critical PRs may start work (branches, drafts) while Phase N's soft PRs are still pending, but no Phase N+1 critical PR is **merged** until Phase N is closed.

Concurrent soft PRs across phases are allowed as long as the merge order respects the critical chain.

## 5. Verification Strategy

| Check | When | Tool |
|---|---|---|
| Build green | Every PR | `make build` |
| Tests green | Every PR | `make verify-fast` (~90s) |
| Full suite green | Every phase boundary | `make verify` (~5min) |
| Perf snapshot | Every phase boundary, archived | `scripts/gen_perf_baseline.sh` (advisory) |
| Architecture diagram still matches code | Phase 1 close, Phase 4 close, Phase 5 close | Manual review of `docs/architecture.{md,svg}` |
| `MEMORY.md` updates | Per phase | When a memo's referenced file/symbol is moved, the memo gets a "moved to X in PR #Y" note rather than silent staleness |

No CI gate is added by this refactor. The advisory perf snapshot is for the author's awareness; regressions documented in the PR body are acceptable if the author justifies them.

## 6. Decomposition Rationale

Why this specific phase order, given the original critique's ten points:

| Critique point | Phase | Why here |
|---|---|---|
| 1. Dead `Graph` IR | 1 | Pure deletion, no risk, sets honesty baseline |
| 3. Attention 10-path dispatcher | 2 | Largest dead-code mass; LOW risk because the dead paths are dead by gate flag, not by latent bug |
| 8. `executor_pre_dequant.cu` 2693 LOC | 3 | Touches no public API; structural split with bounded blast radius |
| 2. `engine.cpp` god class | 4 | Largest diff surface; benefits from Phase 1-3 cleanup landing first (less to split) |
| 4. 1 GiB S-matrix | 5 (soft) | Perf-sensitive rewrite; needs Phase 2 dispatcher simplification as prerequisite |
| 5. `RuntimeConfig` singleton + `gemma4` | 5 | Touches every subsystem; benefits from Phase 4's subsystem extraction |
| 6. Public API two-door | 5 | Same; the consolidated wrapper depends on `Scheduler` from Phase 4 |
| 7. VRAM owner unclear | 5 | Cross-cuts memory + runtime + graph; needs Phase 4 to define which subsystem holds the new `MemoryManager` |
| 9. Hand-written jinja/tokenizer | — | Out of scope. CLAUDE.md "no new third-party deps" rule stands; replacing these is a separate decision |
| 10. Diagram on side branch | 1 | One-shot merge |

## 7. Open Decisions

These are deliberately left for the per-phase plans:

- **Phase 2:** Which single FMHA variant survives as the non-cuBLAS fallback? Decision deferred to Phase 2 plan — depends on a fresh A/B against the simplified dispatcher.
- **Phase 4:** Whether `SamplingHelper` and `StopController` become two classes or one. Decision deferred until the extraction starts and the actual coupling is visible.
- **Phase 5 soft:** Tiled softmax (option a) vs. FMHA-default-switch (option b) for the 1 GiB S-matrix. Decision deferred until Phase 2 closes and the surviving FMHA variant has fresh benchmarks.

## 8. Non-Goals

- Reducing or eliminating CUDA dependencies.
- Adding third-party libraries (Jinja, tokenizers, allocators).
- Cross-architecture portability (sm_80, sm_90, sm_100). The sm_120a-only rule stays.
- Replacing CUTLASS or cuBLAS as the GEMM backends.
- Changing the public C-API symbol table beyond Phase 5's consolidation.

## 9. Per-Phase Plans

Each phase below will, when work starts on it, get its own implementation plan via the `writing-plans` skill:

- `docs/superpowers/specs/<date>-architecture-refactor-phase-1-plan.md`
- `docs/superpowers/specs/<date>-architecture-refactor-phase-2-plan.md`
- … and so on through Phase 5.

This roadmap document is the parent. It is updated only when a phase's scope materially changes during execution (e.g., a planned soft PR escalates to critical, or a phase splits in two).
