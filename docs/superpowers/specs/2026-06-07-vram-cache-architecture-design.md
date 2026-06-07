# VRAM / weight-cache architecture — clean rebuild (design)

Date: 2026-06-07
Status: approved skeleton → implementation planning
Audit basis: `docs/audit/vram_cache_structure_2026_06_07.md`

## Problem

The weight-cache layer is mid-refactor and the migration stalled. Three
independent systems decide a weight's storage tier and disagree by design; two
cleanup passes that look load-bearing are dead no-ops; per GGUF weight up to
three representations are resident. Measured on clean `main`:

- Qwen3-8B Q8_0: GGUF source ~8.0 GB + NVFP4 decode 4.06 GB + CUTLASS SF 0.45 GB
  (dead) → weight held ~1.55×. `GPU memory: 16396 MiB used` for an 8.3 GB model.
- Qwen3-14B Q6_K: ~9.76 GB + 6.69 GB + 0.83 GB (dead) → ~1.8×.

Audit gaps (see audit doc for file:line):
- **G1** — three tier oracles: `StoragePlanner` (correct, source-aware, but
  "diagnostic only (5.1.5)"), the `vram_budget.cpp` heuristic (source-blind,
  actually allocates), and a *second* `nvfp4_beneficial` copy in
  `pre_dequant_internal.h`. Planner projects 10055 MiB vs heuristic 7504+833 —
  nobody reconciles.
- **G2** — Phase 4b "drop redundant GGUF source" is a dead no-op: its skip guard
  `wcache_.nvfp4.count(t.data) > 0` matches every NVFP4 weight (map keyed on the
  source pointer) → `marked_count = 0`. Its sibling diagnostic claims "7.6 GB
  could be freed, overlay covers prefill" — false since IMMA raw-read prefill
  (#617) reads `h.source_data` directly.
- **G3** — the CUTLASS SF buffer is dead weight on GGUF (0.45–0.83 GB): planner
  says `cutlass_nvfp4=0`, GGUF weights are `primary_tier=NVFP4`, the CUTLASS
  prefill path only fires for `primary_tier==CUTLASS_NVFP4`.
- **G4** — GGUF source ⊕ NVFP4 coexistence (the big 1.55–1.8×) is a *deliberate*
  prefill/decode split, both perf-justified (measured). Not a leak. Out of scope
  for the strict-quality-neutral rebuild.
- **G5** — no cross-cache ownership invariant; correctness rests on build-order
  conventions + `borrowed`/`owned` bool flags across ~9 cache maps.

### Evidence that calibrated scope

- **native-NVFP4 (strategic models) is already lean**: `fp16=0`, prefill via
  CUTLASS-NVFP4 TC, decode via NVFP4 GEMV, both from the same source. The 1.55–
  1.8× doubling is a **GGUF-only** (legacy-path) phenomenon.
- VRAM↔speed is **shape/kernel-dependent, measured both ways**: GDN projections
  FP16→NVFP4 *regress* decode −9..−20% (kernel efficiency beats byte count);
  `nvfp4_attn_proj` *gains* +3.8%; dropping the GGUF NVFP4 decode cache costs
  −27% (Gemma 165→130). So the big GGUF posts are real Roofline trades, not
  structural waste.

## Goals & constraints

- **Strictly quality-neutral**: output PPL/greedy must not change. All existing
  arch carve-outs preserved, only encoded cleanly instead of scattered.
- **Performance-gated**: every sub-PR within the 3% decode / 5% prefill gate.
- **Staged**: each stage independently mergeable, benched, reversible.
- VRAM is the scarce resource; recover it where free, never trade it for
  measurable speed/quality loss.

## Roadmap (staged)

1. **Stage 1 — One overlay-tier truth (this spec).** `StoragePlanner` becomes
   authoritative for the overlay-tier decision; heuristic + the duplicate
   `nvfp4_beneficial` die; dead posts (G3, G2, the double CUTLASS build) fall out
   as a consequence. Low risk, ~1 GB (G3 0.45–0.83 + double CUTLASS ~0.4) +
   structural clarity.
2. **Stage 2 — Arena ownership (RAII).** One stream-aware arena owner holds all
   weight allocations; the ~9 cache maps become non-owning views; owning-vs-
   borrowing moves into the type system. Kills the lifecycle-bug class (G5, the
   model-swap leak). Its own spec; builds on Stage 1's reduced cache variety.
3. **Stage 3 (optional) — Honest budget.** The planner's downgrade loop is used
   for real (not diagnostic); the `reserve` fudge is replaced by plan bytes.
   Closes G1 fully.

Stages 2 and 3 are sketched here for context only and get their own specs.

---

## Stage 1 design

### Scope

Stage 1 consolidates the **overlay-tier decision** (which tensor gets an
FP16/FP8/NVFP4/CUTLASS overlay cache). It does **not** touch: the native GGUF
source blocks (`Model::gpu_allocations_`), the GGUF source ⊕ NVFP4 dual-rep
(G4), physical allocation mechanics, or ownership/lifecycle (Stage 2). Only the
decision is unified.

The `StoragePlan` already scopes exactly this: per its header comment it
describes "the overlay layer — tensors whose storage tier is a runtime decision",
while native GGUF blocks "bypass the plan/registry entirely". We make the
already-correct plan authoritative instead of diagnostic.

### Architecture: one producer, many readers

**Producer** — `plan_storage(model, cfg, hints)` runs once, early (before the
pre-dequant phases), and the result is held on the executor for the model's
lifetime (today it runs twice, both discarded as diagnostics).

**Readers** (replace their own tier logic with a plan lookup):

| Site | Today | Stage 1 |
|------|-------|---------|
| `vram_budget.cpp` sizing | `nvfp4_beneficial(qtype)` heuristic estimate | read `plan.projected_vram_bytes` + per-tier sums |
| Phase 1 FP16 build | `plan_routes_to_fp16(kind, qtype)` local lambda | `plan.tier(id) == FP16` |
| Phase 2 FP8 build | local FP8 routing | `plan.tier(id) == FP8` |
| Phase 3 NVFP4 build | `nvfp4_beneficial(qtype, decode_all)` | `plan.tier(id) ∈ {NVFP4, CUTLASS_NVFP4}` |
| Phase 3b CUTLASS build | iterate all `wcache_.nvfp4` | only entries `plan.tier(id) == CUTLASS_NVFP4` |
| Phase 4b drop-source | broken `wcache.count` guard | `plan.tier(id)` says source is overlay-covered |

A small plan-index helper (`tier_of(TensorID)` / `tier_of(const void* src)`)
backs the lookups; `TensorID` already keys the registry.

### Arch carve-outs as plan rules

Today the arch-specific quality rules are scattered across
`engine_init_resolver.cpp`, the phase lambdas, and `effective_capabilities`.
Stage 1 gathers them into **one** post-plan pass `apply_arch_rules(plan, cfg,
hints)` (chosen over expanding `effective_capabilities` so the clean kind×qtype
capability table stays free of arch special-cases):

- **gemma-3**: nvfp4_beneficial weights (Q6_K/Q8_0/Q5_K) require a companion FP16
  entry so the NVFP4 decode cache is built FROM the FP16 copy, not from scratch
  (from-scratch corrupts gemma-3 decode → `<pad>`/IMA). Encoded as: for
  `arch==GEMMA3`, NVFP4-tier entries also get an FP16 backing flag. This requires
  extending `StoragePlan::Entry` with a `bool fp16_companion` (or equivalent), so
  Stage 1 touches the plan schema — Phase 1 reads it to keep the FP16 copy alive.
- **GDN `ssm_in`/`ssm_out`**: FP16 floor by default (recurrent precision); NVFP4
  only when `hints.nvfp4_ssm_proj`. (Wide-GDN NVFP4 measured to *regress* speed —
  keeping FP16 is correct for speed, not just quality.)
- **`nvfp4_attn_proj`, `nvfp4_lm_head`, `nvfp4_lm_head_gdn`**: opt-in hints that
  promote specific recipe-excluded BF16 kinds to an NVFP4 entry.
- **native-NVFP4 prefill**: stays CUTLASS-NVFP4 (no FP16 prefill copy — already
  the runtime reality; the stale "all dense weights get FP16 prefill" comment in
  `pre_dequant_phase4_tensor_registry.cu` is corrected).

One function decides; all phases read. This is the structural insurance against
the scattered-`if(arch==…)` bug class (the rejected 2026-06-07 FP16-gate diff,
the #428×#434 regression).

### VRAM by-catch (consequences, not separate features)

- **G3**: the plan assigns `cutlass_nvfp4` only where `tier==CUTLASS_NVFP4`
  (native-NVFP4). GGUF weights are `tier==NVFP4` → Phase 3b skips them → the
  0.45–0.83 GB dead SF buffer is never built.
- **double CUTLASS (native)**: the plan enumerates each tensor once; Phase 0
  (prequant) and Phase 3 reconcile against the single plan instead of both
  building → the ~0.4 GB overlap goes away. *(Exact overlap to be confirmed in
  the implementation plan — flagged, not assumed.)*
- **G2**: Phase 4b reads the plan to decide droppability honestly. Since IMMA
  raw-read prefill needs the GGUF source, the honest answer for GGUF NVFP4-tier
  weights is "not droppable" — so the misleading "7.6 GB could be freed"
  diagnostic is removed and the dead free-path is either fixed to free genuinely-
  redundant sources or deleted. (No GGUF VRAM recovered here — the value is
  removing false code.)

### Migration (incremental, each sub-PR gated)

1. Hold the plan persistently on the executor; expose `tier_of()` lookups. No
   behaviour change yet (plan still not read by builders) — pure plumbing.
2. Convert **one phase at a time** to plan-query, each as its own PR, proving
   parity (golden + perf + canaries) before the next.
3. Convert `vram_budget.cpp` sizing to plan bytes last (it gates KV size, so it
   moves once the per-phase builds are plan-driven and stable).
4. Delete the heuristic `nvfp4_beneficial` lambda and the
   `pre_dequant_internal.h` copy; the dead posts (G3, double CUTLASS) and the
   Phase-4b diagnostic fall out in the same PRs that make their producers
   plan-driven.

Order rationale: plumbing first (zero risk), then per-phase (smallest provable
steps), sizing last (highest blast radius).

### Verification / canaries

Every sub-PR must be green on all four:
- **golden output** — bit-neutral greedy output (the strict-quality-neutral gate)
- **perf gate** — 3% decode / 5% prefill vs `tests/perf_baseline.json`
- **gemma-3-12b Q4_K_M coherence** — the crash canary (clean main: coherent;
  rejected diff: `<pad>`/IMA). `check-degeneration` battery.
- **Qwen3-14B Q6_K decode** — the north-star, the −15.5% canary from the rejected
  diff. `verify-north-star`.

These four are exactly the net that caught the rejected FP16-gate diff this
morning; making them a per-PR gate is the point.

### Out of scope (Stage 1)

- GGUF source ⊕ NVFP4 dual-rep (G4) — real Roofline trade, strict-neutral keeps it.
- Ownership/RAII (G5) — Stage 2.
- Planner downgrade loop driving real allocation (G1 final) — Stage 3.
- Any kernel or dispatch change. Stage 1 is allocation-decision only.

### Risks

- **Plan/runtime parity gaps**: the plan's per-(kind,qtype) tier must exactly
  reproduce today's per-phase decisions for every model in the zoo, or output
  changes. Mitigation: per-phase migration with golden parity; the plan already
  runs as a diagnostic, so a pre-migration A/B of "plan tier vs actual wcache
  tier" per model surfaces mismatches before any builder is switched.
- **`kind` classification**: the planner uses field position (L.wq→WQ); any
  tensor reaching a builder without a matching plan entry must fall back safely
  (treat as "no overlay", today's uncached path) and log, never silently
  mis-tier.
- **MoE / hybrid coverage**: expert and shared-expert kinds, GDN exclusions
  (`gdn_gate` intentionally not enumerated) must be preserved exactly.
