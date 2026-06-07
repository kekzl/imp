# VRAM / weight-cache structure audit — 2026-06-07

Scope: where model-weight VRAM goes, which copies coexist, and where the
caching/budget architecture has lost structure. All findings verified against
source AND real bench logs (Qwen3-8B Q8_0 GGUF, Qwen3-14B Q6_K GGUF on clean
`main`). Diagnostic-only; no code changed.

## TL;DR

Per GGUF weight, up to **three** representations are resident at once:
the original GGUF blocks (used by prefill), an NVFP4 decode cache (used by
decode), and a CUTLASS scale-factor buffer (**used by nobody on GGUF**). On top
of that, three independent systems decide tiers and budget, only the weakest of
which actually drives allocation, and two cleanup passes that look load-bearing
are dead no-ops. The owner's instinct ("null Struktur") is correct: the cache
layer is mid-refactor and the migration stalled.

Measured residency (clean main, real logs):

| Model | GGUF source (prefill) | NVFP4 decode | CUTLASS SF | weight held ≈ |
|-------|-----------------------|--------------|------------|---------------|
| Qwen3-8B Q8_0  | ~8.0 GB | 4.06 GB | 0.45 GB (dead) | **1.55×** |
| Qwen3-14B Q6_K | ~9.76 GB | 6.69 GB | 0.83 GB (dead) | **1.8×** |

`GPU memory: 16396 MiB used` for an 8B-Q8 model that weighs 8.3 GB on disk.

---

## Gap G1 — three tier-decision systems, only the worst one rules

There are three places that decide "which tier does weight X get", and they
disagree by design:

1. **`StoragePlanner`** (`src/runtime/storage_planner.cpp`) — fully implemented,
   source-qtype-aware (`effective_capabilities(kind, source_qtype)`), has a
   budget-pressure downgrade loop. **Completely unused.**
   `src/runtime/vram_budget.cpp:174`: *"Heuristic still drives allocation;
   planner output diagnostic only (5.1.5)."*
2. **Heuristic** (`vram_budget.cpp:106-271`) — source-*blind*
   (`nvfp4_beneficial(qtype)`, never consults `kind` or the capability table).
   This is what actually sizes the caches.
3. **Runtime `nvfp4_beneficial`** — a **second copy** of the same predicate at
   `src/exec/pre_dequant_internal.h:98`, called per-tensor during the Phase 0-3
   build to decide which weights physically get an NVFP4 slot. Identical logic to
   the `vram_budget.cpp:108` lambda, different file → silent drift risk.

Live divergence in the logs: planner projects **10055 MiB**, heuristic projects
**7504 + 833 MiB**. Nobody reconciles the gap; the planner's answer is logged and
thrown away. The planner has been "diagnostic only" since commit 5.1.5 and the
flip to make it authoritative ("retire heuristic") was never started.

## Gap G2 — Phase 4b "drop redundant GGUF source" is a dead no-op

`pre_dequant_phase4b_drop_redundant_sources_` (`pre_dequant_phase4_tensor_registry.cu:463`)
is wired into the pipeline with `actually_free = true`, and a sibling diagnostic
prints *"7668 MiB of original GGUF could be freed … deferred to Commit 5.1.4.b."*

But it frees **nothing**. The skip guard at lines 489-490:

```cpp
if (wcache_.cutlass_nvfp4.count(t.data) > 0 ||
    wcache_.nvfp4.count(t.data) > 0) { /* skip */ }
```

`wcache_.nvfp4` is keyed on the **source pointer** `t.data`, so this is true for
*every* NVFP4-overlay weight — i.e. exactly the weights `can_drop_source()` just
flagged as droppable. Result: `marked_count = 0`, and the "Phase-4b freed N
sources" log line never appears in any real run (confirmed: only the 4a
diagnostic prints).

Worse, the diagnostic's premise is now **false**: it claims the overlay "covers
prefill + decode", but since the IMMA raw-read prefill work (#617), GGUF prefill
reads the **original GGUF source** directly —
`executor_kernels.cu:1868` → `mmq_q8_imma_gemm(h.source_data, …)`. The GGUF source
is *not* redundant; prefill needs it. So the "7.6/9.8 GB could be freed" number is
misleading, the free path can't fire, and both halves of the feature are dead.

## Gap G3 — the CUTLASS scale-factor buffer is dead weight on GGUF

For every GGUF weight, Phase 3b builds a `cutlass_nvfp4` entry (borrowed packed
data + a freshly-allocated SfAtom scale buffer). Logs: `wcache actual:
nvfp4=253 cutlass_nvfp4=253`.

That CUTLASS SF buffer (0.45 GB Q8 / 0.83 GB Q6K) is **never read** for GGUF
models, verified three independent ways:
- The planner says it shouldn't exist: `plan-ideal tiers: … cutlass_nvfp4=0`.
- GGUF weights get `primary_tier = NVFP4`, but the CUTLASS prefill path only fires
  for `primary_tier == CUTLASS_NVFP4` (`executor_kernels.cu:1894`) — the native
  SafeTensors-NVFP4 case.
- Prefill goes through IMMA raw-read on the source; decode goes through the plain
  `nvfp4` GEMV. Neither touches `cutlass_nvfp4`.

This is pure waste on the GGUF path. (Note: this is *not* double-storage of the
packed weights — `cutlass_nvfp4.data` borrows `nvfp4.packed_data`, only the scale
factors are a second buffer. That part is clean.)

## Gap G4 — GGUF source ⊕ NVFP4 coexistence is structural, not a leak

The big 1.55–1.8× residency is the GGUF source (prefill, IMMA raw-read) plus the
NVFP4 decode cache (decode, GEMV) living side by side. This is a *deliberate*
prefill/decode split — two different representations for two different kernels —
not a bug, and not trivially removable while both paths run in one process. It
*is* the dominant VRAM cost and worth a design decision: is keeping a full FP-rep
for prefill the right call now that IMMA raw-read decode-format kernels exist, or
could decode and prefill share one representation?

## Gap G5 — no cross-cache ownership/dedup; mode-asymmetric FP16 freeing

- `VRAMAllocator` tracks tagged bytes for reporting but does no dedup across the
  ~9 weight-cache maps (`fp16/fp8/nvfp4/nvfp4_moe/cutlass_nvfp4/cutlass_mxfp4/
  q4k_imma/fused_kv/fused_gate_up`). Each map is independent; correctness of
  "this weight lives in exactly one tier" rests entirely on the build-order
  conventions in the Phase 1-3 lambdas, not on any invariant.
- FP16-cache freeing is **mode-asymmetric**: Mode 0 frees each FP16 entry as it
  seeds the NVFP4 build (`phase3…:449-457`); Mode 1 "additive" keeps them. *(Claim
  surfaced by sub-agent; needs a targeted measurement on a model that actually
  builds an FP16 cache before trusting the magnitude — Q8/Q6K early-return out of
  FP16 so they don't exhibit it.)*

---

## Suggested cleanup order (cheap → structural)

1. **G3 (cheap win):** stop building `cutlass_nvfp4` SF buffers for GGUF
   (`primary_tier == NVFP4`) models. Frees 0.45–0.83 GB, zero behaviour change.
   Verify no native-NVFP4 regression.
2. **G2 (correctness/clarity):** either make Phase 4b actually free droppable
   sources (re-key the skip guard off the *owned* packed pointer, not the source
   key) **or** delete the dead pass + the misleading diagnostic. As-is it's
   confusing dead code claiming a 7.6 GB win that can't happen.
3. **G1 (de-dup the truth):** collapse the two `nvfp4_beneficial` copies to one
   shared symbol; decide whether to finish the planner migration or delete the
   planner. Two-and-a-half tier oracles is the root "null Struktur" smell.
4. **G4 (design decision, biggest VRAM):** revisit the prefill/decode dual-rep now
   that IMMA raw-read exists — owner call, not a mechanical fix.
