# Weight-Storage Refactor — Design Spec

**Date:** 2026-04-21
**Author:** Raphael Friedmann (brainstormed with Claude)
**Status:** Spec for implementation — implementation plan not yet written
**Target branch:** feature branch off `main` (not `qwen36-gdn-infrastructure`)

## Problem

Adding new model architectures to imp (Qwen 3.5 GDN, Gemma-4 MoE, Qwen 3.6)
repeatedly triggers bugs in the weight-storage layer. Most recent example
(commit `d0e9b03`): Qwen 3.6 (GDN + MoE) ran in NVFP4-only decode mode, which
freed the entire FP16 weight cache — including `ssm_in` and `ssm_out`, which
have **no NVFP4 or FP8 replacement**. GDN GEMMs then silently fell back to
on-the-fly dequant-to-FP16 scratch, producing per-element drift that cascaded
to incoherent output.

### Root cause

Quantization mode is currently a **global** decision (e.g. "NVFP4-only mode"),
but storage eligibility is **per-tensor** — some tensor kinds have no
quantized replacement. There is no single source of truth for
"this tensor must be kept at FP16 because no quantized path exists".

Two architectural symptoms:

1. **Eviction policy is global.** `pre_dequant_weights` clears entire
   `wcache_.fp16` map when switching mode. The `d0e9b03` fix is a post-hoc
   probe (`if not in nvfp4 and not in fp8 → keep`), not a structural fix.

2. **Dispatch logic is distributed.** 158 accesses to `wcache_.{fp16,fp8,nvfp4,...}`
   scattered across 7 files (`executor_attention.cu`, `executor_ffn.cu`,
   `executor_forward_moe.cu`, `executor_ssm_gdn.cu`, `executor_forward.cu`,
   `executor_pre_dequant.cu`, `executor_workspace*`). Each consumer
   reimplements the "try NVFP4 → try FP8 → try FP16 → fall back to
   dequant-scratch" cascade.

When a new model lands tensors that don't fit the implicit assumptions (e.g.
GDN's `ssm_in` having no quantized form), both the eviction and the dispatch
paths have to be audited by hand. This is error-prone and has cost days of
debugging on every recent model integration.

## Goals

- **Primary:** Eliminate the class of bugs where global mode switches drop
  tensors that specific consumers still need.
- **Primary:** Make per-tensor storage tier decisions visible in one place
  instead of reconstructed by probing at each call site.
- **Secondary:** Make new model integrations lower-risk. A new model with
  novel tensor kinds should require at most: a new `TensorKind` enum entry,
  a capability-table row, and a loader name-matcher case. No architectural
  spelunking.
- **Non-goal:** Fix the Qwen 3.6 GDN per-element drift. That is an unrelated
  precision bug in the scan itself. This refactor only removes the
  contaminating "dequant-scratch fallback" variable from the debug search.
- **Non-goal:** Change which GEMM kernels are used. Same kernels, different
  dispatch layer on top.

## Architecture

### Current pipeline (today)

```
GGUF mmap
  → Tensor { data=ptr, qtype=Q8_0 }
  → 7× WeightCacheManager maps keyed by source ptr
  → global mode flag (use_fp8, nvfp4_decode_mode, dual_path_quant, ...)
  → 158× wcache_.*.find(ptr) in consumers with cascading fallbacks
```

### New pipeline

```
GGUF mmap
  → Tensor { data=ptr, qtype=Q8_0, kind=TensorKind::SSM_IN } ← loader stamps kind
  → WeightRegistry { handles_: vector<WeightHandle> indexed by TensorID }
  → StoragePlanner (pure function: model + budget + hints → StoragePlan)
  → PlanExecutor (allocate tier storage, convert sources, fill handles)
  → gemm_dispatch(handle, x, y, ...) ← central switch on handle.primary_tier
```

### Three single-responsibility components

- **`TensorKindTable`** answers *"which tiers is a tensor kind allowed to
  live in?"* Pure static data.
- **`StoragePlanner`** answers *"which tier does this specific tensor get
  in this specific run?"* Pure function of `(model, budget, hints)`.
- **`gemm_dispatch`** answers *"which GEMM implementation do I call for this
  handle?"* One central switch statement.

Each component can be tested in isolation. None knows the others' internals.

## Components

### 1. `TensorKind` enum

```cpp
// include/imp/tensor_kind.h
enum class TensorKind : uint8_t {
    // Attention projections
    WQ, WK, WV, WO,
    // FFN / expert projections
    W_GATE, W_UP, W_DOWN,
    EXPERT_GATE, EXPERT_UP, EXPERT_DOWN,
    // Fused variants (populated by planner, not loader)
    FUSED_KV, FUSED_GATE_UP,
    // Embeddings
    TOK_EMBED, LM_HEAD,
    // MoE routing
    ROUTER, SHARED_EXPERT_GATE,
    // GDN / Mamba2 (no quantized path today)
    SSM_IN, SSM_OUT, CONV1D_W, A_LOG, DT_BIAS, BETA,
    // Norms (always FP32)
    NORM_GAIN,
    // Positional
    ROPE_FREQS,
    // Vision (SigLIP)
    SIGLIP_ATTN, SIGLIP_FFN, SIGLIP_NORM, MM_PROJ,
    // Fallback
    UNKNOWN,
    _COUNT,
};
```

### 2. `StorageTier` + `TierMask`

```cpp
// include/imp/storage_tier.h
enum class StorageTier : uint8_t {
    Undefined = 0,   // handle not yet populated — FATAL if dispatched
    FP32,
    FP16,
    FP8,             // E4M3 with per-tensor scale
    NVFP4,           // two-level micro-scale, native decode-GEMV path
    CUTLASS_NVFP4,   // block-scaled, native prefill-GEMM path
    MXFP4,           // alternative prefill-GEMM path
};

// Bitmask for capability declarations. Bit index = (int)StorageTier.
using TierMask = uint32_t;
constexpr TierMask mask(StorageTier t) { return TierMask{1} << (int)t; }
```

### 3. `TensorKindTable` (capability matrix)

```cpp
// src/model/tensor_kind_table.cu
struct KindCapabilities {
    TierMask    supported;       // bitmask of allowed tiers
    StorageTier required_floor;  // minimum quality tier — planner must honor
    bool        fusable;         // can be fused with sibling (WK+WV, W_gate+W_up)
};

constexpr KindCapabilities kKindTable[(int)TensorKind::_COUNT] = {
    [WQ]       = { FP16|FP8|NVFP4|CUTLASS_NVFP4|MXFP4, NVFP4, false },
    [WK]       = { FP16|FP8|NVFP4|CUTLASS_NVFP4,       FP8,   true  },
    [WV]       = { FP16|FP8|NVFP4|CUTLASS_NVFP4,       FP8,   true  },
    [WO]       = { FP16|FP8|NVFP4|CUTLASS_NVFP4|MXFP4, NVFP4, false },
    [W_GATE]   = { FP16|FP8|NVFP4|CUTLASS_NVFP4|MXFP4, NVFP4, true  },
    [W_UP]     = { FP16|FP8|NVFP4|CUTLASS_NVFP4|MXFP4, NVFP4, true  },
    [W_DOWN]   = { FP16|FP8|NVFP4|CUTLASS_NVFP4|MXFP4, NVFP4, false },
    [ROUTER]   = { FP16|FP32,                          FP32,  false },
    [SSM_IN]   = { FP16,                               FP16,  false },
    [SSM_OUT]  = { FP16,                               FP16,  false },
    [CONV1D_W] = { FP16,                               FP16,  false },
    [A_LOG]    = { FP32,                               FP32,  false },
    [NORM_GAIN]= { FP32,                               FP32,  false },
    // ... remaining kinds
};
```

Single source of truth. ~30 lines. `constexpr`. Reviewer reads this file
and knows every constraint in the system.

### 4. `WeightHandle` (POD)

```cpp
struct WeightHandle {
    TensorID    id;
    TensorKind  kind;
    StorageTier primary_tier;
    int64_t     shape[2];

    // Tagged payload — exactly one field active, selected by primary_tier.
    union {
        struct { float* data; }                                fp32;
        struct { half* data; }                                 fp16;
        struct { __nv_fp8_e4m3* data; float* d_scale; }        fp8;
        struct { uint8_t* data; uint8_t* block_scales;
                 float* tensor_scale; float* tensor_scale_2; } nvfp4;
        struct { /* cutlass block-scaled layout */ }           cutlass_nvfp4;
        struct { /* mxfp4 layout */ }                          mxfp4;
    } payload;
};
```

No virtual methods. No `unordered_map::find`. Stored directly in `ModelLayer`:

```cpp
struct ModelLayer {
    WeightHandle wq, wk, wv, wo;
    WeightHandle w_gate, w_up, w_down;
    WeightHandle ssm_in, ssm_out;        // populated only on GDN layers
    // ...
};
```

### 5. `gemm_dispatch` (central switch)

```cpp
// src/compute/weight_dispatch.cu — ONE file, ONE switch
void gemm_dispatch(cublasLtHandle_t lt, const WeightHandle& w,
                   const Tensor& x, Tensor& y,
                   float alpha, float beta, cudaStream_t stream) {
    switch (w.primary_tier) {
        case StorageTier::FP16:
            return gemm_fp16(lt, w.payload.fp16.data, w.shape, x, y, alpha, beta, stream);
        case StorageTier::FP8:
            return gemm_fp8(lt, w.payload.fp8.data, w.payload.fp8.d_scale,
                            w.shape, x, y, alpha, beta, stream);
        case StorageTier::NVFP4:
            return gemm_nvfp4(w.payload.nvfp4.data, w.payload.nvfp4.block_scales,
                              w.payload.nvfp4.tensor_scale, w.shape, x, y,
                              alpha, beta, stream);
        case StorageTier::CUTLASS_NVFP4:
            return gemm_cutlass_nvfp4(/* ... */);
        case StorageTier::MXFP4:
            return gemm_cutlass_mxfp4(/* ... */);
    }
    IMP_LOG_FATAL("gemm_dispatch: handle in invalid tier %d", (int)w.primary_tier);
}

void gemv_dispatch(/* ... */);  // decode path, same shape
void gemm_grouped_dispatch(/* ... */, std::span<const WeightHandle>, /* ... */);  // MoE
```

A `grep gemm_dispatch` shows every caller. Opening `weight_dispatch.cu`
shows every tier path. This is the property that's missing today.

**Non-GEMM tensors (norms, conv1d, rope_freqs, A_log):** these also live
as `WeightHandle` for uniform lifecycle, but they are read directly by
their specialized kernels (e.g. `rmsnorm_kernel(handle.payload.fp32.data, ...)`).
They do not go through `gemm_dispatch` because their storage tier is
single-valued and there is nothing to dispatch on. The uniformity buys
one consistent model of ownership and planning, not uniform dispatch.

### 6. `StoragePlanner`

```cpp
// src/runtime/storage_planner.h
struct PlanHints {
    bool   prefer_nvfp4_decode;
    bool   dual_path_attn_fp8_ffn_nvfp4;
    size_t vram_budget_bytes;
};

struct StoragePlan {
    struct Entry { TensorID id; StorageTier tier; };
    std::vector<Entry> entries;
    size_t projected_vram_bytes;
};

// Pure function — no side effects, no allocations of GPU memory.
StoragePlan plan_storage(const Model& model,
                         const ModelConfig& cfg,
                         const PlanHints& hints);
```

Greedy algorithm:

1. Initialize each tensor at `max(required_floor, user_preferred_tier)`,
   clamped to the supported mask.
2. Sum projected bytes. If > budget, downgrade tensors in priority order
   (largest bytes-saved-per-downgrade first) toward their `required_floor`
   until fit.
3. If still doesn't fit at `required_floor`, return failure (see Error
   handling).

Pure, deterministic, unit-testable.

### 7. `PlanExecutor`

`pre_dequant_weights` becomes this: "take plan, allocate VRAM, run
source → tier conversions, fill handles". No policy decisions — only
mechanical execution of the plan.

## Data flow (steady state post-refactor)

**Model load (once):**

1. GGUF parser identifies each tensor by name; loader assigns `TensorKind`
   via name-matcher.
2. Registry allocates `WeightHandle[N]`, one per logical tensor, with
   `primary_tier = Undefined` and zeroed payload.

**Planning + execution (once, before warmup):**

3. `plan_storage(model, cfg, hints)` returns `StoragePlan`.
4. `execute_plan(plan, registry, vram_alloc, stream)` allocates tier
   storage, runs conversions, fills handle payloads.
5. Source mmap'd pages no longer needed are unmapped.
6. Registry is now **immutable** for this run.

**Forward pass (per token, per layer):**

7. Consumer: `gemm_dispatch(layer.wq, x, y, 1.0f, 0.0f, stream)`.
8. Dispatch reads `primary_tier`, calls matching GEMM. Zero lookups.

### Key invariant

After step 6, the Registry is immutable. No cache drops, no mode switches,
no `fp16.clear()`. **The bug class that motivated this spec is structurally
impossible post-refactor.**

## Fused-weight handling

Today: `fused_kv`, `fused_gate_up` are separate maps keyed by layer index.
Populated lazily after primary caches are ready.

New: Planner decides whether to fuse. If fused, it allocates exactly **one**
`WeightHandle` with `kind=FUSED_KV`, `shape=[2*nkv*hd, d_model]`. Attention
consumer calls `gemm_dispatch(layer.kv_fused, ...)`. The individual
`layer.wk` / `layer.wv` handles remain in an "unused" state (their payloads
are not populated).

Fusion becomes a planner decision, consistent with all other tier decisions.

## Migration strategy

Six phases. Each phase = one PR, committable and testable in isolation.
Old and new code paths coexist through phases 0-4. Legacy deleted only in
phase 5.

### Phase 0 — Skeleton

- Add `include/imp/tensor_kind.h`, `src/model/tensor_kind_table.cu`,
  `src/graph/weight_handle.h`, stub `gemm_dispatch` (returns
  `FATAL` for all tiers initially).
- No consumers touched. No behavior change. Tests unchanged.

### Phase 1 — Loader stamps TensorKind

- `Tensor` struct gains `TensorKind kind` field.
- Loader sets it via name-matcher for GGUF and SafeTensors.
- Verification: a new unit test iterates every tensor in test fixtures
  (Qwen3.5-GDN, Gemma-4-MoE, Qwen3.6-35B) and asserts `kind != UNKNOWN`.
- Still no behavior change at forward-pass time. Existing degeneration
  tests continue to pass unchanged.

### Phase 2 — Handles built alongside `wcache_` maps

- `pre_dequant_weights` now populates **both** the legacy `wcache_` maps
  **and** the new `registry.handles_`.
- Phase-2 tier inference shim: right after the existing logic has decided
  which cache a tensor went into, set `handle.primary_tier` to match
  (`wcache_.nvfp4` membership → `NVFP4`, `wcache_.fp8` membership →
  `FP8`, else `FP16`). The planner stays off in this phase; the tier
  assignment is whatever the legacy code path already chose.
- `gemm_dispatch` implemented as a proxy: reads `handle.primary_tier`,
  looks up the corresponding entry in the legacy `wcache_` map, calls
  the existing GEMM function.
- Functionally identical to pre-refactor. All tests must pass.

### Phase 3 — Consumer migration (one file per PR)

- `executor_attention.cu` → `executor_ffn.cu` → `executor_forward_moe.cu`
  → `executor_ssm_gdn.cu` → remainder.
- Each PR replaces `wcache_.nvfp4.find(L.wq.data)`-style cascades with
  `gemm_dispatch(L.wq, ...)`.
- Benchmark + degeneration parity required on: Qwen3-4B Q8_0,
  Gemma-4-26B-A4B Q5_K_M, Qwen3.5-4B-GDN Q8_0, Qwen3-Coder-30B-A3B NVFP4.

### Phase 4 — Storage flip

- `StoragePlanner` implemented with real logic (phases 0-3 can use
  `.tier = existing_inferred_tier` as a shim).
- `pre_dequant_weights` becomes the `PlanExecutor`. Storage now lives in
  handle-owned allocations, not in the `wcache_` maps.
- `wcache_` maps still exist but remain empty. `gemm_dispatch` now reads
  directly from handle payloads.

### Phase 5 — Delete legacy

- `WeightCacheManager` struct deleted.
- All `wcache_` references purged.
- Refactor becomes visible as a "big" diff only at this point, but the
  diff is purely mechanical (no logic changes).

**Rollback point: every PR.** Big-bang disaster is not possible.

## Error handling

Three failure classes, each with a clear diagnostic at the point of
occurrence:

### 1. Loader finds no `TensorKind` for a tensor name

- Action: assign `TensorKind::UNKNOWN`, emit `IMP_LOG_WARN` with the
  tensor name.
- Downstream: planner treats unknown kinds as `required_floor = FP16`
  (safe default).
- Not a crash — unknown tensors are sometimes handled by other code
  paths (e.g. metadata). The warn log is a signal: "add a kind entry".

### 2. Planner cannot satisfy `required_floor` under budget

- Action: hard fail. Model load returns `ImpError::InsufficientVRAM`
  with a message naming the offending tensor kind and the budget gap.
- Rationale: better to fail at load than silently degrade and generate
  incoherent output.

### 3. `gemm_dispatch` receives handle in unknown or `Undefined` tier

- Action: `IMP_LOG_FATAL` + abort.
- This is a programmer error (handle not populated by `PlanExecutor`),
  not a recoverable runtime condition.

### Removed code path: on-the-fly dequant-scratch fallback

The existing "if no cache entry, dequantize to FP16 scratch on the fly"
fallback is **removed entirely**. If a consumer hits a handle without a
populated payload, that is a planning bug or a missing TensorKind entry —
both should fail loud, not be silently papered over with a slow fallback
that also tends to perturb numerics.

## Testing strategy

### 1. Unit: `TensorKindTable` completeness

- `static_assert(std::size(kKindTable) == (int)TensorKind::_COUNT)`.
- Compile-time assertion: `required_floor` is a member of `supported`
  mask for every kind.
- GTest: iterate every tensor in each test-fixture GGUF; assert
  `kind != UNKNOWN`.

### 2. Unit: `StoragePlanner` determinism and constraints

- Given a synthetic model with known tensor sizes + budget: expected
  `StoragePlan` is ground truth, byte-exact comparison.
- Edge cases:
  - budget < sum of `required_floor` → plan returns failure
  - budget >>> required → all tensors on maximum supported tier
  - `dual_path_attn_fp8_ffn_nvfp4` hint → attention on FP8, FFN on NVFP4

### 3. Unit: `gemm_dispatch` correctness per tier

- For each `StorageTier`: dispatched call produces byte-identical output
  to a direct call of the underlying GEMM function.
- Existing `test_fp8_gemm.cu`, `test_nvfp4_quant.cu`, etc., remain.
  New `test_weight_dispatch.cu` covers only the switch logic.

### 4. Integration: Registry+Planner+Executor round-trip

- Load Qwen3-4B → planner runs → executor fills handles → generate 64
  tokens.
- Compare with pre-refactor build: same seed, same prompt → logits
  byte-identical (`cudaMemcpy` + `memcmp`).
- Any divergence → regression, immediate PR stop.

### 5. End-to-end (merge gate, every PR)

- Degeneration test on: Qwen3-4B Q8_0, Gemma-4-26B-A4B Q5_K_M,
  Qwen3.5-4B-GDN Q8_0, Qwen3-Coder-30B-A3B NVFP4.
- Benchmark parity: decode tok/s and prefill tok/s within ±2% of the
  pre-refactor baseline. (Prefill is volatile due to cuBLAS autotuning;
  decode is the hard gate.)

### 6. Regression test (new, added in Phase 4)

- `test_planner_preserves_fp16_only_tensors.cu`:
  construct a synthetic 2-layer GDN model, force NVFP4-only mode via
  hint, verify `registry.handle(ssm_in).primary_tier == FP16` and
  `registry.handle(wq).primary_tier == NVFP4`.
- This is the test that would have caught the original `d0e9b03` bug.

## Expected outcomes

**Primary:**

- "Global cache drop removes FP16 tensors that specific consumers need"
  is structurally impossible post-refactor.
- Cache dispatch logic lives in one file (`weight_dispatch.cu`) instead
  of 158 call sites across 7 files.

**Secondary:**

- Adding a new model architecture requires, at minimum, new `TensorKind`
  entries + a loader name-match clause. No consumer-side changes unless
  truly new ops are needed.
- Debugging new-model precision issues does not have to control for
  "did the cache get silently dropped?" as a hidden variable.

**What this refactor does NOT fix:**

- Qwen 3.6 GDN per-element drift. Unrelated precision bug in the scan.
  This refactor only eliminates the dequant-scratch fallback as a
  contaminating factor in that debug effort.

## Open questions (for implementation plan)

- Exact `TensorID` assignment scheme — dense integers during load vs.
  derived-from-(layer_idx, role) encoding.
- Where `WeightHandle[]` is allocated — inside `Model`, or a separate
  `WeightRegistry` owned by `GraphExecutor`.
- Whether the `TierMask` bitmask representation is worth a proper type
  or can stay as `uint32_t`.

These are implementation details to be resolved in the
writing-plans phase.
