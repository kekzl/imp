# QuantPipeline — first component of the GraphExecutor split (D2)

Date: 2026-06-08

## Problem

`GraphExecutor` (declared in `src/exec/executor.h`, ~1188-line header) is a
god-object: ~53 methods + ~69 data members + 10 supporting structs in one class.
D2 of the structural-debt audit (`docs/audit/structural_debt_2026_06_08.md`)
calls for breaking it into testable components with clear interfaces.

The `.cu` *implementations* are already domain-partitioned
(`executor_attention.cu`, `executor_forward_moe*.cu`, `executor_pre_dequant.cu`,
the `pre_dequant_phase*.cu` set, …). What is monolithic is the **class + header**:
all members are visible to every TU (no encapsulation), no domain is testable in
isolation, and any header edit recompiles the world.

This spec covers the **first** extracted component — `QuantPipeline` — chosen as
the proof-of-concept because it has the cleanest boundary and the lowest hot-path
risk: it runs once at init and the forward path only *reads* the caches it builds.

## Goal & constraints

- Extract the init-time weight-quantization pipeline into a standalone
  `QuantPipeline` class with a single `build()` entry point.
- **Strictly behaviour-neutral.** The forward hot path must stay byte-identical
  and zero-overhead — no new indirection in decode/prefill.
- Gated like the prior refactors: coherence canary across dense / MoE / GGUF /
  gemma + native-NVFP4, plus `make verify-fast`.
- Establish the **pattern** (component class + `build()`-style interface + the
  canary gates) that later components (MoeRunner, Workspace, AttentionRunner, …)
  will follow. Those are out of scope here.

## Architecture

`QuantPipeline` is a **builder component**. `GraphExecutor` owns one
(`QuantPipeline quant_pipeline_;`) and invokes it once during init (today's
`pre_dequant_weights` call site in `init_kv_cache`). The 23 build methods become
`QuantPipeline::` methods; the existing `pre_dequant_phase*.cu` / `executor_pre_dequant.cu`
translation units stay as-is except for the class prefix change.

The key principle that keeps the hot path zero-overhead: **the long-lived caches
stay owned by `GraphExecutor`; `QuantPipeline` fills them by reference.**

### State split

Verified by member-access analysis of the pre-dequant TUs:

| Member | Refs | Decision |
|---|---|---|
| `wcache_` (WeightCaches) | 207 | **stays on GraphExecutor**, filled by ref — hot path reads it everywhere |
| `model_` | 64 | input — passed to `build()` |
| `qscratch_` (QuantScratch) | 37 | **stays**, filled by ref — hot path reads it |
| `vram_alloc_` | 29 | input — passed to `build()` |
| `registry_` (WeightRegistry) | 14 | **stays**, filled by ref — hot path reads it |
| `storage_plan_` (StoragePlan) | 11 | **moves into QuantPipeline** — build-only, no hot-path reader |
| `hints_` (PlanHints) | 2 | **stays**, filled by ref (read by `executor_workspace.cu` sizing) |

`QuantPipeline` owns only build-only transient state: `storage_plan_`, the
`Nvfp4DecodeContext`, and any build scratch. Everything the forward path consumes
remains a `GraphExecutor` member, read identically to today.

### Methods that move (23)

`pre_dequant_weights` (entry) + `apply_arch_rules_` + `pre_dequant_phase0_promote_nvfp4_sidecars_`
+ `pre_dequant_phase0b_register_cutlass_nvfp4_` + `pre_dequant_phase1_fp16_cache_`
+ `pre_dequant_phase2_fp8_cache_` + `pre_dequant_phase3_nvfp4_decode_`
+ `pre_dequant_phase3c_standalone_mxfp4_` + `pre_dequant_phase4_tensor_registry_`
+ `pre_dequant_phase4b_drop_redundant_sources_` + the 10 `nvfp4_decode_*` helpers
+ `gpt_oss_convert_moe_experts_` + `cache_moe_native_nvfp4_`.

## Interface

```cpp
// src/exec/quant_pipeline.h
class QuantPipeline {
public:
    // Runs the full init-time quantization pipeline once. Populates the four
    // long-lived caches (owned by the caller) from the model's weights; owns the
    // transient StoragePlan + decode context internally.
    void build(const Model& model, const RuntimeConfig& rcfg, VRAMAllocator& alloc,
               const VRAMBudget& budget, cudaStream_t stream,
               WeightCaches& wcache, QuantScratch& qscratch,
               WeightRegistry& registry, PlanHints& hints);

private:
    StoragePlan storage_plan_;
    // + the 23 phase/helper methods (moved verbatim), now reading their inputs
    //   from build() params / members instead of GraphExecutor members.
};
```

`GraphExecutor::init_kv_cache` replaces its `pre_dequant_weights(stream, budget)`
call with `quant_pipeline_.build(*model_, *runtime_config_, *vram_alloc_, budget,
stream, wcache_, qscratch_, registry_, hints_)`.

## Open detail (resolved during implementation)

- The 2 `moe_` (MoEWorkspace) references inside the build methods: confirm whether
  they are genuine members (then `moe_` joins the `build()` params) or locals.
  Small; does not change the architecture.

## Data flow

Init: `GraphExecutor::init` → `init_kv_cache` → `quant_pipeline_.build(...)`
→ phases run, filling `wcache_`/`qscratch_`/`registry_`/`hints_` → control returns,
GraphExecutor proceeds. Forward: unchanged — reads `wcache_`/`qscratch_`/`registry_`
directly.

## Error handling

Unchanged. The phases throw on internal errors (translated to `ImpError` at the
API boundary per the project convention); `build()` propagates exceptions the same
way the current `pre_dequant_weights` does. No status-return conversion.

## Testing

- **New**: a standalone `QuantPipelineTest` that constructs a `QuantPipeline`,
  builds against a small loaded model with empty caches, and asserts the caches
  are populated (entry counts / tiers) — the first component testable without the
  full forward machinery. (If a bare-model fixture is impractical in unit scope,
  fall back to asserting via the existing GPU e2e path; note it in the plan.)
- **Canary (behaviour-neutral gate)**: coherence across Qwen3-8B Q8_0 (dense
  GGUF), Qwen3-30B-A3B-NVFP4 (MoE native — exercises `cache_moe_native_nvfp4_`),
  Nemotron-3-Nano-30B (Mamba2/MoE hybrid), gemma-3-12b (dense gemma). Each must
  stay coherent (Paris-class probe) with no `CUDA error / falling back / NaN` and
  no decode-speed drop (the native MoE cache must still fire — same count of
  `NVFP4 MoE native: data-borrow decode cache` log lines).
- `make verify-fast` gtest filter green.

## Out of scope

- Any other component (MoeRunner, Workspace, AttentionRunner, ExpertLRUCache,
  …). This PR only proves the pattern with QuantPipeline.
- Moving `wcache_`/`qscratch_`/`registry_` ownership off GraphExecutor (would add
  hot-path indirection — explicitly rejected).
- Splitting the supporting structs out of `executor.h` (a separate, smaller
  cleanup; not required for this extraction).

## Risks

- **A build method reads a GraphExecutor member not in the planned set.** Caught
  at compile time (the method moves to QuantPipeline and won't see the member);
  resolve by threading it through `build()` params or as a QuantPipeline member if
  build-only. The member-access analysis above bounds this to a short list.
- **Subtle reordering of cache population.** Mitigation: methods move verbatim
  (pure code-motion, like the D3 lambda extraction); the canary's native-MoE-cache
  count + coherence catches any divergence.
