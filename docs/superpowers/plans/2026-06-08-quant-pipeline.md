# QuantPipeline Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the init-time weight-quantization pipeline (23 `pre_dequant_*` / `nvfp4_decode_*` methods + the build-only `StoragePlan`) out of the `GraphExecutor` god-object into a standalone `QuantPipeline` builder class.

**Architecture:** `QuantPipeline` holds the build-only `StoragePlan` plus pointer members to the build inputs/outputs (model, allocator, runtime config, and the four long-lived caches that stay owned by `GraphExecutor`). `build()` sets those pointers and runs the existing phase pipeline, filling the caches by reference — so the forward hot path reads `wcache_`/`qscratch_`/`registry_` exactly as today (byte-identical, zero overhead). `GraphExecutor::pre_dequant_weights` becomes a one-line delegate, so the engine call site is untouched.

**Tech Stack:** C++20 / CUDA, Docker build (`make build`), GTest (`imp-tests`), the 4-arch coherence canary.

Spec: `docs/superpowers/specs/2026-06-08-quant-pipeline-design.md`. Branch: `feat/d2-quant-pipeline` (spec already committed).

---

## File structure

- **Create** `src/exec/quant_pipeline.h` — the `QuantPipeline` class declaration (build-only state + pointer members + the 23 moved method declarations + `build()`).
- **Modify** `src/exec/executor.h` — add `#include "exec/quant_pipeline.h"`, add member `QuantPipeline quant_pipeline_;`, remove the 23 moved method declarations + the `StoragePlan storage_plan_;` member, keep `pre_dequant_weights` (now a delegate).
- **Modify** the 7 pre-dequant TUs — change `GraphExecutor::` → `QuantPipeline::` for the moved methods, change value-cache member access `.` → `->` (now pointers), keep `storage_plan_` as a value member of `QuantPipeline`:
  `executor_pre_dequant.cu`, `pre_dequant_phase0_nvfp4_loader.cu`, `pre_dequant_phase1_fp16_cache.cu`, `pre_dequant_phase2_fp8_cache.cu`, `pre_dequant_phase3_nvfp4_decode.cu`, `pre_dequant_phase3c_mxfp4.cu`, `pre_dequant_phase4_tensor_registry.cu`.
- **Modify** `CMakeLists.txt` — no new TU (the phase TUs already listed); only if a new test file is added.
- **Create** `tests/test_quant_pipeline.cpp` — standalone `QuantPipelineTest` (Task 7).

The build inputs/outputs threaded into `QuantPipeline` (from the spec's member-access analysis, plus `moe_` found in phase4):
- inputs: `const Model* model_`, `VRAMAllocator* vram_alloc_`, `const RuntimeConfig* runtime_config_`, `const VRAMBudget* budget_`, `cudaStream_t stream_`
- outputs (filled, owned by GraphExecutor): `WeightCaches* wcache_`, `QuantScratch* qscratch_`, `WeightRegistry* registry_`, `PlanHints* hints_`, `MoEWorkspace* moe_`
- owned build-only: `StoragePlan storage_plan_`

---

### Task 1: Create the `QuantPipeline` header skeleton

**Files:**
- Create: `src/exec/quant_pipeline.h`
- Reference (copy decls from): `src/exec/executor.h:787-825` (the `pre_dequant_*` / `nvfp4_decode_*` declarations) and `src/exec/executor.h:489-516` (`Nvfp4DecodeContext`).

- [ ] **Step 1: Read the exact current declarations to copy**

Run:
```bash
sed -n '786,825p' src/exec/executor.h
grep -nE "GraphExecutor::(pre_dequant|nvfp4_decode|gpt_oss_convert|apply_arch_rules|cache_moe_native)" src/exec/*.cu | sed 's/.*GraphExecutor:://;s/(.*//' | sort -u
```
Expected: the 23 method names + their signatures. Copy the signatures verbatim for Step 2.

- [ ] **Step 2: Write `quant_pipeline.h`**

```cpp
#pragma once

#include "core/quant.h"          // QType, etc. (match executor.h's includes for these types)
#include "exec/storage_planner.h" // StoragePlan, PlanHints
#include "exec/weight_registry.h" // WeightRegistry
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

class Model;
class VRAMAllocator;
struct RuntimeConfig;
struct VRAMBudget;
struct WeightCaches;
struct QuantScratch;
struct MoEWorkspace;

// Init-time weight-quantization pipeline, extracted from GraphExecutor (D2).
// Runs once via build(); fills the four long-lived caches (owned by the caller)
// and owns only the build-only StoragePlan + decode context. See
// docs/superpowers/specs/2026-06-08-quant-pipeline-design.md.
class QuantPipeline {
public:
    void build(const Model& model, const RuntimeConfig& rcfg, VRAMAllocator& alloc,
               const VRAMBudget& budget, cudaStream_t stream, WeightCaches& wcache,
               QuantScratch& qscratch, WeightRegistry& registry, PlanHints& hints,
               MoEWorkspace& moe);

private:
    // Build context (set at the top of build(); the phase methods read these
    // exactly as they read the same-named GraphExecutor members today).
    const Model* model_ = nullptr;
    VRAMAllocator* vram_alloc_ = nullptr;
    const RuntimeConfig* runtime_config_ = nullptr;
    const VRAMBudget* budget_ = nullptr;
    cudaStream_t stream_ = nullptr;
    WeightCaches* wcache_ = nullptr;
    QuantScratch* qscratch_ = nullptr;
    WeightRegistry* registry_ = nullptr;
    PlanHints* hints_ = nullptr;
    MoEWorkspace* moe_ = nullptr;

    // Owned build-only state.
    StoragePlan storage_plan_;

    // --- moved phase / helper declarations (paste verbatim from executor.h) ---
    // void pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget);  // see note
    // ... the other 22 declarations, with the SAME signatures ...
};

}  // namespace imp
```

Notes for Step 2:
- Paste the 22 inner phase/helper declarations verbatim from `executor.h` (the `pre_dequant_phase*_`, `nvfp4_decode_*`, `gpt_oss_convert_moe_experts_`, `apply_arch_rules_`, `cache_moe_native_nvfp4_` decls).
- The public entry inside QuantPipeline is `build()`. The current `pre_dequant_weights(stream, budget)` body becomes the **private** `run_(...)` of QuantPipeline OR `build()` simply contains that body. Simplest: rename the moved `pre_dequant_weights` body to `build()`'s body (the body already calls the phases). Keep the phase decls private.
- Fix `#include`s by copying the exact ones `executor.h` uses for `StoragePlan`, `PlanHints`, `WeightRegistry`, `Nvfp4DecodeContext`. If `Nvfp4DecodeContext` / `MoeFfnContext` live in `executor.h`, move `Nvfp4DecodeContext` into `quant_pipeline.h` (it's "Per-call state for pre_dequant_phase3" — build-only) and have `executor.h` keep compiling (it no longer needs it).

- [ ] **Step 3: Commit the skeleton (won't fully wire yet)**

```bash
git add src/exec/quant_pipeline.h
git commit -m "feat(quant): QuantPipeline header skeleton (D2 component 1)"
```

---

### Task 2: Move the phase method DEFINITIONS to `QuantPipeline`

**Files:**
- Modify: `src/exec/executor_pre_dequant.cu`, `pre_dequant_phase0_nvfp4_loader.cu`, `pre_dequant_phase1_fp16_cache.cu`, `pre_dequant_phase2_fp8_cache.cu`, `pre_dequant_phase3_nvfp4_decode.cu`, `pre_dequant_phase3c_mxfp4.cu`, `pre_dequant_phase4_tensor_registry.cu`

- [ ] **Step 1: Back up the TUs**

```bash
mkdir -p /tmp/qp_bak && cp src/exec/executor_pre_dequant.cu src/exec/pre_dequant_phase*.cu /tmp/qp_bak/
```

- [ ] **Step 2: Add the include + reclass the method definitions**

In each of the 7 TUs, add `#include "exec/quant_pipeline.h"` (after the existing `#include "exec/executor.h"`), then reclass the moved methods:

```bash
for f in src/exec/executor_pre_dequant.cu src/exec/pre_dequant_phase0_nvfp4_loader.cu \
         src/exec/pre_dequant_phase1_fp16_cache.cu src/exec/pre_dequant_phase2_fp8_cache.cu \
         src/exec/pre_dequant_phase3_nvfp4_decode.cu src/exec/pre_dequant_phase3c_mxfp4.cu \
         src/exec/pre_dequant_phase4_tensor_registry.cu; do
  grep -q 'exec/quant_pipeline.h' "$f" || sed -i '0,/#include "exec\/executor.h"/s//#include "exec\/executor.h"\n#include "exec\/quant_pipeline.h"/' "$f"
  # reclass the moved methods (the 23 names) from GraphExecutor:: to QuantPipeline::
  perl -0pi -e 's/\bGraphExecutor::(pre_dequant_weights|pre_dequant_phase\w+|nvfp4_decode_\w+|gpt_oss_convert_moe_experts_|apply_arch_rules_|cache_moe_native_nvfp4_)\b/QuantPipeline::$1/g' "$f"
done
```

- [ ] **Step 3: Rename the `pre_dequant_weights` definition to `build` and give it the build() signature**

In `executor_pre_dequant.cu`, change:
```cpp
void QuantPipeline::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
```
to the `build()` signature, and stash the context pointers at the top:
```cpp
void QuantPipeline::build(const Model& model, const RuntimeConfig& rcfg, VRAMAllocator& alloc,
                          const VRAMBudget& budget, cudaStream_t stream, WeightCaches& wcache,
                          QuantScratch& qscratch, WeightRegistry& registry, PlanHints& hints,
                          MoEWorkspace& moe) {
    model_ = &model; runtime_config_ = &rcfg; vram_alloc_ = &alloc; budget_ = &budget;
    stream_ = stream; wcache_ = &wcache; qscratch_ = &qscratch; registry_ = &registry;
    hints_ = &hints; moe_ = &moe;
    // ... the rest of the original pre_dequant_weights body unchanged ...
```
If the original body used the parameter names `stream` / `budget` directly, keep doing so (they're still params). The body's internal phase calls (`pre_dequant_phase1_fp16_cache_(...)` etc.) are member calls — unchanged.

- [ ] **Step 4: Convert value-cache member access to pointer access**

The four caches + `moe_` are now POINTER members; `storage_plan_` stays a value member (owned). Apply per TU:
```bash
for f in src/exec/executor_pre_dequant.cu src/exec/pre_dequant_phase*.cu; do
  perl -0pi -e 's/\bwcache_\./wcache_->/g; s/\bqscratch_\./qscratch_->/g; s/\bregistry_\./registry_->/g; s/\bhints_\./hints_->/g; s/\bmoe_\./moe_->/g' "$f"
  # address-of / by-value passes of the now-pointer caches:
  perl -0pi -e 's/&wcache_\b/wcache_/g; s/&qscratch_\b/qscratch_/g; s/&registry_\b/registry_/g; s/&moe_\b/moe_/g' "$f"
done
```
`model_->` / `vram_alloc_->` are unchanged (already pointers). `storage_plan_.` is unchanged (still a value member, now on QuantPipeline). Do NOT touch `storage_plan_`.

- [ ] **Step 5: Commit the move (will not build until Task 3 wires GraphExecutor)**

```bash
git add src/exec/*.cu
git commit -m "refactor(quant): reclass pre_dequant phase methods onto QuantPipeline"
```

---

### Task 3: Wire `GraphExecutor` to own + delegate to `QuantPipeline`

**Files:**
- Modify: `src/exec/executor.h`

- [ ] **Step 1: Include + member**

In `executor.h`, add near the top includes: `#include "exec/quant_pipeline.h"`. In the private member section (near `WeightCaches wcache_;` at line ~953) add:
```cpp
    QuantPipeline quant_pipeline_;
```

- [ ] **Step 2: Remove the moved declarations + the build-only member**

Delete from `executor.h`:
- the 22 inner declarations (`pre_dequant_phase*_`, `nvfp4_decode_*`, `gpt_oss_convert_moe_experts_`, `apply_arch_rules_`, `cache_moe_native_nvfp4_`) — they now live on `QuantPipeline`.
- the `StoragePlan storage_plan_;` member (line ~963) — now owned by `QuantPipeline`.
- `Nvfp4DecodeContext` struct (lines ~489-516) if it was moved into `quant_pipeline.h` in Task 1.

KEEP `void pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget);` declared on GraphExecutor (it becomes a delegate).

- [ ] **Step 3: Make `pre_dequant_weights` a delegate**

The original `GraphExecutor::pre_dequant_weights` body now lives in `QuantPipeline::build`. Add a NEW tiny `GraphExecutor::pre_dequant_weights` definition (put it in `executor_pre_dequant.cu`, replacing the now-renamed one):
```cpp
void GraphExecutor::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
    quant_pipeline_.build(*model_, *runtime_config_, *vram_alloc_, budget, stream,
                          wcache_, qscratch_, registry_, hints_, moe_);
}
```
(`wcache_`/`qscratch_`/`registry_`/`hints_`/`moe_` here are the GraphExecutor value members — passed by reference into `build`.)

- [ ] **Step 4: Commit**

```bash
git add src/exec/executor.h src/exec/executor_pre_dequant.cu
git commit -m "refactor(exec): GraphExecutor owns + delegates to QuantPipeline"
```

---

### Task 4: Build and fix the compile errors

- [ ] **Step 1: Build**

Run: `make build 2>&1 | grep -iE "error:" | head -40`
Expected initially: a short list of errors — most likely (a) a build method that read a GraphExecutor member NOT in the planned set (model_/alloc/rcfg/budget/stream/wcache/qscratch/registry/hints/moe/storage_plan), or (b) a missing include in `quant_pipeline.h`, or (c) a residual `wcache_.` / `&wcache_` the perl missed.

- [ ] **Step 2: Resolve each error by category**

- **Missing GraphExecutor member referenced in a moved method**: add it to the build context — a pointer member on `QuantPipeline` + a `build()` param + set it in Step-3-of-Task-2, and pass it from the delegate. Re-grep to confirm it's genuinely build-time (no hot-path writer).
- **`Nvfp4DecodeContext` / type not found**: add the include or move the type into `quant_pipeline.h`.
- **Residual value-access**: fix the specific `.`/`&` site by hand.

Iterate `make build` until clean. Expected final: `make build` succeeds.

- [ ] **Step 3: Commit the fixes**

```bash
git add -A && git commit -m "fix(quant): thread remaining build-context members + includes"
```

---

### Task 5: Behaviour-neutral canary (the gate)

- [ ] **Step 1: 4-arch coherence + native-MoE-cache count**

```bash
for M in "Qwen3-8B-Q8_0.gguf|dense" "Qwen3-30B-A3B-NVFP4-Modelopt|moe-native" \
         "Nemotron-3-Nano-30B-A3B-NVFP4|hybrid" "gemma-3-12b-it-Q4_K_M.gguf|gemma3"; do
  MODEL="${M%%|*}";
  docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
    imp-cli --model "/models/$MODEL" --prompt "What is the capital of France? One word." \
    --max-tokens 200 --temperature 0 --seed 42 >/tmp/qp.out 2>&1
  echo "== $MODEL =="
  grep -aoiE "paris" /tmp/qp.out | head -1
  grep -acE "NVFP4 MoE native: data-borrow decode cache" /tmp/qp.out
  grep -aiE "CUDA error|falling back|\bNaN\b|legacy.*fallback" /tmp/qp.out | head -1
done
```
Expected: each model prints `Paris`, no `CUDA error/falling back/NaN`, and the MoE models print the SAME native-cache count as `main` (Qwen3-30B 144, Nemotron 46). If the count differs or a model degenerates, the extraction reordered something — bisect against the Task-2/3 commits.

- [ ] **Step 2: `verify-fast`**

Run: `IMP_VERIFY_SKIP_BUILD=1 make verify-fast 2>&1 | grep -E "PASS|FAIL|OK ==="`
Expected: `PASS fast gtest filter`, `=== verify fast: OK ===`.

- [ ] **Step 3: Commit (no-op if clean) / proceed**

No code change here; this is the gate. If anything failed, fix and re-run before continuing.

---

### Task 6: Confirm the hot path is byte-identical

- [ ] **Step 1: Diff the forward TUs vs main**

Run: `git diff main -- src/exec/executor_attention.cu src/exec/executor_ffn.cu src/exec/executor_forward.cu src/exec/executor_forward_moe.cu src/exec/executor_gemm_dispatch.cu | wc -l`
Expected: `0` — the hot-path TUs are untouched (they still read `wcache_`/`qscratch_`/`registry_` as GraphExecutor members). If non-zero, something leaked into the hot path; investigate before proceeding.

---

### Task 7: Add the standalone `QuantPipelineTest`

**Files:**
- Create: `tests/test_quant_pipeline.cpp`
- Modify: `CMakeLists.txt` (add the test source to the GPU test bundle, next to the other `tests/test_*.cu`/`.cpp` entries)

- [ ] **Step 1: Write the test**

A minimal test that loads a small model, runs the pipeline via the public engine path, and asserts the caches got populated. Since `QuantPipeline::build` needs a loaded `Model` + allocator (GPU), the pragmatic first test drives it through the engine and asserts an observable post-build invariant:

```cpp
#include <gtest/gtest.h>
#include "api/imp_c.h"   // or the engine header used by other GPU tests
#include "test_models.h" // existing registry of local test model paths

// QuantPipeline runs inside engine init; assert it produced a usable decode
// cache by checking the model decodes coherently (the cache is what makes the
// native NVFP4 / dequant decode path fire). This is the component's
// post-condition observed end-to-end until a bare-model unit fixture exists.
TEST(QuantPipelineTest, BuildPopulatesDecodeCachesAndDecodes) {
    // Mirror the setup of tests/test_degeneration.cpp (same C API + default model).
    // ... load default model, generate ~16 tokens greedy, assert non-degenerate ...
}
```

Match the exact harness of `tests/test_degeneration.cpp` (same include set, model env var, generation helper). Keep the assertion simple: ≥10 coherent tokens, no repetition. (A true bare-`QuantPipeline` unit test needs a model fixture without the full engine; note it as a follow-up if the fixture doesn't exist yet — do not block this PR on it.)

- [ ] **Step 2: Register + build + run**

```bash
# add tests/test_quant_pipeline.cpp to the test target list in CMakeLists.txt
make build 2>&1 | tail -2
docker run --rm --gpus all -v /home/kekz/models:/models imp:test imp-tests --gtest_filter="QuantPipelineTest.*" 2>&1 | tail -8
```
Expected: `[  PASSED  ] 1 test.`

- [ ] **Step 3: Commit**

```bash
git add tests/test_quant_pipeline.cpp CMakeLists.txt
git commit -m "test(quant): QuantPipeline build post-condition (decode coherence)"
```

---

### Task 8: Push + PR

- [ ] **Step 1: Push + open PR vs main**

```bash
git push -u origin feat/d2-quant-pipeline
gh pr create --base main --head feat/d2-quant-pipeline \
  --title "refactor(exec): extract QuantPipeline — first GraphExecutor component (D2)" \
  --body "<summary: what moved, byte-identical hot path (Task 6 diff=0), 4-arch canary + native-MoE-cache count + verify-fast green; establishes the component pattern for the rest of D2>"
```

- [ ] **Step 2: Watch CI, merge when green** (Build required check; Test skips — GPU local-only; the canary in Task 5 is the behaviour gate run locally).

---

## Notes / gotchas (from prior D-work on this repo)

- Build is Docker-only (`make build`); `build/` is root-owned. GPU canary runs need `-v /home/kekz/models:/models` (the repo `models/` are symlinks).
- `pre_dequant_phase3_nvfp4_decode.cu` already holds the extracted `cache_moe_native_nvfp4_` method (from PR #627) — it reclasses to `QuantPipeline::cache_moe_native_nvfp4_` like the rest.
- Line-surgery lesson: keep `/tmp/*.bak` copies before perl edits; never use `perl -0pi` with an embedded `do{<>}` (it empties the file) — the per-TU perl commands here are plain substitutions, which are safe.
- Qwen3-30B-A3B-NVFP4 is MoE-routing nondeterministic at temp=0; judge it by the native-cache COUNT + coherence, not exact tokens.
