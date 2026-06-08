# executor.h Header Decongestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the ~10 supporting structs out of the 1075-line `src/exec/executor.h` into focused headers, declaration-only, so `executor.h` shrinks to ~600 lines and each type has a home.

**Architecture:** Four new `exec/*.h` headers + moving `VRAMBudget` into the existing `runtime/vram_budget.h`. `executor.h` re-includes the new headers, so every consumer (which sees these types via `executor.h`) compiles unchanged. Pure type-declaration relocation — no logic, no member reordering, no hot-path code touched.

**Tech Stack:** C++20/CUDA, Docker build (`make build`), GTest (`imp-tests`).

Spec: `docs/superpowers/specs/2026-06-08-header-decongestion-design.md`. Branch: `refactor/d2-header-decongestion` (spec committed).

---

## File structure

Struct line ranges currently in `executor.h` (verify with `grep -nE "^struct |^enum |^class GraphExecutor" src/exec/executor.h` before cutting — line numbers shift as you remove blocks, so **work bottom-up or re-grep each task**):
- ExpertLRU family (`ExpertCacheKey`, `ExpertCacheKeyHash`, `ExpertProj`+`kExpertProjCount`, `PerLayerLRU`, `PerLayerAccessRing`, `ExpertLRUCache`): ~39-235
- `VRAMBudget`: ~236-247
- `InferenceState`: ~248-390
- `FP8CacheEntry`: ~391-403
- `WeightCaches` (incl. `Q4kImmaCacheEntry`): ~404-495
- `MoeFfnContext`: ~496-544
- `class GraphExecutor`: ~546+

New files: `src/exec/expert_cache.h`, `src/exec/weight_caches.h`, `src/exec/inference_state.h`, `src/exec/moe_ffn_context.h`. Modified: `src/exec/executor.h`, `src/runtime/vram_budget.h`, `src/exec/expert_cache.cu`.

**General procedure for each "move a struct block" task** (the tasks below specialize it):
1. `cp src/exec/executor.h /tmp/exec_bak.h` (safety, once at start of the whole effort is enough).
2. Re-grep the current line range of the block (numbers shift between tasks).
3. Create the new header: `#pragma once`, namespace `imp`, the includes listed in the task, then paste the struct block **verbatim** (cut from executor.h).
4. Delete that block from `executor.h`.
5. Add `#include "exec/<new>.h"` to `executor.h` in the include section (lines 3-26), grouped with the other `exec/*` includes.
6. `make build 2>&1 | grep -iE "error:" | head -30` → fix missing includes/forward-decls in the new header (add from executor.h's include set) until clean.
7. Commit.

`executor.h`'s current include set (copy from here as needed): `model/model.h`, `memory/kv_cache.h`, `memory/ssm_state.h`, `memory/layer_offload.h`, `compute/moe_routing.h`, `compute/json_constrain.h`, `compute/schema_constrain.h`, `quant/nvfp4_quant.h`, `compute/gemm_cutlass_sm120.h`, `compute/gemm_cutlass_mxfp4_sm120.h`, `core/tensor.h`, `exec/weight_handle.h`, `exec/moe_workspace.h`, `exec/quant_scratch.h`, `exec/quant_pipeline.h`, `runtime/storage_planner.h`, `runtime/config.h`, `<cuda_runtime.h>`, `<cuda_fp16.h>`, `<vector>`, `<unordered_map>`, `<utility>`, `<list>`.

---

### Task 1: `expert_cache.h` — the ExpertLRU family

**Files:** Create `src/exec/expert_cache.h`; Modify `src/exec/executor.h`, `src/exec/expert_cache.cu`.

- [ ] **Step 1: Create the header**

`src/exec/expert_cache.h`:
```cpp
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace imp {

class VRAMAllocator;

// <-- paste ExpertCacheKey, ExpertCacheKeyHash, ExpertProj, kExpertProjCount,
//     PerLayerLRU, PerLayerAccessRing, ExpertLRUCache (incl. Slot) here,
//     verbatim from executor.h lines ~39-235 -->

}  // namespace imp
```

- [ ] **Step 2: Cut the block from `executor.h`** (lines ~39-235) and add `#include "exec/expert_cache.h"` to the include section.

- [ ] **Step 3: Point `expert_cache.cu` at the new header**

In `src/exec/expert_cache.cu`, the existing `#include "executor.h"` already pulls it transitively, but add a direct `#include "exec/expert_cache.h"` after the `#include "executor.h"` line for clarity (the `.cu` defines `ExpertLRUCache::` methods).

- [ ] **Step 4: Build**

Run: `make build 2>&1 | grep -iE "error:" | head -30`
Expected: clean. If `VRAMAllocator` incomplete-type errors appear (a method uses it by value), include `memory/vram_allocator.h` in the header instead of the forward declaration.

- [ ] **Step 5: Commit**

```bash
git add src/exec/expert_cache.h src/exec/executor.h src/exec/expert_cache.cu
git commit -m "refactor(exec): move ExpertLRUCache family to exec/expert_cache.h"
```

---

### Task 2: `weight_caches.h` — FP8CacheEntry + WeightCaches

**Files:** Create `src/exec/weight_caches.h`; Modify `src/exec/executor.h`.

- [ ] **Step 1: Create the header**

`src/exec/weight_caches.h`:
```cpp
#pragma once

#include "core/tensor.h"
#include "quant/nvfp4_quant.h"               // NvFP4QuantResult, NvFP4MoEQuantResult
#include "compute/gemm_cutlass_sm120.h"      // CutlassNvFP4Weight
#include "compute/gemm_cutlass_mxfp4_sm120.h"// CutlassMxFP4Weight
#include <cuda_fp16.h>
#include <unordered_map>
#include <cstddef>

namespace imp {

// <-- paste FP8CacheEntry then WeightCaches (incl. inner Q4kImmaCacheEntry),
//     verbatim from executor.h lines ~391-495 -->

}  // namespace imp
```

- [ ] **Step 2: Cut FP8CacheEntry + WeightCaches from `executor.h`** (re-grep ranges first), add `#include "exec/weight_caches.h"`.

- [ ] **Step 3: Build** — `make build 2>&1 | grep -iE "error:" | head -30`; add any missing include from executor.h's set. Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/exec/weight_caches.h src/exec/executor.h
git commit -m "refactor(exec): move WeightCaches + FP8CacheEntry to exec/weight_caches.h"
```

---

### Task 3: `inference_state.h` — InferenceState

**Files:** Create `src/exec/inference_state.h`; Modify `src/exec/executor.h`.

- [ ] **Step 1: Create the header**

`src/exec/inference_state.h`:
```cpp
#pragma once

#include "core/tensor.h"
#include "memory/kv_cache.h"     // KVCache
#include "memory/ssm_state.h"    // SSMState
#include "memory/gdn_state.h"    // GDNState
#include <cuda_runtime.h>
#include <vector>

namespace imp {

// <-- paste InferenceState verbatim from executor.h lines ~248-390 -->

}  // namespace imp
```

- [ ] **Step 2: Cut InferenceState from `executor.h`** (re-grep range), add `#include "exec/inference_state.h"`.

- [ ] **Step 3: Build** — fix missing includes (e.g. if it references `MoeRoutingResult`, add `compute/moe_routing.h`). Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/exec/inference_state.h src/exec/executor.h
git commit -m "refactor(exec): move InferenceState to exec/inference_state.h"
```

---

### Task 4: `moe_ffn_context.h` — MoeFfnContext

**Files:** Create `src/exec/moe_ffn_context.h`; Modify `src/exec/executor.h`.

- [ ] **Step 1: Create the header**

`src/exec/moe_ffn_context.h`:
```cpp
#pragma once

#include "core/tensor.h"
#include "compute/moe_routing.h"  // MoeRoutingResult
#include <cstddef>

namespace imp {

// <-- paste MoeFfnContext verbatim from executor.h lines ~496-544 -->

}  // namespace imp
```

- [ ] **Step 2: Cut MoeFfnContext from `executor.h`** (re-grep range), add `#include "exec/moe_ffn_context.h"`.

- [ ] **Step 3: Build** — fix missing includes. Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/exec/moe_ffn_context.h src/exec/executor.h
git commit -m "refactor(exec): move MoeFfnContext to exec/moe_ffn_context.h"
```

---

### Task 5: Move `VRAMBudget` into `runtime/vram_budget.h` (drop the god-object include)

**Files:** Modify `src/runtime/vram_budget.h`, `src/exec/executor.h`.

- [ ] **Step 1: Inspect `runtime/vram_budget.h`**

Run: `sed -n '1,20p' src/runtime/vram_budget.h`
Note: it currently has `#include "exec/executor.h"  // VRAMBudget`.

- [ ] **Step 2: Move the struct**

Cut the `VRAMBudget` struct (re-grep its range in `executor.h`, ~236-247 originally — shifted by earlier tasks) and paste it into `src/runtime/vram_budget.h` inside `namespace imp`, ABOVE the `compute_vram_budget` declaration. Then in `runtime/vram_budget.h` REPLACE `#include "exec/executor.h"` with nothing (the struct is now local); keep `model/model.h`, `memory/kv_cache.h`, `<cstddef>`.

- [ ] **Step 3: Make `executor.h` include it**

Remove the `VRAMBudget` block from `executor.h`; add `#include "runtime/vram_budget.h"` to its include section.

- [ ] **Step 4: Build**

Run: `make build 2>&1 | grep -iE "error:|cyclic|redefinition" | head -30`
Expected: clean. A `redefinition of VRAMBudget` error means a stale copy remains — ensure it's deleted from `executor.h`. A cycle error means something `runtime/vram_budget.h` includes pulls `executor.h` back — it shouldn't (it now includes only model/kv_cache/cstddef).

- [ ] **Step 5: Commit**

```bash
git add src/runtime/vram_budget.h src/exec/executor.h
git commit -m "refactor(runtime): move VRAMBudget into vram_budget.h, drop executor.h include"
```

---

### Task 6: Gate — verify-fast + coherence smoke + line-count check

- [ ] **Step 1: Confirm the shrink**

Run: `wc -l src/exec/executor.h`
Expected: ~600 lines (down from ~1075).

- [ ] **Step 2: verify-fast**

Run: `IMP_VERIFY_SKIP_BUILD=1 make verify-fast 2>&1 | grep -E "PASS|FAIL|OK ==="`
Expected: `PASS fast gtest filter`, `=== verify fast: OK ===`.

- [ ] **Step 3: Coherence smoke (dense + MoE)**

```bash
for M in Qwen3-8B-Q8_0.gguf Qwen3-30B-A3B-NVFP4-Modelopt; do
  echo "== $M =="
  docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
    imp-cli --model "/models/$M" --prompt "What is the capital of France? One word." \
    --max-tokens 200 --temperature 0 --seed 42 2>&1 | grep -aoiE "paris|CUDA error|\bNaN\b" | head -1
done
```
Expected: each prints `Paris`, no error/NaN. (Logic is unchanged — this is a backstop.)

- [ ] **Step 4: No further commit needed** (the gate is verification). Proceed to push + PR (handled by the controller, not this plan).

---

## Self-review notes (controller)

- **Spec coverage:** Tasks 1-4 = the four new headers; Task 5 = VRAMBudget + cycle removal; Task 6 = the gate (build/verify-fast/smoke). All spec sections covered.
- **No new behaviour:** every struct moves verbatim; the only executable-code change is include directives. The forward TUs are not edited (no `diff=0` task needed — there's no hot-path edit to check).
- **Type consistency:** header names (`expert_cache.h`, `weight_caches.h`, `inference_state.h`, `moe_ffn_context.h`) used consistently across tasks and `executor.h` includes.
- **Gotcha:** line numbers shift as blocks are removed — every task says re-grep the range. Work one struct at a time, build between, so a break localizes to one header.
