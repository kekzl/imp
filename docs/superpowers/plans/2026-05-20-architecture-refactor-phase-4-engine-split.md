# Architecture Refactor Phase 4 — Engine.cpp Zerteilen

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/runtime/engine.cpp` (3112 LOC) shrinks to ≤800 LOC orchestrator. Methods extracted into per-subsystem translation units, matching the Phase 3 pattern (functions stay `Engine::*` methods; only the TU location changes).

**Architecture:** Sequential extraction of `Engine::*` method groups into their own `.cpp` files. The `Engine` class header (`src/runtime/engine.h`) is **unchanged** — all extractions are pure TU splits, not class redesigns. Each subsystem is one new file containing a coherent group of `Engine::method()` definitions. The `engine.cpp` orchestrator retains constructor/destructor, top-level `init()`, the public C-API-facing methods (`generate`, `add_request`), accessors, and the thin wrappers (MTP, vision).

**Tech Stack:** C++20, CUDA 13.2, CMake, Docker build (`make build`), GTest suite (`make verify-fast`).

---

## Reference: Source spec

`docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md` §3 Phase 4.

## Spec deviation noted upfront

The spec proposed turning each subsystem into a **named class with constructor injection** (`InitResolver`, `WeightUploadOrchestrator`, `KVCacheInitializer`, …). This is a much deeper refactor than the goal of "≤800 LOC orchestrator" requires:

- Current methods mutate `Engine`'s private state directly (`config_.kv_dtype = …`, `wcache_.…`, `kv_manager_ = …`). Pure-functional extraction would require designing `EngineState` structs that subsystems receive by reference, then merging back — a multi-week design with cross-subsystem state-passing decisions.
- The spec's stated goal — "Each subsystem becomes a named owner with constructor injection" — is a class-redesign property. The roadmap's actual measurable target is "engine.cpp shrinks from 3112 LOC to ≤800 LOC."

This plan deviates to **TU-split-only**: methods stay `Engine::method()`, declarations in `engine.h` stay untouched, but each subsystem's definitions move to its own `.cpp` file. Same pattern Phase 3 used for `executor_pre_dequant.cu` (functions stayed `GraphExecutor::`-methods). Achieves the LOC target and the legibility win without the multi-week class redesign.

The "named subsystem owners" goal is preserved as a future opportunistic refactor — once each subsystem's code is in its own TU, promoting to a separate class becomes mechanically easier and can be done one subsystem at a time.

## Reference: Pre-flight inventory

`src/runtime/engine.cpp` at plan-write time (3112 LOC):

| Section | Lines | Method(s) |
|---|---|---|
| File header + includes + anonymous-namespace helpers | 1-140 | (~140 LOC of includes + helpers) |
| MTP wrappers (delegate to MTP state) | 142-325 | `enable_mtp_spec_decode`, `mtp_prefill_prompt`, `mtp_accuracy_reset`, `mtp_draft_one` |
| Accessors / simple state | 326-373 | `prefill_stream`, `decode_stream`, `reset_ssm_state`, `reset_batch_pool_cache`, `invalidate_graphs`, `effective_free_vram` |
| Stop logic | 374-480 | `is_stop_token`, `track_think_state`, `should_stop` |
| Sampling helpers | 481-578 | `fill_sampling_params`, `upload_penalties`, `fill_recurrent_state` |
| Request lifecycle | 580-591 | `finish_request` |
| Vision wrappers | 593-599 | `set_image`, `set_image_from_memory`, `clear_image` |
| Init resolvers | 609-927 | `init_apply_debug_raw_overrides_`, `init_resolve_kv_dtype_policy_`, `init_resolve_ssm_dtype_`, `init_resolve_fp8_prefill_`, `init_resolve_quant_flags_`, `init_compute_max_seq_len_` (~320 LOC across 6 methods) |
| Top-level init | 929-974 | `init()` (~46 LOC orchestrator) |
| Weight upload | 975-1204 | `init_weights` (~230 LOC) |
| KV cache init | 1205-1459 | `init_kv_cache` (~255 LOC) |
| Workspaces | 1460-1536 | `init_features` (~77 LOC) |
| Banned-token list | 1537-1621 | `build_banned_token_list` (~85 LOC) |
| Warmup | 1622-1712 | `warmup()` (~91 LOC) |
| Scheduler group | 1713-2961 | `step`, `step_async_graph_resume`, `step_schedule`, `supports_chunked_prefill_`, `resolve_prefill_chunk_size_`, `step_prefill`, `prefill_allocate_kv_blocks_`, `prefill_upload_metadata_`, `step_prefill_one`, `step_decode`, `decode_build_inference_state_`, `step_decode_forward`, `step_decode_process_outputs` (~1250 LOC across 13 methods — the biggest piece) |
| Top-level driver | 2962-3112 | `generate`, `add_request` |

Engine.cpp also has a sibling: `src/runtime/engine_graph_decode.cpp` (~separate, smaller file; not modified by this phase).

---

## Task 1: Lift shared engine.cpp helpers to internal header

**Why first:** Like Phase 3 Task 1, extract any anonymous-namespace helpers + file-scope statics that subsequent per-subsystem TUs will need.

**Files:**
- Create: `src/runtime/engine_internal.h` — declarations + `inline` definitions of helpers shared across `engine_*.cpp` TUs
- Modify: `src/runtime/engine.cpp` — replace anonymous block + statics with `#include "runtime/engine_internal.h"` + `using`-decls

- [ ] **Step 1: Inspect the current anonymous-namespace block + file-scope statics**

```bash
sed -n '1,140p' src/runtime/engine.cpp
```

Note every helper (signature + body). Determine which are anonymous-namespace and which are file-scope `static`. Common candidates in this kind of orchestration file: timing helpers, log-prefix helpers, parameter-stringification helpers.

If there ARE no shared helpers (the block at 1-140 is just includes + namespace open), report `STATUS: skip-task-1` and move directly to Task 2 — no header to create.

- [ ] **Step 2: If helpers exist, create `src/runtime/engine_internal.h`**

```cpp
#pragma once

// Internal helpers shared across engine_*.cpp translation units.
// Not part of any public API; included only by src/runtime/engine*.cpp.
//
// Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

#include "runtime/engine.h"
// Additional includes based on what the helpers reference.

namespace imp::engine_internal {

// Paste each helper here with:
//   - Free functions: prefix `inline`
//   - Function templates: keep as-is

}  // namespace imp::engine_internal
```

- [ ] **Step 3: Replace anonymous block + statics in `engine.cpp`**

Delete the anonymous-namespace block + each file-scope `static` helper. Add at the top (after existing `#include`s):

```cpp
#include "runtime/engine_internal.h"

using imp::engine_internal::<helper_name>;
// One using-decl per helper.
```

- [ ] **Step 4: Check CMakeLists.txt**

```bash
grep -n 'engine.cpp\|engine_internal' CMakeLists.txt
```

If headers are tracked: add `src/runtime/engine_internal.h`. If only sources are tracked: no change.

- [ ] **Step 5: Build**

```bash
make build
```

Expected: clean.

- [ ] **Step 6: Run tests**

```bash
make verify-fast
```

Expected: `=== verify fast: OK ===`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(engine): lift helpers to shared internal header

Anonymous-namespace and file-scope static helpers move to a new header
src/runtime/engine_internal.h so subsequent per-subsystem extractions
can share them across translation units.

No behavior change. engine.cpp still contains every Engine method —
those move to their own TUs in Tasks 2-7.

Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**If Task 1 was skipped** (no helpers existed), proceed directly to Task 2 with no foundation commit.

---

## Task 2: Extract InitResolver methods

**Goal:** Move 6 `init_resolve_*` + `init_compute_max_seq_len_` + `init_apply_debug_raw_overrides_` methods to their own TU.

**Files:**
- Create: `src/runtime/engine_init_resolver.cpp`
- Modify: `src/runtime/engine.cpp` — remove these 6 method definitions
- Modify: `CMakeLists.txt` — add new source

**Methods to move** (verify line ranges with grep):
- `Engine::init_apply_debug_raw_overrides_` (line 609)
- `Engine::init_resolve_kv_dtype_policy_` (line 641)
- `Engine::init_resolve_ssm_dtype_` (line 704)
- `Engine::init_resolve_fp8_prefill_` (line 726)
- `Engine::init_resolve_quant_flags_` (line 741)
- `Engine::init_compute_max_seq_len_` (line 885)

Total ~320 LOC (lines 609-928 excluding internal blank lines).

- [ ] **Step 1: Identify exact line ranges**

```bash
grep -n '^void Engine::init_\|^bool Engine::init(' src/runtime/engine.cpp | head -10
```

Note the boundary: the last init_resolve method ends just before `Engine::init(...)` at line 929.

- [ ] **Step 2: Create `src/runtime/engine_init_resolver.cpp`**

```cpp
// Engine init phase: resolve quant/KV/SSM dtype policies + compute max
// sequence length from VRAM budget. Pure orchestration of RuntimeConfig
// + Model metadata — no kernel launches, no allocations.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. Methods remain Engine::* with declarations in engine.h.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"  // omit if Task 1 was skipped
// Additional includes the bodies need:
//   - "runtime/config.h" for RuntimeConfig
//   - "core/logging.h" for IMP_LOG_*
//   - "runtime/vram_budget.h" for VRAMBudget (used by init_compute_max_seq_len_)
//   - Any quant/dtype headers the bodies reference

// using-decls for engine_internal:: helpers if any are used.

namespace imp {

// PASTE all 6 method bodies HERE verbatim in original order.

}  // namespace imp
```

- [ ] **Step 3: Remove the 6 methods from engine.cpp**

Delete lines 609-928 (or the actual range from Step 1). The orchestrator `Engine::init(...)` at line 929+ stays — it calls these methods.

- [ ] **Step 4: Update CMakeLists.txt**

```bash
grep -n 'engine.cpp' CMakeLists.txt
```

Add `src/runtime/engine_init_resolver.cpp` to the same `set(...)` block, immediately after `engine.cpp`.

- [ ] **Step 5: Build + tests**

```bash
make build
make verify-fast
```

Expected: both green.

- [ ] **Step 6: Confirm LOC delta**

```bash
wc -l src/runtime/engine.cpp src/runtime/engine_init_resolver.cpp
```

Expected: `engine.cpp` drops by ~320 LOC; new file ~330 LOC.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(engine): extract InitResolver methods to their own TU

Moves Engine::init_apply_debug_raw_overrides_,
Engine::init_resolve_kv_dtype_policy_,
Engine::init_resolve_ssm_dtype_,
Engine::init_resolve_fp8_prefill_,
Engine::init_resolve_quant_flags_, and
Engine::init_compute_max_seq_len_ (~320 LOC across 6 methods) to
src/runtime/engine_init_resolver.cpp.

Declarations in engine.h unchanged. Bodies byte-identical.

Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extract WeightUploadOrchestrator (init_weights)

**Files:**
- Create: `src/runtime/engine_weight_upload.cpp`
- Modify: `src/runtime/engine.cpp` — remove `init_weights` definition
- Modify: `CMakeLists.txt`

**Method to move:** `Engine::init_weights` (line 975, ~230 LOC).

- [ ] **Step 1: Identify line range**

```bash
grep -n '^bool Engine::init_weights\|^bool Engine::init_kv_cache' src/runtime/engine.cpp
```

- [ ] **Step 2: Create `src/runtime/engine_weight_upload.cpp`**

```cpp
// Engine init phase: weight upload to VRAM.
// Calls upload_weight + upload_expert_weights from weight_upload.cu,
// triggers pre-dequant via executor_pre_dequant.cu.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"  // omit if Task 1 was skipped
// Determine other includes from the body — likely:
//   - "model/weight_upload.h"
//   - "exec/executor.h" (for GraphExecutor)
//   - "core/logging.h"
//   - "memory/weight_cache.h"

namespace imp {

// PASTE Engine::init_weights() body HERE verbatim.

}  // namespace imp
```

- [ ] **Step 3: Remove `init_weights` from engine.cpp**

Delete the function definition.

- [ ] **Step 4: Update CMakeLists.txt**

Add `src/runtime/engine_weight_upload.cpp`.

- [ ] **Step 5: Build + tests**

```bash
make build && make verify-fast
```

- [ ] **Step 6: Confirm LOC delta** (engine.cpp drops by ~230 LOC)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(engine): extract WeightUploadOrchestrator (init_weights) to its own TU

Moves Engine::init_weights (~230 LOC) from src/runtime/engine.cpp to
src/runtime/engine_weight_upload.cpp.

Declaration in engine.h unchanged. Body byte-identical.

Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extract KVCacheInitializer (init_kv_cache)

Same pattern as Task 3.

**Files:**
- Create: `src/runtime/engine_kv_cache_init.cpp`
- Modify: `src/runtime/engine.cpp` — remove `init_kv_cache`
- Modify: `CMakeLists.txt`

**Method to move:** `Engine::init_kv_cache` (line 1205, ~255 LOC).

- [ ] **Step 1: Identify line range**

```bash
grep -n '^bool Engine::init_kv_cache\|^bool Engine::init_features' src/runtime/engine.cpp
```

- [ ] **Step 2: Create `src/runtime/engine_kv_cache_init.cpp`**

```cpp
// Engine init phase: paged KV cache allocation.
// Decides block geometry (block_size=16), allocates blocks per KV dtype
// (FP16/FP8/INT8/INT4/NVFP4/MXFP4), wires up KVCacheManager.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"  // omit if Task 1 was skipped
// Likely needs:
//   - "memory/kv_cache.h"
//   - "memory/kv_cache_manager.h"
//   - "core/logging.h"
//   - "runtime/config.h"

namespace imp {

// PASTE Engine::init_kv_cache() body HERE verbatim.

}  // namespace imp
```

- [ ] **Step 3: Remove `init_kv_cache` from engine.cpp**

- [ ] **Step 4: Update CMakeLists.txt**

- [ ] **Step 5: Build + tests**

- [ ] **Step 6: Confirm LOC** (engine.cpp drops by ~255 LOC)

- [ ] **Step 7: Commit** (analogous message to Task 3)

```bash
git commit -m "$(cat <<'EOF'
refactor(engine): extract KVCacheInitializer (init_kv_cache) to its own TU

Moves Engine::init_kv_cache (~255 LOC) to src/runtime/engine_kv_cache_init.cpp.

Declaration in engine.h unchanged. Body byte-identical.

Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extract WorkspaceBuilder + Warmup + banned-token list

**Why combined:** Three smaller methods that all run as part of the init sequence after KV cache is up. Combining keeps related logic colocated and avoids 3 tiny PRs.

**Files:**
- Create: `src/runtime/engine_workspace_warmup.cpp`
- Modify: `src/runtime/engine.cpp` — remove all three methods
- Modify: `CMakeLists.txt`

**Methods to move:**
- `Engine::init_features` (line 1460, ~77 LOC)
- `Engine::build_banned_token_list` (line 1537, ~85 LOC)
- `Engine::warmup` (line 1622, ~91 LOC)

Total ~253 LOC.

- [ ] **Step 1: Identify line ranges**

```bash
grep -n '^bool Engine::init_features\|^void Engine::build_banned_token_list\|^void Engine::warmup\|^bool Engine::step\b' src/runtime/engine.cpp
```

- [ ] **Step 2: Create `src/runtime/engine_workspace_warmup.cpp`**

```cpp
// Engine init phase (tail): workspace buffers + banned-token list + warmup.
//
// init_features:
//   Allocates MMVQ scratch, cuBLAS S-matrix workspace, FP8 activation
//   scratch, split-K attn scratch. See executor_workspace_*.cu for the
//   actual workspace builders.
//
// build_banned_token_list:
//   Constructs the runtime ban-list from RuntimeConfig + model tokenizer.
//
// warmup:
//   Optional first forward pass to prime cuBLAS + CUDA graph capture.
//   Off by default; opt-in via runtime.warmup config.
//
// All three execute at engine-init time and are colocated here as the
// "init tail" — after weights + KV cache are up but before the engine
// becomes ready.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"  // omit if Task 1 was skipped
// Determine other includes from the bodies.

namespace imp {

// PASTE Engine::init_features() body HERE.
// PASTE Engine::build_banned_token_list() body HERE.
// PASTE Engine::warmup() body HERE.

}  // namespace imp
```

- [ ] **Step 3: Remove all three from engine.cpp**

- [ ] **Step 4: Update CMakeLists.txt**

- [ ] **Step 5: Build + tests**

- [ ] **Step 6: Confirm LOC** (engine.cpp drops by ~253 LOC)

- [ ] **Step 7: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(engine): extract WorkspaceBuilder + Warmup + banned-token list

Combines three small init-tail methods into one TU:
  - Engine::init_features (~77 LOC)
  - Engine::build_banned_token_list (~85 LOC)
  - Engine::warmup (~91 LOC)

All three run during engine init after weights + KV cache are set up;
colocating them keeps the init-tail logic in one file. Declarations in
engine.h unchanged. Bodies byte-identical.

Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Extract Scheduler (the big one)

**Why hardest:** ~1250 LOC across 13 methods — the per-step scheduling, prefill, and decode loops. This is engine.cpp's bulk.

**Files:**
- Create: `src/runtime/engine_scheduler.cpp`
- Modify: `src/runtime/engine.cpp` — remove all 13 methods
- Modify: `CMakeLists.txt`

**Methods to move (in original order):**
1. `Engine::step` (line 1713)
2. `Engine::step_async_graph_resume` (line 1746)
3. `Engine::step_schedule` (line 1808)
4. `Engine::supports_chunked_prefill_` (line 1842)
5. `Engine::resolve_prefill_chunk_size_` (line 1891)
6. `Engine::step_prefill` (line 1913)
7. `Engine::prefill_allocate_kv_blocks_` (line 1949)
8. `Engine::prefill_upload_metadata_` (line 2023)
9. `Engine::step_prefill_one` (line 2105)
10. `Engine::step_decode` (line 2379)
11. `Engine::decode_build_inference_state_` (line 2445)
12. `Engine::step_decode_forward` (line 2562)
13. `Engine::step_decode_process_outputs` (line 2859)

Range ends just before `Engine::generate` at line 2962.

- [ ] **Step 1: Enumerate every function in the Scheduler range**

```bash
grep -n '^[A-Za-z].*Engine::step\|^[A-Za-z].*Engine::prefill_\|^[A-Za-z].*Engine::decode_\|^[A-Za-z].*Engine::supports_chunked\|^[A-Za-z].*Engine::resolve_prefill\|^std::string Engine::generate' src/runtime/engine.cpp
```

Expected: 13 step/prefill/decode matches + the `generate` boundary line. Verify count is 13. If different, **stop and report**.

- [ ] **Step 2: Confirm contiguity**

Between the first scheduler method and `Engine::generate`, there should be NO other Engine method definitions (no orphan methods to drag along). Verify with:

```bash
grep -n '^[A-Za-z].*Engine::' src/runtime/engine.cpp | awk -v start=1713 -v end=2961 '$1 >= start && $1 <= end {print}'
```

Expected: exactly the 13 methods.

- [ ] **Step 3: Create `src/runtime/engine_scheduler.cpp`**

```cpp
// Engine scheduler: per-step prefill + decode driver.
// The bulk of engine.cpp at ~1250 LOC across 13 methods.
//
// Top-level flow:
//   step() → step_schedule() → step_prefill() OR step_decode()
//
// Prefill chain:
//   step_prefill → resolve_prefill_chunk_size_ → supports_chunked_prefill_
//                → prefill_allocate_kv_blocks_ → prefill_upload_metadata_
//                → step_prefill_one (per chunk)
//
// Decode chain:
//   step_decode → decode_build_inference_state_ → step_decode_forward
//               → step_decode_process_outputs
//
// Async graph resume:
//   step_async_graph_resume — for the CUDA-graph-captured decode path.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. This is the biggest single TU split in Phase 4.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"  // omit if Task 1 was skipped
// Likely needs many includes — determine from the bodies:
//   - "exec/executor.h"
//   - "memory/kv_cache_manager.h"
//   - "runtime/request.h"
//   - "runtime/batch.h"
//   - "runtime/scheduler.h" (if separate)
//   - "runtime/cuda_graph.h"
//   - "core/logging.h"
//   - "runtime/config.h"
//   - "<cuda_runtime.h>"

namespace imp {

// PASTE all 13 method bodies HERE in original order.

}  // namespace imp
```

- [ ] **Step 4: Remove all 13 methods from engine.cpp**

Delete by line range from Step 1. No orphan signatures or stray closing braces.

- [ ] **Step 5: Update CMakeLists.txt**

- [ ] **Step 6: Build + tests**

```bash
make build && make verify-fast
```

Expected: green. Scheduler is exercised by every gtest that creates an Engine — a failure indicates the extraction lost state.

- [ ] **Step 7: Confirm LOC delta**

```bash
wc -l src/runtime/engine.cpp src/runtime/engine_scheduler.cpp
```

Expected: engine.cpp drops by ~1250 LOC; new file ~1260 LOC.

- [ ] **Step 8: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(engine): extract Scheduler to its own TU

Moves 13 step/prefill/decode methods (~1250 LOC, the bulk of engine.cpp)
to src/runtime/engine_scheduler.cpp:
  - step, step_async_graph_resume, step_schedule
  - supports_chunked_prefill_, resolve_prefill_chunk_size_
  - step_prefill, prefill_allocate_kv_blocks_,
    prefill_upload_metadata_, step_prefill_one
  - step_decode, decode_build_inference_state_,
    step_decode_forward, step_decode_process_outputs

All declarations in engine.h unchanged. Bodies byte-identical.

This is the largest single drop in Phase 4 of the architecture refactor.

Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Extract Sampling + Stop helpers

**Why combined:** Two related concern clusters (sampling parameter prep + stop-token detection) that share state-passing patterns. Combining avoids splitting tightly-related logic across two PRs.

**Files:**
- Create: `src/runtime/engine_sampling_stop.cpp`
- Modify: `src/runtime/engine.cpp` — remove these methods
- Modify: `CMakeLists.txt`

**Methods to move:**
- Stop logic: `Engine::is_stop_token` (line 374), `Engine::track_think_state` (line 391), `Engine::should_stop` (line 439)
- Sampling helpers: `Engine::fill_sampling_params` (line 481), `Engine::upload_penalties` (line 539), `Engine::fill_recurrent_state` (line 564)

Total ~250 LOC.

- [ ] **Step 1: Identify line ranges**

```bash
grep -n '^bool Engine::is_stop_token\|^void Engine::track_think_state\|^bool Engine::should_stop\|^void Engine::fill_sampling_params\|^void Engine::upload_penalties\|^void Engine::fill_recurrent_state\|^void Engine::finish_request' src/runtime/engine.cpp
```

Expected: 6 method matches + the `finish_request` boundary (line 580).

- [ ] **Step 2: Create `src/runtime/engine_sampling_stop.cpp`**

```cpp
// Engine sampling helpers + stop-token detection.
//
// fill_sampling_params:    pull per-request sampling config into InferenceState
// upload_penalties:        copy penalty buffers (repeat/freq/presence/DRY) to device
// fill_recurrent_state:    SSM/GDN per-request state setup
// is_stop_token:           single-token stop check (EOS variants)
// track_think_state:       update <think>...</think> blockcount budget
// should_stop:             aggregate stop check (EOS, max_tokens, stop_strings)
//
// Two related concern clusters colocated because they share the
// per-request state-passing pattern (Request& + InferenceState&) and
// run in the decode loop's tail.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"  // omit if Task 1 was skipped
// Determine includes from the bodies — likely:
//   - "runtime/request.h"
//   - "runtime/batch.h" (for InferenceState)
//   - "compute/sampling.h"
//   - "core/logging.h"
//   - "model/tokenizer.h"

namespace imp {

// PASTE all 6 method bodies HERE in original order:
//   1. is_stop_token (was at line 374)
//   2. track_think_state (was at line 391)
//   3. should_stop (was at line 439)
//   4. fill_sampling_params (was at line 481)
//   5. upload_penalties (was at line 539)
//   6. fill_recurrent_state (was at line 564)

}  // namespace imp
```

- [ ] **Step 3: Remove all 6 methods from engine.cpp**

- [ ] **Step 4: Update CMakeLists.txt**

- [ ] **Step 5: Build + tests**

- [ ] **Step 6: Confirm LOC delta** (engine.cpp drops by ~250 LOC)

- [ ] **Step 7: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(engine): extract Sampling helpers + Stop controller

Moves 6 decode-loop tail methods to src/runtime/engine_sampling_stop.cpp:
  - is_stop_token, track_think_state, should_stop (stop check)
  - fill_sampling_params, upload_penalties, fill_recurrent_state (per-request setup)

Two related clusters colocated because they share the per-request
state-passing pattern (Request& + InferenceState&) and run in the
decode loop tail.

Declarations in engine.h unchanged. Bodies byte-identical.

Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 (SOFT — may slip): Move MTP forward kernel to src/compute/

Per spec §3 Phase 4 soft PRs: "Move `src/runtime/mtp_forward.cu` → `src/compute/mtp_forward.cu`. It is a kernel file, not an orchestrator."

Plan briefly:

- [ ] **Step 1: Move the file**

```bash
git mv src/runtime/mtp_forward.cu src/compute/mtp_forward.cu
```

- [ ] **Step 2: Update `#include`s**

```bash
grep -rln '#include "runtime/mtp_forward' src/ tests/ tools/
```

For each match, rewrite to `compute/mtp_forward`.

- [ ] **Step 3: Update CMakeLists.txt** — move the entry from `IMP_RUNTIME_SOURCES` to `IMP_COMPUTE_SOURCES`.

- [ ] **Step 4: Build + tests** (`make build && make verify-fast`).

- [ ] **Step 5: Commit**.

---

## Task 9 (SOFT — may slip): Move vision_pipeline to src/vision/

Per spec: "Move `src/runtime/vision_pipeline.{cpp,h}` → `src/vision/vision_pipeline.{cpp,h}`."

Plan briefly:

- [ ] **Step 1: Move the files**

```bash
git mv src/runtime/vision_pipeline.cpp src/vision/vision_pipeline.cpp
git mv src/runtime/vision_pipeline.h src/vision/vision_pipeline.h
```

- [ ] **Step 2: Update includes**

```bash
grep -rln '#include "runtime/vision_pipeline' src/ tests/ tools/
```

Rewrite each to `vision/vision_pipeline`.

- [ ] **Step 3: Update CMakeLists.txt** — move from `IMP_RUNTIME_SOURCES` to `IMP_VISION_SOURCES`.

- [ ] **Step 4: Build + tests**.

- [ ] **Step 5: Commit**.

---

## Phase 4 closeout

After Tasks 2-7 are merged (Task 1 may have been skipped; Tasks 8-9 are soft):

- [ ] **Step 1: Full verification suite**

```bash
make verify
```

Expected: green except for pre-existing failures.

- [ ] **Step 2: Confirm engine.cpp final LOC**

```bash
wc -l src/runtime/engine.cpp src/runtime/engine_*.cpp
```

Plan target for `engine.cpp`: **≤800 LOC**. Started 3112 LOC; after Tasks 2-7 cumulative drop: 320 + 230 + 255 + 253 + 1250 + 250 = ~2558 LOC moved out → ~554 LOC remaining. Well under target.

Sub-TUs expected sizes:
- `engine_init_resolver.cpp` ~330 LOC
- `engine_weight_upload.cpp` ~240 LOC
- `engine_kv_cache_init.cpp` ~265 LOC
- `engine_workspace_warmup.cpp` ~265 LOC
- `engine_scheduler.cpp` ~1260 LOC
- `engine_sampling_stop.cpp` ~260 LOC
- Plus the existing `engine_graph_decode.cpp` (unmodified)

- [ ] **Step 3: Perf snapshot** (advisory)

```bash
scripts/gen_perf_baseline.sh
git diff tests/perf_baseline.json
```

Phase 4 is structural — no perf change expected.

- [ ] **Step 4: Update MEMORY.md**

Write `architecture_refactor_phase_4_closed_2026_MM_DD.md` in `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/`. Update the Architecture section header to point at the new memo. Compact older memos if MEMORY.md is over the 200-line target.

- [ ] **Step 5: Update `docs/architecture.md`**

The "engine.cpp is 3112 LOC across one .cpp file" wound at lines 143-148 needs rewriting. Post-Phase-4 it's a thin orchestrator + named subsystem TUs. Update.

- [ ] **Step 6: Mark Phase 4 closed in roadmap spec**

Add Status line at top of Phase 4 section with PR numbers.

- [ ] **Step 7: Commit closeout on `docs/phase-4-closeout` branch + PR.**

Phase 5 (Schichten und APIs) may now begin — final phase of the refactor.
