# Architecture Refactor Phase 3 — Pre-Dequant Aufräumen

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/exec/executor_pre_dequant.cu` (2693 LOC) becomes a thin dispatcher (≤200 LOC) that calls into named per-phase files. Each phase becomes a self-contained source file with one clear responsibility.

**Architecture:** Sequential extraction of the existing phases into their own `.cu` files. Each phase already has a name (`pre_dequant_phase0_promote_nvfp4_sidecars_`, `pre_dequant_phase1_fp16_cache_`, …) and a clean entry-point signature. The refactor moves each phase's body to its own file with no behavior change. The dispatcher in `executor_pre_dequant.cu` becomes the orchestrator that calls the phases in order.

**Tech Stack:** C++20, CUDA 13.2, CMake, Docker build (`make build`), GTest suite (`make verify-fast`).

---

## Reference: Source spec

`docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md` §3 Phase 3.

## Spec deviation noted upfront

The spec's Critical PR 2 suggested splitting **by quant family** (`pre_dequant_q8.cu`, `pre_dequant_q4k.cu`, `pre_dequant_mxfp4.cu`, `pre_dequant_nvfp4.cu`, `pre_dequant_fp16_cache.cu`). Inspection of the actual code shows this is structurally infeasible:

- Phase 1 (`pre_dequant_phase1_fp16_cache_`) handles ALL GGUF Q-types (Q8_0, Q4_K, Q5_K, Q6_K, …) → FP16 in a single shared loop. There is no per-Q-type function to extract.
- Phase 2 handles FP16 → FP8 as a generic conversion.
- Phase 3 handles NVFP4 decode quantization but is spread across ~10 helper functions (`nvfp4_decode_*`) that orchestrate the sub-steps.
- The "Q4_K_M → INT8 via Phase 3 IMMA path" the spec references doesn't exist as a pre-dequant phase yet — IMMA tooling currently lives in `src/exec/mmq_q4k.cu` (Phase 1 v2 plan, not pre-dequant).

The **goal** of the spec (thin dispatcher + one file per concern) still applies. The plan below extracts **by phase** instead of by quant family, which matches the existing function naming and the sequencing constraints (phases run in order: 0 → 0b → 4 → 1 → 2 → 3 → 3c). The result is the same: `executor_pre_dequant.cu` shrinks to ≤200 LOC pure dispatcher, and adding a new phase becomes one file.

The spec's separate registry pattern (`register_pre_dequant(QType src, QType dst, DequantFn fn)`) is left for the **soft PR refactor of `gemm_kernel_*.cu`** (where format-by-format dispatch IS the actual code structure). It does not apply to `executor_pre_dequant.cu`.

## Reference: Pre-flight inventory

`src/exec/executor_pre_dequant.cu` at plan-write time:

| Section | LOC range | Functions |
|---|---|---|
| `namespace imp` open + helpers | 23-187 | `borrow_payload_from_wcache`, `deduct_budget`, `create_fused_weight_pair`, `for_each_dense_weight` |
| `pre_dequant_weights` (entry) | 189-246 | The orchestrator that calls all phases |
| Phase 4 tensor registry | 247-624 | `pre_dequant_phase4_tensor_registry_` |
| Phase 0 NVFP4 sidecars | 625-855 | `pre_dequant_phase0_promote_nvfp4_sidecars_` |
| Phase 0b CUTLASS NVFP4 | 856-948 | `pre_dequant_phase0b_register_cutlass_nvfp4_` |
| Phase 1 FP16 cache | 949-1078 | `pre_dequant_phase1_fp16_cache_` (~130 LOC) |
| Phase 2 FP8 cache | 1079-1220 | `pre_dequant_phase2_fp8_cache_` (~142 LOC) |
| Phase 3 NVFP4 decode | 1221-2546 | `pre_dequant_phase3_nvfp4_decode_` + 10 helpers (`nvfp4_decode_*`) (~1325 LOC, the bulk of the file) |
| Phase 3c standalone MXFP4 | 2547-2692 | `pre_dequant_phase3c_standalone_mxfp4_` |

Total: 2693 LOC.

Helper signatures live on `GraphExecutor` (declared in `src/exec/executor.h`); extraction must keep them as `GraphExecutor::` methods so the existing `pre_dequant_weights` orchestrator can call them across translation units.

---

## Task 1: Lift anonymous-namespace helpers to a shared header

**Why first:** Multiple phase files will reference the same helpers (`borrow_payload_from_wcache`, `for_each_dense_weight`, `deduct_budget`, `create_fused_weight_pair`). These currently live in the file's anonymous namespace and aren't reachable across TUs. Extract them into a shared private header so each phase TU can include it.

**Files:**
- Create: `src/exec/pre_dequant_internal.h` — declarations + `inline` definitions of helpers
- Modify: `src/exec/executor_pre_dequant.cu` — replace the anonymous-namespace block with `#include "exec/pre_dequant_internal.h"`

- [ ] **Step 1: Read the current anonymous-namespace block**

```bash
sed -n '28,115p' src/exec/executor_pre_dequant.cu
```

Note every helper signature + body. There are 4 helpers per the pre-flight inventory.

- [ ] **Step 2: Create `src/exec/pre_dequant_internal.h`**

Write the file with this exact structure (replace `<HELPER_BODIES>` with the verbatim function bodies from Step 1):

```cpp
#pragma once

// Internal helpers shared across pre_dequant_phase*.cu translation units.
// Not part of any public API; included only by exec/pre_dequant_*.cu files.
//
// Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

#include "core/tensor.h"
#include "memory/weight_cache.h"
#include "exec/executor.h"  // WeightHandle, GraphExecutor
#include <cstddef>

namespace imp::pre_dequant_internal {

<HELPER_BODIES — paste the 4 functions here verbatim with `inline` prefix on each non-template free function; keep template signatures unchanged>

}  // namespace imp::pre_dequant_internal
```

The `inline` keyword is required for free functions in a header to avoid ODR violations. Templates stay as-is.

- [ ] **Step 3: Replace the anonymous block in executor_pre_dequant.cu**

In `src/exec/executor_pre_dequant.cu`, delete lines 28-115 (the `namespace {` block) and replace with:

```cpp
#include "exec/pre_dequant_internal.h"

using imp::pre_dequant_internal::borrow_payload_from_wcache;
using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::create_fused_weight_pair;
using imp::pre_dequant_internal::for_each_dense_weight;
```

(Adjust the using-declaration list to match the actual helper names from Step 1.)

- [ ] **Step 4: Add the new header to CMakeLists.txt** if header listings are tracked

```bash
grep -n 'executor_pre_dequant\|pre_dequant_internal' CMakeLists.txt
```

If `executor_pre_dequant.cu` is listed but headers are not, no CMakeLists change is needed (CMake auto-picks up the include). If headers ARE listed (per-target `set(... HEADERS ...)`), add `src/exec/pre_dequant_internal.h` alongside `executor_pre_dequant.cu`.

- [ ] **Step 5: Build**

```bash
make build
```

Expected: clean. If `error: redefinition of 'borrow_payload_from_wcache'` appears, you missed deleting the anonymous-namespace block — re-do Step 3.

- [ ] **Step 6: Run tests**

```bash
make verify-fast
```

Expected: `=== verify fast: OK ===`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(pre-dequant): lift helpers to shared internal header

Anonymous-namespace helpers (borrow_payload_from_wcache,
for_each_dense_weight, deduct_budget, create_fused_weight_pair) move to
a new header src/exec/pre_dequant_internal.h so subsequent per-phase
extractions can share them across translation units.

No behavior change.

Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Extract Phase 1 — FP16 cache

**Why this phase first:** Phase 1 (`pre_dequant_phase1_fp16_cache_`) is the smallest self-contained extraction (~130 LOC), has no MoE-specific helpers, and is the standard pattern that subsequent extractions will repeat. Validates the extraction approach.

**Files:**
- Create: `src/exec/pre_dequant_phase1_fp16_cache.cu`
- Modify: `src/exec/executor_pre_dequant.cu` — remove the phase body, keep only the dispatcher line
- Modify: `CMakeLists.txt` — add the new source file

- [ ] **Step 1: Read the current phase body**

```bash
sed -n '949,1078p' src/exec/executor_pre_dequant.cu
```

Note the exact function signature (`void GraphExecutor::pre_dequant_phase1_fp16_cache_(...)`).

- [ ] **Step 2: Create the new file**

Write `src/exec/pre_dequant_phase1_fp16_cache.cu`:

```cpp
// Pre-dequant Phase 1: FP16 cache.
// Converts all GGUF Q*_K-quantized weights to an FP16 device cache,
// gated by attention.mxfp4_fp16_cache_policy (legacy/pruned).
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. See pre_dequant_internal.h for shared helpers.

#include "core/tensor.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "memory/weight_cache.h"
#include "model/model.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"

using imp::pre_dequant_internal::borrow_payload_from_wcache;
using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::for_each_dense_weight;

namespace imp {

<PHASE1_BODY — paste the entire pre_dequant_phase1_fp16_cache_ function body here, exactly as it was in executor_pre_dequant.cu>

}  // namespace imp
```

The `<PHASE1_BODY>` placeholder must be replaced with the actual function. Include any inner helpers that were local-only to this phase (lambdas, file-scope statics if any — note them while reading Step 1).

Header includes must be the minimal set that compiles. If the original file used additional includes (e.g. `<cuda_runtime.h>`), add them to the new file too.

- [ ] **Step 3: Remove the phase body from executor_pre_dequant.cu**

In `src/exec/executor_pre_dequant.cu`, delete lines 949-1078 (the entire `void GraphExecutor::pre_dequant_phase1_fp16_cache_(...)` definition).

The declaration in `src/exec/executor.h` stays — only the definition moves.

- [ ] **Step 4: Update CMakeLists.txt**

```bash
grep -n 'executor_pre_dequant.cu' CMakeLists.txt
```

Add `src/exec/pre_dequant_phase1_fp16_cache.cu` to the same `set(IMP_EXEC_SOURCES ...)` block, immediately after the `executor_pre_dequant.cu` line.

- [ ] **Step 5: Build**

```bash
make build
```

Expected: clean. If `error: 'pre_dequant_phase1_fp16_cache_' is not a member of 'GraphExecutor'` appears, the declaration in `executor.h` was lost — restore it. If `undefined reference to GraphExecutor::pre_dequant_phase1_fp16_cache_`, the new file isn't in CMakeLists — re-do Step 4.

- [ ] **Step 6: Run tests**

```bash
make verify-fast
```

Expected: `=== verify fast: OK ===`.

- [ ] **Step 7: Confirm LOC delta**

```bash
wc -l src/exec/executor_pre_dequant.cu src/exec/pre_dequant_phase1_fp16_cache.cu
```

Expected: ~2563 LOC in executor_pre_dequant.cu (down from 2693), ~135 LOC in the new file (130 phase body + ~5 lines of header/namespace boilerplate).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(pre-dequant): extract Phase 1 (FP16 cache) to its own TU

Moves GraphExecutor::pre_dequant_phase1_fp16_cache_ (~130 LOC) from
src/exec/executor_pre_dequant.cu to src/exec/pre_dequant_phase1_fp16_cache.cu.

Declaration in executor.h unchanged. Body is byte-identical.

Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extract Phase 2 — FP8 cache

Same pattern as Task 2.

**Files:**
- Create: `src/exec/pre_dequant_phase2_fp8_cache.cu`
- Modify: `src/exec/executor_pre_dequant.cu` — remove Phase 2 body
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Read the current phase body**

```bash
sed -n '1079,1220p' src/exec/executor_pre_dequant.cu
```

- [ ] **Step 2: Create `src/exec/pre_dequant_phase2_fp8_cache.cu`** with this skeleton (replace `<PHASE2_BODY>`):

```cpp
// Pre-dequant Phase 2: FP8 cache.
// Converts FP16 cache (Phase 1 output) to FP8 device tensors for the
// fp8_prefill path, gated by attention.fp8_prefill.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap.

#include "core/tensor.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "memory/weight_cache.h"
#include "model/model.h"
#include "quant/fp8_quant.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"

using imp::pre_dequant_internal::borrow_payload_from_wcache;
using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::for_each_dense_weight;

namespace imp {

<PHASE2_BODY>

}  // namespace imp
```

- [ ] **Step 3: Remove Phase 2 body** from executor_pre_dequant.cu (lines 1079-1220).

- [ ] **Step 4: Update CMakeLists.txt**: add `src/exec/pre_dequant_phase2_fp8_cache.cu`.

- [ ] **Step 5: Build** (`make build`) — expected clean.

- [ ] **Step 6: Tests** (`make verify-fast`) — expected green.

- [ ] **Step 7: Confirm LOC** — executor_pre_dequant.cu drops by ~142 LOC.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(pre-dequant): extract Phase 2 (FP8 cache) to its own TU

Moves GraphExecutor::pre_dequant_phase2_fp8_cache_ (~142 LOC) to
src/exec/pre_dequant_phase2_fp8_cache.cu. Declaration unchanged, body
byte-identical.

Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extract Phase 4 — tensor registry

**Why now:** Phase 4 is large (~378 LOC) but self-contained: it's the tensor-name → role registration step. Lives in the middle of the file (lines 247-624) so extracting it shrinks the orchestrator dramatically.

**Files:**
- Create: `src/exec/pre_dequant_phase4_tensor_registry.cu`
- Modify: `src/exec/executor_pre_dequant.cu` (remove body)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Read the phase body**

```bash
sed -n '247,624p' src/exec/executor_pre_dequant.cu
```

The function name is `GraphExecutor::pre_dequant_phase4_tensor_registry_`. Note any file-local statics or anonymous-namespace adjuncts that need to move with it.

- [ ] **Step 2: Create the new file** (skeleton same as Task 2/3, adapt includes per actual usage of Phase 4 body):

```cpp
// Pre-dequant Phase 4: tensor registry.
// Walks the model's WeightMap and registers each tensor's role +
// runtime location in the GraphExecutor's tensor table.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap.

#include "core/tensor.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "model/model.h"
#include "model/weight_map.h"

namespace imp {

<PHASE4_BODY>

}  // namespace imp
```

Include adjuncts that were file-local-only (lambdas inside the function are fine — they move with the body).

- [ ] **Step 3: Remove from executor_pre_dequant.cu** (lines 247-624).

- [ ] **Step 4: Update CMakeLists.txt**: add new source.

- [ ] **Step 5: Build + tests** (`make build && make verify-fast`).

- [ ] **Step 6: Confirm LOC** — executor_pre_dequant.cu drops by ~378 LOC.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(pre-dequant): extract Phase 4 (tensor registry) to its own TU

Moves GraphExecutor::pre_dequant_phase4_tensor_registry_ (~378 LOC) to
src/exec/pre_dequant_phase4_tensor_registry.cu. Declaration unchanged,
body byte-identical.

Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extract Phase 0 + 0b — NVFP4 sidecars + CUTLASS

**Why combined:** Phase 0 (~230 LOC) and Phase 0b (~93 LOC) are both NVFP4-loader-side concerns and run consecutively. Extract together to a single file to keep related logic colocated.

**Files:**
- Create: `src/exec/pre_dequant_phase0_nvfp4_loader.cu`
- Modify: `src/exec/executor_pre_dequant.cu` (remove both phase bodies)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Read the bodies**

```bash
sed -n '625,948p' src/exec/executor_pre_dequant.cu
```

Two functions: `pre_dequant_phase0_promote_nvfp4_sidecars_` (lines 625-855) and `pre_dequant_phase0b_register_cutlass_nvfp4_` (lines 856-948).

- [ ] **Step 2: Create the new file**:

```cpp
// Pre-dequant Phase 0 + 0b: NVFP4 loader-side setup.
// Phase 0: promote NVFP4 SafeTensors sidecars (scales, codebooks) to
// device tensors. Phase 0b: register CUTLASS-NVFP4 weight metadata for
// the prefill GEMM path.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap.

#include "core/tensor.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "model/model.h"
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "runtime/config.h"

namespace imp {

<PHASE0_BODY>

<PHASE0B_BODY>

}  // namespace imp
```

- [ ] **Step 3: Remove both bodies from executor_pre_dequant.cu** (lines 625-948).

- [ ] **Step 4: Update CMakeLists.txt**.

- [ ] **Step 5: Build + tests**.

- [ ] **Step 6: Confirm LOC** — executor_pre_dequant.cu drops by ~323 LOC.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(pre-dequant): extract Phase 0 + 0b (NVFP4 loader) to its own TU

Moves GraphExecutor::pre_dequant_phase0_promote_nvfp4_sidecars_ (~230 LOC)
and GraphExecutor::pre_dequant_phase0b_register_cutlass_nvfp4_ (~93 LOC)
to a combined file src/exec/pre_dequant_phase0_nvfp4_loader.cu.

Both phases are NVFP4 loader-side concerns that run consecutively;
colocating them keeps related logic in one place. Declarations
unchanged, bodies byte-identical.

Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Extract Phase 3 — NVFP4 decode (the biggest piece)

**Why hardest:** Phase 3 is ~1325 LOC spread across 1 entry point + 10 helper functions (`nvfp4_decode_*`). This is the most complex extraction because the helpers are all `GraphExecutor::` methods that interact via state-passing (`Nvfp4DecodeContext`, `remaining_budget`). All 11 functions move together.

**Files:**
- Create: `src/exec/pre_dequant_phase3_nvfp4_decode.cu`
- Modify: `src/exec/executor_pre_dequant.cu` (remove all 11 functions)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Enumerate every function in the Phase 3 range**

```bash
grep -n '^void GraphExecutor::nvfp4_decode_\|^void GraphExecutor::pre_dequant_phase3_nvfp4_decode_' \
  src/exec/executor_pre_dequant.cu
```

Expected: 11 matches. Plan target functions are:
- `pre_dequant_phase3_nvfp4_decode_` (entry, ~47 LOC)
- `nvfp4_decode_collect_candidates_`
- `nvfp4_decode_quantize_mode2_`
- `nvfp4_decode_quantize_mode1_`
- `nvfp4_decode_free_fp16_and_migrate_fp8_`
- `nvfp4_decode_second_pass_`
- `nvfp4_decode_convert_cutlass_`
- `nvfp4_decode_convert_mxfp4_and_native_`
- `nvfp4_decode_mxfp4_fp16_fallback_`
- `nvfp4_decode_cache_moe_experts_`

Confirm the actual list matches. If grep returns a different count, **stop** and report — the extraction must include every function.

- [ ] **Step 2: Read each function range**

For each function found in Step 1, note the start and end lines. The Phase 3 range is approximately lines 1221-2546 but exact extents must be confirmed function-by-function.

- [ ] **Step 3: Create the new file**

```cpp
// Pre-dequant Phase 3: NVFP4 decode-cache quantization.
// Multi-step quantization of decode-side weights to NVFP4, including
// candidate collection, two-pass mode-1/2 quantize, FP8 migration of
// failed candidates, CUTLASS conversion, MXFP4-source conversion, and
// MoE expert caching.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. The bulk of the pre-dequant file's LOC lives here.

#include "core/tensor.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "memory/weight_cache.h"
#include "model/model.h"
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "quant/fp8_quant.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"

namespace imp {

<PHASE3_ENTRY_BODY>
<NVFP4_DECODE_HELPERS — all 10 in original order>

}  // namespace imp
```

Add additional `#include`s as needed based on what the actual function bodies reference. If `Nvfp4DecodeContext` is declared in `executor.h`, the existing include covers it; if it's declared in a separate `quant/nvfp4_decode.h`, add that too.

- [ ] **Step 4: Remove all 11 functions from executor_pre_dequant.cu**

Be precise: delete each function definition by its exact line range from Step 2. After deletion, the file should have a contiguous gap where Phase 3 used to be (no leftover function signatures, no stray closing braces).

- [ ] **Step 5: Update CMakeLists.txt**: add new source.

- [ ] **Step 6: Build**

```bash
make build
```

If link errors mention any `nvfp4_decode_*` symbol, the new file is missing that function — restore from Step 4's notes.

- [ ] **Step 7: Run tests**

```bash
make verify-fast
```

Expected: green. NVFP4 decode is exercised by the NVFP4 model tests; a failing test indicates the extraction lost state.

- [ ] **Step 8: Confirm LOC**

```bash
wc -l src/exec/executor_pre_dequant.cu src/exec/pre_dequant_phase3_nvfp4_decode.cu
```

executor_pre_dequant.cu should now be ≤700 LOC (started 2693; after Tasks 2-6 cumulative drop: ~130 + ~142 + ~378 + ~323 + ~1325 = ~2298 LOC moved out → ≤395 LOC remaining + some boilerplate gain). pre_dequant_phase3_nvfp4_decode.cu should be ~1335 LOC.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(pre-dequant): extract Phase 3 (NVFP4 decode) to its own TU

Moves GraphExecutor::pre_dequant_phase3_nvfp4_decode_ and 10
nvfp4_decode_* helpers (~1325 LOC, the bulk of the pre-dequant file)
to src/exec/pre_dequant_phase3_nvfp4_decode.cu.

All 11 functions stay GraphExecutor:: methods; their declarations in
executor.h are unchanged. Bodies are byte-identical to pre-extraction.

After this task, executor_pre_dequant.cu is the largest single drop in
the refactor, going from ~1370 LOC (post Tasks 2-5) to ~395 LOC.

Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Extract Phase 3c — standalone MXFP4

The final extraction. Phase 3c is ~146 LOC handling MXFP4 in models that don't go through the NVFP4 pipeline.

**Files:**
- Create: `src/exec/pre_dequant_phase3c_mxfp4.cu`
- Modify: `src/exec/executor_pre_dequant.cu` (remove body)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Read the body**

```bash
sed -n '2547,2692p' src/exec/executor_pre_dequant.cu
```

- [ ] **Step 2: Create the new file**

```cpp
// Pre-dequant Phase 3c: standalone MXFP4.
// Handles MXFP4-source models that don't go through the NVFP4 decode
// pipeline (Phase 3). Loads MXFP4 weights, optionally promotes to an
// FP16 fallback cache per attention.mxfp4_fp16_fallback.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap.

#include "core/tensor.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "memory/weight_cache.h"
#include "model/model.h"
#include "quant/mxfp4_gemm.h"
#include "runtime/config.h"

namespace imp {

<PHASE3C_BODY>

}  // namespace imp
```

- [ ] **Step 3: Remove from executor_pre_dequant.cu** (lines 2547-2692).

- [ ] **Step 4: Update CMakeLists.txt**.

- [ ] **Step 5: Build + tests**.

- [ ] **Step 6: Confirm final LOC**

```bash
wc -l src/exec/executor_pre_dequant.cu src/exec/pre_dequant_phase*.cu
```

Expected (target reached if all goes well):
- `src/exec/executor_pre_dequant.cu`: **≤200 LOC** (the plan target). Contains only the `pre_dequant_weights` orchestrator + helper-import using-declarations + the `for_each_dense_weight` template if it stayed.
- `pre_dequant_phase0_nvfp4_loader.cu`: ~330 LOC
- `pre_dequant_phase1_fp16_cache.cu`: ~135 LOC
- `pre_dequant_phase2_fp8_cache.cu`: ~150 LOC
- `pre_dequant_phase3_nvfp4_decode.cu`: ~1335 LOC
- `pre_dequant_phase3c_mxfp4.cu`: ~155 LOC
- `pre_dequant_phase4_tensor_registry.cu`: ~385 LOC

If executor_pre_dequant.cu is still over 200 LOC, identify what remains in your commit body. Likely cause: large blocks of comments, includes, or the orchestrator itself grew. Don't force further extraction if the remainder is genuinely the dispatcher.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(pre-dequant): extract Phase 3c (standalone MXFP4) — final split

Moves GraphExecutor::pre_dequant_phase3c_standalone_mxfp4_ (~146 LOC)
to src/exec/pre_dequant_phase3c_mxfp4.cu. This is the last phase
extraction: executor_pre_dequant.cu is now the orchestrator only
(<NEW_LOC> LOC, was 2693 LOC at Phase 3 start).

The plan target was ≤200 LOC; achieved <NEW_LOC>.

Adding a new phase now becomes: write one new src/exec/pre_dequant_phase*.cu
file with the body, add it to CMakeLists.txt, declare the method on
GraphExecutor in executor.h, and call it from pre_dequant_weights().

Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `<NEW_LOC>` with the measured value from Step 6.)

---

## Task 8 (SOFT — may slip): gemm_kernel_*.cu registry pattern alignment

Per spec §3 Phase 3 soft PRs: confirm or refactor `src/exec/gemm_kernel_*.cu` (8 files) to use the same registry pattern.

Plan briefly:

- [ ] **Step 1: Inspect the current state**: `cat src/exec/gemm_kernel_registry.cu src/exec/gemm_kernel_registry.h`. The registry pattern probably already exists.
- [ ] **Step 2: If it exists**, document it in `docs/architecture.md` and close the soft PR with a one-line "verified, no change needed" commit.
- [ ] **Step 3: If it doesn't exist**, write a separate spec for the refactor (it's not a Phase 3 closeout item).

---

## Task 9 (SOFT — may slip): Reconcile src/quant/dequant_*.cu with src/exec/pre_dequant_*.cu

Per spec: the two locations have overlapping concerns (in-place dequant for hot paths vs one-shot dequant at init).

Plan briefly:

- [ ] **Step 1: Inspect** `src/quant/dequant_*.cu` files — what they actually contain.
- [ ] **Step 2: Decide**: merge, document the boundary, or leave alone. Probably document.
- [ ] **Step 3: Write a section** in `docs/architecture.md` titled "Dequant pipeline" explaining the boundary between `src/quant/dequant_*.cu` (kernels for hot paths) and `src/exec/pre_dequant_*.cu` (init-time orchestration).
- [ ] **Step 4: Commit**.

---

## Phase 3 closeout

After Tasks 1-7 are merged (Tasks 8-9 may slip):

- [ ] **Step 1: Full verification suite**

```bash
make verify
```

Expected: green (modulo pre-existing failures inherited from earlier phases).

- [ ] **Step 2: Perf snapshot** (advisory)

```bash
scripts/gen_perf_baseline.sh
git diff tests/perf_baseline.json
```

Phase 3 is structural — no perf change expected. Document any surprise.

- [ ] **Step 3: Update MEMORY.md**

Write a new memory file `architecture_refactor_phase_3_closed_2026_MM_DD.md` in `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/`. Add ONE line to MEMORY.md index. Update the Phase 2 entry to past tense if needed.

- [ ] **Step 4: Mark Phase 3 closed in the roadmap spec**

Edit the roadmap spec, add a Status line at the top of the Phase 3 section listing all 7+1 PR numbers.

- [ ] **Step 5: Update `docs/architecture.md`** if it references `executor_pre_dequant.cu` (it does — line 56 mentions it). Note the post-Phase-3 layout: one orchestrator + 6 phase files.

- [ ] **Step 6: Commit closeout on `docs/phase-3-closeout` branch + PR.**

Phase 4 ("engine.cpp zerteilen") may now begin. A new writing-plans output is required.
