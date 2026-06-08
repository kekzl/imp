# executor.h header decongestion (D2 step 2)

Date: 2026-06-08

## Problem

`src/exec/executor.h` (~1075 lines) declares the `GraphExecutor` class plus ~10
supporting structs inline before it. Any edit to that header recompiles every TU
that includes it (most of `src/exec/`), and the supporting types have no focused
home. This is the "1100-line header" half of D2. The QuantPipeline extraction
(PR #629) handled the first real component; this step is the safe,
declaration-only decongestion that shrinks the header and gives each type a home —
making the later runner extractions (Workspace, MoeRunner) cleaner because they
can include only what they need.

Discovery that shaped this: `ExpertLRUCache` is already a full class (methods
defined in `src/exec/expert_cache.cu`) — only its declaration is stranded in
`executor.h`. And `runtime/vram_budget.h` currently `#include "exec/executor.h"`
solely to see the 12-line `VRAMBudget` struct — a god-object include for one type.

## Goal & constraints

- Move the supporting structs out of `executor.h` into focused headers.
- **Strictly behaviour-neutral and declaration-only** — no logic changes, no
  member reordering within a struct, no hot-path code touched. Pure relocation of
  type declarations.
- `executor.h` re-includes the new headers, so every current consumer (which
  sees these types transitively via `executor.h`) keeps compiling unchanged.
- Gated by build + `verify-fast` + a dense+MoE coherence smoke. (No 4-arch canary
  or `diff=0` check needed: no executable code changes — only where types are
  declared. The forward TUs are untouched.)

## Architecture

Five focused headers (four new + one existing) replace the inline struct block in
`executor.h`:

| Header | Types moved | Approx lines |
|---|---|---|
| `src/exec/expert_cache.h` (new) | `ExpertCacheKey`, `ExpertCacheKeyHash`, `ExpertProj` + `kExpertProjCount`, `PerLayerLRU`, `PerLayerAccessRing`, `ExpertLRUCache` (incl. `Slot`) | ~196 |
| `src/exec/weight_caches.h` (new) | `FP8CacheEntry`, `WeightCaches` (incl. `Q4kImmaCacheEntry`) | ~105 |
| `src/exec/inference_state.h` (new) | `InferenceState` | ~143 |
| `src/exec/moe_ffn_context.h` (new) | `MoeFfnContext` | ~49 |
| `src/runtime/vram_budget.h` (existing) | `VRAMBudget` struct (moved INTO it, alongside `compute_vram_budget`) | ~12 |

After the move:
- `executor.h` `#include`s the four new `exec/*.h` headers and
  `runtime/vram_budget.h`, then declares only `GraphExecutor` (plus its private
  inner `SavedWorkspace`, which stays — it is a private member type).
- `expert_cache.cu` `#include`s `exec/expert_cache.h` directly.
- `runtime/vram_budget.h` DROPS its `#include "exec/executor.h"` (it now defines
  `VRAMBudget` itself) — removing a god-object include and the
  `executor.h → vram_budget.h → executor.h` near-cycle.

Each new header is self-contained: `#pragma once`, the minimal includes the moved
types need (copied from `executor.h`'s current include set — e.g. `core/tensor.h`,
`<cuda_fp16.h>`, `<unordered_map>`, `<list>`, the `nvfp4`/`cutlass` weight types
for `weight_caches.h`, and forward declarations such as `class VRAMAllocator;`).

## Data flow / consumers

No consumer changes required in this step. Every `.cu`/`.cpp` that uses these
types includes `executor.h`, which re-includes the new headers. (A future cleanup
could switch consumers to include only the focused header they need — explicitly
OUT OF SCOPE here to keep the diff declaration-only.)

## Error handling

Not applicable — no runtime code changes.

## Testing

- `make build` succeeds (the build is the real check: it catches any missing
  include or forward declaration in a new header, and any include cycle).
- `make verify-fast` gtest filter green.
- Coherence smoke: Qwen3-8B Q8_0 (dense) and Qwen3-30B-A3B-NVFP4 (MoE) each
  decode coherently (Paris). Logic cannot change, so this is a backstop, not a
  behaviour gate.

## Out of scope

- Moving consumers to include the focused headers directly (keeps this diff to
  pure relocation; `executor.h` stays the umbrella include).
- Extracting any runner component (Workspace, MoeRunner, Attention/Ffn/Ssm) —
  those touch the hot path and are separate steps.
- Changing any struct's contents, member order, or methods.

## Risks

- **A moved struct needs an include the new header lacks.** Caught at compile
  time; fix by adding the include (copied from executor.h's set).
- **Include cycle** (e.g. `weight_caches.h` needing a type that includes
  `executor.h`). Mitigation: new headers depend only on leaf types
  (`core/tensor.h`, quant weight headers, std) + forward declarations; none
  include `executor.h`. The `vram_budget.h` change specifically REMOVES a cycle.
- **Member reordering changing struct layout.** Avoided by construction — structs
  move verbatim, contents untouched.
