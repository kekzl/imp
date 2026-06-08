# Workspace Component Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the shared/persistent/decode scratch arena out of `GraphExecutor` into a `Workspace` class, with the forward hot path reading buffers through zero-overhead inline accessors.

**Architecture:** `GraphExecutor` owns `Workspace ws_;`, inits it once (`ws_.init(model, alloc, compute_dtype, max_tokens, use_pdl, moe)`), and the orchestrator `allocate_workspaces` calls `ws_.allocate_*`. The forward path reads `ws_.shared()` / calls `ws_.configure_<phase>(n)` instead of the former members. The cross-cutting buffer-allocation hub (`allocate_auxiliary_buffers`: qscratch/moe/attn/fp32_accum) STAYS on GraphExecutor. NOT `diff=0` — the forward TUs change; canary-gated.

**Tech Stack:** C++20/CUDA, Docker build, GTest, 4-arch canary + decode-perf backstop.

Spec: `docs/superpowers/specs/2026-06-08-workspace-component-design.md`. Branch: `feat/d2-workspace` (spec committed).

---

## File structure

- **Create** `src/exec/workspace.h` — the `Workspace` class (members + init-context pointers + inline accessors + moved-method declarations).
- **Modify** `src/exec/executor.h` — `#include "exec/workspace.h"`; add member `Workspace ws_;`; REMOVE the moved members (shared/persistent workspace + sizes + decode-swap); keep `active_workspace()` public getter as a delegate; keep the hub buffers (`qscratch_`/`moe_`/`attn_scores_buf_`/`fp32_accum_buf_`/`nvfp4_dequant_ws_buf_`).
- **Modify** `src/exec/executor_workspace.cu` — `compute_shared_sizes`, `allocate_persistent_workspace`, `allocate_shared_workspace`, `allocate_decode_workspace`, `workspace_estimate` become `Workspace::` methods; `init` + `allocate_workspaces` stay `GraphExecutor::` (the orchestrator calls `ws_.allocate_*`).
- **Modify** `src/exec/executor_workspace_config.cu` — `configure_attn/ffn/moe/ssm_workspace`, `use_workspace`, `resize_workspace` become `Workspace::`; `view_tokens`, `layer_has_*`, `ensure_logits_pinned` stay `GraphExecutor::`.
- **Leave** `src/exec/executor_workspace_buffers.cu` entirely on GraphExecutor (the hub).
- **Modify** the hot-path TUs (`executor_attention.cu`, `executor_ffn.cu`, `executor_forward.cu`, `executor_forward_moe*.cu`, `executor_ssm_gdn.cu`) — ~40 call sites `shared_workspace_`→`ws_.shared()`, `configure_*_workspace(n)`→`ws_.configure_*(n)`, etc.

Workspace init-context (pointers set in `init`, mirroring QuantPipeline): `const Model* model_`, `VRAMAllocator* vram_alloc_`, `QType compute_dtype_`, `int max_tokens_`, `bool use_pdl_`, `const MoEWorkspace* moe_` (read for moe-workspace sizing only).

Members moving into `Workspace`: `shared_workspace_`(+`_size_`,+`_max_tokens_`), `persistent_workspace_`(+`_size_`), `attn_shared_size_`, `ffn_shared_size_`, `moe_shared_size_`, `ssm_shared_size_`, `decode_persistent_size_`, `decode_shared_size_`, `decode_workspace_`, `decode_shared_workspace_`, `active_workspace_`, `saved_prefill_ws_` (+ its `SavedWorkspace` struct decl).

---

### Task 1: Create `workspace.h` with the class skeleton

**Files:** Create `src/exec/workspace.h`; back up: `cp src/exec/executor.h /tmp/exec_ws_bak.h`.

- [ ] **Step 1: Read the current members + method signatures to copy**

Run:
```bash
grep -nE "SavedWorkspace|shared_workspace_|persistent_workspace_|attn_shared_size_|ffn_shared_size_|moe_shared_size_|ssm_shared_size_|decode_persistent_size_|decode_shared_size_|decode_workspace_|decode_shared_workspace_|active_workspace_|saved_prefill_ws_" src/exec/executor.h
grep -nE "compute_shared_sizes|allocate_persistent_workspace|allocate_shared_workspace|allocate_decode_workspace|workspace_estimate|configure_attn_workspace|configure_ffn_workspace|configure_moe_workspace|configure_ssm_workspace|use_workspace|resize_workspace" src/exec/executor.h
```
Copy the exact signatures + the `SavedWorkspace` struct (currently inner to GraphExecutor) for the header.

- [ ] **Step 2: Write `workspace.h`**

```cpp
#pragma once

#include "core/tensor.h"
#include "core/qtype.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

class Model;
class VRAMAllocator;
struct MoEWorkspace;

// The shared/persistent/decode scratch arena, extracted from GraphExecutor (D2).
// Owns the forward-pass scratch buffers + the decode/prefill workspace swap; the
// cross-cutting auxiliary-buffer hub (qscratch/moe/attn) stays on GraphExecutor.
// See docs/superpowers/specs/2026-06-08-workspace-component-design.md.
class Workspace {
public:
    // Set the build context once (mirrors QuantPipeline's pointer-context).
    void init(const Model& model, VRAMAllocator& alloc, QType compute_dtype,
              int max_tokens, bool use_pdl, const MoEWorkspace& moe) {
        model_ = &model; vram_alloc_ = &alloc; compute_dtype_ = compute_dtype;
        max_tokens_ = max_tokens; use_pdl_ = use_pdl; moe_ = &moe;
    }

    // --- zero-overhead inline accessors (hot path reads these) ---
    void* shared() const { return shared_workspace_; }
    void* persistent() const { return persistent_workspace_; }
    int shared_max_tokens() const { return shared_workspace_max_tokens_; }
    int active() const { return active_workspace_; }

    // --- moved lifecycle methods (paste exact signatures from executor.h) ---
    // bool allocate_persistent_workspace(int max_tokens);
    // bool allocate_shared_workspace(int max_tokens);
    // bool allocate_decode_workspace(...);
    // void compute_shared_sizes(int max_tokens);
    // size_t workspace_estimate() const;
    // void configure_attn_workspace(int n);  (+ ffn/moe/ssm)
    // void use_workspace(...);  void resize_workspace(...);

private:
    const Model* model_ = nullptr;
    VRAMAllocator* vram_alloc_ = nullptr;
    QType compute_dtype_ = QType::F16;
    int max_tokens_ = 0;
    bool use_pdl_ = false;
    const MoEWorkspace* moe_ = nullptr;

    // <-- paste SavedWorkspace struct + the moved members here verbatim -->
};

}  // namespace imp
```
Fill the method decls + members verbatim from executor.h. Match include needs to the moved members (e.g. `SavedWorkspace` holds `Tensor` members → `core/tensor.h`).

- [ ] **Step 3: Commit the skeleton**

```bash
git add src/exec/workspace.h
git commit -m "feat(exec): Workspace class skeleton (D2 component 3)"
```

---

### Task 2: Move the lifecycle method definitions onto `Workspace`

**Files:** Modify `src/exec/executor_workspace.cu`, `src/exec/executor_workspace_config.cu`.

- [ ] **Step 1: Add include + reclass the moved methods**

```bash
for f in src/exec/executor_workspace.cu src/exec/executor_workspace_config.cu; do
  grep -q 'exec/workspace.h' "$f" || sed -i '0,/#include "exec\/executor.h"/s//#include "exec\/executor.h"\n#include "exec\/workspace.h"/' "$f"
done
# executor_workspace.cu: reclass ONLY the 5 arena methods (NOT init / allocate_workspaces)
perl -0pi -e 's/\bGraphExecutor::(compute_shared_sizes|allocate_persistent_workspace|allocate_shared_workspace|allocate_decode_workspace|workspace_estimate)\b/Workspace::$1/g' src/exec/executor_workspace.cu
# config.cu: reclass configure_* + use_workspace + resize_workspace (NOT view_tokens / layer_has_* / ensure_logits_pinned)
perl -0pi -e 's/\bGraphExecutor::(configure_attn_workspace|configure_ffn_workspace|configure_moe_workspace|configure_ssm_workspace|use_workspace|resize_workspace)\b/Workspace::$1/g' src/exec/executor_workspace_config.cu
```

- [ ] **Step 2: Fix member access in the moved bodies**

The moved methods reference the context members (`model_`, `vram_alloc_`, `compute_dtype_`, `max_tokens_`, `use_pdl_`, `moe_`) — those are now `Workspace` members with the SAME names, set in `init()`, so they resolve unchanged. `model_->`/`vram_alloc_` are pointers (unchanged). `moe_` is now a `const MoEWorkspace*` → its access changes `moe_.` → `moe_->` (sizing reads only):
```bash
perl -0pi -e 's/\bmoe_\./moe_->/g' src/exec/executor_workspace_config.cu
```
(In `executor_workspace.cu` the arena methods don't touch `moe_`; confirm with `grep -n "moe_" src/exec/executor_workspace.cu` — if a moved method does, apply the same `.`→`->`.)

- [ ] **Step 3: Commit (won't build until Task 3 wires GraphExecutor)**

```bash
git add src/exec/executor_workspace.cu src/exec/executor_workspace_config.cu
git commit -m "refactor(exec): reclass scratch-arena methods onto Workspace"
```

---

### Task 3: Wire `GraphExecutor` to own `Workspace` + delegate

**Files:** Modify `src/exec/executor.h`, `src/exec/executor_workspace.cu`.

- [ ] **Step 1: executor.h — include, member, remove moved decls/members**

- Add `#include "exec/workspace.h"`.
- Add member `Workspace ws_;` (near where the workspace members were).
- REMOVE the moved member declarations (shared/persistent workspace + sizes + decode-swap + `SavedWorkspace` struct) and the moved method declarations.
- Change the public `active_workspace()` getter to `return ws_.active();`.
- KEEP `qscratch_`, `moe_`, `attn_scores_buf_`(+size), `fp32_accum_buf_`, `nvfp4_dequant_ws_buf_`(+size) and the hub method decls (`allocate_auxiliary_buffers`, `free_buffers`, `release_moe_batch_buf`, `ensure_logits_pinned`, `view_tokens`, `layer_has_*`).

- [ ] **Step 2: executor_workspace.cu — init + allocate_workspaces call ws_**

In `GraphExecutor::init`, after the model is set and `max_tokens_`/`compute_dtype_`/`use_pdl_` are known, add `ws_.init(*model_, *vram_alloc_, compute_dtype_, max_tokens_, use_pdl_, moe_);` (place it where the workspace members were first valid — right before the first `allocate_*` call).

In `GraphExecutor::allocate_workspaces`, replace the calls to the moved methods with `ws_.compute_shared_sizes(...)`, `ws_.allocate_persistent_workspace(...)`, `ws_.allocate_shared_workspace(...)`, `ws_.allocate_decode_workspace(...)` as they appear.

- [ ] **Step 3: Commit**

```bash
git add src/exec/executor.h src/exec/executor_workspace.cu
git commit -m "refactor(exec): GraphExecutor owns + delegates to Workspace"
```

---

### Task 4: Migrate the hot-path call sites (~40)

**Files:** Modify `executor_attention.cu`, `executor_ffn.cu`, `executor_forward.cu`, `executor_forward_moe.cu`, `executor_forward_moe_batch.cu`, `executor_ssm_gdn.cu` (and any TU the build flags).

- [ ] **Step 1: Migrate buffer reads + configure calls**

```bash
HOT="src/exec/executor_attention.cu src/exec/executor_ffn.cu src/exec/executor_forward.cu src/exec/executor_forward_moe.cu src/exec/executor_forward_moe_batch.cu src/exec/executor_ssm_gdn.cu"
for f in $HOT; do
  perl -0pi -e 's/\bshared_workspace_max_tokens_\b/ws_.shared_max_tokens()/g;
                s/\bshared_workspace_\b(?!\w)/ws_.shared()/g;
                s/\bpersistent_workspace_\b(?!\w)/ws_.persistent()/g;
                s/\bconfigure_attn_workspace\(/ws_.configure_attn_workspace(/g;
                s/\bconfigure_ffn_workspace\(/ws_.configure_ffn_workspace(/g;
                s/\bconfigure_moe_workspace\(/ws_.configure_moe_workspace(/g;
                s/\bconfigure_ssm_workspace\(/ws_.configure_ssm_workspace(/g;
                s/\buse_workspace\(/ws_.use_workspace(/g;
                s/\bresize_workspace\(/ws_.resize_workspace(/g' "$f"
done
```
Caveat: `shared_workspace_size_` (the size member) — if the hot path reads it, add an accessor `size_t shared_size() const` to Workspace and migrate `shared_workspace_size_`→`ws_.shared_size()`. The `(?!\w)` guards stop `shared_workspace_` from eating `shared_workspace_size_`/`_max_tokens_`; do those FIRST (the script orders max_tokens before the bare name). Re-grep after: `grep -rn "shared_workspace_\|persistent_workspace_" $HOT` should be empty.

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "refactor(exec): hot-path reads workspace via ws_ accessors"
```

---

### Task 5: Build and fix

- [ ] **Step 1: Build**

Run: `make build 2>&1 | grep -iE "error:" | head -40`
Likely errors + fixes:
- A moved method reads a GraphExecutor member not in the init-context → add it to `Workspace::init` params + a member, pass from `GraphExecutor::init`. (Re-confirm it's read-only for sizing, no hot-path writer.)
- A hot-path TU still reads a workspace member the perl missed, or `shared_workspace_size_` → add the accessor and migrate.
- An accessor name mismatch (`ws_.configure_attn_workspace` vs the method name) → make method names match what the perl produced.
Iterate to clean.

- [ ] **Step 2: Commit fixes**

```bash
git add -A && git commit -m "fix(exec): thread Workspace context + remaining accessors"
```

---

### Task 6: Behaviour + perf gate

- [ ] **Step 1: 4-arch coherence canary**

```bash
for M in "Qwen3-8B-Q8_0.gguf|dense" "Qwen3-30B-A3B-NVFP4-Modelopt|moe" \
         "Nemotron-3-Nano-30B-A3B-NVFP4|ssm" "gemma-3-12b-it-Q4_K_M.gguf|gemma3"; do
  MODEL="${M%%|*}"
  docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
    imp-cli --model "/models/$MODEL" --prompt "What is the capital of France? One word." \
    --max-tokens 200 --temperature 0 --seed 42 >/tmp/ws.out 2>&1
  echo "== $MODEL =="; grep -aoiE "paris" /tmp/ws.out | head -1
  grep -acE "NVFP4 MoE native: data-borrow decode cache" /tmp/ws.out
  grep -aiE "CUDA error|falling back|\bNaN\b|illegal memory" /tmp/ws.out | head -1
done
```
Expected: each `Paris`, no error/NaN/IMA, Qwen3-30B native-cache count = **144**, Nemotron = **46**.

- [ ] **Step 2: Decode-perf backstop (workspace mis-sizing is silent)**

```bash
docker run --rm --gpus all -v /home/kekz/models:/models imp:test bash -lc '
  imp-cli --model /models/Qwen3-8B-Q8_0.gguf --prompt Hi --max-tokens 64 >/dev/null 2>&1
  imp-cli --model /models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 512 --bench-reps 10 2>&1 | grep -iE "tg "'
```
Expected: `tg128`/`tg256` within ~5% of the main baseline (Q8 tg≈268-278). A >5% drop or an OOM/`auto-disable` log = a mis-sized workspace; investigate before proceeding.

- [ ] **Step 3: verify-fast**

Run: `IMP_VERIFY_SKIP_BUILD=1 make verify-fast 2>&1 | grep -E "PASS|FAIL|OK ==="`
Expected: PASS + OK.

- [ ] **Step 4: Coherence battery (hot path changed — run the degen check)**

Run: `make test-gpu GTEST_FILTER="DegenerationTest.*" 2>&1 | tail -5`
Expected: all pass.

---

### Task 7: Push + PR

- [ ] Push `feat/d2-workspace`, open PR vs main, summary: what moved (the scratch arena), what stayed (the alloc hub + fp32_accum/attn_scores), the ~40-site hot-path accessor churn, canary + perf-backstop + verify-fast green. Watch CI; merge when green.

---

## Self-review notes (controller)

- **Spec coverage:** Tasks 1-3 = class + move + wire; Task 4 = the ~40 hot-path sites; Tasks 5-6 = build + canary + perf backstop + degen battery; Task 7 = PR. The refined-scope boundary (hub stays) is encoded in which methods Task 2 reclasses.
- **No `diff=0` task** — the spec explicitly drops it (hot path changes).
- **Type consistency:** `ws_` accessor names (`shared()`, `persistent()`, `shared_max_tokens()`, `active()`) + the `configure_*_workspace`/`use_workspace`/`resize_workspace` method names are used consistently across Tasks 1, 3, 4.
- **Gotcha:** the perl `(?!\w)` guard prevents `shared_workspace_` from clobbering `shared_workspace_size_`/`_max_tokens_`; if `shared_workspace_size_` is read in the hot path, add a `shared_size()` accessor (Task 4 Step 1 caveat). Keep `/tmp/*.bak`; plain substitutions only (no `do{<>}`).
- **Risk:** workspace sizing is silent on failure — the perf backstop + MoE/SSM canary models are load-bearing, not optional.
