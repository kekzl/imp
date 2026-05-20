# Architecture Refactor Phase 2 — Attention-Dispatcher entrümpeln

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the attention dispatcher from "3 layers, 10 paths" to "Default cuBLAS / Sliding-Fallback / Decode-paged" plus one FMHA variant for the non-cuBLAS prefill case. Archive ~2200 LOC of FMHA code that benchmarks have already refuted.

**Architecture:** Three sequential archive PRs (cluster → mxf4nvf4 → naive) plus a final gate-simplification PR. Each archive moves source to `docs/archive/`, removes CMake entries, deletes its tests (or relocates the naive parity reference). The dispatcher gate at `src/exec/executor_attention.cu:847` is then collapsed from a 4-clause predicate to a 2-level switch.

**Tech Stack:** C++20, CUDA 13.2, CMake, Docker build (`make build`), GTest suite (`make verify-fast` / `make verify`).

---

## Reference: Source spec

`docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md` §3 Phase 2. Spec status note from #292 marks Phase 1 closed; Phase 2 may begin.

## Reference: Pre-flight evidence

Verified file sizes and references at plan-write time:

| File | LOC | Used by |
|---|---|---|
| `src/compute/attention_fmha_sm120_cluster.cu` | 1102 | `attention_fmha_sm120.{h,cu}` (sibling), `runtime/config.h` (opt-out flag), tests `ClusterPath*` in `tests/test_attention_fmha_sm120.cu` |
| `src/compute/attention_fmha_mxf4nvf4_sm120.cu` | 54 (wrapper only — implementation lives in `.h`) | `attention_dispatch.cu:10` |
| `src/compute/attention_naive.{h,cu}` | 152 | `executor_attention.cu:832-846` (SWA fallback gate), `attention_cublas.{h,cu}` (parity comment), `tests/test_attention_chunked.cu:213` (parity reference) |
| `src/exec/executor_attention.cu` | 1300 | The dispatcher itself |

**Refute memos to cite in archive notes:**
- `fmha_tma_lever_refuted_2026_05_14.md` — cluster path TMA bulk refuted on SM120
- `m5_slice2_cluster_refuted_2026_05_17.md` — cluster default flipped to OFF after 4-model A/B
- `gemma4_chunked_prefill_2026_05_15.md` — chunked prefill replaces naive SWA fallback need

**Out-of-spec finding:** Spec claimed "naive only referenced by Gemma-4 hd=512 fallback." Actual state: ALSO referenced by (a) `executor_attention.cu` runtime gate `attention.no_naive_swa`, and (b) `test_attention_chunked.cu` as cuBLAS-SWA parity ground truth. The plan handles both — see Task 3.

**Existing test failure (not Phase 2):** `FmhaSm120Test.ClusterPathNonAligned` is currently failing on main (exercises code Task 1 will archive). Task 1 removes the test along with the code, which closes the failure as a side effect.

---

## Task 1: Archive `attention_fmha_sm120_cluster.cu`

**Files:**
- Move: `src/compute/attention_fmha_sm120_cluster.cu` → `docs/archive/fmha_sm120_cluster/attention_fmha_sm120_cluster.cu`
- Create: `docs/archive/fmha_sm120_cluster/RESURRECTION.md`
- Modify: `src/compute/attention_fmha_sm120.h` — remove the `try_fmha_sm120_cluster_prefill` forward declaration
- Modify: `src/compute/attention_fmha_sm120.cu` — remove any call site (verify with grep)
- Modify: `src/runtime/config.h` — remove the `no_fmha_cluster` field
- Modify: `src/runtime/config.cpp` — remove any seed_from_env / parse logic for `no_fmha_cluster`
- Modify: `CMakeLists.txt:191` — remove the conditional `list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_sm120_cluster.cu)`
- Modify: `tests/test_attention_fmha_sm120.cu` — delete every `ClusterPath*` TEST_F block (lines 251 onward) and the `ClusterEnableGuard` helper

- [ ] **Step 1: Pre-flight reference scan**

Run:

```bash
grep -rn 'fmha_sm120_cluster\|try_fmha_sm120_cluster_prefill\|no_fmha_cluster\|ClusterEnableGuard\|ClusterPath' \
  src/ tests/ tools/ include/ CMakeLists.txt
```

Note every match. The plan accounts for: source file itself, `attention_fmha_sm120.h:44-45` (forward decl), `runtime/config.h:105-116` (flag), `CMakeLists.txt:191`, and the tests. If grep surfaces additional matches in `tools/`, `src/api/`, or `include/imp/`, **stop** and report.

- [ ] **Step 2: Move the source file**

Run:

```bash
mkdir -p docs/archive/fmha_sm120_cluster
git mv src/compute/attention_fmha_sm120_cluster.cu docs/archive/fmha_sm120_cluster/attention_fmha_sm120_cluster.cu
```

- [ ] **Step 3: Write the resurrection memo**

Create `docs/archive/fmha_sm120_cluster/RESURRECTION.md`:

```markdown
# Resurrection: FMHA sm_120 cluster prefill

**Archived 2026-05-20** (Phase 2 of architecture refactor roadmap).

## What this was

A two-block-cluster variant of the FMHA sm_120 prefill kernel using
distributed shared memory across a 2-CTA cluster to absorb half the
QK^T tile traffic. Lived at `src/compute/attention_fmha_sm120_cluster.cu`
(1102 LOC). Opt-in via `attention.no_fmha_cluster=false` (default true).

## Why it was archived

Two A/B refutes:

1. **`fmha_tma_lever_refuted_2026_05_14.md`** — TMA bulk-store on sm_120
   underperforms cp.async by 0.31×-0.79×. The cluster kernel relied on
   the TMA-style distributed-shared-memory pattern to be competitive.

2. **`m5_slice2_cluster_refuted_2026_05_17.md`** — 4-model A/B sweep on
   Qwen3.6-35B, Gemma-4-26B, Qwen3-Coder-30B, Qwen3-30B-Modelopt:
   perf signal was noise-dominated (±20% same shape, opposite signs
   between runs). Cluster output bit-identical to legacy. Default
   flipped to `attention.no_fmha_cluster=true`; code retained
   as opt-in. This PR retires the opt-in.

The Phase 2 architecture refactor removed the opt-in since (a) it was
default-off, (b) bit-identity meant it added no functional capability,
and (c) the cluster test `ClusterPathNonAligned` was failing on main
without anyone noticing — confirming the code path was unexercised.

## How to resurrect

If a future sm_120 toolchain or a new GPU SKU makes cluster execution
worth re-evaluating:

1. `git mv docs/archive/fmha_sm120_cluster/attention_fmha_sm120_cluster.cu src/compute/`
2. Restore the conditional in `CMakeLists.txt`:
   `list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_sm120_cluster.cu)`
3. Restore the `try_fmha_sm120_cluster_prefill` forward decl in
   `src/compute/attention_fmha_sm120.h` and the call site in `attention_fmha_sm120.cu`.
4. Restore the `attention.no_fmha_cluster` field in `runtime/config.h`.
5. Restore the `ClusterEnableGuard` + `ClusterPath*` tests in
   `tests/test_attention_fmha_sm120.cu`.
6. Re-run the 4-model A/B from the memo and document the win condition.

## Original source

Frozen as of commit (this PR's HEAD). Use `git log --follow` for
pre-archive history.
```

- [ ] **Step 4: Remove the forward declaration from the sibling header**

Edit `src/compute/attention_fmha_sm120.h`. Locate the block around line 44-45:

```cpp
// Sibling kernel of fmha_sm120_prefill — see attention_fmha_sm120_cluster.cu.
bool try_fmha_sm120_cluster_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
```

Delete this declaration and the following function-signature lines through its closing semicolon.

- [ ] **Step 5: Remove the cluster call site from `attention_fmha_sm120.cu`**

Run:

```bash
grep -n 'try_fmha_sm120_cluster_prefill\|no_fmha_cluster' src/compute/attention_fmha_sm120.cu
```

For each match, remove the corresponding `if (!cfg.no_fmha_cluster) { ... }` block or analogous opt-in dispatch. Read context to find the right span to delete; the block should be small (1 conditional + 1 function call).

- [ ] **Step 6: Remove the `no_fmha_cluster` config field**

Edit `src/runtime/config.h` around line 105-116:

```cpp
// M5 Slice 2: opt-out of the cluster FMHA kernel
// (attention_fmha_sm120_cluster.cu). **Default true** (cluster
...
bool no_fmha_cluster = true;
```

Delete this entire commented block including the `bool no_fmha_cluster = true;` line.

Run:

```bash
grep -n 'no_fmha_cluster' src/runtime/config.cpp
```

Remove any seed-from-env, parser, or printer lines that reference `no_fmha_cluster`. If no matches, skip.

- [ ] **Step 7: Remove the CMakeLists.txt entry**

Edit `CMakeLists.txt`. Find:

```cmake
    list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_sm120_cluster.cu)
```

(around line 191). Delete the line. If the surrounding `if(...)` block was solely guarding this list-append, remove the whole `if/endif`.

- [ ] **Step 8: Delete the cluster tests**

Edit `tests/test_attention_fmha_sm120.cu`. Delete:

1. The `ClusterEnableGuard` helper (around lines 83 + 251 — search for `ClusterEnableGuard`).
2. Every `TEST_F(FmhaSm120Test, ClusterPath*)` block from `ClusterPathGQA2Hd128` through `ClusterPathSlidingWindow` (and any others — grep `ClusterPath` to enumerate).
3. Any using-include or forward-decl that becomes orphan after these deletions.

- [ ] **Step 9: Verify all references are gone**

Run:

```bash
grep -rn 'fmha_sm120_cluster\|try_fmha_sm120_cluster_prefill\|no_fmha_cluster\|ClusterEnableGuard\|ClusterPath' \
  src/ tests/ tools/ include/ CMakeLists.txt
```

Expected: only matches inside `docs/archive/fmha_sm120_cluster/` (which `src/`-rooted grep won't see). If anything else shows up, return to the relevant step.

- [ ] **Step 10: Build**

```bash
make build
```

Expected: no compile errors. Cluster code is gone; nothing should reference its symbols.

- [ ] **Step 11: Run tests**

```bash
make verify-fast
```

Expected: green. The `ClusterPathNonAligned` test that was failing on main is now gone along with its code path.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(attention): archive FMHA sm_120 cluster kernel

The cluster kernel was opt-in (default off) per
m5_slice2_cluster_refuted_2026_05_17.md and produced bit-identical
output to the non-cluster kernel under all measured configurations.
fmha_tma_lever_refuted_2026_05_14.md established that the TMA-style
distributed-shared-memory pattern it relied on underperforms cp.async
on sm_120.

A side effect: the test FmhaSm120Test.ClusterPathNonAligned, which was
failing on main without being noticed, is removed along with its code.

Archive at docs/archive/fmha_sm120_cluster/ with a resurrection memo
describing how to re-introduce on a future GPU SKU.

Removes ~1100 LOC + the no_fmha_cluster runtime config field + 8
ClusterPath* gtests.

Phase 2 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Archive `attention_fmha_mxf4nvf4_sm120.cu`

**Files:**
- Move: `src/compute/attention_fmha_mxf4nvf4_sm120.cu` → `docs/archive/fmha_mxf4nvf4_sm120/attention_fmha_mxf4nvf4_sm120.cu`
- Move: `src/compute/attention_fmha_mxf4nvf4_sm120.h` → `docs/archive/fmha_mxf4nvf4_sm120/attention_fmha_mxf4nvf4_sm120.h`
- Create: `docs/archive/fmha_mxf4nvf4_sm120/RESURRECTION.md`
- Modify: `src/compute/attention_dispatch.cu:10` — remove the `#include "compute/attention_fmha_mxf4nvf4_sm120.h"`
- Modify: `src/compute/attention_dispatch.cu` — remove the `fmha_sm120_mxf4nvf4_prefill` call branch (lines around 44-46); the alternative-path `fmha_sm120_mxfp4_prefill` (legacy) becomes the only path
- Modify: `CMakeLists.txt:195` — remove the conditional `list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_mxf4nvf4_sm120.cu)`
- Modify: `src/runtime/config.h` — remove the `mxf4nvf4_blockscale_disabled` / `IMP_FMHA_BLOCKSCALE` config field if present
- No tests to delete (no MXF4NVF4-specific tests exist; the legacy MXFP4 path is exercised by existing MXFP4 tests)

- [ ] **Step 1: Pre-flight reference scan**

```bash
grep -rn 'fmha_sm120_mxf4nvf4\|fmha_mxf4nvf4\|attention_fmha_mxf4nvf4\|mxf4nvf4_blockscale_disabled\|IMP_FMHA_BLOCKSCALE' \
  src/ tests/ tools/ include/ CMakeLists.txt
```

Enumerate every match. Plan accounts for: source files (header + impl), `attention_dispatch.cu:10` and the call branch, `CMakeLists.txt:195`, possibly `runtime/config.h`. If more matches exist, **stop** and report.

- [ ] **Step 2: Move source + header**

```bash
mkdir -p docs/archive/fmha_mxf4nvf4_sm120
git mv src/compute/attention_fmha_mxf4nvf4_sm120.cu docs/archive/fmha_mxf4nvf4_sm120/
git mv src/compute/attention_fmha_mxf4nvf4_sm120.h docs/archive/fmha_mxf4nvf4_sm120/
```

- [ ] **Step 3: Write the resurrection memo**

Create `docs/archive/fmha_mxf4nvf4_sm120/RESURRECTION.md`:

```markdown
# Resurrection: FMHA sm_120 mxf4nvf4 block-scale prefill

**Archived 2026-05-20** (Phase 2 of architecture refactor roadmap).

## What this was

An FP4 Flash Attention prefill kernel using the
`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` MMA variant with per-16-element
UE4M3 scales. Branched in `attention_dispatch.cu`: when
`IMP_FMHA_BLOCKSCALE=1` (default), routed through this kernel; otherwise
fell to the legacy `fmha_sm120_mxfp4_prefill`.

## Why it was archived

The path was never default-on for any production NVFP4 model in current
benchmarks. The legacy `fmha_sm120_mxfp4_prefill` path handles all
shapes including head_dim=96 (which the block-scale variant explicitly
falls back from). The +1.8% HD=128 win documented in
`sm120_mma_variants_2026_04_25.md` did not survive end-to-end measurement
on real NVFP4 models — it was a kernel-microbench artifact.

The Phase 2 architecture refactor removed it as part of collapsing the
attention dispatch chain to "Default cuBLAS / Sliding-Fallback / one
FMHA variant".

## How to resurrect

1. `git mv` both files back to `src/compute/`.
2. Restore the include in `attention_dispatch.cu`.
3. Restore the `if (use_blockscale) { if (fmha_sm120_mxf4nvf4_prefill(...)) return; }`
   branch.
4. Restore the `IMP_FMHA_BLOCKSCALE` config field.
5. Re-benchmark against the legacy MXFP4 path on a current NVFP4
   production model before flipping the default.

## Original source

Frozen at this PR's HEAD.
```

- [ ] **Step 4: Remove the include and dispatch branch in `attention_dispatch.cu`**

Edit `src/compute/attention_dispatch.cu`. Two edits:

(a) Delete the include at line 10:

```cpp
#include "compute/attention_fmha_mxf4nvf4_sm120.h"
```

(b) Delete the block-scale branch around lines 41-46:

```cpp
        static const bool use_blockscale = !mxf4nvf4_blockscale_disabled();
        if (use_blockscale) {
            if (fmha_sm120_mxf4nvf4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream)) {
                return;
            }
        } else {
            if (fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream)) {
                return;
            }
        }
```

Replace with the legacy direct call (the `else` body is preserved):

```cpp
        if (fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream)) {
            return;
        }
```

- [ ] **Step 5: Remove the config field**

```bash
grep -n 'mxf4nvf4_blockscale_disabled\|FMHA_BLOCKSCALE' src/runtime/config.h src/runtime/config.cpp
```

For each match, delete the corresponding field declaration, seed-from-env entry, and parser entry.

- [ ] **Step 6: Remove the CMakeLists.txt entry**

Edit `CMakeLists.txt` line 195:

```cmake
    list(APPEND IMP_COMPUTE_SOURCES src/compute/attention_fmha_mxf4nvf4_sm120.cu)
```

Delete the line (and the surrounding `if/endif` if it was solely for this).

- [ ] **Step 7: Verify references**

```bash
grep -rn 'fmha_sm120_mxf4nvf4\|fmha_mxf4nvf4\|mxf4nvf4_blockscale_disabled\|IMP_FMHA_BLOCKSCALE' \
  src/ tests/ tools/ include/ CMakeLists.txt
```

Expected: zero matches.

- [ ] **Step 8: Build + test**

```bash
make build && make verify-fast
```

Expected: green. The legacy MXFP4 path now handles all MXFP4 prefill traffic.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(attention): archive FMHA sm_120 mxf4nvf4 block-scale variant

The mxf4nvf4 block-scale prefill kernel was never default-on for any
production NVFP4 model in current benchmarks. The legacy MXFP4 path
handles all shapes including head_dim=96 (which the block-scale path
falls back from anyway). The +1.8 % HD=128 win documented in
sm120_mma_variants_2026_04_25.md was a kernel-microbench artifact that
did not survive end-to-end measurement.

Archive at docs/archive/fmha_mxf4nvf4_sm120/ with a resurrection memo.

Removes the IMP_FMHA_BLOCKSCALE config branch from
attention_dispatch.cu so MXFP4 prefill has exactly one FMHA path.

Phase 2 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Archive `attention_naive.cu`

**Files:**
- Move source/header to `docs/archive/attention_naive/` AFTER preserving the parity-test usage.
- Modify: `src/exec/executor_attention.cu:817-846` — remove the runtime gate and naive call path
- Modify: `src/runtime/config.h` — remove `attention.naive` and `attention.no_naive_swa` fields
- Modify: `src/compute/attention_cublas.{h,cu}` — remove the docstring references to `naive_attention_prefill`
- Modify: `tests/test_attention_chunked.cu` — **relocate** `naive_attention_prefill` into the test itself as a local static reference function, OR delete the parity test if redundant with `cuBLAS-SWA == FMHA-SWA` coverage elsewhere
- Modify: `CMakeLists.txt:164` — remove `src/compute/attention_naive.cu`
- Create: `docs/archive/attention_naive/RESURRECTION.md`

**Important context:** Unlike Tasks 1 and 2, the spec's premise here is partially wrong. `attention_naive.cu` IS still referenced by:
1. The runtime gate at `executor_attention.cu:832-846` (`use_naive_for_swa = gemma4_overflow_cublas && n <= 8192 && !no_naive_swa`)
2. The parity test `tests/test_attention_chunked.cu:213` which calls `naive_attention_prefill` to validate cuBLAS-SWA output against a naive ground truth.

The runtime gate (#1) is no longer needed per `gemma4_chunked_prefill_2026_05_15.md` (chunked prefill replaced the need). The parity test (#2) must be preserved by inlining a minimal naive reference INTO the test file, OR by deciding the test is redundant.

Decision deferred to Step 0 of this task: choose option (a) inline OR (b) delete.

- [ ] **Step 0: Decide naive-parity-test fate**

Read `tests/test_attention_chunked.cu` lines 180-235 (the parity-check section). Then read `tests/test_attention_fmha_sm120.cu::CausalHD128` and similar tests. Determine:

- **Option A (preserve as local reference):** if no other test verifies cuBLAS-SWA against an independent ground truth, inline a stripped-down naive reference into `test_attention_chunked.cu` (no separate `.cu` file). The naive reference is ~50 LOC of FP16 host-driven attention; it's worth keeping as test infrastructure.
- **Option B (delete the parity test):** if `test_attention_fmha_sm120.cu` already provides ground-truth equivalent for the same shapes, delete the parity assertion entirely.

Report decision before proceeding. If unclear, default to Option A (safer).

- [ ] **Step 1: Pre-flight reference scan**

```bash
grep -rn 'naive_attention\|attention_naive\|attention.naive\|no_naive_swa\|use_naive_attn\|use_naive_for_swa' \
  src/ tests/ tools/ include/ CMakeLists.txt
```

Note every match.

- [ ] **Step 2: Move naive source to archive**

```bash
mkdir -p docs/archive/attention_naive
git mv src/compute/attention_naive.cu docs/archive/attention_naive/
git mv src/compute/attention_naive.h docs/archive/attention_naive/
```

- [ ] **Step 3: Write the resurrection memo**

Create `docs/archive/attention_naive/RESURRECTION.md`:

```markdown
# Resurrection: naive attention reference

**Archived 2026-05-20** (Phase 2 of architecture refactor roadmap).

## What this was

A pure-FP16 reference attention prefill (no FMHA, no cuBLAS, no flash):
straightforward QK^T + softmax + PV with optional sliding window. Lived
at `src/compute/attention_naive.{h,cu}` (152 LOC).

Two callers existed before archival:
1. **Runtime SWA fallback** in `executor_attention.cu`, gated by
   `attention.no_naive_swa=false`. Used to be the only safe path for
   Gemma-4 SWA layers when cuBLAS S-matrix overflowed.
2. **Parity test** in `tests/test_attention_chunked.cu` — ground-truth
   reference to validate cuBLAS-SWA output.

## Why it was archived

1. **Runtime path:** Replaced by chunked prefill (PR documented in
   `gemma4_chunked_prefill_2026_05_15.md`). The Gemma-4 SWA layers
   now use cuBLAS sliding-window mask via the chunked path, with no
   S-matrix overflow.
2. **Test parity:** Either (a) the reference function was inlined into
   `test_attention_chunked.cu` as a local static, keeping the parity
   coverage without a public symbol; OR (b) the test was deleted as
   redundant with `test_attention_fmha_sm120.cu` coverage.

## How to resurrect (runtime fallback)

If a future model needs a non-tiled SWA fallback again:

1. `git mv docs/archive/attention_naive/attention_naive.{cu,h} src/compute/`
2. Restore `src/compute/attention_naive.cu` in `CMakeLists.txt`.
3. Restore the gate + call in `executor_attention.cu` (was at
   `:817-846` at archive time; check pre-archive history with
   `git log --follow` on this file).
4. Restore `attention.naive` and `attention.no_naive_swa` in
   `runtime/config.h`.

## Original source

Frozen at this PR's HEAD.
```

- [ ] **Step 4: Update `test_attention_chunked.cu` per Step 0 decision**

**If Option A (inline reference):**

Read `docs/archive/attention_naive/attention_naive.cu` to find the
`naive_attention_prefill` function body. Copy it into
`tests/test_attention_chunked.cu` as a `static` function inside an
anonymous namespace at the top of the file. Rename to
`naive_attention_prefill_ref` to make its test-only role explicit.
Update the call site at the previously-`:213` line to call the new
local name.

Remove the `#include "compute/attention_naive.h"` at the top of the test
file.

**If Option B (delete the parity test):**

Delete the parity-check TEST_F block (around lines 180-235 of
`test_attention_chunked.cu`). Remove the `#include
"compute/attention_naive.h"`.

- [ ] **Step 5: Remove the runtime gate from `executor_attention.cu`**

Edit `src/exec/executor_attention.cu`. Locate lines around 817-846:

```cpp
        const bool use_naive_attn = RuntimeConfig::current().attention.naive;
        ...
        bool use_naive_for_swa = (gemma4_overflow_cublas && n <= 8192 &&
                                  !RuntimeConfig::current().attention.no_naive_swa);
        if ((use_naive_attn && n <= 2048) || use_naive_for_swa) {
            ...
            naive_attention_prefill(...);
        } else if ((force_cublas_attn || ...
```

Delete the entire `if ((use_naive_attn ...) || use_naive_for_swa) { ... }`
branch. The dispatcher proceeds directly to the cuBLAS branch (`else if (force_cublas_attn || !no_cublas_attn) ...` becomes the leading `if`).

Also remove the `#include "compute/attention_naive.h"` at the top of the file.

- [ ] **Step 6: Remove the config fields**

Edit `src/runtime/config.h`. Locate the `AttentionConfig` section. Remove:

```cpp
bool naive = false;
bool no_naive_swa = false;
```

(Names may differ slightly — confirm with grep.)

Run:

```bash
grep -n 'attention.naive\|naive_swa\|use_naive' src/runtime/config.cpp
```

Remove corresponding seed/parse/print lines.

- [ ] **Step 7: Clean docstrings in `attention_cublas.{h,cu}`**

Edit `src/compute/attention_cublas.h:22` — remove the docstring sentence:

```cpp
// semantics of naive_attention_prefill's sliding_window parameter. Defaults to 0 (off).
```

Replace with:

```cpp
// sliding_window parameter (number of past tokens visible; 0 = off).
```

Edit `src/compute/attention_cublas.cu:104` — remove:

```cpp
// (matches naive_attention_prefill semantics).
```

(Delete the parenthetical clause; preserve the surrounding context.)

- [ ] **Step 8: Remove CMakeLists.txt entry**

Edit `CMakeLists.txt:164`. Find:

```cmake
    src/compute/attention_naive.cu
```

Delete the line.

- [ ] **Step 9: Verify references gone**

```bash
grep -rn 'naive_attention\|attention_naive\|attention.naive\|no_naive_swa\|use_naive_attn\|use_naive_for_swa' \
  src/ tests/ tools/ include/ CMakeLists.txt
```

Expected: zero matches if Option B was taken, or one match for the local-static reference function inside `tests/test_attention_chunked.cu` (under its renamed `_ref` suffix) if Option A was taken.

- [ ] **Step 10: Build + test**

```bash
make build && make verify-fast
```

Expected: green.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(attention): archive naive attention reference

The naive attention reference at src/compute/attention_naive.{cu,h} had
two callers:

1. Runtime SWA fallback in executor_attention.cu, kept around for
   Gemma-4 SWA layers when cuBLAS S-matrix overflowed. Made obsolete
   by chunked prefill (gemma4_chunked_prefill_2026_05_15.md) — Gemma-4
   SWA now uses cuBLAS sliding-window mask via the chunked path.

2. Parity ground truth in test_attention_chunked.cu. [Option A:
   inlined as a local static reference function under the test file
   to preserve coverage without a public symbol.] [Option B: parity
   test deleted as redundant with test_attention_fmha_sm120.cu.]

Archive at docs/archive/attention_naive/ with a resurrection memo.

Removes attention.naive + attention.no_naive_swa config fields and the
runtime gate from executor_attention.cu.

Phase 2 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Pick the Option A or Option B sentence from the brackets according to Step 0's decision; delete the unused bracketed alternative.)

---

## Task 4: Simplify the dispatcher gate

**Files:**
- Modify: `src/exec/executor_attention.cu` — the prefill gate around line 847 and surrounding context
- Target: file goes from 1300 LOC to ≤700 LOC. Decode switch at `:996` is untouched (already flat).

**Important:** After Tasks 1-3, the naive branch is gone, so the dispatcher's branching is already simpler. This task collapses what remains.

- [ ] **Step 1: Re-read the current state of the gate**

```bash
wc -l src/exec/executor_attention.cu
sed -n '800,900p' src/exec/executor_attention.cu
```

Confirm Tasks 1-3 landed: no `naive_attention_prefill` calls, no `no_fmha_cluster` references, no `mxf4nvf4_blockscale` references. Note the current file LOC.

- [ ] **Step 2: Document the target dispatch matrix**

Decide and write down (in a Bash comment in the commit message OR as a header comment in `executor_attention.cu`) the post-simplification dispatch matrix:

```
Prefill:
  IF n > attn_scores_capacity OR sliding_active (non-Gemma-4)
    → FMHA fallback (attention_prefill_dispatch → fmha_sm120_prefill, the one remaining variant)
  ELSE
    → cuBLAS QK^T + causal softmax + cuBLAS PV (attention_cublas_prefill)

Decode:
  switch (cache_dtype) — unchanged at :996+
```

The 4-clause predicate `(force_cublas || !no_cublas) && attn_scores_buf_ && n ≤ cap && (force_cublas || !sliding)` collapses to: "use cuBLAS unless the S-matrix doesn't fit or sliding is active". The `force_cublas` override (for Gemma-4 hd=512) becomes "ALWAYS cuBLAS if hd=512, regardless of n/sliding, because FMHA hd=512 OOMs the 100 KiB smem on sm_120".

- [ ] **Step 3: Rewrite the prefill branch**

In `src/exec/executor_attention.cu`, replace the current gate (formerly at lines 817-885 ish, but offsets will have shifted after Tasks 1-3) with:

```cpp
// Prefill dispatch (post-Phase-2 simplification):
//   cuBLAS QK^T + causal softmax + cuBLAS PV is the default. Falls back
//   to the FMHA chain only when the S-matrix cap can't hold the [nh, n, n]
//   tensor, or when the layer uses sliding-window attention on a model
//   that isn't Gemma-4 (Gemma-4 SWA uses cuBLAS via the chunked path).
//
// force_cublas_attn: set per-layer for Gemma-4 hd=512 global layers,
// where FMHA OOMs the 100 KiB smem; cuBLAS handles arbitrary head_dim
// at the cost of the materialized S-matrix.
const bool s_matrix_fits = attn_scores_buf_ != nullptr &&
                            n <= static_cast<int>(attn_scores_.shape[1]);
const bool non_gemma4_sliding = !force_cublas_attn && sliding_active;

if (s_matrix_fits && !non_gemma4_sliding) {
    attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, nh, nkv, hd, scale,
                             /*causal=*/true, cfg.attn_logit_softcap,
                             /*q_offset=*/0, stream, layer_sliding_window);
} else {
    // FMHA fallback: tiled O(n) memory chain.
    int64_t q4s[4] = {1, n, nh, hd};
    int64_t kv4s[4] = {1, n, nkv, hd};
    int64_t o4s[4] = {1, n, nh, hd};
    Tensor q4 = qv.reshape(4, q4s);
    Tensor k4 = kk.reshape(4, kv4s);
    Tensor v4 = vv.reshape(4, kv4s);
    Tensor o4 = ao.reshape(4, o4s);
    attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true,
                               layer_sliding_window, cfg.attn_logit_softcap, stream);
}
```

Delete the old `if (use_naive_attn ...)` / `else if (force_cublas || !no_cublas) ...` / `else { ... reshape + FMHA ... }` cascade.

- [ ] **Step 4: Drop now-unused `no_cublas_attn` config field if applicable**

```bash
grep -n 'no_cublas_attn\|no_cublas' src/runtime/config.h src/runtime/config.cpp src/exec/executor_attention.cu
```

If `no_cublas_attn` is still referenced elsewhere (e.g., for a separate debug switch), leave it. If its only consumer was the gate this task rewrote, remove it from `runtime/config.h` and `runtime/config.cpp` as part of this commit.

- [ ] **Step 5: Verify LOC target**

```bash
wc -l src/exec/executor_attention.cu
```

Expected: ≤700 LOC. If still over 700, the file likely contains other code beyond the prefill+decode dispatch — note in commit body but don't force further reduction. Phase 4 (engine split) may absorb additional cleanup.

- [ ] **Step 6: Build + tests**

```bash
make build && make verify-fast
```

Expected: green. The behavior change should be zero — cuBLAS-on-default was already the case; this task only collapses the predicate.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(attention): collapse prefill dispatcher gate

After archiving the naive, cluster, and mxf4nvf4 paths in Tasks 1-3,
the dispatcher's prefill gate at executor_attention.cu collapses from
a 4-clause predicate to a clean two-branch switch:

  if S-matrix fits AND not non-Gemma-4 sliding:
      cuBLAS QK^T + softmax + PV
  else:
      FMHA fallback chain

The decode switch at :996 is untouched (already flat per Phase 2 spec).

executor_attention.cu shrinks from 1300 LOC to <N> LOC (target ≤700).

Phase 2 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `<N>` with the actual measured LOC from Step 5.)

---

## Task 5 (SOFT — may slip): Write `docs/attention-dispatch.md`

Documents the final dispatch matrix. Single doc file. Branch from
post-Task-4 main. Plan briefly:

- [ ] **Step 1: Create `docs/attention-dispatch.md`** with three sections:
  1. **Prefill matrix** — table of (dtype × sliding × force_cublas) → kernel
  2. **Decode matrix** — table of (cache_dtype) → kernel (mirroring the `switch` at `:996`)
  3. **Decision flow** — short prose explaining when each path fires
- [ ] **Step 2: Link from README documentation table** as second row
  after `docs/architecture.md`
- [ ] **Step 3: Run `make verify-fast`** (doc-only, no-op)
- [ ] **Step 4: Commit** with conventional message + Co-Authored-By trailer

(Full content not specified here — the matrix is best generated by
reading the post-Task-4 dispatcher state. Treat the file as a 100-200
line markdown doc summarizing what the dispatcher does, with line-number
references into `src/exec/executor_attention.cu` and
`src/compute/attention_dispatch.cu`.)

---

## Task 6 (SOFT — may slip): Compile-only resurrection tests

For each archived variant, add a CMake test target that compiles the
archived source against current headers to detect bitrot. Branch from
post-Task-4 main.

- [ ] **Step 1: Create `tests/archive_compile_check/CMakeLists.txt`**:

```cmake
# Compile-only "bitrot check" for archived FMHA variants. Not linked
# into the main build. Run with `make archive-compile-check`.
add_executable(archive_compile_check
    ${CMAKE_SOURCE_DIR}/docs/archive/fmha_sm120_cluster/attention_fmha_sm120_cluster.cu
    ${CMAKE_SOURCE_DIR}/docs/archive/fmha_mxf4nvf4_sm120/attention_fmha_mxf4nvf4_sm120.cu
    ${CMAKE_SOURCE_DIR}/docs/archive/attention_naive/attention_naive.cu
)
set_target_properties(archive_compile_check PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_include_directories(archive_compile_check PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(archive_compile_check PRIVATE imp_compute imp_core)
```

- [ ] **Step 2: Add a Makefile target** in the top-level `Makefile`:

```make
archive-compile-check:
\tcmake --build build --target archive_compile_check
```

- [ ] **Step 3: Include the subdirectory** from the top-level
`CMakeLists.txt`:

```cmake
if(IMP_ENABLE_ARCHIVE_COMPILE_CHECK)
    add_subdirectory(tests/archive_compile_check)
endif()
```

(Gated by an option so the default build is unaffected.)

- [ ] **Step 4: Document** in the resurrection memos that
`-DIMP_ENABLE_ARCHIVE_COMPILE_CHECK=ON` builds the bitrot-detection
target.

- [ ] **Step 5: Verify** the target builds (`cmake -B build
-DIMP_ENABLE_ARCHIVE_COMPILE_CHECK=ON && cmake --build build --target
archive_compile_check`).

- [ ] **Step 6: Commit** with conventional message.

---

## Task 7 (SOFT — may slip): Eliminate `attention_paged_common.cuh`

Per spec §3 Phase 2 soft PRs: "Move `attention_paged_common.cuh`
includes into the per-dtype paged files (eliminate the umbrella header)."

Plan briefly:

- [ ] **Step 1: Inspect `attention_paged_common.cuh`** to find which
declarations are used where.
- [ ] **Step 2: Inline** each declaration into the file(s) that need it
(usually one per quant dtype: `attention_paged_fp8.cu`,
`attention_paged_int4.cu`, `attention_paged_int8.cu`,
`attention_paged_nvfp4.cu`, `attention_paged_nvfp4_tc.cu`,
`attention_paged.cu`).
- [ ] **Step 3: Delete `attention_paged_common.cuh`** if no symbols
remain shared.
- [ ] **Step 4: Build + test.**
- [ ] **Step 5: Commit.**

---

## Phase 2 closeout

After Tasks 1-4 are merged (soft Tasks 5-7 may slip):

- [ ] **Step 1: Full verification suite**

```bash
make verify
```

Expected: green except for any pre-existing failures unrelated to Phase 2. Note that `FmhaSm120Test.ClusterPathNonAligned` (which failed on main at Phase 1 closure) is removed by Task 1 — that failure should no longer appear.

- [ ] **Step 2: Capture perf snapshot** (advisory, per spec §5)

```bash
scripts/gen_perf_baseline.sh
git diff tests/perf_baseline.json
```

Phase 2 is expected to be perf-neutral on the default path (cuBLAS prefill was already default-on; the gate simplification is structural). If decode shows any change, document the surprise.

- [ ] **Step 3: Update MEMORY.md**

Write a new memory file
`architecture_refactor_phase_2_closed_2026_MM_DD.md` in
`/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/`.
Add ONE line to `MEMORY.md` index pointing at it. Compact older
Phase-1 + memo lines if `MEMORY.md` is over 200 lines.

- [ ] **Step 4: Mark Phase 2 closed in the roadmap spec**

Edit `docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md`.
At the top of the Phase 2 section, add:

```markdown
**Status (2026-MM-DD):** Closed. PRs #<N1> (archive cluster),
#<N2> (archive mxf4nvf4), #<N3> (archive naive), #<N4>
(dispatcher gate simplification), [#<N5> (attention-dispatch.md doc)],
[#<N6> (archive compile-check)], [#<N7> (drop attention_paged_common.cuh)].
```

- [ ] **Step 5: Commit closeout** on a `docs/phase-2-closeout` branch + PR.

Phase 3 ("Pre-Dequant + Quant-Zoo aufräumen") may now begin. A new
implementation plan is required — invoke writing-plans with the Phase 3
section of the spec.
