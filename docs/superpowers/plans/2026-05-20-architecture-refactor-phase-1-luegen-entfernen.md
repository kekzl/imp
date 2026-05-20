# Architecture Refactor Phase 1 — "Lügen entfernen" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the dead `Graph` IR (~250 LOC), bring the architecture diagram from the side branch onto `main`, and write a canonical `docs/architecture.md` narrative — so what the repo claims structurally matches what the code does.

**Architecture:** Three sequential cleanup tasks plus one optional rename. Each task is a single PR. No code-execution-logic changes; only deletions, file moves, doc creation, and a mechanical rename. The existing test suite (`make verify-fast`) is the regression net — there is no new test logic to write because we are removing dead code, not changing behavior.

**Tech Stack:** C++20, CUDA 13.2, CMake, Docker build (`make build`), GTest suite (`make verify-fast` / `make verify`).

---

## Reference: Source spec

`docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md` §3 Phase 1. This plan implements the three Phase 1 critical PRs plus the soft PR. Both spec and plan live on `main`.

## Reference: Pre-flight evidence

Establishes that `Graph`/`OpNode`/`OpType` is dead:

```
$ grep -rln 'OpType::\|imp::Graph\|imp::OpNode' src/ include/ tests/ tools/
src/graph/op.cpp
src/graph/op.h
src/graph/graph.cpp
src/graph/graph.h
```

All four occurrences are the dead-IR files referencing each other. No executor, kernel, test, or tool consumes the type. Diagram files (`docs/architecture.{dot,svg,png}`) exist on branch `docs/arch-diagram` at commit `163bcf7` and were never merged.

---

## Task 1: Delete the dead Graph IR

**Files:**
- Delete: `src/graph/op.h`
- Delete: `src/graph/op.cpp`
- Delete: `src/graph/graph.h`
- Delete: `src/graph/graph.cpp`
- Modify: `CMakeLists.txt:237-238` (remove the two source-list lines)

- [ ] **Step 1: Verify deadness one more time (pre-condition)**

Run:

```bash
grep -rln 'OpType::\|imp::Graph\b\|imp::OpNode' src/ include/ tests/ tools/
```

Expected output (exactly four lines, all in `src/graph/`):

```
src/graph/op.cpp
src/graph/op.h
src/graph/graph.cpp
src/graph/graph.h
```

If any other file appears, **stop**. The IR has a live consumer and the deletion is no longer safe. Surface the new finding before continuing.

- [ ] **Step 2: Verify no string-based dispatch (defense-in-depth)**

Run:

```bash
grep -rn '"EMBEDDING"\|"RMSNORM"\|"ATTENTION_PREFILL"\|op_type_name' src/ include/ tests/ tools/
```

Expected output (one line, in the dead IR itself):

```
src/graph/op.h:33:const char* op_type_name(OpType type);
```

If any other file references `op_type_name` or stringified OpType values, **stop** and investigate.

- [ ] **Step 3: Delete the four source files**

Run:

```bash
git rm src/graph/op.h src/graph/op.cpp src/graph/graph.h src/graph/graph.cpp
```

- [ ] **Step 4: Update CMakeLists.txt**

Edit `CMakeLists.txt`. Find these two lines (lines 237-238 at time of writing — confirm with `grep -n 'op.cpp\|graph.cpp' CMakeLists.txt`):

```cmake
    src/graph/op.cpp
    src/graph/graph.cpp
```

Delete both lines.

- [ ] **Step 5: Build to confirm no compile-time consumer was missed**

Run:

```bash
make build
```

Expected: build completes successfully, no errors about missing `op.h` / `graph.h` / `Graph` / `OpNode` / `OpType` symbols. If a build error references those symbols, **stop**: the grep in Step 1 missed a consumer (e.g., a generated file or a macro-hidden include). Investigate before deleting.

- [ ] **Step 6: Run the fast test suite**

Run:

```bash
make verify-fast
```

Expected: all tests pass in ~90 seconds. Decode/prefill perf gate is advisory (per spec §5); deletion of unreachable code cannot affect runtime perf, so any reported drift is variance.

- [ ] **Step 7: Commit**

Run:

```bash
git add -u
git commit -m "$(cat <<'EOF'
refactor(graph): delete dead Graph/OpNode IR

Graph, OpNode, OpType, and op_type_name had no live consumers — only
the four IR source files referenced each other. Confirmed via:
  grep -rln 'OpType::\|imp::Graph\|imp::OpNode' src/ include/ tests/ tools/
returning only src/graph/op.{h,cpp} and src/graph/graph.{h,cpp}.

Removes ~250 LOC of misleading scaffolding. The "graph executor" is and
remains an imperative sequence of executor_*.cu calls; the directory
name will be revisited in a follow-up rename.

Phase 1 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Bring the architecture diagram from the side branch onto main

**Files:**
- Create: `docs/architecture.dot` (from branch `docs/arch-diagram` commit `163bcf7`)
- Create: `docs/architecture.svg` (from same)
- Create: `docs/architecture.png` (from same)
- Modify: `README.md` (insert link in the Documentation table)
- Modify: `CLAUDE.md` (add diagram to "Where to look" section)

- [ ] **Step 1: Verify the source commit and files**

Run:

```bash
git show --stat 163bcf7 | head -25
git ls-tree 163bcf7 docs/ | grep architecture
```

Expected: commit `163bcf7` is "docs(arch): add E2E load + execution pipeline diagram" on branch `docs/arch-diagram`, and the tree contains `docs/architecture.dot`, `docs/architecture.png`, `docs/architecture.svg`.

- [ ] **Step 2: Check out the three diagram files into main**

Run:

```bash
git checkout 163bcf7 -- docs/architecture.dot docs/architecture.svg docs/architecture.png
git status --short
```

Expected: three new files staged for add (`A  docs/architecture.dot`, `A  docs/architecture.svg`, `A  docs/architecture.png`).

- [ ] **Step 3: Verify the SVG is renderable (sanity check)**

Run:

```bash
head -5 docs/architecture.svg
file docs/architecture.svg docs/architecture.png
```

Expected: SVG starts with `<?xml version="1.0"` or `<svg ...`, `file` reports SVG/XML and PNG image data respectively.

- [ ] **Step 4: Add the diagram to the README documentation table**

Edit `README.md`. Find this block (current `## Documentation` section, ~line 115-126):

```markdown
## Documentation

| Document | Description |
|---|---|
| [Usage & reference](docs/usage.md) | Build, server, CLI, C API |
| [Supported models](docs/supported-models.md) | Tested model families with VRAM + tok/s |
| [Quantization](docs/quantization.md) | GGUF Q*_K, NVFP4, MXFP4, FP8 KV — formats, pipelines, trade-offs |
```

Insert this new row as the **first** documentation row, directly above `Usage & reference`:

```markdown
| [Architecture diagram](docs/architecture.svg) ([source](docs/architecture.dot)) | End-to-end load + execution pipeline (load → engine init → prefill → decode) |
```

The full Documentation table after the edit:

```markdown
## Documentation

| Document | Description |
|---|---|
| [Architecture diagram](docs/architecture.svg) ([source](docs/architecture.dot)) | End-to-end load + execution pipeline (load → engine init → prefill → decode) |
| [Usage & reference](docs/usage.md) | Build, server, CLI, C API |
| [Supported models](docs/supported-models.md) | Tested model families with VRAM + tok/s |
| [Quantization](docs/quantization.md) | GGUF Q*_K, NVFP4, MXFP4, FP8 KV — formats, pipelines, trade-offs |
| [Performance](docs/performance.md) | Decode + prefill throughput, methodology |
| [imp.conf reference](imp.conf.example) | All runtime configuration keys |
| [sm_120a kernels](docs/sm120.md) | Kernel optimization notes |
| [Roadmap](docs/roadmap.md) | Open bugs and in-flight performance work |
| [Changelog](CHANGELOG.md) | Per-release notes |
```

- [ ] **Step 5: Add the diagram to CLAUDE.md "Where to look"**

Edit `CLAUDE.md`. Find the "Where to look" section near the end. Add a new bullet as the **first** bullet under that section:

```markdown
- `docs/architecture.{svg,dot}` — canonical end-to-end pipeline diagram (load → init → prefill → decode + attention dispatcher)
```

- [ ] **Step 6: Re-rendering instructions**

Add a small note to the bottom of `docs/architecture.md` if it already exists, otherwise defer to Task 3. For this task, just verify the re-render command works (only if Docker is available locally for the dot renderer):

```bash
docker run --rm -v "$(pwd)/docs:/d" nshine/dot dot -Tsvg /d/architecture.dot -o /tmp/architecture_rerender.svg
diff -q docs/architecture.svg /tmp/architecture_rerender.svg && echo "byte-identical re-render" || echo "differs (acceptable — graphviz may produce non-determ ordering)"
```

This is a sanity check, not a gate. If `nshine/dot` is unavailable, skip — re-rendering is documented in the commit message of `163bcf7`.

- [ ] **Step 7: Run the fast test suite**

Run:

```bash
make verify-fast
```

Expected: green. Documentation-only PR; tests should not be affected.

- [ ] **Step 8: Commit**

Run:

```bash
git add docs/architecture.dot docs/architecture.svg docs/architecture.png README.md CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(arch): merge architecture diagram from docs/arch-diagram to main

Brings the end-to-end load + execution pipeline diagram (originally
added in commit 163bcf7 on branch docs/arch-diagram) onto main. Adds
README and CLAUDE.md links so the diagram is discoverable.

Re-render: docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tsvg /d/architecture.dot -o /d/architecture.svg

Phase 1 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Write `docs/architecture.md` as the canonical narrative companion

**Files:**
- Create: `docs/architecture.md`
- Modify: `README.md` (update the "Architecture diagram" row to point at the markdown narrative, with the SVG linked from it)

- [ ] **Step 1: Create `docs/architecture.md`**

Write `docs/architecture.md` with this exact content:

```markdown
# imp — Architecture

This document is the canonical narrative companion to [`architecture.svg`](architecture.svg).
The SVG shows the structural overview; this file explains each phase in
prose and points at the source files that implement it.

If the code and this document disagree, the code wins — but a disagreement
is a bug in this document and should be fixed.

## At a glance

imp runs LLM inference end-to-end in four phases:

1. **Load** — read a GGUF file or a Hugging-Face SafeTensors directory into a
   `Model` object with a `WeightMap`, a `Tokenizer`, and a `ModelConfig`.
2. **Engine init** — resolve runtime config, upload weights to VRAM, build
   the paged KV cache, allocate workspaces, capture CUDA graphs for decode.
3. **Prefill** — run the prompt through the per-layer forward pass (chunked
   if the architecture supports it), producing the first-token logits.
4. **Decode** — replay the captured CUDA graph per token: attention → FFN →
   LM head → penalties → sampler → stop check, looping until EOS or limit.

See [`architecture.svg`](architecture.svg) for the full graph including
the attention dispatcher, memory subsystem, and kernel subsystem.

## Phase 1 — Load (one-time, `src/model/`)

Entry: `imp_model_load(path) → ImpModel` (`include/imp/imp.h`, dispatched
via `src/api/imp_api.cpp`).

Format detection inspects the path: a `.gguf` file routes to
`src/model/gguf_loader.cpp`; a directory containing `config.json` and
`*.safetensors` routes to `src/model/safetensors_loader.cpp` with optional
LLM-Compressor recipe handling in `src/model/llm_compressor_loader.cpp`.

Both loaders produce:

- A **WeightMap** (`src/model/weight_map.cpp`) — tensor name → role.
- A **Tokenizer** (`src/model/tokenizer.cpp` + `chat_template.cpp` +
  `jinja.cpp` + optional `sentencepiece_loader.cpp`).
- A **ModelConfig** + **Model** object (`src/model/model.cpp`,
  `src/model/model_arch.h`).

## Phase 2 — Engine init (one-time, `src/runtime/engine.cpp`)

Entry: `imp_context_create(model, ImpConfig) → ImpContext`.

`Engine::init()` orchestrates the init pipeline. The major steps are
distinct private methods on `Engine`:

| Step | Method | Notes |
|---|---|---|
| Load runtime config | `RuntimeConfig::load()` | `imp.conf` + `--config` CLI + legacy env-var seeds (`src/runtime/config.cpp`) |
| Resolve quant/KV/SSM dtypes | `init_resolve_*` group | `init_resolve_kv_dtype_policy_`, `init_resolve_ssm_dtype_`, `init_resolve_fp8_prefill_`, `init_resolve_quant_flags_` |
| Compute max sequence length | `init_compute_max_seq_len_` | VRAM budget → max context (`src/runtime/vram_budget.cpp`) |
| Upload weights | `init_weights` | `upload_weight` + `upload_expert_weights` in `src/model/weight_upload.cu`; pre-dequant in `src/graph/executor_pre_dequant.cu` |
| Init KV cache | `init_kv_cache` | Paged blocks (block_size=16); dtype is FP16 / FP8 / INT8 / INT4 / NVFP4 / MXFP4 |
| Allocate workspaces | `init_features` | MMVQ scratch, cuBLAS S-matrix (~1 GiB — see Known wounds), FP8 activation scratch, split-K attn scratch |
| Warm up | `warmup()` | Captures CUDA graph for decode (`src/runtime/cuda_graph.cu`) |

The Engine surface is currently a large class — splitting it into named
subsystems is Phase 4 of the refactor roadmap.

## Phase 3 — Prefill (per request, `Engine::step_prefill`)

Entry: `imp_prefill_with_params(tokens, n) → status`.

Per-chunk loop in `src/graph/executor_forward.cu` (or
`executor_forward_moe.cu` for MoE architectures). Each layer runs:

```
RMSNorm → QKV GEMM + RoPE + KV-cache write → Attention → O proj →
RMSNorm + residual → FFN (dense SwiGLU or MoE top-k grouped GEMM)
```

After the last layer of the last chunk: final RMSNorm + LM head → logits.

### Attention dispatcher (the central choice)

`executor_attention.cu` decides which attention kernel to call. The gate
at `executor_attention.cu:847` checks:

```
if (force_cublas || !no_cublas) && attn_scores_buf_ && n ≤ cap
    && (force_cublas || !sliding)
```

When true (the default for typical Qwen3 / Gemma-4 configs), prefill goes
through `attention_cublas_prefill`: cuBLAS QK^T → ~1 GiB S-matrix buffer
→ causal softmax → cuBLAS PV. When false, it falls through to
`attention_prefill_dispatch` (`attention_dispatch.cu:30`), which selects
among the per-dtype FMHA variants.

Decode attention uses a separate switch on `cache_dtype` at
`executor_attention.cu:996`, dispatching to one of the paged kernels
(INT4 / NVFP4 ± TC / MXFP4-KV / INT8 / FP8 / FP16-paged).

**Status:** the FMHA dispatch chain currently has more variants than are
default-enabled. Phase 2 of the refactor roadmap archives the unused
variants.

## Phase 4 — Decode loop (per token, `Engine::step_decode_forward`)

Entry: `imp_decode_step(params) → next_token` (or the streaming wrapper
`imp_generate_streaming`).

Per token:

1. **Replay** the captured CUDA graph (`src/runtime/cuda_graph.cu`).
   Graph capture is enabled for most architectures; non-Gemma-4 MoE
   disables it because of host-side routing (see CLAUDE.md).
2. **Paged attention decode** — kernel chosen by KV dtype.
3. **FFN GEMV** — dp4a / mma.sync / NVFP4 variants in
   `executor_ffn.cu`.
4. **LM head GEMV** → logits.
5. **Apply penalties** (repeat / freq / presence / DRY) — `request.cpp`.
6. **Sampler** (temp / top-p / top-k / min-p / typical / mirostat) —
   `request.cpp`.
7. **Stop check** — EOS, max_tokens, stop strings.
8. **(Optional) MTP spec-decode draft** — `src/runtime/mtp_forward.cu`.

## Subsystems referenced across phases

- **Memory** — `src/memory/vram_allocator.cu`, `src/memory/kv_cache.cu`,
  `src/memory/kv_cache_manager.cpp`, `src/memory/layer_offload.cu`,
  `src/runtime/vram_budget.cpp`, `src/runtime/storage_planner.cpp`. The
  cross-cutting ownership here is intentionally being consolidated in
  Phase 5 of the refactor roadmap.
- **Kernels** — `src/compute/` (attention, GEMM, RMSNorm, RoPE, SwiGLU,
  softmax, sampling) and `src/quant/` (dequant, FP8 quant, NVFP4 quant).
- **Public C API** — `include/imp/{imp,types,error,config}.h`,
  implemented in `src/api/imp_api.cpp`. ABI-stable per CONTRIBUTING.md.

## Known structural wounds

These are documented for honesty. See the refactor roadmap in
[`superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md`](superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md)
for the resolution plan.

- **`Engine::init` is 700+ LOC of orchestration in one method.** Will be
  split into named subsystems in Phase 4.
- **The cuBLAS attention path allocates ~1 GiB of S-matrix workspace.**
  Caps maximum context length. Phase 5 evaluates tiled streaming softmax
  or FMHA-default-switch.
- **`RuntimeConfig::current()` is a global singleton.** Phase 5
  de-globalizes it to per-Engine.
- **The directory `src/graph/` does not contain a graph IR** (the
  `OpNode`/`Graph` types were deleted in Phase 1). Rename to `src/exec/`
  is a Phase 1 soft PR, possibly slipping to Phase 5.

## Re-rendering the diagram

```bash
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tsvg /d/architecture.dot -o /d/architecture.svg
docker run --rm -v "$(pwd)/docs:/d" nshine/dot \
  dot -Tpng /d/architecture.dot -o /d/architecture.png
```

Edit `architecture.dot` first, then regenerate both raster forms.
```

- [ ] **Step 2: Update README to point at the markdown narrative**

Edit `README.md`. Replace the row added in Task 2 Step 4. Find:

```markdown
| [Architecture diagram](docs/architecture.svg) ([source](docs/architecture.dot)) | End-to-end load + execution pipeline (load → engine init → prefill → decode) |
```

Replace with:

```markdown
| [Architecture](docs/architecture.md) ([diagram](docs/architecture.svg)) | End-to-end load + execution pipeline (load → engine init → prefill → decode) |
```

- [ ] **Step 3: Run the fast test suite**

Run:

```bash
make verify-fast
```

Expected: green. Documentation-only PR.

- [ ] **Step 4: Commit**

Run:

```bash
git add docs/architecture.md README.md
git commit -m "$(cat <<'EOF'
docs(arch): add narrative companion docs/architecture.md

The SVG shows the structural overview; this markdown explains each phase
in prose, points at the source files that implement it, and lists the
known structural wounds with pointers to the refactor roadmap.

Phase 1 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 (SOFT — may slip to a later phase or be dropped): Rename `src/graph/` → `src/exec/`

**Why soft:** The Phase 1 critical PRs are complete after Task 3. This
rename is mechanical but touches 85+ include lines and the CMakeLists.
If you are short on time, defer; the directory is no longer lying about
containing a Graph IR (deleted in Task 1).

**Files:**
- Rename: all files under `src/graph/` to `src/exec/`
- Modify: every `#include "graph/..."` (85 occurrences across src/) → `#include "exec/..."`
- Modify: `CMakeLists.txt` — `IMP_GRAPH_SOURCES` variable name + source paths

- [ ] **Step 1: Pre-flight scan — confirm the rename surface**

Run:

```bash
grep -rln '#include "graph/' src/ tools/ tests/ | wc -l
grep -rn 'src/graph/' CMakeLists.txt | head
grep -rn 'IMP_GRAPH_SOURCES\|graph/' CMakeLists.txt | head
```

Note the file count (should be ~85 — exact number may have shifted if
unrelated PRs landed between tasks). If the count differs by more than
±10, refresh this plan task.

- [ ] **Step 2: Move all files**

Run:

```bash
git mv src/graph src/exec
git status --short | head
```

Expected: every file under `src/graph/` shows as renamed to
`src/exec/`. CMakeLists is not yet updated; build will fail until
Step 3.

- [ ] **Step 3: Rewrite include paths in source**

Run:

```bash
grep -rl '#include "graph/' src/ tools/ tests/ \
  | xargs sed -i 's|#include "graph/|#include "exec/|g'
grep -rn '#include "graph/' src/ tools/ tests/
```

Expected: second `grep` returns no matches.

- [ ] **Step 4: Update CMakeLists.txt**

Edit `CMakeLists.txt`. Find the `IMP_GRAPH_SOURCES` variable (around
line ~236). Two changes:

1. Rename the variable: `IMP_GRAPH_SOURCES` → `IMP_EXEC_SOURCES`.
2. Rewrite each `src/graph/...` path to `src/exec/...`.

Then find every reference to `IMP_GRAPH_SOURCES` elsewhere in
`CMakeLists.txt` and rename it:

```bash
grep -n 'IMP_GRAPH_SOURCES' CMakeLists.txt
```

Replace each occurrence.

- [ ] **Step 5: Pre-flight grep for any missed `graph/` reference**

Run:

```bash
grep -rn 'src/graph/\|"graph/' src/ include/ tests/ tools/ CMakeLists.txt
```

Expected: empty output. Any remaining match is a missed reference and
must be fixed before build.

- [ ] **Step 6: Build**

Run:

```bash
make build
```

Expected: build completes. If `fatal error: graph/...: No such file or
directory` appears, return to Step 3.

- [ ] **Step 7: Full test run**

Run:

```bash
make verify-fast
```

Expected: green. The rename is a no-op for behavior; any failure is a
plumbing bug from Steps 2-4.

- [ ] **Step 8: Update CLAUDE.md "Where to look" and docs/architecture.md**

`CLAUDE.md` "Where to look" lists `src/{api,compute,core,graph,memory,...}`.
Change `graph` → `exec`.

`docs/architecture.md` (created in Task 3) references `src/graph/` in
multiple places — search and replace `src/graph/` → `src/exec/`.

- [ ] **Step 9: Commit**

Run:

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: rename src/graph/ → src/exec/

The Graph IR (OpNode, OpType, Graph) was deleted in an earlier Phase 1
commit. The directory name "graph" was misleading — its contents are
imperative GraphExecutor / GEMM-kernel-registry / workspace / scratch
code, not a graph. Rename clarifies intent.

Mechanical change: file moves + include-path rewrite (85 occurrences) +
CMake variable rename (IMP_GRAPH_SOURCES → IMP_EXEC_SOURCES).
make verify-fast green.

Phase 1 (soft PR) of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 closeout

After Tasks 1-3 are merged (and Task 4 either merged or deferred):

- [ ] **Step 1: Run the full verification suite at the phase boundary** (per spec §5)

Run:

```bash
make verify
```

Expected: full GTest suite passes in ~5min.

- [ ] **Step 2: Capture a perf snapshot** (advisory, per spec §5)

Run:

```bash
scripts/gen_perf_baseline.sh
git diff tests/perf_baseline.json
```

If the diff is within ~3% decode / ~5% prefill of pre-Phase-1 numbers,
no action. If outside, document the surprise in the next Phase plan —
since Phase 1 only deleted dead code and added docs, any non-zero perf
delta is variance and should be noted.

- [ ] **Step 3: Update MEMORY.md** (per spec §5)

Add an entry under "Architecture" or "Workflow feedback" pointing to
the completed Phase 1 work. Note that `src/graph/op.{h,cpp}` and
`graph.{h,cpp}` no longer exist (in case any memo references them) and
that `src/graph/` was renamed to `src/exec/` (if Task 4 landed).

- [ ] **Step 4: Mark Phase 1 closed in the roadmap spec**

Edit `docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md`.
At the top of the Phase 1 section, add:

```markdown
**Status (2026-MM-DD):** Closed. PRs #<N1> (Graph IR delete), #<N2>
(diagram merge), #<N3> (architecture.md narrative), [#<N4> (rename)].
```

- [ ] **Step 5: Commit the closeout**

Run:

```bash
git add MEMORY.md docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md
git commit -m "docs(arch): close Phase 1 (Lügen entfernen) of refactor roadmap

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Phase 2 ("Attention-Dispatcher entrümpeln") may now begin. A new
implementation plan is required — invoke the writing-plans skill with
the Phase 2 section of the spec as input.
