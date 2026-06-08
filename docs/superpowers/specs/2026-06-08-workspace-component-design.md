# Workspace component — D2 step 3 (first hot-path runner extraction)

Date: 2026-06-08

## Problem

`GraphExecutor` owns the engine's scratch-memory management: ~13 buffer members
(persistent + shared workspace pointers/sizes, the per-phase shared sizes, the
fp32 accumulator, the attention-scores buffer, the NVFP4 dequant workspace, the
decode/prefill workspace swap state) and ~25 methods in `executor_workspace*.cu`
(`allocate_*`, `configure_*_workspace`, `compute_shared_sizes`, `use_workspace`,
`resize_workspace`, `free_buffers`, `ensure_logits_pinned`,
`release_moe_batch_buf`, `workspace_estimate`). This is the most-shared piece of
GraphExecutor state, read all over the forward path.

D2 step 3 extracts it into a `Workspace` class. Unlike the QuantPipeline
extraction (init-time, hot path untouched, `diff=0`), **this touches the forward
hot path**: the attention/ffn/moe/ssm/forward TUs read `shared_workspace_`,
`fp32_accum_buf_`, `attn_scores_buf_` and call `configure_*_workspace()` — those
become `ws_.x()` calls. The `diff=0` guarantee does not hold; correctness is
canary-gated instead.

## Scope (what the analysis settled)

Two things that live in `executor_workspace*.cu` are NOT workspace and stay on
`GraphExecutor`, which keeps the extraction clean and avoids needless churn:

- **`view_tokens(buf, n)`** — a stateless one-liner (`return slice_rows(buf, n);`)
  read 121× across 8 TUs. It carries no workspace state, so it stays a
  GraphExecutor helper. Moving it would churn 121 hot-path sites for zero benefit.
- **`layer_has_moe / _gdn / _ssm / _attention / _dense_ffn`** — per-layer model
  classification (`model_->layer(i).moe_gate.data != nullptr`). Model territory
  (ModelProfile-adjacent), not workspace. Left in place.

**In scope (moves into `Workspace`):** the buffer members + the allocation/sizing
lifecycle —
- members: `persistent_workspace_`(+size), `shared_workspace_`(+size,+max_tokens),
  `attn_shared_size_`, `ffn_shared_size_`, `moe_shared_size_`, `ssm_shared_size_`,
  `fp32_accum_buf_`, `attn_scores_buf_`(+size) + `attn_scores_` Tensor,
  `nvfp4_dequant_ws_buf_`(+size), `decode_persistent_size_`, `decode_shared_size_`,
  `active_workspace_`, `saved_prefill_ws_` (the `SavedWorkspace` swap state).
- methods: `allocate_persistent_workspace`, `allocate_shared_workspace`,
  `allocate_decode_workspace`, `allocate_auxiliary_buffers`, `allocate_workspaces`,
  `compute_shared_sizes`, `configure_attn_workspace`, `configure_ffn_workspace`,
  `configure_moe_workspace`, `configure_ssm_workspace`, `use_workspace`,
  `resize_workspace`, `free_buffers` (workspace portion), `ensure_logits_pinned`,
  `release_moe_batch_buf`, `workspace_estimate`.

The exact member/method set is finalized during implementation (build catches any
straggler); this list is the cohesive cluster.

## Goal & constraints

- Extract `Workspace` into `src/exec/workspace.h` (+ the existing
  `executor_workspace*.cu` become `Workspace::` methods).
- **Behaviour-neutral**, **zero hot-path overhead**: the forward path reads buffers
  through `inline` accessors (`ws_.shared()`, `ws_.fp32_accum()`,
  `ws_.attn_scores()`, …) that compile to the same member load.
- The `diff=0` hot-path check does NOT apply (the forward TUs change). Gated by the
  4-arch coherence canary + verify-fast + a decode-perf backstop.

## Architecture

`GraphExecutor` owns `Workspace ws_;`. `Workspace` gets the inputs it needs
(`const Model*`, `VRAMAllocator*`, `const RuntimeConfig*`, plus the
`compute_dtype_` / `max_tokens_` it sizes against) via an `init(...)` call from
`GraphExecutor::init`, mirroring the QuantPipeline pointer-context pattern.

Access pattern (zero overhead):
```cpp
// Workspace exposes inline accessors; the hot path reads through them.
void* shared() const { return shared_workspace_; }
void* fp32_accum() const { return fp32_accum_buf_; }
const Tensor& attn_scores() const { return attn_scores_; }
// ... etc.
```
Hot-path call sites change mechanically:
- `shared_workspace_` → `ws_.shared()` (~17 sites)
- `fp32_accum_buf_` → `ws_.fp32_accum()` (~23 sites, 5 TUs — the gemma-4 fp32 path)
- `attn_scores_buf_` / `attn_scores_` → `ws_.attn_scores*()` (~7 sites)
- `configure_attn_workspace(n)` → `ws_.configure_attn(n)` (and ffn/moe/ssm) (~9)
- `persistent_workspace_` / `use_workspace` / `resize_workspace` → `ws_.x()` (~10)

`GraphExecutor` keeps a couple of thin accessors it already exposes publicly
(`active_workspace()`, the attn-scores width getter at executor.h:190) — they
delegate to `ws_`.

## Data flow

Init: `GraphExecutor::init` → `ws_.init(model, alloc, rcfg, compute_dtype, max_tokens)`
→ `ws_.allocate_workspaces(...)`. Forward: each phase calls
`ws_.configure_<phase>(n)` at its top (as today) and reads buffers via the inline
accessors. Teardown: `ws_.free_buffers()` from `GraphExecutor::free_buffers`.

## Error handling

Unchanged — allocation failures throw / log exactly as today; no status-return
conversion.

## Testing

- `make build` clean.
- **4-arch coherence canary** (the behaviour gate, since hot path changed):
  Qwen3-8B Q8_0 (dense), Qwen3-30B-A3B-NVFP4 (MoE — exercises moe workspace),
  Nemotron-3-Nano-30B (SSM/GDN — exercises ssm workspace), gemma-3-12b (dense
  gemma) — each coherent (Paris), no `CUDA error / falling back / NaN`. The
  Qwen3-30B native-MoE-cache count must still match main (144).
- **Decode-perf backstop**: Qwen3-8B Q8_0 `tg` within noise of main (a mis-sized
  workspace silently degrades perf or forces a fallback). Warm, isolated single
  compare; flag if >5% down.
- `make verify-fast` gtest filter green.
- (No `diff=0` check — the forward TUs intentionally change.)

## Out of scope

- `view_tokens` (stays a GraphExecutor helper) and `layer_has_*` (model
  classification, stays).
- Any other component (MoeRunner, Attention/Ffn/Ssm runners).
- Splitting the `attn_scores`/`fp32_accum` buffers' *allocation policy* — moved
  verbatim, not redesigned.

## Risks

- **Larger, hot-path churn (~66 sites).** Mechanical, but unlike a verbatim move a
  wrong accessor (e.g. returning the wrong buffer) compiles fine and fails at
  runtime. Mitigation: the accessor returns the exact same member; the 4-arch
  canary + perf backstop catch divergence. Per-TU, build between.
- **Workspace sizing bugs are silent** — manifest as IMA / OOM / perf regression,
  not compile errors. Mitigation: the methods move verbatim (no sizing-logic
  edits); the MoE + SSM canary models exercise the moe/ssm workspace paths; the
  perf backstop catches a silent fallback.
- **Init ordering** — `ws_.init` must run where `allocate_workspaces` runs today
  (after model load, before the first forward). Mirror the current call site.
