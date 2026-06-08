# Structural audit — 2026-06-09

Scope: a fresh codebase-wide pass over the areas the 2026-06-08 audit
(`structural_debt_2026_06_08.md`, D1–D4) did **not** cover. That audit focused on
`src/exec/` and the `GraphExecutor` god-object; this one targets `src/compute/`
(40,942 LOC — 40 % of the codebase, previously untouched), the loader family in
`src/model/`, header/dependency hygiene, and the test suite. Counts are verified
with grep/wc, not estimated. Diagnostic + an opening cheap-sweep PR.

## TL;DR

After D1–D4 and the D2/D3 PRs (#621–#631) the `exec/` layer is structurally
healthy and — pleasantly — the **header/dependency layer is clean**: no module
cycle, 100 % `#pragma once`, the public C-API boundary leaks nothing. The real
untouched debt lives in **`src/compute/`** (god-files + heavy attention-variant
duplication) and in **`weight_map.cpp`** (an 858-line matcher ladder). The two
findings deferred on 2026-06-08 are still open.

A note on method: several agent-reported "dead flags/files" were **false
positives** caught by re-verifying with the right search scope (the exact trap
documented on 2026-06-08). They are recorded under "Refuted" so they are not
re-chased.

---

## C1 — `src/compute/` paged-attention duplication (highest new value)

Six variants — `attention_paged{,_nvfp4,_nvfp4_tc,_int8,_fp8,_int4}.cu`
(5,261 LOC together) — each reimplement the **same per-token online-softmax loop**
(m/l/O accumulation, KV-tile load, score reduction, block range helpers). Only the
dequant of K/V differs (~2,000 LOC of repeated core loop). The drift risk is real:
a softmax/range fix has to be applied six times.

**Fix:** an `attention_paged_core_loop.cuh` template parametrised by dequant
traits — exactly the pattern `gemv_dp4a_traits.cuh` already uses for GEMV. High
effort, own brainstorm→spec→plan cycle, each variant verified against its current
output. Rank: high pain × high risk.

## C2 — `src/compute/` god-files (conflated responsibilities)

| File | LOC | Conflated concerns |
|------|-----|--------------------|
| `gdn.cu` | 2,061 | delta-rule scan + RMSNorm/gated-SiLU + V-head layout conversion |
| `gemm.cu` | 1,806 | GEMM dispatch + 12 GEMV kernel variants + MoE gate/up-fused |
| `sampling.cu` | 1,763 | 7 sampling algorithms + 3 penalty fns + logprob computation |
| `json_schema.cpp` | 1,333 | JSON parser + schema tree + RegexNfa (Thompson) + GBNF transform |

All are verbatim-extractable (D3-style, low risk): e.g. `gdn_scan.cu` /
`gdn_activation.cu` / `gdn_layout.cu`; pull GEMVs out of `gemm.cu`; lift
`regex_nfa.cpp` out of `json_schema.cpp`. Size alone is not the smell —
*conflated* responsibilities are (`jinja.cpp` 2,629 and the FA2 kernels stay:
single responsibility). Rank: medium pain × low risk.

## C3 — `weight_map.cpp` 858-line matcher ladder

`apply_weights()` (lines 312–1170) is 23 sequential `if (!matched && …)` string-
matching blocks. Concrete duplication inside it:
- GPTQ field assignment duplicated (self_attn vs mlp, ~58 LOC, lines 1079–1138).
- NVFP4 fused-scale slicing duplicated (qkv vs gate_up, ~40 LOC, lines 668–730).
- `moe_gate` assigned at 4 separate sites (Gemma-4 / Mixtral / DeepSeek / gpt-oss).

**Fix:** a matcher-registry table (`{pattern, field-assignment}`) instead of a
linear scan; extract the slicing/GPTQ helpers. Rank: medium pain × low risk.

## C4 — Loader config-parse paths diverge (GGUF ↔ SafeTensors)

GGUF parses arch-specific config **inline** (gemma4 SWA array, LongRoPE,
`rope_local_theta`; `gguf_loader.cpp:1106–1400`); SafeTensors delegates to
`hf_config_loader.cpp` + `apply_arch_defaults()`. The Gemma SWA setup has no
SafeTensors equivalent — an arch change can land on one path only. This is the
exact #514/#516 bug class D1's ModelProfile was meant to retire, surviving in the
*loaders* rather than the hot path. **Fix:** move metadata-agnostic config logic
into a shared post-load routine both loaders call after arch detection. Rank:
medium pain × high risk.

## C5 — Deferred 2026-06-08 findings, still open (re-verified)

- **Dead `WeightCaches::q4k_imma` cache — REMOVED (follow-up).** Re-confirmed
  never populated (no `use_q4k_imma = true`, no insertion anywhere). **Correction
  to the "~1,200 LOC tile/reorder stack" estimate:** the call-graph shows the
  *live* `mmq_q4k_imma_gemm` (the `q4k_imma_prefill` path) calls
  `mmq_q4k_imma_reorder` + `mmq_q4k_imma_tile` on-the-fly — those kernels are
  **live, not dead**. The only dead code was the unused cache struct itself
  (`Q4kImmaCacheEntry` map + `use_q4k_imma` + `q4k_imma_bytes`, ~30 LOC in
  `weight_caches.h`) and its always-empty free loop in
  `executor_workspace_buffers.cu`. Removed those; the kernels stay.
- **Unbridged config keys** — `server.prefix_cache`, `server.green_contexts`,
  `server.prefix_pin_budget_pct`, `paths.mmproj` are parsed into `RuntimeConfig`
  but **never read** (`grep cfg.server.` finds only the parse lines); the live
  enablement flows through the C-API (`imp_api.cpp`). Silently inert from
  `imp.conf`. Needs a bridge-or-retire decision (user-facing), not a blind delete.

## C6 — Header micro-findings (the layer is otherwise clean)

- `exec/inference_state.h` includes `compute/json_constrain.h` +
  `schema_constrain.h` (~190 LOC each) but only stores `JsonConstrainer*` /
  `SchemaConstrainer*` — forward-declarable. The includes ride transitively into
  every `executor.h` consumer. **(Fixed in this PR — see Progress.)**
- **Dependency inversion:** `compute/weight_dispatch.h` includes
  `exec/weight_handle.h` — `compute` (lower layer) depending on `exec` (higher).
  The only violation of the api→runtime→exec→compute DAG. Fix: split a
  `weight_dispatch_types.h` interface so compute includes only the interface.

## C7 — Test-fixture adoption gap

`SKIP_IF_NO_CUDA()` is defined in **11 files** with three divergent bodies
(`::imp::test::HasCudaDevice()`, a raw `cudaGetDevice` check, an unqualified
`HasCudaDevice()`); `HasCudaDevice()` itself is re-defined in 6 files;
`get_model_path()` in 3. A shared layer (`test_models.h`, `test_model_builder.h`)
exists but only 17 of ~117 test files use it. **Fix:** a lightweight
`tests/test_cuda_skip.h` owning the one `HasCudaDevice()` + macro (kept separate
from the heavy `test_model_builder.h` so host-compiled `.cpp` tests don't pull in
`cuda_fp16`/`half`). **(Fixed in this PR — see Progress.)**

## C8 — Candidate dead kernels (verify launch-by-pointer before deleting)

0 `<<<` launch sites found, each symbol confined to its own TU:
`softmax_sum_kernel`, `softmax_to_pairs_kernel` (`sampling.cu` — note the *device*
variant `softmax_to_pairs_device_kernel` **is** launched at line 868, so only the
non-device pair looks dead); `zero_int32_kernel`, `exclusive_scan_kernel`,
`count_tokens_per_expert_kernel` (`moe_routing.cu`, look like leftovers from an
older routing implementation). Rule out `cudaLaunchKernel`-by-pointer before
deletion. Rank: low pain × low risk.

---

## Refuted (do not re-chase)

Re-verification with `tools/` included and a fresh tree caught these agent
false positives — the same local-scope trap noted on 2026-06-08:

- **`diagnostics.dump_tokens`, `generation.force_bos`, `bench.generate` are NOT
  dead** — all read in `tools/imp-cli/main.cpp` (lines 638 / 631 / 812). The
  search that flagged them omitted `tools/`.
- **`tests/golden/` and `tests/api/test_outputs/` do not exist** (count 0) — the
  "dead artifacts" finding was stale.
- `generation.think_budget` (global field) is the one marginal residue: live use
  is the *per-request* `req.think_budget` (`engine_graph_decode.cpp:87`); the
  global config field appears unread → at most a bridge-or-retire, not a safe
  delete.

## Clean — no action

Public API boundary (`include/imp/`, 0 internal leaks), module DAG acyclic,
151/151 headers `#pragma once`, `CMakeLists.txt` structured per-module, `tools/`
cohesive (no duplicated dequant/tokenizer logic), `executor.h` at 584 LOC post
PRs #629–#631.

---

## Recommended order

1. **Cheap, now:** test-fixture consolidation (C7) + header forward-decls (C6) —
   verbatim, low risk. *(this PR)*
2. **Medium, high value:** `compute/` god-file splits (C2) — D3-style verbatim,
   each behind the coherence canaries.
3. **Biggest:** paged-attention traits refactor (C1) — own brainstorm→spec→plan,
   each variant verified against current output.
4. **Decision, not code:** unbridged config keys (C5) — bridge or retire.

---

## Progress

- **C6 — DONE (this PR):** `inference_state.h` forward-declares `JsonConstrainer`
  / `SchemaConstrainer` instead of including their headers; `executor.cu` gains a
  direct `#include "compute/schema_constrain.h"` (it dereferences
  `schema_constrainer->apply_mask`). `engine_scheduler.cpp` only assigns/null-
  checks the pointers → forward decl suffices.
- **C7 — DONE (this PR):** new `tests/test_cuda_skip.h` owns
  `imp::test::HasCudaDevice()` + `SKIP_IF_NO_CUDA()`; the 11 local macro
  definitions and 6 local `HasCudaDevice()` re-definitions removed and replaced
  with the include. `get_model_path()` dedup (3 files) left for a follow-up.

---

## Verification pass — 2026-06-09 (same session, after shipping C5–C8)

Before acting on the bigger findings each was re-verified against the actual
code + the 2026-06-08 principle **"split on conflation, not size; no speculative
abstraction."** The cheap/genuine items shipped; the big-ticket findings did
**not survive verification** — the fresh fan-out audit over-flagged them (the
same false-positive pattern flagged under "Refuted" above). Evidence-backed
verdicts:

- **C5 — DONE (PR #636).** Bridged at engine init (imp.conf → `EngineConfig`,
  OR semantics); RuntimeConfig defaults realigned to the real (off) defaults so
  no-imp.conf embedders are unaffected. Functionally verified:
  `prefix_cache=true` → engine logs "Prefix caching enabled". This was a genuine
  bug (documented keys silently inert), not just structure.
- **C8 — DONE (PR #635).** 5 dead kernels removed (116 LOC); clean link proves
  no launch site.
- **C3 — partial DONE (PR #637), rest DECLINED.** Real GPTQ field-assignment
  duplication extracted (`assign_gptq_field`, 58→24 LOC). The NVFP4 fused-scale
  slicing is **parallel structure, not duplication** (3-way-by-head vs
  2-way-by-half, correctness-critical with inline bug-history comments) — left
  alone. The full ladder→registry rewrite is high-risk (mis-mapped tensor name
  = garbage output) for cosmetic gain — declined.
- **C1 — REFUTED.** The "~2,000 LOC repeated core loop" is **already factored**:
  `attention_paged_common.cuh` owns `online_softmax_step`,
  `compute_context_range`, `block_token_range`, `compute_kv_tile_bounds`,
  `apply_score_masks`, `crosswarp_reduce_*`, cp.async wrappers — and all 6
  variants `#include` it and call those helpers (4–18 uses each, verified). What
  remains per-variant is format-specific K/V dequant, inherently divergent and
  correctly separated. No traits-template dedup warranted.
- **C2 — REFUTED.** `gdn.cu` / `gemm.cu` / `sampling.cu` / `json_schema.cpp` are
  each one cohesive domain (GDN layer / GEMM / sampling / schema→grammar), not
  conflated god-files — the same call the 2026-06-08 audit already made for
  `gdn.cu` ("domain-cohesive, don't touch"). Size ≠ smell. Splitting would be
  low-value churn against the project's own guidance.
- **C4 — REFUTED as a bug-class.** Both loaders handle Gemma SWA/RoPE-local: the
  SafeTensors path via `hf_config_loader.cpp` (`sliding_window`, `swa_layers`,
  `rope_local_theta` at lines 293/512-556) and GGUF inline from metadata. They
  parse the same concepts format-appropriately — not a divergence bug.
  Consolidating two working format-specific parsers is speculative; declined
  absent a concrete bug.

**Net:** the genuine, safe, high-value debt (C5 bug + C6/C7 hygiene + C8 dead
code + the real C3 dedup) is shipped. The remaining big-ticket findings were
verified non-debt or low-value-high-risk. Lesson reinforced: a fan-out audit
surfaces candidates; each must be re-checked against the real code (with `tools/`
in scope) and the conflation-not-size principle before it is acted on.
