# Structural audit — 2026-07-10

Sixth structural pass (prior: `structural_debt_2026_07_07.md`, whose findings
#888–#897 are all closed — the server-debt backlog from pass #5 was fully worked
off). Since that pass: ~40 PRs (#898–#940), so this sweep focused on the surfaces
they touched — the VRAM ordering/balloon rework (#923/#924/#926/#927/#935), the
FA2 hd=256 arc (#930–#933), GDN verify-capture (#933), the server thinking-state
pipeline (#937/#939), the C++23 migration (#916/#919), and the deterministic-GEMM
validation (#929). Method per the codebase-audit skill: five fan-out sweeps
(server, VRAM/init, attention/capture, config/docs, dead-code/comments), then
every candidate re-verified against the code before filing.

File-size gate: 0 violations, 34 warnings (27 allowlisted). New over the warn
threshold since pass #5: `tools/imp-server/handlers_chat.cpp` (623 code-LOC,
warn>600 — it was deliberately relocated under the gate in #901 and has crept
back) and `src/exec/executor_forward_moe.cu` (521, warn>500). Neither is near
hard-review; no action.

## Confirmed findings → issues

| # | Finding | Issue |
|---|---------|-------|
| 1 | `/v1/responses` **streaming** updates no metrics (`requests_total`, token counters, TTFT/duration histograms — only `requests_cancelled` at `handlers_responses.cpp:343`) and sends no ~10s idle keepalive (chat: `handlers_chat_stream.cpp:263-270`, messages: `handlers_messages.cpp:331`, responses: bare `continue;` at `:356-358`). Third + fourth concrete drift instance in the triplicated SSE loop family (#892 was the first two) — the inner state machines are shared (`stream_pipeline.h`/`reasoning_split.h`/`tool_stream_filter.h`) but the outer per-token loop is still hand-copied 3×, plus 5 hand-copies of the OpenAI-params→`imp::Request` field mapping (`handlers_chat.cpp:44-88`, `:301-342`, `handlers_chat_core.cpp:777-819`, `handlers_messages.cpp:767-801`, `handlers_responses.cpp:673-705`). Extraction of the shared loop is no longer speculative — the drift class keeps producing real bugs. | [#941](https://github.com/kekzl/imp/issues/941) |
| 2 | Pre-upload KV reserve (`engine_weight_upload.cpp:125-136`) uses raw `dtype_size(kv_cache_dtype)`: `qtype_elem_bytes` has no NVFP4/MXFP4_KV case → **0 bytes** reserved (offload keeps too many experts on-device, KV pool collapses toward the #927 floor); INT4 → 2× over-reserve; scale bytes omitted. Same bug class `vram_budget.cpp:220-241` already fixed for itself. | [#942](https://github.com/kekzl/imp/issues/942) |
| 3 | `workspace_estimate()` (`executor_workspace.cu:351-359`) still adds up to 256 MiB S-matrix for every non-MoE model, while the allocator skips it on FA2-served configs since #932 (`executor_workspace_buffers.cu:326-335`). The estimate feeds the pre-upload reserve and the #926 balloon floor, so the #932 reclaim never reaches the planners. | [#943](https://github.com/kekzl/imp/issues/943) |

## Confirmed nits → fixed in the audit-cleanup PR

- **Dead code:** `MemAccount::total_vram_` (write-only: `mem_account.h:103`,
  `mem_account.cu:67`, zero reads across src/tests/tools); `KVCacheManager::swa_window()`
  / `swa_slack()` public getters (added #924, zero callers — the backing members are
  live internally; independently confirmed by two sweeps); redundant local forward
  declaration of `run_chat_stream_` (`handlers_chat_stream.cpp:37`, already declared
  in `handlers_internal.h:163`).
- **Stale comments:** four sites still describe `attention.fa2_hd256` as
  "default off"/"opt-in" after the #932 default-ON flip
  (`attention_fmha_sm120.cu:1839`, `process_diag.h:68`,
  `executor_attention_prefill.cu:48-49`, `executor_attention_internal.h:39`);
  `gemm.cu:380` says FP8-KV "forces this path model-wide on head_dim!=128 models"
  (post-#932 the gate is "not FA2-served"); `handlers_internal.h:128-132` documents
  `g_in_anthropic_shim` as messages-only (now also set by `handle_responses` and
  `handle_count_tokens`).
- **Latent default mismatch:** `process_diag.cpp:36` seeds
  `attention_fa2_hd256 = false` while the config default is `true` — inert today
  (overwritten from cfg at `:93`) but a wrong fallback if ever read pre-seed.
- **Docs drift (hd=256 arc):** `docs/attention-dispatch.md` still presents
  cuBLAS/WMMA as the default hd=256 prefill path (lines 11-12, 20, 37, 46, 48 —
  pre-#930/#932); `docs/architecture.md:100-101,162` still calls hd=256 a blanket
  FA2-decline; `docs/architecture.md:163` claims `RuntimeConfig::current()` is a
  global singleton — that accessor was retired (Phase 5 Track D; per-Engine
  snapshot, `engine.h:287`). README.md was already correct, making the docs
  internally inconsistent.

## Noted, not actioned (low priority)

- `conv_ch = ssm_inner_size + 2*ssm_group_count*ssm_state_size` is derived
  identically at **5** sites (`engine_kv_cache_init.cpp:292`,
  `engine_weight_upload.cpp:149`, `vram_budget.cpp:162`,
  `executor_workspace_buffers.cu:739`, `gguf_tensor_assign.cpp:156`), and the full
  per-batch SSM-state footprint formula is hand-copied at the first two + the
  authoritative alloc. No drift bug found yet — a one-line `ModelConfig` accessor
  becomes worth it the moment one appears (or fold into the #942 fix if it touches
  the same lines).
- `compute_native_cache_demand()` re-walks every layer 3× per init (resolver ×2 +
  weight-upload balloon). Init-time only; not worth caching yet.
- `/v1/completions` carries a fourth, bespoke think-strip implementation
  (`handlers_chat.cpp:394-503`) — documented as deliberate (`:386-393`, no
  `reasoning_content` field on that API). Consistency note only.

## Checked and NOT flagged — do not re-chase

- **Admission control (#888 fix) is complete**: `is_inference_endpoint`
  (`main.cpp:26-29`) covers all five capacity-consuming routes incl.
  `/v1/responses` and `/v1/embeddings`; count_tokens/tokenize/detokenize are
  deliberately outside (no engine submission). No bypass.
- **Thinking-state pipeline (#937 reconcile vs #939 force) is complementary, not
  duplicated**: `force_thinking` acts at render time (explicit requests on
  closed-default templates), `reconcile_thinking_with_prompt_tail` post-render
  (prompt tail = ground truth, explicit requests left untouched). No dead
  parameter, no contradictory branch.
- **Old hd=256 WMMA path is LIVE**, not orphaned: `fmha_sm120_prefill` `case 256`
  is the dispatch fallback for `fa2_hd256=false` and FA2-declined chunk
  continuations (`q_offset>0`), with live tests (`FmhaSm120Test.HD256_LongSeq*`).
- **FP8-KV forcing removal (#932) left no dead branch**: the forcing is still
  legitimately live for `!fa2_serves_attention` (sinks/heterogeneous/flag-off),
  hd=256 folded into `fa2_hd_ok` (`engine_init_resolver.cpp:237`).
- **No stale "GDN can't capture" gate** after #933; the #858 fail-loud guard is
  gone, `chunk_capture_supported` accepts hd=256 hybrids. No stale "N=32
  unsupported under capture" comment after #937.
- **Config surface is clean**: every key added/changed since 07-07 (`fa2_hd256`,
  `fp8_tile`/`fp8_tile_gqa`, `swa_sizing`, all six `[rope]` keys, all three
  `[vram]` keys) is parsed AND value-read on a live path; all 145 dotted keys are
  present in `imp.conf.example` with defaults matching `config.h`; `docs/usage.md`
  defaults correct. The rope override lives in `engine_init_resolver` → live on
  the C-API path too.
- **`MemAccount::device_peak_used` (pass-#5 #894) was actually removed** — does
  not linger. `VramOwned` still does not exist (recurring hallucination).
- **#926 balloon fields all live** (`mandatory_sf_bytes`/`mandatory_moe_bytes`
  written in vram_budget, consumed in pre_dequant phase 3); ordering comments
  (`executor_pre_dequant.cu:57`, `engine_weight_upload.cpp:277`) are consistent
  with the new reserve-before/build-last order.
- **Deterministic-GEMM (#929) left no dead old path**: the remaining `results[0]`
  uses (`gemm.cu:428-429`, `:284`) are legitimate no-scratch fallback /
  post-heuristic reselect, not the old blind path.
- **No debug code left enabled** in the touched set (every `fprintf`/printf is
  flag-gated); **no C++23 half-migrations**; **no stale TODOs** referencing closed
  issues; **no duplicated new helpers** (`align_up` etc. single-sourced).
- **SWA sizing (#924)**: `layer_swa_window()` is the single centralized window
  predicate shared by budget/write/attention paths; no duplicated group logic.
- `workspace_estimate()` itself is single-sourced (3 call sites, no
  reimplementation) — the S-matrix gate (#943) is its only staleness.
