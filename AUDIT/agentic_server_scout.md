# Agentic Server Hardening — Phase 0 Scout / Ground-Truth Audit

**Date:** 2026-06-23 · **Base:** `main` @ v0.12.2 · **Method:** 3 parallel read-only scouts over
`tools/imp-server/` + `src/runtime/` + `src/memory/` + `src/compute/`. All findings carry file:line.
**No source was edited.** This report decides *whether and where* to edit.

## TL;DR — the May-2026 audit is largely stale

Of the five documented gaps, **only ONE is genuinely open as written** (per-request spec toggle).
Two are already CLOSED, two are PARTIAL (the hard part already shipped). The scouts also surfaced
**three items not in the original doc**, including a latent **correctness hazard** (persisted prefix
cache key) that outranks most of the planned work.

| # | Documented gap | Reality | Verdict |
|---|---|---|---|
| 1 | `/v1/messages` streaming is synthetic | Real per-token SSE since #754; obsolete-comment at `main.cpp:196` | **CLOSED** |
| 2 | No `cache_control` support | Coarse-boolean pin + correct usage accounting work; breakpoint semantics missing | **PARTIAL** |
| 3 | No per-request spec-decode toggle | Global-only; no request-struct field | **OPEN** |
| 4 | MoE nondeterminism, no deterministic mode | Deterministic MoE kernels exist + wired to `runtime.deterministic`; not per-request/API-exposed | **PARTIAL** |
| 5 | No p50/p99 latency histograms | Prometheus `/metrics` + TTFT & E2E histograms already exist; ITL/queue-wait + agent counters missing | **PARTIAL** |

**New findings (not in the doc):**
| N | Finding | Severity | Cite |
|---|---|---|---|
| N1 | Persisted prefix-cache header validates KV **geometry+dtype only**, NOT model/tokenizer identity → two same-geometry models can serve each other's KV silently (wrong output) | **HIGH (correctness)** | `kv_cache_manager.cpp:1003-1014`, header `:851-860` |
| N2 | Non-stream requests have **no client-disconnect detection** → dropped client wastes compute until `request_timeout` (streaming is fine) | MEDIUM | `handlers.cpp:1484-1499` vs streaming `:1951-1952` |
| N3 | No `ping` keepalive on Anthropic SSE → silence during long prefill until TTFT | LOW | absent repo-wide |
| N4 | Schema-constrained **tool-argument generation** not implemented — mask bypassed in tool body, only post-hoc best-effort validation | MEDIUM (quality) | `preamble_gate.h:191`, `schema_constrain.cu:388`, `tool_call.cpp:686` |

---

## Phase 1 — Real `/v1/messages` streaming → **CLOSED**

- Route comment already states "native incremental SSE (real per-token, not synthetic replay)"
  (`main.cpp:196-197`). Path: `handle_messages` (`handlers.cpp:4528`) builds a real `imp::Request`
  with `stream=true` and stays on per-step decode (`:4650`, #754), submits (`:4686`), registers a
  chunked provider → `run_anthropic_stream_` (`handlers.cpp:3811`) which runs the same `pop_token()`
  loop as OpenAI and emits events as tokens arrive. TTFT captured on first token (`:4070-4073`).
- **Event coverage — all real, none faked:** `message_start` (`:3851`), `content_block_start`
  text/thinking/tool_use (`:3873/:3885/:3935`), `content_block_delta` text/thinking/input_json
  (`:3895/:3905/:3948`), `content_block_stop` (`:3861`), `message_delta` (`:4499`), `message_stop`
  (`:4503`). **Missing: `ping`** (N3).
- **usage:** `input_tokens` in `message_start` (`:3849`); `output_tokens` incremental & accurate
  (`n_output_tokens++` `:4069`, shipped final `message_delta` `:4489`).
- OpenAI `/v1/chat/completions` shares the identical real-streaming infra (`run_chat_stream_`
  `handlers.cpp:1793`, `pop_token` `:1969`). **CLOSED** too.
- **Action:** none required. Optional low-risk nicety: emit `ping` after `message_start` / on the
  empty-`pop_token` branch (`handlers.cpp:4029`).

## Phase 2 — Prompt caching / `cache_control` → **PARTIAL**

Working end-to-end but as a **coarse boolean**, not the Anthropic breakpoint model.
- `has_cache_control()` (`anthropic.cpp:269-288`) = boolean OR over any block carrying the key;
  **position deliberately ignored** (`:264-268`) → `oai["cache_prompt"]=true` (`:378-379`).
- Flows to `req->pin_kv_prefix`; pin fires at `finish_request` → `pin_prefix(req->id, full_blocks)`
  (`engine.cpp:394-398`) → `KVCacheManager::pin_prefix` (`kv_cache_manager.cpp:632-683`), FIFO budget
  `prefix_pin_budget_pct`=25% (`config.h:425`). Pin protects the prefix for *future* requests.
- **Usage split is correct & surfaced:** `cached_tokens` (`scheduler.cpp:87`), non-stream OAI
  (`handlers.cpp:1730-1742`), stream OAI (`:2618-2641`), Anthropic mapping → `cache_read/creation_input_tokens`
  (`anthropic.cpp:484-502`). *Caveat: Anthropic streaming usage path not directly traced — verify in P2.*
- **Missing vs spec:** (a) `cache_control.type` value unused; (b) `ttl` not parsed (no `"ttl"` anywhere);
  (c) 4-breakpoint limit unenforced; (d) per-breakpoint boundary collapsed to whole-prompt; (e) → see N1.
- **Smallest changes:** parse at `anthropic.cpp:269-288`, carry breakpoint token-offset on `imp::Request`,
  pass `offset/block_size` as `num_blocks` at `engine.cpp:398` (`pin_prefix` is already block-granular).

## NEW N1 — Persisted prefix-cache key lacks model identity → **HIGH / correctness**

- In-process key = block content hash only (`compute_block_hash` FNV-1a over token IDs,
  `kv_cache_manager.cpp:222-232`) — safe in-process (one model per process, per-instance table).
- **Disk-persisted cache is the hazard:** `save/load_prefix_cache` (`:862`, `:977`) persist token-hash→KV;
  load-time header validation (`:1003-1014`) checks only `n_layers/n_kv_heads/head_dim/dtype/block_bytes/magic/version`.
  **No model fingerprint, no tokenizer hash.** Two different models with identical KV geometry (common across
  same-family fine-tunes) would match each other's token-hashes and serve **wrong KV silently**.
- **Fix:** add `model_fingerprint`/`tokenizer_hash` to `PrefixCacheHeader` (`:851-860`), reject mismatch at `:1003`.
  Cheap, low-risk, and removes a silent-corruption class. **Recommend doing this first.**

## Phase 3 — Streaming tool_use → **PARTIAL** (blocks CLOSED, schema-constraint OPEN)

- **CLOSED:** `emit_tool_use` (`handlers.cpp:3929`) → `content_block_start {type:tool_use,id,name,input:{}}`
  (`:3935`), chunked `input_json_delta` 48-byte slices (`:3948`), `content_block_stop` (`:3956`);
  monotonic block index (`:3871/:3932`, re-scan for next tool `:4283-4294`); `stop_reason:"tool_use"`
  (`:4479`). Parallel/sequential tool blocks supported. OpenAI mirror (`:2273/:2299`).
- **OPEN (N4):** tool args are free-generated then post-hoc validated, **not** schema-constrained. In tool
  body the `PreambleGate` enters `TOOL_BODY` and absorbs everything (`preamble_gate.h:191`); `apply_mask`
  early-returns while `preamble_.active()` (`schema_constrain.cu:388`, `json_constrain.cu:420`) → FSM mask
  fully bypassed. Only `validate_tool_call` (`tool_call.cpp:686-729`) checks required keys/top-level types
  post-hoc; invalid args are still streamed (`handlers.cpp:4271-4275`).
- **Cost to close:** real multi-day change — the gate doesn't know *which* tool opened until the model emits
  the `name`, so mask activation must defer until after `"name"` + a name→schema lookup the engine doesn't
  hold. Today's "free-gen + post-hoc validate" is deliberate (`handlers.cpp:810-814`). **Lower priority** —
  validation already prevents the worst outcomes.

## Phase 4 — Cancellation & slot reclamation → mostly **CLOSED**, one gap (N2)

- **Cooperative cancellation CLOSED:** `ServerRequest{std::atomic<bool> cancelled}` (`batching_engine.h:25-69`);
  checked between steps (`batching_engine.cpp:151-158`) → frees KV (`free_sequence`) + SSM/GDN state eagerly
  (≤1 step latency), erased from active (`:282`). No leak.
- **Cancelled-request cache safety CLOSED:** cancel path bypasses `finish_request` so `register_block_hashes`/
  `pin_prefix` never fire (`engine.cpp:388-400`) → no partial/incomplete KV ever enters the hash table.
- **Eviction safety CLOSED (P2/#538):** reclaim removes hash mapping before freeing (`kv_cache_manager.cpp:600-608`);
  `-1` sentinel + ref-count discipline (`:291-303`); evicted prefix cleanly recomputes, never wrong KV.
- **Streaming disconnect CLOSED:** `!sink.is_writable()` → `cancel()` (`handlers.cpp:1951-1952`).
- **OPEN (N2):** **non-stream has no disconnect detection** (`handlers.cpp:1484-1499` only checks
  `request_timeout`). httplib buffered handlers expose no `DataSink`, so this needs a connection-liveness
  hook or periodic poll. Medium risk/effort.
- Note: recurrent (SSM/GDN) models disable prefix caching entirely (`engine_kv_cache_init.cpp:123-126`) →
  `cache_control` is a silent no-op there; worth surfacing to clients.

## Phase 5 — Per-request spec toggle → **OPEN**; determinism mode → **PARTIAL**

- **Spec toggle OPEN:** master switch is process-global `runtime_config_.speculative.ngram`
  (`engine_scheduler.cpp:867`, gate `engine_spec_ngram.cpp:143-184`). Neither `ChatRequestParams`
  (`handlers.cpp:~30-62`, parse `:681-779`) nor `imp::Request` (`request.h:27-125`, which already holds spec
  *counters* `:59-68`) carries an enable override. **Smallest change:** add `optional<bool>` to `Request`
  (`request.h:68`), thread through `ChatRequestParams` + body parse, change gate at `engine_scheduler.cpp:867`
  to `override.value_or(global)`. **Risk:** must stay inside the `batch==1`/greedy gate or it silently no-ops;
  toggling under continuous batching changes co-batched kernel path — evaluate per-request inside the existing guard.
- **Determinism PARTIAL:** deterministic MoE kernels EXIST — `moe_scatter_deterministic_kernel_impl`
  (`moe_routing.cu:529`), `moe_fused_permute_deterministic_kernel` (`:633`); selected when
  `process_diag_deterministic_gemm()` (sites `:804/:869/:998`). Driven by `runtime.deterministic`
  (`config.h:39`, ⇒ `deterministic_gemm`, `config.cpp:108`; promoted `engine.cpp:470-471`). Atomics
  nondeterminism lives in `moe_scatter_kernel_impl` (`atomicAdd` `:519`) + `moe_fused_permute_kernel`
  (bucket scatter `:606`). **Works today via `--set runtime.deterministic=true`.** Gaps: (a) no independent
  "deterministic routing only" switch (rides `deterministic_gemm`); (b) process-global, NOT per-request —
  and **per-request is unsafe under continuous batching** (kernel selection is global per launch; can't
  co-batch det + nondet). Cost: cheap at decode/batch-1, severe at large prefill (single-block permute,
  historical 536ms stall, #682). **Recommend: expose as documented server flag, NOT per-request.**

## Phase 6 — Observability → **PARTIAL** (better than the doc assumes)

- `/health` (`main.cpp:180`) + `/metrics` (`main.cpp:214`), auth-exempt. `handle_metrics` (`handlers.cpp:3420`)
  emits **real Prometheus** (`# HELP/# TYPE`, `text/plain; version=0.0.4` `:3509`).
- **Histograms already exist:** `struct LatencyHistogram` 11 buckets (`handlers.h:63-83`); `imp_request_duration_seconds`
  (E2E) + `imp_ttft_seconds` (TTFT) emitted `:3491-3493`. Counters: requests/failed/prompt/completion/cached
  tokens/model_loads; gauges: uptime/last_duration/last_ttft/queue_depth (`:3428-3507`, struct `handlers.h:85-98`).
- **Missing for agents:** ITL & queue-wait histograms; cancellation counter (cancels happen `:1489/:1952/:3031`
  but never counted); spec-accept-rate (tracked engine-side `spec_stats_`/`Request::spec_accepted` but only
  logged to stderr `engine_spec_ngram.cpp:57-65` — needs a new C-API getter to surface); cache-miss counter;
  active-vs-queued split.
- **Insertion point:** per-request timing in `ChatRequestContext` (`handlers.cpp:85-97`, `t_start` `:91`,
  `t_first` `:2027`, record block `:2654-2662`). Add `metrics.itl.observe()` there; queue-wait needs an
  arrival timestamp at request entry (none today). Mostly low-risk host-side additions.

---

## Dependency-ordered plan (revised to ground truth)

Ordered by value × (1/effort × 1/risk), respecting deps. **Each step gated by the §Benchmark protocol;
GPU work waits for a free card.**

1. **N1 — persisted prefix-cache key hardening** (correctness, ~½ day, low risk). `kv_cache_manager.cpp`.
   Do first: removes a silent-wrong-KV class. No perf impact, no GPU needed beyond a correctness test.
2. **Phase 7 (partial) — agent benchmark harness** in `tools/` (TTFT/ITL p50/p99 under 1/4/16/64 concurrency,
   warm-vs-cold cache, cancel cleanup). Needed to *gate* everything below and to prove there's nothing to fix
   in already-CLOSED streaming. GPU-bound → schedule on a free card.
3. **Phase 6 — metrics gaps** (ITL + queue-wait histograms, cancellation counter, cache-miss, active/queued
   split; spec-accept needs a small C-API getter). Cheap, additive, <1% overhead target. Mostly host-side.
4. **N2 — non-stream disconnect detection** (medium; httplib liveness hook at `handlers.cpp:1484`).
5. **Phase 5a — per-request spec toggle** (small plumbing; insertion points identified). Validate token-identity.
6. **Phase 5b — determinism: expose `runtime.deterministic` as a documented server flag** + 5-run identical-output
   proof + perf-delta log. (Per-request determinism = NON-GOAL; unsafe under continuous batching.)
7. **Phase 2 — cache_control breakpoint semantics** (type/ttl parse, 4-limit, per-boundary pin via offset
   plumbing). Boolean already delivers most value → refinements are lower ROI.
8. **N3 — `ping` keepalive** (trivial, optional).
9. **Phase 3 — schema-constrained tool args** (multi-day, real design change; post-hoc validation already
   mitigates). Lowest priority unless a concrete schema-violation failure is observed.

**Phase 1 = no work** (already real streaming).

### Invariant check
No step above touches hot-path kernels except Phase-5b (which only *selects* the existing deterministic MoE
variant behind an opt-in flag — zero default-path change). Decode/prefill throughput is not on the line for
steps 1–8. The −2% `tg256`/prefill gate is only at risk if Phase-6 histogram observe() is mis-placed in the
decode loop — keep it in the per-request record block, not per-token.

**STOP — Phase 0 complete. No edits made. Awaiting go/no-go per phase.**
