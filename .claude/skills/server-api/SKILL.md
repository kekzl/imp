---
name: server-api
description: Use when working on or testing imp-server and its HTTP APIs — OpenAI/Anthropic endpoints, /v1/chat/completions, /v1/messages, SSE streaming, tool calling, json_schema/constrained decoding, thinking/reasoning_content, cache_control/prefix cache, model loading semantics, "server returns 404/garbage", API compliance. Do NOT use for CLI-only inference (imp-cli), kernel work (sm120-cuda-expert), or pure output-quality checks (check-degeneration).
---

# Server & API — imp-server

Source: `tools/imp-server/` — `main.cpp` (routes) · `args.cpp` (CLI flags + `--help` text) · `handlers_chat_core.cpp`/`handlers_chat.cpp`/`handlers_chat_stream.cpp` (OpenAI chat: params→render→stream) · `stream_driver.cpp` (shared per-token SSE loop; the three streaming handlers are thin dialect adapters over it) · `handlers_messages.cpp`+`anthropic.cpp` (Anthropic dialect) · `responses.cpp`/`handlers_responses.cpp` (Responses API) · `reasoning_split.h` (think/content channel split) · `tool_call.cpp`/`tool_stream_filter.h` (tool calling) · `batching_engine.cpp` (scheduler).

## Start the server

```bash
docker run --rm --gpus all -p 8080:8080 -v $HOME/models:/models imp:test \
  imp-server --host 0.0.0.0 --model /models/<MODEL> [--set server.prefix_cache=true]
```

Key flags (`imp-server --help` for all): `--port` (8080) · `--chat-template auto|chatml|llama3|nemotron|gemma` · `--lora NAME=PATH` (repeatable, select per request via `"lora"`) · `--kv-fp8|--kv-int8|--kv-int4|--kv-nvfp4` · `--think-budget <frac>` (default 0.5 of max_tokens) · `--api-key` · `--max-concurrent` (64) · `--log-requests <jsonl>` · `--mmproj` (vision) · `--vram-budget <mb>` (per-process cap; binds only since #1109, and still overshoots by the CUDA context ~1.7 GiB plus ~1.8 GiB of un-migrated tenants — leave headroom) · `--mem-report` (VRAM attribution table at init: lifecycle checkpoints, per-pool notes, named charges, and `own_peak` vs the cap with an `[OVER BUDGET]` marker) · `--set section.key=value` for any `RuntimeConfig` override.

## Endpoints

| Route | Dialect | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | SSE streaming, tools, json_schema |
| `POST /v1/completions` | OpenAI legacy | raw text |
| `POST /v1/responses` | OpenAI Responses | Agents SDK / Codex dialect; stateless shim over the chat path (`responses.cpp`); native SSE events incl. incremental `function_call_arguments.delta` |
| `POST /v1/messages` | **Anthropic** | thinking, tool use, `cache_control`, **real per-token SSE** (old "synthetic replay" info is obsolete) |
| `POST /v1/embeddings` | OpenAI | embedding vectors |
| `GET /v1/models` | both | Lists the loaded model (`loaded: true`, with vLLM `max_model_len` + llama.cpp `meta.n_ctx_train`) **plus the rest of the models directory** (`loaded: false`) since #1080 — a harness cannot request what it cannot see. |
| `GET /` | imp | Single-page web UI, embedded into the binary at build time (`cmake/embed_webui.cmake`, source `tools/imp-server/webui/index.html`). Editing the page requires a rebuild. |
| `GET /props` | llama.cpp | context-window probe: `n_ctx` (top-level + `default_generation_settings.n_ctx`) |
| `GET /info` | TGI | context-window probe: `max_total_tokens` / `max_input_tokens` |
| `POST /tokenize`, `/detokenize` | imp | |
| `GET /health`, `/metrics` | imp | healthcheck / Prometheus. `/metrics` carries request counters + TTFT/ITL histograms **and** the memory surface: per-tier `imp_memory_reserved_bytes` / `imp_memory_live_bytes` (capacity vs occupancy — a pool 90% full and one 90% reserved and empty are different incidents), `imp_kv_blocks_total` / `imp_kv_blocks_used`, and `imp_vram_budget_bytes` / `imp_vram_own_bytes` / `imp_vram_own_peak_bytes`. |

## Semantics that bite

- **Thinking**: default-ON for think-capable models in plain chat (json/tools requests excluded). Reasoning splits into `reasoning_content` vs `content`; gpt-oss uses Harmony channels (analysis/final) mapped the same way. Think-budget guarantees answer headroom — the `started_in_think` edge cases were a three-bug chain, so tread carefully there.
- **Thinking intent vs. rendered prompt — the prompt tail is ground truth.** Pipeline: request intent → Jinja render → `reconcile_thinking_with_prompt_tail`. A template may default `enable_thinking` to a **closed** `<think></think>` block; treating that as "model is thinking" lands the answer in `reasoning_content` with empty `content`. Explicit `enable_thinking:true` is honored via `force_thinking` (stamps it into the render). Don't re-tighten `mentions_thinking()` — deliberately rejected; the shortcut is load-bearing for spontaneous-`<think>` models.
- **Regex-constrained decoding**: `response_format: {"type":"regex","regex":"..."}` or vLLM's top-level `guided_regex`. The whole reply must match; EOS is only allowed from an accepting state. Engine is `RegexNfa` (`compute/json_schema.h`), shared with JSON-Schema `pattern`; the decode-time wrapper is `RegexConstrainer`. Unsupported constructs (lookaround, anchors, `\b`, backrefs) are refused, not silently mis-enforced — note `RegexNfa` itself PARSES some of them, so the refusal lives in the constrainer.
- **GBNF grammar-constrained decoding**: `response_format: {"type":"grammar","grammar":"root ::= ..."}`, llama.cpp's top-level `grammar`, or vLLM's `guided_grammar`. The whole reply must derive from `root`; EOS is gated on a complete derivation. Engine is a pushdown simulator (`compute/gbnf_grammar.{h,cpp}` + `gbnf_parser.cpp`), wrapper `GrammarConstrainer` — this is the one constrainer a regex could not have been, since a stack is what balances brackets. Grammars it cannot enforce (left recursion incl. indirect and star-over-nullable, undefined rules, no `root`, absurd repetition bounds) are refused at compile time, and the request then decodes unconstrained rather than under a wrong grammar. Perf note: a cold mask costs a 151k-token vocabulary walk, so the per-state mask cache and the interned-stack successor memo are load-bearing, not polish.
- **Adding a constrainer? Every bypass path must be closed, or the mask silently does nothing.** The checklist, learned the hard way: the two `InferenceState` sites in the scheduler AND the constrained-pipeline state in `engine_graph_decode.cpp`; the mask application in `executor.cu` — now ONE `apply_constraint_mask` helper (it used to be four copies, and two were easy to miss; keep it that way); the spec-ngram and graph-loop eligibility gates (`engine_spec_ngram.cpp`, `engine_graph_decode.cpp`) which otherwise route around the FSM; the request-field gates in `Engine::ensure_constraints_` and `pipeline_row_eligible_`; the thinking default in `handlers_chat_core.cpp` (structured output must suppress it, or the model spends the budget reasoning); and `ConstraintManager`'s active_* flags plus any per-constrainer cache — the manager is POOLED, so stale state leaks into the next request. That last one bit GBNF too: the grammar's stack arena was cleared on recompile but its memoised transitions were not, so a second grammar would have decoded with the first one's.
- **Constrained decoding** (`response_format: json_schema`): the per-token FSM simulator `sim_advance` is the SINGLE grammar source (`src/compute/schema_constrain.{h,cu}`, schema parsing in `json_schema.h`). Supports `$ref`/`$defs`; termination is exact-JSON, so the server returns exactly the object.
- **Constrained-decode perf**: category prefilter + in-string shortcut on the schema mask, then `ConstrainedPipeline` enqueues forward N+1 *before* the host FSM advances (json_schema ~102→235 tok/s). In-pipeline: greedy/top-k/top-p, rep/freq/presence penalties, banned tokens, think-budget — deliberately, so the server's own defaults don't disqualify it. **Falls back to eager** (slow) for logprobs, min_p, typical_p, mirostat, DRY, logit_bias, MTP, batch>1 — when measuring constrained perf, verify none are set or you are benchmarking the eager path.
- **`cache_control` / prefix cache**: Anthropic `cache_control` pins prompt-KV blocks (budget `server.prefix_pin_budget_pct`, FIFO) and reports `cache_read`/`cache_creation_input_tokens`. The LAST marked system/message block bounds the pin; a marker on tools pins the whole prompt; TTL tiers are accepted but not modeled. `PrefixCacheE2ETest` is the ship gate.
- **Speculative decoding** (n-gram prompt-lookup, greedy/batch-1): default-ON for dense, gated off for MoE; disable with `--set speculative.ngram=false`. Knobs in `config.h` → `struct Speculative`. Output stays token-identical to plain greedy. Bench confound: self-repetitive `--bench` prompts hit ~99.9% accept — see `benchmark-cuda` before A/B-ing decode kernels.
- **Model swapping** (`server.model_swap`, default ON): a request naming another model in the models directory swaps to it instead of 404; unknown names still 404, so a typo cannot trigger a load. It is safe only because both failure modes that killed the first auto-swap are closed — in-flight generations **drain** (`batching->pause`, the `/admin/suspend` contract, never cancelled) and a failed load **restores** the previous model. Keep both if you touch this path. **Third hazard, unclosable in software:** WSL2/WDDM never returns a process's peak VRAM commitment, so each in-process swap permanently costs the previous model's footprint (measured: free VRAM 30927 → 23113 MiB after one load/generate/free cycle, then stable). Every CUDA-level release succeeds — the memory is gone at the platform layer. A long-lived server that swaps repeatedly walks itself into host-spill territory; restart it rather than debugging the allocator.
- **Streaming must never cut inside a character.** A BPE token can end mid-character and each delta is serialized alone, so a raw byte cut becomes U+FFFD in the client (`größer` → `gr??ßer`). Two guards exist: `Utf8Stitch` holds partial bytes at detokenization — *before* any consumer, since the think splitter and tool filter match raw bytes too — and `holdback_decision` pulls its byte-offset cut back to a codepoint boundary. A new streaming sink must not reintroduce one.
- **Config keys** (`config.h` → `struct Server`): `prefix_cache`, `prefix_pin_budget_pct`, `green_contexts`, `model_swap`, `model_swap_drain_ms`.
- **Stop handling**: the server stops on turn markers at high temperature — don't remove that guard.

## Validation (run before claiming server work done)

```bash
# 1. Degeneration/think-leak/adherence battery against the running server
python3 tools/analysis/degen_suite.py --url http://localhost:8080   # --skip-deterministic for Qwen3.6

# 2. API contract suite (mock server, no GPU)
pytest tests/api/        # or tests/api/run_mock_tests.sh

# 3. SafeTensors model-level validation battery
python3 scripts/validate_safetensors.py --help
```

**Testing the web UI (or anything browser-driven) on this host**: the Playwright
MCP tool does NOT work — there is no Chrome, and installing Node/Chrome on the
host violates the clean-host rule. Drive a browser from a container instead:

```bash
docker run --rm --network host -e PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 -e HOME=/sp \
  -v "$PWD":/work:ro -v /tmp/<scratch>:/sp -v /tmp/<scratch>/shots:/out \
  -w /sp mcr.microsoft.com/playwright:v1.56.0-noble \
  sh -c 'npm i --silent playwright@1.56.0 && node /sp/script.mjs'
```

Playwright is NOT preinstalled in that image (only the browsers), hence the
`npm i`; `--network host` reaches `localhost:8080`; write screenshots to `/out`
and read them back. Assert on the DOM, not on screenshots alone.

**degen_suite `constrained` category** covers json_object, json_schema validity under three sampler states (greedy / temp=1.3 / min_p — the last forces the eager path, so it guards constrained-pipeline↔eager parity), and forced-`tool_choice` tool-call emission + argument JSON validity. Tool-arg *enforcement* is FSM-backed since #1002 (forced/required + `strict:true` + `parallel_tool_calls` on ChatML-JSON, forced on Llama3) and covers the Qwen-Coder/Qwen3.6 XML dialect (`<function=`/`<parameter=` raw-text bodies — templates teaching that format get the XML grammar, never the JSON body FSM, which would mask raw newlines and mangle multi-line code args). Categories now: repetition, think-leak, special-tokens, adherence, long-context, multi-turn, stream, **constrained**, anthropic-thinking. For deeper constrained-decoding changes also run `tests/api/` schema cases.

## Diagnostic fingerprints

| Symptom | Likely cause |
|---|---|
| CLI output fine, server output broken | `step()` path or NVFP4 cache divergence — not the model |
| Think text leaking into `content` | template-injected `<think>` without `</think>` (non-stream spill) — degen_suite catches this |
| Empty `content`, `reasoning_content` holds the **finished answer** (no `</think>` needed, reads like a reply) | channel mis-routing: imp's thinking state disagrees with the rendered prompt tail (closed-`<think>`-block template) — see the reconcile bullet above |
| Empty `content`, `reasoning_content` is **genuine but truncated reasoning** (cut off mid-thought) | think-budget eaten (`--think-budget 1.0` footgun) — discriminator vs the row above: is the reasoning text an answer or an unfinished chain of thought? |
| 404 `model_not_found` | the name did not resolve in the models directory (or `server.model_swap=false`) — not a bug. Check the models dir and `GET /v1/models`. |
| 503 "model swap … failed" | the requested model could not load; the previous one was restored and the server keeps serving. Usually VRAM or a bad file. |

After any server-side inference change, run the `check-degeneration` skill battery (section 0 targets exactly this layer).

## Known structural debt (don't re-discover)

The 2026-07-07 structural audit (`docs/audit/structural_debt_2026_07_07.md`) flagged the server layer as the main debt source — open issues #888–#897 (p1: admission-control bypass, `/health` lock contention, SSE-loop drift). Check those issues before filing "new" findings in this layer.
