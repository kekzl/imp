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

Key flags (`imp-server --help` for all): `--port` (8080) · `--chat-template auto|chatml|llama3|nemotron|gemma` · `--lora NAME=PATH` (repeatable, select per request via `"lora"`) · `--kv-fp8|--kv-int8|--kv-int4|--kv-nvfp4` · `--think-budget <frac>` (default 0.5 of max_tokens) · `--api-key` · `--max-concurrent` (64) · `--log-requests <jsonl>` · `--mmproj` (vision) · `--set section.key=value` for any `RuntimeConfig` override.

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
| `GET /health`, `/metrics` | imp | healthcheck / Prometheus |

## Semantics that bite

- **Thinking**: default-ON for think-capable models in plain chat (json/tools requests excluded). Reasoning is split into `reasoning_content` vs `content` (`--reasoning-format deepseek`); gpt-oss uses Harmony channels (analysis/final) mapped the same way. Think-budget guarantees answer headroom — `started_in_think` edge cases were a 3-bug chain (PR #518), be careful there.
- **Thinking intent vs. rendered prompt — the prompt tail is ground truth** (PRs #937/#939). The pipeline is: request intent → Jinja render → `reconcile_thinking_with_prompt_tail` (a template may default `enable_thinking` to a **closed** `<think></think>` block; imp must NOT treat the model as thinking then, or the answer lands in `reasoning_content` with empty `content`). Explicit `enable_thinking:true` on such templates is honored via `force_thinking` (stamps `enable_thinking=true` into the render — `chat_template.h:60`, `handlers_chat_core.cpp` ~538). Don't re-tighten `mentions_thinking()` — deliberately rejected (no-op on real models; the shortcut is load-bearing for spontaneous-`<think>` models).
- **Constrained decoding** (`response_format: json_schema`): the per-token FSM simulator `sim_advance` is the SINGLE grammar source (`src/compute/schema_constrain.{h,cu}`; schema parsing in `src/compute/json_schema.h`; chain PRs #497–#499). Supports `$ref`/`$defs`. Termination is exact-JSON (CAT_EOS only at DONE, whitespace cap) — the server returns exactly the JSON object.
- **Constrained-decode perf** (PR #651; the #650 SIGBUS/ctrl-char fixes are prerequisites): category prefilter + in-string shortcut on the schema mask, then `ConstrainedPipeline` (`src/runtime/engine.h`) enqueues forward N+1 *before* the host FSM advances — json_schema 102→235 tok/s (plain ≈270). In-pipeline: greedy/top-k/top-p, rep/freq/presence penalties, banned tokens, think-budget (deliberately, so the server defaults `repetition_penalty=1.05`/`think_budget=0.5` don't silently disqualify it). **Falls back to eager** (slow) for: logprobs, min_p, typical_p, mirostat, DRY, logit_bias, MTP, batch>1 — when measuring constrained perf, check none of these are set or you're benchmarking the eager path.
- **`cache_control` / prefix cache**: Anthropic `cache_control` pins prompt-KV blocks (budget `server.prefix_pin_budget_pct` = 25% FIFO) and reports `cache_read_input_tokens`/`cache_creation_input_tokens`. Since #1046 the LAST marked system/message block bounds the pin (internal `cache_prefix_messages` → truncated re-render → `pin_kv_prefix_tokens`); a marker on tools = whole-prompt pin; TTL tiers accepted but not modeled. `server.prefix_cache` is default ON since the #536/#538 stale-block-table fix; `PrefixCacheE2ETest` is the ship gate.
- **Speculative decoding** (n-gram prompt-lookup, greedy/batch-1): **default-ON for dense models, gated off for MoE** since PR #781 — disable via `--set speculative.ngram=false`. Knobs in `src/runtime/config.h` → `struct Speculative`: `k` (16), `min_match` (6), `max_match` (12), `give_up_after` (64), `burst` (128), `miss_burst` (8), `burst_rearm` (default ON — the #683 wrong-token artifact was the conditional-graph `setup()` position off-by-one, fixed in PR #692, not rearm). Output stays token-identical to plain greedy. Bench confound: on self-repetitive `--bench` prompts the draft accept rate hits ~99.9% — see `benchmark-cuda` before A/B-ing decode kernels.
- **Model swapping** (`server.model_swap`, default ON since #1080): a request naming another model in the models directory swaps to it instead of 404. Unknown names still 404 (the name must resolve, so a typo cannot trigger a load). The swap is safe because both failure modes that killed the FIRST auto-swap are closed — in-flight generations **drain** via `batching->pause(drain_ms)` (the `/admin/suspend` contract, never cancelled), and a failed load **restores** the previous model. If you touch this path, keep both: an auto-swap that cancels streams or leaves the server model-less is why it was removed once already.
- **Streamed non-ASCII was corrupted until #1078** — worth knowing when touching the stream path. A BPE token can end mid-character, and each delta is serialized alone, so `dump_safe` turned the halves into U+FFFD (`größer` → `gr??ßer`). Two independent causes, both fixed: `Utf8Stitch` (`utils.h`) holds partial bytes at detokenization *before any consumer sees the piece* — the think splitter and tool filter match on raw bytes too — and `holdback_decision` now pulls its byte-offset cut back to a codepoint boundary. Any new streaming sink must not reintroduce a raw byte cut.
- **Config keys** (`src/runtime/config.h` → `struct Server`): `prefix_cache`, `prefix_pin_budget_pct`, `green_contexts`, `model_swap`, `model_swap_drain_ms`.
- **Stop handling**: server stops on turn markers at high temperature (PR #442) — don't remove that guard.

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

**degen_suite `constrained` category** (added 2026-07-15) covers json_object, json_schema validity under three sampler states (greedy / temp=1.3 / min_p — the last forces the eager path, so it guards constrained-pipeline↔eager parity), and forced-`tool_choice` tool-call emission + argument JSON validity. Tool-arg *enforcement* is FSM-backed since #1002 (forced/required + `strict:true` + `parallel_tool_calls` on ChatML-JSON, forced on Llama3) and covers the Qwen-Coder/Qwen3.6 XML dialect (`<function=`/`<parameter=` raw-text bodies — templates teaching that format get the XML grammar, never the JSON body FSM, which would mask raw newlines and mangle multi-line code args). Categories now: repetition, think-leak, special-tokens, adherence, long-context, multi-turn, stream, **constrained**, anthropic-thinking. For deeper constrained-decoding changes also run `tests/api/` schema cases.

## Diagnostic fingerprints

| Symptom | Likely cause |
|---|---|
| CLI output fine, server output broken | `step()` path or NVFP4 cache divergence — not the model |
| Think text leaking into `content` | template-injected `<think>` without `</think>` (non-stream spill) — degen_suite catches this |
| Empty `content`, `reasoning_content` holds the **finished answer** (no `</think>` needed, reads like a reply) | channel mis-routing: imp's thinking state disagrees with the rendered prompt tail (closed-`<think>`-block template) — see the reconcile bullet above (PR #937) |
| Empty `content`, `reasoning_content` is **genuine but truncated reasoning** (cut off mid-thought) | think-budget eaten (`--think-budget 1.0` footgun) — discriminator vs the row above: is the reasoning text an answer or an unfinished chain of thought? |
| SIGSEGV on model reload | prewarm on dangling cuBLAS stream — reload is intentionally unsupported, restart instead |
| 404 `model_not_found` | strict #507 semantics, not a bug — request must name the loaded model |

After any server-side inference change, run the `check-degeneration` skill battery (section 0 targets exactly this layer).

## Known structural debt (don't re-discover)

The 2026-07-07 structural audit (`docs/audit/structural_debt_2026_07_07.md`) flagged the server layer as the main debt source — open issues #888–#897 (p1: admission-control bypass, `/health` lock contention, SSE-loop drift). Check those issues before filing "new" findings in this layer.
