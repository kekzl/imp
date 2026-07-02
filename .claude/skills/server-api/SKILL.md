---
name: server-api
description: Use when working on or testing imp-server and its HTTP APIs — OpenAI/Anthropic endpoints, /v1/chat/completions, /v1/messages, SSE streaming, tool calling, json_schema/constrained decoding, thinking/reasoning_content, cache_control/prefix cache, model loading semantics, "server returns 404/garbage", API compliance. Do NOT use for CLI-only inference (imp-cli), kernel work (sm120-cuda-expert), or pure output-quality checks (check-degeneration).
---

# Server & API — imp-server

Source: `tools/imp-server/` (`main.cpp` routes, `handlers.cpp` OpenAI dialect, `anthropic.cpp` Anthropic dialect, `batching_engine.cpp` scheduler).

## Start the server

```bash
docker run --rm --gpus all -p 8080:8080 -v /home/kekz/models:/models imp:test \
  imp-server --host 0.0.0.0 --model /models/<MODEL> [--set server.prefix_cache=true]
```

Key flags (`imp-server --help` for all): `--port` (8080) · `--chat-template auto|chatml|llama3|nemotron|gemma` · `--lora NAME=PATH` (repeatable, select per request via `"lora"`) · `--kv-fp8|--kv-int8|--kv-int4|--kv-nvfp4` · `--think-budget <frac>` (default 0.5 of max_tokens) · `--api-key` · `--max-concurrent` (64) · `--log-requests <jsonl>` · `--mmproj` (vision) · `--set section.key=value` for any `RuntimeConfig` override.

## Endpoints

| Route | Dialect | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | SSE streaming, tools, json_schema |
| `POST /v1/completions` | OpenAI legacy | raw text |
| `POST /v1/responses` | OpenAI Responses | Agents SDK / Codex dialect; stateless shim over the chat path (`responses.cpp`); native SSE events incl. incremental `function_call_arguments.delta` |
| `POST /v1/messages` | **Anthropic** | thinking, tool use, `cache_control`, **real per-token SSE** (`main.cpp:192` — old "synthetic replay" info is obsolete) |
| `POST /v1/embeddings` | OpenAI | embedding vectors |
| `GET /v1/models` | both | **strict semantics since PR #507**: only the loaded model is listed; a foreign `model` field → 404 `model_not_found`; switching models = restart (auto-swap was removed after a reload SIGSEGV) |
| `POST /tokenize`, `/detokenize` | imp | |
| `GET /health`, `/metrics` | imp | healthcheck / Prometheus |

## Semantics that bite

- **Thinking**: default-ON for think-capable models in plain chat (json/tools requests excluded). Reasoning is split into `reasoning_content` vs `content` (`--reasoning-format deepseek`); gpt-oss uses Harmony channels (analysis/final) mapped the same way. Think-budget guarantees answer headroom — `started_in_think` edge cases were a 3-bug chain (PR #518), be careful there.
- **Constrained decoding** (`response_format: json_schema`): the per-token FSM simulator `sim_advance` is the SINGLE grammar source (`src/compute/schema_constrain.{h,cu}`; schema parsing in `src/compute/json_schema.h`; chain PRs #497–#499). Supports `$ref`/`$defs`. Termination is exact-JSON (CAT_EOS only at DONE, whitespace cap) — the server returns exactly the JSON object.
- **Constrained-decode perf** (PR #651; the #650 SIGBUS/ctrl-char fixes are prerequisites): category prefilter + in-string shortcut on the schema mask, then `ConstrainedPipeline` (`src/runtime/engine.h`) enqueues forward N+1 *before* the host FSM advances — json_schema 102→235 tok/s (plain ≈270). In-pipeline: greedy/top-k/top-p, rep/freq/presence penalties, banned tokens, think-budget (deliberately, so the server defaults `repetition_penalty=1.05`/`think_budget=0.5` don't silently disqualify it). **Falls back to eager** (slow) for: logprobs, min_p, typical_p, mirostat, DRY, logit_bias, MTP, batch>1 — when measuring constrained perf, check none of these are set or you're benchmarking the eager path.
- **`cache_control` / prefix cache**: Anthropic `cache_control` pins prompt-KV blocks (budget `server.prefix_pin_budget_pct` = 25% FIFO) and reports `cache_read_input_tokens`/`cache_creation_input_tokens`. `server.prefix_cache` is default ON since the #536/#538 stale-block-table fix; `PrefixCacheE2ETest` is the ship gate.
- **Speculative decoding** (n-gram prompt-lookup, opt-in, greedy/batch-1): no dedicated server flags — enable via `--set speculative.ngram=true`. Knobs in `src/runtime/config.h` → `struct Speculative` (parsed in `config.cpp:247-254`): `k` (16), `min_match` (6), `max_match` (12), `give_up_after` (64), `burst` (128), `miss_burst` (8), `burst_rearm` (default ON — the #683 wrong-token artifact was the conditional-graph `setup()` position off-by-one, fixed in PR #692, not rearm). Output stays token-identical to plain greedy. CLI win ~+6%; opt-in because draft-poor workloads regress.
- **Config keys** (`src/runtime/config.h` → `struct Server`): `prefix_cache`, `prefix_pin_budget_pct`, `green_contexts`.
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

**degen_suite has NO json_mode/json_schema category** (categories: repetition, think-leak, special-tokens, adherence, long-context, multi-turn, stream, anthropic-thinking). Constrained-decoding changes need explicit validation: `tests/api/` schema cases + live `response_format: json_schema` requests against a real model.

## Diagnostic fingerprints

| Symptom | Likely cause |
|---|---|
| CLI output fine, server output broken | `step()` path or NVFP4 cache divergence — not the model |
| Think text leaking into `content` | template-injected `<think>` without `</think>` (non-stream spill) — degen_suite catches this |
| Empty `content`, long reasoning | think-budget eaten (`--think-budget 1.0` footgun) |
| SIGSEGV on model reload | prewarm on dangling cuBLAS stream — reload is intentionally unsupported, restart instead |
| 404 `model_not_found` | strict #507 semantics, not a bug — request must name the loaded model |

After any server-side inference change, run the `check-degeneration` skill battery (section 0 targets exactly this layer).
