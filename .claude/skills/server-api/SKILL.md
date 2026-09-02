---
name: server-api
description: Use when working on or testing imp-server and its HTTP APIs - OpenAI/Anthropic endpoints, /v1/chat/completions, /v1/messages, /v1/responses, SSE streaming, tool calling, json_schema/regex/GBNF constrained decoding, thinking/reasoning_content, cache_control/prefix cache, model loading and swapping, request priority, X-Request-Id, OTLP tracing / traceparent / Jaeger, /metrics, container env (IMP_SET), "server returns 404/garbage", API compliance. Do NOT use for CLI-only inference (imp-cli), kernel work (sm120-cuda-expert), or pure output-quality checks (check-degeneration).
---

# Server & API - imp-server

## Source map (`tools/imp-server/`)

| File | Owns |
|---|---|
| `main.cpp` | routes, `post_routing` (X-Request-Id echo) |
| `args.cpp` | server flags + `--help`; shared flags in `tools/common/args_common.cpp`; `tools/common/mtp_auto.cpp` decides the MTP head before an engine exists |
| `handlers_chat_core.cpp`, `handlers_chat.cpp`, `handlers_chat_stream.cpp`, `handlers_chat_params.cpp` | OpenAI chat: params -> render -> stream; `handlers_internal.h` holds `ChatRequestParams` |
| `stream_driver.cpp`, `stream_pipeline.h` | shared per-token SSE loop; the three dialect handlers are thin adapters |
| `handlers_messages.cpp` + `anthropic.cpp` | Anthropic dialect |
| `responses.cpp`, `handlers_responses.cpp` | Responses API shim |
| `handlers_rerank.cpp`, `handlers_admin.cpp`, `handlers_misc.cpp` | `/v1/rerank`, suspend/resume, tokenize/health/props |
| `reasoning_split.h`, `tool_call.cpp`, `tool_call_gemma.cpp`, `tool_stream_filter.h` | think/content split, tool calling |
| `constraint_validation.cpp` | uncompilable constraint = 400 |
| `batching_engine.cpp/.h` | scheduler + the deferred-delivery notifier thread (#1758, "Deferred delivery") |
| `tracing.cpp/.h` | W3C `traceparent`, OTLP/HTTP JSON exporter (background thread, 1 s batches); emission in `log_request_jsonl`, once per request |
| `rate_limit.cpp`, `image_fetch.cpp`, `metrics_memory.cpp` | per-key rate limit, remote images, memory surface of `/metrics` |

## Start

```bash
docker run --rm --gpus all -p 8080:8080 -v $HOME/models:/models imp:test \
  imp-server --host 0.0.0.0 --model /models/<MODEL> [--set key=value ...]
# dev binary: docker run --gpus all -v $PWD:/src -w /src -v $HOME/models:/models imp:toolchain build-dev/imp-server ...
```

Flags (`imp-server --help`): `--port` 8080, `--models-dir`, `--chat-template auto|none|chatml|llama2|llama3|nemotron|gemma`, `--lora NAME=PATH` (repeatable, per request `"lora"`), `--kv-fp8|--kv-int8|--kv-int4|--kv-nvfp4|--kv-mxfp4`, `--think-budget` 0.5, `--reasoning-format`, `--api-key`, `--rate-limit`, `--metrics-require-auth`, `--trusted-proxy`, `--max-concurrent` 64 (also sizes the HTTP worker pool to `max_concurrent + 8` since #1762; a streamed completion holds its worker), `--max-batch`, `--max-batch-items`, `--max-input-tokens`, `--max-n`, `--max-logit-bias`, `--request-timeout`, `--http-read-timeout|--http-write-timeout|--http-keep-alive-max`, `--log-requests <jsonl>`, `--mmproj`, `--allow-remote-images`, `--vram-budget <mb>` (binds since #1109; overshoots by the CUDA context ~1.7 GiB plus ~1.8 GiB of un-migrated tenants), `--mem-report`, `--prefix-cache`, `--config`, `--set section.key=value`.

Container: only `IMP_CONFIG` (-> `--config`) and `IMP_SET` (-> one `--set` per whitespace-separated `key=value`) reach every config key (#1823); the 19 legacy `IMP_*` names in `docker-entrypoint.sh` are frozen. `IMP_KV_FP8=1` on a Qwen3.5-family model (Qwen3.5/3.6/3.8 dense hybrids, e.g. Qwen3.8-27B-NVFP4-vllm) DOUBLES KV bytes (auto = NVFP4 there since #1750) and now logs the pin (#1823). `docs/DEPLOYMENT.md` "From a container".

## Endpoints

| Route | Dialect | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | SSE, tools, json_schema, `priority` |
| `POST /v1/completions` | OpenAI legacy | raw text |
| `POST /v1/responses` | OpenAI Responses | stateless shim over chat (`responses.cpp`); native SSE incl. `function_call_arguments.delta` |
| `POST /v1/messages` | Anthropic | thinking (opt-in since #1560/#1743), tool use, `cache_control`, real per-token SSE |
| `POST /v1/embeddings`, `/v1/rerank`, `/rerank` | OpenAI / Cohere-Jina-vLLM | rerank = causal-LM cross-encoder, gate `make test-rerank` |
| `POST /v1/messages/count_tokens` | Anthropic | |
| `POST /admin/suspend`, `/admin/resume` | imp | drain, release/re-acquire the GPU |
| `GET /v1/models`, `/v1/models/{id}` | both | loaded model (`loaded: true`, vLLM `max_model_len`, llama.cpp `meta.n_ctx_train`) + the rest of the models dir (`loaded: false`, #1080) |
| `GET /` | imp | web UI embedded at build (`cmake/embed_webui.cmake`, `tools/imp-server/webui/index.html`); rebuild to change |
| `GET /props`, `/info` | llama.cpp / TGI | context probes (`n_ctx`, `max_total_tokens`) |
| `POST /tokenize`, `/detokenize` | imp | |
| `GET /health` | imp | `kv_blocks_total` vs `kv_ceiling_blocks`, "this server takes concurrent requests" (why MTP auto declined) |
| `GET /metrics` | Prometheus | request counters, TTFT/ITL/duration histograms, `imp_queue_depth`, `imp_tokens_cached_total`, `imp_spec_drafted_total`, per-tier `imp_memory_reserved_bytes`/`imp_memory_live_bytes`, `imp_kv_blocks_total`/`_used`, `imp_vram_budget_bytes`/`_own_bytes`/`_own_peak_bytes`, `imp_otlp_*` |

## Semantics that bite

| Topic | Fact |
|---|---|
| Thinking | default ON for think-capable models in plain chat (json/tools excluded); `reasoning_content` vs `content`; gpt-oss Harmony channels map the same way. Think budget guarantees answer headroom (`started_in_think` was a three-bug chain). |
| Thinking vs rendered prompt | pipeline: intent -> Jinja -> `reconcile_thinking_with_prompt_tail`; the prompt tail is ground truth. A closed `<think></think>` template block treated as "thinking" puts the answer in `reasoning_content`. `enable_thinking:true` honored via `force_thinking`. Do not re-tighten `mentions_thinking()` (load-bearing for spontaneous-`<think>` models). |
| `reasoning_effort` | threaded through `ChatTemplate::apply*`/`render_jinja` + server snapshot (#1750); identical prompt-token counts across efforts = it is not reaching the template |
| Regex constraints | `response_format: {"type":"regex"}` or `guided_regex`; `RegexNfa` (`src/compute/json_schema.h`) + `RegexConstrainer`; lookaround, anchors, `\b`, backrefs refused in the constrainer (the NFA parses some of them) |
| GBNF constraints | `{"type":"grammar"}`, top-level `grammar`, `guided_grammar`; pushdown simulator `src/compute/gbnf_grammar.{h,cpp}` + `gbnf_parser.cpp`, `GrammarConstrainer`; left recursion, undefined rules, no `root`, absurd bounds refused at compile time (400); cold mask = 151k-vocab walk, so the per-state mask cache and interned-stack memo are load-bearing |
| JSON schema | `sim_advance` in `src/compute/schema_constrain.{h,cu}` is the single grammar source; `$ref`/`$defs`; exact-JSON termination |
| Adding a constrainer | close every bypass: `InferenceState` sites in `engine_decode_pipeline.cpp` AND `engine_scheduler.cpp`, constrained-pipeline state in `engine_graph_decode.cpp`; the ONE `apply_constraint_mask` helper in `src/exec/executor.cu`; spec-ngram and graph-loop eligibility (`engine_spec_ngram.cpp`, `engine_graph_decode.cpp`); `Engine::ensure_constraints_` and `pipeline_row_eligible_`; the thinking default in `handlers_chat_core.cpp`; `ConstraintManager` is POOLED (active flags + per-constrainer caches must reset; GBNF's memoised transitions once leaked into the next grammar) |
| Tool calling | tool-arg enforcement is FSM-backed since #1002 (forced/required, `strict:true`, `parallel_tool_calls` on ChatML-JSON; forced on Llama3); the Qwen-Coder/Qwen3.6 XML dialect (`<function=`/`<parameter=` raw-text bodies) gets the XML grammar, never the JSON body FSM (which would mask raw newlines in code args); degen_suite `constrained` covers forced `tool_choice` |
| Constrained perf | category prefilter + in-string shortcut; `ConstrainedPipeline` enqueues forward N+1 before the host FSM advances (~102 -> 235 tok/s). Falls back to eager for logprobs, min_p, typical_p, mirostat, DRY, logit_bias, MTP, batch>1 |
| `cache_control` / prefix cache | pins prompt-KV blocks (`server.prefix_pin_budget_pct` 25, FIFO); reports `cache_read`/`cache_creation_input_tokens`; last marked block bounds the pin; TTL accepted, not modeled; `PrefixCacheE2ETest` is the gate. Hybrids need `server.recurrent_snapshot_mb` (256 = 3 slabs of 79.5 MiB on the 27B) + host tier `server.recurrent_snapshot_host_mb` (2048 = 25 slabs; 8 interleaved sessions x 3 turns: turn-2 TTFT 324 -> 163 ms, #1854) |
| Speculation | `speculative.mtp_k=-1` auto (#1809): single-stream (`max_batch_size=1`) + head + not deterministic -> `mtp_k=2, ngram=false`; concurrent serving declines (head 0.79 GiB per batch slot); reads the RESOLVED batch size (`resolve_max_batch_size()`: flag > imp.conf > per-load, #1811). Manual few-stream serving: `--set speculative.mtp_k=2 --set speculative.ngram=false` as a PAIR. Adaptive depth `speculative.mtp_adaptive_k` on. n-gram default ON dense, off MoE. `imp_spec_drafted_total` = 1 after an essay = dead |
| Long context | `attention.sparse_topk_tokens` (off; 4096-8192 for long-ctx serving, floor 8192 on Qwen3.8 for NIAH 8/10), `sparse_min_ctx` 12288; the key min/max pool must stay VRAM-resident (a `kv_cache.max_blocks` pin without headroom spilled every prefill kernel +11%). `kv_cache.growable` opt-in (#1794: 32x8k+512 wall -24%). `--prompt-file` for 32k prompts |
| Priority | `"priority": int` body field (vLLM semantics, lower = earlier, default 0), primary sort key, aging within a class, admission only (#1803). Smoke needs a 400-token occupier or the slot frees before the high-prio request lands |
| Request ids | client `X-Request-Id` echoed on every response (sanitized, 128 chars); server completion id otherwise; JSONL `client_request_id` |
| OTLP tracing | `server.otlp_endpoint=http://host:4318/v1/traces` (off by default), `server.otlp_service_name`; one SERVER span per generation with queue/prefill/decode children, joined to the caller's `traceparent`; unsampled (`-00`) not exported; rejected requests (4xx/429/503) emit no span; no `traceresponse` header. Test: `tests/test_server_tracing.py` (in `make test-server`; its collector binds 4318, use `IMP_OTLP_PORT=4319` beside Jaeger). Jaeger recipe: `--add-host=host.docker.internal:host-gateway`, read back `GET :16686/api/traces/<id>`; Jaeger storage is in-memory. Compare span attributes against the response `usage` (that is what found `imp.cached_tokens` = 0, #1856) |
| Model swapping | `server.model_swap` on: another model name in the models dir swaps (drain, never cancel; failed load restores the previous). WSL2/WDDM never returns a process's peak VRAM: each swap costs the previous footprint (30927 -> 23113 MiB free after one cycle); restart a long-lived swapping server |
| Streaming UTF-8 | `Utf8Stitch` holds partial bytes before any consumer; `holdback_decision` cuts at codepoint boundaries; a new sink must not reintroduce a byte cut |
| TCP | `TCP_NODELAY` on accepted sockets (#1803; delayed ACK cost up to ~40 ms ITL for network clients) |
| Serving loop | deferred token delivery (#1758; `push_token` does not wake the SSE handler inline; diagnose with `diagnostics.step_timing`, #1759, whose `sample` phase includes the GPU sync), ragged prefill batching (`runtime.prefill_batch`, `src/runtime/engine_prefill_ragged.cpp`, #1780; serial for vision, constraints, logprobs, embeddings, rerank; off for Mamba2, MLA, MTP), prefill pacing (256-token floor charged once per ragged forward, #1781; `runtime.prefill_chunk_decode_cap` 1024, 4096 for bursts = +~10%), id-based rotor (#1762), graph prewarm (`runtime.graph_prewarm`, #1761, ~2.3 s at init), scheduler TUs split in #1782 (`engine_prefill.cpp`, `engine_prefill_ragged.cpp`, `engine_decode_pipeline.cpp`), hybrid decode pipeline and prefill||decode overlap both measured NEUTRAL and kept off (#1755, #1792) |
| Stop handling | the server stops on turn markers at high temperature; keep the guard |
| Config keys (`struct Server`) | `server.prefix_cache`, `prefix_pin_budget_pct`, `green_contexts` (off, sm_120 race), `model_swap`, `model_swap_drain_ms`, `recurrent_snapshot_mb`, `recurrent_snapshot_host_mb`, `otlp_endpoint`, `otlp_service_name` |

## Validation (before claiming server work done)

```bash
python3 tools/analysis/degen_suite.py --url http://localhost:8080   # --skip-deterministic for Qwen3.6-35B
pytest tests/api/            # mock contract, no GPU (tests/api/run_mock_tests.sh)
python3 scripts/validate_safetensors.py --help
make test-server             # the only end-to-end run of handlers + batching_engine (+ degen_suite, tracing)
make test-rerank; make test-agents; make test-agents-external   # aider, Claude Code, OpenAI Agents SDK drive imp
```

- The mock lane runs the nomodel tests too: a new contract test must be lane-agnostic AND `tests/api/mock_server.py` must mirror the contract (X-Request-Id echo cost a red `Mock API contract` on #1803).
- `--log-requests`: bind-mount a DIRECTORY (`-v dir:/out`); a single-file bind mount stays at 0 bytes.
- `imp-cli --version` does not exist (usage, rc=1): a control arm on it distinguishes nothing; bare `imp-cli` validates `--set` keys before the load (`no such key`).
- Browser-driven checks: no Chrome on the host; `docker run --rm --network host -e PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 -e HOME=/sp -v "$PWD":/work:ro -v /tmp/<scratch>:/sp -v /tmp/<scratch>/shots:/out -w /sp mcr.microsoft.com/playwright:v1.56.0-noble sh -c 'npm i --silent playwright@1.56.0 && node /sp/script.mjs'`; assert on the DOM.

## Diagnostic fingerprints

| Symptom | Likely cause |
|---|---|
| CLI fine, server broken | `step()` path or NVFP4 cache divergence, not the model |
| Think text in `content` | template `<think>` without `</think>` (non-stream spill); degen_suite catches it |
| Empty `content`, `reasoning_content` holds a finished answer | channel mis-routing (closed `<think>` template block): reconcile bullet |
| Empty `content`, `reasoning_content` truncated mid-thought | think budget eaten (`--think-budget 1.0`) or `max_tokens < ~400` on Qwen3.8 |
| Empty `content`, exit 0, speculation on | fixed #1796 (verify argmax ignored the banned mask); pair `mtp_k` with `ngram=false` |
| 404 `model_not_found` | name not in the models dir or `server.model_swap=false` |
| 503 "model swap ... failed" | load failed, previous model restored (VRAM or bad file) |
| `/health` reports a batch you did not configure | `runtime.max_batch_size` from imp.conf vs `--max-batch` flag: flag wins |
| Span attributes disagree with `usage` | dead context fields on the non-stream path (class of #1856): assert span vs body in the test |

After any server-side inference change run the **check-degeneration** battery (section 0).

## Known structural debt

Server layer flagged in `docs/archive/structural_debt_2026_07_07.md`: issues #888-#897 (admission-control bypass, `/health` lock contention, SSE-loop drift). Check them before filing new findings.
