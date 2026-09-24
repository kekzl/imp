<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# API

What the HTTP surface actually accepts.

- Status legend from [`FEATURES.md`](FEATURES.md): ✅ code path plus a gated test, 🟡 code path, no test.
- Constrained decoding, tool calling, thinking/reasoning and images: [`API_FEATURES.md`](API_FEATURES.md).

**Two dialects, both native.** `/v1/messages` is implemented against the
Anthropic wire format directly, no shim in either direction. All three
dialects share one per-token streaming driver, so a streaming fix lands in
all of them at once.

## Endpoints

| endpoint | status | notes |
|---|---|---|
| `POST /v1/chat/completions` | ✅ | the main one. Text and image content parts |
| `POST /v1/completions` | ✅ | legacy text completion |
| `POST /v1/messages` | ✅ | Anthropic. Real per-token SSE, `ping` keepalives |
| `POST /v1/messages/count_tokens` | ✅ | |
| `POST /v1/responses` | ✅ | OpenAI Responses, the dialect Codex and the Agents SDK speak by default; stateless, so use `store: false` and resend the transcript in `input` |
| `POST /v1/embeddings` | ✅ | needs an embedding model loaded |
| `POST /v1/rerank`, `POST /rerank` | ✅ | Cohere/Jina/vLLM shape |
| `POST /tokenize`, `POST /detokenize` | ✅ | `/tokenize` takes `content` (llama.cpp) or `prompt` (vLLM) |
| `GET /v1/models` | ✅ | loaded model plus the rest of the directory, each with `loaded: true|false`; the loaded entry carries `meta.reasoning_effort` `{values, default}` when its chat template names the list (Qwen3.8: `xhigh`, `medium`, `low`) |
| `GET /health`, `/metrics`, `/props`, `/info` | ✅ | `/props` is the llama.cpp shape, `/info` the TGI one |
| `POST /admin/suspend`, `/admin/resume` | ✅ | see [`DEPLOYMENT.md`](DEPLOYMENT.md) |
| `GET /` | ✅ | built-in chat UI |

## Sampling and generation fields

| field | status | notes |
|---|---|---|
| `model` | ✅ | **required**; basename of the file or directory |
| `messages`, `prompt` | ✅ | |
| `max_tokens` | ✅ | on a reasoning model the answer reserve scales with it (`max(256, max_tokens/4)`) |
| `temperature`, `top_p`, `top_k`, `min_p` | ✅ | **imp's defaults are not OpenAI's** - see the table below |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | ✅ | `repetition_penalty` has no OpenAI field; imp applies 1.05 by default |
| `seed` | ✅ | greedy is reproducible; see [`determinism.md`](determinism.md) for the exact guarantee, which is narrower than "same seed, same bytes" |
| `stop` | ✅ | |
| `stream` | ✅ | per token, all three dialects |
| `n` | ✅ | documented and tested as `[1,4]` |
| `logprobs` | ✅ | **of the processed distribution**: the row the sampler drew from, after penalties, `logit_bias`, banned tokens, a constraint mask, `min_p` and `typical_p` (vLLM's `processed_logprobs`; a masked token reads as probability 0, and at temperature 0 the emitted token is always `top_logprobs[0]`). `tests/test_server_logprobs.py` in `make test-server` pins that with a `logit_bias` case, plus `tests/test_logprobs_shapes.cpp` in the CPU lane. Streaming emitted none whenever a `stop` sequence was set until #1588; `/v1/completions` returned the Chat shape until #1589 |
| `ignore_eos` | ✅ | vLLM-compatible extension on chat and completions: EOS and stop tokens are counted as output tokens without text, the think-model implicit `\nHuman` stop is not injected, user `stop` strings still apply, the request ends at `max_tokens` with `finish_reason: "length"`; for benchmark clients that need equal token counts across arms (`tools/analysis/burst_stream_client.py` with `IGNORE_EOS=1`). `tests/test_server_ignore_eos.py` in `make test-server` |
| `best_of` | ⚪ | `best_of > 1` is a 400: imp generates no candidate set to choose from (#1598) |
| DRY, mirostat, typical_p, logit_bias | ✅ | `logit_bias`: token-id keys, values in [-100, 100], ids inside the vocabulary; anything else is a 400 |
| `"speculative": true/false/{"mtp_k": N}` | ✅ | per-request override, carried by all three dialects (chat, `/v1/messages`, `/v1/responses`). The boolean form switches every drafter on or off; `false` reaches **all three** (n-gram, MTP head, token recycling) since #1639. The object form addresses the trained MTP head alone and is orthogonal to the boolean: `{"mtp_k": 0}` turns the head off for this request and leaves the matcher running, `{"mtp_k": 2}` asks for a depth-2 chain. `N` outside `0..<armed depth>` is a 400 naming the range (the bound is the device chain cap, 16, when no head is armed). Neither form can conjure a head the checkpoint lacks or the process did not upload. When a verify step ran, or the request asked for MTP and did not get it, `usage.completion_tokens_details` carries `imp_spec_drafted`, `imp_spec_accepted`, `imp_spec_emitted`, `imp_spec_verify_steps` and, on a decline, `imp_spec_declined` + `imp_spec_declined_detail`. All six travel on every dialect from one shared table (`tools/imp-server/spec_usage_keys.h`): Anthropic lifts them into top-level `usage`, Responses into `usage.output_tokens_details`. The request span carries the counters as `imp.spec_*` attributes. Decline reasons: `no_mtp_head`, `mtp_head_not_loaded` (the concurrency decline - `speculative.mtp_k=auto` refuses the head on a server taking concurrent requests), `mtp_head_not_armed`, `mtp_depth_clamped` |
| `"lora": "name"` | ✅ | PEFT adapter selected per request, every quant path. One adapter is active at a time: a request naming a different one waits until the in-flight requests finished, then the worker switches (decode graphs re-capture); it is never batched with them. The prefix cache is keyed by adapter, so a shared system prompt is prefilled once per adapter. Adapter shapes are checked against the model at load |
| `"priority": int` | ✅ | vLLM-compatible admission priority, **lower value schedules earlier**, default 0. Strictly dominates the scheduler's shortest-first-with-aging order; a caller that sets priorities owns starvation across classes. Accepted on all three dialects |
| `"cache_prompt": true` | ✅ | pins the prompt's full KV blocks against eviction (the OpenAI-route spelling of Anthropic `cache_control`). Prefix reuse runs regardless: `false` does **not** disable it, unlike llama.cpp (`usage.prompt_tokens_details.cached_tokens` reports the reuse either way) |

### Defaults, and where they differ from OpenAI

A request with no sampling fields uses imp defaults, not OpenAI's (#1596):
they suit local models better than `temperature 1.0` with no truncation, so an
identical request returns different output than against the OpenAI API.

| field | imp | OpenAI | to get OpenAI's behaviour |
|---|---|---|---|
| `temperature` | 0.7 | 1.0 | send `"temperature": 1.0` |
| `top_p` | 0.95 | 1.0 | send `"top_p": 1.0` |
| `top_k` | 40 | (no field) | send `"top_k": 999999` (see below) |
| `repetition_penalty` | 1.05 | (no field) | send `"repetition_penalty": 1.0` |

Two of these do not switch off the way the field name suggests:

- **`top_k: 0` is not "off", it is 50.** Every sampling site spells
`top_k > 0 ? top_k : 50` (`src/exec/executor.cu:191`, `:290`, `src/runtime/engine_decode_pipeline.cpp:74`).

- Zero and "unset" both land on 50, a *tighter* truncation than the 40 default.
- Disabling top-k needs a value at or above the vocabulary size.
- The dispatcher clamps to the full vocabulary (`engine_decode_pipeline.cpp:75`).
- **`repetition_penalty` has no OpenAI field**, so a strictly spec-compliant
  client cannot switch it off. Sending the non-OpenAI field with value `1.0`
  disables it: the engine skips the penalty pass when all three penalties are
  neutral (`src/runtime/engine_sampling_stop.cpp:178`).

### Metrics for what the server decided

`/metrics` carries counters (#1640, #1641):

| counter | what moves it |
|---|---|
| `imp_requests_timed_out_total` | the server ended a request at `--request-timeout`. The client sees `finish_reason: "length"`, which is also what a spent token budget produces - this counter is the only way to tell them apart |
| `imp_kv_pressure_rejections_total` | a request was cancelled because the KV pool could not give it blocks (admission or mid-decode). Not incremented for a failed metadata allocation or a snapshot mismatch, which are different faults |
| `imp_kv_pool_growths_total` | the growable pool committed more memory. A pool that keeps growing under load is the signal that arrives before it stops being able to |
| `imp_streaming_kv_auto_enables_total` | the KV pool ran nearly full and StreamingLLM eviction switched itself on (every KV dtype; F16 only before v0.43). The same event demotes CUDA graphs; both are lifted again once a fifth of the pool is free, unless a sequence was actually evicted, which pins them for the rest of the process. `usage.prompt_tokens_details.evicted_tokens` is the per-request size, this is the rate |
| `imp_prefix_cache_evictions_total` | a cached prefix block was reclaimed for a new allocation. Rising while `imp_tokens_cached_total` stalls means the pool is smaller than the working set |

`imp_requests_cancelled_total` remains client-disconnect only.

Speculative decoding is counted twice: once in aggregate, once per draft source.

- Aggregate: `imp_spec_drafted_total`, `_accepted_total`, `_verify_steps_total`, `_miss_steps_total`.
- Per source: `imp_spec_mtp_*` against `imp_spec_ngram_*`, each with `drafted_total`, `accepted_total`, `emitted_total`, `verify_steps_total`, `verify_wall_ms_total`.
- Aggregate cannot price the MTP head on its own: the n-gram/suffix matcher, the prompt prediction and token recycling fill the same verify chunk and land in the same totals, so a server running the documented MTP pair reports the same four numbers as one the matcher carried.
- `imp_spec_mtp_accepted_total / imp_spec_mtp_drafted_total`: the head's acceptance rate.
- `imp_spec_mtp_emitted_total / imp_spec_mtp_verify_steps_total`: what its verify bought.
- `imp_spec_mtp_verify_wall_ms_total`: what it cost.
- `imp_spec_ngram_*`: every non-MTP drafter together.
- All ten hang off a live engine; a model-less server emits none of them.

Per endpoint, the same counter and ladders carry an `endpoint` label (`chat_completions`, `completions`, `messages`, `responses`, `embeddings`, `rerank`).

- Labels apply to `imp_endpoint_requests_total`, `imp_endpoint_request_duration_seconds`, `imp_endpoint_ttft_seconds`, `imp_endpoint_inter_token_seconds`, `imp_endpoint_queue_time_seconds`.
- Every endpoint emitted at zero, so a panel can be built before traffic arrives.
- The unlabelled series stay the totals.
- `imp_model_loaded` carries `model="<name>"`, so a swapped model's numbers are not its predecessor's.

`imp_decode_batch_last_rows` (gauge) is the number of sequences in the most
recent decode step, 0 while the worker idles: the live batch, where
`imp_decode_batch_rows_total / imp_decode_batch_steps_total` is a windowed
mean and `imp_decode_batch_max` never resets.

The four latency histograms (`imp_request_duration_seconds`, `imp_ttft_seconds`, `imp_queue_time_seconds`, `imp_inter_token_seconds`) are fed by every generation path: chat stream and non-stream, `/v1/completions` stream and non-stream.

- A request cancelled or timed out before the worker admitted it contributes its wait to `imp_queue_time_seconds` as well.
- `imp_queue_time_seconds` ends when the scheduler puts the request into its first batch: the wait behind `max_batch_size` and KV admission.
- `imp_queue_waiting` / `imp_queue_running` split `imp_queue_depth` on the same boundary.
- Gate: `tests/test_server_metrics.py` in `make test-server`.
- The serving KPI harness reads the histograms and counters back per concurrency level (`tools/analysis/serving_kpi.py`, definitions in [`internals/BENCHMARKING.md`](internals/BENCHMARKING.md)).

`imp_kv_blocks_reserved` (gauge) is what admission has promised to running
requests for the rest of their generation and not yet written (#1635). Free
blocks minus this gauge is what the next request is admitted against, so a
queue in front of a pool with free blocks is explained by this number and by
nothing else in `/metrics`.

## Prompt caching

✅ by default. Prefix blocks reused across requests sharing a prefix.

`cache_control` is honoured per breakpoint: the **last** marked block bounds the
pinned region, rather than pinning the whole prompt. Usage reporting carries
`cache_read_input_tokens` and `cache_creation_input_tokens`.

The cache is keyed on the picture as well as the token ids for image requests,
and it is model-fingerprint-gated on disk, so a cache file from another model is
never replayed.

## Request tracing

Every response echoes back the `X-Request-Id` header, refusals and unmatched routes included, sanitized to printable ASCII and capped at 128 chars.

- Generation endpoints answer with the server's own completion id when no client id was sent, so every generation response carries some id a caller can quote.
- With `--log-requests`, the JSONL record carries the client id as `client_request_id` next to the server `req_id`, the join an agent framework needs to attribute its own latency to this hop.
- With `server.otlp_endpoint` set (the full traces URL of an OTLP/HTTP collector, JSON encoding, `http://` only), every generation request is exported as an OpenTelemetry span: a SERVER span named after the endpoint (`/v1/chat/completions`, `/v1/messages`, ...).
- Span attributes: `imp.request_id`, `imp.client_request_id`, `gen_ai.request.model`, `gen_ai.usage.input_tokens` / `output_tokens`, `imp.cached_tokens`, `gen_ai.response.finish_reasons`, `imp.stream`, `imp.queue_ms`, `imp.ttft_ms`.
- Child spans on the request's timeline: `queue`, `prefill`, `decode` (the last two for streaming requests, where the first token is observed).
- A W3C `traceparent` request header puts the hop inside the caller's trace (its trace id, the caller's span as parent); without one a trace id is minted.
- A header with the sampled flag clear (`...-00`) is honoured the way a parent-based sampler would: ids are minted, nothing is exported.
- The JSONL record carries the same `trace_id` / `span_id` either way.
- Only requests that reach generation are traced; a request refused before it (4xx, 429, 503) emits no span.
- Export runs on a background thread in one-second batches; an unreachable collector costs one warning and dropped batches, never request latency.
- Measured 2026-09-02 on Qwen3.8-27B: median request latency 113.6 ms with the collector up, 111.5 ms with it stopped.
- `/metrics` counts `imp_otlp_spans_exported_total`, `imp_otlp_export_failures_total`, `imp_otlp_unsampled_requests_total`.
- Service name: `server.otlp_service_name` (default `imp-server`).
- Verified against Jaeger v2.20 (OTLP/HTTP receiver, GenAI view) with an OpenTelemetry-SDK client: the hop lands under the client's span with `queue` / `prefill` / `decode` children on its timeline.

## Errors

Every error is a JSON envelope, never a bare status with an empty body, and
`/v1/messages*` paths get the Anthropic error shape rather than the OpenAI one.
An unmatched route answers with an envelope too.

On `/v1/messages` the `error.type` is always one of Anthropic's own -
`invalid_request_error`, `authentication_error`, `billing_error`,
`permission_error`, `not_found_error`, `request_too_large`, `rate_limit_error`,
`api_error`, `overloaded_error`, `timeout_error`. Every response from that
endpoint also carries a `request-id` header, and error bodies repeat it as
`request_id`.

Stream ending on server-side fault emits `error` SSE event instead of
`message_delta`/`message_stop`.

`anthropic-version` and `anthropic-beta`: read and echoed back, not enforced.
Unknown beta logged once per value.

Internal engine errors are translated to `ImpError` at the C API boundary
(`src/api/imp_api.cpp`); this is intentional and is why a load failure surfaces
as a typed message rather than a crash.

### `GET /ready`

Readiness as a status code. 200 `{"ready": true, ...}` only when an inference
request would be taken right now; otherwise 503 with a stable `code`:

| `code` | state |
|---|---|
| `no_model` | started model-less and nothing loaded yet |
| `engine_faulted` | a device fault poisoned the CUDA context; a swap does not clear it, restart the process |
| `suspended` | `/admin/suspend` parked the weights; `POST /admin/resume` |
| `swapping` | a model load or swap is in progress |
| `draining` | the process received SIGTERM and is finishing in-flight work |

`/health` stays liveness and answers 200 in all four (an orchestrator must not
restart a server that is swapping or draining). Both routes need no API key.

### `GET /health`, and which 503 is worth retrying

`/health` answers 200 with `status: "ok"` when process can serve.
Load, queueing, transient OOM: 200 (server alive; restart would worsen them).

503 only for persistent states, with stable `code`:

| `code` | what it means | what a client should do |
|---|---|---|
| `kv_pool_floored` | the KV pool fell back to its rescue floor, so it holds a few hundred tokens instead of a context. Sized once at startup, usually because another process still held the card | do **not** retry, the process cannot recover. Restart it on a free card |
| `engine_faulted` | a device fault (illegal address, launch failure) poisoned the CUDA context; the worker stopped | restart |

A device fault is also visible on the request that hit it and on every later
one: non-streaming answers 500 `internal_error` with `code` `engine_faulted`
(`engine_step_failed` when the step failed on the host and the engine
recovered), a stream carries the same in its `error` event. No request after
the fault is queued or served.

Whenever a model is loaded the body carries the pool capacity, healthy or
not, so a caller can check a prompt against server capacity without scraping
`/metrics`:

```json
{"status": "unhealthy", "code": "kv_pool_floored",
 "model_loaded": true, "queue_depth": 0, "suspended": false,
 "kv_blocks_total": 16, "kv_block_size": 32, "kv_capacity_tokens": 512,
 "kv_ceiling_blocks": 16, "kv_pool_growable": false, "kv_pool_bandwidth_gbps": 1554.0}
```

`kv_pool_bandwidth_gbps` is a device-to-device copy timed inside the pool at init, GB/s counting read plus write (also `imp_kv_pool_bandwidth_gbps` on `/metrics`).

- On WSL2/WDDM a successful allocation proves nothing: a pool the driver spilled into host memory serves at a sixth of the bandwidth with every other signal reading ok.
- Resident memory on the reference card reads ~1500, a spilled pool ~240.
- Below 500 the server logs a WARN at load and keeps serving (the threshold is one driver on one card, so it is a gauge, not a refusal).
- 0 = not measured (a pool too small to time).

`kv_ceiling_blocks` is what the pool may still grow to.

- Equal to `kv_blocks_total`: fixed pool at final size.
- Greater than `kv_blocks_total`: growable, not there yet (`kv_cache.growable`).
- `kv_pool_growable` disambiguates the case both report ceiling == total (fixed pool vs growable at its ceiling), which want opposite reactions: wait for the card to free, or stop waiting.
- A pool that is small **and** can still grow is not reported unhealthy: it heals as the card frees; wait for the total to climb rather than restart.
