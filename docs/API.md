<!--
layer: L1
audience: operators
verified: 2026-08-28
commit: be825e4a
-->

# API

What the HTTP surface actually accepts. Status legend from
[`FEATURES.md`](FEATURES.md): ✅ code path plus a gated test, 🟡 code path, no
test.

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
| `POST /v1/responses` | ✅ | OpenAI Responses, the dialect Codex and the Agents SDK speak by default |
| `POST /v1/embeddings` | ✅ | needs an embedding model loaded |
| `POST /v1/rerank`, `POST /rerank` | ✅ | Cohere/Jina/vLLM shape |
| `POST /tokenize`, `POST /detokenize` | ✅ | |
| `GET /v1/models` | ✅ | loaded model plus the rest of the directory, each with `loaded: true|false` |
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
| `logprobs` | ✅ | `tests/test_server_logprobs.py` in `make test-server`, plus `tests/test_logprobs_shapes.cpp` in the CPU lane. Streaming emitted none whenever a `stop` sequence was set until #1588; `/v1/completions` returned the Chat shape until #1589 |
| `best_of` | ⚪ | `best_of > 1` is a 400: imp generates no candidate set to choose from (#1598) |
| DRY, mirostat, typical_p, logit_bias | ✅ | |
| `"speculative": true/false` | ✅ | per-request override; also bridged from the Anthropic shape. `false` switches off **all three** drafters (n-gram, MTP head, token recycling) since #1639 - it used to reach only the n-gram matcher. `true` enables what the model and config allow; it cannot conjure an MTP head the checkpoint lacks |
| `"lora": "name"` | ✅ | PEFT adapter hot-swap, works with every quant path |
| `"priority": int` | ✅ | vLLM-compatible admission priority, **lower value schedules earlier**, default 0. Strictly dominates the scheduler's shortest-first-with-aging order; a caller that sets priorities owns starvation across classes. Accepted on all three dialects |

### Defaults, and where they differ from OpenAI

A request that sets no sampling fields is not served with OpenAI's defaults
(#1596). Deliberate: the values suit local models better than
`temperature 1.0` with no truncation, but an identical request returns
different output than against the OpenAI API.

| field | imp | OpenAI | to get OpenAI's behaviour |
|---|---|---|---|
| `temperature` | 0.7 | 1.0 | send `"temperature": 1.0` |
| `top_p` | 0.95 | 1.0 | send `"top_p": 1.0` |
| `top_k` | 40 | (no field) | send `"top_k": 999999` (see below) |
| `repetition_penalty` | 1.05 | (no field) | send `"repetition_penalty": 1.0` |

Two of these do not switch off the way the field name suggests:

- **`top_k: 0` is not "off", it is 50.** Every sampling site spells
  `top_k > 0 ? top_k : 50` (`src/exec/executor.cu:193`, `:290`,
  `src/runtime/engine_decode_pipeline.cpp:82`): zero and "unset" both land on
  50, a *tighter* truncation than the 40 default. Disabling top-k needs a
  value at or above the vocabulary size, which the dispatcher clamps to the
  full vocabulary (`engine_decode_pipeline.cpp:83`).
- **`repetition_penalty` has no OpenAI field**, so a strictly spec-compliant
  client cannot switch it off. Sending the non-OpenAI field with value `1.0`
  disables it: the engine skips the penalty pass when all three penalties are
  neutral (`src/runtime/engine_sampling_stop.cpp:212`).

### Metrics for what the server decided

`/metrics` carries counters for the decisions that used to be visible only in
the server log (#1640, #1641):

| counter | what moves it |
|---|---|
| `imp_requests_timed_out_total` | the server ended a request at `--request-timeout`. The client sees `finish_reason: "length"`, which is also what a spent token budget produces - this counter is the only way to tell them apart |
| `imp_kv_pressure_rejections_total` | a request was cancelled because the KV pool could not give it blocks (admission or mid-decode). Not incremented for a failed metadata allocation or a snapshot mismatch, which are different faults |
| `imp_kv_pool_growths_total` | the growable pool committed more memory. A pool that keeps growing under load is the signal that arrives before it stops being able to |

`imp_requests_cancelled_total` remains client-disconnect only.

`imp_kv_blocks_reserved` (gauge) is what admission has promised to running
requests for the rest of their generation and not yet written (#1635). Free
blocks minus this gauge is what the next request is admitted against, so a
queue in front of a pool with free blocks is explained by this number and by
nothing else in `/metrics`.

## Constrained decoding

Four flavours, all enforced by masking at decode time rather than by
post-validation.

| form | status |
|---|---|
| `response_format: {"type": "json_object"}` | ✅ |
| `response_format: {"type": "json_schema", ...}` | ✅ whole-token validated. An `integer` is bounded at 19 digits (int64's width): unbounded, the sampler could stay in the digit state above temperature 0 and emit a 40-digit population (#1540) |
| `response_format: {"type": "regex"}` / `guided_regex` | ✅ |
| `response_format: {"type": "grammar"}` / `grammar` / `guided_grammar` | ✅ GBNF, a pushdown simulator, so recursive and bracket-balanced formats work |

**A constraint imp cannot compile is a `400`, not an unconstrained answer.** That
changed in v0.23.0 and it is a breaking difference from servers that log the
rejection and answer anyway. Left recursion, undefined rules and a missing
`root` are rejected at compile time, and since #1567 so are the JSON-Schema
assertion keywords imp does not enforce. Which ones, and the three shapes that
are accepted with a weaker guarantee than asked for, are listed in
[`LIMITATIONS.md`](LIMITATIONS.md#known-bad-and-known-limited-behaviour).

Schema, regex and grammar inputs are bounded: nesting past 64 levels, a `{n,m}`
repeat above 1024, or a pattern needing more than 100k NFA states is a `400`
rather than a stack overflow or an allocation storm (#1608, #1609).

Constrained replies parse even when they hit `max_tokens`: the closer-narrowing
and the no-closer-after-comma rules used to cancel each other out and release the
constraint entirely.

## Tool calling

✅, gated by real clients: `make test-agents-external` drives imp with aider
over the OpenAI dialect, Claude Code over the Anthropic one and the OpenAI
Agents SDK over `/v1/responses`, each having to land an actual edit in a
throwaway repository.

`tool_choice` that contradicts the request is a `400`: naming a function absent
from `tools`, or `"required"` with no tools.

**`tool_choice` that this server cannot enforce is also a `400`** (#1592), with
`code: "tool_choice_unenforceable"`. The decode FSM constrains the tool envelope
only where the loaded model's chat template has a grammar for it:

| `tool_choice` | enforced on |
|---|---|
| `"required"` | `chatml` |
| a named function | `chatml`, `llama3` |
| `"auto"` / `"none"` / absent | nothing to enforce, every family |

On every other family the constraint used to degrade to a sentence in the prompt
and the request was answered `200` with prose. Measured before the refusal, 10
requests each at `temperature 0.7` with one `get_weather` function:

| model | family | `tool_choice` | tool calls |
|---|---|---|---|
| gemma-3-12b Q4_K_M | `gemma` | `required` / named | **0 / 10** each |
| gemma-4-26B Q4_K_M | `gemma` | `required` | **0 / 10** |
| gpt-oss-20b MXFP4 | `harmony` | `required` / named | **0 / 10** each |
| Qwen3-4B Q8_0 | `chatml` | `required` / named | 10 / 10 each |

`"auto"` is untouched: Gemma-4 produced 1 of 10 there, and a best-effort call is
what `auto` asks for.

**gpt-oss calls tools now** (#1716). Its envelope is a channel with a recipient,
not a tag:

```
<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Berlin"}<|call|>
```

Before #1716 the parser had no Harmony branch: the call fell through to the
ChatML `<tool_call>` scanner and was dropped, an **empty `content` with
`finish_reason: "stop"`**. Measured on `gpt-oss-20b-mxfp4`,
`tool_choice: "auto"`, 10 requests per row:

| path | before | after |
|---|---|---|
| `/v1/chat/completions` | 0 / 10 | **10 / 10** |
| the same, streaming | 0 / 10 | **10 / 10** |

`tool_choice: "required"` on `harmony` is still a 400: the FSM has no grammar
for this envelope, so the call is the model's choice rather than a guarantee.

Reasoning models separate their chain of thought into `reasoning_content`
(Anthropic: `thinking`) rather than emitting it as the answer. This holds on the
streaming path too, which is where it was once wrong.

**On `/v1/messages`, thinking is opt-in** (#1541). A request without a `thinking`
field gets no thinking block, so `content[0]` is the text - which is what the
Anthropic dialect promises. Ask for it and the thinking block comes first, ahead
of the answer, the way upstream orders it. Measured on `Qwen3.6-27B-Text-NVFP4-MTP`:

| request `thinking` | `content` blocks | `content[0].text` |
|---|---|---|
| absent | `[text]` | `"Hi"` |
| `{"type":"adaptive"}` | `[thinking, text]` | `""` |
| `{"type":"adaptive","display":"omitted"}` | `[text]` | `"Hi"` |
| `{"type":"disabled"}` | `[text]` | `"Hi"` |

Only this dialect changed. On `/v1/chat/completions` the reasoning is a separate
`reasoning_content` field, so nothing shifts an index and the server's
`think_budget` default still applies.

**When you do ask for thinking, `content[0]` is not the text.** Select by
`type` rather than by index:

```python
text = "".join(b["text"] for b in resp["content"] if b["type"] == "text")
```

Verified on Qwen3-8B-Q8_0: `content` = `[{type: thinking, ...}, {type: text,
text: "The capital of France is Paris."}]`.

### Turning thinking on and off

On `/v1/messages` it is off unless asked for. On `/v1/chat/completions` the
server default (`--think-budget`, 0.5) applies and these fields turn it off -
**either one alone is enough**:

| field | dialect | effect |
|---|---|---|
| `think_budget` | OpenAI | fraction of `max_tokens` reserved for reasoning. **0 disables thinking** |
| `enable_thinking` | OpenAI | `false` disables thinking |
| `thinking: {type: "disabled"}` | Anthropic | same, and zeroes the budget |
| `thinking: {type: "enabled"\|"adaptive"}` | Anthropic | thinking on. **Required to get it at all on this dialect** (#1541); `adaptive` is what current SDKs send |
| `thinking: {budget_tokens: N}` | Anthropic | converted to a fraction of `max_tokens`. `0` disables thinking outright |
| `thinking: {display: "omitted"}` | Anthropic | the model still reasons; the `thinking` block is not returned, on either transport |

`thinking` blocks carry a `signature`, and the stream emits `signature_delta`
before the block's `content_block_stop` (SDKs round-trip the pair). It is a
deterministic digest of the block text, not an attestation: it proves the
block came back unedited, nothing more.

Either field alone suffices. Measured on Qwen3.8-27B, JSON prompt at
`max_tokens: 400`, `reasoning_content` characters: nothing set 160,
`think_budget: 0` alone **0**, `enable_thinking: false` alone **0**.

**Why disable it:** the answer shares the token budget with the thinking, so a
small `max_tokens` on a thinking model can be consumed before the reply
starts: empty `content`, `finish_reason: stop`. For short structured calls
(a JSON classifier at `max_tokens: 400`), disable thinking rather than raising
every budget. See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

Structured output disables thinking on its own: `json_mode`, `json_schema`,
`tools`, `regex` and `grammar` all suppress it without either field.

## Prompt caching

✅ on by default for the server. Prefix blocks are reused across requests that
share a prefix: a growing agent transcript gets cheaper per turn, not more
expensive.

`cache_control` is honoured per breakpoint: the **last** marked block bounds the
pinned region, rather than pinning the whole prompt. Usage reporting carries
`cache_read_input_tokens` and `cache_creation_input_tokens`.

The cache is keyed on the picture as well as the token ids for image requests,
and it is model-fingerprint-gated on disk, so a cache file from another model is
never replayed.

## Images

✅ on `/v1/chat/completions`, as `image_url` content parts. Several images in one
request are encoded in prompt order.

**A data URI works out of the box; an `http(s)` URL does not.** Fetching one
opens a server-side connection to a caller-named host with an unauthenticated
caller, so it is behind `--allow-remote-images` (#1610). With the flag on, the
destination is refused if it resolves to loopback, link-local (including the
cloud metadata address), RFC1918, CGNAT or ULA; redirects are not followed,
body capped at 32 MiB, 10 s read timeout. Residual:
[`LIMITATIONS.md`](LIMITATIONS.md).

Two deliberate refusals:

- An `image_url` that cannot be read is a `400`, not a skipped picture:
  dropping one would slide every later image onto the wrong placeholder. The
  message is the same whatever went wrong and does not echo the URL, so the
  endpoint cannot distinguish an open port from a closed one.
- A model whose vision tower imp cannot read loads **text-only** and says so;
  an image request to it gets `400 vision_unavailable` rather than a confident
  description of a picture the model never received.

No video. `temporal_patch_size` is parsed but used only as a still-image repeat.

## Request tracing

Send an `X-Request-Id` header and every response echoes it back - refusals
and unmatched routes included - sanitized to printable ASCII and capped at
128 chars. The generation endpoints answer with the server's own completion
id when no client id was sent, so every generation response carries some id
a caller can quote. With `--log-requests`, the JSONL record carries the
client id as `client_request_id` next to the server `req_id`, which is the
join an agent framework needs to attribute its own latency to this hop.
There is no OpenTelemetry export; the id propagation here is the wire half.

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

A stream that ends on a server-side fault emits an `error` SSE event instead of
`message_delta`/`message_stop`: a request timeout and an admission refusal used
to arrive as an ordinary completed turn, indistinguishable from the model
finishing.

`anthropic-version` and `anthropic-beta` are read and echoed back, neither
enforced (upstream: missing version = 400, unknown beta = refused; a client
that works here can fail there). An unknown beta is logged once per value: imp
implements no beta surface, and a silent 200 would be a false accept.

Internal engine errors are translated to `ImpError` at the C API boundary
(`src/api/imp_api.cpp`); this is intentional and is why a load failure surfaces
as a typed message rather than a crash.

### `GET /health`, and which 503 is worth retrying

`/health` answers 200 with `status: "ok"` whenever the process can serve.
Load, queueing and a transient out-of-memory stay 200 on purpose: the server
is alive, and an orchestrator restarting on them makes things worse.

**503 only for states that outlast the request that hit them**, with a stable
`code`:

| `code` | what it means | what a client should do |
|---|---|---|
| `kv_pool_floored` | the KV pool fell back to its rescue floor, so it holds a few hundred tokens instead of a context. Sized once at startup, usually because another process still held the card | do **not** retry, the process cannot recover. Restart it on a free card |
| `engine_faulted` | the engine is wedged | restart |

Whenever a model is loaded the body carries the pool capacity, healthy or
not, so a caller can check a prompt against server capacity without scraping
`/metrics`:

```json
{"status": "unhealthy", "code": "kv_pool_floored",
 "model_loaded": true, "queue_depth": 0, "suspended": false,
 "kv_blocks_total": 16, "kv_block_size": 32, "kv_capacity_tokens": 512,
 "kv_ceiling_blocks": 16, "kv_pool_growable": false}
```

`kv_ceiling_blocks` is what the pool may still grow to: equal to
`kv_blocks_total` = fixed pool at final size; greater = growable, not there
yet (`kv_cache.growable`). `kv_pool_growable` disambiguates the case both
report ceiling == total (fixed pool vs growable at its ceiling), which want
opposite reactions: wait for the card to free, or stop waiting. A pool that is
small **and** can still grow is not reported unhealthy: it heals as the card
frees; wait for the total to climb rather than restart.

Before this field existed the state was silent: `/health` said ok,
`/v1/models` kept advertising 131 072 tokens, every real prompt came back
cancelled with a message about the prompt.
