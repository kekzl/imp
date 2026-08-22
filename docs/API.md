<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# API

What the HTTP surface actually accepts, as opposed to what an OpenAI-shaped
client might assume. Status legend is the one from
[`FEATURES.md`](FEATURES.md): ✅ code path plus a gated test, 🟡 code path, no
test.

**Two dialects, both native.** imp is not an OpenAI server with an Anthropic
shim bolted on, nor the reverse. `/v1/messages` is implemented against the
Anthropic wire format directly, and all three dialects share one per-token
streaming driver, so a fix in streaming lands in all of them at once.

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

### Defaults, and where they differ from OpenAI

A request that sets no sampling fields is not served with OpenAI's defaults
(#1596). The values are deliberate - they suit local models better than
`temperature 1.0` with no truncation - but they change what an identical
request returns compared to the OpenAI API, so they are stated here rather than
left in the source.

| field | imp | OpenAI | to get OpenAI's behaviour |
|---|---|---|---|
| `temperature` | 0.7 | 1.0 | send `"temperature": 1.0` |
| `top_p` | 0.95 | 1.0 | send `"top_p": 1.0` |
| `top_k` | 40 | (no field) | send `"top_k": 999999` (see below) |
| `repetition_penalty` | 1.05 | (no field) | send `"repetition_penalty": 1.0` |

Two of these do not switch off the way the field name suggests.

**`top_k: 0` is not "off", it is 50.** Every sampling site spells
`top_k > 0 ? top_k : 50` (`src/exec/executor.cu:193`, `:290`,
`src/runtime/engine_scheduler.cpp:2572`), so zero and "unset" both land on 50 -
a *tighter* truncation than the 40 default. The only way to disable top-k is a
value at or above the vocabulary size, which the dispatcher clamps to the full
vocabulary (`engine_scheduler.cpp:2573`).

**`repetition_penalty` has no OpenAI field at all**, so a strictly
spec-compliant client cannot switch it off and gets a mild anti-repetition bias
it never asked for. Sending the non-OpenAI field with value `1.0` does disable
it: the engine skips the penalty pass entirely when all three penalties are
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

## Constrained decoding

Four flavours, all enforced by masking at decode time rather than by
post-validation.

| form | status |
|---|---|
| `response_format: {"type": "json_object"}` | ✅ |
| `response_format: {"type": "json_schema", ...}` | ✅ whole-token validated |
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

✅, and gated by real clients rather than by our own idea of correct: `make
test-agents-external` drives imp with aider over the OpenAI dialect, Claude Code
over the Anthropic one and the OpenAI Agents SDK over `/v1/responses`, each
having to land an actual edit in a throwaway repository.

`tool_choice` that contradicts the request is a `400`: naming a function absent
from `tools`, or `"required"` with no tools.

Reasoning models separate their chain of thought into `reasoning_content`
(Anthropic: `thinking`) rather than emitting it as the answer. This holds on the
streaming path too, which is where it was once wrong.

**On `/v1/messages` that means `content[0]` is often not the text.** A reasoning
model returns a `thinking` block first and the answer second, so a client reading
`content[0].text` gets an empty string and concludes the model said nothing.
Select by `type`:

```python
text = "".join(b["text"] for b in resp["content"] if b["type"] == "text")
```

Verified on Qwen3-8B-Q8_0: `content` = `[{type: thinking, ...}, {type: text,
text: "The capital of France is Paris."}]`.

### Turning thinking off

Two request fields do it, and **either one alone is enough**:

| field | dialect | effect |
|---|---|---|
| `think_budget` | OpenAI | fraction of `max_tokens` reserved for reasoning. **0 disables thinking** |
| `enable_thinking` | OpenAI | `false` disables thinking |
| `thinking: {type: "disabled"}` | Anthropic | same, and zeroes the budget |

Measured on Qwen3.8-27B with a JSON prompt at `max_tokens: 400`, counting
`reasoning_content` characters: nothing set 160, `think_budget: 0` alone **0**,
`enable_thinking: false` alone **0**. The server's toggle test sets both because
it covers the combination, not because both are required.

**Why you would.** The answer shares the token budget with the thinking, so a
small `max_tokens` on a thinking model can be consumed before the reply starts,
and you get an empty `content` with `finish_reason: stop`. Short structured
calls (a JSON classifier at `max_tokens: 400`, say) are the usual case: disable
thinking there rather than raising every budget. See
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

Structured output disables thinking on its own: `json_mode`, `json_schema`,
`tools`, `regex` and `grammar` all suppress it without either field.

## Prompt caching

✅ on by default for the server. Prefix blocks are reused across requests that
share a prefix, which is what makes a growing agent transcript cheaper per turn
rather than more expensive.

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
makes the server open a connection to a host the caller names, and the caller is
unauthenticated by default, so it is behind `--allow-remote-images` (#1610).
With the flag on, the destination is resolved and refused if it is loopback,
link-local (including the cloud metadata address), RFC1918, CGNAT or ULA;
redirects are not followed, the body is capped at 32 MiB and the read has a
10 s timeout. See [`LIMITATIONS.md`](LIMITATIONS.md) for the residual.

Two refusals worth knowing, both deliberate:

- An `image_url` that cannot be read is a `400`, not a skipped picture. Dropping
  one would slide every later image onto the wrong placeholder. The message is
  the same whatever went wrong, and does not echo the URL, so the endpoint
  cannot be used to tell an open port from a closed one.
- A model whose vision tower imp cannot read loads **text-only** and says so; a
  request that sends it an image gets `400 vision_unavailable` rather than a
  confident description of a picture the model never received.

No video. `temporal_patch_size` is parsed but used only as a still-image repeat.

## Errors

Every error is a JSON envelope, never a bare status with an empty body, and
`/v1/messages*` paths get the Anthropic error shape rather than the OpenAI one.
An unmatched route answers with an envelope too.

Internal engine errors are translated to `ImpError` at the C API boundary
(`src/api/imp_api.cpp`); this is intentional and is why a load failure surfaces
as a typed message rather than a crash.

### `GET /health`, and which 503 is worth retrying

`/health` answers 200 with `status: "ok"` whenever the process can serve. Load,
queueing and a transient out-of-memory all stay 200 on purpose: the server is
alive and an orchestrator restarting on them makes things worse.

It answers **503 only for states that outlast the request that hit them**, and
then carries a stable `code` so a client can tell those apart from a retryable
one:

| `code` | what it means | what a client should do |
|---|---|---|
| `kv_pool_floored` | the KV pool fell back to its rescue floor, so it holds a few hundred tokens instead of a context. Sized once at startup, usually because another process still held the card | do **not** retry, the process cannot recover. Restart it on a free card |
| `engine_faulted` | the engine is wedged | restart |

Whenever a model is loaded the body also carries the pool capacity, healthy or
not, so a caller can check what it is about to send against what the server can
hold without scraping `/metrics`:

```json
{"status": "unhealthy", "code": "kv_pool_floored",
 "model_loaded": true, "queue_depth": 0, "suspended": false,
 "kv_blocks_total": 16, "kv_block_size": 32, "kv_capacity_tokens": 512,
 "kv_ceiling_blocks": 16, "kv_pool_growable": false}
```

`kv_ceiling_blocks` is what the pool may still grow to. Equal to
`kv_blocks_total` means a fixed pool at its final size; greater means it is
growable and has not got there yet (`kv_cache.growable`). `kv_pool_growable`
resolves the case where that pair cannot: a fixed pool and a growable one that
has already reached its ceiling both report ceiling == total, and the two want
opposite reactions — wait for the card to free, or stop waiting. A pool that is small
**and** can still grow is not reported as unhealthy at all: it heals as the card
frees, and a client should wait for the total to climb rather than restart the
server.

This exists because the quiet version of that state cost an operator an
afternoon: `/health` said ok, `/v1/models` kept advertising 131 072 tokens, and
every real prompt came back cancelled with a message about the prompt.
