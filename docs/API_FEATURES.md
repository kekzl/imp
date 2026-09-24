<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# API features

Constrained decoding, tool calling, thinking/reasoning and images. Endpoints, sampling fields, metrics, prompt caching, request tracing, errors: [`API.md`](API.md).

## Constrained decoding

Four flavours, all enforced by masking at decode time rather than by
post-validation.

| form | status |
|---|---|
| `response_format: {"type": "json_object"}` | ✅ |
| `response_format: {"type": "json_schema", ...}` | ✅ whole-token validated. An `integer` is bounded at 19 digits (int64's width): unbounded, the sampler could stay in the digit state above temperature 0 and emit a 40-digit population (#1540). `enum`/`const` members may be strings, numbers, booleans or `null`; a non-string member is emitted exactly as written (`1.0` does not match `1`) |
| `response_format: {"type": "regex"}` / `guided_regex` | ✅ |
| `response_format: {"type": "grammar"}` / `grammar` / `guided_grammar` | ✅ GBNF, a pushdown simulator, so recursive and bracket-balanced formats work |

**A constraint imp cannot compile is a `400`, not an unconstrained answer.** That changed in v0.23.0 and it is a breaking difference from servers that log the rejection and answer anyway.

- Left recursion, undefined rules and a missing `root` are rejected at compile time.
- Since #1567, the JSON-Schema assertion keywords imp does not enforce are rejected too.
- Which ones, and the three shapes accepted with a weaker guarantee than asked for: [`LIMITATIONS.md`](LIMITATIONS.md#known-bad-and-known-limited-behaviour).

All four forms are carried by `/v1/responses` as `text.format` too.

- Until #1930 that transform mapped only `json_object` and `json_schema` and wrote nothing for the rest, so the same `regex` or `grammar` request was constrained on `/v1/chat/completions` and free text on `/v1/responses`, at 200.
- A `text.format` or a `tool_choice` shape the transform cannot map is now a `400` naming it, rather than a field deleted before the parser's own check could see it.

Schema, regex and grammar inputs are bounded: nesting past 64 levels, a `{n,m}`
repeat above 1024, or a pattern needing more than 100k NFA states is a `400`
rather than a stack overflow or an allocation storm (#1608, #1609).

Constrained replies parse even when they hit `max_tokens`: the closer-narrowing
and the no-closer-after-comma rules used to cancel each other out and release the
constraint entirely.

## Tool calling

✅. Tested: `make test-agents-external` (aider on OpenAI, Claude Code on Anthropic, Agents SDK on `/v1/responses`).

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

ChatML parses `<tool_call>` envelopes; other families have no enforced structure.

On every other family (measurement: 10 requests @ temperature 0.7 with one `get_weather` function):

| model | family | `tool_choice` | tool calls |
|---|---|---|---|
| gemma-3-12b Q4_K_M | `gemma` | `required` / named | **0 / 10** each |
| gemma-4-26B Q4_K_M | `gemma` | `required` | **0 / 10** |
| gpt-oss-20b MXFP4 | `harmony` | `required` / named | **0 / 10** each |
| Qwen3-4B Q8_0 | `chatml` | `required` / named | 10 / 10 each |

`"auto"` is not enforced: a best-effort call is what `auto` asks for (Gemma-4: 1 of 10).

**gpt-oss calls tools** (#1716). Envelope: channel with recipient, not tag:

```
<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Berlin"}<|call|>
```

Measurement on `gpt-oss-20b-mxfp4`, `tool_choice: "auto"` (10 requests each):

| path | before | after |
|---|---|---|
| `/v1/chat/completions` | 0 / 10 | **10 / 10** |
| the same, streaming | 0 / 10 | **10 / 10** |

`tool_choice: "required"` on `harmony` is still a 400: the FSM has no grammar
for this envelope, so the call is the model's choice rather than a guarantee.

Reasoning models separate their chain of thought into `reasoning_content` (Anthropic: `thinking`) rather than emitting it as the answer.

- On streaming, with thinking off: stream scans first tokens for `<think>` the model may open anyway.
- Without tools: releases on first word (client-side TTFT on Qwen3.8-27B-NVFP4, 27-token prompt, 97-105 -> 32-62 ms).
- With tools: holds up to 256 tokens so chain of thought does not stream as answer.

**On `/v1/messages`, thinking is opt-in** (#1541).

- Without `thinking` field: no thinking block, `content[0]` is text.
- With it: thinking block first.
- Measured on `Qwen3.6-27B-Text-NVFP4-MTP`:

| request `thinking` | `content` blocks | `content[0].text` |
|---|---|---|
| absent | `[text]` | `"Hi"` |
| `{"type":"adaptive"}` | `[thinking, text]` | `""` |
| `{"type":"adaptive","display":"omitted"}` | `[text]` | `"Hi"` |
| `{"type":"disabled"}` | `[text]` | `"Hi"` |

Only this dialect changed. On `/v1/chat/completions` the reasoning is a separate
`reasoning_content` field, so nothing shifts an index and the server's
`think_budget` default still applies.

A prior assistant message may carry its `reasoning_content` back (the Anthropic dialect folds `thinking` blocks into it); the Jinja template decides whether to render it (Qwen3.8 keeps it, `preserve_thinking`).

- Sending it back is what lets a hybrid (GDN) model restore its recurrent state at the end of the previous reply instead of at the prompt boundary (`server.transcript_snapshot`).
- The reply then re-tokenizes exactly, a forced think end included (it emits `"\n"` before `</think>` like the model's own close).
- Where the model wrote a non-canonical BPE split, the server splices the ids it forwarded instead of re-encoding the text (`server.transcript_token_reuse`, 256 transcripts).
- The built-in web UI sends it.

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

### The reasoning budget, and what the answer gets

Three knobs, and only one of them caps tokens:

| knob | dialect | what it does |
|---|---|---|
| `think_budget` / `--think-budget` | OpenAI | FRACTION of `max_tokens` (default 0.5) the model may spend reasoning |
| `thinking.budget_tokens` | Anthropic | a token count, converted to that same fraction (`N / max_tokens`, clamped to 1.0); `0` disables thinking |
| `reasoning_effort` / `reasoning.effort` | OpenAI / Responses | a TEMPLATE instruction, no token cap of its own. Identical prompt-token counts across efforts mean it never reached the template. On `/v1/responses` the effort additionally maps to a fraction (0.0 / 0.25 / 0.5 / 0.8). The loaded model's values: `/v1/models` `meta.reasoning_effort` |

Engine enforces answer reserve: reasoning force-closed (injected `</think>`) when it reaches `max_tokens - max(runtime.think_answer_reserve, max_tokens / 4)` or the fraction, whichever is later.

- `runtime.think_answer_reserve` default 256.
- Example: `max_tokens: 260` + 0.5 default = `max(130, 4) = 130` reasoning tokens max.

When answer never starts, response signals it:

| field | where |
|---|---|
| `usage.completion_tokens_details.reasoning_tokens` | OpenAI, non-stream and `include_usage` chunk |
| `usage.output_tokens_details.reasoning_tokens` | Anthropic (`message_delta` streaming), `/v1/responses` |
| `imp_finish_detail: "reasoning_budget_exhausted"` | beside `finish_reason` / `stop_reason` (imp-namespaced extra, same shape as `imp_spec_*` usage keys); appears only when `content` empty, no tool call, non-empty reasoning channel |
| `imp_requests_reasoning_exhausted_total` | `/metrics` |

`thinking` blocks carry `signature`, stream emits `signature_delta` before
`content_block_stop` (deterministic digest of block text, not attestation).
Prior-turn block with mismatched signature: `400`.

Stream holds back up to `server.agent_scan_limit` tokens (default 256) with tools
to scan for reasoning model (`<think>`). Without tools: holds 8 tokens, releases on first word.

On `/v1/chat/completions` at `max_tokens: 400` with JSON prompt, `think_budget: 0` and `enable_thinking: false` both fully disable reasoning.

Structured output disables thinking on its own: `json_mode`, `json_schema`, `tools`, `regex` and `grammar` all suppress it without either field.

- Answer and thinking share the token budget: a small `max_tokens` on a thinking model can be spent before the reply starts (empty `content`, `finish_reason: stop`).
- For short structured calls, disable thinking instead of raising every budget; see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## Images

✅ on `/v1/chat/completions`, as `image_url` content parts.

- Multiple images encoded in prompt order, max `--max-images-per-request` (default 8).
- Picture wider or taller than 16384 px: `400`.

`http(s)` URLs require `--allow-remote-images` (#1610); destination refused if loopback, link-local, RFC1918, CGNAT or ULA.

- No redirects.
- Body cap 32 MiB, 10 s read timeout.
- Data URIs work by default.

Refusal rules:

- Unreadable content part or block: `400` naming it, all three dialects.
| Endpoint | Reads |
|---|---|
| `/v1/chat/completions`, `/v1/responses` | `text`, `image_url` |
| `/v1/messages` | `text`, `image` (base64 or url), `tool_use`, `tool_result`, `thinking` |

- On `/v1/messages` the `system` field takes string or array of `text` blocks.
- Unreadable `image_url`: `400`, not skipped (would misalign later images).
  Message same regardless of error, does not echo URL.
- Model with unreadable vision tower: loads text-only, image request gets `400 vision_unavailable`.

No video. `temporal_patch_size` is parsed but used only as a still-image repeat.
