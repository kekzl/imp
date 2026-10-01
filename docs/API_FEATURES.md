<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# API features

Constrained decoding, tool calling, thinking/reasoning, images and fill-in-the-middle. Endpoints, sampling fields, metrics, prompt caching, request tracing, errors: [`API.md`](API.md).

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
| a named function | `chatml`, `llama3`, `harmony` (gpt-oss), `gemma` with the `<\|tool_call>` token (Gemma-4) |
| `"auto"` / `"none"` / absent | nothing to enforce, every family |

Named-function envelopes forced on the bare-args families (#2279), body = the function's `parameters` schema as JSON:

| family | forced open literal | close |
|---|---|---|
| `llama3` | `<function=NAME>` | `</function>` |
| `harmony` | `<\|channel\|>commentary to=functions.NAME <\|constrain\|>json<\|message\|>` | `<\|call\|>` |
| `gemma` (Gemma-4) | `<\|tool_call>call:NAME` | `<tool_call\|>` |

gemma-3 has no `<|tool_call>` token: a named function stays a 400 there. The 400 body names the family and the families that enforce.

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

`tool_choice: "required"` on `harmony` and `gemma` is still a 400: the name sits in the
envelope header and the FSM has no name enum there. A named function is forced (#2279).

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
| `enable_thinking` | OpenAI | `false` disables thinking; `chat_template_kwargs.enable_thinking` (vLLM/SGLang form) is read when the top-level field is absent |
| `stop` | OpenAI | matched against the answer only, never the reasoning, on both transports |
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
- Formats: JPEG and PNG. GIF, BMP, WebP and every other format: `400` (#2401).

`http(s)` URLs require `--allow-remote-images` (#1610); destination refused if loopback, link-local, RFC1918, CGNAT or ULA.

- No redirects.
- Body cap 32 MiB, 10 s read timeout.
- Data URIs work by default.

Refusal rules:

- Unreadable content part or block: `400` naming it, all three dialects.
| Endpoint | Reads |
|---|---|
| `/v1/chat/completions`, `/v1/responses` | `text`, `image_url`; `video` on chat completions |
| `/v1/messages` | `text`, `image` (base64 or url), `tool_use`, `tool_result`, `thinking` |

- On `/v1/messages` the `system` field takes string or array of `text` blocks.
- Unreadable `image_url`: `400`, not skipped (would misalign later images).
  Message same regardless of error, does not echo URL.
- Model with unreadable vision tower: loads text-only, image request gets `400 vision_unavailable`.

## Video

✅ on `/v1/chat/completions` for Qwen3-VL, as a `video` content part of frames the client sampled; no container decoding (mp4 is out by decision).

```json
{"type": "video", "video": ["data:image/png;base64,...", "data:image/png;base64,..."], "fps": 2}
```

| field | rule |
|---|---|
| `video` | 2..768 frame image URLs, same rules as `image_url` (data URI; `http(s)` only with `--allow-remote-images`), all frames the same size |
| `fps` | frame k sits at k/`fps` seconds; default 24 (HF `Qwen3VLProcessor` fallback) |
| `timestamps` | one number per frame, seconds; overrides `fps` |

- Frames pair up on the temporal axis (odd count repeats the last frame); each pair becomes one `<x.x seconds><\|vision_start\|>` + tokens + `<\|vision_end\|>` group, as `Qwen3VLProcessor` emits it.
- Resize bound: t*h*w in 4096..25165824 pixels (`video_preprocessor_config.json`), capped at frames x the per-image patch budget; `cap_pixels_per_frame` off (transformers 5.17.0 default).
- `imp-cli`: `--video-frames a.png,b.png,... [--video-fps F | --video-timestamps s0,s1,...]`.
- Wrong frame count, unreadable or mixed-size frames: `400`. Model without a Qwen3-VL tower: `400 vision_unavailable`.

## Fill-in-the-middle

Two entry points, one generation path (#2201):

| Entry | Fields | Response |
|---|---|---|
| `POST /infill` | `input_prefix`, `input_suffix`, `input_extra` `[{filename, text}]`, `prompt` (appended to the prefix), `n_predict` (alias of `max_tokens`), `model` optional; every `/v1/completions` sampling field | `text_completion`, plus top-level `content` and `stop` (llama.cpp clients read these), streaming too |
| `POST /v1/completions` | `suffix`: non-empty string = FIM with `prompt` as the prefix; `""` or absent = plain completion | `text_completion` |

- FIM token ids come from the tokenizer: GGUF `tokenizer.ggml.fim_{pre,suf,mid,pad,rep,sep}_token_id` (or the older `prefix/suffix/middle_token_id`), else the vocab's special or added tokens `<|fim_prefix|>` (Qwen2.5/3-Coder), `<fim_prefix>` (StarCoder), `<PRE>` (CodeLlama), `<｜fim▁begin｜>` (DeepSeek-Coder), `<|code_prefix|>` (GLM-4) and their suffix/middle partners.
- Prompt order is PSM: `[BOS] PRE prefix SUF suffix MID`, BOS only when the tokenizer asks for it.
- `input_extra` non-empty: the llama.cpp repo layout goes in front, `<|repo_name|>myproject\n` then `<|file_sep|>{filename}\n{text}` per chunk and `<|file_sep|>filename\n`; without a file separator token each chunk is preceded by `\n\n--- snippet ---\n\n`.
- The FIM marker texts are added as stop strings, so the model stops at `<|file_sep|>` or `<|fim_pad|>`.
- Model without FIM prefix, suffix and middle tokens: `400`, `code: "fim_not_supported"` (`param: "suffix"` on `/v1/completions`). A `suffix` with a token-id prompt, or a non-string field: `400`.
- The GPU acceptance run is `scripts/accept_2201.sh` (Qwen3-Coder-30B-A3B-Instruct-FP4, a Gemma-4 model for the 400).

## Responses store

`POST /v1/responses` with `store: true` keeps the response in process memory; a later request names it in `previous_response_id` instead of resending the transcript (#2206).

| item | behaviour |
|---|---|
| stored | the conversation input as the model saw it (earlier turns flattened), the output items (reasoning, message, function_call), the response object |
| `previous_response_id` | stored input + stored output + the new `input`, then the normal transform: the same prompt a client resending the transcript with `store: false` builds. Reasoning items are skipped on replay, as in a stateless resend |
| not carried over | `instructions`, `tools`, sampling fields: send them on every turn (OpenAI does not carry `instructions` either) |
| `store` absent | stored, same as `store: true` (OpenAI default), so Agents SDK clients can use `previous_response_id` without setting it. On a disabled store: stateless, no error |
| `store: false` | stateless, as before; `previous_response_id` still works against a stored predecessor |
| lifetime | `--responses-store-ttl` s from insertion (default 3600); a hit refreshes LRU order, not the TTL |
| caps | `--responses-store-max-entries` (default 1000) and `--responses-store-max-mib` (default 256); the least recently used entry goes first. An entry larger than the byte cap alone is not stored (logged) |
| off | any `--responses-store-*` flag at 0 (e.g. `--responses-store-max-entries 0`): nothing is stored, explicit `store: true` answers 400 (`param: "store"`) |
| unknown or expired id | 404, `code: "response_not_found"`, `param: "previous_response_id"` |
| `GET /v1/responses/{id}` | the stored response object; 404 `response_not_found` otherwise |
| `DELETE /v1/responses/{id}` | `{"id", "object": "response.deleted", "deleted": true}`; 404 otherwise |
| `GET /v1/responses/{id}/input_items` | not implemented: input items carry no ids to page by |
| ids | `resp_imp<counter><64 random bits>`: GET serves stored content, so ids are not enumerable |
| scope | one process, no persistence: a restart or a second replica does not see the entry. Auth is the server's `--api-key`; there is no per-key isolation |
| `/metrics` | `imp_responses_store_entries`, `imp_responses_store_bytes` (gauges), `imp_responses_store_evictions_total` (caps), `imp_responses_store_expired_total` (TTL) |

- Streaming responses are stored when `response.completed` / `response.incomplete` was written; a client that disconnects earlier stores nothing.
- Token identity of a two-turn continuation against the stateless resend on a real model: `scripts/accept_2206.sh` (GPU).
