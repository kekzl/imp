<!--
layer: L1
audience: operators
verified: 2026-09-29
commit: 1989d10c
-->

# API: closed-choice scoring

Endpoint list and shared error rules: [`API.md`](API.md).

One prefill per item, no sampling: the answer is a softmax over candidate-token logits at the last prompt position (#2198).

`POST /v1/decide` request:

| field | type | rule |
|---|---|---|
| `model` | string | optional, as on every route |
| `evidence` | string | required; shared by every item |
| `system` | string | optional; default `Answer only with the letter of the correct option.` |
| `mode` | string | `auto` (= `serial`), `serial`, `direct`, `shared` |
| `items[]` | array | required, non-empty, at most `--max-batch-items` |
| `items[].id` | any | optional, echoed; default the item index |
| `items[].criterion` | string | required |
| `items[].options` | string array | 2 to 16 entries, labelled `A` to `P` |

Prompt per item:

- `[system, user]` through the model's chat template, generation prompt appended, thinking suppressed.
- user message: compact JSON, keys in this order: `{"evidence":...,"criterion":...,"options":{"A":...,"B":...}}`.
- evidence first, so every item shares the evidence prefix.

Response (values illustrative):

```json
{"object": "decide", "model": "...", "mode_used": "serial",
 "items": [{"id": "paid", "probs": {"A": 0.93, "B": 0.05, "C": 0.02}, "argmax": "A",
            "argmax_index": 0, "prompt_tokens": 71, "cached_tokens": 48}],
 "usage": {"prompt_tokens": 71, "cached_tokens": 48, "total_tokens": 71}}
```

| mode | behaviour |
|---|---|
| `serial` | item k+1 is submitted after item k finished: its evidence prefix is a prefix-cache hit (`cached_tokens` > 0 from item 2 on). Hybrid models: each item also saves a recurrent snapshot at the block floor of the prefix all items share (`Request::snapshot_hint_tokens`) |
| `direct` | all items submitted at once, each bypassing the prefix cache (`cached_tokens` = 0). `cache_prompt: false` does not do this; it only controls pinning |
| `shared` | item 1 alone; after it finished (its evidence blocks are published at finish), items 2..n are submitted at once, each reusing the evidence prefix, and run as rows of one ragged prefill (`runtime.prefill_batch`, row cap `prefill_chunk_size`, at most `max_batch_size` admitted per step). Hybrid models: each row restores the shared-prefix recurrent snapshot. Mamba2 and MLA models have no ragged prefill: rows prefill one by one, still reusing the prefix. `/v1/score` has one prompt: same as `serial` |

`POST /v1/score` request: `model`, exactly one of `prompt` (raw string, tokenized like `/v1/completions`) or `messages` (`[{role, content}]`, chat template + generation prompt), `candidates` (2 to 256 token strings or integer token ids), `mode` as above. Response: `candidates[] {candidate, token_id, logit, prob}`, `argmax_index`, `prompt_tokens`, `cached_tokens`, `mode_used`, `usage`.

Token guards, each a 400:

| check | example |
|---|---|
| a letter or candidate string encodes to exactly one token | `"Xylophone"` is several |
| `tokenize(prefix + c) == tokenize(prefix) + [id]`, prefix = prompt text after its last control token | a prompt ending in `:` or `>`: `":A"`, `">A"` are single tokens in Qwen, gpt-oss and Nemotron vocabularies |
| candidate ids are distinct and inside the vocabulary | |
