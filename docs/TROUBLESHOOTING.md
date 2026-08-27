<!--
layer: L1
audience: operators
verified: 2026-08-28
commit: be825e4a
-->

# Troubleshooting

Symptom, cause, fix. Ordered by frequency.

## The model loads and answers nonsense

**Repeated tokens, or a fluent answer that contradicts itself.**

Check `stderr` for `packed NVFP4 weight reached the generic cuBLAS path`:
weights reached a kernel that cannot read them, the multiply was skipped, the
layer contributed nothing. Since #1403 an NVFP4 MoE checkpoint whose experts
do not fit is refused at load; on a current build this line is a new defect,
worth an issue.

**Fluent but wrong is the harder case.** Partial corruption stays grammatical.
Do not judge output quality from one short prompt; run the degeneration
battery (`tools/analysis/degen_suite.py`).

## Decode is far slower than the numbers in the docs

In order of likelihood:

1. **Something else is on the GPU.** On WSL2 `nvidia-smi` does **not** show a
   container holding the card. Check `docker ps` as well.
2. **CUDA graphs did not capture.** The log prints `Resolved dispatch: ...
   graphs=1` when they did. `graphs=0` typically costs 2-3x on decode.
3. **The NVFP4 decode cache is partial.** Look for
   `NVFP4 decode caches: FULL (n/n MoE layers)`. Anything less means some
   tensors decode from their slower source, and a partial cache also aborts
   graph capture.
4. **VRAM spilled to host.** A successful allocation proves nothing on
   WSL2/WDDM: the driver oversubscribes into host memory and returns success.
   The tell is bandwidth, roughly 1530 GB/s resident against 237 GB/s spilled,
   so the symptom is a ~6.5x cliff rather than an error. Reduce
   `runtime.max_seq_len` or the KV dtype.
5. **You are comparing against a number that was measured differently.** The
   prefill pin changed meaning on 2026-07-26 when one-shot runs stopped hitting
   the prefix cache. See [`PERF.md`](PERF.md).
6. **It is the host, not the change.** Decode on this box moves several percent
   between sessions. A single slow reading is not a regression until it
   reproduces in a paired A/B.

## The perf gate is red

A red gate is **not** a regression until it reproduces. Before investigating:

- re-run it; the host's own between-session movement is several percent
- check `docker ps` and `nvidia-smi` for a co-tenant
- if it still fails, A/B against `main` with alternating rounds, not two runs
  twenty minutes apart

If the change genuinely and intentionally moves performance, refresh the
baseline with `scripts/gen_perf_baseline.sh` **and say so in the PR**. A refresh
without that sentence is indistinguishable from a regression papered over.

## A request hangs, or streams nothing until the end

Almost always a reverse proxy buffering the response: set
`proxy_buffering off` (nginx) or equivalent, see
[`DEPLOYMENT.md`](DEPLOYMENT.md).

No proxy: a very large grammar can take long to build its first token mask.
Grammars are compiled and memoised; the first mask inside a deeply nested
state is the expensive one.

## Requests fail after the first one

`finish_reason: "cancelled"`, or the second identical request returns nothing.

Usually the KV pool is too small to hold one full-length sequence: the load
succeeds, every full-length request is cancelled at admission. Since v0.23.0
the condition is reported at load time for an operator-set `max_seq_len`.
Reduce `runtime.max_seq_len`, or pick a smaller KV dtype.

Also check `kv_cache.swa_snapshot_mb`: a value **below one snapshot size**
silently disables prefix caching, which is worse than setting it to zero. It
warns since #1092.

## Requests queue while `/metrics` shows free KV blocks

`imp_kv_blocks_reserved` is the answer (gauge since v0.29.0).

Admission reserves prompt **plus `max_tokens`** (#1635): a request admitted on
its prompt alone can run the pool dry mid-answer, truncating whichever
request needs the next block. Queueing is the better failure, so the promise
holds from admission until the blocks are written.

Cost: concurrency against a client that does not set `max_tokens`. The server
default is 8192, so each such request reserves `ceil(8192/block_size) + 1`
blocks whatever it emits. Set `max_tokens` to what the answer needs.

On a pool too small to ever hold prompt + `max_tokens` the reserve is clamped
to the pool (pre-#1635 behaviour): the mid-stream cancel stays possible, no
admission rule can promise memory that does not exist.

## After a restart, every prompt is cancelled

Restarting while the previous process still holds the card sizes the KV pool
against unreleased VRAM: the server comes up in seconds, loads the model,
then cancels every prompt past a few hundred tokens with "KV cache too small
for prompt", which reads as a statement about the prompt.

`GET /health` says so since v0.28.0: **503 with `code: "kv_pool_floored"`**,
plus `kv_capacity_tokens` in the body always. On an older build the tell is
`imp_kv_blocks_total` in `/metrics`: tens where a clean start reads
thousands, while `/v1/models` kept advertising the full context.

Fix: restart on a free card, not a retry. The pool is sized once, at startup;
wait for the old process to release the GPU.

## The server came up fine, but its numbers are wrong

Quieter half of the same fault: a false finding instead of a cancelled
request. A server started beside another process does not fail; it gets a
smaller KV pool, and `/health` reports that pool at its own ceiling because
the ceiling was computed against the same occupied card. No `/health` field
separates it from a healthy start, both read `ok`.

The load log says so since v0.29.0:

```
WARN  Weight upload consumed 8446 MiB of device free VRAM for a 3263 MiB
      checkpoint. The excess is not weights: another process is holding the card
      (on WSL2 it is invisible until this upload), or the upload spilled to host
      memory.
```

The weight upload is the first moment a neighbour is visible at all: under
WSL2/WDDM the driver reports the whole card as free until a process allocates
against it. Everything sized after that point (KV pool, decode caches) reads
the same shrunken residual and the load still reports success. Treat the
warning as a refusal to measure: free the card and restart before recording
any number from that process.

## The server answers 503

Either the model is suspended (`POST /admin/resume`), or the server was started
without `--model` and the request resolved to no model. `GET /v1/models` shows
what is available.

## An image is ignored, or refused

- `400 vision_unavailable` means the checkpoint loaded **text-only**: imp does
  not recognise its vision tower. This is a refusal on purpose, so you do not get
  a confident description of a picture the model never saw.
- `400` on an `image_url` means it could not be fetched or decoded. Dropping it
  silently would shift every later image onto the wrong placeholder.
- Check `MODELS.md` for which checkpoints can actually see.

## Constrained output comes back unconstrained

Since v0.23.0 this should be impossible: a pattern imp cannot compile is a `400`.
If you get free text at HTTP 200 with a `response_format` set, that is a bug
worth reporting, and it was the exact defect fixed in #1256.

## `content` is empty and the answer sits in `reasoning_content`

On a thinking model in a long conversation, raise `max_tokens` before
assuming a defect. The model thinks first and the budget is shared: once
thinking fills it, the reply never starts and imp returns empty `content`
with `finish_reason: stop`, an honest report of what was generated.

Measured on Qwen3.8-27B, one session grown to ~8k tokens over 74 turns:

| `max_tokens` | result |
|---|---|
| 260 | several turns with empty `content`, one reply that was a single non-Latin token |
| 600 | **74 of 74 turns clean**, every recall correct |

Not one model's quirk: Qwen3.6-35B-A3B runs the same 54-turn probe with 53
turns clean at `max_tokens` 600; its one failure is the same shape, a
`wrapup` turn cut off mid-sentence at `finish_reason: length` with 2 395
characters of thinking behind it.

The model, not the engine: the same conversation on vLLM with the same
checkpoint degenerates the same way, only visibly (vLLM has no reasoning
parser, thinking streams into `content`). A fixed conversation replayed at
identical depth (up to 5 005 prompt tokens) is answered correctly by both
engines, so context length alone is not the trigger.

The server logs `empty content: the answer never started because the token
budget went to reasoning`, with the amount of thinking and the finish reason.
Reproduce with `tools/analysis/multiturn_deep.py`.

## Build or test problems

- `build/` and `build-dev/` are root-owned by the container. Remove them with
  `make dev-clean` or a throwaway container, never `sudo`.
- `make test-unit` runs a **different binary** from the CI lane. Green there is
  not green in CI. The CI lane is `ctest -L unit`, which is `make dev-test`.
- `build-dev/` carries whichever branch was last compiled in it; `git checkout`
  does not rebuild.
- A `--gtest_filter` naming a value-parameterised suite without wildcards matches
  zero tests and reports `PASSED`. `DetEvalE2ETest.*` matches nothing;
  `*DetEvalE2ETest*` is correct.

More: [`CONTRIBUTING.md`](../CONTRIBUTING.md) and
[`internals/BENCHMARKING.md`](internals/BENCHMARKING.md).
