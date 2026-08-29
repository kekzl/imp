# Qwen3.8-27B long-context posture: KV dtype, batch, and four ways to measure it wrong

Status: settled for the configuration questions, one item open (2026-08-29).
Trigger: an operator asked whether more context is available at good speed on
Qwen3.8-27B-NVFP4. It is, and the first four answers were all measurement
artefacts. Two sessions measured against each other on the same RTX 5090; the
peer numbers are marked, both hosts idle-checked over five samples.

## The recommendation

| deployment | posture | why |
|---|---|---|
| single user, long context | do NOT pin the KV dtype; pin `runtime.max_batch_size` (8 measured) | auto resolves NVFP4 on QWEN35; a pinned small batch frees recurrent-state VRAM into the KV pool |
| concurrent serving | leave the batch on auto, still do not pin the KV dtype | auto batch converts freed VRAM into slots, so the context gain is split, not lost |
| decode latency matters more than context | keep FP8 | NVFP4 costs ~11% decode at 78k, spec off |

`attention.sparse_topk_tokens` does not apply here: its v1 gates require F16 or
FP8 KV, so the NVFP4 default excludes it.

## What the configuration is worth

Server form, Qwen3.8-27B-NVFP4, `/v1/models` `max_model_len` is what a client sees:

| arm | KV B/tok | batch | SSM state | KV blocks | pool tokens | max_model_len | binding limit |
|---|---:|---:|---:|---:|---:|---:|---|
| FP8 pinned, batch auto (peer) | 32768 | 22 | 1749 MiB | 2714 | 86 848 | ~90 528 | the pool |
| NVFP4 auto, batch auto (peer) | 16384 | 28 | 2226 MiB | 3951 | 126 432 | 126 432 | the pool |
| NVFP4 auto, batch 8 pinned (peer) | 16384 | 8 | 636 MiB | 6949 | 222 368 | 131 072 | `max_seq_len` auto cap |

+45.6% context from dropping the FP8 pin alone. The pinned batch then moves the
binding limit off the pool entirely: past that point more pinning buys
concurrency headroom, not sequence length.

Two mechanisms behind those rows:

- `kv_cache.dtype=auto` resolves **NVFP4 on QWEN35 only** (`kv_nvfp4_default_safe`,
  measured +0.3% PPL for 2.7x the context). Everything else lands on FP8 (author
  hint or verified family) or FP16. So the same `--kv-fp8` pin halves KV on most
  families and DOUBLES it here.
- `max_batch_size: auto` on a GDN hybrid spends freed VRAM on slots: NVFP4 raised
  the auto batch 22 -> 28, and the recurrent state 1749 -> 2226 MiB ate part of
  the KV saving. Per-slot state is 79.5 MiB; a single-user deployment pays it for
  slots that stay empty.

Beyond the cap: `--max-seq-len 262144` (imp-cli, batch 1) took the pool to 5089
blocks / 162 848 tokens with unchanged decode, and warned that it could not reach
the request. The `max_seq_len: auto -> 131072` line names both bounds:
`vram_cap=388816, auto_cap=131072`.

## Quality at that length

`tools/analysis/niah_check.py`, v0.32.1, server, batch 8 pinned, depths
0.05/0.25/0.5/0.75/0.95. Lengths are `usage.prompt_tokens`, not the harness's
word-based estimate (which undershot 100 000 -> 81 908):

| arm | 81 908 tokens | 126 908 tokens |
|---|---|---|
| NVFP4 | 5/5 | **5/5** |
| FP8 | 5/5 | cannot serve it (`max_model_len` 101 024 in this shape) |

Retrieval only. It does not say long-chain reasoning holds at that length.

## Speed, and what it is not

Spec OFF is the only comparable pair (peer, 78 733 tokens, streaming, decode
between first and last token): **FP8 89.5 tok/s (1.2%) vs NVFP4 79.2 (2.0%)** —
NVFP4 is 11.5% SLOWER. Its win is capacity, not throughput.

Spec ON is not comparable, and the evidence is a mirror flip of the same script
on the same box:

| run | NVFP4 | FP8 |
|---|---|---|
| peer | 95.1 tok/s, speculation pays +20% | 89.5, speculation pays 0% |
| this session, same reproducer and shape | 88.0, `drafted=0` | 101.5, `drafted=384, accept 32.3%` |

The arms emitted different token counts (131 vs 155 at the same `max_tokens`).
Different KV precision gives different logits, greedy trajectories diverge, and
the n-gram matcher drafts from the request's own emitted text — so whether it
fires is a property of the generated TEXT, not of the KV path. Both the "+20% for
NVFP4" and the "+13% for FP8" readings are trajectory luck.

## Four ways this was measured wrong first

Each one produced a confident number that a report would have carried:

1. **An inherited dtype pin.** Every arm of the first table ran under
   `IMP_KV_FP8=1` from a compose file. The pin was correct when `auto` meant
   FP16; after the QWEN35 default it doubles KV bytes. It reads as a batch
   effect. (The variable reaches imp through the image's
   `docker-entrypoint.sh`, which bridges 19 `IMP_*` names into CLI flags — the
   engine itself retired that surface in #879, so a source grep says the pin is
   dead while it is live.)
2. **tok/s at short context.** The first speed table was measured with a few
   hundred prompt tokens, where per-step KV traffic is negligible — the one
   operating point at which the KV dtype cannot show a difference. It reported
   "costs nothing" from data that could not have shown a cost.
3. **Mismatched library-reserve state.** A `--rm` container with no
   `/home/imp/.cache/imp` mount plans with the 3900 MiB constant; a deployment
   with the mount plans with its measurement. Measured difference here: 716 MiB
   = 716 blocks at FP8 (3876 with mount vs 3157 without, predicted 3157 before
   the run). Block counts are only comparable within the same reserve state, and
   the gap is larger than most effects being hunted. On Qwen3-8B-Q8_0 the same
   mismatch line reads +4389 MiB.
4. **Diverging trajectories with speculation on.** See the mirror flip above.
   Compare KV configurations with `speculative.ngram=false`, and reject any pair
   whose generated token counts differ.

## Open

- The ~11% decode cost is measured on text with an identical sentence skeleton
  (numbers and nouns rotate, structure repeats). Real agent traffic is less
  structured; whether the gap holds there is unmeasured.
- The entrypoint env surface (19 names) outlives #879's "29 legacy reads -> 2"
  in the engine. Deliberate compatibility or drift is undecided; at least one
  name (`IMP_KV_FP8`) now has an inverted effect on one family.
