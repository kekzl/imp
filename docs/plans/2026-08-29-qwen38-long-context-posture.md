# Qwen3.8-27B long-context posture: KV dtype, batch, and five ways to measure it wrong

Status: closed (2026-08-30). The configuration questions settled on 2026-08-29;
the decode cost that was left open turned out to be a kernel defect and is fixed
in #1817, which reverses the one recommendation that had a trade in it.
Trigger: an operator asked whether more context is available at good speed on
Qwen3.8-27B-NVFP4. It is, and the first five answers were all measurement
artefacts. Two sessions measured against each other on the same RTX 5090; the
peer numbers are marked, both hosts idle-checked over five samples.

## The recommendation

| deployment | posture | why |
|---|---|---|
| single user, long context | do NOT pin the KV dtype; pin `runtime.max_batch_size` (8 measured) | auto resolves NVFP4 on QWEN35; a pinned small batch frees recurrent-state VRAM into the KV pool |
| concurrent serving | leave the batch on auto, still do not pin the KV dtype | auto batch converts freed VRAM into slots, so the context gain is split, not lost |
| decode latency matters more than context | still do not pin | since #1817 NVFP4 decodes 74.1 vs FP8's 72.4 tok/s in the same shape; the dtype pin now costs on both axes |

**Superseded 2026-08-30.** The third row used to read "keep FP8, NVFP4 costs
13.5% decode". That cost was a load-width defect in the NVFP4 decode kernel, not
a property of 4-bit KV, and it is fixed in #1817. There is no longer a
deployment shape in which pinning FP8 on this family is the better choice.

`attention.sparse_topk_tokens` applies here since #1818 (the v1 gates required
F16 or FP8 KV; the metadata kernel now has an NVFP4 arm). It stays default-off
and buys speed for retrieval accuracy - the table under "Sparse decode" below.

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

Measured on REAL prose (2.0 MB of repo text: README, docs, Python sources;
most frequent 8-gram appears 20x), 76 705 `prompt_tokens`, batch 8, v0.32.1,
both arms emitting the same 111 tokens - so this compares configurations, not
trajectories:

| arm | decode | spread | speculation |
|---|---:|---:|---|
| NVFP4 | 80.5 tok/s | 0.7% | `drafted=0` |
| FP8 | **93.2 tok/s** | 0.0% | `drafted=0` |

**FP8 is 15.8% faster**, and the n-gram matcher never fires in either arm. Two
consequences: the whole speculation axis is inert at this operating point (every
spec-on number in this ledger describes an effect real traffic does not see), and
the earlier structured-text pair (peer, spec off, 78 733 tokens: FP8 89.5 vs
NVFP4 79.2, 11.5%) UNDERSTATED the gap.

**Superseded 2026-08-30 by a stricter protocol.** Both runs above got equal
emitted-token counts by luck, not by construction; a third run of the same pair
read the opposite sign purely because one arm emitted 120 tokens and the other
111. With `max_tokens` set BELOW what the task needs, both arms are cut off at
the same number, and the configuration is read back from the startup line
instead of assumed:

| arm | startup line | decode | spread |
|---|---|---:|---:|
| NVFP4 | `dtype=NVFP4  attn_decode=paged_nvfp4` | 63.9 tok/s | 0.3% |
| FP8 | `dtype=FP8_E4M3  attn_decode=paged_fp8` | **72.5 tok/s** | 0.3% |

**FP8 is 13.5% faster.** Release image, no repo mount, 3 rounds, forced length.
This is the number to quote; the 15.8% and 11.5% above are the same finding
measured with weaker controls.

**Closed 2026-08-30 by #1817.** The gap was a defect, and fixing it reverses the
sign. Same harness, two images built from the same tree, arms alternating, all
18 runs emitting exactly 125 chunks:

| arm | dispatch line | decode (3 trials) | spread |
|---|---|---:|---:|
| NVFP4 before | `attn_decode=paged_nvfp4` | 64.1 / 64.1 / 64.0 | 0.1-0.5% |
| NVFP4 after | `attn_decode=paged_nvfp4` | 74.2 / 74.2 / 74.0 | 0.3-0.6% |
| FP8, both images | `attn_decode=paged_fp8` | 72.3 / 72.4 | 0.4-0.5% |

**+15.7% on NVFP4; FP8 unchanged**, which is also the control: an accidentally
shared arm would have moved both. NVFP4 now decodes 2.3% faster than FP8 and
still holds +45.6% context.

## Sparse decode on this KV dtype (#1818)

One binary, alternating arms, 77k prompt tokens, forced-equal emitted tokens,
`sparse decode attention ACTIVE` present in every sparse arm and absent in every
dense one (the positive control - a null result on a feature that never engaged
is the failure mode this guards):

| KV dtype | dense | sparse | NIAH (5 depths x 2 lengths) |
|---|---:|---:|---|
| FP8 | 72.4 | 96.8 | 5/5 at 81 908; cannot serve 126 908 |
| NVFP4 | 74.3 | **100.2** | dense 10/10, sparse 8/10 |

NVFP4 with sparse is the fastest arm and keeps the context advantage. The cost
is retrieval accuracy, and it is a budget curve, not a defect - the failures are
digit garbles of the right needle ("ZEBRA-1550-25" for "ZEBRA-155000-25"), and
they thin out monotonically as the budget grows:

| effective budget | decode | NIAH |
|---|---:|---|
| dense | 74.3 | 10/10 |
| 32 768 tokens | 89.4 | 9/10 |
| 8 192 tokens | 100.2 | 8/10 |

**Budgets here are EFFECTIVE tokens, not the configured value.** `budget_blocks`
is computed with the compile-time `kKVBlockSize` (16), while a model with
`n_kv_heads <= 4` runs `block_size = 32` (`engine_kv_cache_init.cpp:101`), so
`sparse_topk_tokens=N` delivers 2N tokens of budget and `sparse_min_ctx` engages
at twice its stated length. The startup line reports the 16-based arithmetic and
the ACTIVE line the real one, which is why the two disagree. Pre-existing, not
introduced by #1818, and it applies to FP8 models on the same geometry - fixing
it changes what the knob means for existing deployments, so it is its own change.

## Why the 4-bit cache used to be the slower one

Counter-intuitive on bytes (16 384 vs 32 768 B/tok), and the bytes are not what
decides it. Two candidate explanations died on the source:

- The GQA-batched tile kernel that makes FP8 fast elsewhere needs
  `head_dim == kTileHeadDim == 128`; this model has 256, so it never runs here.
- Both paths map `grid.y = n_heads` (24 Q heads over 4 KV heads), so neither
  re-reads KV more often than the other.

What is left is how the two kernels that DO run are built:

| | FP8 `splitk_fp8_pipeline_kernel` | NVFP4 `splitk_nvfp4_kernel` |
|---|---|---|
| KV staging | 4 `cp_async` sites, double-buffered | **none** |
| load width | `uint32_t` per 4 bytes | byte-wise `k_bytes[i]` in the inner loop |
| per token | - | two dependent scale-byte loads before any compute |
| per element | one convert | nibble unpack + UE4M3 scale + FMA |

**Row two was the whole cost; #1817 fixed it and rows one and three were red
herrings.** SASS at HD=256 before: 20 `LDG.E.U8` and 7 wide loads. `uint8_t*`
carries no provable alignment, so ptxas cannot merge four adjacent byte reads,
while the FP8 twin had been reading `uint32_t` since it was written. Reading the
same bytes one word at a time, and issuing K and V before the warp reduction
rather than V after it, leaves 2 `LDG.E.U8` (the two scale bytes) at 56
registers and zero spills, unchanged from before.

The staging row is a genuine dead end, separately measured: a double-buffered
smem variant regressed 3% on 2026-05-08 (147.0 → 142.7 tok/s) and the note in
`attention_paged_nvfp4.cu` still says don't re-attempt it. Same for #1785's
GQA-tile variant at -9%. Both were traffic-sharing levers on a kernel whose
traffic is L2-served; the instruction count was the thing nobody had counted.

The asymmetry is historical: FP8 has five decode kernels from two optimisation
rounds (#899/#900 added the tile and GQA-tile variants), NVFP4 has two. An
earlier investigation (`docs/audit/PERF_LOG.md`, P1) found the NVFP4 attention
path was dead code on the model it examined, because KV-FP8 is the auto default
there - so it never came under optimisation pressure. On QWEN35 it is the
default.

Closed by #1817: the vectorised half of that plan shipped and NVFP4 now wins on
both axes. The `cp_async` half was not built and should not be, for the reason
above. No nsys profile was needed in the end - `cuobjdump -sass` answered it,
and the two void profile attempts (one shape where speculation masked
everything, one against the stray config below) never had to be repeated.

## Five ways this was measured wrong first

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
4. **Diverging trajectories.** See the mirror flip above. Different KV precision
   gives different logits, and imp's forward is not bit-deterministic either, so
   two runs of the SAME arm can stop at different lengths. Equal counts must be
   FORCED (a `max_tokens` below what the task needs), not observed: three of the
   pairs here were compared on counts that happened to match, and the one that
   did not match read the opposite sign.
5. **Config identity.** A dev build run with the repo mounted as its working
   directory reads `./imp.conf`; the release image does not. A stray, gitignored
   `imp.conf` in this repo (`kv_cache.dtype = "fp8"`, `server.prefix_cache =
   true`) therefore made both arms of two profiling runs the SAME arm - identical
   kernel lists, 12 169 vs 12 143 ms total, every row within 3 ms. The tell was
   the FP8 attention kernel appearing in the NVFP4 arm's profile, not the
   timings, which looked plausible. imp prints `imp.conf loaded from <path>` at
   startup; the failure was filtering it out of the log. Record that line with
   the measurement.

## Open

- ~~The decode cost is measured on structured text only.~~ Closed 2026-08-30:
  on real prose the gap is larger (15.8%, above), and the matcher fires in
  neither arm.
- The entrypoint env surface (19 names) outlives #879's "29 legacy reads -> 2"
  in the engine. Deliberate compatibility or drift is undecided; at least one
  name (`IMP_KV_FP8`) now has an inverted effect on one family.
