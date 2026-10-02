# Sparse decode attention: Quest-class top-k page selection

Status: SHIPPED opt-in (2026-08-28). Roadmap Open item 2 (long context served
by a 2023-era answer). Mechanism trigger per the BitDecoding shelf note: paged
attention is 19.9%/43.9% of the dense decode window at 8k/32k and 29.1%/50.6%
on MoE (ceiling 1.76-2.0x at 32k if attention were free).

## Measured (Qwen3-8B-Q8_0, fp8 KV, budget 4096, 3/3 alternating rounds)

| ctx | dense tok/s | sparse tok/s | delta | regime |
|---|---:|---:|---:|---|
| 32768 | 160.3 | 199.5 | +24.5% | selection |
| 16384 | 202.1 | 212.0 | +4.9% | selection |
| 8192 | 230.4 | 223.8 | -2.9% | identity (`sparse_min_ctx`) |
| 2048 | 258.1 | 251.5 | -2.6% | identity |

```
[PROV: commit=899301c6 date=2026-08-28 hw=RTX5090 model=Qwen3-8B-Q8_0
       quant=Q8_0 (fp8 KV) cuda=13.3 path=imp-cli n=3 alternating rounds,
       fresh process per arm, make-build image
       cmd=`imp-cli --kv-fp8 --bench --bench-pp 32768|16384|8192|2048
       --bench-reps 1 --max-tokens 136 --max-seq-len 40960
       --set speculative.ngram=false [--set attention.sparse_topk_tokens=4096]`]
```

Kernel budget per layer per step at 32k (nsys, dev build, same code): score
14.7 us + select 11.7 + batched minmax update (amortized) + paged attention
11.6 vs dense attention 74.4. Two build-out lessons, measured: sizing the
scores row from `max_tokens_` (the 4k per-forward chunk cap) silently disabled
the gate past 4k ctx - the first NIAH/perf pass measured dense vs dense; and
the identity regime cost -11..-14% before the score kernel exited ahead of its
q-smem staging and the per-layer metadata updates were batched into one
launch. Under CUDA graphs a host-side "feature active" log line can never fire
(dispatch code runs at capture, kernels at replay) - activity proof is nsys
kernel presence, not logs.

## Mechanism

Keep the whole KV, read it sparsely. Per attention layer, per decode step:

1. Per-block key min/max metadata (FP16, per kv_head x head_dim, updated at KV
   write time) gives an upper bound on any query dot product against the block:
   `bound_h(b) = sum_d max(q_h[d]*min[d], q_h[d]*max[d])` (Quest, MIT-HAN-lab).
2. Score every context block with `max_h bound_h(b)`; select top
   `budget_blocks` (sink + recent blocks always forced in).
3. Build a compacted block table + context length, ascending block order,
   device-side. The unmodified paged attention kernel runs on the compacted
   table: block-table remap only, zero kernel variants touched, every KV dtype
   would work (v1 gates to F16 + FP8 read-back).

Device-side selection makes it CUDA-graph-safe (ctx grows during replay; all
inputs are device arrays). `n_blocks <= budget` short-circuits to an identity
copy of the table: bit-identical to dense attention.

## v1 gates (checked at init unless noted)

| gate | why |
|---|---|
| `attention.sparse_topk_tokens > 0` (default 0 = off) | opt-in |
| KV dtype F16 or FP8_E4M3 | metadata kernel reads keys back from the cache (post-RoPE, exact w.r.t. what attention reads); 2 dequants in v1. FP8 stores raw scale-1 min/max: the per-layer scale is a positive constant factor per score and cannot change the ranking |
| uniform KV geometry (scalar KVCache ctor) | per-layer offsets not wired; excludes Gemma-4 dual geometry |
| `kv_cache.growable=false` | metadata pool sized once at init |
| not MLA | absorbed decode has its own latent cache |
| `speculative.token_recycling=false` | copy_blocks_device does not copy metadata |
| no persistent prefix cache (`prefix_cache_path` empty) | disk-restored blocks bypass the KV write path and would carry empty metadata. In-memory prefix reuse is fine: metadata lives per block and the reused blocks were written normally; the full-hit last-token re-write is idempotent |
| per layer (dispatch time): `sliding_window == 0 && n_sinks == 0` | SWA/StreamingLLM layers are already bounded |
| ~~per step: plain decode only~~ closed 2026-08-29 | spec verify chunks ride the sparse table too: chunk rows are already per-row "sequences" with own context lens and replicated tables - the exact shape the selection kernels take. Pad rows attend 1 token (identity path); repeated pad positions are handled by the consecutive-slot span clamp |

Rollback/overwrite after rejected speculation only loosens the bound (min/max
over a superset), never tightens it: selection quality degrades marginally,
correctness does not.

## Metadata maintenance (race-free without atomics)

Owner-CTA scheme in one kernel over the written tokens, launched after every
KV write (both the generic `write_kv_cache` funnel and the fused-RoPE decode
write): CTA i is active iff its block differs from token i-1's block; it scans
forward over the launch's same-block tokens (adjacent in every real call
shape: prefill contiguous, ragged row-range per seq, verify chunk contiguous,
multi-seq decode 1 token/seq with exclusive blocks). slot 0 initializes,
otherwise merge with stored metadata. Block reuse is covered by the slot-0
init (a fresh block's first write is always slot 0).

## Cost model (32k dense, fp8 KV, budget 4096)

Metadata bytes = 12.5% of K+V bytes (fp8) / 6.25% (fp16). Per step per layer:
scan all metadata (12.5%) + read selected pages (12.5%) ~ 25% of full
attention traffic. e2e bound at 43.9% attention share: ~1.44x. Overhead at
short ctx: +3 graph-replayed launches per attention layer per step
(~1-2 us each); feature default-off, documented.

## Files

- `src/compute/attention_sparse_select.cu/.h` - minmax update, block scoring,
  top-k select + table build (ballot-compaction, ascending)
- `src/memory/kv_cache.{h,cu}` - optional `key_minmax` pool (charged as
  `kv_cache_minmax`)
- `src/exec/executor_kv_write.cu`, `executor_attention_decode.cu` - update +
  dispatch wiring (pointer swap)
- `src/core/config/attention.h`, `src/runtime/config.cpp` -
  `attention.sparse_topk_tokens|sparse_sink_tokens|sparse_recent_tokens`
- `src/runtime/engine_kv_cache_init.cpp` - eligibility + pool pricing

## Verify chunks on the sparse table (2026-08-29)

Speculation ON (n-gram default) at 32k on the NIAH-filler workload (echo-heavy,
5.25-5.67 tok/verify), 3/3 alternating rounds, make-build images:

| arm | tok/s | ms/verify |
|---|---:|---:|
| all dense | 124.5 | - |
| sparse, chunks dense (#1805 = main) | 137.4 | 233 |
| sparse incl. verify chunks | 176.1 | 133 |

+28.2% over #1805, +41.4% over dense; NIAH 32k with spec ON: dense 15/15,
sparse 15/15 (`fp8_sparse4k_spec` config, --max-gen-tokens 768).

Two gate traps that made the first two B-vs-C measurements read NEUTRAL
(the change was silently inactive both times - launch counts, not logs,
proved it):

- scratch rows were sized from max_batch (8 at M=1); chunk rows present as
  n_sequences up to the 33-row chunk cap (`engine_spec_capture.cpp`) - a
  17-row chunk failed the row gate.
- spec verify row tables carry 16 slack blocks past the context ceiling
  (`table_cap = ctx_blocks + 16`); the scores-row capacity gate compared
  against the unslacked ceiling and failed every chunk.

## Serving regime (2026-08-29)

Concurrent long-context serving, Qwen3-8B-Q8_0 fp8 KV, imp-server, decode
rate via the tg8/tg520 differential (per-arm prefill wall cancels), fresh
server per arm, 3 alternating trials
(`tools/analysis/serving_sparse_ab.sh`):

| geometry | dense (median, spread) | sparse budget 4096 | delta |
|---|---:|---:|---:|
| 3 streams x 25k ctx, resident | 155.6 (150.3-173.8) | 197.7 (194.4-198.2) | **+27%** |
| 3 streams x 30k / 6 x 15.5k, ON arm at 689 MiB free | numbers invalid | numbers invalid | WDDM spill |

Findings that gate the numbers:

- **The metadata pool is the #1103 spill trap at serving scale.** 928 MiB at
  6600 blocks; an operator `kv_cache.max_blocks` pin that does not include it
  ran the ON arm at 689 MiB "free" - cudaMalloc still succeeds, WDDM spills,
  and EVERY prefill kernel ran uniformly +11% (launch counts identical; the
  per-kernel inflation and its disappearance under `cuda_graphs=never` -
  which frees enough VRAM to fit - were the fingerprints). The pinned-pool
  path now WARNS with the exact MiB; auto-sized pools log the size (pricing
  it inside `plan_memory` is the open follow-up - a post-sizing deflation
  broke the admission guarantee and was reverted).
- Serving decode variance is one-sided: the dense arm spans 150-174 tok/s
  across fresh servers, the sparse arm holds 194-198.
- KV capacity, not the selection, binds stream count at long context:
  73.7 KB/token (fp8, this model) means 3 x 25k+gen is what ~5000 blocks
  hold; a 32-stream x 16k experiment does not fit this card with this model.

Per-forward batched metadata update (one launch per prefill chunk / decode
step, ragged mapping via `seq_offsets`) replaced the per-(seq, chunk, layer)
inline launches while chasing the spill; it was not the mechanism, but it is
the cheaper shape and the ragged mapping is now unit-tested.

## Quality (NIAH, Qwen3-8B-Q8_0 fp8 KV, 16k ctx, 5 depths x 3 seeds)

| arm | pass | note |
|---|---:|---|
| dense (`fp8_ng`) | 15/15 | |
| sparse budget 4096 | 12/15 | 3/3 repeat rounds fail the IDENTICAL 3 cells; all 3 retrieve the needle VERBATIM at `--max-tokens 768` - the harness's 384-token cap is think-budget exhaustion (Qwen3 shares think+answer budget), not a retrieval miss |
| sparse budget 2048 | 15/15 | 8x page sparsity |

`speculative.ngram=false` in every arm: prompt-lookup would draft the answer
straight from the needle and verify it with FULL attention, masking a broken
selection.

32k follow-up (2026-08-28, after `imp-cli --prompt-file` unblocked long
prompts): dense, budget-4096 (8x sparsity) and budget-2048 (16x) all 15/15 at
`--max-gen-tokens 768` (the budget that separates retrieval failure from
think-budget exhaustion).

## Measurement plan (done, results above)

- Identity: budget >= n_blocks output bit-identical vs dense (unit + e2e).
- Quality: `tools/eval/niah/niah_bench.py` at 16k, budgets 4096/2048 vs dense.
- Perf: decode A/B at 2k/8k/16k/32k, alternating arms, `make build` image.

## ROADMAP CLOSED (2026-08-30, recorded 2026-09-04)

SHIPPED opt-in (`attention.sparse_topk_tokens`). Standing evidence: Qwen3-8B
32k 160.3 -> 199.5 tok/s (+24.5%), verify chunks on the sparse table +28.2% at
32k, concurrent 3 x 25k 155.6 -> 197.7 (+27%, #1808); NVFP4-KV arm 77k 74.3 ->
100.2 (#1818); block-size fix #1819 (`sparse_topk_tokens` doubled on
`n_kv_heads <= 4`, configure 2N to keep an old budget). Retrieval price on
Qwen3.8-27B under the original min/max corner bound: NIAH 10/10 dense, 8/10 at
8192, 5/10 at 4096, which is why it stayed opt-in.

## Page score: the corner bound was the retrieval price (2026-09-12)

`max(q*min, q*max)` is identically `q*(min+max)/2 + |q|*(max-min)/2`, so the
bound's width per dimension is set by whichever single token in the page is most
extreme there. Replacing the stored pair with the page's own mean and standard
deviation (arXiv 2605.27740) keeps the layout, the loads and the arithmetic
count, and changes only what the metadata pass writes: one Welford pass plus
Chan's merge, with `slot` as the count already covered.

One image, one model (Qwen3.8-27B-NVFP4-vllm, NVFP4 KV, `kv_cache.growable=false`),
`niah_check.py` at 5 depths x 81 908 / 126 908 prompt tokens, `sparse decode
attention ACTIVE` asserted in every arm:

| budget | min/max corner | mean + std |
|---|---|---|
| 4096 | 2/10 | 10/10 |
| 8192 | 7/10 | 10/10 |

Wall time on a 77k-token prompt at 128 emitted tokens, arms alternating over two
rounds: corner 11.54 / 11.70 / 11.84 / 11.67 s, mean+std 11.53 / 11.54 / 11.56 /
11.69 s. Neutral, so the default is now mean+std and
`attention.sparse_score_meanstd=false` restores the corner bound.

**The first run of this A/B measured nothing**: both arms read 10/10 because
`kv_cache.growable` defaults true and `enable_key_minmax()` refuses a growable
pool, so neither arm was sparse. The startup line
(`Sparse decode attention: ... score ...`) proves the scratch, not the
selection; only `sparse decode attention ACTIVE` proves the selection. The A/B
script treats a zero count as an abort.

Follow-ups live in `docs/roadmap.md` Open 3, not here: MLA models, prefill
sparsity, and StreamingLLM eviction as the only answer under KV-pool pressure.

## Budget floor under the mean+std score (2026-09-12)

Same harness and model as the score A/B above, `tools/analysis/sparse_score_niah_ab.sh` and a
single-arm sweep:

| budget | min/max corner | mean + std |
|---|---|---|
| 1024 | 0/10 | 10/10 |
| 2048 | 1/10 | 10/10 |
| 3072 | - | 10/10 |
| 4096 | 2/10 | 10/10 |
| 8192 | 7/10 | 10/10 |

NIAH saturates under the new score at every budget down to 1024, so it can no longer discriminate
between them, and a selection refinement aimed at small budgets (an uncertainty gate that widens
the kept set on near-tied cuts, arXiv 2607.07724) has no target on this workload.

The budget is also not a speed lever here. Decode throughput, prefill cancelled by measuring the
slope between two generation lengths on the same 77k-token prompt, speculation off, three rounds:

| arm | decode tok/s |
|---|---|
| dense | 82.0 / 82.9 / 82.7 |
| sparse 1024 | 91.1 / 88.7 / 95.5 |
| sparse 8192 | 96.2 / 87.4 / 89.9 |

**Corrected 2026-10-01** (dead-end re-check S3; the table above ran before #2364, when the NVFP4
metadata pool of the sparse arms was unpriced): same shape, image c43392b8, growable pool, pool probe
1493-1608 GB/s resident in all 9 boots, `sparse decode attention ACTIVE` in every sparse arm, batch 4,
harness `tools/analysis/sparse_mla_ab.sh`, log `~/imp-e2-s3.log`:

| arm | decode tok/s |
|---|---|
| dense | 68.63 / 68.54 / 68.70 |
| sparse 1024 | 87.72 / 88.75 / 87.25 |
| sparse 8192 | 84.11 / 84.30 / 84.16 |

Sparse 1024 is +28 % over dense (median 87.72 vs 68.63), and 1024 beats 8192 by 3.56 tok/s against a
largest arm spread of 1.50: the budget is a small speed lever (+4.2 %). Rules fixed before the run.
State before: "Sparse is worth about +9% over dense on this shape and 1024 against 8192 is inside the
spread: only 16 of this model's 64 layers are attention, so the pages read are a small share of a
decode step. Configure the budget for retrieval headroom, not for speed."

**Two harness traps this cost.** A single wall-clock reading at a 77k prompt prices nothing: prefill
is ~80% of it and every budget read ~13 s. And the slope must use the tokens actually emitted, not
the requested `max_tokens` - assuming the request value made an 8192 budget read faster than a 1024
one. With the embedded MTP head on (the default for a single stream) the same arm spread 111 to 268
tok/s between rounds, so speculation has to be off to price an attention-side knob.

## Sparse prefill (2026-10-01)

SHIPPED opt-in: `attention.sparse_prefill_topk_tokens` (0 = off), `sparse_prefill_rows` (16),
`sparse_prefill_recent_tokens` (1024). Trigger: FMHA was 45.6 % of the Qwen3.8-27B pp77824 kernel
window (bound 1.52x at 4x cheaper attention).

| step | shape |
|---|---|
| select | once per continuation chunk per layer: score the past blocks for `rows` evenly spaced query rows (last row included), merge as max over rows of `score_r(b) - max_b score_r(b)`, top-k with sink + recent forced (same kernels as decode) |
| attend | gather only the selected past pages (compacted, ascending), append the chunk, one FA2 pass with `q_offset` = selected past tokens; keys are post-RoPE, so compaction changes nothing else |
| identity | past blocks <= budget: the dense table and `q_offset` (no launch) |
| not this | the QSA dead end (dense FA2 chunk, then recompute rows): no dense pass |

| budget | pp77824 tok/s | vs dense |
|---|---:|---:|
| dense | 6857.16 | - |
| 4096 | 11311.44 | 1.65x |
| 8192 | 10539.24 | 1.54x |
| 16384 | 9347.64 | 1.36x |

Shipped measurement at 8192 (1.54x clears the 1.26x gate with headroom for retrieval and PPL),
3 alternating rounds, fresh process per run:

| arm | pp77824 tok/s | FMHA ms (nsys, window) | selection launches |
|---|---|---:|---:|
| dense | 6835.55 / 6838.01 / 6828.79 | 5220.4 (45.5 %) | 0 |
| sparse 8192 | 10529.05 / 10537.84 / 10524.38 | 1203.1 (16.4 %) | 528 per kernel (16 layers x 33 chunks); score 52.5 ms, select 9.1, merge 8.7 |

| quality | dense | sparse 8192 |
|---|---|---|
| NIAH 5 depths x 81908 / 126908 tokens (`niah_check.py`) | 10/10 | 10/10 |
| PPL `ppl_corpus_45k.txt` (13811 tokens), `runtime.deterministic=true` | 4.5842 (2 runs) | 4.5962 (+0.26 %) |

Activity proof: `sparse prefill attention ACTIVE` in every sparse arm (prefill is not graph-captured,
so the host line fires), and the nsys kernel counts above.

```
[PROV: commit=afd9b9c4 date=2026-10-01 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4 (NVFP4 KV)
       cuda=13.4.1 image=scripts/build_image.sh of the branch n=3 alternating rounds
       cmd=`imp-cli --bench --bench-pp 77824 --bench-reps 1 --max-tokens 8
       --set speculative.ngram=false --set speculative.mtp_k=0 [--set attention.sparse_prefill_topk_tokens=8192]`]
```

## Prefill attention share (2026-10-01)

nsys, `imp-cli --bench --bench-pp N --bench-reps 1 --max-tokens 8`, mtp and n-gram off, graphs traced per node, window = NVTX `bench:pp`, attention = `fmha_sm120_fa2*` launches; bound = 1 / (1 - share x 0.75), the speedup if attention got 4x cheaper. Harness `tools/analysis/prefill_attn_share.sh`, image `86b079f2`.

| model | pp | kernels ms | attention ms (launches) | share | bound |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Q8_0 | 32768 | 3294.3 | 1283.1 (576) | 38.9 % | 1.41x |
| Qwen3-8B Q8_0 | 77824 | 21696.7 | 16742.5 (1368) | 77.2 % | 2.37x, suspect: 12.2 vs 2.2 ms per launch for 2.4x the mean KV length, not explained |
| Qwen3.8-27B NVFP4 | 32768 | 3495.3 | 944.3 (512) | 27.0 % | 1.25x |
| Qwen3.8-27B NVFP4 | 77824 | 11339.8 | 5176.4 (1216) | 45.6 % | 1.52x |

Decision rule fixed before the run: build prefill sparsity only if the bound is >= 1.15x on Qwen3.8-27B at 77k. It is 1.52x: the build is queued as its own unit. Earlier record: query-side QSA prefill measured slower (pp4503 650 vs 912 tok/s, CHANGELOG).

## MLA (2026-10-01, #2372)

The metadata gate refused every MLA model; the materialized decode reads the paged keys like any other model, only `attention.mla_absorb` bypasses them and stays refused. DeepSeek-V2-Lite NVFP4 (`imp-quantize`), F16 KV, `paged_fp16`, imp-server per arm, mtp, n-gram and prefix cache off; decode by slope (320 vs 64 `ignore_eos` tokens), 3 rounds; harness `tools/analysis/sparse_mla_ab.sh`.

| ctx | dense tok/s | sparse 4096 tok/s | ratio |
|---|---|---|---:|
| 16000 | 6.44 / 6.47 / 6.50 | 22.99 / 23.15 / 22.91 | 3.55x |
| 32000 | 3.18 / 3.20 / 3.20 | 22.75 / 23.03 / 23.00 | 7.18x |

NIAH, 28k and 30k x 5 depths, DeepSeek-V2-Lite-Chat NVFP4 (the base model answers `<jupyter_code>` to the chat-framed probe in every arm): dense 10/10, sparse 1024 10/10, sparse 4096 10/10, one `sparse decode attention ACTIVE` line per sparse arm. Dense MLA decode itself is the slow part (48x under the KV-bandwidth ceiling at 32k): #2374.

### MLA after the dense decode fix (2026-10-01, #2374)

The dense MLA numbers above ran on `paged_attention_decode_kernel_generic` (98.9 % of decode, 28 GB/s). With HD 192 split-K plus compaction (#2374), same harness and model, 3 rounds:

| ctx | dense tok/s | sparse 4096 tok/s | ratio |
|---|---|---|---:|
| 16000 | 182.45 / 180.49 / 181.81 | 247.99 / 249.95 / 253.77 | 1.38x |
| 32000 | 111.22 / 112.38 / 111.35 | 259.95 / 235.40 / 244.98 | 2.21x |

Sparse still wins on MLA; the 3.55x / 7.18x of the first table measured the slow dense kernel. NIAH result unchanged.

## Default on per arch family (2026-10-02, #2405, #2406)

`attention.sparse_topk_tokens` and `attention.sparse_prefill_topk_tokens` default to -1 (auto):
`sparse_decode_default_tokens` / `sparse_prefill_default_tokens` (`src/model/model.cpp`) give the
budget per `ModelArch`, 0 elsewhere; `init_resolve_sparse_attention_` logs the resolved value.
Explicit 0 is the opt-out.

Gates, fixed before the run: NIAH (`niah_check.py`, 5 depths x 16k/32k/64k where the model
context allows, 4096 answer tokens on the reasoning models) sparse >= dense; PPL
(`imp-cli --perplexity`, `runtime.deterministic=true`) <= +0.5 % (one-sided, replaced by the
two-sided gate in #2529 below) on `ppl_corpus_45k.txt`
(13.5k tokens) and on the first 110000 bytes of `calib_corpus.txt` (25.5k tokens); tg at
pp32512 and pp32512 itself >= dense, median of 3 alternating rounds, fresh process per run,
n-gram and MTP off. Decode sparsity has no PPL arm: teacher-forced PPL runs the prefill path
(`--prefill-chunk-size 1` ran FMHA, 0 `sparse decode attention ACTIVE` lines).
Every sparse arm logged its `ACTIVE` line except gpt-oss prefill and Gemma-4 (see decision); no dense arm logged one.

| model | NIAH dense / dec / pre / both | PPL 45k dense -> pre | PPL long dense -> pre | tg32k dense -> dec | pp32512 dense -> pre |
|---|---|---|---|---|---|
| Qwen3-4B Q8_0 | 15/15 / 15/15 / 15/15 / 15/15 | 11.5332 -> 11.5584 (+0.22 %) | 10.9551 -> 10.9592 (+0.04 %) | 194.42 -> 284.51 (+46.3 %) | 13876.37 -> 18998.94 (+36.9 %) |
| Qwen3-8B Q8_0 | 10/10 / 10/10 / 10/10 / 10/10 | 10.7522 -> 10.7853 (+0.31 %) | 9.1309 -> 9.1471 (+0.18 %) | 156.86 -> 207.85 (+32.5 %) | 10176.48 -> 12641.83 (+24.2 %) |
| Qwen3-14B Q6_K | 10/10 / 10/10 / 10/10 / 10/10 | 9.2035 -> 9.2263 (+0.25 %) | 6.1084 -> 6.1333 (+0.41 %) | 86.19 -> 134.09 (+55.6 %) | 6636.09 -> 7956.77 (+19.9 %) |
| Qwen3-Coder-30B-A3B NVFP4 | 15/15 / 15/15 / 15/15 / 15/15 | 9.8901 -> 9.8930 (+0.03 %) | 2.7671 -> 2.7649 (-0.08 %) | 213.63 -> 248.46 (+16.3 %) | 15009.60 -> 23087.63 (+53.8 %) |
| Qwen3.8-27B NVFP4 | 15/15 / 15/15 / 15/15 / 15/15 | 4.5842 -> 4.5962 (+0.26 %) | 3.8139 -> 3.8214 (+0.20 %) | 81.42 -> 88.04 (+8.1 %) | 9438.01 -> 10881.41 (+15.3 %) |
| Qwen3.6-35B-A3B NVFP4 (pre 8192) | 15/15 / 15/15 / 15/15 / 15/15 | 6.7750 -> 6.8202 (+0.67 %) | 1.5529 -> 1.5620 (+0.59 %) | 284.02 -> 299.28 (+5.4 %) | 24062.83 -> 28661.85 (+19.1 %) |
| Nemotron-3-Nano-30B-A3B NVFP4 | 15/15 / 15/15 / 15/15 / 15/15 | 9.3707 -> 9.3749 (+0.04 %) | 7.9624 -> 7.8591 (-1.30 %) | 349.52 -> 383.61 (+9.8 %) | 35199.74 -> 38966.20 (+10.7 %) |
| Phi-4-reasoning-plus NVFP4 | 6/10 / 10/10 / 6/10 / 8/10 | 12.9442 -> 12.9616 (+0.13 %) | 3.8711 -> 3.8672 (-0.10 %) | 110.13 -> 137.23 (+24.6 %) | 12538.86 -> 17545.39 (+39.9 %) |
| gpt-oss-20b MXFP4 | 15/15 / 15/15 / 15/15 / 15/15 | 262.6413 -> 262.6413 (+0.00 %) | 7065.3958 -> 7065.3958 (+0.00 %) | 270.97 -> 330.60 (+22.0 %) | 25553.11 -> 25474.12 (-0.3 %) |
| Qwen3.6-35B-A3B NVFP4 (pre 16384) | 15/15 / 15/15 / 15/15 / 15/15 | 6.7750 -> 6.7750 (identity, 13.5k < budget) | 1.5529 -> 1.5537 (+0.05 %) | 285.23 -> 299.97 (+5.2 %) | 24142.12 -> 25863.03 (+7.1 %) |

| decision | decode (4096) | prefill |
|---|---|---|
| on | qwen3, qwen3moe, qwen35, qwen36moe, nemotron_h_moe, llama (Phi-4), gpt_oss | superseded by the two-sided gate below (#2529): qwen35 8192; qwen3, qwen36moe 16384; qwen3moe, llama 24576 |
| off | gemma4: per-layer KV geometry refuses the metadata pool | qwen36moe at 8192 (PPL +0.67 %); nemotron_h_moe (#2529); gpt_oss (learned sinks, prefill selection never engages: pp -0.3 %); gemma4 |

Phi-4 dense fails 4 of the 5 32k cells at 4096 answer tokens (reasoning without an answer); the
decode arm passes them. Qwen3.6-35B-A3B NIAH ran with `runtime.max_batch_size=1` (the auto batch
of 26 recurrent slots left 1625 KV blocks, 32k/64k probes 503 in every arm). Gemma-4 dense NIAH is
0/5 at 16k (#2519); Llama-3.2-3B is dense-broken past 16k (#2520), so it cannot gate the llama family.

### Two-sided prefill gate (2026-10-02, #2529)

The +0.5 % bound only failed increases; Llama-3.2-3B (dense fixed by #2528) read -3.27 % at 8192.
Gate now: |PPL delta| <= 0.5 % vs dense on windows [C/2, C-2] for C = 16384 and 32768
(`diagnostics.ppl_first/ppl_last`), corpus = first 160000 bytes of the #2520 `long.txt`
(38.2k to 40.0k tokens), `runtime.deterministic=true`, n-gram and MTP off. A changed budget also
re-ran NIAH (sparse >= dense) and pp32512 (median of 3 alternating, >= dense) with decode sparse off
in both arms. Budget >= context is identity: every 16k row of 16384 and 24576 equals dense.

| model | family | budget | PPL 16k dense -> sparse | PPL 32k dense -> sparse | pass |
|---|---|---|---|---|---|
| Llama-3.2-3B Q8_0 | llama | 8192 | 6.4539 -> 6.3332 (-1.87 %) | 13.4171 -> 12.9788 (-3.27 %) | no |
| | | 16384 | identity | 13.4171 -> 13.2550 (-1.21 %) | no |
| | | 24576 | identity | 13.4171 -> 13.4002 (-0.13 %) | yes |
| Phi-4-reasoning-plus NVFP4 | llama | 8192 | 2.5117 -> 2.5151 (+0.14 %) | 2.8675 -> 2.8244 (-1.50 %) | no |
| | | 16384 | identity | 2.8675 -> 2.8334 (-1.19 %) | no |
| | | 24576 | identity | 2.8675 -> 2.8632 (-0.15 %) | yes |
| Qwen3-4B Q8_0 | qwen3 | 8192 | 9.8863 -> 9.8896 (+0.03 %) | 14.3093 -> 14.3095 (+0.00 %) | yes |
| | | 16384 | identity | 14.3093 -> 14.2855 (-0.17 %) | yes |
| Qwen3-8B Q8_0 | qwen3 | 8192 | 7.4424 -> 7.4686 (+0.35 %) | 11.3594 -> 11.4389 (+0.70 %) | no |
| | | 16384 | identity | 11.3594 -> 11.3659 (+0.06 %) | yes |
| Qwen3-14B Q6_K | qwen3 | 8192 | 4.2820 -> 4.2977 (+0.37 %) | 6.0408 -> 6.0319 (-0.15 %) | yes |
| | | 16384 | identity | 6.0408 -> 6.0235 (-0.29 %) | yes |
| Qwen3-Coder-30B-A3B NVFP4 | qwen3moe | 8192 | 1.7787 -> 1.7712 (-0.42 %) | 2.1710 -> 2.1320 (-1.80 %) | no |
| | | 16384 | identity | 2.1710 -> 2.1348 (-1.67 %) | no |
| | | 24576 | identity | 2.1710 -> 2.1698 (-0.06 %) | yes |
| Qwen3.8-27B NVFP4 | qwen35 | 8192 | 2.6295 -> 2.6399 (+0.40 %) | 3.5418 -> 3.5513 (+0.27 %) | yes |
| Qwen3.6-35B-A3B NVFP4 | qwen36moe | 16384 | identity | 1.1592 -> 1.1604 (+0.10 %) | yes |
| Nemotron-3-Nano-30B-A3B NVFP4 | nemotron_h_moe | 8192 | 6.1427 -> 6.0631 (-1.30 %) | 8.2164 -> 8.0174 (-2.42 %) | no |
| | | 16384 | identity | 8.2164 -> 8.1404 (-0.92 %) | no |
| | | 24576 | not run | 8.2164 -> 8.1718 (-0.54 %) | no |
| gpt-oss-20b MXFP4 | gpt_oss | 0 (never engages) | 16979.2805 both | 12832.8206 both | identity |

gpt-oss reads ~1.7e4 on this prose in llama.cpp too (`llama-perplexity -c 16384`: 16377.62).

Changed budgets, NIAH dense / sparse and pp32512 dense -> sparse (tok/s):

| model | budget | NIAH | pp32512 |
|---|---|---|---|
| Qwen3-4B Q8_0 | 16384 | 15/15 / 15/15 | 13829.27 -> 15499.81 (+12.1 %) |
| Qwen3-8B Q8_0 | 16384 | 10/10 / 10/10 | 10100.37 -> 10970.27 (+8.6 %) |
| Qwen3-14B Q6_K | 16384 | 10/10 / 10/10 | 6640.71 -> 7112.65 (+7.1 %) |
| Qwen3-Coder-30B-A3B NVFP4 | 24576 | 15/15 / 15/15 | 14928.13 -> 15270.51 (+2.3 %) |
| Llama-3.2-3B Q8_0 | 24576 | 10/10 / 10/10 | 19999.44 -> 20311.74 (+1.6 %) |
| Phi-4-reasoning-plus NVFP4 | 24576 | 6/10 / 7/10 | 12457.52 -> 12740.69 (+2.3 %) |

Why sparse reads lower (Llama-3.2-3B, budget 8192, per-position NLL dumps,
`diagnostics.ppl_dump=full`): the corpus switches from Pride and Prejudice to Shakespeare at
row 27701. On the 5066 rows after the switch (31 % of the 32k window) sparse carries 59.2 % of the
window's NLL drop. Same rows: dense 26.0767, sparse 24.4503 (-6.24 %), dense with the unrelated
prefix cut away 21.4799 (-17.63 %). Dense is HF-exact (#2520), so the drop is the model being
distracted by long unrelated context; page selection removes part of it. Before the switch
(rows [16384, 27700]) sparse is -1.94 %; dropping the oldest 11540 tokens moves dense only
-0.07 % (7396 tokens: +5.52 %), so plain truncation does not explain that part.

Qwen3.8-27B pp77824 (#2406 acceptance), 3 alternating pairs: dense 6871.80 / 6873.68 / 6873.02,
sparse 8192 10670.21 / 10670.54 / 10671.67 tok/s (1.55x).

KV ceiling cost of the metadata pool (growable pool ceiling, server at `runtime.max_seq_len`
81920): Qwen3-4B 8172 -> 7691 blocks (-5.9 %), Qwen3-8B 5809 -> 5467 (-5.9 %), Qwen3-Coder-30B
8712 -> 7744 (-11.1 %), Phi-4 9473 -> 8420 (-11.1 %), gpt-oss-20b 27343 -> 25734 (-5.9 %);
Qwen3.8-27B, Nemotron-3-Nano, Qwen3-14B unchanged (ceiling bound by `max_seq_len`).

`copy_blocks_device` now copies the key min/max metadata, so the hybrid transcript snapshot
(`server.transcript_snapshot`, default on) stays on with the pool; before, the pool turned it off.

```
[PROV: commit=7858db2d date=2026-10-02 hw=RTX5090 cuda=13.4.1 image=scripts/build_image.sh of main
       n=3 alternating rounds, fresh process per run
       cmd=`imp-cli --bench --bench-pp 32512 --bench-reps 1 --max-tokens 128 --max-seq-len 32768
       --set speculative.ngram=false --set speculative.mtp_k=0
       [--set attention.sparse_topk_tokens=4096 | --set attention.sparse_prefill_topk_tokens=8192]`]
```
