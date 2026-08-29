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
