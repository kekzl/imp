# Sparse decode attention: Quest-class top-k page selection

Status: in progress (2026-08-28). Roadmap Open item 2 (long context served by a
2023-era answer). Mechanism trigger per the BitDecoding shelf note: paged
attention is 19.9%/43.9% of the dense decode window at 8k/32k and 29.1%/50.6%
on MoE (ceiling 1.76-2.0x at 32k if attention were free).

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
| per step (dispatch time): plain decode only (`!chunk_decode_attn`, n == n_sequences) | verify chunks keep full attention; metadata is still maintained for their writes |

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

## Measurement plan

- Identity: budget >= n_blocks output bit-identical vs dense (unit + e2e).
- Quality: `tools/eval/niah/niah_bench.py` at 8k-32k, budget 4096, vs dense;
  degen suite.
- Perf: decode A/B at 8k/32k prefill on a dense model, alternating arms,
  `make build` image.
