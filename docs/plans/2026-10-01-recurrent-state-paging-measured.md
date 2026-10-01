# Recurrent-state paging: what binds admission at 32 streams x long context

Roadmap row 5, closed 2026-10-01. Question: at 32 streams x 8k-30k tokens on Qwen3.8-27B, does the
recurrent-state slab or the KV pool bind admission? Paging the slabs pays only if the slab binds.

## Setup

| | |
|---|---|
| server | `imp-server --model /models/Qwen3.8-27B-NVFP4-vllm --set runtime.max_batch_size=<32\|8> --set runtime.max_seq_len=32768`, stock otherwise |
| client | `tools/analysis/longctx_conc_client.py` with `stream=True` (the per-request `prompt + completion ... queue=` line logs on the streaming path only), 32 concurrent, `max_tokens` 64, temperature 0, `TARGET_CHARS` 36000 / 71000 / 131000 |
| arms | bs=32 (32 slabs) and the control bs=8 (8 slabs): the KV the freed slabs buy is the most paging could buy |
| peak active | max of `imp_queue_running`, `/metrics` sampled every 0.5 s |
| KV-queued | requests with logged `queue=` > 2000 ms |
| slab refusals | count of `cannot be admitted - the recurrent state slab` (`src/runtime/scheduler.cpp:84`); slab holds = count of `SSM state: slot N not committed` |

## Result

| arm | L | prompt tokens | ssm MiB | KV blocks (plan / ceiling) | peak active | admitted at t0 | slab refusals | slab holds | KV-queued | max queue ms |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| bs=32 | 8k | 8090-8533 | 2544 | 2048 / 12425 | 30 | 24 | 0 | 0 | 8 | 14993.3 |
| bs=32 | 16k | 15988-16498 | 2544 | 2048 / 12425 | 19 | - | 0 | 0 | 20 | 37702.5 |
| bs=32 | 30k | 29901-30235 | 2544 | 2048 / 12425 | 12 | 6 | 0 | 0 | 26 | 77663.7 |
| bs=8 | 8k | 8089-8532 | 636 | 5753 / 16384 | 8 | - | 0 | 0 | 24 (batch cap) | 25908.4 |
| bs=8 | 16k | 15987-16497 | 636 | 5753 / 16384 | 8 | - | 0 | 0 | 24 (batch cap) | 51669.3 |
| bs=8 | 30k | 29900-30234 | 636 | 5753 / 16384 | 8 | - | 0 | 0 | 24 (batch cap; 4 samples at running 6-7 with waiting > 0) | 97656.9 |

Demand at 32 x L (sum of logged prompt + completion, 16-token blocks): 8k 266535 tokens = 16674 blocks,
16k 523142 = 32711, 30k 961682 = 60124. Ceiling: 12425 blocks = 198800 tokens at 32 slabs,
16384 blocks = 262144 tokens at 8 slabs. 24 fewer slabs free 1908 MiB and buy 3959 blocks = 63344
tokens; demand exceeds the bs=8 ceiling too, at every length (16674 > 16384 at 8k).

bs=32 peak active above "admitted at t0" (30 vs 24 at 8k, 12 vs 6 at 30k) comes after the
StreamingLLM valve armed (`KV cache >90% full ... auto-enabling StreamingLLM (sinks=4, window=4096)`,
06:52:56, once): KV pressure, not the slab.

## Decision

CLOSED, DO NOT BUILD: 0 slab refusals and 0 slab holds at every length, KV-queued 8 / 20 / 26 at
8k / 16k / 30k. KV binds first; paging all but 8 slabs would add 63344 tokens against a 266535-961682
token demand.

## Side finding

bs=8: 8 sequences cancelled at decode (`KV pool exhausted at decode`, 2 in the 16k wave, 6 in the 30k
wave), StreamingLLM never armed: #2361.

Both arms logged `KV cache: pool copy bandwidth 62 GB/s` (bs=32) / `304 GB/s` (bs=8), below the
500 GB/s spill threshold: throughput and queue ms are not comparable to a clean card, admission
counts are block arithmetic and unaffected.

[PROV: commit=3a8a5bd9 date=2026-10-01 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4 cuda=13.4.1
       image=imp:test-imp-r5-4aeeca55 (make build of 3a8a5bd9) n=1 wave of 32 per arm x length
       cmd=`run_arm.sh 32` then `run_arm.sh 8`, 06:52-06:59 UTC; logs ~/.cache/imp/roadmap/R5/bs{32,8}/server.log,
       sampler_{8k,16k,30k}.txt, client_{8k,16k,30k}.txt (ok=32 err=0 every wave)]
