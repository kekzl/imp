# Allocator headroom from a measurement

Roadmap row 91, closed 2026-10-04 (#2482). `kAllocatorHeadroomPct` (`src/memory/vram_query.h`) was 5 % of total, never measured. Question: how much device memory do allocations take after the growable KV pool hit its cap, and what does a smaller headroom buy?

## Setup

| | |
|---|---|
| server | `imp-server --model /models/Qwen3.8-27B-NVFP4-vllm --max-concurrent 32`, stock otherwise |
| load | `tools/analysis/longctx_conc_client.py`, 32 streams, 256 tokens out, 10 min of waves; 62000 chars per prompt (~15.5k tokens), or `MIXED=1`: 4k..60k chars drawn per wave |
| samples | `nvidia-smi` memory.used and `imp_kv_blocks_total` every second |
| growth | device used at the first sample with the pool at its run maximum vs the peak after it |

## Results

| arm | KV max blocks | device used at cap / peak after (MiB) | growth after cap | waves, wall per wave |
|---|---:|---|---:|---|
| 5 %, 62000 chars | 12827 (capped by headroom) | 28875 / 28875 (cap in the last 32 s) | 0 MiB | 10, 58.56..62.04 s |
| 2 %, 62000 chars | 13599 (ceiling) | 29047 / 29091 | 44 MiB | 11, 55.48..56.86 s |
| 2 %, mixed | 13599 | 29410 / 30018 | **608 MiB** | 18 |
| 4 %, mixed | 13599 | 29994 / 30018 | 24 MiB | 20 |

[PROV: commit=6d63045f date=2026-10-04 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm cuda=13.4.2
       path=imp-server n=1-per-arm harness_md5=32d64ca3
       cmd=`tools/analysis/headroom_soak.sh <img> <tag> 10` (+ `MIXED=1`), arms = kAllocatorHeadroomPct 5 / 2 / 4
       note=no errors, OOM or cancelled request in any arm; device free never below 2589 MiB (the gap above the allocator's reading is uncommitted lazy-pool charges)]

## Decision

- 4 % = 1304 MiB on 32 GB: 696 MiB above the largest measured growth (608 MiB); frees 325 MiB, the pool needs 197 MiB to reach its 13599-block ceiling (+772 blocks, +6.0 %).
- 2 % leaves 44 MiB above the 608 MiB growth: refused.
- The issue's 5796-block figure was computed from the 1630 MiB alone; the planner's 3900 MiB reserve floor sets the ceiling, so 772 blocks is the reachable gain.
