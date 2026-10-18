# Cold KV pages in host RAM: decision measurement

Roadmap row 112, sub-item 112.1, 2026-10-18. Question: what share of a sparse decode step (`attention.sparse_topk_tokens=4096`) would fetching the selected pages that are not resident in VRAM cost, if cold pages of live sessions sit in mapped pinned host RAM?

Decision: NO-GO for 112.2-112.7. The page selection replaces 43 % of its pages every step, so a VRAM ring that keeps 25-50 % of the context resident still pulls 24.9-75.2 MB per token over PCIe.

## Setup

| | |
|---|---|
| Models | Qwen3-8B Q8_0, Qwen3-4B-Instruct-2507 Q8_0 (262k native context, same KV geometry: 36 layers, 8 KV heads, hd 128) |
| KV | FP8 E4M3, 16-token blocks: 16 KiB K + 16 KiB V = 32 KiB per block and layer |
| Text | `docs/*.md` + `docs/internals/*.md` concatenated (464802 bytes), `imp-cli --bench --bench-prompt-file --bench-teacher-force`, 256 decode tokens |
| Selection dump | scratch patch after `sparse_select_blocks` in `src/exec/executor_attention_decode.cpp`: sync, copy `sparse_block_tables` row 0 + `sparse_context_lens`, write per layer and step; `--no-cuda-graphs`, never committed |
| Ring model | per layer an LRU of R blocks over the selected ids; a miss is a selected block not in the ring; every first sight counts (physical ids are not logical: the last step's tail ids are 578, 577, 576, 575 at 2064 context blocks) |
| Fetch cost | `tools/analysis/h2d_gather_probe.cu` mode C (one gather kernel over mapped pinned memory, 32 blocks per copy), 4 GiB pinned source, distinct destination slots |

## Selection churn

| Run | Blocks selected per layer | New vs previous step, mean / p50 / p99 | Lowest-churn layer | Highest |
|---|---:|---|---|---|
| Qwen3-8B, 32k | 256 | 110.28 / 107 / 199 | L0 47.9 | L3 148.9 |
| Qwen3-4B-2507, 32k | 256 | 110.71 / 108 / 200 | L0 68.7 | L7 146.5 |
| Qwen3-4B-2507, 64k | 256 | 116.03 / 113 / 213 | L0 37.2 | L7 165.7 |

## Ring misses per decode step (sum over 36 layers)

| Run | Ring per layer | Resident share of the context | Misses mean / p99 | Bytes per step |
|---|---|---:|---|---:|
| Qwen3-8B, 32k | 256 (1x budget) | 12.4 % | 6139.1 / 7646 | 201.2 MB |
| Qwen3-8B, 32k | 512 (2x) | 24.8 % | 2293.7 / 4218 | 75.2 MB |
| Qwen3-8B, 32k | 1024 (4x) | 49.6 % | 760.6 / 2985 | 24.9 MB |
| Qwen3-4B-2507, 64k | 512 (2x) | 12.5 % | 2705.4 / 4539 | 88.7 MB |
| Qwen3-4B-2507, 64k | 1024 (4x) | 24.9 % | 1385.8 / 3196 | 45.4 MB |

Cold start (first decode step after the prefill): 9216 blocks = 256 per layer in every run.

## Gather cost

| Slots of 32 KiB | Mode C total | GB/s |
|---:|---:|---:|
| 21 | 0.03 ms | 19.9-21.5 |
| 64 | 0.06-0.08 ms | 26.4-36.6 |
| 170 | 0.14 ms | 38.7-39.0 |
| 512 | 0.34-0.35 ms | 47.6-49.7 |

`cudaMemcpyBatchAsync` (mode B) reaches 39.1 GB/s at 512 slots; per-slot `cudaMemcpyAsync` (mode A) 2.7-6.3 GB/s.

## Share of the decode step

Step times (graphs on, same patched image, the dump inert without `IMP_SPARSE_DUMP`), tg 128 x 2 reps:

| Run | Sparse 4096 | Dense |
|---|---:|---:|
| Qwen3-8B, 32k | 4.820 ms | 5.979 ms |
| Qwen3-4B-2507, 32k | 3.533 ms | 4.732 ms |
| Qwen3-4B-2507, 64k | 3.884 ms | 7.114 ms |

| Run, ring | Floor: bytes / 49.7 GB/s | Share | Per layer serialized: probe time x 36 | Share |
|---|---:|---:|---:|---:|
| Qwen3-8B 32k, 1x | 4.048 ms | 84.0 % | 5.04 ms | 104.6 % |
| Qwen3-8B 32k, 2x | 1.512 ms | 31.4 % | 2.16-2.88 ms | 44.8-59.7 % |
| Qwen3-8B 32k, 4x | 0.501 ms | 10.4 % | 1.08 ms | 22.4 % |
| Qwen3-4B 64k, 2x | 1.784 ms | 45.9 % | | |
| Qwen3-4B 64k, 4x | 0.914 ms | 23.5 % | | |

## Why NO-GO

| Point | Number |
|---|---|
| Half the KV out of VRAM (32k, ring 4x) | +10.4 % (bandwidth floor) to +22.4 % (per-layer gather on the critical path) per token, one session |
| Same saving without PCIe | `--kv-nvfp4` halves FP8 KV; sparse decode takes NVFP4 caches (`src/exec/executor_attention_decode.cpp:112 F16/FP8/NVFP4 caches`) |
| Three quarters out (32k ring 2x, 64k ring 4x) | +23.5 % to +31.4 % at the floor, one session |
| Concurrency | PCIe bytes scale with sessions, the weight read per step does not: the target case (more sessions than VRAM) is the worst case |
| Metadata stays resident | key min/max pool = 12.5 % of the K+V pool (`engine_kv_cache_init.cpp` log), needed to score cold pages |

## Reopen condition

A selection that keeps its pages across steps: ring 2x misses fall to about 21 blocks per layer per step (the 32k ring 4x point, 10.4 % floor).

That changes which pages attention reads, so it needs a quality A/B first (`tools/analysis/sparse_score_niah_ab.sh`).
