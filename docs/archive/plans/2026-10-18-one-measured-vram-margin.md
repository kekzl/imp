# One measured VRAM margin

AUDIT item 24 (backlog B-29f), owner decision 2026-10-18: collapse the stacked reserves into one measured margin.

## Change

| Reserve | Before | After |
|---|---|---|
| Library reserve | measured per model (cache), 3900 MiB constant on a cold start | unchanged |
| Allocator headroom | 4 % = 1304 MiB on 32 GB, measured (#2482) | unchanged: the margin |
| Reserve floor (`vram.reserve_floor_pct`, `vram_reserve_floor`) | 10 % = 3260 MiB, absolute floor 256 MiB; pre-dequant phases always 10 % | default 0, floor = the 4 % headroom; the key stays as an opt-in |
| `vram.kv_fraction` | 0.8 of post-reserve VRAM, clamp [0.05, 0.95] | 1.0 (no cap), clamp [0.05, 1.0]; the key stays |
| 256 MiB workspace term in `planned_post_cache_reserve` | | kept: a workspace estimate in the mode-2 cache guard, not a policy reserve |

Where the floor acted: live-pass reserve (`vram_budget.cpp`, always below the 3900 MiB library charge on 32 GB), native-NVFP4 phase-3 reserve (floor + 1 GiB: 4281.9 -> 2327.2 MiB), GGUF planner-driven reserve, `pre_dequant_phase3_cutlass.cpp` `kCtReserve`, `pre_dequant_phase3_nvfp4_decode.cpp` safety, `executor_pre_dequant.cpp` minimum reserve.

## Follow-on fix: the ceiling raise skipped pools with key min/max

`apply_measured_library_reserve_` re-plans the KV pool once warmup has measured the library claim (#2436). `KVCache::raise_ceiling` refused every pool with a sparse-decode key min/max pool, so the computed ceiling was dropped without a log line.

Qwen3.8-27B-NVFP4 server, 32 streams: the replan computed 19063 blocks, the ceiling stayed at 8396. The growable min/max region is now rebuilt at the new ceiling (no live blocks at that point); `KVCacheGrowTest.RaiseCeilingKeepsTheCommitAndGrowsPastTheOldCeiling` asserts the move, the stride and the backed grown block; a mutant without the swap fails it.

## Single load (`imp-cli`, 16 tokens, `diagnostics.vram_audit`), 10 models

| Run | Models with identical max_seq_len, KV ceiling, weight caches, peak | Moved |
|---|---|---|
| batch 1, sampled | 9 of 10 | Qwen3.6-35B-A3B-NVFP4: KV ceiling 118128 -> 143184 tokens, peak 26662 -> 26525 MiB |
| batch 16, greedy | 9 of 10, greedy text identical on all 10 except Qwen3.8-27B | Qwen3.6-35B-A3B-NVFP4: KV ceiling 131072 -> 143184 tokens |

Qwen3.8-27B-NVFP4 greedy, 24 tokens, batch 16: base run 1 = base run 2 = B-29f, and base = B-29f under `IMP_DETERMINISTIC=1`; the sweep's differing text was run-to-run noise.

Models: Qwen3-8B Q8_0, Qwen3-14B NVFP4, Qwen3.8-27B NVFP4, gemma-3-12b Q4_K_M, Gemma-4-26B-A4B Q4_K_M, Qwen3-30B-A3B NVFP4, gpt-oss-20b MXFP4, Qwen3.6-35B-A3B NVFP4, Qwen3-8B FP8, Nemotron-3-Nano-30B NVFP4.

## Server soak (`tools/analysis/headroom_soak.sh`, `MIXED=1`, 10 min, Qwen3.8-27B-NVFP4, 32 streams)

| | Base (main) | Defaults only | Defaults + ceiling fix |
|---|---:|---:|---:|
| Ceiling after warmup | 22187 | 8396 | 19063 |
| Max KV blocks | 20768 | 8396 | 19063 |
| Peak device used | 29530 MiB | 25018 MiB | 28679 MiB |
| Growth after the pool maximum | 133 MiB | 1545 MiB | 14 MiB |
| Failed requests, I2 violations, server ERROR lines | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |

Base reached 20768 blocks through the live-pass fit backstop (`vram_budget.cpp`, "KV clamped"): when the weight-cache estimate (1213.4 + 5883.5 MiB) exceeds `available` (6240.2 MiB) it sizes on all of `available` and ignores the caches. With the smaller phase-3 reserve the caches fit, the backstop subtracts them, and the ceiling comes from the plan at the measured reserve (19063).

Open: at 19063 blocks the card still had 3928 MiB free at peak, 2624 MiB above the headroom; the plan charges that somewhere outside this item.
