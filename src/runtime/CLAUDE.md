<!--
layer: L3
audience: agents
verified: 2026-10-24
commit: 28185525
-->

# src/runtime - engine, scheduler, config, KV

Admission, continuous batching, KV allocation, CUDA-graph capture, config surface.

## Invariants

- `RuntimeConfig` (`config.h`) is the only config surface, one per engine, no process-global config. Env vars seeded in `config.cpp` only (list: root `CLAUDE.md`).
- Unknown `--set` key = error.
- Internal errors throw; `src/api/imp_api.cpp` translates to `ImpError`. No status returns.
- Never size from a live `cudaMemGetInfo` reading.

## Entry points

- `engine.cpp`: lifecycle, suspend/resume, top-level step
- `engine_scheduler.cpp`, `scheduler.cpp`: admission, continuous batching
- `engine_prefill.cpp`, `engine_prefill_ragged.cpp`: prefill; `engine_decode_pipeline.cpp`: pipelined batched decode
- `engine_kv_cache_init.cpp`: KV block geometry, pool sizing
- `engine_graph_decode.cpp`, `cuda_graph.cu`: capture and replay
- `config.h` / `config.cpp`: every key with default and rationale

## Test

CPU tests: `make dev-test`, new ones in `test-core`. KV, graphs, scheduling: `make verify-fast` before shipping (CI has no GPU).

## Pitfalls

- `runtime.prefill_graph` (default `true`) covers serial offset-0 prefill only, not ragged. Eager: legacy host-args MoE prefill, KV dtypes other than F16 / calibrated FP8 (`engine_prefill.cpp`). Other prefills replay only FFN/MoE from per-(layer, padded rows) graphs (`runtime.prefill_ffn_graph`, `exec/ffn_graph_cache.h`). Source of truth: `config.h`.
- `kv_cache.swa_snapshot_mb` below one snapshot disables prefix caching entirely (worse than 0).
- A/B on anything sized off free VRAM: pin `runtime.max_batch_size`.

## Do not touch

Generated config docs: edit the inline comments in `config.h`.

## See also

[`docs/internals/MEMORY.md`](../../docs/internals/MEMORY.md) (VRAM, ownership, lifetime), [`docs/DETERMINISM.md`](../../docs/DETERMINISM.md).
