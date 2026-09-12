<!--
layer: L3
audience: agents
verified: 2026-09-13
commit: 81d22eff
-->

# src/runtime - engine, scheduler, config, KV

Request lifecycle: admission, continuous batching, KV allocation, CUDA-graph capture, and the config surface everything else reads.

## Invariants

- `RuntimeConfig` (`config.h`) is the only config surface: one per engine, no process-global config. Env vars are seeded in `config.cpp` only (list: root `CLAUDE.md`).
- A `--set` key that does not exist is an error, not a warning.
- Internal errors throw; `src/api/imp_api.cpp` translates them to `ImpError`. Do not convert them to status returns.
- Never size anything from a live `cudaMemGetInfo` reading (root `CLAUDE.md`, VRAM).

## Entry points

- `engine.cpp`: lifecycle, suspend/resume, the top-level step
- `engine_scheduler.cpp`, `scheduler.cpp`: admission, continuous batching
- `engine_prefill.cpp`, `engine_prefill_ragged.cpp`: prefill; `engine_decode_pipeline.cpp`: pipelined batched decode
- `engine_kv_cache_init.cpp`: KV block geometry, pool sizing
- `engine_graph_decode.cpp`, `cuda_graph.cu`: capture and replay
- `config.h` / `config.cpp`: every key, with default and rationale inline

## Test

CPU tests run in `make dev-test`; new ones go to `test-core`. Anything touching KV, graphs or scheduling: `make verify-fast` before it ships (CI has no GPU).

## Pitfalls

- `runtime.prefill_graph` defaults to `true`; the legacy host-args MoE prefill path and non-F16 KV append still run eager (`engine_prefill.cpp`). Read `config.h`, not old comments.
- `kv_cache.swa_snapshot_mb` below one snapshot size disables prefix caching entirely, which is worse than 0.
- Anything sized off free VRAM: pin `runtime.max_batch_size` for an A/B, or the arms differ.

## Do not touch

Generated config documentation: edit the inline comments in `config.h`, which are the source.

## See also

[`docs/internals/MEMORY.md`](../../docs/internals/MEMORY.md) (VRAM, ownership, lifetime), [`docs/determinism.md`](../../docs/determinism.md) (what reproducibility guarantees).
