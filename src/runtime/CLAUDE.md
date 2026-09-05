<!--
layer: L3
audience: agents
verified: 2026-09-06
commit: b5de0dd7
-->

# src/runtime - engine, scheduler, config, KV

The request lifecycle: admission, continuous batching, KV allocation, CUDA-graph
capture, and the configuration surface everything else reads.

## Invariants

- **`RuntimeConfig` (`config.h`) is the only configuration surface.** No ad-hoc
  env reads. Six env vars are seeded into it, in `config.cpp` and nowhere else:
  `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`, and the four trace knobs
  `IMP_SPEC_TRACE`, `IMP_JUMP_TRACE`, `IMP_PPL_DUMP`, `IMP_WORKER_TIMING`, which
  land in `diagnostics.*` keys (#1207, AUDIT_arch_2026 J-10).
- **There is no process-global config.** It hangs off the engine, one per engine.
- **A `--set` key that does not exist is an error**, not a warning. A typo used
  to measure the default silently.
- **Internal errors throw** and are translated to `ImpError` at
  `src/api/imp_api.cpp`. Do not convert them to status returns.
- **Capacity is planned, not discovered.** A successful `cudaMalloc` proves
  nothing about free VRAM on WSL2, and free VRAM only ever decreases within a
  process. Never size anything from a live `cudaMemGetInfo` reading.

## Entry points

- `engine.cpp`: lifecycle, suspend/resume, the top-level step
- `engine_scheduler.cpp`, `scheduler.cpp`: admission and continuous batching;
  prefill execution lives in `engine_prefill.cpp` / `engine_prefill_ragged.cpp`,
  the pipelined batched decode in `engine_decode_pipeline.cpp`
- `engine_kv_cache_init.cpp`: KV block geometry and pool sizing
- `engine_graph_decode.cpp`, `cuda_graph.cu`: capture and replay
- `config.h` / `config.cpp`: every key, with its default and rationale inline

## Test

This directory's tests run in the CPU lane (`make dev-test`); new CPU tests go
to **test-core**. Anything touching KV, graphs or scheduling needs
`make verify-fast` before it ships: CI has no GPU and cannot see a wedge.

## Pitfalls

- Prefill graph capture is default-**on** (`runtime.prefill_graph`, flipped
  2026-05-17). The legacy host-args MoE prefill path and non-F16 KV append run
  eager (`engine_prefill.cpp`, #874). Read the config value, not old comments.
- `kv_cache.swa_snapshot_mb` below one snapshot size disables prefix caching
  entirely, which is worse than zero.
- Anything sized off free VRAM must pin `runtime.max_batch_size` for an A/B, or
  the arms differ by ~1.6 GB of pre-upload noise.

## Do not touch

Generated config documentation; edit the inline comments in `config.h`, which
are the source.

## See also

[`docs/internals/MEMORY.md`](../../docs/internals/MEMORY.md) before anything
about VRAM, ownership or lifetime,
[`docs/determinism.md`](../../docs/determinism.md) for what reproducibility
actually guarantees.
