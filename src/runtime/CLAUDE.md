<!--
layer: L3
audience: agents
verified: 2026-08-31
commit: 01799405
-->

# src/runtime — engine, scheduler, config, KV

The request lifecycle: admission, continuous batching, KV allocation, CUDA-graph
capture, and the configuration surface everything else reads.

## Invariants

- **`RuntimeConfig` (`config.h`) is the only configuration surface.** No ad-hoc
  env reads. The env vars that remain seeded are `IMP_DETERMINISTIC` and
  `IMP_FMHA_FA2`; three former trace knobs became config keys in #1207.
- **There is no process-global config.** It hangs off the engine, one per engine.
- **A `--set` key that does not exist is an error**, not a warning. A typo used
  to measure the default silently.
- **Internal errors throw** and are translated to `ImpError` at
  `src/api/imp_api.cpp`. Do not convert them to status returns.
- **Capacity is planned, not discovered.** A successful `cudaMalloc` proves
  nothing about free VRAM on WSL2, and free VRAM only ever decreases within a
  process. Never size anything from a live `cudaMemGetInfo` reading.

## Entry points

- `engine.cpp` — lifecycle, suspend/resume, the top-level step
- `engine_scheduler.cpp`, `scheduler.cpp` — admission and continuous batching;
  prefill execution lives in `engine_prefill.cpp` / `engine_prefill_ragged.cpp`,
  the pipelined batched decode in `engine_decode_pipeline.cpp`
- `engine_kv_cache_init.cpp` — KV block geometry and pool sizing
- `engine_graph_decode.cpp`, `cuda_graph.cu` — capture and replay
- `config.h` / `config.cpp` — every key, with its default and rationale inline

## Build & test

```
make dev && make dev-test        # this directory's tests are in the CPU lane
make verify-fast                 # anything touching KV, graphs or scheduling
```

New CPU tests go to **test-core**. `make test-unit` is a different binary from
the CI lane; green there is not green in CI.

## Pitfalls

- Changing KV or graph code without `make verify-fast` is how a wedge ships: CI
  cannot see it.
- Prefill graph capture is default-**on** (`config.h`, `runtime.prefill_graph`,
  flipped 2026-05-17). A comment at `src/runtime/engine_init_resolver.cpp:731`
  still says "prefill is never graph-captured"; it predates the flip. Read the
  value, not the comment - and note this pointer is itself unguarded, since the
  citation gate covers `docs/` and the root docs, not the `CLAUDE.md` tree (it
  read `:565` from 2026-08-13 until 2026-08-31 while the comment sat 166 lines
  further down).
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
