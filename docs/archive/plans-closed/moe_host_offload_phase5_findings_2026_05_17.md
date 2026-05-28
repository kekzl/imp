# MoE host-offload Phase 5 — empirical findings

**Status (2026-05-17): blocked on dispatch refactor. Flag shipped opt-in for research.**

Phase 5 of the MoE host-offload + CUDA Graphs design (`moe_host_offload_graphs_design_2026_05_17.md` §5) was supposed to be a 3-5-day finish: drop the `experts_on_host_ && use_cuda_graphs` guard at `engine.cpp:1158`, capture the decode loop with prefetch + compute nodes, replay. The empirical attempt surfaced two blockers that the original design didn't account for. Both are structural — neither can be fixed by a config-level tweak.

## 1. Architectural mismatch: dispatch is host-driven, capture wants device-driven

The decode dispatch path goes through `ExpertLRUCache::get_or_load(layer, proj, key, src_host, expert_bytes, stream)`. That call does, on the host:

1. `std::unordered_map::find(key)` — branches on a value that's data-dependent on the router output.
2. `std::list` LRU splice (recency update).
3. Slot allocation (round-robin / LRU within the layer's sub-pool) — host branch on slot occupancy.
4. `cudaMemcpyAsync(slot.gpu_ptr, src_host, expert_bytes, ...)` issued with arguments computed from the host branches above.

Inside a `cudaStreamBeginCapture` window, the captured graph snapshots **the exact pointer values and node sequence that the host computed during capture**. On replay the captured `cudaMemcpyAsync` node fires with **the same** `slot.gpu_ptr` and `src_host`. If a different token's router picks a different expert E', the host-side `get_or_load` is not re-run during graph replay — the captured memcpy still copies expert E into the same slot, and the dispatch kernel reads expert E's bytes regardless of what the router picked. Silent data corruption.

This isn't an oversight in `get_or_load`. It's that the dispatch architecture was always host-driven; making it capture-correct requires moving the slot decision into a device kernel (read the Phase 2 mirror, compute slot pointer at runtime), which is a multi-week refactor of every `dequant_expert` / `expert_gemm` site.

## 2. Cross-stream event wait fails during capture

Phase 4's prefetcher uses a separate `prefetch_stream_` and per-layer `cudaEvent_t prefetch_done_`. The dispatch waits on these events from the compute stream:

```cpp
cudaStreamWaitEvent(compute_stream, prefetch_done_[layer], 0);
```

Under `cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeRelaxed)` this fails immediately:

```
[ERROR] expert_cache.cu:419: CUDA error: cudaStreamWaitEvent(...) —
        dependency created on uncaptured work in another stream
```

CUDA Graphs require all dependencies referenced inside the capture window to be **part of the same capture** (or pre-recorded events whose work is fully outside the capture). The prefetch stream is *not* under capture; it's making forward progress concurrently with the compute stream's capture. Relaxed mode loosens many constraints but not this one — cross-stream events between captured and uncaptured streams are explicitly disallowed.

Fixes that would unblock this:

- **Capture both streams** (`cudaStreamBeginCapture` on the prefetch stream too) — then the prefetcher's H2Ds become memcpy nodes in the graph with fixed src/dst (the same problem #1 above resurfaces: fixed pointers don't adapt to per-token routing).
- **Move prefetch out of the capture window** — issue prefetch *before* `BeginCapture` and *after* `EndCapture`, leaving only deterministic compute inside the graph. Loses the layer-L+1 prefetch overlap with layer-L compute.
- **Use `cudaStreamCaptureModeThreadLocal`** — still doesn't permit cross-stream-event wait inside the capture window, only loosens the cross-thread sync rules.

None of these are config-level changes; each is a meaningful refactor of how the engine schedules work.

## What ships in this PR

- `moe.allow_graphs_under_offload` config flag (default `false`). When set, the `engine.cpp:1158` guard is skipped and graph capture is attempted under host-offload. A loud `IMP_LOG_WARN` fires explaining the architectural caveat.
- An `IMP_LOG_INFO` tip on the disable path pointing at the new flag, so users discover the experimental path through the existing log surface.

**The flag does not currently produce useful graphs.** Enabling it on Qwen3.6-35B-A3B Q4_K_M with `force_host_experts=10` + `prefetch_top_k=3` triggers the blocker in §2 immediately:

```
[INFO]  CudaGraphCapture: using cudaStreamCaptureModeRelaxed
[ERROR] cudaStreamWaitEvent(...) — dependency created on uncaptured work in another stream
[ERROR] cudaMemcpyAsync(...) — operation failed due to a previous error during capture
[ERROR] (... ~30 propagated capture errors per token across MoE + GDN + SSM ...)
```

The flag is shipped anyway so future work has a hookpoint and the architectural state is discoverable from `config.h`'s documentation block, not buried in this memo. Phase 5 proper requires either (a) the kernel-side mirror read refactor or (b) moving prefetch outside the capture window. Both are tracked in the design memo and are pre-conditions on any further perf claims.

## Phase 4 perf claim correction

A side-finding during Phase 5 implementation: the Phase 4 (#236) PR reported **+43 % decode** (37.74 → 53.92 tok/s) at `prefetch_top_k=3` on Qwen3.6-35B-A3B Q4_K_M with `force_host_experts=10`.

That measurement was an artifact of a silent prefetch bug (per-(layer, proj) byte size mismatch — see PR #237). The broken prefetch's `cudaMemcpyAsync` returned "invalid argument" but the executor's `cudaGetLastError()`-as-clear path swallowed it, and the dispatch's compute-stream fallback re-loaded the same expert from the same host pointer — saving one real H2D per nominal-but-failed prefetch.

After the fix, the honest Phase 4 perf at `prefetch_top_k=3` on the same config is **+10 %** (30.80 → 33.78 tok/s). The Phase 1 nsys spike's projected +348 % over current host-offload remains the right ceiling target; Phase 4 alone moves ~3 % of that distance, not ~12 %.

## Path forward

The next meaningful slice is **Phase 5.1: kernel-side mirror read** — refactor `dequant_expert` and `expert_gemm` so each dispatched kernel takes the per-layer `d_lookup_` slice + the per-layer slot-pool base as parameters, and reads `slot_idx = d_lookup_[proj * n_experts + expert]` at runtime to compute its src pointer. Once dispatch is kernel-driven, the captured graph's compute nodes adapt to per-token routing without re-running host code, and §1's data-corruption blocker dissolves.

Phase 5.1 is **multi-week work** — every gemv variant (`gemv_q6k`, `gemv_q8_0`, `gemv_dp4a_kpar_*`, the dequant-then-cuBLAS fallback) needs the new parameter convention, and the testing surface includes all currently-supported MoE quants (Q*_K, NVFP4, F16-cache). The design memo's "3-5 days for Phase 5" estimate underestimated this by a wide margin; the realistic range is 2-4 weeks.

Until Phase 5.1 lands, the `moe.allow_graphs_under_offload` flag remains experimental scaffolding. The right user-facing advice continues to be `IMP_EXPERT_OVERHEAD_PCT=10` — keep all experts on device and the existing graph fast-path delivers +97–234 % over host-offload as already documented (`cuda_graphs_moe_works_2026_05_07.md`).
