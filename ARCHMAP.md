# ARCHMAP.md — code-derived architecture, ownership & hot-path map

Derived from source on 2026-06-24 (audit pass 2), not from prose. Where docs and
code disagreed it is noted. Companion to `docs/architecture.md` (narrative);
this file is the ownership/lifetime/hot-path slice an auditor needs.

## Layer DAG (`src/`)

```
api ──▶ runtime ──▶ exec ──▶ compute ──▶ quant
                 │        └──▶ memory ──▶ core
                 ├──▶ model ──▶ core
                 └──▶ memory ──▶ core
vision ──▶ compute/model        lora ──▶ runtime/model
```

- **core** — `Buffer`, `Tensor`, `cuda_raii.h` (`CudaStream`/`CudaEvent`, move-only),
  logging + `IMP_CUDA_CHECK*` macros, `ModelProfile` (centralized arch facts).
- **memory** — `DeviceAllocator` (stream-ordered `cudaMallocAsync`/`cudaMemPool`),
  `KVCache` (paged, block_size 16) + `KVCacheManager` (block tables, LRU, prefix
  cache, pinning), pinned allocator.
- **quant** — GGUF Q4_0…Q8_0/Q*_K + NVFP4/FP8/INT8 dequant & quant kernels.
- **compute** — GEMM (cuBLASLt dense, CUTLASS grouped NVFP4 MoE), FA2 attention
  (`attention_fmha_sm120`), GDN, sampling, layernorm, MoE routing.
- **exec** — `GraphExecutor` (forward pass; intrinsically forward-pass-coupled —
  prior audit settled, do NOT split into runner classes), pre-dequant phases.
- **runtime** — `Engine` + scheduler (`engine_scheduler.cpp`), CUDA-graph decode
  (`cuda_graph.cu`, `engine_graph_decode.cpp`), spec-ngram, `RuntimeConfig`.
- **api** — `imp_api.cpp` C-ABI boundary (all entry points wrap try/catch →
  `ImpError`; nothing throws across the ABI).
- **tools/imp-server** — httplib server, `handlers.cpp` (~4600 LOC, OpenAI +
  Anthropic), `BatchingEngine` (the HTTP→GPU bridge).

Layer note (pass-1 D1): a few `compute/quant/memory → runtime` includes exist for
diagnostics/PDL only (instrumentation, not algorithmic coupling).

## Concurrency / async model (HTTP → GPU bridge)

```
N httplib handler threads ──submit()──▶ pending_queue_ (queue_mutex_)
                                              │
                          single BatchingEngine::worker_loop thread
                                              │ exclusive caller of
                                   Engine::step() / add_request()
                                              ▼
                                   Scheduler → GraphExecutor (GPU)
```

- The worker thread is the **sole** caller of `Engine`/`Scheduler`, so those are
  effectively single-threaded. **No mutex is held across GPU work.** `state.mtx`
  guards only short state snapshots + `submit()`. The concurrency model is sound
  and is not the bottleneck (the previously-suspected "concurrency cliff" was a
  Python-GIL harness artifact — see MEMORY).
- Cancellation: HTTP disconnect → `sink.is_writable()` false → `server_req->cancel()`;
  worker checks `is_cancelled()` between steps and at stream loop top. **Gap
  (F-A2):** the non-streaming unbounded conditional-graph burst does not re-poll
  between device-side tokens.
- `worker_loop` wraps `step()` in try/catch — a host throw cancels the batch and
  recovers; (audit) now also probes device health to fail-fast on a poisoned context.

## Decode hot path (per token, batch≥1)

```
step() ─▶ step_decode() ─▶ [spec-ngram gate?] ─▶ build batch from sched_decode_batch_
        ├─ per seq: append_block if needed  (evict_lru under pressure — F-A1 guarded)
        ├─ step_decode_forward(valid_decode)
        │     └─ GraphExecutor::forward_logits  (CUDA-graph replay; bucketed by
        │        n_sequences-1 and pow2 max_blocks_per_seq)
        ├─ sample (mask applied here for json_schema/constrained)
        └─ touch(seq) at step end  ◀── (F-A1: too late; batch protected at eviction)
```

Decode is at the HBM ceiling (~341 tok/s on 30B-MoE NVFP4); the only wall-breaker
is speculation. Steady-state decode is allocation-free.

## CUDA Graph ↔ allocator coupling (the highest-leverage soundness invariant)

- **Address stability holds.** The batched-decode graph captures only
  `forward_logits(state,…)`. `state.block_tables` point into `decode_batch_pool_`
  (a fixed arena allocated once, `batch.cpp:158-202`); contents are re-uploaded
  each step, pointers stay put. `state.kv_cache` is the single persistent pool
  (`kv_cache.cu:42-49`). The graph bakes stable addresses; only buffer *contents*
  change.
- **Batch-size change** → per-size graph pool keyed `n_sequences-1`. **max_blocks
  growth** → pow2 buckets → `cudaGraphExecUpdate`, full reinstantiate on failure.
- **Workspace arena vs dynamic KV are separated** by lifetime: workspace/persistent
  buffers are pre-sized to `max_tokens` at init *before* KV is sized against
  remaining free VRAM. Conditional-graph `d_block_tables` is alloc-once + re-upload
  with a capacity guard. The #683/#692 position off-by-one is fixed.
- Degraded-mode per-step `cudaMalloc` paths exist (`batch.cpp` raw upload, MoE
  `owns_memory`, `force_cublas_decode`) but are all OFF the live dispatch (init-pool
  fallbacks / debug flags); they would break capture if ever taken.

## Ownership / lifetime

- **`Model`** — `const Model*` in the executor; const-after-load, lock-free
  shareable by `shared_ptr` across contexts. The only mutations (`const_cast` in
  `pre_dequant_phase*`) run once at `init_kv_cache`, before any request.
- **KV blocks** — owned by `KVCacheManager`/`KVCache`; a `SequenceState` *borrows*
  via the `seq_id` block table. `free_sequence` is refcount-correct, idempotent,
  keeps pinned/prefix-cached blocks alive, handles `-1` StreamingLLM sentinels, and
  is called on completion, cancellation, and error. `lru_order_` holds **only live
  sequences** (finished ones are removed) — the fact underlying F-A1/F-A1b.
- **Device resources** — `core/cuda_raii.h` wrappers are move-only; a handful of
  managers still create streams/events raw with manual dtor cleanup (pass-1 C1;
  F-A13 fixed the weight-upload one; LayerOffload/ExpertLRUCache/GreenCtx parked).

## Determinism sources (catalog)

Gated by `[runtime] deterministic`: MoE permute/scatter atomics, top-k softmax
stats, cuBLASLt split-K. Documented exceptions: CUB top-k `>128`, `typical_p` smem
atomicAdd, GDN cross-context. **Audit gap (F-A9):** NVFP4 grouped-MoE CUTLASS GEMM
does not consult the flag (whether that is observable is unverified — see report).
