# Phase 2 — Decode Batching Design Spec

**Goal:** imp-server serves up to 4 parallel decode requests. Weight reads are amortized across sequences. Prefill remains serial (one request at a time). This enables parallel AI agents on a single RTX 5090.

**Scope:** Remove artificial batch=1 constraints in the decode path. The infrastructure (scheduler, KV manager, BatchBuilder, GPUBatchPool, CUDA graph pool) already supports n_sequences > 1. The work is primarily constraint removal and wiring.

**Out of scope:** Ragged prefill batching, multi-GPU, changes to the C API (`imp_generate` stays single-request blocking).

---

## Architecture

```
HTTP Requests → BatchingEngine.submit() → pending_queue_
                                               ↓
Worker Thread: engine.step() loop {
  1. Scheduler: promote pending → active (up to max_batch_size=4)
  2. For each request in PREFILLING state:
       step_prefill_one(request)  [serial, one at a time]
       On completion: request.status = DECODING, joins decode batch next step
  3. Collect all DECODING requests → decode_batch (n_sequences=1..4)
  4. step_decode_batch(decode_batch):
       - Build GPUBatch via BatchBuilder (per-seq token, position, block_table)
       - Select graph from graph_pool[n_sequences-1]
       - Run forward_logits() [batched GEMV — weights read once, 4 sequences computed]
       - Per-sequence sampling (temperature, top-p/k, penalties, stop check)
  5. Route sampled tokens back to per-ServerRequest token queues
  6. Remove finished requests (EOS / max_tokens), freeing their KV blocks
}
```

### Prefill/Decode Interleaving

- Prefill is serial: if 3 requests arrive simultaneously, they prefill one-by-one (request 1 prefills while 2+3 wait in pending)
- Once a request finishes prefill, it immediately joins the decode batch
- Decode runs for ALL active decoding requests each step (batched)
- New prefills can interleave with active decodes: step does prefill-chunk for one request, then decode-batch for all decoding requests

### Why this works for agents

Agent workloads generate many tokens (long decode phase) with moderate-length prefills. The decode phase dominates wall time. Batching decode gives near-linear throughput scaling: 4 requests at ~70% of single-request decode speed = ~2.8x total throughput.

---

## Changes Required

### 1. Remove batch=1 decode constraint

**File:** `src/runtime/engine_scheduler.cpp`

The scheduler already collects decode requests into `sched_decode_batch_`. Today, a guard forces single-sequence decode for non-SSM models (line ~723 in engine_scheduler.cpp). This guard must be relaxed.

**Change:** Allow `n_sequences > 1` in the decode batch for standard transformer and MoE models. SSM/GDN models (Qwen3.5, Qwen3.6 SSM layers, Nemotron-H Mamba2) keep `force_single_decode = true` because their recurrent state is not batched.

### 2. Enable async graph loop for batch > 1

**File:** `src/runtime/engine_scheduler.cpp`

The async graph runner (`async_graph_runner_`) has a guard at line ~1262: `valid_decode.size() == 1`. This prevents batched decode from using CUDA Graphs.

**Change:** Extend the async graph loop to work with `n_sequences > 1`. The `decode_graph_pool_[batch_idx]` already pre-allocates separate graphs for each batch size. Select the graph matching the current decode batch size. If batch size changes between steps (request finishes or new one joins), fall back to eager execution for that step and re-capture on the next stable step.

### 3. Per-sequence sampling after batched decode

**File:** `src/runtime/engine_scheduler.cpp`

Sampling is already per-sequence in the decode path (lines 966-1000 show per-request penalty extraction, temperature, top-k/p). The code loops over `valid_decode` and samples each sequence independently.

**Change:** Verify this works correctly with `n_sequences > 1`. The logits tensor from batched forward is `[n_sequences, vocab_size]` — sampling already indexes per-sequence. Main check: penalty tokens and stop-sequence state must be correctly isolated per-request.

### 4. Server token routing

**File:** `tools/imp-server/batching_engine.cpp`

The worker loop must map sampled tokens back to the correct `ServerRequest`. Today with batch=1 there's only one active request to route to.

**Change:** After `engine->step()`, iterate over all active requests and push their latest token (if any) to the corresponding `ServerRequest::push_token()`. The request ID links engine request to server request.

### 5. KV budget partitioning

**File:** `src/runtime/engine_scheduler.cpp` (or `vram_budget.cpp`)

With 4 requests each needing KV blocks, total KV budget must be split. The LRU evictor already handles this gracefully (evicts oldest blocks when budget exhausted), but aggressive prefills could starve decode requests of KV space.

**Change:** Add a soft per-sequence KV cap: `max_context_per_request = total_kv_tokens / max_batch_size`. This isn't a hard limit (a single long request can use more if others are short), but prevents one request from consuming all KV and forcing eviction of other active requests. Scheduler checks available KV capacity before admitting new requests.

### 6. Config

**File:** `src/runtime/config.h`

```cpp
struct Runtime {
    int max_batch_size = 4;  // Max concurrent decode sequences
    // ... existing fields
};
```

This replaces the current `max_batch_size` in `ImpConfig` (C API) with a runtime config field that the server reads. The C API `ImpConfig.max_batch_size` stays for the single-context use case.

---

## What Does NOT Change

- **Prefill path** — stays serial (`step_prefill_one`, one request at a time)
- **FMHA / cuBLAS attention** for prefill — no changes
- **C API** — `imp_generate` remains single-request blocking. Batching is server-internal.
- **Kernel interfaces** — decode kernels already accept `n_sequences` via `InferenceState`
- **KV cache manager** — already per-sequence block tables, no changes
- **BatchBuilder / GPUBatchPool** — already support multi-sequence, no changes
- **Paged attention decode** — already handles multi-sequence (`context_lens[]`, `block_tables[]`)

---

## Validation

### Correctness
- Multi-request decode produces identical per-token output as single-request (same model, same prompt, same seed → same tokens regardless of what other requests run in parallel)
- KV isolation: request A's KV blocks never read by request B's attention
- Penalty/stop isolation: request A finishing doesn't affect request B's generation

### Performance
- Single-request decode: no regression (batch=1 path unchanged)
- 4-request decode: throughput ≥ 2.5x single-request (conservative, weight-read amortization)
- Prefill: no regression (still serial, same path)

### Testing
- Unit: extend `test_continuous_batching.cpp` with multi-decode scenario
- Integration: multi-request server test (4 concurrent `curl` clients)
- Degeneration check: verify coherent output from all 4 requests simultaneously

---

## Effort Estimate

~5-7 days. Most changes are guard removal + wiring. No new kernels, no new data structures. The riskiest part is the async graph loop extension (graph invalidation on batch size change).
