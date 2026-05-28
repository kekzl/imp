# MoE host-offload CUDA Graphs design memo — 2026-05-17

**TL;DR — defer.** The 4-6-week engineering required to restore CUDA Graphs
under host-offloaded experts only pays back if a model is **both** (a) too
big to fit on the RTX 5090 at the aggressive 10% expert-overhead headroom
**and** (b) decode-rate-critical. No model in imp's current support matrix
satisfies both: every host-offload-triggering model that exists today either
fits with `IMP_EXPERT_OVERHEAD_PCT=10` (Qwen3-Coder-30B-A3B Q6_K, Qwen3.6-35B
Q4_K_M, Gemma-4-26B Q4_K_M — all confirmed +97-234% decode wins via the
existing workaround) or is so far over the VRAM ceiling that even host-offload
+ Graphs would still be PCIe-bound. The right next step is **Phase 1 only**
(profile current host-offload mode to measure the actual cost) — a 2-day
investment that decides whether the multi-week effort is ever worth starting.

## Table of contents

1. [Status — what's the cost of disabled Graphs in host-offload?](#1-status--whats-the-cost-of-disabled-graphs-in-host-offload)
2. [The two reasons Graphs are blocked under host-offload](#2-the-two-reasons-graphs-are-blocked-under-host-offload)
3. [The "device-side LRU + async pipeline" plan](#3-the-device-side-lru--async-pipeline-plan)
4. [Risks](#4-risks)
5. [Implementation phases](#5-implementation-phases)
6. [Decision recommendation](#6-decision-recommendation)

---

## 1. Status — what's the cost of disabled Graphs in host-offload?

### Where Graphs get turned off

`src/runtime/engine.cpp:1150-1173` — after `init_weights()` finishes the
expert upload phase, the engine scans every layer for a host-resident
`expert_up_packed`. Any one host-resident expert flips `experts_on_host_ =
true`, which then forces `config_.use_cuda_graphs = false` (line 1164):

```cpp
if (experts_on_host_ && config_.use_cuda_graphs) {
    IMP_LOG_INFO("Disabling CUDA graphs: expert weights on host");
    IMP_LOG_INFO(
        "  Tip: if model+KV fits in VRAM, set IMP_EXPERT_OVERHEAD_PCT=10 "
        "(default 30) to upload ALL experts and re-enable CUDA graphs "
        "(+~180%% decode on Qwen 3.6 35B Q4_K_M).");
    config_.use_cuda_graphs = false;
}
```

The companion comment two blocks down (line 1170-1172) calls out the
asymmetry the memo `cuda_graphs_moe_works_2026_05_07.md` already established:
the **decode fast-path** is fully device-side and graph-safe; only the
**prefill** path uses D2H `expert_offsets` sync (and prefill isn't captured
anyway). So host-offload is the only remaining decode-side blocker.

### The H2D staging code

`src/graph/executor_forward_moe.cu:1256-1278` and `:1413-1426` — the dequant
helper and the fused-GEMV helper both branch on `packed.on_device`. When
false, they call into `expert_cache_.get_or_load()` (LRU on GPU pool) or fall
back to a single `cudaMemcpyAsync` into `moe_.raw_staging_buf`:

```cpp
if (!packed.on_device) {
    const char* host_ptr = static_cast<const char*>(packed.data) + offset;
    if (expert_cache_.n_slots_ > 0) {
        ExpertCacheKey ck{packed.data, expert_idx};
        void* cached = expert_cache_.get_or_load(ck, host_ptr, expert_raw,
                                                 stream);
        src = static_cast<const char*>(cached);
    } else if (moe_.raw_staging_buf) {
        cudaMemcpyAsync(moe_.raw_staging_buf, host_ptr, expert_raw,
                        cudaMemcpyHostToDevice, stream);
        src = static_cast<const char*>(moe_.raw_staging_buf);
    }
    ...
}
```

`ExpertLRUCache` (`src/graph/expert_cache.cu`) lives entirely on the host:
`std::unordered_map<ExpertCacheKey, ...> lookup_` for the index,
`std::list<int> lru_order_` for the recency chain, both manipulated CPU-side
per `get_or_load()` call. The `cudaMemcpyAsync` source pointer
(`slot.gpu_ptr`) comes from `slots_[slot_idx]`, where `slot_idx` is chosen by
the host-side LRU walk.

### Quantified cost — decode tok/s with vs without Graphs

From `moe_expert_offload_fix_2026_04_24.md` (RTX 5090 native Linux,
Qwen3-Coder-30B-A3B Q6_K, tg256):

| Config | tok/s | CUDA Graphs | Notes |
|---|---:|---|---|
| 30% overhead default (host-offload triggers) | **77.81** | OFF | partial offload, LRU path |
| `IMP_EXPERT_OVERHEAD_PCT=10` (all on GPU) | 237.88 | ON | every expert on-device |
| Auto-pick (10% if fits) | 240.30 | ON | same as manual |

So host-offload at the moment is a **3.08×** decode penalty for this model —
and that's the combined effect of (a) per-token H2D copy on the critical
path plus (b) no CUDA-Graph launch coalescing. Splitting the two
contributions:

From `cuda_graphs_moe_works_2026_05_07.md` (same RTX 5090, all experts
on-device, Qwen3-Coder-30B Q6_K, tg128):

| | tok/s | Graphs |
|---|---:|---|
| `--no-cuda-graphs` | 117.40 | OFF |
| default | **231.67** | ON |

So **Graphs alone are worth +97% decode** on Qwen3-Coder-30B Q6_K when the
H2D problem is already eliminated (experts on-device). That sets the Graphs
ceiling for this kernel: **117 → 231 tok/s if we kept host-offload but
re-enabled Graphs**.

Compare with the 77.81 → 237.88 measurement: ~40 tok/s of the gap (77 → 117)
comes from "host-offload at all" (PCIe + LRU bookkeeping); ~115 tok/s
(117 → 232) comes from "Graphs vs no-Graphs". The Graphs-recovery target
under this design is **the second number** (the 117 → 231 step) — *assuming*
the prefetch pipelining keeps PCIe out of the critical path, which is the
big unknown.

### For Qwen3.6-35B Q4_K_M specifically

From `MEMORY.md` baselines: this model is the canonical "+180% decode via
`IMP_EXPERT_OVERHEAD_PCT=10`" case (current best 143 tok/s with the trick,
the README quotes the +180% gain). The same per-model table in
`moe_expert_offload_fix_2026_04_24.md` shows expert weights at 18.22 GiB —
fits with both 10% and 30% headroom, **so this model doesn't actually
trigger host-offload today** (the table marks 30% as "✅ (barely)"). The
180% number quoted in the engine.cpp tip is a historical artifact from when
the default was 30% and the partial-offload path was reached anyway; current
main with the auto-probe lands `IMP_EXPERT_OVERHEAD_PCT=10` automatically
and decode is already at the "Graphs ON, all on device" tier.

That makes Qwen3.6-35B Q4_K_M a **bad poster child for this design**: the
workaround already wins; restoring Graphs in host-offload buys nothing
because host-offload doesn't get hit. The honest target model is
Qwen3-Coder-30B-A3B Q6_K on a config that **forces** host-offload (e.g.
WSL2/WDDM where the 30% safety margin is real, or a different SKU with
<32 GiB VRAM). For those configs, the design would recover the 117 → 231
tok/s Graphs win — roughly **+97% decode**.

## 2. The two reasons Graphs are blocked under host-offload

CUDA Graph capture demands every operation be replayable with no
host-resident decisions and stable device pointers. Two violations exist on
the host-offload path:

### 2a. Host-pointer dereference per layer per token

`ExpertLRUCache::get_or_load()` performs CPU-side work:

1. `lookup_.find(key)` — a `std::unordered_map` probe on the host
2. On hit: splice `lru_order_` (`std::list<int>`) to move the slot to the
   front
3. On miss: scan `slots_` for an unoccupied entry, or evict from the back
   of `lru_order_`
4. Emit `cudaMemcpyAsync` with the chosen slot's `gpu_ptr` as destination

Steps 1-3 are CPU branches whose outcomes change every token (LRU order is
data-dependent on routing history). A captured graph would replay the
**recorded** sequence — wrong slot, wrong copy direction, wrong dependency
edges. Even step 4 alone breaks capture: the `gpu_ptr` chosen this token
may not be the slot the kernel reads next token.

### 2b. `cudaMemcpyAsync` per expert per token with mutating source pointer

The source pointer (`host_ptr = packed.data + expert_idx * expert_raw`)
varies per token because **`expert_idx` is router output**, which differs
each token. Graphs require static source/destination pointers at capture
time — `cudaGraphAddMemcpyNode` records the exact pointer; replay copies
from that same address. A router that picks expert 7 at capture time and
expert 42 at replay would silently read expert 7's bytes every step.

`cudaGraphExecMemcpyNodeSetParams` exists for runtime pointer updates, but
the imp graph executor doesn't drive it on a per-token cadence today
(decode-fast-path graphs only update activations / KV pointers, not weight
memcpy sources), and it would still need a host-side decision about which
node to update — putting us right back at problem 2a.

## 3. The "device-side LRU + async pipeline" plan

The fix shape that unblocks Graphs has three pieces: bookkeeping on the
device, static pointers in the captured graph, and prefetch that hides the
PCIe cost.

### 3a. Move the LRU table to device memory

Allocate a device-resident table:

```
device_expert_cache_table_[num_layers][cache_slots]  // slot → expert_id
device_expert_cache_lookup_[num_layers][n_experts]   // expert_id → slot (-1 = miss)
```

Update from CPU via a **single** `cudaMemcpyAsync` once per request
(prefetch decision baked in), or — better — keep the lookup pure-device by
having a small device kernel decide which expert occupies which slot when
the router commits its top-K choice. The per-token routing flow at
`executor_forward_moe.cu:1256` (the `dequant_expert` lambda) already has
`expert_idx` in scope from the existing routing-output buffer; it would
index `device_expert_cache_lookup_[layer][expert_idx]` device-side to find
the slot index, then read from `cache_slot_buf_[layer][slot]` directly.

The hot read in `executor_forward_moe.cu:1413-1426` (the fused-GEMV path)
gets one extra device-side indirection (table lookup) but **zero
host-pointer dereferences**.

### 3b. Static src/dst pointers

Pre-allocate a fixed bank of device buffers:

```
cache_slot_buf_[layer][slot]   // device-resident, slot_size_ = max expert raw bytes
```

— this is exactly `ExpertLRUCache::pool_` today, but partitioned per layer
to make the graph capture deterministic. The captured graph contains
**fixed `cudaMemcpyAsync` nodes** from a fixed host buffer (engine-owned,
pinned, `cudaHostAlloc` with `cudaHostAllocPortable`) to a fixed device
slot. Both endpoints known at capture time.

LRU eviction becomes: write a new entry into the device lookup table
(replacing whatever was there). No host pointer math, no `std::list`
splicing — eviction is *implicit* in the table-overwrite.

The dispatch then reads from `cache_slot_buf_[layer][routed_slot]` —
`routed_slot` is the device-side lookup result. Static destination pointer,
data-dependent index.

### 3c. Async pipelining

The fundamental constraint is the **per-expert PCIe latency**. An H2D copy
of one Q6_K expert (Qwen3-Coder-30B-A3B, ~22 GiB / 48 layers / 128 experts
≈ 3.7 MiB/expert) at WSL2-typical 12 GiB/s PCIe Gen5 takes ~300 µs. That's
catastrophic on a 4 ms decode budget. The only way this works is if the
copy is **already done** before the layer needs it:

- **Phase 1**: at layer L's start, kick off prefetch H2D for layer L+1's
  hot experts. "Hot" = predicted from the previous token's routing pattern
  (high-locality assumption — same K experts dominate over short windows)
  or a static "top-2K most-used experts" heuristic baked at warmup.
- **Phase 2**: layer L's compute uses already-prefetched experts (which the
  L-1 phase staged). No H2D on the critical path.

Pipeline depth: **prefetch L+1 while computing L**. Two slots per layer
minimum (one being read this token, one being filled for next layer).
Cross-layer pipelining via separate `cudaStream_t` (the upload stream
pattern is already established in `init_weights()` —
`src/runtime/engine.cpp:1140-1141` uses `upload_stream` + `cudaEvent` for
H2D overlap).

This *also* requires the prefetch decision itself to be graph-safe — i.e.
the "which experts do I prefetch for L+1" choice must be device-side.
Either (a) prefetch top-K most-recent-used per layer (read from the
device-side LRU directly, no host involvement), or (b) prefetch a
static-at-warmup top-2K shortlist (graph nodes hold fixed pointers).

## 4. Risks

### 4a. Prefetch misprediction

If the router picks a non-prefetched expert mid-replay, we need a
synchronous H2D fallback — which **breaks Graph replay**. Options:

- **Conservative**: prefetch top-2K per layer (where K is router top-K,
  e.g. 8 → prefetch 16 candidates). Doubles PCIe bandwidth requirement
  per token but lets us survive routing churn. Qwen3-Coder-30B router has
  fairly stable top-8 over short windows (token-to-token Jaccard ~0.7
  empirically — see `nvfp4_moe_prefill_landscape_2026_05_10`), so top-16
  prefetch likely covers >95% of selections.
- **Fallback re-launch**: detect the miss device-side (via an atomic
  device counter incremented by the dispatch), end the graph early,
  recover with eager mode, re-capture. Adds latency but stays correct.
  Mainline path becomes "graph until first miss, then eager for the rest
  of the request" — degrades to current host-offload behavior. Acceptable
  if misses are rare.

### 4b. Pinned host memory ceiling

The plan assumes the entire expert weight pool is in pinned host memory
(`cudaHostAlloc`) so that `cudaMemcpyAsync` is true async DMA, not
synchronous-via-pageable. Numbers for Qwen3.6-35B Q4_K_M (60 layers × 128
experts × ~8 MB/expert ≈ 60 GiB) **don't fit in a 32 GiB-RAM WSL2 host**.

This forces a two-level cache: pinned ring buffer (e.g. 4 GiB) acting as a
host-side LRU staging area, with `cudaHostRegister` / `cudaHostUnregister`
churn — but `cudaHostRegister` on a per-expert cadence is itself slow
(~ms-per-call). Alternatives:

- **Host-side LRU on top of a pinned pool** — pinned pool sized to fit
  some N>K experts per layer, populated via `memcpy` from pageable
  `mmap`'d weight file. The `memcpy` is CPU-bound but happens at request
  start, not per-token.
- **Selective pinning** — only the top-2K most-used experts per layer get
  pinned; tail experts stay pageable and pay the bounce-buffer cost
  (~half the bandwidth) when they're picked. Skews the misprediction risk
  worse but bounds pinned-memory cost.
- **Skip the model** — if pinned RAM can't hold even the working set, the
  config is genuinely infeasible. Document and reject at init time.

For native Linux with 64+ GiB host RAM the whole 60 GiB fits and this risk
collapses; for WSL2/laptops it's the dominant blocker.

### 4c. Cache coherence between device-side table and graph replay

Device-side table updates (writes to `device_expert_cache_lookup_`) must
become visible **before** the consumer kernel reads — but the captured
graph contains the kernel launch as a fixed node. Two patterns:

- **Atomic device flag + spin** in the consumer kernel: producer writes
  table entry, then `__threadfence_system()` + atomic store to a "ready"
  flag; consumer kernel reads the flag (graph-captured `cuStreamWaitValue`
  node) before dispatching. CUDA 13.x supports `cuStreamWaitValue` inside
  graphs.
- **Stream dependency**: prefetch stream writes the table, signals an
  event, compute stream waits on the event before the dispatch kernel.
  This is the standard cross-stream pattern already used in
  `LayerOffloadManager` (`src/memory/layer_offload.h:30`) — `ensure_layer`
  waits, `prefetch_layer` signals.

The second is simpler and already proven in the codebase. Likely the right
default; first is the fallback if event overhead becomes a bottleneck.

### 4d. Graph cache size

`config_.max_seq_len` × `num_layers` × `top_K` plausible expert combinations
explodes the captured graph variant count if we go down the per-routing
specialization rabbit hole. The design keeps **one graph per
(n_tokens=1, prefill_chunk_size)** — i.e. the routing-dependence is encoded
in *device-side table reads*, not in graph-node selection. Graph cache
stays the same size as today's decode-fast-path graph cache.

## 5. Implementation phases

### Phase 1 — Profile current host-offload mode (2 days, gate)

**Goal:** answer "is the engineering even worth it?"

- Pick a configuration that *actually triggers* host-offload today. The
  natural candidates are (a) WSL2/WDDM with `IMP_EXPERT_OVERHEAD_PCT=30`
  forced on Qwen3-Coder-30B-A3B Q6_K, or (b) a synthetic OOM via a small
  artificial `vram_alloc_.init(0.50f)` to leave room for KV but force
  experts to host. (b) is reproducible; (a) is the real user pain point.
- Run `nsys profile --trace=cuda,nvtx imp-cli --bench --bench-pp 0
  --bench-reps 5 --max-tokens 256` against Qwen3-Coder-30B-A3B Q6_K under
  forced host-offload.
- Breakdown the per-token timeline:
  - Wall-clock per token (target: 1000/77.81 ≈ 12.8 ms)
  - Time in `cudaMemcpyAsync` (PCIe transfer cost)
  - Time in `ExpertLRUCache::get_or_load()` (CPU bookkeeping cost)
  - Time in dispatch kernels (the work itself)
  - Idle gaps (would-be Graph-launch coalescing wins)
- Sanity-check the ceiling: project "PCIe perfectly overlapped, Graphs
  fully restored" — does it land in the +97% range that
  `cuda_graphs_moe_works_2026_05_07.md` predicts, or are we actually
  PCIe-bound regardless?

**Decision gate:** if the projected ceiling is <40% improvement over
today's host-offload (i.e. PCIe is fundamentally the bottleneck), **stop**
— the engineering doesn't pay back. If ≥80%, proceed.

### Phase 2 — Migrate `expert_cache_` to device-side table (1 week)

Bookkeeping only — no perf change, just lays foundation. The host-side
`ExpertLRUCache` continues to make the eviction decisions; the lookup
table merely **mirrors** the host state onto the device. Hot read sites
in `executor_forward_moe.cu:1262` and `:1419` keep using the existing
`expert_cache_.get_or_load()` API; under the hood it now also writes
`device_expert_cache_lookup_[layer][expert_id] = slot_idx` after every
update.

Correctness validation: a debug mode that asserts host-LRU and device-LRU
return identical slot_idx for every lookup. All 574 tests pass; no perf
regression in non-offload configs (which don't touch this code).

### Phase 3 — Static device cache slots, per-token dispatch through table (1-2 weeks)

Replace the per-call `get_or_load()` indirection at `:1262` and `:1419`
with **direct device-side reads**:

- New `cache_slot_buf_[layer][slot]` allocation in
  `MoeWorkspaceBuffers` (next to `raw_staging_buf`).
- Dispatch kernels gain an extra parameter: `const int* d_lookup_table`
  for this layer. They look up `slot = d_lookup_table[expert_id]`,
  read from `cache_slot_buf_[layer][slot]` directly.
- LRU policy: simplest first — round-robin slot replacement on every
  miss. Per-layer recency tracking added in Phase 4 if needed.
- `cudaMemcpyAsync` source pointer becomes
  `host_expert_buf_[layer][expert_id]` (still expert-id-keyed, but a fixed
  host pointer with no host-side computation — capture-safe).

Validation: with Graphs still disabled, prove bit-identical output to
Phase 2 across 5 prompts × 256 tokens. This is the harder rewrite of the
two helper sites — keep their fallback paths (`raw_staging_buf` direct
copy) for the no-cache configuration.

### Phase 4 — Async prefetch L+1 from L (1 week)

- Second stream (`prefetch_stream_`) issues H2D for layer L+1's predicted
  hot experts while compute stream runs layer L.
- Event-based handshake: `prefetch_done_[L+1]` event signaled by prefetch
  stream, awaited by compute stream's first layer-(L+1) dispatch.
- Prediction policy: round-robin device-side recency wins for the MVP
  (write recency back into a small device-side ring per layer, prefetch
  the top-K most-recent). Static top-2K shortlist as a fallback if the
  recency-based predictor underperforms.

Measure PCIe overlap with nsys: target is `prefetch_stream` H2D
**finishing before** compute stream needs it. If not, the prefetch lead
needs to grow (L+2, L+3, …) and the pinned-memory footprint with it.

### Phase 5 — Capture decode CUDA Graphs (3-5 days)

Drop the `experts_on_host_ && config_.use_cuda_graphs` guard at
`engine.cpp:1158-1165`. Capture decode loop:

- Graph nodes for layer L's compute (static slot reads, table lookups)
- Graph nodes for layer L+1's prefetch (fixed src host ptr, fixed dst
  slot)
- Event nodes for the handshake

Bench Qwen3-Coder-30B-A3B Q6_K host-offload Graphs ON vs OFF. Target:
+50-90% decode over the current 77.81 tok/s baseline (the +97% in-memory
ceiling from `cuda_graphs_moe_works_2026_05_07.md`, discounted for the
PCIe overhead that remains).

**Total: 4-6 weeks** (Phase 1 → 2 days; Phase 2 → 1 week; Phase 3 → 1-2
weeks; Phase 4 → 1 week; Phase 5 → 3-5 days).

## 6. Decision recommendation

**Defer with a Phase-1 spike.** Two days of profiling closes the question
of whether the multi-week effort can clear the PCIe ceiling. If profiling
shows we're already PCIe-bound at 77 tok/s, no amount of Graph restoration
will help — the conclusion is "host-offload is correctly slow, recommend
users hit the `IMP_EXPERT_OVERHEAD_PCT=10` path or move to a bigger card."

The full Phase 1-5 only makes sense if (a) Phase 1 shows ≥80% headroom
over current host-offload **and** (b) a real production workload appears
that can't be served by the existing workaround — e.g. a Qwen3-MoE-405B-
class model that overflows 32 GiB even at 10% overhead, on hardware that
can't be upgraded. No such workload exists in imp's current support
matrix; every host-offload candidate model in `MEMORY.md` either fits with
the workaround or is gated by a different bottleneck.

**Justification (one sentence):** the existing `IMP_EXPERT_OVERHEAD_PCT=10`
workaround already unlocks +97% to +234% decode on every production MoE
model in scope today, and 4-6 weeks of device-side LRU + async prefetch
plumbing only matters for hypothetical future models that don't fit even
at the lowered headroom — none of which are on the roadmap.

## Cross-references

- Memos: `moe_expert_offload_fix_2026_04_24.md` (the workaround that
  obviates this design today), `cuda_graphs_moe_works_2026_05_07.md`
  (the +97-234% Graph ceiling), `moe_prefill_graphs_plan_2026_05_10.md`
  (the device-side bookkeeping pattern, already shipped for prefill
  via PR #164)
- Code: `src/graph/executor_forward_moe.cu:1256-1278` (dequant H2D
  staging), `:1413-1426` (fused-GEMV H2D staging),
  `src/runtime/engine.cpp:1150-1173` (where Graphs get disabled),
  `src/graph/expert_cache.cu` (current host-side LRU),
  `src/graph/executor.h:34-94` (ExpertCache types),
  `src/memory/layer_offload.h` (the prefetch/event pattern already in
  the codebase)
