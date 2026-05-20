# K8 CPU offload design memo — 2026-05-17

**TL;DR — defer.** Lever 2 NVFP4 KV (PR 2026-05-07/08, 3.9× compression at
parity decode) already puts every imp production model at 100K+ context on
a 32 GB RTX 5090. The roadmap entry K8 ("async prefetch for cold tokens,
enables 100K+ context") was scoped before that lever shipped, and is now
solving a problem the engine no longer has at production scale. The real
demand surface — workloads needing >256K context on consumer Blackwell —
has not appeared from any imp user. If/when it does, the cheapest first
move is **Phase 1 bookkeeping only** (cold/hot flag on the block_table,
no kernel change), which keeps the option open without commiting to a
multi-week kernel + scheduler effort.

## Table of contents

1. [Status — what's the actual VRAM ceiling today?](#1-status--whats-the-actual-vram-ceiling-today)
2. [The K8 design space](#2-the-k8-design-space)
3. [PCIe budget](#3-pcie-budget)
4. [Sub-problem 1 deeper — what to offload](#4-sub-problem-1-deeper--what-to-offload)
5. [Async prefetch pipeline](#5-async-prefetch-pipeline)
6. [Risks](#6-risks)
7. [Implementation plan](#7-implementation-plan)
8. [Decision recommendation](#8-decision-recommendation)

---

## 1. Status — what's the actual VRAM ceiling today?

### Current KV sizing path

`Engine::init_kv_cache()` in `src/runtime/engine.cpp:1190-1279`:

- `kv_block_size` default 16 tokens/block (`kKVBlockSize` in `src/memory/kv_cache.h:12`),
  auto-bumped to 32 for `n_kv_heads ≤ 4` models (Qwen3.6 etc.).
- `max_blocks` comes from `compute_vram_budget()` (`src/runtime/vram_budget.cpp`):
  takes 80 % of free VRAM after weight/activation caches for KV
  (`NVFP4_DECODE_ONLY`, `FP8_PREFILL_NVFP4_DECODE`, `FP16_ONLY` strategies all
  land here), divides by `per_block_total` bytes = `K + V + scales` summed
  across all attention layers.
- `KVCache` owns one contiguous device pool (`pool_` in `kv_cache.h:103`)
  + scale pool + (optional) sketch / mscale pools. Per-layer-shape ctor
  (Gemma-4 dual head_dim) builds `layer_block_bytes_` / `layer_k_offset_` /
  `layer_v_offset_` vectors so 256-vs-512 head_dim is sized correctly.
- A short auto-cap of 16 384 tokens is enforced for `max_seq_len` when the
  user does not pass `--max-seq-len` (`engine.cpp:944`). This is policy,
  not a physical ceiling: explicit user value bypasses it.

### Per-model context ceiling, FP16 KV, 32 GB VRAM

Numbers below are the per-model figures from `nvfp4_kv_potential_2026_04_25.md`
(measured 2026-04-25) cross-checked against the current `vram_budget.cpp`
allocation path. They assume default budget split (~80 % of post-weight
free VRAM goes to KV).

| Model | layers (attn) | n_kv_heads | head_dim | KV B/tok | FP16 cap |
|---|---|---|---|---|---|
| Qwen3-4B Q8_0 | 36 | 8 | 128 | ~144 KiB | ~80 K |
| Qwen3-8B Q8_0 | 36 | 8 | 128 | 144 KiB | 16 K¹ |
| Qwen3.6-35B Q4_K_M | 40 (10 attn) | 2 | 256 | ~20 KiB | model native 256 K fits |
| Gemma-4-26B Q4_K_M | 30 (dual 256/512) | 8 | 256/512 | ~228 KiB | **13.2 K (capped)**² |

¹ Auto `max_seq_len` cap; physical ceiling for Qwen3-8B Q8 at FP16 KV is
~40 K. The auto cap exists because users rarely need it.
² Real measured cap with FP16 KV — VRAM-limited even on 32 GB.

### After Lever 2 NVFP4 KV (`lever2_nvfp4_kv_implemented_2026_05_07`)

3.9× compression (4 bits/elem + UE4M3 per-16-elem scale, no per-tensor
FP32 master). Decode parity with FP16 after PTX `cvt.rn.f16x2.e2m1x2`
vectorization (2026-05-08): Qwen3-8B Q8 147.0 tok/s NVFP4 vs 147.5 FP16.

Re-projection of the ceiling table:

| Model | FP16 cap | NVFP4 cap (3.9×) |
|---|---|---|
| Qwen3-4B Q8_0 | ~80 K | ~310 K |
| Qwen3-8B Q8_0 | ~40 K (physical) | **40 K (model-native max)**³ |
| Qwen3.6-35B Q4_K_M | 256 K (model native) | already fits |
| Gemma-4-26B Q4_K_M | 13.2 K | ~50 K |

³ Confirmed empirically: `--kv-nvfp4` lifted Qwen3-8B from cap 16 K to
its model-native maximum 40 K within the same VRAM budget, story
coherent.

### Where the wall hits

For the imp production set (Qwen3-4B, Qwen3-8B, Qwen3.6-35B, Gemma-4-26B):
**NVFP4 KV alone clears 100 K on every dense model that has a 100 K+ trained
context window** (Qwen3-4B 256 K, Qwen3.6-35B 256 K). Gemma-4 at 13.2 K → 50 K
post-NVFP4 still under-shoots its native 128 K window, but no current user
reports needing >50 K Gemma-4 in production.

The remaining wall (citing `nvfp4_kv_potential_2026_04_25` table):

- Dense Q4 models trained for ≥128 K context where 3.9× compression
  still doesn't fit (only Gemma-4-26B is in scope, and Gemma-4 has
  separate code-gen quality issues at long context — see
  `gemma4_q4km_vs_q8_2026_04_19`).
- Hypothetical >256K context (e.g. RAG over multi-million-token corpora,
  long-running agents). No imp user has asked for this regime.
- Single-GPU multi-million-token inference scenarios (research-grade).

K8 is the right tool *only for that residual wall*. For everything imp
ships today, NVFP4 KV already cleared the design intent of the roadmap
bullet.

---

## 2. The K8 design space

Three sub-problems, one strategy each for single-RTX-5090 imp:

### 2.1 What to offload

**Recommendation: entire paged blocks (16 or 32 tokens) on an LRU /
recency rule, not individual K/V tensors.**

- Block granularity matches imp's existing paged abstraction in
  `KVCache::k_ptr(layer, block_id)` / `v_ptr(layer, block_id)`. The
  pointer-into-pool design (`pool_ = single contiguous GPU alloc`) makes
  per-block H2D copies trivial: each block already has a known byte
  range (`layer_k_offset_[l]` + `block_id * layer_block_bytes_[l]`).
- Sub-block granularity (offload just K, keep V) gives 2× finer control
  but doubles the bookkeeping cardinality and forces split scatter/gather
  in the attention kernel. Not worth it for first cut.
- "SWA-eligible blocks" (Gemma-4 SWA layers) is a real opportunity but
  belongs to a separate refactor: those blocks are *already known unused
  past the window*, and the right move is to **never allocate them** in
  the first place (StreamingLLM / `evict_middle_blocks` in
  `kv_cache_manager.h:141`). Coupling K8 to SWA muddles two policies.

### 2.2 When to prefetch

**Recommendation: speculative prefetch at decode-step start, driven by
the block_table of the active sequence.**

Imp's decode is single-prompt-at-a-time on the hot path
(`max_batch_size` defaults to 1 for CLI, server batches but each
sequence has its own block_table). The block sequence read by attention
is **fully known at decode-step start** — it's `block_table[0..seq_len/block_size]`.
There is no branch-mispredict equivalent: prefetch what the kernel is
about to read.

Compare-and-contrast:

- **Compute-time on-demand (synchronous)**: stall the decode kernel
  waiting for H2D. Adds full PCIe round-trip to per-token latency.
  Unacceptable on 200 tok/s = 5 ms/token budget.
- **Speculative based on attention pattern**: needs cumulative attention
  scores (this is K5 / H2O eviction policy, separate research item).
  Overkill for K8.
- **Block-table-driven prefetch**: at decode-step start, iterate
  block_table, kick off H2D for any cold block. By the time the
  attention kernel issues its load for block N, it's resident.

### 2.3 Where to read from

**Recommendation: pinned host memory only (PCIe), no NVMe.**

- Pinned host (`PinnedAllocator` in `src/memory/pinned_allocator.h`)
  already exists for KV cache prefix save/load and other H2D paths.
  64 MiB default pool with bump + free-list, expandable to 100s of MiB.
- NVMe (cold storage) is the next tier and adds 100 µs+ latency per page
  read. At decode budget 5 ms/token that's still fine, but only matters
  for the >100 GB regime that requires multi-million-token contexts.
  Defer.
- HBM-on-CPU (Grace Hopper, MI300A) doesn't exist on the 5090 box —
  consumer Blackwell is on PCIe Gen5. Out of scope.

---

## 3. PCIe budget

PCIe Gen5 x16 = 64 GB/s theoretical, ~50 GB/s real (matches
`layer_offload.cu` observations during expert weight staging).

Per-token KV cost depends on model shape. The task prompt computed
256 KB for Qwen3.6-35B (32 × 2 × 16 × 128 × 2). The real Qwen3.6-35B
shape is 10 attention layers × 2 × 2 kv_heads × 256 head_dim × 2 bytes
**= ~20 KiB/tok at FP16, ~5 KiB/tok at NVFP4**. Updated computation:

| Model | KV B/tok FP16 | KV B/tok NVFP4 |
|---|---|---|
| Qwen3-4B Q8 (36 attn, 8 kv, 128 hd) | 144 KiB | 36 KiB |
| Qwen3-8B Q8 (36 attn, 8 kv, 128 hd) | 144 KiB | 36 KiB |
| Qwen3.6-35B Q4 (10 attn, 2 kv, 256 hd) | 20 KiB | 5 KiB |
| Gemma-4-26B Q4 (30 dual hd) | 228 KiB | 58 KiB |

At 200 tok/s steady-state decode (Qwen3.6-35B target), each new token
appends to one fresh block (write side) — that's not the H2D cost.
What costs is **reading back cold blocks the attention kernel needs**.
At 100 K context, that's 100 K / 16 = ~6 250 blocks × 20 KiB = 125 MiB
of KV per layer per sequence. If *all of that* lived host-side, at
50 GB/s the full attention sweep over cold blocks would take ~2.5 ms
per layer per token — 25 ms summed across 10 attention layers, way
over budget.

The right framing is **only the eviction tail is cold**. If "recent
4 K tokens hot, everything older cold", the cold-block read per
attention step is just the K/V the kernel actually touches: ~96 K tokens
worth × 20 KiB / (200 tok/s × 5 ms budget) = ~96 K × 20 KiB = ~2 GiB,
which at 50 GB/s = 40 ms. Still over budget on a 5 ms/token decode.

So the real PCIe budget is fine **only if the kernel doesn't need the
entire cold tail every step**. That requires one of:

1. **Attention sparsity** — H2O-style heavy-hitter eviction. But that's
   K5, which has retrieval-quality issues
   (`kv_research_grade_eval_2026_05_09` Item 3).
2. **Sliding-window models** where the attention kernel mathematically
   skips the cold tail — Gemma-4 SWA layers, Qwen-with-window, etc. K8
   becomes essentially free here because cold blocks are never read.
3. **Sub-linear retrieval** — RAG-style sparse attention over a
   recently-retrieved subset. Out of scope for K8.

**Updated conclusion**: PCIe bandwidth is **not** the bottleneck for
write-side (KV append) or for the recent-warm window read. PCIe **is**
the bottleneck if you naively pull the entire cold tail each step.
Production-viable K8 needs an attention pattern that consults a small
fraction of the cold tail per step (sliding window, H2O, retrieval).
On dense full-attention dense models, K8 will not deliver 100 K+
context at user-acceptable decode tok/s.

This is a meaningful course-correction from the task prompt's "PCIe is
not the bottleneck" framing: PCIe is fine for write-back, but reading
the cold tail every step at full-attention semantic correctness is
infeasible at 200 tok/s. K8 only pays off in attention regimes that
naturally skip the cold tail.

---

## 4. Sub-problem 1 deeper — what to offload

### Recent-N hot, everything-older cold

Static rule: `hot_blocks_per_seq = N / kv_block_size`. Last N tokens
always in VRAM, used by every attention call.

Concrete N candidates:

- N = 4 K (256 blocks at bs=16, 8 KiB block_table per seq): covers
  every recent-attention model's relevance window.
- N = 8 K: covers Gemma-4 SWA (8 192 window) end-to-end. Cold tail
  is mathematically unread.
- N = 16 K: covers `kAutoMaxSeqLenCap` — eliminates K8 entirely for
  default workloads, only fires when the user explicitly requests
  long context.

### Cold block classification options

| Option | Pros | Cons |
|---|---|---|
| LRU on per-block attention score | Tracks actual hot set | Needs attention-score tracking (K5 H2O infra); per-block reduction kernel per step |
| Static "blocks > N behind cursor" | O(1) decision, no extra kernels | Discards old-but-relevant blocks (NIAH-style retrieval fails) |
| Hybrid: H2O heavy-hitter + recent-N | Best quality | Implements K5 first |
| SWA-aware: cold = `position < cursor - window` | Mathematically lossless on SWA models | Only helps SWA archs |

**Recommendation: start with static "blocks > N behind cursor are
cold", with N user-configurable. Layer SWA-awareness on top as a free
optimization for Gemma-4-style models (since the kernel never reads
those blocks anyway, we can flag them cold without quality risk).**

H2O integration is a future option that gates on K5 shipping first.

### Bookkeeping

Per-block flag is one byte; per-block host pointer is 8 bytes. For a
100 K context at bs=16: 6 250 blocks × 9 bytes = 56 KiB per sequence.
Negligible.

The block_table layout in `KVCacheManager::seq_blocks_` is
`std::unordered_map<int, std::vector<int>>`. Adding a parallel
`std::vector<HostPtr>` (or a struct `{int block_id; void* host_ptr;}`)
is a localized edit.

---

## 5. Async prefetch pipeline

### Reusing `LayerOffloadManager` semantics

`src/memory/layer_offload.{h,cu}` already implements the exact pattern
K8 needs, applied to expert weight blocks:

- Double-buffered GPU staging slots (`slots_[2].gpu_buf`).
- Dedicated `transfer_stream` per slot (`cudaStreamCreateWithFlags(...,
  cudaStreamNonBlocking)`).
- `cudaEvent_t ready_event` per slot.
- `ensure_layer()` does `cudaStreamWaitEvent(compute_stream, ready_event)`
  to gate compute on transfer completion.
- `prefetch_layer(next)` kicks off async H2D into the inactive slot.
- Uses `cudaMemcpyWithAttributesAsync` with explicit src/dst location
  hints — the modern API.

**This is the exact pipeline K8 wants.** Replace "layer's worth of
weights" with "a KV block (or batch of cold blocks)", and the same
double-buffered prefetch + event-gate logic applies. The right move
is a parallel `KVBlockOffloadManager` that mirrors the structure,
not a rewrite of `LayerOffloadManager`.

### Per-decode-step pipeline

1. Decode step starts; engine knows `seq_id` and current `seq_len`.
2. Engine builds the list of blocks the next attention call needs
   (already done — that's the `block_table`).
3. For each cold block in that list, check if it's resident in a
   staging slot. If not, issue `cudaMemcpyAsync(d_pool_slot, h_pinned,
   block_bytes, transfer_stream)`.
4. Update the `block_table` device-side to point at the staging slot's
   address for cold blocks (cool blocks must point at staging copies,
   not the original host ptr — paged attention kernel reads device
   memory).
5. Record event after the H2D batch.
6. `cudaStreamWaitEvent(compute_stream, prefetch_event)` before
   launching attention.
7. Launch attention kernel.
8. After attention, optionally evict warm-but-not-recent blocks back
   to host to make room.

### Throwaway cost

Per task prompt: if the prefetched-but-unused block costs 256 KiB H2D,
that's ~5 µs at PCIe Gen5. Negligible. The real cost is making sure
the staging slot pool has space.

### Staging slot sizing

For a 100 K context at NVFP4 KV on Qwen3.6-35B: 6 250 blocks × ~320 B
per block per attention layer × 10 attention layers = ~20 MiB total
KV. Even if we resident only the cold-tail blocks needed for the next
attention call (say 4 K tokens worth on a sliding-window model), the
staging area is single-digit MiB. Compare to the 2× `max_layer_bytes`
that `LayerOffloadManager` already burns on expert weights (often
100s of MiB).

---

## 6. Risks

### 6.1 Pinned host memory pressure

Per task prompt math: 100 K × 256 KiB = 25 GiB pinned. With imp's actual
per-token NVFP4 footprint on Qwen3.6-35B (5 KiB/tok), 100 K context is
500 MiB pinned. Easily workable on a 64 GiB WSL2 host. Even Gemma-4
NVFP4 at 100 K is 5.8 GiB — fits.

The 25 GiB scenario the prompt hypothesizes happens for FP16 KV on
Gemma-4 at 100 K — but **that case is exactly what NVFP4 KV avoids**.
K8 is layered *on top* of Lever 2 in production, not as a replacement.

### 6.2 Block-table cardinality

A "cold flag + host pointer" doubles the per-block bookkeeping
(`seq_blocks_` entries grow from `int` to `{int, void*, bool}`).
Worst case 100 K × 6 250 blocks × 16 bytes = 100 KiB per sequence.
Trivial.

The harder bookkeeping cost is **device-side**: the paged-attention
kernel reads `d_block_tables` as a flat `int*`. To handle cold blocks
the kernel needs either (a) a parallel device-side hot/cold bitmap +
indirection table (extra load per kernel step), or (b) the engine
rewrites the block_table to point at staging slots when cold blocks
get resident.

(b) is cheaper at kernel time but requires a per-decode-step block_table
edit. With (a), the attention kernel has one extra indirection but the
block_table stays stable. Pick (b) for cache-friendliness.

### 6.3 Reattach latency

Per task prompt: 5 µs PCIe DMA per cold-block-read at 200 tok/s budget
of 5 ms/token. 0.1 % per cold-block-read. **But** at 100 K cold tail on
a full-attention model the kernel touches *thousands* of cold blocks
per token (one per relevant past block × layers). 0.1 % × 6 250 reads
× 10 layers = 6 250 % — i.e. K8 is infeasible on full-attention dense
models. See §3 — this risk is mathematically real and is why the
recommendation is to defer.

### 6.4 Quality

If the kernel does eventually read every block it would have read in
VRAM, the result is bit-identical (just slower). **No quality risk.**
The only quality risk is if K8 is composed with K5 H2O eviction, which
*does* drop blocks — but that's K5's risk, not K8's.

### 6.5 CUDA Graph compatibility

The graph executor currently captures decode steps (`engine.cpp:936`
disables graphs when MoE experts are host-offloaded, but enables them
otherwise). A naive cudaMemcpyAsync from a not-pre-known host pointer
**breaks graph capture** — the source pointer is determined per-step,
not at capture time.

Workaround: use the BitDecoding Phase 3 trick (device-resident state
buffers updated by an `advance_residual_state_kernel`, see
`kv_cache_manager.h:217`). The block_table-rewrite would need a
device-side "cold staging table" that gets populated by a tiny
copy-table kernel each step.

This is solvable but adds engineering surface. Real risk: K8 with
CUDA Graphs enabled is a multi-day spike on top of the kernel work.

### 6.6 Multi-sequence batching

Server mode batches multiple sequences. K8 needs per-sequence staging
slot pools or a global pool with eviction. The first version should
restrict to `max_batch_size == 1` to side-step this, matching the
typical long-context use case (a single very long prompt, not many
parallel short prompts).

---

## 7. Implementation plan

Five phases, each independently shippable. Phases 1–2 are bookkeeping
only and incur zero perf cost; Phases 3–5 commit to the kernel + bench
investment.

### Phase 1 — Bookkeeping plumb-through (~1 week)

- Add `cold` flag + `void* host_ptr` to per-block tracking in
  `KVCacheManager::seq_blocks_` (or a parallel data structure if
  invasive).
- Add `KVBlockOffloadManager` skeleton at
  `src/memory/kv_block_offload.{h,cu}`, mirroring
  `LayerOffloadManager`'s slot + transfer-stream + event structure.
  Initially every block is hot — no offloading happens, the manager
  is a no-op.
- Wire `KVBlockOffloadManager::ensure_block(block_id, compute_stream)`
  into the executor at the paged-attention dispatch site
  (`src/exec/executor_attention.cu`).
- Validate: zero perf change in `make verify-fast` baseline.

**Deliverable:** the plumbing is in place. Future phases can flip the
policy without touching the executor.

### Phase 2 — Static cold-block offload policy (~1 week)

- Add config knob `kv_cache.cpu_offload_hot_tokens = N` (default 0 =
  disabled). When set, blocks older than `N` tokens behind cursor are
  classified cold.
- Implement `KVBlockOffloadManager::evict_to_host(block_id)`:
  `cudaMemcpyAsync` to pinned host, free the device-side block back
  to the KV pool free list.
- Implement `KVBlockOffloadManager::ensure_block(block_id)`: on
  attention dispatch, for each block in the seq's block_table that is
  cold AND not already in a staging slot, kick off H2D. Wait on event
  before kernel launch.
- Rewrite the device-side `d_block_tables` entries for cold blocks to
  point at staging slots. Use a tiny copy-table kernel for
  graph-capture safety (Phase 3 BitDecoding trick).
- Skip CUDA Graph compatibility for this phase — capture is disabled
  when `cpu_offload_hot_tokens > 0`. Document the trade-off.

**Deliverable:** opt-in cold offload that works at `max_batch_size=1`,
no graphs.

### Phase 3 — Long-context bench (~3 days)

- A/B at 64 K / 128 K / 256 K context on:
  - Qwen3.6-35B Q4_K_M NVFP4 KV (the model that easily fits 256 K
    without K8 — should show ~0 % overhead from K8 plumbing).
  - Gemma-4-26B Q4_K_M NVFP4 KV at 128 K (the realistic K8 target —
    NVFP4 caps at ~50 K, K8 should unlock more).
  - Qwen3-4B Q8 at 256 K (already fits — overhead measurement only).
- Acceptance: K8 enables target context, decode tok/s within 30 % of
  in-VRAM baseline. Sub-30 % means PCIe is the bottleneck and K8 is
  not viable for that model.

**Deliverable:** measured perf envelope. Decision point: ship or
shelve.

### Phase 4 — Quality A/B (~3 days)

- NIAH (Needle In A Haystack) at long context: K8 enabled vs disabled
  on the same model + same prompt. Expected result: bit-identical
  decode tokens — the kernel sees the same K/V values, just via a
  H2D round trip.
- If divergence: bug, not policy issue. Fix before Phase 5.

**Deliverable:** quality parity confirmed (or bug fixed).

### Phase 5 — Default flip + CUDA Graph compatibility (~1 week, optional)

- If Phases 3-4 land green and a user has asked for >256 K context:
  flip `cpu_offload_hot_tokens` from 0 to a sensible default (e.g.
  16 384). Document trade-off in `docs/usage.md`.
- Restore CUDA Graphs compatibility via device-resident block-table
  rewrite kernel (matches BitDecoding Phase 3 graph-safety approach).

**Total wall time if all phases ship: ~3–4 weeks for one engineer.**

---

## 8. Decision recommendation

**Defer K8 for now; consider Phase 1 (bookkeeping) as the cheapest
hedge.**

Justification (one sentence): Lever 2 NVFP4 KV already gives every imp
production model 100 K+ context at parity decode tok/s — the only
residual demand is the >256 K regime, which no user has asked for, and
which would require attention-sparsity techniques (K5 H2O or SWA)
layered on top of K8 to be feasible on PCIe Gen5 bandwidth.

The cheap hedge: ship **Phase 1 (bookkeeping only)** in a future
slow-week. The block_table struct change + `KVBlockOffloadManager`
skeleton is ~200 LoC and zero kernel risk; it preserves the option to
ship Phases 2-5 in days rather than weeks once a real >256 K workload
materializes.

The expensive bet: full Phase 1-5 is multi-week work to solve a
problem nobody has reported. The combinatorial risk surface (PCIe
bottleneck on dense attention, CUDA Graph incompatibility, host pinned
memory pressure under multi-seq batching) only justifies the investment
once a concrete user need exists.

## Cross-references

- `lever2_nvfp4_kv_implemented_2026_05_07.md` — the reason K8's design
  intent is largely met for production models.
- `nvfp4_kv_potential_2026_04_25.md` — per-model KV ceiling math
  (FP16 vs NVFP4).
- `kv_research_grade_eval_2026_05_09.md` — K5 H2O evaluation; K8
  composability discussion.
- `int4_kv_chunked_prefill_2026_05_15.md` — most recent KV-related ship,
  pattern for incremental KV-dtype work.
- `gemma4_chunked_prefill_2026_05_15.md` — SWA + chunked prefill path
  that K8 would compose with cleanly on Gemma-4.
- `bitdecoding_phase3_continuation_2026_05_09.md` — residual ring +
  device-resident state buffer pattern reusable for K8 graph safety.
- `src/memory/layer_offload.{h,cu}` — exact prefetch pipeline pattern
  to mirror in `KVBlockOffloadManager`.
- `src/memory/kv_cache_manager.h:178-243` — existing per-seq ring +
  slot allocator (BitDecoding residual) provides a working template
  for per-seq device-resident state.
- `src/runtime/vram_budget.cpp:160-235` — KV budget allocation that K8
  would extend with a host-pool budget alongside `kv_cache_bytes`.
- `src/runtime/engine.cpp:1190-1279` — `init_kv_cache()` where K8
  manager would be instantiated.
