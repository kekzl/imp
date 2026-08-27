<!--
layer: L2
audience: kernel-devs
verified: 2026-08-28
commit: be825e4a
-->

# imp Memory Architecture

> **Status: Phase B in progress.** A1 is measured current state; A2-A7 are the design. Implementation follows the A7 migration order, one mergeable commit per subsystem. Divergences are recorded in [§B0 Implementation log](#b0-implementation-log), in the same commit that diverges.

Target: `sm_120a` (RTX 5090 / GB202, 32 607 MiB) with the `sm_120f` PTX fallback. No SM80/SM90 paths.

---

## 0. The defect

- No memory architecture: 380 allocation sites across 81 files; ownership implicit, lifetime by convention.
- Capacity falls out of a 480-line arithmetic ladder in `src/runtime/vram_budget.cpp` whose input is a live `cudaMemGetInfo` reading that is **wrong by ~3.9 GiB** at the moment it is taken (A1.5).
- #874, #926, #934, #963, #1100, #1103: six incidents whose fixes are stacked clamps in `vram_budget.cpp`, each correcting the previous one.

Five structural facts the design has to change:

| # | Fact | Evidence |
|---|---|---|
| F1 | Only **31 of 336** device-allocation sites route through `VRAMAllocator`. The rest call the driver directly and are invisible to it. | census, A1.1 |
| F2 | `VRAMAllocator` is **a tracker, not an owner**: its own destructor comment says so, and it enforces its headroom in three separately-defeatable ways. | `vram_allocator.{h,cu}`, A1.2 |
| F3 | Capacity is **discovered, not planned**: the KV pool is sized from live free VRAM *before* the weight caches exist, and the caches then re-derive their own budgets from live free VRAM again. | `vram_budget.cpp`, `split_pre_dequant_budget` docstring, A4.1 |
| F4 | A fixed **~3.9 GiB is claimed on the first forward pass**, after the plan is final, attributed to nothing. Invariant to batch (1→16) and context (1024→4096). | measured, A1.5 |
| F5 | `src/core/allocator.h` already contains an `ArenaAllocator` and a `PoolAllocator`. **Both are dead**: zero references anywhere in `src/ include/ tools/ tests/`. | grep, A1.6 |

---

## A1. Current-state inventory

### A1.1 Allocation-site census

Measured on the working tree (`main` + the two `#1104` constrain commits + the staged `#1103` budget fix, see A1.4). Comment-only mentions excluded.

```
380 source lines containing an allocation call, in 81 files under src/
395 individual calls (some lines carry two)
```

| API | calls | note |
|---|---:|---|
| `cudaMalloc(` | 248 | plain, untracked |
| `cudaMallocAsync` | 57 | default pool, release threshold pinned to `UINT64_MAX` |
| `cudaHostAlloc` | 49 | pinned host staging |
| `vram_alloc(` → `VRAMAllocator` | 31 | the *only* tracked path |
| `cudaMallocHost` | 6 | |
| `cudaHostRegister` | 4 | |

Device-side total: **336 sites**, of which **31 (9 %)** are tracked. `cudaFree`/`cudaFreeAsync`/`cudaFreeHost`: 422 sites. Sites outside `src/memory/`: **365, in 74 files** = the initial I1 allowlist.

By subsystem:

| dir | sites | dominant consumer |
|---|---:|---|
| `exec/` | 132 | executor workspaces, pre-dequant weight caches |
| `compute/` | 115 | per-kernel scratch, GEMM pointer arrays, sampling |
| `runtime/` | 69 | graph buffers, scheduler staging, spec-decode |
| `model/` | 21 | weight upload |
| `quant/` | 17 | quantization scratch |
| `memory/` | 15 | KV pool, SSM state, snapshots |
| `vision/` | 6 | tower weights, pixel + embedding buffers |
| `core/` | 4 | `Buffer` (1 user), `ArenaAllocator`/`PoolAllocator` (**dead**) |
| `lora/` | 1 | |

Top files by site count: `exec/executor_workspace_buffers.cu` (47), `model/weight_upload.cu` (19), `runtime/engine_prefill.cpp` + `engine_scheduler.cpp` (17 combined, split 2026-08-26), `runtime/cuda_graph.cu` (17), `compute/gemm_grouped_nvfp4_smallM.cu` (14), `runtime/engine_graph_decode.cpp` (13), `exec/pre_dequant_phase3_nvfp4_decode.cu` (13).

### A1.2 What `VRAMAllocator` actually enforces

`src/memory/vram_allocator.{h,cu}`. Headroom defeated three ways, all live:

1. `can_allocate()` returns `true` unconditionally for anything `< 16 MiB`.
2. `allocate(..., bypass_headroom = true)` skips the check entirely (used by the NVFP4 MoE cache).
3. On a headroom failure `allocate()` retries against raw free VRAM and proceeds anyway if `free >= bytes + 64 MiB`, logging a warning.

Destructor deliberately frees nothing ("The allocator is a tracker, not an owner"); every one of its 31 callers holds a raw `void*` and does its own `vram_free`.

The one hard constraint, `free >= bytes + 5 % of total` for allocations `>= 16 MiB`, was until the staged `#1103` fix **unknown to the planner**: the planner planned down to a 512 MiB reserve while the allocator refused anything leaving less than 1630 MiB free. The plan was unexecutable by construction; the starved caches failed mid-build and cost ~7x decode on gpt-oss-20b at server defaults.

### A1.3 Measured footprint: three configs

Harness: `MemAccount` (`src/memory/mem_account.{h,cu}`, gated by `diagnostics.vram_audit`), lifecycle checkpoints + per-pool notes + a 2 ms device-used peak sampler. Driver: `tools/analysis/vram_audit_load.py`, 2 rounds x N concurrent streaming completions, 0 errors, GPU healthy throughout (2857-2932 MHz SM / 13801 MHz mem / 310-444 W under load).

Card total 32 607 MiB. `00_pre_init` (CUDA primary context + WSL2/WDDM driver) is **1679.6 MiB** on all three.

| config | model | max_batch | ctx | load-time peak | steady state | peak under load | free |
|---|---|---:|---:|---:|---:|---:|---:|
| **dense** | Qwen3-4B-Instruct-2507 Q8_0 (36 L, d=2560, kv_heads=8) | 8 | 4096 | 18 226 | 18 226 | **18 416** | 14 191 |
| **MoE** | Qwen3-Coder-30B-A3B NVFP4 (48 L, d=2048, kv_heads=4) | 8 | 4096 | 23 872 | 23 872 | **24 050** | 8 557 |
| **vision** | gemma-3-4b-it Q4_K_M + mmproj-F16 (34 L, d=2560) | 4 | 4096 | 14 792 | 14 792 | **14 992** | 17 617 |

All MiB. Aggregate throughput during the load: 876 / 567 / 215 tok/s.

> **Substitution, stated:** the dispatch names Gemma-3-12B + mmproj; no 12B mmproj exists on this host (`~/models/gemma-3-4b-vl/` is the only vision pair). The vision row is gemma-3-4b + `mmproj-F16.gguf`, driven with three real image requests plus the text load. The shape of the vision finding (tower resident, `04_features` is where it lands) is size-independent.

**Peak - steady state is +190 / +178 / +200 MiB.** That is the entire steady-state per-request allocation surface: there is no transient prefill spike to cap, every workspace is statically pre-allocated (confirms the 2026-06-12 finding). The +190 MiB is the I2 violation.

### A1.4 Lifecycle phase deltas (measured, full coverage incl. raw `cudaMalloc`)

| checkpoint | dense Δ | MoE Δ | vision Δ | what it is |
|---|---:|---:|---:|---|
| `00_pre_init` | 1679.6 | 1679.6 | 1679.6 | CUDA context + WDDM driver (absolute, not Δ) |
| `01_prewarm_gemm` | +676 | +676 | +676 | `gemm_init` + `attention_cublas_prewarm` + `gemm_grouped_3x_nvfp4_prewarm` |
| `02_weights+decode_cache` | +4700 | +19962 | +3026 | weight upload + pre-dequant caches |
| `03_kv_cache` | +6990 | +1552 | +7798 | KV pool + executor workspaces + SSM state |
| `04_features` | 0 | 0 | **+1610** | vision tower (only config that has one) |
| `05_post_warmup` | **+4180** | +2 | +2 | see A1.5 |

Per-pool `note()` attribution at steady state:

| pool | dense | MoE | vision |
|---|---:|---:|---:|
| `WEIGHTS` | 4076.1 | 15467.4 | 2367.6 |
| `WEIGHT_CACHE_NVFP4` | 2157.7 | - | 263.0 |
| `WEIGHT_CACHE_CUTLASS_SF` | - | 1800.5 | 29.2 |
| `WEIGHT_CACHE_FP16` | - | - | 5185.0 |
| `KV_BLOCK_POOL` | 4608.0 | 1536.0 | 2176.0 |
| `EXEC_WORKSPACES` | 396.6 | 507.1 | 452.4 |
| **tracked total** | **11238.3** | **19311.0** | **10473.2** |
| **untracked residual** | **7177.2 (39 %)** | **4738.5 (20 %)** | **4516.4 (30 %)** |

The table above is the state that **motivated** the campaign, kept because the shape of the gap is what the fixes were aimed at. It is not the state today.

**Criterion 6 (≥95 % accounted) is met.** The pool ledger is always on rather than gated behind `--mem-report` (`AUDIT.md` B80/B81), and the library charge is measured over the whole init instead of the warmup-forward window. Re-measured 2026-08-03 with `--mem-report`, one run each:

| config | accounted | residual |
|---|---:|---:|
| Qwen3-4B-Q8_0 (dense GGUF, was 39 % untracked) | **99.9 %** | 16 MiB |
| Qwen3-30B-A3B-NVFP4 (MoE, was 20 %) | **99.9 %** | 12 MiB |
| Qwen3-8B-NVFP4 | **99.9 %** | 0 MiB |
| Qwen3-14B-NVFP4 | **99.9 %** | 0 MiB |

The 07-29 audit's F-6 was written against the old table; retired by this measurement, closed in `docs/audit/SETTLED.md`. Do not re-open it from the numbers above.

### A1.5 F4: the ~3.9 GiB first-forward claim

The dense `05_post_warmup` delta of +4180 MiB is the cost of the **first forward pass**, whenever that happens:

| probe | result |
|---|---|
| `runtime.warmup=false`, measured before any request | init ends at **14 046 MiB** (`05_post_warmup` Δ = 0) |
| same process, after **one** 32-token request | **18 234 MiB**, i.e. **+4188 MiB** |

Invariant to everything the planner knows about:

| max_batch | ctx | init used | after 1 request (smi) | Δ |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 10 566 | 14 439 | **3873** |
| 8 | 4096 | 14 046 | 17 901 | **3855** |
| 8 | 1024 | 10 268 | 14 116 | **3848** |
| 16 | 4096 | 18 676 | 22 562 | **3886** |

(`nvidia-smi` reads ~180 MiB below the in-process `cudaMemGetInfo` used; the delta is flat.)

Sub-hypotheses tested and **refuted**:

- **imp's own lazy CUDA module loading.** `CUDA_MODULE_LOADING=EAGER` moved only **+124 MiB** into `00_pre_init` (1679.6 → 1803.6) and +270 MiB into `01_prewarm_gemm`. The first-request delta stayed at +4188 MiB.
- **The default `cudaMallocAsync` pool.** `reserved`/`used` are 4096/4076 MiB before *and* after the request.
- **Scaling with batch or context.** Flat, per the table above.

What remains: CUDA/cuBLAS/CUTLASS library-internal lazy reservation claimed on first matmul dispatch. **21 % of the dense config's total footprint, unknown to the planner**: at budget time the dense run logs `available=22290.3 MiB` and hands the KV pool 4608 MiB, from a number ~3.9 GiB too optimistic. On the MoE config (8.5 GiB free at end) the same constant is 46 % of the remaining headroom.

**Design consequence:** the planner charges a measured, arch-and-driver-specific *library reservation* as a first-class line item (A4); `--mem-report` attributes it explicitly instead of dumping it into a residual (A5.3).

### A1.6 Verified-dead and verified-clean (do not re-chase)

- **`ArenaAllocator` / `PoolAllocator`** (`src/core/allocator.{h,cpp}`, 68 + 100 LOC): zero references in `src/ include/ tools/ tests/`. Dead. They implement bump-arena and fixed-block-pool disciplines, two of the five tiers this design needs, and were never wired to anything.
- **`Buffer`** (`src/core/buffer.{h,cpp}`): exactly **one** producer (`engine.cpp:699`, vision embeddings) and one holder (`Request::vision_emb`). A single-purpose helper, not a general RAII layer.
- **COW-fork / Best-of-N does not exist.** Grep for `cow|copy_on_write|fork_seq|best_of|n_best` across `src/ tools/` returns one hit, a comment in `scheduler.cpp:117` about a *hypothetical* site. The dispatch's "block with three referents" referents are **sequence block table + prefix-cache hash table + pin set** (plus the on-disk persisted cache), not a fork (A5.1).
- **No persistent GEMM autotuning state.** `src/compute/gemm.cu` contains no file I/O; the cuBLASLt algo cache is a process-local map rebuilt every start. Prefill variance across process starts is therefore not explained by persisted autotune state; the algo-selection share measured 3.50 % (the 2.6x once quoted here was a carried-forward citation, see `docs/PERF.md`). Question closed; not a memory-design input.

### A1.7 Per-site inventory by subsystem

Lifetime classes use the A2 taxonomy. "Graph" = the buffer's address is baked into a captured CUDA graph (prefill and decode are both graphified).

| subsystem | what | size formula | lifetime | graph | tracked |
|---|---|---|---|:-:|:-:|
| `model/weight_upload.cu` | model weights | Σ tensor bytes | model-resident | ✓ | note only |
| `exec/pre_dequant_phase*` | FP16 / FP8 / NVFP4 / CUTLASS-SF weight caches | ≈ `elems/2 + elems/16` (NVFP4), `elems` (FP8), `2·elems` (FP16); SF slab `Σ align256(cutlass_nvfp4_sf_size(N,K))` | model-resident | ✓ | partial |
| `memory/kv_cache.cu` | paged KV block pool | `kv_max_blocks × kv_block_bytes_per_layer(dtype,bs,kv_heads,hd) × n_kv_layers` | engine-persistent | ✓ | note |
| `memory/kv_cache_manager` | residual FP16 ring | `max_seqs × n_layers × 2 × residual_n × kv_heads × hd × 2 B` | engine-persistent | ✓ | via `VRAMAllocator` |
| `memory/ssm_state.cu` | SSM/GDN conv + h state | `n_ssm × max_batch × (conv_ch·(k−1)·4 + heads·hd·state·dtype)` | engine-persistent | ✓ | via `VRAMAllocator` |
| `memory/recurrent_snapshot_store` | hybrid prefix snapshots | `entries × ssm_slab_bytes` (LRU-bounded) | engine-persistent | ✗ | ✗ |
| `exec/executor_workspace*` | persistent workspace | `max_tokens·d_model·2 B·3 + max_tokens·vocab·4 B` | engine-persistent | ✓ | via `VRAMAllocator` |
| `exec/executor_workspace*` | shared workspace | `max(attn, ffn, moe, ssm)` shared sizes at `max_tokens` | engine-persistent | ✓ | via `VRAMAllocator` |
| `exec/executor_workspace*` | decode workspace (2nd copy) | same, at `max_batch` | engine-persistent | ✓ | via `VRAMAllocator` |
| `exec/executor_workspace_buffers.cu` | `attn_scores` S-matrix | `attention.attn_scores_mib` (default 384 MiB) | engine-persistent | ✓ | via `VRAMAllocator` |
| `exec/executor_workspace_buffers.cu` | MLA QKV scratch (4 buffers) | `max_tokens × {kv_a, latent, k_rope, kv_b} × 2 B` | engine-persistent | ✓ | ✗ (raw) |
| `exec/moe_workspace.cu`, `expert_cache.cu` | MoE dequant / staging / expert cache | `max_expert_raw`, `expanded × d_ff` | engine-persistent | ✓ | partial |
| `compute/gemm.cu` | cuBLASLt workspace | 64 MiB (ladder 64/32/8/2) | engine-persistent | n/a | ✗ (static) |
| `compute/gemm.cu` | algo-bench scratch | 32 MiB fixed | engine-persistent | n/a | ✗ (static) |
| `compute/gemm_cutlass_sm120.cu` | CUTLASS fallback workspace | `GemmT::get_workspace_size(M,N,K)`, **grown lazily at GEMM time** | engine-persistent | ✗ | ✗ (static) |
| `runtime/engine_graph_decode.cpp` | block tables, banned-token list | `max_blocks_per_seq × 4 B`, `n_banned × 4 B` | **per-request, `cudaMallocAsync`** | ✓ | ✗ |
| `runtime/engine_prefill.cpp` (split from engine_scheduler.cpp 2026-08-26) | prefill metadata (token ids, positions, block tables, ctx lens) | `chunk_len × 4 B` etc., pooled when it fits, else `cudaMallocAsync` | **per-request** | ✓ | ✗ |
| `runtime/engine_spec_*` | draft/verify staging + `spec_graphs_` | `k_max × …` per bucket | engine-persistent (invalidated) | ✓ | ✗ |
| `vision/` | tower weights, pixel buffer, embedding buffer | `Σ tower tensors`; `image_size²·3·2 B` (mmproj) or `vision_max_patches × features × 2 B` (Qwen3-VL); `num_image_tokens × d_model × 2 B`, plus one such buffer per DeepStack tap | model-resident | ✗ | via `VRAMAllocator` |
| `memory/layer_offload.cu` | double-buffered H2D layer staging | `2 × max_layer_bytes` | engine-persistent | ✗ | ✗ |
| `memory/weight_snapshot`, `weight_cache_file` | suspend/resume + warm-cache staging | host-side only | transient host-staging | ✗ | n/a |
| - | **library reservation (F4)** | **~3.9 GiB, constant** | engine-persistent | n/a | ✗ |

---

## A2. Lifetime taxonomy

Five tiers. **Each tier gets exactly one allocator whose discipline makes that tier's failure mode structurally impossible.**

| Tier | Lifetime | Allocator | Failure mode made impossible | Address stability |
|---|---|---|---|---|
| **T1 Model-resident** | model load → unload | bump arena, freed wholesale | per-object leak (nothing is individually freed) | stable |
| **T2 Engine-persistent** | process | bump arena | per-object leak | stable |
| **T3 Pooled fixed-block** | request-scoped, refcounted | free-list over one slab | external fragmentation (all blocks identical) | stable |
| **T4 Forward-scratch** | one forward pass | LIFO stack | fragmentation (LIFO cannot fragment) + leak (stack unwinds) | stable per slot |
| **T5a Transient host-staging** | load only | ordinary host alloc | surviving load (asserted at phase transition) | n/a |
| **T5b Engine-persistent pinned host** | process | `PinnedBuffer` over `HostPinnedAllocator`; `HostRegistration` for memory imp does not own | per-object leak (move-only owner, no free to forget); asymmetric unregister | n/a |

Confirmed against A1.7. Three corrections the inventory forced:

- **T5 splits.** 26 pinned-host acquisition sites are engine-persistent by construction (e.g. the per-step D2H gather staging buffer is pinned once and reused every decode step), so they must survive into `AllocPhase::Serving`. T5a keeps the "load only, asserted at phase transition" discipline; T5b is the tier those 26 sites move to, owned by `PinnedBuffer` (move-only, frees exactly once, empty-on-failure so degradation paths still work). `Backend` covers device memory only.
- **T1 and T2 need separate arenas.** `server.model_swap` (shipped 2026-07-26) unloads a model and loads another without a process restart. T1 must be releasable wholesale at swap; T2 (KV pool geometry, cuBLAS workspace, graph buffers) must survive it or be torn down explicitly.
- **T3 is a group of pools, not one.** The KV cache runs two block groups (global + SWA); the residual FP16 ring, SSM state and recurrent snapshots are all fixed-stride slabs with per-sequence slots. One `BlockPool<Stride>` template, four instantiations.

No tier for the library reservation (F4): it is not imp's memory, it is a **charge** the planner subtracts before distributing anything (A4).

One entry straddles tiers: the per-request `cudaMallocAsync` traffic in `engine_graph_decode.cpp` / `engine_scheduler.cpp` is logically T4 but implemented as driver calls on the hot path. That is the I2 violation; moving it to T4 is the fix.

---

## A3. Layer design

Three layers. Each states what it is *not* responsible for.

```
   ┌──────────────────────────────────────────────────────────────┐
   │ L3  Handles          typed RAII ownership, stability in the  │
   │                      type system                             │
   │     NOT: sizing, policy, physical acquisition                │
   ├──────────────────────────────────────────────────────────────┤
   │ L2  Allocators       one per lifetime tier (A2)              │
   │     NOT: talking to the driver, deciding how much            │
   ├──────────────────────────────────────────────────────────────┤
   │ L1  Backend          physical acquisition, phase guard,      │
   │                      accounting                              │
   │     NOT: lifetime, tiering, policy                           │
   └──────────────────────────────────────────────────────────────┘
```

All of it lives in `src/memory/`. Nothing above L1 calls the driver.

### A3.1 L1 - Backend: VMM or `cudaMalloc`?

**Recommendation: `cudaMalloc` backend for every tier now; a VMM backend for the KV block pool only, in migration step 7, gated on a WSL2 spike. No VMM anywhere else.**

*Against VMM everywhere:* peak - steady state is +190 / +178 / +200 MiB (A1.3); imp allocates once at init and holds until teardown, so there is no churn to fragment. VMM's costs (2 MiB minimum granularity, explicit `cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`/`cuMemSetAccess` lifecycle, handle bookkeeping, different behaviour under WDDM) are a bad trade against a 190 MiB problem.

**Superseded 2026-07-31 (B84): the defect this section describes no longer exists.** 6.4 reordered the build so the caches come first and KV takes the measured residual; 6.9 replaced the wrong free-VRAM reading with a measured library charge (attribution ~100 %). The pool grants the full requested context at 4k, 32k and 128k; holding a 128k pool at 81 MiB free costs 1 % of decode. The reasoning below stands as the record of why VMM looked necessary.

*For VMM on the KV pool:* the pool must be sized **before** the weight caches are built, from a free-VRAM reading wrong by ~3.9 GiB (F3, F4). With VMM the guess disappears:

- `cuMemAddressReserve` the VA range for the **maximum** KV the config could want (`max_batch × max_seq_len`); reserving address space costs no physical memory.
- Commit physical pages in coarse chunks (64-256 MiB, multiples of the 2 MiB granularity) as the free-list runs dry.
- Decommit chunks when a chunk's blocks are all free and the pool has been under-subscribed for N steps.

The pool cannot be mis-sized because it is no longer sized: bounded by the reservation, backed on demand. The planner's job shrinks from "predict the residual exactly" to "prove the maximum commitment fits". The `kv_max_blocks` clamp ladder (`vram_budget.cpp:455-539`) collapses to one check. Addresses stay stable under VMM, so I3 holds and graph-captured KV pointers remain valid across growth; plain `cudaMalloc` growth cannot offer that.

*Why step 7 and not step 1:*

1. **`cuMemCreate`/`cuMemMap` under WSL2/WDDM: measured 2026-07-29, gate OPEN.** `tools/analysis/vmm_wsl2_probe.cu`, 24/24 checks, two reproducible runs on the RTX 5090 (driver 13030, `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1`):

   | property | result |
   |---|---|
   | 24 GiB `cuMemAddressReserve` | succeeds in 0.4-1.0 ms, **+0.000 MiB** physical |
   | commit in 256 MiB granules | exactly `+256.00 MiB` each, no overhead |
   | base address across grow/shrink (8→2→6→4→10 chunks) | **invariant**, `0x2000000000` at every step |
   | data in another region across decommit/recommit | intact, 0 / 67 108 864 words wrong |
   | **graph-captured kernel over a fixed VA, across +1.5 GiB growth** | **same checksum three times, no re-instantiate** |
   | full decommit | free VRAM returns to baseline, **+0.00 MiB** residual |

   Two numbers change the design:
   - **Granularity is 2 MiB** (minimum and recommended identical on sm_120a), not 64 KiB. A commit chunk must be a multiple of 2 MiB; 256 MiB = 128 granules.
   - **Decommit costs ~2x commit**: commit 256 MiB ~1.2 ms mean, decommit ~2.4-2.6 ms. The decommit policy must be generous and hysteretic or flapping costs more than the memory it returns. A growth event is ~1.2 ms, a third of a decode step at ~280 tok/s, only every few hundred steps.

   No WDDM tax of the 165 µs `memcpyAsync` kind appeared.

2. **The tiers above must exist first**, or there is nothing to grow into.

*Not considered:* `cudaMallocAsync` pools as the backend. imp pins the default pool's release threshold to `UINT64_MAX` (a de-facto arena without an arena's guarantees), and its reserved-vs-used split is a recurring accounting-confusion source (A1.4 residuals). Kept only for the weight-upload path during migration, then retired.

**Backend interface:**

```cpp
// src/memory/backend.h - the ONLY place in imp that calls the driver.
class Backend {
public:
    virtual ~Backend() = default;
    // Physical acquisition. Fails cleanly; never throws, never aborts.
    virtual std::expected<Region, MemError> acquire(size_t bytes,
                                                    Alignment a,
                                                    RegionTag tag) = 0;
    virtual void release(Region&&) = 0;
    // Growable regions (VMM backend only; CudaMallocBackend returns
    // MemError::NotGrowable).
    virtual std::expected<void, MemError> commit(Region&, size_t new_bytes) = 0;
    virtual void decommit(Region&, size_t new_bytes) = 0;
    virtual BackendStats stats() const = 0;
};
```

`Region` is `{void* base; size_t committed; size_t reserved; RegionTag tag;}`: move-only, and the *only* type in imp that holds a raw device pointer obtained from the driver.

### A3.2 L1 - the phase guard (I2)

```cpp
enum class AllocPhase { Loading, Planning, Serving };
```

Process-global, monotonic, set by the engine. Every entry point that acquires physical memory consults it: `acquire()`, `acquire_growable()`, and since #1649 `commit()` and `commit_range()` as well.

- `Loading` / `Planning`: allowed.
- `Serving`: **debug:** `IMP_ASSERT_FAIL` with the tag and a backtrace. **release:** increment `steady_state_allocations_total{tag}`, log once per tag at WARN, proceed (never crash a production server over an accounting bug).

The counter is the I2 test surface: acceptance criterion 3 is `steady_state_allocations_total == 0` after a soak. Also the migration progress bar: starts at ~190 MiB worth of allocations per config, must reach zero.

One deliberate exception: `Serving` is temporarily re-entered as `Planning` during `server.model_swap`, bracketed and logged.

**Growth is an acquisition** (#1649). `commit()` and `commit_range()` are non-virtual wrappers around `do_commit()` / `do_commit_range()`, same reason `acquire()` wraps `do_acquire()`: a backend cannot forget the guard. Before #1649 a growable KV pool committing pages under load was counted by none of the three instruments: not the phase counter, not the `--wrap` interposer (wraps `cudaMalloc*`, the VMM backend calls `cuMemCreate`/`cuMemMap`), not `check_alloc_sites.py` (scans for driver APIs, and the call is inside `src/memory/`).

The guard runs **after** the call, on the delta the backend actually committed: a range may be partly mapped already, and `commit(new_total)` is a target, not an amount. Shrinking is not counted; it hands memory back.

### A3.3 L2 - Allocators, one per tier

```cpp
class ArenaAllocator;   // T1, T2 - bump; reset() frees wholesale
template <size_t Stride> class BlockPool;  // T3 - free-list over one slab
class ScratchStack;     // T4 - LIFO; scope guard restores the mark
```

`ArenaAllocator` and `BlockPool` exist in `src/core/allocator.h` (A1.6, dead). They move to `src/memory/`, get the Backend as acquisition source and the handle types below.

`ScratchStack` is new:

```cpp
class ScratchStack {
public:
    class Mark {                       // RAII; dtor rewinds
        ~Mark() { stack_->rewind(off_); }
    };
    [[nodiscard]] Mark mark();
    StableSpan<std::byte> take(size_t bytes, Alignment);  // nullptr if exhausted
};
```

A forward pass opens one `Mark` at entry; every intermediate takes from the stack; the `Mark` destructor rewinds. Cannot fragment (LIFO), cannot leak (unwinds on exception too). Its high-water mark is the number the planner needs, so the planner sizes it from a **measured** warmup high-water mark rather than from `max(attn, ffn, moe, ssm)` heuristics recomputed in three places.

### A3.4 L3 - Handles: making the illegal states unrepresentable

The load-bearing type distinction is I3.

```cpp
// Non-owning view. May point at anything. Cheap, copyable.
template <class T> class DeviceSpan {
    T* p_; size_t n_;
public:
    T* data() const; size_t size() const;
};

// Non-owning view whose address is guaranteed stable for the lifetime of the
// region it came from. Constructible ONLY by the tier allocators that can make
// that promise (T1-T4). Friend-restricted ctor; no public ctor from T*.
template <class T> class StableSpan {
    T* p_; size_t n_;
    StableSpan(T*, size_t);                       // private
    friend class ArenaAllocator;
    template <size_t S> friend class BlockPool;
    friend class ScratchStack;
public:
    operator DeviceSpan<T>() const;               // widening: always OK
    // NOTE: there is deliberately no DeviceSpan -> StableSpan conversion,
    // no StableSpan(T*) ctor, and no as_stable() escape hatch.
};
```

Every graph-capturable kernel launch wrapper takes `StableSpan`:

```cpp
// exec/: signature enforces I3 at the call site.
void launch_paged_attention_decode(StableSpan<const half> q,
                                   StableSpan<const int>  block_table,
                                   StableSpan<half>       out,
                                   cudaStream_t);
```

A relocatable buffer is a `DeviceSpan`; passing it where a `StableSpan` is expected does not compile. The tier allocators are the only producers of `StableSpan` because the promise is theirs to make.

Ownership:

```cpp
template <class T, Tier Ti> class Owned {   // move-only RAII
    ~Owned();                               // returns to its allocator
    StableSpan<T> span() const;             // T1..T4 only
};
```

`Owned<T, Tier::ModelResident>` is not convertible to `Owned<T, Tier::EnginePersistent>`: a subsystem cannot smuggle a model-lifetime buffer into engine-lifetime storage, so `server.model_swap` cannot leave a dangling pointer behind.

Request-scoped blocks:

```cpp
class BlockRef {                     // move-only; NOT copyable
    KVBlockId id_; BlockPool<kKVBlockBytes>* pool_;
public:
    ~BlockRef();                     // dec_ref, exactly once
    BlockRef share() const;          // explicit inc_ref - the ONLY way to alias
};
```

"A request-scoped block cannot outlive its request" is not compile-time enforceable in C++ without lifetime annotations. What the design does enforce:

- `BlockRef` is move-only: an accidental copy is a compile error, every additional referent is a visible, greppable `share()` call.
- Every `BlockRef` a request owns lives in that request's `SequenceSlot`; the slot destructor asserts (debug) / counts (release) that its refcount contribution nets to zero.
- Acceptance criterion 4 (post-drain live blocks return to post-load baseline ±1 %) is the soak-time test for "no block outlived its request".

**Criterion 4 is pool-level, not device-level, and that is a platform fact** (AUDIT B36). Measured: after one load → generate → free cycle, every CUDA-level release succeeds (async pool trims to `reserved 0 / used 0`, graph memory zero) and `cudaMemGetInfo` still drops by the model's full footprint and never recovers for the life of the process. Release threshold zero + re-trim returns nothing. WSL2/WDDM does not hand a process's peak VRAM commitment back. So "live blocks return to baseline" is what the soak checks; "device-used returns to baseline" is not achievable here by any allocator design, and a `cudaMalloc` probe cannot prove otherwise because the driver oversubscribes into host memory and returns success (G18; time it instead: ~1530 GB/s resident vs ~237 GB/s spilled).

This is also the sharpest argument for **I4**: within a process, free VRAM only ever decreases; any capacity decision read from `cudaMemGetInfo` measures a moving floor.

---

## A4. The planner

### A4.1 What is wrong today

`compute_vram_budget()` (`src/runtime/vram_budget.cpp`, 480 LOC) is "pure computation" per its header, but its dominant input is `free_vram`, a live `cudaMemGetInfo` reading, and it is called **after** the weights are uploaded and **before** the weight caches are built. So:

1. It sizes the KV pool from a free-VRAM number that still contains ~3.9 GiB of library reservation claimed later (F4).
2. The pre-dequant phases then re-derive **their own** reserves from live free VRAM again. That is #1100: "the KV pool is allocated before the cache build, so its bytes are already gone from free_vram, and every one of them then came out of the decode cache a second time."
3. The engine works around the ordering with a **balloon**: a physical `cudaMalloc` held across `init_weights` purely to hide bytes from the KV planner, released just before phase 3 (`engine_weight_upload.cpp:325`, `engine_kv_cache_init.cpp:434`).
4. Six incident-driven clamps stack on the result: `target_blocks`, the post-weight `max_fit_blocks` backstop, `min_kv_blocks`, the `kv_fraction` affordability cap, the SWA batch-shaped charge, and the `#1103` allocator-headroom floor.

### A4.2 The replacement

```cpp
struct PlanInput {
    ModelShape        model;        // layers, dims, per-tensor byte demand
    FeatureSet        features;     // spec decode, vision, SWA, residual ring, LoRA
    ConcurrencyLimits limits;       // max_batch, max_seq_len, kv dtype/block size
    size_t            budget_bytes; // --vram-budget, or device total
    LibraryReserve    library;      // A1.5 - measured constant, not a guess
};

struct MemoryPlan {
    size_t model_resident;       // T1 arena
    size_t engine_persistent;    // T2 arena
    ScratchSizes scratch;        // T4 high-water, per phase
    KvPlan kv;                   // block count / VA reservation / commit chunk
    std::vector<PoolPlan> pools; // SWA group, residual ring, SSM state, snapshots
    size_t library_reserve;
    size_t total() const;
};

std::expected<MemoryPlan, PlanFailure> plan_memory(const PlanInput&);
```

Three properties the current code does not have:

- **`plan_memory` never calls `cudaMemGetInfo`.** Only capacity input is `budget_bytes`. Fully testable on the host with no GPU; same config yields the same plan on every boot, ending the "free VRAM before weight upload swings by 1.6 GB between identical invocations → different auto-batch → different KV clamp" trap recorded against #1103.
- **Allocation follows the plan in tier order**, KV allocated from a **computed** residual: `library reserve → T2 engine-persistent → T1 model-resident + weight caches → T3 pools → KV`. The balloon is deleted, not fixed.
- **It fails at load time with a report**, never mid-generation.

### A4.3 `--vram-budget` and the failure report

`--vram-budget` today is a *sizing view* (`src/memory/vram_query.{h,cpp}`): it rewrites what `cudaMemGetInfo` returns so the heuristics size smaller; its own header calls it "best-effort hard cap, not an OS limit ... leave ~1 GiB of real headroom." Under the new planner, `budget_bytes` is the plan's total, the plan fits it or fails, and `Backend` refuses any acquisition that would exceed it. The rewriting view is retired with the heuristics that needed it.

`PlanFailure` carries the full arithmetic; the operator-facing message says what to change:

```
Cannot fit this configuration in the 32607 MiB budget.

  requested                             MiB
    model weights                    15467
    NVFP4 decode cache + SF slab      1800
    KV pool (batch 8 x 4096 tok)     12288
    executor scratch (high-water)      507
    engine-persistent                  310
    CUDA context + driver             1680
    library reserve (measured)        3900
                                     -----
    total                            35952   over by 3345 MiB

  the three largest levers
    runtime.max_seq_len 4096 -> 2048        frees 6144 MiB
    kv_cache.dtype f16 -> fp8               frees 6144 MiB
    runtime.max_batch_size 8 -> 6           frees 3072 MiB
```

Not "VRAM budget: KV clamped 25600 -> 512 blocks", which is what the log says today after the fact.

---

## A5. Subsystem boundaries

| subsystem | may hold | must request | must never touch |
|---|---|---|---|
| `compute/` | nothing | all buffers as `StableSpan`/`DeviceSpan` parameters | any allocation API; any static workspace |
| `exec/` | `Owned<_, EnginePersistent>` workspace handles; a `ScratchStack::Mark` per forward | scratch from the stack | driver calls; `cudaMallocAsync` |
| `model/` | `Owned<_, ModelResident>` weight handles | the T1 arena | KV, workspaces, driver calls |
| `quant/` | `Owned<_, ModelResident>` cache handles | the T1 arena, budgeted by the plan | live free-VRAM queries |
| `graph` (`runtime/cuda_graph.*`, `engine_graph_decode.cpp`) | graph + exec objects | `StableSpan` for everything captured | any allocation inside a capture region |
| `runtime/` | the plan, the allocators, the phase | - | per-request driver allocation |
| `vision/` | `Owned<_, ModelResident>` tower + `Owned<_, EnginePersistent>` staging | T1 + T2 | KV, executor workspaces |
| `api/` | nothing device-side | - | everything |

### A5.1 Paged KV + prefix cache + pinning - who owns a block

> **Corrected 2026-07-29 during step 3 (B3).** This section originally said a block has three refcount holders (sequence, prefix cache, pin set). Wrong: there are **two** holders; the pin set is an eviction *policy* overlay, not an owner. Verified: `kv_cache_manager.cpp` contains exactly **one** `cache_->inc_ref` call site (line 536, prefix reuse) and no pin path touches the KV refcount.

A KV block has **two refcount holders** (COW-fork does not exist, A1.6):

1. **`seq_blocks_[seq_id]`**: an active sequence's positional block table. Takes a reference on allocation, drops it in `free_sequence()`.
2. **The cached LRU** (`cached_blocks_lru_` + `block_hash_to_id_`): the content-addressed prefix cache. Holds no reference of its own; it survives because `free_sequence()` *deliberately skips* the free, leaving the count at 1. That is the defect: liveness by omission rather than by ownership.

Not a holder, though it behaves like a referent:

3. **`pinned_blocks_` / `pin_refcount_`**: agentic prefix pinning. Its counts are *pin-owner* counts, not KV refcounts. A pinned block stays alive because it sits in the cached LRU at count 1; pinning only makes `reclaim_cached_block()` rotate past it. The pin set migrates as **policy**, not ownership.

Plus one out-of-process referent: `save_prefix_cache()` / `load_prefix_cache()` serialise blocks to disk and re-register hashes on load.

**Ownership rule: the `BlockPool` owns the memory; nobody else does. Both holders hold `BlockRef`s.** A block returns to the free list when and only when its last `BlockRef` is destroyed. No path frees a block by id.

Concretely:

- `seq_blocks_` becomes `std::vector<BlockRef>`; `free_sequence()` clears the vector, the refs drop.
- The prefix cache holds its own `BlockRef` per entry. `evict_cached_block()` drops that ref; a block still used by a sequence is untouched. Today this is a manual refcount plus `free_block_dropping_stale_hash`, whose comment documents the double-ownership bug it prevents: "a stale hash->id entry on a free-listed block lets a later prefix match inc_ref a block the allocator still hands out".
- The pin set holds `BlockRef`s keyed by owner. `unpin_prefix()` drops them; budget eviction drops the front owner's refs. Pins surviving `free_sequence()` is just another live reference, not a special case.

**Cancellation, disconnect, error paths** leak refcounts today because each path frees by hand. Under `BlockRef` they are one path: the `SequenceSlot` is destroyed and its refs unwind, including on exception. The slot destructor asserts its net contribution is zero (A3.4); criterion 4's cancellation-heavy soak is the system-level proof.

**Nuance the design must not lose:** `evict_middle_blocks()` (StreamingLLM) replaces freed slots with sentinel `-1` while keeping the table *length*; the attention kernels depend on positional alignment. So `seq_blocks_` is `std::vector<std::optional<BlockRef>>` and the sentinel is `nullopt`. Same for the SWA positional table (documented `-1` holes by design).

### A5.2 CUDA graph pool

Current state, verified:

| pool | keyed by | bound |
|---|---|---|
| `decode_graph_pool_[64]` | `n_sequences − 1` | `kMaxGraphPoolSize = 64`, fixed array |
| `prefill_graph_runner_` | - | 1 |
| `async_graph_runner_` | - | 1 (conditional-node loop) |
| `spec_graphs_` | `std::tuple<n_tokens, ctx_capacity, rec_slot>` | **no explicit cap**, all three axes bucketed; cleared wholesale by `free_spec_graphs_()` |

`spec_graphs_` bucketing: `n_tokens` to 3-5 draft buckets (`spec_capture_bucket_`), `ctx_capacity` to power-of-two tiers from 4096 up to `speculative.capture_ctx_cap` (`spec_capture_ctx_tier_`, ~6 tiers), `rec_slot` to `max_batch + 1`. Worst case ~5 × 6 × (max_batch+1) graph execs, ~1950 at `--max-batch 64`. Bounded but **uncounted**: nothing in the plan charges the graph memory those execs hold. The design gives it a plan-derived LRU capacity and counts it.

Interaction with I3: everything a graph captures is a `StableSpan`, so it comes from T1-T4, all arena- or pool-backed and never moving. The `workspace_generation` invalidation hook stays as a belt-and-braces assert; it should never fire.

Interaction with `cudaDeviceGraphMemTrim`: **measured 2026-07-29, already zero.** `cudaDeviceGetGraphMemAttribute` reports `used=0.0 / reserved=0.0 / high_since_serving=0.0 MiB` after 12 serving requests on the dense config, so no captured region allocates today. Consequences, settled: the trim calls at `cuda_graph.cu:366,1291` are dead code (pure deletion, not a step-5 deliverable), and graph memory is **not** part of the 20-39 % residual (AUDIT B26/B27).

### A5.3 cuBLAS / CUTLASS workspaces

Current: three file-scope statics. `gemm.cu:s_workspace` (64 MiB via a 64/32/8/2 try-down ladder), `gemm.cu:s_bench_scratch` (32 MiB), `gemm_cutlass_sm120.cu:s_cutlass_workspace` (0, grown lazily inside `gemm_nvfp4_cutlass_sm120_impl` at GEMM time: a `cudaFree` + `cudaMalloc` pair on a path that can run under graph capture).

**Design: shared from the T2 arena, sized by the plan, per-process; not per-handle and not per-stream.** cuBLASLt takes a workspace pointer + size as an argument, so one arena slice sized at the plan's maximum serves every call; per-handle multiplies by handle count for no benefit; per-stream is meaningless with one compute stream plus a prefill stream.

The lazy CUTLASS growth path is deleted. `gemm_nvfp4_cutlass_sm120_workspace(M, N, K)` already pre-sizes `qscratch_.cutlass_workspace`; the planner calls it over the model's shape set and takes the max. A shape exceeding the plan fails cleanly and falls back; it does not allocate.

**Implemented 2026-07-31 (B73); the growth path was unreachable, not merely undesirable.** Every in-tree caller sizes with the same `..._workspace()` it passes; the one caller passing `nullptr, 0` (FP32 LM head at `executor_forward.cu` / `executor_perplexity.cu`) needs zero: these kernel configurations ask for **0 bytes at every shape measured**, pinned by `CutlassWorkspaceContract`. Same commit moved `gemm.cu`'s two statics to T2 and re-sized the **grouped** path: the 512 MiB reservation was guesswork against a measured 152 320 B (170 SMs x 896 B persistent-scheduler state, invariant across expert count, N, K and prefill length); it now takes 1 MiB, freeing 488 MiB of resident VRAM on every MoE model. The grouped path keeps a growth path, safe because **a bump-arena take is pointer arithmetic, not a CUDA call, so it is legal under stream capture.**

**Prefill variance:** not explained by persistent autotuning state; there is none (A1.6, zero file I/O in `gemm.cu`). Question closed, not a design input. (The dispatch quoted 2.6x; retracted, see `docs/PERF.md`.)

**The measurement window, not the charge, is what varies (B79).** The reserve reads 0 / 2 / 4182 / 7460 MiB across configs because `report_library_reserve()` anchors immediately before the warmup forward, and which library claims before that point depends on the model's execution path: the NVFP4 cache build runs CUTLASS two phases earlier. On 30B-A3B-NVFP4 the unmeasured remainder is the 2239 MiB residual, confirmed by the invariance A1.5 defines the charge by (identical at batch 1 and 8, +27.7 MiB across a 4x context). Fix: an earlier anchor; it changes what the plan charges on every model, so it is its own pass.

**Not covered here: the ~3.9 GiB library reservation (F4).** Claimed by the libraries themselves on first dispatch. The plan charges it (A4.2), `--mem-report` names it, reducing it is out of scope.

### A5.4 Vision tower

Currently **resident**: `VisionPipeline::init()` runs during `init_features()`/warmup whenever `--mmproj` is given, loading the tower and pre-allocating pixel + embedding buffers through `VRAMAllocator`. Measured cost on the gemma-3-4b pair: **+1610 MiB** at `04_features` (A1.4), for a server that may never receive an image.

Qwen3-VL (2026-07-31): tower ships *inside* the checkpoint, so presence is the signal, no `--mmproj` to withhold. Operator knob: `runtime.vision_max_patches` (default 4096 ≈ 1024x1024) sets the image-token budget every encoder workspace is sized from; a hard ceiling, so a larger image is scaled down rather than refused. DeepStack adds one embedding buffer per tap (`engine_qwen3vl.cpp`) on top of the merged one. The tower is not measured here yet; that number belongs in this table when it is.

**Design: keep it resident, as a planned, declared T1 cost.** Lazy loading rejected: the tower is model-resident weights, so a first-image request would allocate ~1.6 GiB *while serving*, violating I2, and would need admission control for a memory event unrelated to request size. On the mmproj path, not passing `--mmproj` is the switch; Qwen3-VL has no such switch.

One hot-path hole to fix: `vision_pipeline.cpp:97` falls back to a raw `cudaMalloc` per image if the pre-allocated pixel buffer was too small. Becomes a `ScratchStack` take that fails cleanly.

### A5.5 Speculative decoding

Draft/verify staging buffers and `spec_graphs_` are T2, sized by the plan from `speculative.k` / `suffix_k_max` / MTP depth. Already invalidated together (`free_spec_buffers_` → `free_spec_graphs_`), which is the right coupling.

**Per-request toggling** (`spec-ngram: gates failed ...`: spec disabled per request by sampling params) must not resize anything: buffers are planned for the maximum `k` the config allows and simply unused when a request does not qualify. Already the behaviour; the design states it as an invariant so nobody "optimises" it into a per-request allocation later.

---

## A6. Testability

`Backend` is the substitution seam. `FakeBackend` allocates host memory (`std::aligned_alloc`) and hands out the same `Region` type, so every allocator, the planner, and the refcount logic run on CPU-only CI (no GPU runner; the CI lane is `ctest -L unit`).

`FakeBackend` provides:

- Configurable capacity: budget-exhaustion paths testable without a 32 GiB card.
- Full allocation journal: `(seq, phase, tag, bytes, op)` for every acquire / release / commit / decommit.
- Poison-on-release (`0xDE`): use-after-free becomes a deterministic data comparison, not a GPU fault.
- Injectable failure (fail the *n*-th acquisition), exercising the rollback paths that today are hand-written per call site (`rollback_partial_allocation`, the `moe_3x_packed`/`sf` unwind in `executor_workspace_buffers.cu:641`, the KV `allocate_blocks` rollback).
- Growth simulation for the VMM path: `commit`/`decommit` succeed or fail on command; the fake asserts the base address never changes (host-side proof of I3 under growth).

Invariants asserted against it:

| # | Invariant | Test shape |
|---|---|---|
| V1 | Conservation | journal replay: Σ acquired − Σ released == live bytes, after every op |
| V2 | No allocation in `Serving` | drive a synthetic decode loop; assert the journal has no acquire with `phase == Serving` |
| V3 | Arena resets free wholesale | after `reset()`, live bytes attributable to that arena == 0 |
| V4 | Block pool conservation | randomised alloc/free/share/evict/pin sequence; free_count + live refs == num_blocks, always |
| V5 | Refcount balance under faults | inject an exception at every point in a request's lifecycle; assert net block refcount delta == 0 for each |
| V6 | LIFO discipline | `ScratchStack` marks are asserted to rewind in reverse order; out-of-order rewind is a hard failure |
| V7 | Plan determinism | `plan_memory` is a pure function: same input → byte-identical plan, 1000 randomised configs |
| V8 | Plan sufficiency | replay a recorded real allocation journal against the plan; assert no tier is exceeded |
| V9 | Stability under growth | VMM fake: commit/decommit across a 10× growth; assert `Region::base` is invariant |

V8 makes the migration safe: record the journal from a real GPU run once per model config, check it in, assert the planner covers it. An under-provisioned plan is then a CI failure, not a production OOM.

Acceptance criterion 8 (peak VRAM per model config vs checked-in thresholds): **built 2026-07-29, with one design change.** The gate lives in `scripts/verify.sh` next to the perf gate (CI has no GPU runner) and the threshold in the existing `tests/perf_baseline.json` (`metrics.memory_mb.own_peak_mb` vs `thresholds.vram_increase_pct`) rather than a new file; that file carried both fields since it was written and nothing read either, so criterion 8 was structurally unmeetable while looking provided-for (B39). It gates **`own_peak`, not the `--mem-report` device total**: device-used carries the CUDA primary context and any neighbour process, while `own_peak` is this process's allocations since engine init, a delta immune to the 1.6 GB run-to-run free-VRAM swing #1103 documented. Measured byte-identical across repeat runs.

---

## A7. Migration plan

Strangler fig. The I1 allowlist starts at **365 sites / 74 files** and shrinks monotonically; `tools/check_alloc_sites.py` fails the build if a file not on the list allocates or the list grows. Every step leaves the tree green, keeps decode and prefill within 1 %, and records both in `docs/audit/PERF_LOG.md`.

| # | Step | Removes from allowlist | Why here |
|---|---|---:|---|
| 0 | `Backend` + `FakeBackend` + phase guard + `check_alloc_sites.py` (allowlist = everything, gate green from day one). No behaviour change. | 0 | The gate must exist before anything moves, or the list can silently grow. |
| 1 | Move `ArenaAllocator`/`BlockPool` to `src/memory/`, wire to `Backend`, add `StableSpan`/`Owned`/`BlockRef`. Still unused. | 0 | Pure addition; A1.6 says the code already exists and is dead. |
| 2 | **`plan_memory()` alongside `compute_vram_budget()`** - computed alongside, and since 2026-07-30 (B69) the KV block count is APPLIED from it. Log both; assert agreement within a tolerance in CI via V8. | 0 | Establishes the plan is right *before* anything depends on it. Highest-information, lowest-risk step. |
| 3 | **KV pool + `KVCacheManager`** → `BlockPool` + `BlockRef`. Deletes the manual refcount, `free_block_dropping_stale_hash`, and the free-by-id paths. | ~15 | The hardest ownership question (A5.1), with the best existing test coverage. Doing it early proves the refcount machinery before four more subsystems depend on it. |
| 4 | **Executor workspaces** (`exec/executor_workspace*.cu`, 47+ sites in one file) → T2 arena + `ScratchStack`. | ~70 | Biggest single-file win; sizes already computed centrally in `compute_shared_sizes`. |
| 5 | **Per-request allocations** (`engine_graph_decode.cpp`, `engine_scheduler.cpp`, `executor_attention_prefill.cu`, `executor_attention.cu`, MoE per-call arrays) → `ScratchStack`. **The step that satisfies I2**; the step-0 counter must reach zero. **Plan it per BUFFER FAMILY, not per file** (B59): the KV block tables alone are 8 acquisitions against **23 releases in 5 files**, one allocation freed at 8 sites depending on which path unwinds; a file-at-a-time pass cannot close them. | ~40 | Depends on the stack existing (4) and the plan sizing it (2). |
| 6 | **Weight upload + pre-dequant caches** (`model/`, `quant/`, `exec/pre_dequant_*`) → T1 arena. **Deletes the balloon** and the phase-local free-VRAM re-derivation. Switch allocation order to the plan's tier order. **Re-scoped 2026-07-30 (B61): both original justifications are spent.** B6's acceptance test (`03_kv_cache` at 0.0 MiB free on gpt-oss) was closed by 6.4's reordering (now ends at 10613 MiB free). The balloon binds on exactly ONE of five measured configurations (35B-MoE-NVFP4 @32k: 120 vs 108 covered MoE caches, KV 16 vs 87 blocks), where the capture-abort it exists to prevent does **not** reproduce and the uncovered path is correct. What remains open here is I4's other half: the live `cudaMemGetInfo` sizing, not the balloon's rescue. | ~90 | Largest blast radius; after the plan is trusted (2) and the KV pool no longer competes for a residual (3). |
| 7 | **VMM backend for the KV pool**, gated on the WSL2 spike (A3.1). If the spike fails, stop here. **The spike passed, B84 measured the premises spent, and B85 built it anyway once the condition B84 named for reopening actually happened in production.** | ~450 | Optional by construction; `kv_cache.growable`, off by default. |
| 8 | **`compute/` statics**: cuBLAS/cuBLASLt/CUTLASS workspaces from the T2 arena; delete the lazy CUTLASS growth path. | ~115 | Mechanical once the arena exists; `compute/` sites are small per-kernel scratch. |
| 9 | **Guardrails**: `--vram-budget` as a real cap, admission control per I6, `/metrics` tagged breakdown per I7, `--mem-report`, the peak-VRAM CI gate. | remainder | Needs everything above to have real numbers to report. |

Ordering rationale, one line each: gate before moves (0); tools before users (1); the plan before it is trusted (2); the hardest ownership problem while isolated (3); the biggest single file (4); I2 once scratch has somewhere to go (5); the balloon last among big consumers because its removal changes init ordering (6); the optional backend behind a gate (7); the mechanical sweep (8); the operator surface once it has something true to say (9).

Steps 3, 4 and 6 each need a coherence check (`check-degeneration`): KV cache, forward pass and weight caches respectively.

---

## Invariant compliance

"Design-time" is the state Phase A was written against, kept as the record. "Measured" is where Phase B stands; three of the seven are still open.

| | Invariant | Design-time | **Measured (rows carry their own date; latest 2026-07-30)** | Target |
|---|---|---|---|---|
| I1 | Single acquisition point | ✗ - 365 sites / 74 files outside `src/memory/` | **✗ - 499 calls / 74 files (2026-07-31), of which 224 are acquisitions and 275 releases.** The **pinned-host class is CLOSED: 26 → 0** (B58/B60) - every one moved to T5b's `PinnedBuffer`/`HostRegistration`, which is why the total fell 638 → 582 (each migration took its releases with it). What remains is device memory only. Progress since B48's 696/309 came from migrating whole clusters rather than sites (B52-B57); what stalled it before that was migrations keeping their original path as a fallback, which the gate cannot see (B34/B47) | ✓ - `Backend`, allowlist empty, CI gate |
| I2 | No allocation on the hot path | ✗ - measured +190 MiB/config of steady-state allocation | **✓ - `0 cudaMalloc, 0 cudaMallocAsync, 0 pinned-host allocations while serving`**, 15 requests, dense. Was 414 → 238 → 0 | ✓ - `ScratchStack`, phase guard, counter == 0 |
| I3 | Stable addresses for graph memory | ~ - true in practice, enforced by comments + a `workspace_generation` hook | **~ - `StableSpan` exists and is passkey-enforced**, so only allocators that can promise stability can mint one; kernel signatures still take raw pointers | ✓ - `StableSpan` in kernel signatures; no conversion from `DeviceSpan` |
| I4 | Capacity planned, not discovered | ✗ - live `cudaMemGetInfo`, a balloon, six stacked clamps | **~ - `plan_memory()` is pure and runs in shadow**; the live pass now charges the library reserve it always omitted (B37). **The balloon is GONE (2026-07-30, B62)** - the mandatory-cache guarantee is a planned floor instead of a physical hold, which also freed 72x more KV on the config where the hold bound. **The KV block count is now the PLAN's** (B69) - `plan_memory()` decides it, the live pass is the fallback when the plan rejects, and the measured-residual clamp can still only shrink it. What still enters through `cudaMemGetInfo` is the *distributable* figure the plan is handed - **measured 2026-07-31 (B82) and it does not move**: five identical starts produce a byte-identical plan, and a co-tenant holding 31 949 MiB changes neither the plan nor decode throughput (293.4 vs 292.7 tok/s, i.e. no WDDM spill). Replacing it is the invariant's letter, not a fix for anything that fails. **B82 holds only while the config cap binds before VRAM does** - with a VRAM-bound plan the same co-tenant takes 25 % of the KV pool (B8) | ✓ - `plan_memory()` never queries the device; fails at load with a report |
| I5 | Unidirectional ownership | ✗ - `VRAMAllocator` is a tracker; raw `void*` cross module boundaries | **~ - KV blocks and the T2/T3 tiers own through move-only RAII** (`Region`, `BlockRef`, `GraphSlotLease`); raw `void*` still crosses boundaries in `exec/` and `compute/` | ✓ - `Owned<T, Tier>`, no cross-tier conversion, no raw device pointers above L1 |
| I6 | OOM is typed and recoverable | ~ - `RequestStatus::CANCELLED`, plus a warning that fires *after* prefill | **✓ - both halves.** Plan-time: an unservable `--vram-budget` refuses at load with the block arithmetic. Admission-time: `IMP_ERROR_CAPACITY` → HTTP 503 `capacity_error`, distinct from a client cancel (B40) | ✓ - plan-time failure at load; admission-time 429/503 at runtime |
| I7 | Capacity ≠ occupancy | ✗ - 20-39 % of device memory unattributed | **~ - per-tier reserved *and* live is served** on `/metrics`, plus KV blocks and budget-vs-own. Accounting reaches **98.3 %** once the library reserve is charged at its measured value rather than the 3900 MiB constant (B41/B42); with the constant it is 82.5 % on Qwen3-8B. MoE's **102.0 %** - a negative residual - proves the note double-booking (B32), so that config's gap is the counters, not the attribution | ✓ - per-tier reserved *and* live, library reserve named, ≥95 % accounted |

Nothing in the invariant set was dropped. Two are weakened in a stated way: I3 is type-enforced for *stability* with the graph-invalidation hook kept as a runtime assert; I5's "request-scoped block cannot outlive its request" is type-enforced against aliasing and assert-plus-soak-enforced against outliving (A3.4).

---

## Open questions for Phase B

1. **WSL2 VMM spike** (A3.1): gates step 7 only.
2. **`LibraryReserve` calibration**: the ~3.9 GiB constant is measured on this driver/CUDA/card. Needs a boot-time self-check (assumed vs actual after the first forward, metric on >10 % divergence) rather than a hardcoded number, plus a documented re-measure procedure after a driver or CUDA bump.
3. **`ScratchStack` under concurrency**: one compute stream plus a prefill stream. Two stacks or one with a mutex? Lean toward two (one per stream): keeps LIFO per-stream, no lock on the forward path. Decide with the step-5 measurement.

---

## B0. Implementation log

What has landed, and every divergence from A2-A7 with the reason. Divergences are recorded here rather than by editing the design above.

### Landed

| A7 step | State | Commit |
|---|---|---|
| 0 - Backend, FakeBackend, phase guard, I1 gate | **done** | `feat(memory): L1 backend, allocation-phase guard, I1 CI gate` |
| 1 - tier allocators + typed spans | **done** | `feat(memory): arena, block pool, scratch stack, stable spans` |
| 2a - `plan_memory()`, pure and tested | **done** | `feat(memory): plan_memory - capacity planned, not discovered` |
| 2b - shadow plan: run `plan_memory()` next to the live budget at init, log both | **done** | `feat(memory): shadow plan - log what plan_memory would decide` |
| 3.1 - `KVCache` block ids + refcounts → `BlockPool` (slots mode) | **done** | `refactor(memory): KVCache block ids and refcounts move to BlockPool` |
| 3.2 - prefix cache holds its own `BlockRef` | **done** | `refactor(memory): the KV prefix cache owns its blocks instead of inheriting them` |
| 3.3-3.4 - `seq_blocks_` → `BlockRef`, drop the scaffolding | **done** | `refactor(memory): sequences own their KV blocks through BlockRef` |
| **A7 step 3 complete** | | |
| 4a - engine-persistent (T2) arena in production, first tenant | **done** | `feat(memory): engine-persistent arena, with the MMVQ scratch as its first tenant` |
| 4b.1 - T2 arena sized from exact per-tenant demand | **done** | `feat(memory): size the engine arena from exact tenant demand` |
| 4b.2 - the remaining `exec/` workspaces | **done (B52-B57, B83)**. The last holdout - the chunk-capture K/V pair - was mis-diagnosed, not unmigratable: "growable, a bump arena strands it" is true of a staircase of takes, not of taking the bound once, and the bound is exactly what `engine_spec_capture.cpp` already computes. Left outside T2 in `exec/`: the model-resident weight caches (T1, step 6) and one-shot init transients | `fix(exec): ...the MLA cluster to the T2 arena` / `fix(exec): the chunk-capture K/V pair to the T2 arena` |
| 5.1 - banned-token list: one engine-owned copy | **done** | `fix(runtime): one engine-owned device copy of the banned-token list` |
| 5.2 - `calibrate_fp8_scale` reduction scratch | **done** | `fix(quant): persistent reduction scratch for calibrate_fp8_scale` |
| 5.3 - conditional-graph-loop buffers → T2 slot pool | **done** | `feat(memory): T2 slot pool for the conditional graph loop` |
| 5.4 - pre-size the speculative verify path's lazy scratch | **done - criterion 3 reads zero** | `fix(runtime): pre-size the speculative verify scratch at init` |
| 9b.1 - reserve floored at the library charge; `--vram-budget` starts binding | **done** | `fix(vram): charge the library reserve in the live budget pass` |
| 9b.2 - plan-time refusal when a budget cannot serve one sequence (I6) | **done** | same |
| 9b.3 - `/metrics` per-tier reserved *and* live, KV blocks, budget (I7) | **done** | `feat(memory): tagged memory metrics and the peak-VRAM gate` |
| 9b.4 - peak-VRAM gate in `verify.sh`, pinned in `perf_baseline.json` | **done** | same |
| 9b.5 - admission-time 503 on KV exhaustion (I6) | **done** | `feat(api): a typed capacity refusal instead of a generic cancel` |
| 6.4 - weight caches before the KV pool; KV takes the measured residual | **done** | `fix(vram): build the weight caches before the KV pool` |
| 6.6 - delete the balloon; the mandatory-cache floor becomes planned, not pre-held | **done** | `fix(vram): delete the native-cache balloon - the floor was already the guarantee` |
| 6.0/6.5 - exact demand, T1 arena | not started | |
| 6.9 - the library charge is measured across the WHOLE init, not the warmup-forward window | **done (B79-B81)** - the ledger is always on, so the residual means what it says on every start; attribution 89.7 -> 100 % on the config that was worst, and the 35B's second start gains 4x KV | `fix(memory): measure the library charge across the whole init` |
| **criterion 4 - post-drain live blocks return to baseline** | **RUN and PASSES: live blocks 0 → 0 across 48 requests.** The earlier "one block per request leaks" reading was a gauge conflating live with cached - see B46, #1115 closed invalid | |
| 6.7 - the library reserve is measured and reported, not assumed | **done** | `feat(memory): measure the library reserve instead of assuming it` |
| 6.8 - the measurement is remembered; criteria 5+6 hold from the 2nd start | **done, with a condition B77 made explicit**: "the 2nd start" needs the cache path to outlive the process, and the default lands inside the container imp is run in - `vram.library_reserve_cache` must point at a mounted path or the constant is charged forever (+639 MiB of distributable VRAM lost per start on Qwen3-14B-Q6_K) | `feat(memory): remember the measured library reserve` |
| 7a - WSL2 VMM spike (the gate) | **done - GO** | `test(memory): WSL2 VMM spike - the step-7 gate is open` |
| 7b - VMM backend implementation | **built (B85), after the reopening condition B84 named actually occurred.** A server started while another process still held the card came up with a KV pool at its rescue floor, 16 blocks against a planned 3066, and stayed there for its whole life: capacity is planned once, against a free-VRAM reading taken while the previous process was still letting go. `kv_cache.growable` reserves address space for the pre-clamp plan and commits the clamped number, so the clamp becomes a starting point instead of a verdict. Off by default. The spike was re-run against CUDA 13.3 first: base invariant across growth, a captured graph correct after 1.5 GiB was committed under it, `cuMemRelease` returning VRAM to the driver within 2 MiB of the baseline, 1.18 ms per 256 MiB commit | `feat(memory): a KV pool that grows into what it asked for` |
| 8 - `compute/` statics | **done, by drawing a line rather than by reaching zero.** Four PRs: the DRY penalty pair (B72), the cuBLAS/CUTLASS workspace family and the grouped-3x reserve (B73, −488 MiB on every MoE model), **AUDIT B13's grow-on-demand statics (B74), which closes that bug class**, and the CUB sort + top-M scratches (B75). What is left in `compute/` belongs to other steps, enumerated in B75: weight-derived caches keyed by source pointer (T1, step 6), per-call result tensors (I2, step 5 - criterion 3 already reads zero because the serving path does not use them), one-shot init transients, and test-only overloads. One buffer migrated and **handed back**: `attention_cublas.cu`'s pointer array has no degradation contract, and a tenant whose caller cannot refuse does not belong in a tier that can run out | `fix(compute): DRY penalty buffers to the T2 arena` / `fix(compute): the cuBLAS and CUTLASS workspaces to the T2 arena` / `fix(compute): the grow-on-demand IMMA and MXFP4 scratches to T2` / `fix(compute): the CUB sort and top-M scratches to T2` |
| 9a - `--mem-report` with named charges | **done** | `feat(memory): --mem-report - name the charges the pool notes cannot see` |
| **A7 step 9 complete** (9a + 9b.1-9b.5) | | criterion 5 is *not* claimed - see B38 |

### I1 allowlist baseline

`tools/alloc_allowlist.txt` opens at **79 files / 717 sites**. Larger than A1.1's "365 lines in 74 files" because the gate also counts the `cudaFree`/`cudaFreeAsync`/`cudaFreeHost` side: I1 names them, and a free outside `src/memory/` means ownership lives outside too. A1.1 counts allocations only. Both are correct; the gate uses the wider one on purpose.

### Divergences from the A3 sketch

| # | Design said | Implementation does | Why |
|---|---|---|---|
| D1 | `Owned<T, Tier>`, with `Tier` a template parameter, so a model-resident buffer cannot be stored in an engine-persistent slot | `StableSpan<T>` everywhere; the arena carries a runtime `RegionTag` for reporting and a `generation()` counter for staleness | The compile-time tier tag would have to appear in every storage member and signature, to prevent a bug class that has not bitten - `server.model_swap` shipped and works. imp already has `workspace_generation` for exactly this staleness check (the spec-graph cache invalidates on it). **Cost of the divergence, stated: cross-tier smuggling is caught at runtime by the generation counter, not at compile time.** |
| D2 | `template <size_t Stride> class BlockPool` | stride is a runtime constructor argument | KV block bytes are computed at init from `(kv_dtype, block_size, n_kv_heads, head_dim)` - see `kv_block_bytes_per_layer()`. A compile-time stride cannot express that, and instantiating per dtype would multiply the TU for no gain. |
| D3 | `std::expected<Region, MemError>` | `AcquireResult { Region; MemError; explicit operator bool }` | These headers are included from `.cu` TUs. nvcc's frontend is the constraint, not the language version; a plain aggregate costs nothing and cannot surprise it. |
| D4 | `Backend::acquire()` for everything | separate `acquire()` and `acquire_growable(reserve, initial_commit, …)` | A VMM reservation has two sizes, not one. Overloading `bytes` to sometimes mean "reserve" would make the KV pool's call site ambiguous at exactly the place the design removes ambiguity. `CudaMallocBackend` inherits the default and returns `NotGrowable`. |
| D5 | `StableSpan`'s private constructor with the allocators as template friends | passkey idiom: a `detail::StableKey` only the allocators can construct | One friend list in one place instead of forward declarations of every allocator inside `span.h`; granting a new type the right to promise stability stays a single greppable edit. Same guarantee. |
| D6 | (not specified) | `FakeBackend` quarantines the last 16 released regions instead of freeing them | Poison-on-release is only useful if a test can read the poison back. Freeing immediately would make `is_poisoned()` a use-after-free *in the test*. |
| D7 | A4 sketched `PlanInput` as `{ModelShape, FeatureSet, ConcurrencyLimits, feature flags, budget}` | plus two explicit measured charges: `context_bytes` and `LibraryReserve` | The two things the old planner could not see. `context_bytes` (1679.6 MiB measured) is gone before imp allocates anything; `LibraryReserve` is the ~3.9 GiB of A1.5. Named inputs are the difference between a plan and a guess - `MemoryPlan.ChargesTheLibraryReserveAsAFirstClassLineItem` fails if the KV pool stops noticing. |
| D12 | A3.3 has the arena owned by the Engine | `engine_arena()` is a **process global**, opened by `Engine::init` and closed by `~Engine` | Its tenants are file-scope statics in `compute/` and `exec/` with no Engine to reach through - same reason `gemm.cu`'s cuBLAS workspace is a static today. Single-engine-per-process is the supported deployment (`vram_query.h` says so explicitly). A `generation()` counter makes a stale cached pointer detectable across an engine teardown, which the pre-arena statics could not do. |
| D13 | A2 says a T2 tenant draws from the arena, full stop | the MMVQ scratch **falls back** to its own allocation when the arena is short, with a WARN | Measured, not hypothetical: the demand is `max_tokens x ceil(K/32) x 36 x 2`, and `max_tokens` belongs to the *executor*, not to anything the engine can bound when it opens the arena - 24 MiB on a bench run, **108 MiB on a server default** at K=12288. A guessed reservation that is too small must not be what breaks a model. The fallback goes when the planner sizes T2 from measured high-water marks (B5); until then the shortfall is reported rather than fatal. |
| D10 | B2 said "`KVCache` acquires its region from `BlockPool`" | `BlockPool` gained an **id-space-only** mode (`open_slots`); `KVCache` keeps its own region | Reading the layout inverted the assumption. `BlockPool::block(id)` resolves `base + id * stride`, but the KV pool is laid out **layer-major**: one block id's bytes are scattered across per-layer K and V regions whose sizes differ per layer (Gemma-4 dual geometry) and per group (SWA layers hold only the trailing window, `layer_capacity()`). A uniform stride cannot express that - and the addressing was never the part that needed fixing. What had to move was the **ownership**: the free list and the refcounts. `BlockPool` now owns the id space and optionally a uniform region; the KV cache takes the former and keeps the latter. `open()` is unchanged and still what the SSM state, residual ring and snapshot store will use. |
| D11 | (not specified) | `BlockRef::release()`, `BlockPool::acquire_raw/release_raw/adopt_raw/abandon()` | Strangler scaffolding, marked as such in the headers and deleted in step 3's final commit. They let the int-based `allocate_block`/`free_block`/`inc_ref` API keep **exactly** its current semantics - including tolerating a free of an already-free block, which `KVCache::free_block` does today and a `BlockRef` drop must not - while the manager's three referents migrate one at a time. Without them every caller would have to move in one commit, which is precisely the big-bang this migration is shaped to avoid. |
| D9 | A7 step 2 said "assert agreement in CI via V8" | the shadow plan **logs** the comparison; there is no CI assertion of agreement. **Resolved 2026-07-30 (B65/B66): they now agree, and the assertion exists** - the divergence was one `target_blocks = needed_blocks * 2`, removed after measuring that it bought no prefix-cache reuse. V8 is `LivePassNeverExceedsThePlan`, an inequality rather than an equality (the two compute differently on purpose; what must hold is that the live read never spends what the plan committed elsewhere) | The two did not agree, and should not have: measured on the dense server default, the live pass hands KV **4096 blocks (4608 MiB)** while the plan takes **2048 (2304 MiB)** - see B1 below. Asserting agreement would have meant either encoding the old pass's 2x overshoot as correct, or writing a tolerance so wide it asserts nothing. V8 is asserted instead against the plan's own contract (a successful plan never exceeds its budget), which is the property that actually has to hold. |
| D8 | A4 said the planner "fails at load time with a report" | it also fails when the KV pool would sit **below the admission floor**, even though the plan technically fits | A pool that cannot hold one advertised sequence is not a working configuration: it prefills fine and then cancels on the first block append, while `/v1/models` keeps advertising `max_seq_len`. Observed on Qwen3.6-35B-A3B-NVFP4 at `--max-batch 64`, where KV collapsed to 16 blocks = 512 tokens and every longer prompt came back `finish_reason=cancelled` with no hint why. Reporting that as a successful plan would preserve exactly the failure I4 exists to remove. |

### B1 - what the shadow plan found

Measured at server defaults (Qwen3-4B Q8_0, `--max-batch 8`, `runtime.max_seq_len=4096`), both sides fed the same demand figures:

```
  distributable               25551 MiB
  weight-cache demand          2397 MiB  (same figure the live pass used)
  library reserve              3900 MiB  (the live pass does not charge this)
  engine-persistent             397 MiB
  KV: live 4096 blocks -> plan 2048 blocks (4608 -> 2304 MiB)
```

**The live pass gives the KV pool exactly twice what the configuration asks for.** `needed_blocks = ceil(4096/16) x 8 = 2048`; `vram_budget.cpp:457` sets `target_blocks = needed_blocks * 2` for every non-mode-2 strategy. 2304 MiB of the pool cannot be reached by any request the server will accept: 8 concurrent sequences at the advertised context fill 2048 blocks and stop.

Invisible on the dense config (14 GiB free either way). It stops being invisible where the incidents happened: the surplus is drawn from the same post-weight headroom the pre-dequant caches compete for, and it is the same order of magnitude as the library reserve the pass cannot see. #1100 and #1103 are both cases of that headroom being mis-divided.

Not fixed here: the shadow plan computes and does not apply; flipping `target_blocks` alone would change KV sizing for every model without the rest of the plan behind it. Step 6 switches it over.

### B2 - step 3 scope (KV pool → `BlockPool`/`BlockRef`)

Written down before starting: an error here does not crash, it silently corrupts cross-request KV (the #1044/#1045 class).

**Blast radius:** `kv_cache.{h,cu}` (615 LOC), `kv_cache_manager.{h,cpp}` (1519 LOC), 13 call sites in `runtime/scheduler.cpp` and `runtime/engine_scheduler.cpp`, 58 tests across `test_kv_cache.cpp`, `test_fp8_kv_cache.cu`, `test_prefix_cache_equiv.cpp`.

**What moves.** `KVCache` today owns the pool *and* the free list *and* the refcount (`allocate_block` / `free_block` / `inc_ref` / `ref_count`). Only the middle two move: `BlockPool` takes the region, free list and refcount; `KVCache` keeps the geometry (per-layer K/V offsets, scale and sketch regions) and becomes a pure address calculator over `BlockPool::block(id)`.

**Order, so the tree stays green at every commit:**

1. `KVCache` acquires its region from `BlockPool` but keeps its own free list and refcount. Pure plumbing; every test unchanged. The two-id-space problem is settled here (below).
2. `KVCacheManager::seq_blocks_` becomes `std::vector<std::optional<BlockRef>>`
   - `std::optional`, not `BlockRef`, because `evict_middle_blocks()` (StreamingLLM) frees slots while keeping the table *length*, and the attention kernels depend on that positional alignment. `-1` becomes `nullopt`; the kernel-facing block table is still built as ints.
3. **Done.** The cached LRU takes its own `BlockRef` per entry and `free_sequence()` transfers rather than omits. `free_block_dropping_stale_ hash()` is redundant in the common case but deliberately left in place - it still guards the rollback of a *shared* cached block; removing it is a separate decision.
4. **Done.** The pin set needed no ownership change (see the correction in A5.1); its `reclaimable_cached_count_` bookkeeping was already keyed on the cached LRU and is unchanged.
5. **Done, partially.** `allocate_block_with_eviction()` (int), `free_block_dropping_stale_hash()` and `BlockPool::adopt_raw()` are gone. `KVCache::allocate_block`/`free_block`/`inc_ref`/`ref_count` and their `BlockPool` backing (`acquire_raw`/`release_raw`/`BlockRef::release`/ `abandon`) **stay**: they are KVCache's own public int API, exercised directly by `KVCacheTest.KVCacheRefCounting` and used by the SWA group, which has no sharing and therefore no need for handles. No longer scaffolding; the comments say so.

**Three traps, all load-bearing:**

- **Two id spaces.** SWA blocks live in a separate space with their own free list (`allocate_swa_block`/`free_swa_block`). That is two `BlockPool`s (`RegionTag::KvBlockPool`, `RegionTag::SwaBlockPool`), not one pool with a partition; `BlockRef` must not cross between them.
- **SWA blocks are deliberately unshared.** Never hashed, pinned, persisted or shared; `share()` on one is a bug. Assert it.
- **The persisted prefix cache is a fourth referent.** `save_prefix_cache()` / `load_prefix_cache()` serialise blocks and re-register hashes on load; the load path must take refs, not raw ids.

**Verification bar:** the 58 existing KV tests unchanged and green, plus a coherence check (`check-degeneration`): the failure to look for is a repetition loop or cross-request bleed, which the CPU tests cannot see.

### B6 - the measurement campaign for step 6, and what it found

Run before touching the ordering, as the control arm. Two configs, server defaults, `diagnostics.vram_audit=true`.

**#1100's own repro is already fixed; no win left to demonstrate there.** Qwen3-14B-Q6_K, `--max-batch 8`, auto `max_seq_len` 32768: `Phase-4 overlay: registry=282 cached / plan-ideal=282`, KV 5154 blocks (6442 MiB), decode cache 280 tensors (7087 MiB), 2314 MiB free at the end. #1102 closed the arithmetic; the ordering change cannot improve on 282/282. Recorded so nobody re-derives the expectation.

**gpt-oss-20b-mxfp4 at server defaults is still broken, mechanism exact.** ~16 tok/s (120 tokens in 7.1-7.8 s across two requests). Lifecycle checkpoints:

```
02_weights+decode_cache   used  6127.6   free 26479.0   delta  3708.0
03_kv_cache               used 32606.6   free     0.0   delta 26479.0
04_features               used 32606.6   free     0.0   delta     0.0
```

**`03_kv_cache` consumed 26 479 MiB, every remaining byte, down to 0.0 MiB free.** The chain, from the same log:

1. `max_seq_len: auto → 131072` (full model context; KV costs 49 152 B/token, one sequence alone is 6.4 GiB).
2. The planner-driven weight-cache reserve *does* fire: 5811.6 MiB, capped so KV keeps one full sequence (`kv guarantee=8192 blocks`).
3. `KV clamped 65536 → 25382 blocks to fit post-weight VRAM`: **19 036 MiB**, 406 112 tokens, for a `--max-batch 8` server.
4. The pre-dequant phases expand into what is left; `vram_alloc_force` (8 sites in `pre_dequant_phase3_moe.cu`) bypasses the headroom check entirely.
5. The loser is whoever allocates *after* the cache build: `VRAMAllocator: rejecting nvfp4_dequant allocation of 31.64 MiB (0 MiB free, need 1630 MiB headroom)`.

A **31 MiB** workspace fails on a 32 GiB card because a 19 GiB KV pool and an unbounded cache build got there first. Every step is locally reasonable; the composition is not. Cleanest evidence in the whole audit for I4: the engine allocating until the device says no, the last tenant paying. Another clamp cannot fix it; a clamp is what produced step 3.

**Superseded in part - see AUDIT B61 (2026-07-30):** the acceptance test below now passes without step 6, and the balloon's premise ("caches built last, after KV init") stopped being true when 6.4 reordered them.

**Consequence for the step-6 order.** A KV reservation threaded into `split_pre_dequant_budget` (the cheap way to make 6.4 safe without 6.5) would *not* have fixed this config: `vram_alloc_force` consults no reserve. The guarantee has to come from the tiers owning their memory (6.5), not a term subtracted before a live-free read. 6.4 and 6.5 are one change, and this config is its acceptance test: **`03_kv_cache` must not end at 0.0 MiB free, and `nvfp4_dequant` must not be rejected.**

### B5 - what step 4 still needs

4a landed the arena and proved it on one tenant. The rest of `exec/` needs two things 4a deliberately did not guess at:

1. **A planner-supplied capacity.** The arena opens at a fixed `kEngineArenaDefaultBytes` (64 MiB) because the engine cannot bound its tenants' demand at open time. Measured on Qwen3-8B-Q8_0: the MMVQ scratch alone wants 23.62 MiB at `max_tokens=896` and **108 MiB at 4096**. The fix is not a bigger constant: `plan_memory()`'s `engine_persistent_bytes` becomes the capacity, fed by the high-water marks the arena and `ScratchStack` now report.
2. **A decision on the three rollback groups.** `moe_3x_packed`/`_sf`, `cutlass_act_data`/`_sf`/`_workspace` and `mxfp4_act_sf`/`_workspace` each allocate a set and, if one member fails, free the others and disable the feature. A bump arena cannot reclaim, so those bytes would be stranded. Bounded and one-time, but it has to be a decision per group - and with a planner-sized arena, a member failing means the plan was wrong, a better signal than a silent feature downgrade.

Leverage point for the mechanical part: `vram_alloc()` in `exec/executor_helpers.h`. 31 of the sites funnel through it, so the tier switch is one function once (1) and (2) are settled.

### B4 - step 3 as built

Final ownership shape:

| owner | holds | drops when |
|---|---|---|
| `KVCacheManager::SeqBlocks` (per sequence) | one `BlockRef` per positional slot; an **empty** ref is a hole | the sequence is freed, rolled back, or StreamingLLM-evicted - in every case by the reference going out of scope |
| `cached_blocks_map_` (prefix cache) | one `BlockRef` per entry | the entry is erased (reclaim, or a reuse that moves it to a sequence) |
| `pinned_blocks_` / `pin_refcount_` | **nothing** - an eviction-policy overlay | n/a |

Every hand-over is a `std::move` of a reference rather than an omitted free. `free_sequence()` has no "skip the free" branches left; it moves the reference into the cache or lets it drop. `evict_middle_blocks()` calls `make_hole(i)`, which drops the reference and keeps the slot: table length, and with it the kernels' positional alignment, unchanged by construction rather than by a `-1` convention.

`block_table()` still returns `const std::vector<int>&`, derived from the refs and rebuilt lazily on mutation, so the ~10 consumers in `runtime/` are untouched and the two representations cannot drift.

Deleted: `free_block_dropping_stale_hash()` (its guard survives as `drop_stale_hash_if_last()`, which no longer also frees), `allocate_block_with_eviction()` (int), `BlockPool::adopt_raw()`.

### B3 - why the remaining step-3 work does not split further

Attempted 3.2 (`seq_blocks_` → `BlockRef`) and stopped before committing: the code says the B2 decomposition is wrong. Recorded so the next attempt starts from the real shape.

`free_sequence()` has three branches, two of which **deliberately do not free**:

```cpp
if (pinned_blocks_.contains(block_id)) {          // keep alive at count 1
    if (cache_->ref_count(block_id) > 1) cache_->free_block(block_id);
    ... add to cached LRU ...; continue;
}
if (prefix_caching_enabled_ && cache_->ref_count(block_id) == 1) {
    ... add to cached LRU ...; continue;           // "Skip cache_->free_block()"
}
cache_->free_block(block_id);                      // normal
```

A sequence's reference is not dropped; it is *left behind* for the cache to inherit implicitly. Converting `seq_blocks_` to `BlockRef` first means writing scaffolding that deliberately leaks the ref in both branches, deleted one commit later; converting the cache first requires `free_sequence()` to stop skipping in the same change, or the count is off by one. **The two are one commit.**

The hard part is not the storage type: `ref_count(block_id) > 1` in the pinned branch means "another sequence also holds this block via prefix reuse", and the code drops exactly one count and lets the cache inherit the remainder. Making the cache a real owner forces a decision about what those retained counts mean, taken against what the 58 KV tests actually assert, not the happy path. A reading task before an editing one. D11's scaffolding (`adopt_raw`, `release_raw`, `abandon`) is the right shape: `adopt_raw` is "the sequence hands its reference to the cache", which is what the skip branches mean.

### B7 - criterion 3, measured

`IMP_ALLOC_INTERPOSE=ON` links `src/memory/alloc_interpose.cpp` and puts `-Wl,--wrap=` on the executables, so imp's references to the CUDA allocation symbols resolve to recorders that forward to `__real_*`. Calls made *inside* libcudart/cuBLAS/CUTLASS are not redirected (resolved at library link time), which is right: the ~3.9 GiB library reserve (A1.5) stays out of the counter and is charged separately.

First measurement, dense config, 15 serving requests:

```
[alloc-interpose] I2 VIOLATIONS while serving:
    cudaMalloc            414 calls        1.15 MiB
    cudaMallocAsync         0 calls        0.00 MiB
    pinned host            72 calls        0.01 MiB
```

Two things this settles:

1. **`calibrate_fp8_scale()` allocated twice per call** (`fp8_quant.cu:211/212`), 144 of the 414, listed by no inventory. One-shot per layer (`executor_kv_write.cu` gates on `kv_calibrated_[kv_layer]`), so an I2 violation by the letter rather than hot-path traffic. Fixed via persistent arena scratch: 414 → 315. The rest of the named sites are `CudaGraphConditionalRunner::setup`, which the step-5 inventory did list and which is genuinely per burst.
2. **The bytes are irrelevant; the count is not.** 1.15 MiB is three orders of magnitude below M2's +190 MiB, so that delta is library/driver internal growth, not imp's allocations (AUDIT B30). Step 5's value is removing 414 driver round-trips from the hot path, not reclaiming memory.

### Invariants now under test

`tests/test_memory_backend.cpp` and `tests/test_memory_allocators.cpp`, both in the CPU lane (`test-core`), cover V1 (conservation), V2 (no allocation while serving), V3 (arena reset frees wholesale), V4 (block-pool conservation under randomised churn), V5 (refcount balance when an exception unwinds mid-sequence), V6 (LIFO discipline), and V9 (address stability across commit/decommit). `tests/test_memory_plan.cpp` adds V7 (plan determinism, 1000 randomised configs) and V8 (a successful plan never exceeds its budget, 2000 randomised configs).

The I3 mechanism is asserted at compile time in `tests/test_memory_allocators.cpp`: `DeviceSpan` does not convert to `StableSpan`, `StableSpan` is not constructible from a raw pointer, and the widening direction stays implicit. A refactor that reopens the hole fails to compile.

### B8 - a co-tenant moves the plan, and no `/health` field can say so (2026-08-19)

Same command, same model, same build; only the card differs:

| card at startup | KV blocks | `kv_ceiling_blocks` | what `/health` says |
|---|---:|---:|---|
| idle | 23 339 | 23 339 | `ok`, pool at its ceiling |
| 23.4 GiB in use by a neighbour | 17 406 | 17 406 | `ok`, pool at its ceiling |

A client comparing total against ceiling reads "at capacity, healthy" in both, at a quarter of the pool. Nothing is faulted, nothing is floored: the second server loads, serves, and every number it produces is a statement about a card it shared.

**This bounds B82.** B82 found a co-tenant holding 31 949 MiB changing neither plan nor decode throughput, and it is correct *in its regime*: at `runtime.max_seq_len=8192` the plan asks `max_seq_len x max_batch / block_size` = 16 384 blocks, the config cap binds before VRAM does, and a neighbour is invisible. The table above is the same box with `max_seq_len=65536`, where the plan is VRAM-bound, and the neighbour takes 25 % of it. "The plan does not move" holds only while something other than free VRAM is the binding constraint.

**Two `/health` fields were built for this state and both were measured to fail.** Recorded so they are not rebuilt:

1. **The pre-clamp plan beside the total** (`kv_blocks_planned`). The plan is handed `effective_free_vram()`, so it shrinks with the neighbour like everything else. Occupied arm: planned 17 673 against a total of 17 512, a 0.9 % gap that reads as a clean bill of health while the pool sits 25 % below an idle-card start. The gap it does expose (measured residual undercutting the plan) is the ordinary case the `KvResidualSizing::clamped` comment already calls normal.
2. **Device-used at `vram_budget_install`** (`vram_used_at_install_bytes()`). 1679 MiB in **both** arms. Under WSL2/WDDM the driver reports the whole card as free until a process allocates against it, so a snapshot taken before the first allocation cannot see a neighbour at all. Every figure derived from that baseline inherits the blindness; `vram_own_used_bytes()` cannot answer this either.

**What does see it is the process's own upload.** For the same 3263 MiB checkpoint, `free_before - free_after` across `upload_weights_gpu` measured 3264 MiB on the idle card and 8446 MiB beside the neighbour: the first moment the card's real occupancy is observable at all. Until now the line printed the delta as `weights ~8446 MiB`, labelling a co-tenant as this model's weights. It now names the delta for what it is and warns when it exceeds the checkpoint on disk by more than a quarter (`engine_weight_upload.cpp`). One-sided on purpose: consuming *less* than the file is ordinary (host-resident experts, dropped sources); consuming a quarter more cannot be weights. Checked against a 9.9 GiB NVFP4 checkpoint directory on an idle card: consumed 10 080 MiB, stays silent.

```
[PROV: commit=f39441d0 date=2026-08-19 hw=RTX5090 model=Llama-3.2-3B-Instruct-Q8_0
       quant=Q8_0 cuda=13.3 path=imp-server n=2 arms x 2 configs
       cmd=`imp-server --model <m> --set runtime.max_seq_len=65536`, neighbour =
       a second imp-server pinned to `--set kv_cache.max_blocks=14000`; blocks and
       ceiling read from GET /health, upload delta from the server log]
```


---

## Provenance

Every number in A1: measured on this host 2026-07-28 against `imp:test` built from the working tree at `fix/1104-json-number-grammar` = `main` + the two `#1104` constrain commits (no memory impact) + the **staged, uncommitted `#1103` fix** to `vram_budget.cpp`/`vram_query.h` flooring the mode-2 reserve at the allocator's 5 % headroom. GPU otherwise idle (0 containers, no compute processes), healthy under load (2857-2932 MHz SM, 13801 MHz mem, 310-444 W). Harness: `MemAccount` via `diagnostics.vram_audit`, driver `tools/analysis/vram_audit_load.py`.

Findings, including the refuted ones, are recorded in `AUDIT.md`.
