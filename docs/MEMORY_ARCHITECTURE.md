# imp Memory Architecture

> **Status: Phase A — design, not implemented.** Nothing in this document is in
> the tree yet. Section A1 is measured current state; A2–A7 are the proposal.
> Implementation follows the migration order in A7, one mergeable commit per
> subsystem, and this document is updated in the same commit whenever the
> implementation diverges from it.

Target: `sm_120a` (RTX 5090 / GB202, 32 607 MiB) with the `sm_120f` PTX
fallback. No SM80/SM90 paths.

---

## 0. The defect

imp has no memory architecture. It has 380 allocation sites across 81 files.
Ownership is implicit, lifetime is by convention, and capacity is whatever falls
out of a 480-line arithmetic ladder in `src/runtime/vram_budget.cpp` whose input
is a live `cudaMemGetInfo` reading that is **wrong by ~3.9 GiB** at the moment it
is taken (A1.5). Leaks and OOMs are downstream of that; so are #874, #926, #934,
#963, #1100 and #1103 — six incidents whose fixes are all still visible in
`vram_budget.cpp` as stacked clamps, each correcting the previous one.

The five structural facts that the design has to change:

| # | Fact | Evidence |
|---|---|---|
| F1 | Only **31 of 336** device-allocation sites route through `VRAMAllocator`. The rest call the driver directly and are invisible to it. | census, A1.1 |
| F2 | `VRAMAllocator` is **a tracker, not an owner** — its own destructor comment says so, and it enforces its headroom in three separately-defeatable ways. | `vram_allocator.{h,cu}`, A1.2 |
| F3 | Capacity is **discovered, not planned**: the KV pool is sized from live free VRAM *before* the weight caches exist, and the caches then re-derive their own budgets from live free VRAM again. | `vram_budget.cpp`, `split_pre_dequant_budget` docstring, A4.1 |
| F4 | A fixed **~3.9 GiB is claimed on the first forward pass**, after the plan is final, attributed to nothing. Invariant to batch (1→16) and context (1024→4096). | measured, A1.5 |
| F5 | `src/core/allocator.h` already contains an `ArenaAllocator` and a `PoolAllocator`. **Both are dead** — zero references anywhere in `src/ include/ tools/ tests/`. The disciplines this design needs were written once and never wired up. | grep, A1.6 |

---

## A1. Current-state inventory

### A1.1 Allocation-site census

Measured on the working tree (`main` + the two `#1104` constrain commits + the
staged `#1103` budget fix, see A1.4). Comment-only mentions excluded.

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

Device-side total: **336 sites**, of which **31 (9 %)** are tracked.
`cudaFree`/`cudaFreeAsync`/`cudaFreeHost`: 422 sites.

**Sites outside `src/memory/`: 365, in 74 files.** That is the initial I1
allowlist.

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

Top files by site count: `exec/executor_workspace_buffers.cu` (47),
`model/weight_upload.cu` (19), `runtime/engine_scheduler.cpp` (17),
`runtime/cuda_graph.cu` (17), `compute/gemm_grouped_nvfp4_smallM.cu` (14),
`runtime/engine_graph_decode.cpp` (13),
`exec/pre_dequant_phase3_nvfp4_decode.cu` (13).

### A1.2 What `VRAMAllocator` actually enforces

`src/memory/vram_allocator.{h,cu}`. Its headroom is defeated three ways, all
live:

1. `can_allocate()` returns `true` unconditionally for anything `< 16 MiB`.
2. `allocate(..., bypass_headroom = true)` skips the check entirely (used by the
   NVFP4 MoE cache).
3. On a headroom failure `allocate()` retries against raw free VRAM and proceeds
   anyway if `free >= bytes + 64 MiB`, logging a warning.

Its destructor deliberately frees nothing ("The allocator is a tracker, not an
owner"). Every one of its 31 callers therefore holds a raw `void*` and is
responsible for its own `vram_free`.

The single hard constraint it does impose — `free >= bytes + 5 % of total` for
allocations `>= 16 MiB` — was until the staged `#1103` fix **unknown to the
planner**, which planned down to a 512 MiB reserve while the allocator refused
anything leaving less than 1630 MiB free. The plan was unexecutable by
construction; the caches it starved failed mid-build and cost ~7× decode on
gpt-oss-20b at server defaults.

### A1.3 Measured footprint — three configs

Harness: existing `MemAccount` (`src/memory/mem_account.{h,cu}`, gated by
`diagnostics.vram_audit`), lifecycle checkpoints + per-pool notes + a 2 ms
device-used peak sampler. Driver: `tools/analysis/vram_audit_load.py`,
2 rounds × N concurrent streaming completions, 0 errors, GPU healthy throughout
(2857–2932 MHz SM / 13801 MHz mem / 310–444 W under load — no depressed-host
artefact).

Card total 32 607 MiB. `00_pre_init` (CUDA primary context + WSL2/WDDM driver)
is **1679.6 MiB** on all three.

| config | model | max_batch | ctx | load-time peak | steady state | peak under load | free |
|---|---|---:|---:|---:|---:|---:|---:|
| **dense** | Qwen3-4B-Instruct-2507 Q8_0 (36 L, d=2560, kv_heads=8) | 8 | 4096 | 18 226 | 18 226 | **18 416** | 14 191 |
| **MoE** | Qwen3-Coder-30B-A3B NVFP4 (48 L, d=2048, kv_heads=4) | 8 | 4096 | 23 872 | 23 872 | **24 050** | 8 557 |
| **vision** | gemma-3-4b-it Q4_K_M + mmproj-F16 (34 L, d=2560) | 4 | 4096 | 14 792 | 14 792 | **14 992** | 17 617 |

All MiB. Aggregate throughput during the load: 876 / 567 / 215 tok/s.

> **Substitution, stated:** the dispatch names Gemma-3-12B + mmproj. There is no
> 12B mmproj on this host (`/home/kekz/models/gemma-3-4b-vl/` is the only vision
> pair). The vision row is gemma-3-4b + `mmproj-F16.gguf`, driven with three real
> image requests in addition to the text load. The *shape* of the vision finding
> (tower is resident, `04_features` is where it lands) is unaffected by the size.

**Peak − steady state is +190 / +178 / +200 MiB.** This is the entire
steady-state per-request allocation surface, and it confirms the 2026-06-12
finding still holds: there is no transient prefill spike to cap, because every
workspace is statically pre-allocated. The +190 MiB is the I2 violation, and it
is small — which is why nobody has fixed it, and why the fix is cheap.

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
| `WEIGHT_CACHE_NVFP4` | 2157.7 | — | 263.0 |
| `WEIGHT_CACHE_CUTLASS_SF` | — | 1800.5 | 29.2 |
| `WEIGHT_CACHE_FP16` | — | — | 5185.0 |
| `KV_BLOCK_POOL` | 4608.0 | 1536.0 | 2176.0 |
| `EXEC_WORKSPACES` | 396.6 | 507.1 | 452.4 |
| **tracked total** | **11238.3** | **19311.0** | **10473.2** |
| **untracked residual** | **7177.2 (39 %)** | **4738.5 (20 %)** | **4516.4 (30 %)** |

Acceptance criterion 6 asks for ≥95 % accounted. Today it is **61–80 %**.

### A1.5 F4 — the ~3.9 GiB first-forward claim

The dense config's `05_post_warmup` delta of +4180 MiB is not a warmup cost. It
is the cost of the **first forward pass**, whenever that happens:

| probe | result |
|---|---|
| `runtime.warmup=false`, measured before any request | init ends at **14 046 MiB** (`05_post_warmup` Δ = 0) |
| same process, after **one** 32-token request | **18 234 MiB** — **+4188 MiB** |

It is invariant to everything the planner knows about:

| max_batch | ctx | init used | after 1 request (smi) | Δ |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 10 566 | 14 439 | **3873** |
| 8 | 4096 | 14 046 | 17 901 | **3855** |
| 8 | 1024 | 10 268 | 14 116 | **3848** |
| 16 | 4096 | 18 676 | 22 562 | **3886** |

(`nvidia-smi` reads ~180 MiB below the in-process `cudaMemGetInfo` used; the
delta is what matters and it is flat.)

Sub-hypotheses tested and **refuted**:

- **imp's own lazy CUDA module loading.** `CUDA_MODULE_LOADING=EAGER` moved only
  **+124 MiB** into `00_pre_init` (1679.6 → 1803.6) and +270 MiB into
  `01_prewarm_gemm`. The first-request delta stayed at +4188 MiB.
- **The default `cudaMallocAsync` pool.** `reserved`/`used` are 4096/4076 MiB
  before *and* after the request — unchanged.
- **Scaling with batch or context.** Flat, per the table above.

What remains is CUDA/cuBLAS/CUTLASS library-internal lazy reservation claimed on
first matmul dispatch. **This is 21 % of the dense config's total footprint and
the planner does not know it exists**: at budget time the dense run logs
`available=22290.3 MiB` and hands the KV pool 4608 MiB — from a number that is
~3.9 GiB too optimistic. It survives today only because the dense config happens
to leave 14 GiB free. On the MoE config, which ends with 8.5 GiB free, the same
constant is already 46 % of the remaining headroom.

**Design consequence:** the planner must charge a measured, arch-and-driver-
specific *library reservation* as a first-class line item (A4), and `--mem-report`
must attribute it explicitly rather than dumping it into a residual (A5.3).

### A1.6 Verified-dead and verified-clean (do not re-chase)

- **`ArenaAllocator` / `PoolAllocator`** (`src/core/allocator.{h,cpp}`, 68 + 100
  LOC): zero references in `src/ include/ tools/ tests/`. Dead. They implement
  bump-arena and fixed-block-pool disciplines — exactly two of the five tiers
  this design needs — and were never wired to anything.
- **`Buffer`** (`src/core/buffer.{h,cpp}`): exactly **one** producer
  (`engine.cpp:699`, vision embeddings) and one holder
  (`Request::vision_emb`). Not a general RAII layer; a single-purpose helper.
- **COW-fork / Best-of-N does not exist.** Grep for `cow|copy_on_write|fork_seq|
  best_of|n_best` across `src/ tools/` returns one hit, a comment in
  `scheduler.cpp:117` noting that a *hypothetical* site would need COW. The
  dispatch's "block with three referents" is real, but the three referents are
  **sequence block table + prefix-cache hash table + pin set** (plus the
  on-disk persisted cache), not a fork. Design accordingly (A5.1).
- **No persistent GEMM autotuning state.** `src/compute/gemm.cu` contains no
  file I/O of any kind; the cuBLASLt algo cache is a process-local map rebuilt
  every start. The known **2.6× prefill variance across container restarts is
  therefore not explained by persisted autotune state** — there is none. It is
  per-process heuristic selection plus host/driver state. Not a memory-design
  input; recorded here so the question is closed.

### A1.7 Per-site inventory by subsystem

Lifetime classes use the A2 taxonomy. "Graph" = the buffer's address is baked
into a captured CUDA graph (prefill and decode are both graphified).

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
| `runtime/engine_scheduler.cpp` | prefill metadata (token ids, positions, block tables, ctx lens) | `chunk_len × 4 B` etc., pooled when it fits, else `cudaMallocAsync` | **per-request** | ✓ | ✗ |
| `runtime/engine_spec_*` | draft/verify staging + `spec_graphs_` | `k_max × …` per bucket | engine-persistent (invalidated) | ✓ | ✗ |
| `vision/` | tower weights, pixel buffer, embedding buffer | `Σ tower tensors`; `image_size²·3·2 B`; `num_image_tokens × d_model × 2 B` | model-resident | ✗ | via `VRAMAllocator` |
| `memory/layer_offload.cu` | double-buffered H2D layer staging | `2 × max_layer_bytes` | engine-persistent | ✗ | ✗ |
| `memory/weight_snapshot`, `weight_cache_file` | suspend/resume + warm-cache staging | host-side only | transient host-staging | ✗ | n/a |
| — | **library reservation (F4)** | **~3.9 GiB, constant** | engine-persistent | n/a | ✗ |

---

## A2. Lifetime taxonomy

Five tiers. The taxonomy is not aesthetic: **each tier gets exactly one
allocator whose discipline makes that tier's failure mode structurally
impossible.**

| Tier | Lifetime | Allocator | Failure mode made impossible | Address stability |
|---|---|---|---|---|
| **T1 Model-resident** | model load → unload | bump arena, freed wholesale | per-object leak (nothing is individually freed) | stable |
| **T2 Engine-persistent** | process | bump arena | per-object leak | stable |
| **T3 Pooled fixed-block** | request-scoped, refcounted | free-list over one slab | external fragmentation (all blocks identical) | stable |
| **T4 Forward-scratch** | one forward pass | LIFO stack | fragmentation (LIFO cannot fragment) + leak (stack unwinds) | stable per slot |
| **T5 Transient host-staging** | load only | ordinary host alloc | surviving load (asserted at phase transition) | n/a |

Confirmed against A1.7. Two corrections the inventory forced:

- **T1 and T2 need separate arenas even though both are "stable forever."**
  `server.model_swap` (shipped 2026-07-26) unloads a model and loads another
  without restarting the process. T1 must be releasable wholesale at swap;
  T2 (KV pool geometry, cuBLAS workspace, graph buffers) must survive it or be
  torn down explicitly. Collapsing them makes model swap a process restart.
- **T3 is a *group* of pools, not one.** The KV cache already runs two block
  groups (global + SWA), and the residual FP16 ring, SSM state and recurrent
  snapshots are all fixed-stride slabs with per-sequence slots. One
  `BlockPool<Stride>` template, four instantiations.

What does **not** get a tier: the library reservation (F4). It is not imp's
memory. It is a **charge** the planner subtracts before distributing anything
(A4).

Everything in A1.7 maps cleanly. The one entry that today straddles tiers is the
per-request `cudaMallocAsync` traffic in `engine_graph_decode.cpp` /
`engine_scheduler.cpp`: it is logically T4 but is implemented as driver calls on
the hot path. That is precisely the I2 violation, and moving it to T4 is what
fixes it.

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

### A3.1 L1 — Backend: VMM or `cudaMalloc`?

**Recommendation: `cudaMalloc` backend for every tier now; a VMM backend for the
KV block pool only, in migration step 7, gated on a WSL2 spike. Do not use VMM
anywhere else.**

Reasoning, from the measurements rather than from general principle:

*Against VMM everywhere.* The fragmentation problem VMM solves does not exist
here. Measured peak − steady state is +190 / +178 / +200 MiB across all three
configs (A1.3): imp allocates essentially everything once at init and holds it
until teardown. There is no allocate/free churn to fragment the heap. Paying
VMM's costs — 2 MiB minimum granularity, explicit
`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`/`cuMemSetAccess` lifecycle,
handle bookkeeping, and a driver API surface that behaves differently under
WDDM — to fix a problem that measures at 190 MiB is a bad trade.

*For VMM on the KV pool specifically.* The KV pool is where the actual defect
lives, and VMM is a direct structural fix for it, not a performance tweak.
Today the pool must be sized **before** the weight caches are built, from a
free-VRAM reading that is wrong by ~3.9 GiB (F3, F4). Every clamp in
`vram_budget.cpp` exists to guess that number better. With VMM the guess
disappears:

- `cuMemAddressReserve` the VA range for the **maximum** KV the config could
  ever want (`max_batch × max_seq_len`). Reserving address space costs no
  physical memory.
- Commit physical pages in coarse chunks (64–256 MiB, a multiple of the 2 MiB
  granularity, each covering many KV blocks) as the free-list runs dry.
- Decommit chunks when a chunk's blocks are all free and the pool has been
  under-subscribed for N steps.

The pool then *cannot* be mis-sized, because it is no longer sized — it is
bounded by the reservation and backed on demand. The planner's job shrinks from
"predict the residual exactly" to "prove the maximum commitment fits", which is
a statement about declared demand, not about a live measurement. The
`kv_max_blocks` clamp ladder (`vram_budget.cpp:455–539`) collapses to one check.

Addresses stay stable under VMM — that is the whole point of mapping into a
reserved VA range — so I3 is satisfied and graph-captured KV pointers remain
valid across growth. That is a property plain `cudaMalloc` growth cannot offer
at all.

*Why it is step 7 and not step 1.* Two things must be established first, and
neither is assumable:

1. **`cuMemCreate`/`cuMemMap` under WSL2/WDDM.** imp already has scars from this
   platform (`memcpyAsync` D2D costing a 165 µs host block; the driver update
   that broke GPU containers). A 200-line spike that reserves 24 GiB of VA,
   commits and decommits 256 MiB chunks, and verifies a graph-captured kernel
   still reads correct data after a growth, is a hard gate. If it fails, the
   `cudaMalloc` backend stays and the planner keeps a conservative fixed pool —
   the rest of this design is unaffected, which is why the backend is an
   interface.
2. **The tiers above must exist first**, or there is nothing to grow into.

*Not considered:* `cudaMallocAsync` pools as the backend. imp already pins the
default pool's release threshold to `UINT64_MAX`, which makes it a de-facto
arena with none of an arena's guarantees, and its reserved-vs-used split is a
recurring source of accounting confusion (A1.4 residuals). The design keeps the
async pool only for the weight-upload path during migration and retires it.

**Backend interface:**

```cpp
// src/memory/backend.h — the ONLY place in imp that calls the driver.
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

`Region` is `{void* base; size_t committed; size_t reserved; RegionTag tag;}` —
move-only, and the *only* type in imp that holds a raw device pointer obtained
from the driver.

### A3.2 L1 — the phase guard (I2)

```cpp
enum class AllocPhase { Loading, Planning, Serving };
```

Process-global, monotonic, set by the engine. `Backend::acquire()` consults it:

- `Loading` / `Planning` — allowed.
- `Serving` — **debug:** `IMP_ASSERT_FAIL` with the tag and a backtrace.
  **release:** increment `steady_state_allocations_total{tag}`, log once per tag
  at WARN, and proceed (never crash a production server over an accounting bug).

The counter is the I2 test surface: acceptance criterion 3 is
`steady_state_allocations_total == 0` after a soak. It is also the migration
progress bar — it starts at ~190 MiB worth of allocations per config and must
reach zero.

One deliberate exception: `AllocPhase::Serving` is temporarily re-entered as
`Planning` during `server.model_swap`, bracketed and logged.

### A3.3 L2 — Allocators, one per tier

```cpp
class ArenaAllocator;   // T1, T2 — bump; reset() frees wholesale
template <size_t Stride> class BlockPool;  // T3 — free-list over one slab
class ScratchStack;     // T4 — LIFO; scope guard restores the mark
```

`ArenaAllocator` and `BlockPool` already exist in `src/core/allocator.h` (A1.6,
dead). They are moved to `src/memory/`, given the Backend as their acquisition
source, and given the handle types below. This is the cheapest part of the whole
migration: the disciplines are already written and reviewed, they were just
never connected.

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

A forward pass opens one `Mark` at entry; every intermediate takes from the
stack; the `Mark` destructor rewinds. It cannot fragment (LIFO), cannot leak
(unwinds on exception too), and its high-water mark is exactly the number the
planner needs — so the planner sizes it from a **measured** warmup high-water
mark rather than from `max(attn, ffn, moe, ssm)` heuristics recomputed in three
places.

### A3.4 L3 — Handles: making the illegal states unrepresentable

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
// that promise (T1–T4). Friend-restricted ctor; no public ctor from T*.
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

A relocatable buffer is a `DeviceSpan`. Passing it where a `StableSpan` is
expected does not compile. That is the whole mechanism, and it is why the tier
allocators must be the only producers of `StableSpan` — the promise is theirs to
make.

Ownership:

```cpp
template <class T, Tier Ti> class Owned {   // move-only RAII
    ~Owned();                               // returns to its allocator
    StableSpan<T> span() const;             // T1..T4 only
};
```

`Owned<T, Tier::ModelResident>` is not convertible to
`Owned<T, Tier::EnginePersistent>`: a subsystem cannot smuggle a model-lifetime
buffer into engine-lifetime storage, so `server.model_swap` cannot leave a
dangling pointer behind.

Request-scoped blocks:

```cpp
class BlockRef {                     // move-only; NOT copyable
    KVBlockId id_; BlockPool<kKVBlockBytes>* pool_;
public:
    ~BlockRef();                     // dec_ref, exactly once
    BlockRef share() const;          // explicit inc_ref — the ONLY way to alias
};
```

The dispatch asks that a request-scoped block cannot outlive its request.
**Stated honestly: C++ cannot enforce that at compile time without lifetime
annotations.** What the design does enforce:

- `BlockRef` is move-only, so an accidental copy is a compile error and every
  additional referent is a visible, greppable `share()` call.
- Every `BlockRef` a request owns lives in that request's `SequenceSlot`. The
  slot's destructor asserts (debug) / counts (release) that its refcount
  contribution nets to zero.
- A soak-time invariant closes the residual gap: acceptance criterion 4
  (post-drain live blocks return to post-load baseline ±1 %) is exactly the
  test for "no block outlived its request."

Type system where it can carry the weight; asserted invariants plus a soak where
it cannot. Not claiming more than the language gives.

---

## A4. The planner

### A4.1 What is wrong today

`compute_vram_budget()` (`src/runtime/vram_budget.cpp`, 480 LOC) is described in
its own header as "pure computation — just arithmetic." It is pure in the
functional sense and completely impure in the useful sense: its dominant input
is `free_vram`, a live `cudaMemGetInfo` reading, and it is called **after** the
weights are uploaded and **before** the weight caches are built. So:

1. It sizes the KV pool from a free-VRAM number that still contains ~3.9 GiB of
   library reservation that will be claimed later (F4).
2. The pre-dequant phases then re-derive **their own** reserves from live free
   VRAM again — which is why #1100 exists: "the KV pool is allocated before the
   cache build, so its bytes are already gone from free_vram, and every one of
   them then came out of the decode cache a second time."
3. The engine works around the ordering with a **balloon**: a physical
   `cudaMalloc` held across `init_weights` purely to hide bytes from the KV
   planner, released just before phase 3
   (`engine_weight_upload.cpp:325`, `engine_kv_cache_init.cpp:434`). A balloon is
   what you build when you have no plan.
4. Six incident-driven clamps stack on the result: `target_blocks`, the
   post-weight `max_fit_blocks` backstop, `min_kv_blocks`, the `kv_fraction`
   affordability cap, the SWA batch-shaped charge, and the `#1103` allocator-
   headroom floor.

### A4.2 The replacement

```cpp
struct PlanInput {
    ModelShape        model;        // layers, dims, per-tensor byte demand
    FeatureSet        features;     // spec decode, vision, SWA, residual ring, LoRA
    ConcurrencyLimits limits;       // max_batch, max_seq_len, kv dtype/block size
    size_t            budget_bytes; // --vram-budget, or device total
    LibraryReserve    library;      // A1.5 — measured constant, not a guess
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

Three properties that the current code does not have:

- **`plan_memory` never calls `cudaMemGetInfo`.** Its only capacity input is
  `budget_bytes`. It is therefore fully testable on the host with no GPU, and
  its output is reproducible — the same config yields the same plan on every
  boot, ending the "free VRAM before the weight upload swings by 1.6 GB between
  identical invocations → different auto-batch → different KV clamp" trap
  recorded against #1103.
- **Allocation follows the plan in tier order**, and the KV pool is allocated
  from a **computed** residual rather than a measured one:
  `library reserve → T2 engine-persistent → T1 model-resident + weight caches →
  T3 pools → KV`. The balloon is deleted, not fixed.
- **It fails at load time with a report**, never mid-generation.

### A4.3 `--vram-budget` and the failure report

`--vram-budget` today is a *sizing view*
(`src/memory/vram_query.{h,cpp}`): it rewrites what `cudaMemGetInfo` returns so
the heuristics size smaller. Its own header calls it "best-effort hard cap, not
an OS limit ... leave ~1 GiB of real headroom." Under the new planner it becomes
what its name says: `budget_bytes` is the plan's total, the plan either fits it
or fails, and `Backend` refuses any acquisition that would exceed it. The
`cudaMemGetInfo`-rewriting view is retired along with the heuristics that needed
it.

`PlanFailure` carries the full arithmetic, and the operator-facing message says
what to change:

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

Not "VRAM budget: KV clamped 25600 -> 512 blocks", which is what the log says
today after the fact.

---

## A5. Subsystem boundaries

| subsystem | may hold | must request | must never touch |
|---|---|---|---|
| `compute/` | nothing | all buffers as `StableSpan`/`DeviceSpan` parameters | any allocation API; any static workspace |
| `exec/` | `Owned<_, EnginePersistent>` workspace handles; a `ScratchStack::Mark` per forward | scratch from the stack | driver calls; `cudaMallocAsync` |
| `model/` | `Owned<_, ModelResident>` weight handles | the T1 arena | KV, workspaces, driver calls |
| `quant/` | `Owned<_, ModelResident>` cache handles | the T1 arena, budgeted by the plan | live free-VRAM queries |
| `graph` (`runtime/cuda_graph.*`, `engine_graph_decode.cpp`) | graph + exec objects | `StableSpan` for everything captured | any allocation inside a capture region |
| `runtime/` | the plan, the allocators, the phase | — | per-request driver allocation |
| `vision/` | `Owned<_, ModelResident>` tower + `Owned<_, EnginePersistent>` staging | T1 + T2 | KV, executor workspaces |
| `api/` | nothing device-side | — | everything |

### A5.1 Paged KV + prefix cache + pinning — who owns a block

A KV block has **three concurrent referents** (COW-fork does not exist — A1.6):

1. **`seq_blocks_[seq_id]`** — an active sequence's positional block table.
2. **`block_hash_to_id_`** — the content-addressed prefix cache. Holds a block
   after its sequence is freed so a later sequence can reuse it.
3. **`pinned_blocks_` / `pin_refcount_`** — agentic prefix pinning, with its own
   FIFO owner list and budget, deliberately surviving `free_sequence()`.

Plus a fourth, out-of-process referent: `save_prefix_cache()` / `load_prefix_
cache()` serialise blocks to disk and re-register hashes on load.

**Ownership rule: the `BlockPool` owns the memory; nobody else does. All three
referents hold `BlockRef`s.** A block is returned to the free list when — and
only when — its last `BlockRef` is destroyed. There is no path that frees a
block by id.

Concretely:

- `seq_blocks_` becomes `std::vector<BlockRef>`; `free_sequence()` clears the
  vector, and the refs drop.
- The prefix cache holds its own `BlockRef` per entry. `evict_cached_block()`
  drops that ref; if a sequence is still using the block, nothing happens to the
  memory — which is the correct behaviour and today is achieved by a manual
  refcount plus a hand-written invariant (`free_block_dropping_stale_hash`,
  whose comment documents the exact double-ownership bug it exists to prevent:
  "a stale hash->id entry on a free-listed block lets a later prefix match
  inc_ref a block the allocator still hands out").
- The pin set holds `BlockRef`s keyed by owner. `unpin_prefix()` drops them.
  Budget eviction drops the front owner's refs. Pins surviving `free_sequence()`
  is then not a special case — it is just another live reference.

**Cancellation, disconnect, error paths** are where refcounts leak today,
because each path frees by hand. Under `BlockRef` they are all the same path:
the `SequenceSlot` is destroyed and its refs unwind, including on exception.
There is no `free_sequence()` to forget to call. The slot destructor asserts its
net contribution is zero (A3.4), and criterion 4's cancellation-heavy soak is the
system-level proof.

**Nuance the design must not lose:** `evict_middle_blocks()` (StreamingLLM)
replaces freed slots with sentinel `-1` while keeping the table *length* — the
attention kernels depend on positional alignment. So `seq_blocks_` is
`std::vector<std::optional<BlockRef>>`, not `std::vector<BlockRef>`, and the
sentinel is `nullopt`. Same for the SWA positional table, which is documented as
holding `-1` holes by design.

### A5.2 CUDA graph pool

Current state, verified:

| pool | keyed by | bound |
|---|---|---|
| `decode_graph_pool_[64]` | `n_sequences − 1` | `kMaxGraphPoolSize = 64`, fixed array |
| `prefill_graph_runner_` | — | 1 |
| `async_graph_runner_` | — | 1 (conditional-node loop) |
| `spec_graphs_` | `std::tuple<n_tokens, ctx_capacity, rec_slot>` | **no explicit cap**, but all three axes bucketed (see below); cleared wholesale by `free_spec_graphs_()` |

`spec_graphs_` is the only one without an explicit cap, and the first instinct —
"unbounded, `ctx_capacity` grows with the conversation" — is **wrong**. All three
key axes are bucketed: `n_tokens` to 3–5 draft buckets
(`spec_capture_bucket_`), `ctx_capacity` to power-of-two tiers from 4096 up to
`speculative.capture_ctx_cap` (`spec_capture_ctx_tier_`, ~6 tiers), and
`rec_slot` to `max_batch + 1`. Worst case is ~5 × 6 × (max_batch+1) graph
execs — ~1950 at `--max-batch 64`. Bounded, then, but **uncounted**: nothing in
the plan charges the graph memory those execs hold. The design gives it a
plan-derived LRU capacity so the bound is declared rather than emergent, and
counts it.

Interaction with I3: this is the sharpest constraint in the engine and the design
respects it directly — everything a graph captures is a `StableSpan`, so it comes
from T1–T4, all of which are arena- or pool-backed and never move. The
`workspace_generation` invalidation hook (which today exists because workspaces
*can* move) stays as a belt-and-braces assert, but it should never fire.

Interaction with `cudaDeviceGraphMemTrim`: graphs that capture stream-ordered
allocations acquire their own graph memory pool, invisible to the default pool's
`reserved`/`used` attributes. Two consequences: (a) once per-request
`cudaMallocAsync` is removed from the captured regions (A7 step 5), graph-owned
memory drops to zero and the trim calls in `cuda_graph.cu:366,1291` become dead;
(b) until then, `--mem-report` must query `cudaDeviceGetGraphMemAttribute` and
report it as its own line — it is currently part of the 20–39 % residual.

### A5.3 cuBLAS / CUTLASS workspaces

Current: three file-scope statics —
`gemm.cu:s_workspace` (64 MiB via a 64/32/8/2 try-down ladder),
`gemm.cu:s_bench_scratch` (32 MiB), and
`gemm_cutlass_sm120.cu:s_cutlass_workspace` (0, **grown lazily inside
`gemm_nvfp4_cutlass_sm120_impl` at GEMM time** — a `cudaFree` + `cudaMalloc` pair
on a code path that can run under graph capture).

**Design: shared from the engine-persistent (T2) arena, sized by the plan,
per-process — not per-handle and not per-stream.** Rationale: cuBLASLt takes a
workspace pointer + size as an argument, so one arena slice sized at the plan's
maximum serves every call; per-handle would multiply it by handle count for no
benefit, and per-stream is meaningless when imp runs one compute stream plus a
prefill stream.

The lazy CUTLASS growth path is deleted. `gemm_nvfp4_cutlass_sm120_workspace(M,
N, K)` already exists and is already used by the executor to pre-size
`qscratch_.cutlass_workspace`; the planner calls it over the shape set the model
can produce and takes the max. If a shape exceeds the plan, the call fails
cleanly and falls back — it does not allocate.

**On the 2.6× prefill variance:** the dispatch asks whether persistent autotuning
state explains it. It does not — there is no persistence (A1.6, verified: zero
file I/O in `gemm.cu`). Recording it here closes the question; it is not a design
input.

**Not covered by any of this: the ~3.9 GiB library reservation (F4).** It is not
a workspace imp allocates; it is claimed by the libraries themselves on first
dispatch. The plan charges it (A4.2) and `--mem-report` names it. Reducing it is
out of scope for this work.

### A5.4 Vision tower

Currently **resident**: `VisionPipeline::init()` runs during
`init_features()`/warmup whenever `--mmproj` is given, loading the tower and
pre-allocating both the pixel buffer and the embedding buffer through
`VRAMAllocator`. Measured cost on the gemma-3-4b pair: **+1610 MiB** at
`04_features` (A1.4) — for a server that may never receive an image.

**Design: keep it resident, but make it a planned, declared T1 cost.** Lazy
loading is rejected: the tower is model-resident weights, so a first-image
request would have to allocate ~1.6 GiB *while serving*, which violates I2
outright and would need admission control for a memory event that has nothing to
do with request size. Declared-and-resident is the honest trade — and if an
operator does not want the cost, not passing `--mmproj` is already the switch.

`VisionPipeline` keeps one genuine hot-path hole to fix: `vision_pipeline.cpp:97`
falls back to a raw `cudaMalloc` per image if the pre-allocated pixel buffer was
too small. That becomes a `ScratchStack` take that fails cleanly.

### A5.5 Speculative decoding

Draft/verify staging buffers and `spec_graphs_` are T2 (engine-persistent), sized
by the plan from `speculative.k` / `suffix_k_max` / MTP depth. They are already
invalidated together (`free_spec_buffers_` → `free_spec_graphs_`), which is the
right coupling.

**Per-request toggling** (`spec-ngram: gates failed ...` in the logs — spec is
disabled per request by sampling params) must not resize anything: the buffers
are planned for the maximum `k` the config allows and are simply unused when a
request does not qualify. This is already how it behaves; the design makes it a
stated invariant rather than an emergent one, so nobody "optimises" it into a
per-request allocation later.

---

## A6. Testability

`Backend` is the substitution seam. `FakeBackend` allocates host memory
(`std::aligned_alloc`) and hands out the same `Region` type, so every allocator,
the planner, and all the refcount logic run on CPU-only CI — which is what imp
has (no GPU runner; the CI lane is `ctest -L unit`).

`FakeBackend` provides:

- A configurable capacity, so budget-exhaustion paths are testable without a
  32 GiB card.
- A full allocation journal: `(seq, phase, tag, bytes, op)` for every acquire /
  release / commit / decommit.
- Poison-on-release (`0xDE`) so a use-after-free is a deterministic data
  comparison, not a GPU fault.
- Injectable failure: fail the *n*-th acquisition, to exercise the rollback
  paths that today are hand-written per call site
  (`rollback_partial_allocation`, the `moe_3x_packed`/`sf` unwind in
  `executor_workspace_buffers.cu:641`, and the KV `allocate_blocks` rollback).
- Growth simulation for the VMM path: `commit`/`decommit` succeed or fail on
  command, and the fake asserts the base address never changes — the host-side
  proof of I3 under growth.

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

V8 is the one that makes the migration safe: record the journal from a real GPU
run once per model config (the harness in A1 already produces the phase deltas),
check it in, and assert the planner covers it. A plan that under-provisions is
then a CI failure, not a production OOM.

Acceptance criterion 8 (peak VRAM per model config vs checked-in thresholds)
sits on top: a GPU-side job, run manually or on a self-hosted runner, comparing
`--mem-report` totals against `tests/vram_thresholds.json`.

---

## A7. Migration plan

Strangler fig. The I1 allowlist starts at **365 sites / 74 files** and shrinks
monotonically; `tools/check_alloc_sites.py` fails the build if a file not on the
list allocates, or if the list grows. Every step leaves the tree green, keeps
decode and prefill within 1 %, and records both in `docs/audit/PERF_LOG.md`.

| # | Step | Removes from allowlist | Why here |
|---|---|---:|---|
| 0 | `Backend` + `FakeBackend` + phase guard + `check_alloc_sites.py` (allowlist = everything, gate green from day one). No behaviour change. | 0 | The gate must exist before anything moves, or the list can silently grow. |
| 1 | Move `ArenaAllocator`/`BlockPool` to `src/memory/`, wire to `Backend`, add `StableSpan`/`Owned`/`BlockRef`. Still unused. | 0 | Pure addition; A1.6 says the code already exists and is dead. |
| 2 | **`plan_memory()` alongside `compute_vram_budget()`**, computing but not applying. Log both; assert agreement within a tolerance in CI via V8. | 0 | Establishes the plan is right *before* anything depends on it. Highest-information, lowest-risk step. |
| 3 | **KV pool + `KVCacheManager`** → `BlockPool` + `BlockRef`. Deletes the manual refcount, `free_block_dropping_stale_hash`, and the free-by-id paths. | ~15 | The hardest ownership question (A5.1) and the one with the best existing test coverage. Doing it early means the refcount machinery is proven before four more subsystems depend on it. |
| 4 | **Executor workspaces** (`exec/executor_workspace*.cu`, 47+ sites in one file) → T2 arena + `ScratchStack`. | ~70 | Biggest single-file win; the sizes are already computed centrally in `compute_shared_sizes`. |
| 5 | **Per-request allocations** (`engine_graph_decode.cpp`, `engine_scheduler.cpp`, `executor_attention_prefill.cu`, `executor_attention.cu`, MoE per-call arrays) → `ScratchStack`. **This is the step that satisfies I2**; the counter from step 0 must reach zero. | ~40 | Depends on the stack existing (4) and on the plan sizing it (2). |
| 6 | **Weight upload + pre-dequant caches** (`model/`, `quant/`, `exec/pre_dequant_*`) → T1 arena. **Deletes the balloon** and the phase-local free-VRAM re-derivation. Switch the allocation order to the plan's tier order. | ~90 | Largest blast radius; must come after the plan is trusted (2) and the KV pool no longer competes for a residual (3). |
| 7 | **VMM backend for the KV pool**, gated on the WSL2 spike (A3.1). If the spike fails, stop here — everything above stands. | 0 | Optional by construction. |
| 8 | **`compute/` statics**: cuBLAS/cuBLASLt/CUTLASS workspaces from the T2 arena; delete the lazy CUTLASS growth path. | ~115 | Mostly mechanical once the arena exists; `compute/` sites are small per-kernel scratch. |
| 9 | **Guardrails**: `--vram-budget` as a real cap, admission control per I6, `/metrics` tagged breakdown per I7, `--mem-report`, the peak-VRAM CI gate. | remainder | Needs everything above to have real numbers to report. |

Ordering rationale in one line each: the gate before the moves (0); the tools
before the users (1); the plan before it is trusted (2); the hardest ownership
problem while it is still isolated (3); the biggest single file (4); I2 once
there is somewhere for scratch to go (5); the balloon last among the big
consumers, because it is the one whose removal changes init ordering (6); the
optional backend behind a gate (7); the mechanical sweep (8); the operator
surface once it has something true to say (9).

Steps 3, 4 and 6 each need a coherence check (`check-degeneration`) — they touch
the KV cache, the forward pass and the weight caches respectively.

---

## Invariant compliance

| | Invariant | Today | After |
|---|---|---|---|
| I1 | Single acquisition point | ✗ — 365 sites / 74 files outside `src/memory/` | ✓ — `Backend`, allowlist empty, CI gate |
| I2 | No allocation on the hot path | ✗ — measured +190 MiB/config of steady-state allocation | ✓ — `ScratchStack`, phase guard, counter == 0 |
| I3 | Stable addresses for graph memory | ~ — true in practice, enforced by comments + a `workspace_generation` hook | ✓ — `StableSpan` in kernel signatures; no conversion from `DeviceSpan` |
| I4 | Capacity planned, not discovered | ✗ — live `cudaMemGetInfo`, a balloon, six stacked clamps | ✓ — `plan_memory()` never queries the device; fails at load with a report |
| I5 | Unidirectional ownership | ✗ — `VRAMAllocator` is a tracker; raw `void*` cross module boundaries | ✓ — `Owned<T, Tier>`, no cross-tier conversion, no raw device pointers above L1 |
| I6 | OOM is typed and recoverable | ~ — `RequestStatus::CANCELLED`, plus a warning that fires *after* prefill | ✓ — plan-time failure at load; admission-time 429/503 at runtime |
| I7 | Capacity ≠ occupancy | ✗ — 20–39 % of device memory unattributed | ✓ — per-tier reserved *and* live, library reserve named, ≥95 % accounted |

Nothing in the invariant set had to be dropped. Two are weakened in a stated way:
I3 is enforced by the type system for *stability* but the graph-invalidation hook
stays as a runtime assert; I5's "request-scoped block cannot outlive its request"
is type-enforced against aliasing and assert-plus-soak-enforced against outliving
(A3.4).

---

## Open questions for Phase B

1. **WSL2 VMM spike** (A3.1) — gates step 7 only.
2. **`LibraryReserve` calibration** — the ~3.9 GiB constant is measured on this
   driver/CUDA/card. It needs a boot-time self-check (compare assumed vs actual
   after the first forward, emit a metric on >10 % divergence) rather than a
   hardcoded number, and a documented procedure for re-measuring after a driver
   or CUDA bump.
3. **`ScratchStack` under concurrency** — imp runs one compute stream plus a
   prefill stream. Two stacks, or one with a mutex? Lean toward two (one per
   stream), which keeps the LIFO discipline per-stream and avoids a lock on the
   forward path. Decide with the step-5 measurement.

---

## Provenance

Every number in A1 was measured on this host on 2026-07-28 against `imp:test`
built from the working tree at `fix/1104-json-number-grammar` — that is `main`
plus the two `#1104` constrain commits (no memory impact) plus the **staged,
uncommitted `#1103` fix** to `vram_budget.cpp`/`vram_query.h` that floors the
mode-2 reserve at the allocator's 5 % headroom. Measurements taken with the GPU
otherwise idle (0 containers, no compute processes) and healthy under load
(2857–2932 MHz SM, 13801 MHz mem, 310–444 W). Harness: `MemAccount` via
`diagnostics.vram_audit`, driver `tools/analysis/vram_audit_load.py`.

Findings, including the refuted ones, are recorded in `AUDIT.md`.
