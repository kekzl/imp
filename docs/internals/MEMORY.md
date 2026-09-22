<!--
layer: L2
audience: kernel-devs
verified: 2026-09-22
commit: 9cbb8004
-->

# imp Memory Architecture

Target: `sm_120a` (RTX 5090 / GB202, 32 607 MiB) with the `sm_120f` PTX fallback. No SM80/SM90 paths.

Design: five lifetime tiers (A2) over a three-layer stack (A3), a planner that never queries the device (A4), subsystem ownership rules (A5), a CPU-only test seam (A6), seven invariants I1-I7. Site census that motivated the design, the migration steps, and the full implementation/divergence log (B0, B1-B8, D1-D16): [`docs/archive/memory_census_2026.md`](../archive/memory_census_2026.md).

---

## A2. Lifetime taxonomy

Five tiers. Each gets exactly one allocator whose discipline makes that tier's failure mode structurally impossible.

| Tier | Lifetime | Allocator | Failure mode prevented | Address stability |
|---|---|---|---|---|
| T1 Model-resident | model load -> unload | bump arena, freed wholesale | per-object leak | stable |
| T2 Engine-persistent | process | bump arena | per-object leak | stable |
| T3 Pooled fixed-block | request-scoped, refcounted | free-list over one slab | external fragmentation | stable |
| T4 Forward-scratch | one forward pass | LIFO stack (`ScratchStack`) | fragmentation + leak | stable per slot |
| T5a Transient host-staging | load only | ordinary host alloc | surviving load | n/a |
| T5b Engine-persistent pinned host | process | `PinnedBuffer` / `HostRegistration` | per-object leak, asymmetric unregister | n/a |

Corrections the site census (archive A1.7) forced:

| Correction | Why |
|---|---|
| T5 splits a/b | 26 pinned-host sites are reused every decode step, so T5a's "load only" discipline could not hold them; moved to T5b |
| T1 and T2 need separate arenas | `server.model_swap` unloads/loads without a process restart: T1 releases wholesale at swap, T2 survives it |
| T3 is a group of pools, not one | KV runs two block groups (global + SWA); residual FP16 ring, SSM state and recurrent snapshots are separate fixed-stride slabs. One `BlockPool` class, four instances (stride is a runtime constructor argument, not a template parameter) |

No tier holds the library reservation: not imp's memory, a **charge** the planner subtracts before distributing anything (A4). Per-request `cudaMallocAsync` traffic (`engine_graph_decode.cpp`/`engine_scheduler.cpp`) is logically T4; since 2026-09-07 it is one T2 pool carved at init (`Engine::init_serving_metadata_pool_()`, `runtime/serving_metadata_layout.h`), the driver calls remaining only as the fallback for a pool that failed to allocate.

---

## A3. Layer design

| Layer | Responsibility | Not responsible for |
|---|---|---|
| L3 Handles | typed RAII ownership, stability in the type system | sizing, policy, physical acquisition |
| L2 Allocators | one per lifetime tier (A2) | talking to the driver, deciding how much |
| L1 Backend | physical acquisition, phase guard, accounting | lifetime, tiering, policy |

All in `src/memory/`; nothing above L1 calls the driver.

### A3.1 L1 Backend: VMM or `cudaMalloc`

`cudaMalloc` backend for every tier; VMM backend for the KV block pool only (`kv_cache.growable`, `src/core/config/kv_cache.h`, default **on**; `growable_initial_pct` default **25** - commits a fraction at startup, grows at admission, capped so it cannot overshoot into a WDDM spill).

| Decision | Reason |
|---|---|
| No VMM elsewhere | peak minus steady state is +190/+178/+200 MiB (archive A1.3); imp allocates once at init and holds until teardown, no churn to fragment against VMM's 2 MiB granularity and lifecycle cost |
| VMM for the KV pool | the pool must be sized before the weight caches exist, from a free-VRAM reading wrong by ~3.9 GiB (archive 0, F3/F4); `cuMemAddressReserve` the maximum KV the config could want, commit 64-256 MiB chunks on demand, decommit when idle |
| WSL2/WDDM spike: GO (2026-07-29, `tools/analysis/vmm_wsl2_probe.cu`, 24/24 checks) | 24 GiB reserve costs 0 MiB physical; commit granularity 2 MiB (not 64 KiB); base address invariant across an 8-chunk grow/shrink cycle; a graph captured over a fixed VA survives +1.5 GiB growth with the same checksum three times; decommit costs ~2x commit (~2.4-2.6 ms per 256 MiB) |
| Not `cudaMallocAsync` as the backend | imp already pins its default pool's release threshold to `UINT64_MAX` (a de-facto arena without an arena's guarantees); its reserved-vs-used split is a recurring accounting-confusion source |

```cpp
// src/memory/backend.h - the ONLY place in imp that calls the driver.
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::expected<Region, MemError> acquire(size_t bytes, Alignment a, RegionTag tag) = 0;
    virtual void release(Region&&) = 0;
    virtual std::expected<void, MemError> commit(Region&, size_t new_bytes) = 0;   // VMM only
    virtual void decommit(Region&, size_t new_bytes) = 0;
    virtual BackendStats stats() const = 0;
};
```

`Region = {void* base; size_t committed; size_t reserved; RegionTag tag;}`: move-only, the only type holding a raw device pointer obtained from the driver.

### A3.2 L1 phase guard (I2)

`enum class AllocPhase { Loading, Planning, Serving }` (`backend.h`), process-global, monotonic. Every acquisition entry point (`acquire()`, `acquire_growable()`, `commit()`, `commit_range()`) checks it before proceeding.

| Phase | Debug build | Release build |
|---|---|---|
| Loading / Planning | allowed | allowed |
| Serving | `IMP_LOG_ERROR` + `std::abort()` | `steady_state_allocations(tag)` incremented, `IMP_LOG_WARN` once per tag, request proceeds |

One deliberate exception: `Serving` is temporarily re-entered as `Planning` during `server.model_swap`, bracketed and logged. Growth is an acquisition too (#1649: `commit`/`commit_range` are guarded the same way): before this, a growable pool committing pages under load was invisible to the phase counter, the `--wrap` interposer and `check_alloc_sites.py` alike.

### A3.3 L2 Allocators

| Class | Tier | Discipline |
|---|---|---|
| `ArenaAllocator` | T1, T2 | bump; `reset()` frees wholesale |
| `BlockPool` | T3 | free-list over one slab; stride is a runtime constructor argument |
| `ScratchStack` | T4 | LIFO; RAII `Mark` rewinds on scope exit |

A forward pass opens one `ScratchStack::Mark` at entry; every intermediate takes from the stack; the mark's destructor rewinds.

- Cannot fragment (LIFO), cannot leak (unwinds on exception too).
- Its high-water mark sizes the planner's T4 charge from a measured warmup, not the `max(attn, ffn, moe, ssm)` heuristic recomputed in three places.

### A3.4 L3 Handles (I3, I5)

| Type | Guarantee |
|---|---|
| `DeviceSpan<T>` | non-owning view, may point at anything, cheap and copyable |
| `StableSpan<T>` | non-owning view whose address is stable for the region's lifetime; constructible only by a tier allocator (`detail::StableKey` passkey, friended to `ArenaAllocator`/`ScratchStack`/`BlockPool`); widens to `DeviceSpan` implicitly, never narrows back, no raw-pointer constructor |
| `BlockRef` | move-only, request-scoped KV reference; `~BlockRef()` decrements exactly once; `share()` is the only way to alias (explicit inc_ref) |
| `GraphSlotLease` | move-only lease over the T2 conditional-graph-loop slot pool; releases on destruction |

Every graph-capturable kernel launch wrapper takes `StableSpan`, so a relocatable `DeviceSpan` passed where a `StableSpan` is expected does not compile. Ownership tagging is runtime, not compile-time: the arena carries a `RegionTag` and a `generation()` counter, so a stale pointer surviving an engine teardown (`server.model_swap`) is caught by an assert on the generation, not by the type system.

"A block cannot outlive its request" is not compile-time enforceable in C++: `BlockRef` makes an accidental copy a compile error, and each request's `SequenceSlot` destructor asserts (debug) / counts (release) that its net refcount contribution is zero.

**Platform fact behind I4's other half:** within a process, free VRAM only ever decreases.

- After a load -> generate -> free cycle every CUDA-level release succeeds (async pool trims to 0, graph memory zero) but `cudaMemGetInfo` never recovers for the process's life, WSL2/WDDM does not hand a process's peak commitment back.
- "Live blocks return to baseline" is what criterion 4 checks; "device-used returns to baseline" is not achievable here by any allocator design.
- Bandwidth, not a `cudaMalloc` probe, tells resident from spilled (~1530 GB/s resident vs ~237 GB/s spilled).

---

## A4. The planner

| Problem the old pass has (`compute_vram_budget()`, `src/runtime/vram_budget.cpp`) | Consequence |
|---|---|
| Dominant input is a live `cudaMemGetInfo`, read after weights upload, before caches build | KV pool sized against a number that still contains ~3.9 GiB of library reservation claimed later |
| Pre-dequant phases re-derive their own reserve from live free VRAM again | #1100: cache bytes counted twice against the same headroom |
| A physical `cudaMalloc` balloon held across `init_weights`, released before phase 3 | hid bytes from the KV planner instead of ordering the build correctly (deleted, archive B0 6.6) |
| Six incident-driven clamps stacked on the result | `target_blocks`, `max_fit_blocks`, `min_kv_blocks`, `kv_fraction` cap, SWA batch charge, `#1103` headroom floor |

```cpp
// src/memory/plan.h - current fields, condensed
struct PlanInput {
    ModelShape model; FeatureSet features; ConcurrencyLimits limits;
    LibraryReserve library;          // measured constant, not a guess
    size_t budget_bytes;             // --vram-budget, or the device total
    size_t context_bytes;            // CUDA primary context + driver (measured, ~1679.6 MiB)
    size_t forward_scratch_bytes;    // T4 high-water: a recorded warmup, else a conservative estimate
    size_t engine_persistent_bytes;  // workspaces, cuBLAS/CUTLASS scratch, graph buffers
};
struct MemoryPlan {
    size_t model_resident;      // T1: weights + mandatory weight caches
    size_t optional_caches;     // rest of the weight-cache demand, granted above the KV floor
    size_t engine_persistent;   // T2
    size_t forward_scratch;     // T4
    KvPlan kv;                  // blocks, bytes, below_floor (loud, not silent, on underflow)
    std::vector<PlanLine> pools;         // SWA group, SSM state, residual ring, ...
    size_t library_reserve; size_t context_reserve;
    size_t total() const;
    std::vector<PlanLine> lines() const;  // every line item, largest first - the --mem-report body
};
PlanResult plan_memory(const PlanInput& in);   // pure, deterministic, never touches the device
```

| Property `plan_memory()` has that the old pass does not |
|---|
| Never calls `cudaMemGetInfo`: only capacity input is `budget_bytes`; testable on the host with no GPU; same config gives the same plan on every boot |
| Allocation follows tier order (library reserve -> T2 -> T1 + weight caches -> T3 -> KV), computed residual, no balloon |
| Fails at load time with an itemised report (`PlanFailure::report()`: every line plus the three largest levers), never mid-generation |

`--vram-budget <mb>` (`tools/imp-cli/args.cpp`) still installs a process-wide cap via `vram_budget_install()` (`src/memory/vram_query.h`): `free' = min(free_now, budget - my_used)`, its own header calling it "best-effort hard cap, not an OS limit ... leave ~1 GiB of real headroom." `plan_memory()`'s `budget_bytes` is fed from the same distributable-VRAM probe (`runtime/plan_shadow.cpp`), so the plan's report is bounded by it even though the plan itself never calls `cudaMemGetInfo`.

---

## A5. Subsystem boundaries

| Subsystem | May hold | Must request | Must never touch |
|---|---|---|---|
| `compute/` | nothing | buffers as `StableSpan`/`DeviceSpan` parameters | any allocation API; static workspace |
| `exec/` | T2 workspace handles; one `ScratchStack::Mark` per forward | scratch from the stack | driver calls; `cudaMallocAsync` |
| `model/` | T1 weight handles | the T1 arena | KV, workspaces, driver calls |
| `quant/` | T1 cache handles | the T1 arena, budgeted by the plan | live free-VRAM queries |
| `runtime/cuda_graph.*`, `engine_graph_decode.cpp` | graph + exec objects | `StableSpan` for everything captured | allocation inside a capture region |
| `runtime/` | the plan, the allocators, the phase | - | per-request driver allocation |
| `vision/` | T1 tower + T2 staging | T1 + T2 | KV, executor workspaces |
| `api/` | nothing device-side | - | everything |

### A5.1 Paged KV + prefix cache + pinning

A KV block has **two** refcount holders, not three (COW-fork does not exist, archive A1.6): `seq_blocks_[seq_id]` (an active sequence's positional block table) and the cached LRU (`cached_blocks_lru_`, the content-addressed prefix cache; holds a `BlockRef` per entry). `pinned_blocks_`/`pin_refcount_` is an **eviction-policy overlay**, not a holder: a pinned block stays alive because the cached LRU already holds it at count 1, pinning only stops `reclaim_cached_block()` rotating past it.

**Rule: the `BlockPool` owns the memory; nobody else does.** A block returns to the free list when and only when its last `BlockRef` is destroyed.

- `free_sequence()` moves its reference into the cache or lets it drop, no "skip the free" branches.
- StreamingLLM eviction (`evict_middle_blocks()`) must keep the block-table *length* for kernel positional alignment, so `seq_blocks_` is `std::vector<std::optional<BlockRef>>`, `nullopt` the hole.

**The KV-pressure valve counts reclaimable blocks (#1879).** Until 2026-09-03 the "pool over 90% full" check compared the free list against the blocks live sequences hold, so a pool one third full of reclaimable prefix-cache blocks read as full and every wave after the first ran eager (measured: 2387 -> 1443-1485 tok/s). Fixed by adding `num_reclaimable_cached_blocks()` to the comparison.

### A5.2 CUDA graph pool

| Pool | Keyed by | Bound |
|---|---|---|
| `decode_graph_pool_[64]` | `n_sequences - 1` | `kMaxGraphPoolSize = 64`, fixed array |
| `prefill_graph_runner_` | - | 1 |
| `async_graph_runner_` | - | 1 (conditional-node loop) |
| `spec_graphs_` | `(n_tokens, ctx_capacity, rec_slot)` | uncounted, ~1950 execs at `--max-batch 64`; cleared wholesale by `free_spec_graphs_()` |

`cudaDeviceGraphMemTrim`: measured 2026-07-29, `used=reserved=high_since_serving=0 MiB` after 12 serving requests - no captured region allocates today, so `cuda_graph.cu`'s trim calls are dead code and graph memory is not part of the residual.

### A5.3 cuBLAS / CUTLASS workspaces

Shared from the T2 arena, sized by the plan, per-process (not per-handle, not per-stream: one compute stream plus one prefill stream).

- `gemm.cu`'s two statics (64 MiB workspace, 32 MiB bench scratch) moved to T2.
- The grouped-GEMM reserve was guesswork at 512 MiB against a measured 152 320 B (170 SMs x 896 B persistent-scheduler state), now 1 MiB, freeing 488 MiB on every MoE model.
- The lazy CUTLASS growth path (`cudaFree`+`cudaMalloc` at GEMM time, unsafe under graph capture) is deleted: `gemm_nvfp4_cutlass_sm120_workspace(M, N, K)` pre-sizes, the planner takes the max over the model's shape set.
- The FP32 LM head caller needs 0 bytes at every shape measured, pinned by the `CutlassWorkspaceContract` test suite.

### A5.4 Vision tower

Resident: loaded during `init_features()`/warmup whenever `--mmproj` is given, or always when the checkpoint carries its own tower (Qwen3-VL).

- Measured cost: +1610 MiB at `04_features` on the gemma-3-4b pair (archive A1.4).
- `runtime.vision_max_patches` (default 4096, ~1024x1024) bounds the image-token budget every encoder workspace is sized from, a hard ceiling: an oversized image is scaled down rather than refused.
- **Kept resident by design:** lazy loading would allocate ~1.6 GiB *while serving* on the first image (an I2 violation) and would need admission control for a memory event unrelated to request size.

### A5.5 Speculative decoding

Draft/verify staging and `spec_graphs_` are T2, sized by the plan from `speculative.k`/`suffix_k_max`/MTP depth, invalidated together. Per-request spec toggling must not resize anything: buffers are planned for the config's maximum `k` and simply unused when a request does not qualify.

---

## A6. Testability

`Backend` is the substitution seam. `FakeBackend` allocates host memory (`std::aligned_alloc`) and hands out the same `Region` type, so the allocator stack, the planner and the refcount logic run on CPU-only CI (`ctest -L unit`, no GPU runner).

| `FakeBackend` provides |
|---|
| Configurable capacity: budget-exhaustion paths testable without a 32 GiB card |
| Full allocation journal: `(seq, phase, tag, bytes, op)` per acquire/release/commit/decommit |
| Poison-on-release (`0xDE`): use-after-free becomes a deterministic data comparison, not a GPU fault |
| Injectable failure (fail the n-th acquisition): exercises rollback paths hand-written per call site otherwise |
| Growth simulation for the VMM path: commit/decommit succeed or fail on command; asserts the base address never changes |

| # | Invariant | Test shape |
|---|---|---|
| V1 | Conservation | journal replay: sum(acquired) - sum(released) == live bytes, after every op |
| V2 | No allocation in `Serving` | synthetic decode loop; journal has no acquire with `phase == Serving` |
| V3 | Arena resets free wholesale | after `reset()`, live bytes attributable to that arena == 0 |
| V4 | Block pool conservation | randomised churn; free_count + live refs == num_blocks, always |
| V5 | Refcount balance under faults | exception injected at every lifecycle point; net refcount delta == 0 |
| V6 | LIFO discipline | `ScratchStack` marks rewind in reverse order; out-of-order rewind is a hard failure |
| V7 | Plan determinism | `plan_memory` is pure: same input -> byte-identical plan, 1000 randomised configs |
| V8 | Plan sufficiency | replay a recorded real allocation journal against the plan; no tier exceeded |
| V9 | Stability under growth | VMM fake: commit/decommit across a 10x growth; `Region::base` invariant |

V8 is the migration safety net: record a journal from a real GPU run once per model config, check it in, assert the planner covers it - an under-provisioned plan is then a CI failure, not a production OOM. Acceptance criterion 8 (peak VRAM vs checked-in thresholds) gates `own_peak_mb` (`tests/perf_baseline.json`, this process's allocations since init) rather than the `--mem-report` device total, in `scripts/verify.sh` next to the perf gate.

---

## Invariant compliance

| ID | Invariant | Status | Target |
|---|---|---|---|
| I1 | Single acquisition point | partial - device-memory sites remain outside `src/memory/` (`tools/alloc_allowlist.txt`); the pinned-host class is closed, every site moved to T5b's `PinnedBuffer`/`HostRegistration` | `Backend`, empty allowlist, CI gate |
| I2 | No allocation on the hot path | partial - `make check-alloc-interpose` (2026-09-07): the ragged-prefill/constrained-decoding path reads 0 driver calls while serving; the upload family (block tables, token ids, positions, M-RoPE) is one T2 pool | `ScratchStack`, phase guard, counter == 0 |
| I3 | Stable addresses for graph memory | partial - `StableSpan` exists and is passkey-enforced, so only a tier allocator can mint one; some kernel signatures still take raw pointers | `StableSpan` in every kernel signature, no `DeviceSpan` conversion |
| I4 | Capacity planned, not discovered | partial - `plan_memory()` is pure and deterministic (byte-identical across 5 identical starts); the balloon is deleted; the KV block count is the plan's, the live pass only a fallback | `plan_memory()` never queries the device; fails at load with a report |
| I5 | Unidirectional ownership | partial - KV blocks and the T2/T3 tiers own through move-only RAII (`Region`, `BlockRef`, `GraphSlotLease`); raw `void*` still crosses boundaries in `exec/` and `compute/` | `Owned`-style non-copyable handles everywhere, no raw device pointers above L1 |
| I6 | OOM is typed and recoverable | done - plan-time refusal at load with the block arithmetic; admission-time `IMP_ERROR_CAPACITY` -> HTTP 503 `capacity_error`, distinct from a client cancel | plan-time failure at load; admission-time 429/503 |
| I7 | Capacity != occupancy | partial - per-tier reserved and live served on `/metrics`, plus KV blocks and budget-vs-own; accounting improves once the library reserve is charged at its measured value rather than a fixed constant | per-tier reserved and live, library reserve named, >=95 % accounted |

Nothing in the set was dropped.

- I3 is type-enforced for stability with the graph-invalidation `generation()` check kept as a runtime assert rather than a compile-time one.
- I5's "a request-scoped block cannot outlive its request" is type-enforced against aliasing (`BlockRef` move-only) and assert-plus-soak-enforced against outliving (A3.4), not compile-time enforced end to end.
- Dated measurements behind every "partial" and the full divergence log (D1-D16, B1-B8): [`docs/archive/memory_census_2026.md`](../archive/memory_census_2026.md).

---

## Open questions

| # | Question | Note |
|---|---|---|
| 1 | WSL2 VMM spike | resolved GO (A3.1); gated migration step 7 only |
| 2 | `LibraryReserve` calibration | the ~3.9 GiB constant is measured on this driver/CUDA/card; needs a boot-time self-check (assumed vs. actual after the first forward, on >10 % divergence) plus a documented re-measure procedure after a driver or CUDA bump |
| 3 | `ScratchStack` under concurrency | one compute stream plus a prefill stream: two stacks (one per stream, no lock, LIFO stays per-stream) vs. one stack with a mutex - undecided |

---

## Provenance

The site census this design was measured against (archive A1) was taken 2026-07-28 on `imp:test` (`main` + the two `#1104` constrain commits + the staged `#1103` budget fix), GPU otherwise idle, healthy under load (2857-2932 MHz SM, 13801 MHz mem, 310-444 W).

- Harness: `MemAccount` via `diagnostics.vram_audit`; driver `tools/analysis/vram_audit_load.py`.
- Findings including refuted ones: `AUDIT.md`.
