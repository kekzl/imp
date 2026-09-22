<!--
layer: L1
audience: records
verified: 2026-09-22
commit: 9cbb8004
-->

# Memory subsystem census 2026

Moved verbatim from `docs/internals/MEMORY.md` during the docs prose cleanup (2026-09-22). Record of the state the memory redesign was measured against and its implementation log. Current design: [`docs/internals/MEMORY.md`](../internals/MEMORY.md).

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

**RESOLVED 2026-09-04 (#1899): on Q8_0 configs this was imp's own allocation, not a library one.** `mmq_q8_imma`'s `imma_ensure_weight()` `cudaMalloc`s an s8 + (α, β) SoA copy of every Q8_0 weight it prefills - 1.125 B per element, taken on that tensor's first prefill, untracked and best-effort (a failed `cudaMalloc` silently drops that GEMM to the dequant path). That is why the delta is invariant to batch and context, why `CUDA_MODULE_LOADING=EAGER` moved nothing, and why the default async pool was clean: all three sub-hypotheses were refuted because the answer was a direct `cudaMalloc` in `src/compute/`.

Discriminator, one flag, same tree and model (Qwen3-8B-Q8_0, cold, v0.37.0):

| arm | measured "library reserve" | forward window |
|---|---:|---:|
| defaults | 6780 MiB | 5612 MiB |
| `gemm.q8_imma_enabled=false` | **967 MiB** | **5 MiB** |

And it was never a constant: the same model measured 5612 / 7839 / 7942 MiB at `vram.library_reserve_mb` = default / 6782 / 12000, because the cache takes whatever the KV pool left free. Models without a Q8_0 prefill path never showed it (Qwen3-14B-Q6_K 1366, Qwen3-8B-NVFP4 1535, Qwen3.8-27B-NVFP4 3260 MiB, all with a ~0 MiB forward window).

**Design consequence:** `compute_vram_budget()` charges the planes as `imma_plane_bytes` (the same arithmetic `imma_q8_plane_bytes()` allocates, capped at what is left after the KV guarantee), the engine caps the allocator at the charge via `mmq_q8_imma_set_plane_budget()`, and `--mem-report` shows them as `imma_q8_planes`. What remains under "library reserve" is the real library claim: **763 MiB** on that config, and the measurement is now repeatable (charged 762, measured 763). The planner still charges a measured, arch-and-driver-specific *library reservation* as a first-class line item (A4); `--mem-report` attributes it explicitly instead of dumping it into a residual (A5.3).

### A1.6 Verified-dead and verified-clean (do not re-chase)

- **`ArenaAllocator` / `PoolAllocator`** (`src/core/allocator.{h,cpp}`, 68 + 100 LOC): zero references in `src/ include/ tools/ tests/`. Dead. They implement bump-arena and fixed-block-pool disciplines, two of the five tiers this design needs, and were never wired to anything.
- **`Buffer`** (`src/core/buffer.{h,cpp}`): exactly **one** producer (`engine.cpp:617`, vision embeddings) and one holder (`Request::vision_emb`). A single-purpose helper, not a general RAII layer.
- **COW-fork / Best-of-N does not exist.** Grep for `cow|copy_on_write|fork_seq|best_of|n_best` across `src/ tools/` returns one hit, a comment in `scheduler.cpp:81` about a *hypothetical* site. The dispatch's "block with three referents" referents are **sequence block table + prefix-cache hash table + pin set** (plus the on-disk persisted cache), not a fork (A5.1).
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
| `memory/recurrent_snapshot_store` | hybrid prefix snapshots | `entries × ssm_slab_bytes` (LRU-bounded, `server.recurrent_snapshot_mb`); evicted entries move to a pinned host tier (`server.recurrent_snapshot_host_mb`, not VRAM) | engine-persistent | ✗ | ✗ |
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
| 9c - the KV pool measures its own residency (AUDIT_arch_2026 B-6) | **done (2026-09-07)**: `KVCache::probe_residency()` right after the pool exists, one timed pass of copies spanning up to 512 MiB of the fresh (all-zero) pool after a 300 ms clock warm-up, `kKvPoolSpillGbps` = 500 as the WARN line, gauge on `/health` and `/metrics`. Falsifier in `test-kv`: 256 MiB pool 1287 GB/s, mapped pinned host memory 130-139 GB/s. Two traps: a cold single pass reads 280 GB/s on resident VRAM (floor clocks), a repeated slice inside the 96 MB L2 reads 4681. Only the KV pool is probed; weights and caches still show a spill only as the throughput cliff | `mem(kv): block size as an operator key, the pool probes its own residency` |
| 9a - `--mem-report` with named charges | **done** | `feat(memory): --mem-report - name the charges the pool notes cannot see` |
| **A7 step 9 complete** (9a + 9b.1-9b.5) | | criterion 5 is *not* claimed - see B38 |
| 10 - lazy pools (`vram.lazy_commit`, 2026-09-10) | **done.** The slot-shaped pools reserve at init and commit on demand, the way the KV pool already grew: `SSMState` reserves every slot (stride padded to the 2 MiB granule, payload unchanged) and commits one slot when the scheduler admits a sequence (`Scheduler::set_admission_gate` -> `Engine::recurrent_slot_admissible_`, which commits the slot the next acquire pops; a refused commit holds the round, the request stays pending); the engine arena opens growable and `take_bytes` commits the prefix a take reaches into, so the Qwen3-VL tower (1107 MiB on Qwen3.8-27B) is uploaded and taken on the first image (`Qwen3VLPipeline::init(lazy)` / `ensure_ready_`). The plan charges every byte as before; what is charged and not committed sits in `vram_reserved_uncommitted_bytes()` and the KV growth cap subtracts it, so the opportunistic grower cannot eat a promised slot. Commits inside a reservation are `planned_serving_commits()`, not I2 violations. A slot commit is refused, never spilled: `ensure_slot` reads free VRAM (which already excludes every pool's pending charge) plus its own pending charge against the allocator headroom (#1103). `kv_cache.growable_initial_pct` 100 -> 25. Idle figures: table below | `perf(memory): commit the slot-shaped pools on demand` |

Three things the first measurement taught, all in the same PR: (1) every free-VRAM reader is a planner, so `vram_budget_mem_get_info()` itself subtracts the ledger (the KV plan had taken the arena's deferred 1946 MiB: 3561 -> 9556 blocks, the scale pool 414 -> 602 MiB); (2) the graph prewarm captures one decode graph per batch row and touches every recurrent slot, so `Engine::trim_recurrent_slots_after_warmup_` decommits the free ones (28/28 committed at init_complete before it); (3) `MemAccount::unattributed_bytes()` reads the raw view (`vram_budget_mem_get_info_ex`, pending left in), or the first-forward library charge counts pending address space as used (7212 measured against 3519), and it reads it in ONE call, or the reading races the prewarm's slot commits (4185 against 3416). The KV-pressure valve grows a pool below its ceiling before it demotes graphs, or a 25 % start armed StreamingLLM twice (`ServingSignalsTest.GraphsComeBackWhenThePressureClearsWithoutEvictions`).

Lazy pools, Qwen3.8-27B-NVFP4-vllm, defaults (`runtime.max_batch_size` auto = 28), arm A `imp:ab-a7c5e49d` (main) vs arm B this tree, one run each, 2026-09-10:

| | main | lazy |
|---|---|---|
| device used, idle after warmup (nvidia-smi, MiB) | 30053 | 25455 |
| device used at init_complete (audit, MiB) | 29663 | 25507 |
| library reserve measured on the first forward (MiB) | 3338 | 3393 |
| ssm_state committed at init_complete (MiB) | 2226 | 0 (2240 reserved) |
| engine arena at init_complete (MiB) | 1946 named | 578 tracked (1946 reserved) |
| kv_cache + scales at init_complete (MiB) | 956 + 414 | 280 + 414 |
| KV blocks committed / ceiling | 3561 / 13258 | 875 / 13264 |
| 28 concurrent chat completions, 64 tokens, wall (s) | 1.37 | 1.43 |
| device used after the burst (MiB) | 30067 | 27691 (28 slots committed) |
| first image request, 64x64 PNG (s) | 0.35 | 1.28 (tower upload + 1107 MiB commit) |
| second image request (s) | 0.21 | 0.20 |
| device used after the images (MiB) | 30131 | 28861 |
| unattributed residual at init_complete (MiB) | 188 | 247 |

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
| D14 | D8 left a plan rejection without `--vram-budget` as a WARN plus the live-pass fallback | the live pass never charged the batch-shaped SSM/GDN state, so the fallback served `max_batch_size` slots the card could not hold: Qwen3.8-27B-NVFP4 at 64 slots (2026-09-08) allocated 5088 MiB of state through the allocator's "exceeds headroom, allowing" hatch, the library reserve claimed at the first forward oversubscribed the device and the pool probe read 528 GB/s (spilled; 32-stream step 15.0 -> 17.4 ms, and a lottery: a later start of the same config read 1453). Now `plan_fitting_batch()` (pure, `memory/plan.cpp`) finds the largest batch the plan accepts, `init_kv_cache` clamps `max_batch_size` and the scheduler's admission cap to it and re-plans (`max_batch_size clamped 64 -> 41`, probe 1621 GB/s). With an operator KV pin only the fixed charges decide the batch. The plan's batch lever counts the state per slot; it read `64 -> 63 frees 0 MiB` before | `test_memory_plan.cpp` `BatchLeverCountsTheBatchShapedSsmState`, `FittingBatchIsTheLargestThePlanAccepts` |
| D15 | A4 had `FeatureSet` as the plan's declared demand | four of its five byte fields were read by `plan_memory()` and written by nobody, and the SSM formula existed twice | `shadow_plan_input` now fills `recurrent_snapshot_bytes` (the snapshot store cudaMallocs `server.recurrent_snapshot_mb`, 256 MiB by default, AFTER the KV pool is sized, so the pool was sized over memory the same init was about to take; charged at `floor(budget / per_seq) * per_seq`, the figure the store claims). `vision_tower_bytes` stays 0 on purpose - the tower is an engine-arena tenant and the arena is open before this probe, so charging it would double-count - and `spec_decode_bytes` / `residual_ring_bytes` stay 0 because neither is sized at plan time (the MTP workspace is allocated by `imp_enable_mtp_spec_decode` after context creation; the residual ring is per-sequence state in `KVCacheManager`, not a pool). `vision tower` and `speculative decode staging` are their own plan lines instead of a silent addition to `engine-persistent`. The SSM byte formula moved to `memory/ssm_state_size.h`, used by both `vram_budget.cpp` and `ssm_state.cu`: they disagreed by conv taps and 256-byte alignment, 4968 MiB charged against 5088 MiB taken at 64 slots. The plan report now closes with `ceiling: recurrent N seqs, KV M seqs at max_seq_len L (B blocks/seq)` in both the accepted and the rejected branch, and WARNs when `M < N` on a model with per-slot state | `test_memory_plan.cpp` `EveryFeatureFieldReachesALine`, `PinsTheQwen38GeometryTheAllocatorTakes`, `KvSeqCeilingIsBlocksOverBlocksPerSeq`, `ReportStatesBothPoolCeilings`, `ARejectedPlanStillStatesTheCeilings` |

### B1 - what the shadow plan found

Measured at server defaults (Qwen3-4B Q8_0, `--max-batch 8`, `runtime.max_seq_len=4096`), both sides fed the same demand figures:

```
  distributable               25551 MiB
  weight-cache demand          2397 MiB  (same figure the live pass used)
  library reserve              3900 MiB  (the live pass does not charge this)
  engine-persistent             397 MiB
  KV: live 4096 blocks -> plan 2048 blocks (4608 -> 2304 MiB)
```

**The live pass gives the KV pool exactly twice what the configuration asks for.** `needed_blocks = ceil(4096/16) x 8 = 2048`; `vram_budget.cpp:379` sets `target_blocks = needed_blocks * 2` for every non-mode-2 strategy. 2304 MiB of the pool cannot be reached by any request the server will accept: 8 concurrent sequences at the advertised context fill 2048 blocks and stop.

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

1. **`calibrate_fp8_scale()` allocated twice per call** (`fp8_quant.cu:193/212`), 144 of the 414, listed by no inventory. One-shot per layer (`executor_kv_write.cu` gates on `kv_calibrated_[kv_layer]`), so an I2 violation by the letter rather than hot-path traffic. Fixed via persistent arena scratch: 414 → 315. The rest of the named sites are `CudaGraphConditionalRunner::setup`, which the step-5 inventory did list and which is genuinely per burst.
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

