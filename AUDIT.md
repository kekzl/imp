# Memory Subsystem Audit

Running findings log for the memory-architecture work. Design lives in
[`docs/MEMORY_ARCHITECTURE.md`](docs/MEMORY_ARCHITECTURE.md). **Negative results
are recorded here too** — a suspected problem that turns out to be clean is worth
exactly as much as one that is real, and stops the next pass re-chasing it.

Convention: **CONFIRMED** = verified against the code or measured on the device.
**REFUTED** = tested and found not to be true. **OPEN** = suspected, not yet
verified.

---

## 2026-07-28 — Phase A (inventory + design)

Tree: `fix/1104-json-number-grammar` = `main` + two `#1104` constrain commits
(no memory impact) + the **staged, uncommitted `#1103`** budget fix. Card idle
before every run (0 containers, no compute processes); healthy under load
(2857–2932 MHz SM / 13801 MHz mem / 310–444 W).

### CONFIRMED — structural

| # | Finding | Evidence |
|---|---|---|
| C1 | **380 source lines with an allocation call (395 calls) in 81 files.** 336 calls are device-side; only **31 (9 %)** route through `VRAMAllocator`. 365 lines in 74 files sit outside `src/memory/` — that is the initial I1 allowlist. | grep census, comment lines excluded |
| C2 | **`VRAMAllocator` is a tracker, not an owner** — stated in its own destructor comment. Its headroom is defeatable three ways, all live: `<16 MiB` always allowed; `bypass_headroom=true`; and an override that proceeds anyway when `free >= bytes + 64 MiB`. | `src/memory/vram_allocator.{h,cu}` |
| C3 | **Capacity is discovered, not planned.** `compute_vram_budget()`'s dominant input is a live `cudaMemGetInfo`; it runs after the weight upload and before the cache build; the pre-dequant phases then re-derive their own reserves from live free VRAM again. | `src/runtime/vram_budget.cpp`; `split_pre_dequant_budget` docstring (#1100) |
| C4 | **The engine holds a balloon** — a physical `cudaMalloc` kept across `init_weights` purely to hide bytes from the KV planner, released just before phase 3. | `engine_weight_upload.cpp:325`, `engine_kv_cache_init.cpp:434` |
| C5 | **Six incident-driven clamps stack** on the KV block count: `target_blocks`, post-weight `max_fit_blocks`, `min_kv_blocks`, `kv_fraction` affordability cap, SWA batch-shaped charge, `#1103` allocator-headroom floor. Each corrects the previous one. | `vram_budget.cpp:455–539` |
| C6 | **`spec_graphs_` has no explicit cap, but its key space is bounded** — a `std::map` keyed by `(n_tokens, ctx_capacity, rec_slot)`, cleared only wholesale on invalidation. All three axes are bucketed: `n_tokens` to 3–5 draft buckets, `ctx_capacity` to power-of-two tiers from 4096 up to `speculative.capture_ctx_cap` (~6 tiers), `rec_slot` to `max_batch + 1`. Worst case is therefore ~5 × 6 × (max_batch+1) graph execs, each with its own graph memory — ~1950 at `--max-batch 64`, not infinite. **Bounded but uncounted:** nothing in the plan charges it. | `engine.h:738`, `engine_spec_capture.cpp:65,145` |
| C7 | **CUTLASS workspace grows lazily at GEMM time** (`cudaFree` + `cudaMalloc` inside the impl), on a path reachable under graph capture, whenever the executor's pre-sized buffer is too small or failed to allocate. | `gemm_cutlass_sm120.cu:808–816` |
| C8 | **Per-request `cudaMallocAsync` inside graph-captured regions**: block tables, SWA block tables, banned-token list. A comment at `executor_attention_prefill.cu:14` already concedes the violation ("cudaMallocAsync per layer here violates CLAUDE.md 'No cudaMalloc in hot…'"). | `engine_graph_decode.cpp:142,150,178`; `engine_scheduler.cpp:597–605` |
| C9 | **`--vram-budget` is a sizing view, not a cap.** It rewrites what `cudaMemGetInfo` returns; its own header says "best-effort hard cap, not an OS limit … leave ~1 GiB of real headroom." | `src/memory/vram_query.{h,cpp}` |
| C10 | **`--mem-report` does not exist.** No flag, no handler. Acceptance criterion 6 is a new deliverable, not a repair. | grep `mem-report\|mem_report` → 0 hits |

### CONFIRMED — measured

| # | Finding | Numbers |
|---|---|---|
| M1 | **Steady-state footprint, three configs** (32 607 MiB card, `00_pre_init` = 1679.6 MiB on all): dense Qwen3-4B Q8_0 **18 226**, MoE Qwen3-Coder-30B-A3B NVFP4 **23 872**, vision gemma-3-4b + mmproj **14 792**. | `MemAccount` init tables |
| M2 | **Peak under load − steady state = +190 / +178 / +200 MiB.** There is no transient prefill spike: every workspace is statically pre-allocated. This delta *is* the entire I2 violation surface. | 2 rounds × 8/8/4 concurrent, 0 errors |
| M3 | **20–39 % of device memory is unattributed.** Tracked/total: dense 11 238 / 18 416 (61 %), MoE 19 311 / 24 050 (80 %), vision 10 473 / 14 992 (70 %). Criterion 6 wants ≥95 %. | per-pool `note()` vs device used |
| M4 | **~3.9 GiB is claimed on the first forward pass**, after the plan is final, attributed to nothing. `runtime.warmup=false` → init ends at 14 046 MiB; **one** 32-token request → 18 234 MiB (**+4188**). | dense config |
| M5 | **M4 is invariant to batch and context**: batch 1/8/16 at ctx 4096 → 3873 / 3855 / 3886 MiB; batch 8 at ctx 1024 → 3848 MiB. It scales with nothing the planner knows about. | 4-point sweep |
| M6 | **The planner over-reports available capacity by exactly M4.** The dense run logs `available=22290.3 MiB` at budget time and hands KV 4608 MiB — from a number ~3.9 GiB too optimistic. Survives today only because dense leaves 14 GiB free; on the MoE config the same constant is 46 % of the remaining headroom. | `vram_budget.cpp:571` log vs M1 |
| M7 | **The vision tower costs +1610 MiB at `04_features`**, resident from init whether or not an image ever arrives. | vision config |

### REFUTED — tested, not true

| # | Hypothesis | How it was killed |
|---|---|---|
| R1 | The ~3.9 GiB first-forward claim is **imp's lazy CUDA module loading**. | `CUDA_MODULE_LOADING=EAGER` moved only **+124 MiB** into `00_pre_init` and +270 MiB into `01_prewarm_gemm`; the first-request delta stayed at **+4188 MiB**. |
| R2 | It is the **default `cudaMallocAsync` pool** (release threshold pinned to `UINT64_MAX`, so frees are retained). | Pool `reserved`/`used` = 4096/4076 MiB before *and* after the request. Unchanged. |
| R3 | It **scales with batch or context** (i.e. it is a workspace the planner could size). | Flat across batch 1→16 and ctx 1024→4096 (M5). |
| R4 | **Persistent GEMM autotuning state** explains the known 2.6× prefill variance across container restarts. | `src/compute/gemm.cu` contains **no file I/O of any kind**. The cuBLASLt algo cache is a process-local map rebuilt every start. There is nothing to persist. Not a memory-design input; question closed. |
| R5 | imp has a **COW-fork / Best-of-N** path, making a KV block's ownership a three-way fork problem. | Grep for `cow\|copy_on_write\|fork_seq\|best_of\|n_best` across `src/ tools/` returns **one** hit — a comment at `scheduler.cpp:117` noting a hypothetical site *would* need COW. The three referents are real but are **sequence table + prefix-cache hash + pin set** (plus the on-disk persisted cache). |
| R6 | The CUDA **graph memory pool** explains the first-forward claim. | Attempted A/B was **invalid** — `--set runtime.cuda_graphs=off` and `=never` both still captured 2 graphs, so the variable was never controlled (see G1). Untested, not refuted; the design counts graph memory separately regardless (`cudaDeviceGetGraphMemAttribute`, A5.2). |
| R7 | A `VramOwned` RAII type exists and just needs extending. | It does not exist. Prior audits hallucinated it (already flagged in the `codebase-audit` skill priors); re-verified. |

### CONFIRMED — dead / near-dead code

| # | Finding |
|---|---|
| D1 | **`ArenaAllocator` and `PoolAllocator` (`src/core/allocator.{h,cpp}`, 168 LOC) are entirely dead** — zero references in `src/ include/ tools/ tests/`. They implement bump-arena and fixed-block-pool disciplines, i.e. two of the five tiers this design needs, written once and never wired up. Migration step 1 revives them rather than writing new ones. |
| D2 | **`Buffer` (`src/core/buffer.{h,cpp}`) has exactly one producer** (`engine.cpp:699`, vision embeddings) and one holder (`Request::vision_emb`). It is a single-purpose helper, not a general RAII layer. Not dead, but not a foundation either. |

### GOTCHAS (cost real time; do not repeat)

| # | Trap |
|---|---|
| G1 | **`runtime.cuda_graphs` takes `"auto" \| "always" \| "never"` — and silently ignores anything else.** `--set runtime.cuda_graphs=off` parsed fine, changed nothing, and produced a byte-identical A/B that looked like a clean refutation. `=never` *also* left 2 graph captures in the log. Always verify the arm actually moved (`grep -c 'capturing CUDA graph'`) before believing an A/B. Same class as the `kv_cache.prefix_cache` non-key trap from 2026-07-27. |
| G2 | **`tools/analysis/vram_audit_load.py` sends `prompt = 0.6 × ctx` and asks for `0.4 × ctx` more**, so `prompt + max_tokens == ctx` exactly and the server returns **400 on every request** when the driver's `--ctx` equals the server's. The warmup round (256/128) succeeds, so the run *looks* fine and the summary reads `requests: 0 errors: 16` at the very bottom. Drive it with `--ctx = 0.75 × server ctx`. |
| G3 | **`nvidia-smi` reads ~180 MiB below the in-process `cudaMemGetInfo` used** on this box. Fine for deltas, wrong for absolutes — quote `MemAccount`, not smi. |
| G4 | **There is no `src/graph/`.** CUDA-graph code lives in `src/runtime/cuda_graph.{cu,h}` + `engine_graph_decode.cpp` + `engine_spec_capture.cpp`. |

### OPEN

| # | Question |
|---|---|
| O1 | What exactly claims the ~3.9 GiB (M4)? R1–R3 exclude imp's modules, the async pool, and anything that scales with the plan. Remaining candidate: CUDA/cuBLAS/CUTLASS library-internal reservation on first matmul dispatch. Needs either a `cudaDeviceGetGraphMemAttribute` + per-library probe, or acceptance as a measured constant with a boot-time self-check (the design takes the latter route). |
| O2 | Does `cuMemCreate`/`cuMemMap` behave under WSL2/WDDM? Gates migration step 7 only; a ~200-line spike answers it. |

---

## Housekeeping note (not a memory finding)

The working tree carries a **staged but uncommitted `#1103` fix** —
`src/memory/vram_query.h`, `src/runtime/vram_budget.cpp`,
`src/runtime/engine.cpp`, `tests/test_vram_budget_reserve.cpp` — on the
unrelated branch `fix/1104-json-number-grammar`. It floors the mode-2 reserve at
the `VRAMAllocator`'s 5 % headroom, which the planner previously undercut by
1118 MiB on a 32 GB card. All A1 numbers were measured **with** it applied. It
needs its own branch off `main` before it is lost or lands in the wrong PR.
