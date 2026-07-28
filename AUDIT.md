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

## 2026-07-28 — Phase B, steps 0–2a

### CONFIRMED

| # | Finding | Evidence |
|---|---|---|
| B1 | **The I1 gate works as designed on its first real test.** Step 1 deleted the dead `src/core/allocator.{h,cpp}`; the gate immediately failed on the now-stale allowlist entry and refused to pass until it was removed. That is the mechanism that makes "the allowlist shrinks monotonically" structural rather than aspirational. Baseline 79 files / 717 sites → **78 / 713**. | `tools/check_alloc_sites.py` |
| B2 | **The new code has no engine call sites yet.** Nothing outside `src/memory/` (and the three new test files) includes `backend.h`, `arena.h`, `block_pool.h`, `scratch_stack.h`, `span.h`, `plan.h` or `fake_backend.h`. Steps 0–2a are provably inert; the measured −0.43% prefill / −1.01% decode is noise, not a regression. | grep; `docs/audit/PERF_LOG.md` |
| B3 | **The whole allocator stack + planner runs GPU-free.** 608 CPU tests green in `test-core`, including block-pool conservation under 5000-step randomised churn, refcount balance across every exception point in a sequence's lifecycle, and plan determinism over 1000 configs. imp has no GPU runner, so this was a hard requirement, not a nicety. | `docker run --rm imp:test test-core` |

| B4 | **The live budget pass gives the KV pool exactly 2x what the configuration asks for.** `vram_budget.cpp:457` sets `target_blocks = needed_blocks * 2` for every non-mode-2 strategy. Dense server default: `needed = ceil(4096/16) x 8 = 2048`, pool = **4096 blocks / 4608 MiB**, of which 2304 MiB is unreachable by any request the server accepts. Harmless where VRAM is slack; not harmless where the surplus is drawn from the headroom the pre-dequant caches compete for — which is exactly #1100 and #1103. | shadow plan at init, `docs/MEMORY_ARCHITECTURE.md` B1 |

| B5 | **`test-e2e` leaks ~15 GiB of device memory across engine teardowns — PRE-EXISTING, measured both sides.** `MtpForwardTest.DraftStepProducesValidToken` passes in isolation (`24.76 GiB free` → all 15 GiB of NVFP4 experts upload) and fails in the full binary (`9.12 GiB free` → "experts exceed on-device budget … will produce garbage" → the MTP head upload fails → `mtp_->loaded == false`). It runs directly after six `ChunkedPrefillTest`s whose engines are destroyed but whose memory is not returned. **A baseline build from the pre-step-3 commit fails identically, with a byte-identical `9.12 GiB free`** — so the KVCache→BlockPool migration neither caused it nor changes device-memory retention. Prime suspect: imp pins the default async mempool's release threshold to `UINT64_MAX`, so `cudaFreeAsync`'d weights are retained rather than returned, and `trim_device_mempool()` is not reached on every teardown path. **This is acceptance criterion 4's exact failure mode** (post-drain live bytes must return to the post-load baseline) and it belongs to A7 step 6, not step 3. | full `test-e2e` vs isolated; stash-and-rebuild A/B |

| B6 | **The KV pin set holds no refcount — A5.1's "three referents" was wrong.** `kv_cache_manager.cpp` has exactly **one** `cache_->inc_ref` call site (line 536, prefix reuse); no pin path touches the KV refcount. `pinned_blocks_`/`pin_refcount_` are *pin-owner* counts. A pinned block stays alive because it sits in the cached LRU at refcount 1, and pinning only makes `reclaim_cached_block()` rotate past it. There are **two** refcount holders — the sequence and the cached LRU — and the second holds nothing: it survives because `free_sequence()` deliberately skips the free. Liveness by omission, not by ownership. Design doc corrected; the pin set migrates as policy, not ownership. | grep of every `inc_ref`/`free_block`/`ref_count` site |

### GOTCHAS (Phase B)

| # | Trap |
|---|---|
| G5 | **`make build \| tail` swallows the exit code** — a known imp trap, and it bit again here: a failed build reported `BUILD_EXIT=0` because the pipeline's last stage was `tail`. Capture to a file and echo `$?` on its own. |
| G6 | **Two of my own planner tests encoded wrong premises and failed on first run.** `ChargesTheLibraryReserve...` compared KV block counts at the full 32 GiB budget, where KV stops at what it *needs* and the residual never binds — both arms were identical, so the test would have passed while proving nothing had the arithmetic differed slightly. `KvShrinksToTheResidual...` asserted a 12 GiB budget still plans, when the fixed charges alone consume 12.3 GiB. Both were fixed by correcting the test, not the implementation. Worth recording because a green test here would have been worse than a red one. |
| G9 | **`PrefixCacheE2ETest.*` and `DetEvalE2ETest.*` silently SKIP without `IMP_TEST_MODEL`** — they print `[ RUN ]` and then nothing, and the summary counts only the tests that did run, so a filtered run reports `[  PASSED  ] 5 tests` and looks green. I nearly shipped the prefix-cache ownership change believing I had e2e coverage of exactly the path I had changed. Run them as `-e IMP_TEST_MODEL=/models/<model>` and check the RUN/OK counts match. |
| G10 | **A −2.8% decode reading with < 0.04% variance across trials is still not a regression.** It looked stable enough to be real. A back-to-back A/B (stash, rebuild, bench) put the unmodified parent at 277.84 and the change at 277.60 — the whole delta was host state (the user was streaming, invisible in nvidia-smi; the same trap as the "evening drift" in #526/#1018). **Never compare against a number taken 40 minutes ago; rebuild the parent and measure both arms in the same window.** |
| G8 | **A GPU test that passes in isolation and fails in the suite is a VRAM-retention story, not a logic story** — and the free-VRAM figure in the log is the tell. `MtpForwardTest` looked like a regression from the KV-cache migration; it is B5, and the giveaway was that the two runs report the *same* `9.12 GiB free`. Always compare against a rebuilt baseline before attributing a suite-only failure to your change. |
| G7 | **A `docker build` can fail on a transient DNS lookup for `docker/dockerfile:1`** (the `# syntax=` directive resolves over the network). Retry before debugging anything. |

---

## Housekeeping note (not a memory finding)

The working tree carries a **staged but uncommitted `#1103` fix** —
`src/memory/vram_query.h`, `src/runtime/vram_budget.cpp`,
`src/runtime/engine.cpp`, `tests/test_vram_budget_reserve.cpp` — on the
unrelated branch `fix/1104-json-number-grammar`. It floors the mode-2 reserve at
the `VRAMAllocator`'s 5 % headroom, which the planner previously undercut by
1118 MiB on a 32 GB card. All A1 numbers were measured **with** it applied. It
needs its own branch off `main` before it is lost or lands in the wrong PR.
