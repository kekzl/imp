# Phase 5 — Synthesis Master Report

Anchor: `f58eb9e`. Target: sm_120a / RTX 5090 / GB202 only. Synthesizes
`phase1_inventory.md` (751 LOC), `phase2_perf.md` (1 159 LOC),
`phase3_maint.md` (1 283 LOC), `phase4_ext.md` (1 044 LOC).

---

## 0. TL;DR

imp is an architecturally clean 90 KLOC sm_120a-only inference engine
whose kernels sit at 90–100 % of the bandwidth-limited decode ceiling
on every supported model class, but whose **prefill is 14.7× slower
than vLLM on the identical CUTLASS NVFP4 grouped-GEMM template**, and
whose **`src/graph/` is a 15.8 KLOC clay tablet** where every
quantization format, every model architecture and every dispatch
decision is encoded as inline `if`-ladders. The codebase is small
enough to refactor, mature enough to merit it, and broken in exactly
two places — perf (around the kernel, not in it) and extensibility
(per-arch behavior is `if (cfg.arch == ModelArch::GEMMA4)` sprinkled
in 40 sites).

- **Biggest brecher-opportunity:** the `WeightCaches` god-struct +
  21-param `gemm_dispatch_impl` + dual dispatch tables — collapsing
  these into a `GemmKernel` registry wins on perf, maintainability
  and extensibility simultaneously and removes ~1 000 LOC.
- **Biggest streichkandidat:** `src/compute/mmq_q4k_v2.cu` (1 667 LOC,
  −4 % E2E on Qwen3.6-35B Q4_K_M, opt-in only, 5 of 7 phase templates
  dead at runtime).
- Evidence anchored in `review/phase1_inventory.md`,
  `review/phase2_perf.md`, `review/phase3_maint.md`,
  `review/phase4_ext.md`.

---

## 1. Executive Summary

**imp today.** A single-target sm_120a engine with healthy kernel
choices (HMMA, mxf4nvf4 block-scaled FP4, FP8 m16n8k32, cluster
launches with spread scheduling, PDL-augmented decode graphs), a thin
24-function public C ABI, ~574 GTest tests, and a maintained MEMORY
ledger of 60+ shipped / archived memos. Decode throughput on every
supported model is within ~10 % of the bandwidth ceiling
[P2 §1, §5.4]. The architecture maps the right way at the directory
level (`core/` is a strict leaf, `compute → core` is the dominant
edge), but the next layer up (`graph/`) is a god-layer that bends
every other subsystem around itself.

### Top-3 Wins (what is already brecher)

- **Decode is at-the-roof bandwidth-bound.** `gemv_nvfp4_moe_decode_kernel`
  with `__launch_bounds__(128, 12)` hits ~261 tok/s vs ~270 tok/s
  ceiling on Qwen3-Coder NVFP4 [P2 §1, P2 §2.3 — `src/quant/nvfp4_gemm.cu:855`].
- **Public C ABI is clean.** 24 functions, 4 headers, zero internal-
  header leakage in `include/imp/` [P1 §2 — `include/imp/imp.h:1-142`].
  The single public→internal bridge is `src/api/imp_internal.h:1`.
- **Hot-loop allocator hygiene is honored.** No `cudaMalloc/cudaFree`
  in true per-token loops; surviving lazy allocs are all monotonic-
  grow-only with pre-warm hooks [P2 §2.6, P2 §5.6].

### Top-3 Brüche (what must change)

- **NVFP4 MoE prefill is 14.7× slower than vLLM on the same CUTLASS
  template.** Same `MainloopSm120ArrayTmaWarpSpecializedBlockScaled`,
  same `<128, 128, 128>` tile, same `Sm120` arch tag. The gap is
  100 % around-the-kernel: activation-quant fusion missing, per-layer
  launch overhead, scheduler maturity [P2 §6 — `src/compute/gemm_cutlass_grouped_3x.cu:30-86`,
  `src/graph/executor_forward_moe.cu:559-563`].
- **`graph/` is one 15.8 KLOC god-layer** with a 21-param
  `gemm_dispatch_impl`, a 6-map `WeightCaches` god-struct, and a
  duplicate per-qtype dispatch table in `compute/weight_dispatch.cu`
  [P3 §1.2, P3 §2 #1, P3 §5.2 #4 — `src/graph/executor_kernels.cu:2003-2269`,
  `src/graph/executor.h:286`, `src/compute/weight_dispatch.cu:73-125`].
- **Per-arch behavior is `if (cfg.arch == ModelArch::GEMMA4)` × 40.**
  14 in `executor_attention.cu`, 19 in `executor_forward_moe.cu`,
  2 in `executor_forward.cu`, 5+ in `engine.cpp`. Adding a new arch
  is a 25-file diff [P3 §1.5, P4 §0, P4 §2.4 —
  `src/graph/executor_attention.cu:161,310,387,464,472,493,534,596,658,678,821,893,1198,1274`].

### What changes in 90 days if everything below ships

After 90 days of focused execution, `src/graph/` shrinks from 15.8 KLOC
to ~10 KLOC; `executor_kernels.cu` shrinks from 2 327 to ~500 LOC;
adding a new model arch becomes a single ~80-LOC file in
`src/model/plugins/`; the NVFP4 MoE prefill gap closes from 14.7× to
~3× (CUTLASS scheduler maturity ceiling); ~5 KLOC of validated dead
code disappears from the runtime tree; the test suite gains
parameterized per-arch coverage and at least one graph-capture
sanity gate. The engine remains correct on every currently-supported
model and gains a clean seam for the next ten.

---

## 2. Performance Roadmap (TTFT-first, decode second)

Decode is at-the-roof; perf wins live almost entirely in **prefill**
and **graph-capture coverage**. Decode-side wins are squeezed.

### 2.1 Quick Wins (< 1 week each)

| # | Action | File anchor | Expected impact | Effort | Risk | Cross-axis bonus |
|---:|---|---|---|---|---|---|
| 1 | Pre-warm `mmvq_kernel` scratch (kill lazy `cudaMalloc/Free`) | `src/graph/executor_kernels.cu:2175-2181` [P2 #8] | First-call latency drop; capture-safe everywhere | 1 hour | none | maint (one less lazy alloc) |
| 2 | Delete stale wgmma docstring in FMHA header | `src/compute/attention_fmha_sm120.h:8-10` [P2 #7, P3 §10 #11] | 0 perf; un-misleads readers | 5 min | none | maint |
| 3 | Add L2-streaming hint to LM-head GEMV | `src/graph/executor_forward.cu` (LM head call site) [P2 #10] | ≤1 % decode at long ctx | 1 hour | none | none |
| 4 | Cache `IMP_NVFP4_FORCE_DEQUANT` and `IMP_LOG_GEMM_ALGO` env reads in `static const` | `src/compute/weight_dispatch.cu:106`, `src/compute/gemm.cu:326` [P3 §9.3] | Microscopic; removes per-call `getenv` | 30 min | none | maint |
| 5 | Tighten per-layer SFA zero-memset in MoE prefill | `src/graph/executor_forward_moe.cu:557-558` [P2 #9] | ~2.3 ms / session | hours | none | none |
| 6 | Gate BitDecoding TC residual-arg marshalling behind a single null-check | `src/graph/executor_attention.cu:1027-1083` [P2 #6] | Few µs/layer × n_decode_steps | hours | low | maint (kill 8 nullable trailing args from hot dispatch) |
| 7 | Remove dead `mxfp4_act_sf` branch inside NVFP4 path | `src/graph/executor_kernels.cu:2083-2094` [P3 §5.1, P3 §10 #13] | 0 perf; -12 LOC, -1 nested arm | 30 min | none | maint |
| 8 | Hard-fail (not log-INFO) when NVFP4 `da_cache` populates < 100 %  | `src/graph/executor_forward_moe.cu:566-578` [P4 #10] | Prevents silent 5× decode regression on VRAM-tight boxes | hours | low | ext (new NVFP4 MoE arches survive prod) |

### 2.2 Medium (1-4 weeks each)

| # | Action | File anchor | Expected impact | Effort | Risk | Cross-axis bonus |
|---:|---|---|---|---|---|---|
| M1 | **Fuse SwiGLU + activation-quant for MoE down-phase** (Bucket B of NVFP4 prefill gap) | `src/graph/executor_forward_moe.cu:559-563, 646-657` [P2 #1, P2 §6.9] | +5–10 % pp512 on Qwen3-Coder NVFP4; first slice of 14.7× gap | 2-3 days kernel + 2 days plumbing | low (numerics identical, prefill-only) | ext (clears the largest hot-path code surface) |
| M2 | **Remove per-token D2H sync in `try_run_moe_gemma4_ggml_prefill`** | `src/graph/executor_forward_moe.cu:2121-2123` [P2 #3] | Restores prefill graph capture for Gemma-4 ggml fallback path; large for that path | 1-2 days | low (function only runs at n>1) | maint (kill one of the 5 D2H sync sites) |
| M3 | **Complete MoE-prefill graph capture Phase 4** (per `moe_prefill_graphs_plan_2026_05_10`) | `src/graph/executor_forward_moe.cu:1131-1135, 1189-1192, 1996-2000` and `src/runtime/cuda_graph.cu:228-240` [P2 §4.5, P2 §6.8] | +10–15 % pp on NVFP4 MoE | 2-3 weeks | medium (multi-graph pool sizing) | maint (5 D2H sites collapse into one capture) |
| M4 | **Bucket-allocate ctx_len in pow-2 buckets + enlarge `kMaxGraphPoolSize`** | `src/runtime/cuda_graph.cu:228-240`, `src/runtime/engine.cpp:2637, 2643` [P2 #4] | -10–30 ms per shape change at decode | days | low | maint |
| M5 | **Cluster-launch on FMHA prefill** (DSMEM K-broadcast across cluster CTAs) | `src/compute/attention_fmha_sm120.cu` (no cluster today) [P2 §3.7, P2 §8 #1] | +10–20 % pp4096+ on FP16 prefill non-MoE | 2-3 weeks | medium (tile re-layout) | none |

### 2.3 Big Bets (> 4 weeks)

| # | Action | File anchor | Expected impact | Effort | Risk | Cross-axis bonus |
|---:|---|---|---|---|---|---|
| B1 | **Close the rest of the Qwen3-Coder NVFP4 prefill 14.7× gap** (Buckets A+B+C from P2 §6.8) | `src/compute/gemm_cutlass_grouped_3x.cu:150` + `src/graph/executor_forward_moe.cu:508-665` [P2 #2, P2 §6] | Realistic ceiling: 14.7× → ~3× (CUTLASS scheduler maturity is the residual) | 6-12 weeks staged | medium (CUTLASS upstream dependency on residual) | ext (everything in this work-stream lands in `executor_forward_moe.cu` which Phase 3 §11 #3 wants split anyway — wins maint as side effect) |
| B2 | **Device-side LRU expert prefetch with cudaMemcpyAsync overlap** (re-enables CUDA Graphs on host-offloaded experts) | `src/runtime/engine.cpp:1158-1164` [P2 S7, P4 #11, `docs/roadmap.md:53`] | ~5× decode lift on >32GB MoE arches that today fall back to host experts | 4-6 weeks | medium (pinned-memory pool sizing) | ext (any 50B+ MoE arch inherits this win) |
| B3 | **PV-FP4 in FMHA (Phase 3 mxfp4 block-scaled P×V)** with two-level accumulator (SageAttention3-style) | `src/compute/attention_fmha_sm120.cu:563`, `src/compute/attention_fmha_mxfp4_sm120.cu` [P2 §3.3, §8 #2] | +10–13 % attention-bound prefill | 4-6 weeks (quality A/B mandatory) | high (softmax-output FP4 quant has quality risk) | none |

**"If we ship nothing else this quarter, ship this one":**

- **B1 wins** the quarter. The 14.7× vLLM gap is the headline number
  that defines whether imp is competitive. B2 is bigger per-feature
  but only matters for one model class (>32GB MoE). B3 is risky and
  upper-bound +13 %. **B1 (specifically: ship M1 first, then prefill
  graph completion M3, then iterate to bucket A scheduler tuning) is
  the only big bet that simultaneously moves the dominant TTFT
  metric AND lands inside `executor_forward_moe.cu` which Phase 3
  wants split anyway** — it pays the refactor cost as a side effect.

- **Counter-justification for B2:** if business priority is
  35B+ MoE serving (Qwen3.5/3.6, Llama-4, future 70B-MoE), B2
  should jump the queue — a 5× decode lift dwarfs prefill in user-
  visible TTFB on those models. **This is a maintainer call.**

- **Counter-justification for B3:** PV-FP4 ships only if the quality
  A/B harness is built first (which itself needs ~1 week). With
  quality risk and only +13 % upside, this is not the quarter's win.

---

## 3. Maintainability Roadmap

### 3.1 Streichliste (deletable today)

Consolidated from P1 §7 + P3 §10 + P4 #15. Risk: lo = mechanical;
med = test-pinned; hi = policy call.

| # | Item | LOC | Risk | Source |
|---:|---|---:|---|---|
| 1 | `mmq_q4k_v2.cu` phase-template tails (5 of 7 instantiations) | 700 | lo | P3 §3.3, §10 #1 |
| 2 | `mmq_q4k_v2.cu` entire TU (alt to #1; opt-in, −4 % E2E) | 1 667 | hi | P3 §10 #2 |
| 3 | `gemm_grouped_nvfp4_smallM.cu` relocate to `tests/bench/` | 948 (move) | lo | P1 §7.5, P3 §10 #3 |
| 4 | `attention_tc.cu` (subsumed by Blackwell variant) | 411 | med | P1 §7.4, P3 §10 #5 |
| 5 | `gemm_moe_fused_tc.cu` (WMMA-based; needs profile confirm) | ~520 | med | P1 §7.5, P3 §10 #6 |
| 6 | `gemm_capture_fp16_sm120.cu` (~600 LOC, dispatch unconfirmed) | ~600 | med | P1 §7.5, P3 §10 #7 |
| 7 | `__CUDA_ARCH__ >= 1200` `#else` branches across 17 TUs | ~600 | lo | P1 §7.2, P3 §10 #8 |
| 8 | Three runtime sm_120 availability flags | ~9 | lo | P1 §7.3, P3 §10 #9 |
| 9 | `sm_80/90/100` comments + 1 dead auto-select branch | ~13 | lo | P1 §7.1, P3 §10 #10 |
| 10 | Stale wgmma docstring (`attention_fmha_sm120.h:8-10`) | ~5 | lo | P3 §10 #11 |
| 11 | Bench/probe TUs (relocate `src/compute/*_bench.cu` → `tests/bench/`) | 1 772 (move) | lo | P1 §7.8, P3 §10 #12 |
| 12 | Dead `mxfp4_act_sf` branch | 12 | lo | P3 §5.1, §10 #13 |
| 13 | Two dispatch tables merged into one | ~150 | med | P3 §1.2 #2, §10 #14 |
| 14 | Dead `RuntimeConfig` fields (~8-10 of ~50, esp. Gemma-4 stabilization toggles) | ~50 | lo | P3 §9.5, §10 #15 |
| 15 | `compute/preamble_gate.h` back-edge into `graph/quant_scratch.h` | 3 | lo | P3 §1.3, §10 #16 |
| 16 | Move `compute/warp_reduce.cuh` + `compute/ptx92_utils.cuh` to `core/` | 0 net | lo | P3 §1.3, §10 #17 |
| 17 | Drop `<cuda_runtime.h>` from `model/model.h` (forward-decl `cudaStream_t`) | -1 +5 | lo | P3 §4.3, §10 #18 |
| 18 | `throw`-based error handling in `core/`, `memory/` (~12 sites + 40 callers) | 0 net (rewrite) | med | P3 §6.1, §10 #19 |
| 19 | Convert hot-path `assert()` (55 sites) to logged-runtime check | 0 net | lo | P3 §6.4, §10 #20 |

| Bucket | LOC |
|---|---:|
| Hard delete (lo risk) | **~1 942** |
| Soft delete (med/hi — needs profiling or policy call) | **~3 198** |
| Relocate (no LOC change in the binary) | **~2 720** moved |
| **Grand total kahlschlag (post-de-dup, hard+soft+relocate)** | **~5 460 LOC = 6.1 % of `src/`** |

This matches Phase 1 §7.9's independent estimate of ~5 390 LOC.

### 3.2 Refactoring sequence

Re-using Phase 3 §11's #1–#5 sequence, validated against Phase 4 §6
roadmap for conflicts. **The numbering below is the master execution
order**, not Phase 3's internal numbering.

| Order | Refactor | Depends-on | Files touched | LOC delta | What becomes easier after |
|---:|---|---|---|---:|---|
| **R0 (FIRST DOMINO)** | Mechanical env-var + error-handling sweep [P3 §11 #5] | none | `runtime/config.{h,cpp}` + 16 grep-targeted TUs + ~10 in `core/`/`memory/` | -20 net | Hot-path env reads disappear; uniform `if (RuntimeConfig::current().X.Y)` pattern enables grep-able deprecation; `throw → return ImpError` lets every later refactor propagate errors consistently |
| R1 | Test parameterization + `docs/integration.md` [P4 §6 Step 1] | R0 | `tests/test_e2e_models.cpp` (rewrite -300+200), new `docs/integration.md` (~200 LOC) | -100 net | Every later arch / kernel refactor gets free e2e coverage; new contributor day-7 cost drops from 2-3 weeks to ~3 days |
| R2 | `ArchPlugin` interface + Qwen3 first migration [P4 §6 Step 3] | R1 | `model/arch_plugin.h` (~120), `model/arch_registry.cpp` (~60), `model/plugins/qwen3.cpp` (~80), -18 across `model.cpp` + `chat_template.cpp` | +250 net | Validates the plugin design before R3 commits to it; future arches land in 1 file |
| R3 | Migrate remaining 13 arches to plugins [P4 §6 Step 4] | R2 | 13 plugin files (~80 LOC × 13), -80 in `model.cpp` + `chat_template.cpp` | +960 net | Per-arch behavior is centralized in plugin file; `kArchRegistry` deletes |
| R4 | `ModelArchAdapter` / kill GEMMA4 hot-path branches [P3 §11 #1, P4 §6 Step 5] | R3 | `executor_attention.cu` -80 (14 branches), `executor_forward_moe.cu` -60 (19), `engine.cpp` -20, `executor_workspace.cu` -2, `executor_forward.cu` -5; new `arch_adapter.h` + `arch_adapter_gemma4.cu` (~400 LOC moved) | -150 net | Adding a new attention variant is 1 file; **§5 cross-axis champion lands here** |
| R5 | `GemmKernel` registry — collapse 21-param dispatch + `WeightCaches` god-struct + dual dispatch tables [P3 §11 #2, P4 §6 Step 6] | R0 (parallel with R4) | `executor_kernels.cu` -266, `executor.h` -150, `weight_dispatch.cu` -100, 8 new `compute/gemm_kernel_*.cu` (~150 LOC each), `executor_pre_dequant.cu` -500 | **-1 000 net** | New qtype = 1 file; biggest single refactor win |
| R6 | Split `executor_forward_moe.cu` into 5 TUs [P3 §11 #3] | R4 + R5 | `executor_forward_moe.cu` (delete; split into decode_fast, prefill, shared_expert, routing, gemma4_overrides) | 0 net | Phase 2 leak #1 (gate+up+SwiGLU+quant fusion) becomes a single-file change; 60 % cognitive load drop per file |
| R7 | Split `engine.cpp` into 5 TUs [P3 §11 #4] | R6 | `engine.cpp` (split into init/step_prefill/step_decode/mtp/residual) | 0 net | Per-file cognitive load drop |
| R8 | `RecurrentState` polymorphism [P4 §6 Step 7] | R5 | `memory/recurrent_state.h` (new ~40), `memory/ssm_state.{h,cu}` + `memory/gdn_state.{h,cu}` refactored, ~10 sites in `engine.{h,cpp}`, ~5 in `graph/executor.h` + `executor_ssm_gdn.cu` | +0 net | Hybrid arch #3 (RWKV/RetNet) lands cleanly |

**The first domino is R0.** Reasoning: every other refactor changes
hot-path code that reads env vars and propagates errors through
`throw`. Without R0, R1-R8 either keep the inconsistency or fight it
mid-refactor. R0 is also the smallest LOC delta and lowest risk —
ideal first move.

---

## 4. Extensibility Roadmap

### 4.1 Target state: "<500 LOC = working model"

Per Phase 4 §5: each arch becomes a single ~80-LOC plugin file in
`src/model/plugins/` that implements `ArchPlugin` (sketched at
`phase4_ext.md` §5.1). Net-new kernel work (e.g. a new MoE routing
variant) lives separately in `compute/`, but the plugin file is the
contributor's only entry point for the registry, parser, chat
template, attention/MoE/engine policy and per-tensor-kind matchers.

| Arch family | Today | Post-roadmap |
|---|---:|---:|
| Dense LLaMA-style (LLaMA, Mistral, Qwen3) | ~60-120 LOC across 25 files | **~50-80 LOC** in 1 plugin file |
| MoE (Qwen3-Coder, Mixtral, Qwen3.5-A3B) | ~280-650 LOC | **~120-200 LOC** (plugin + optional routing variant) |
| Hybrid (Mamba2 + attn) | ~400-800 LOC + weeks of debugging | **~200-300 LOC** (plugin + RecurrentState subclass) once R8 + chunk-state cliff fixed |
| New attention variant (e.g. hypothetical Gemma-5) | ~410-980 LOC across 25 files | **~150-300 LOC** if AttentionPolicy struct covers it |

### 4.2 The 3 sequenced steps that get there

These are R1+R2+R4 from §3.2 above, restated with cost / payoff in
extensibility terms.

| Step | Cost | Payoff |
|---|---:|---|
| **Test parameterization + integration doc** (R1) | 3 days | Removes per-arch fixture copy-paste; new contributor day-7 cost: 3 days vs 2-3 weeks |
| **`ArchPlugin` interface + Qwen3 migration** (R2) | 1 week | Validates the design; first plugin lands without disturbing un-migrated arches |
| **Migrate 13 arches + collapse `if`-ladder via `ModelArchAdapter`** (R3 + R4) | 2.5 weeks | Per-arch behavior is centralized in 1 file; future arches land as 1 file |

After Step 3, **80 % of the extensibility win is in**. Steps R5-R8
are per-feature (qtype, hybrid state) rather than per-arch — their
cost amortizes over future archs.

### 4.3 In-scope vs. nice-to-have

**In scope (commit to doing):**
- R0, R1, R2, R3, R4 (the first 5 weeks). These deliver 95 % of the
  extensibility win and are mechanical post-R0.

**Nice-to-have (do if budget allows):**
- R5 (GemmKernel registry) — strictly speaking is a perf/maint win,
  not extensibility. But it unblocks R6 cleanly. Defer if quarter is
  tight; ship if not.
- R8 (RecurrentState) — only matters when a non-Mamba2/non-GDN
  hybrid actually shows up. Defer until then.
- Cross-chunk SSM state handoff fix (Nemotron-H cliff) — 2-6 weeks
  LOW confidence per P4 §3.6. **Defer** unless Nemotron-H is the
  next priority model.

---

## 5. Cross-axis Big Picture

A senior architect picks the refactors that move multiple axes. The
table below ranks every recommendation in §§2-4 by how many axes it
moves.

| Refactor | Perf win | Maint win | Ext win | Combined ROI rank |
|---|---|---|---|---:|
| **R5 GemmKernel registry** [P3 §11 #2, P4 §6 Step 6] | indirect (clears the dispatch indirection that hides leak #2) | **HUGE** (-1 000 LOC, kills god-struct + dual tables) | **HUGE** (new qtype = 1 file) | **#1** |
| **R4 ModelArchAdapter / kill GEMMA4 if-ladder** [P3 §11 #1, P4 §6 Step 5] | small (clears `if`-noise from hot path) | **HUGE** (-150 LOC, kills 40 inline branches) | **HUGE** (new arch attention/MoE variant = 1 method override) | **#2** |
| **R6 Split `executor_forward_moe.cu`** [P3 §11 #3] | indirect (perf leak #1 fix becomes 1-file change) | medium (-2 563 LOC monolith → 5 files) | medium (MoE routing variant lands in 1 file) | **#3** |
| **M1 Fuse SwiGLU + activation-quant** [P2 #1] | **+5–10 % pp** | small (fewer kernels in the code path) | small | **#4** |
| **M3 Complete MoE prefill graph capture Phase 4** [P2 §6.8] | **+10–15 % pp** | medium (kills 5 D2H sync sites) | small | **#5** |
| **R0 env-var + error-handling sweep** [P3 §11 #5] | microscopic (hot-path env reads vanish) | medium (uniform error model, central env) | medium (every later refactor relies on this invariant) | **#6 (but R0 is the FIRST DOMINO regardless of rank)** |
| **R1 Test parameterization** [P4 §6 Step 1] | **HUGE for the long term** (CI gates new perf regressions) | small | **HUGE** (free e2e coverage for every future arch) | **#7** |
| **R2 + R3 ArchPlugin migrations** [P4 §6 Steps 3-4] | none | medium (-net code in `model.cpp` + `chat_template.cpp`) | **HUGE** (target state §4.1) | **#8** |
| Streichliste (S1-S19 above) | none | **medium** (-5 460 LOC) | tiny (less surface to grep) | **#9** |
| B2 Device-side LRU expert prefetch | **+5× decode on >32GB MoE** | small | medium (any future 50B+ MoE inherits) | **#10 (per-feature)** |

### The single biggest brecher

**R5 — collapse `WeightCaches` god-struct + 21-param `gemm_dispatch_impl`
+ duplicate `weight_dispatch.cu` table into a `GemmKernel` registry.**

Defended by all three phase reports:

- **P2 — perf evidence:** Phase 2 §1.4 hot-path tables show that
  `gemm_dispatch_impl` at `executor_kernels.cu:2003-2269` is on every
  decode token. The 21-param signature + 8 cache-pointer branches add
  µs-grade dispatch overhead per call. More importantly, the dispatch
  shape is what hides leak #2 (the 14.7× NVFP4 prefill gap) inside
  a code surface no one wants to touch.

- **P3 — maint evidence:** Phase 3 §1.2 #1 (the worst coupling),
  Phase 3 §2 #1 (the worst danger zone), Phase 3 §5.2 #4 (the
  duplicate dispatch tables) all point at this single artifact.
  Refactor R5 alone deletes ~1 000 LOC net (-266 in
  `executor_kernels.cu`, -150 in `executor.h`, -100 in
  `weight_dispatch.cu`, -500 in `executor_pre_dequant.cu` from the
  cache-population virtualization, +1 200 across 8 new
  `compute/gemm_kernel_*.cu` files of moved-not-added code) [P3 §11 #2].

- **P4 — ext evidence:** Phase 4 §4 #9 names this as the exact
  blocker for adding a new qtype (INT3, INT2, BFP16, future). The
  current pattern requires touching 6 cache maps, the
  `gemm_dispatch_impl` arm, the `weight_dispatch.cu` arm, and the
  `executor_pre_dequant` populator — five places for one logical
  thing.

- **Calendar weeks:** 2-3 weeks (per P3 §11 estimate).

- **Engineering risk:** medium. The `GemmKernel` interface is
  straightforward, but the move touches 5 hot-path TUs and the
  test surface is wide. Mitigation: use the per-qtype microbenches
  to verify bit-identity before/after each kernel migration.

- **Counterfactual cost (next 6 months):** every new qtype is +1
  week of plumbing; every NVFP4 MoE prefill optimization (P2 #1, M3,
  Bucket B of B1) is harder than it should be because the dispatch
  surface obscures the call graph; the split refactors R6+R7 don't
  trivialize cleanly because the executor still has to know about 6
  cache types. **R5 unblocks R6 and is parallelizable with R4.**

---

## 6. Risks & Mitigations

For each major recommendation in §§2-5. Categorized by what could go
wrong, what test should exist before the refactor, and the rollback
strategy.

### 6.1 Performance refactors

| Item | Regression risk | Pre-refactor test that should exist | Rollback strategy |
|---|---|---|---|
| M1 (SwiGLU + activation-quant fusion) | Numerical drift in NVFP4 MoE expert outputs | Bit-exact A/B test of fused vs unfused at fp16 reference (does NOT exist today; P3 §8.2 confirms 0/10 perf leaks have regression gates) | Feature flag `RuntimeConfig.moe.fused_swiglu_quant`; default OFF for one release |
| M2 (Gemma-4 ggml D2H sync removal) | Silent wrong tokens at n>1 prefill on Gemma-4 ggml fallback | E2E Gemma-4 prefill coherence test at n=512 (`tests/test_e2e_models.cpp::Gemma4ModelTest`) — exists but does not assert n=512 specifically | Branch park; Gemma-4 prefill graphs already opt-in via `IMP_PREFILL_GRAPH` |
| M3 (MoE prefill graph Phase 4) | Silent hang at chunk re-instantiation (similar class to Nemotron-H bug) | Graph-capture-validation test asserting "n=512 prefill MoE captures cleanly to 1 graph" — **does not exist today** [P3 §8.5] | `IMP_PREFILL_GRAPH` already gates; default-OFF for opt-in |
| B1 (close NVFP4 prefill 14.7×) | Many — staged refactor across CUTLASS template, host scheduler, kernel launch overhead | Per-bucket isolated A/B harness (do not exist today) | Each bucket as separate PR with feature flag |
| B2 (device-side LRU expert prefetch) | Pinned-memory exhaustion; expert-cache thrashing under concurrent batching | Multi-request expert cache stress test — does not exist | Branch park; today's `experts_on_host_=true` path is the rollback target |

### 6.2 Maintainability refactors

| Item | Regression risk | Pre-refactor test that should exist | Rollback strategy |
|---|---|---|---|
| R5 (GemmKernel registry) | Silent dispatch divergence on edge-case (input.qtype, output.qtype, cache-state) tuples | Per-qtype dispatch-coverage test (matrix of (input qtype, weight qtype, M-shape) × expected kernel) — **does not exist today**; 12 internal-namespace test files exist [P3 §8.3] but no dispatch matrix | Per-kernel migration is independently revertable; commit one qtype migration per PR |
| R4 (ModelArchAdapter) | Lost arch-specific behavior (e.g. Gemma-4 V=K compaction) | Each Gemma-4 e2e test (`tests/test_e2e_models.cpp::Gemma4ModelTest`, `Gemma4GraphsTest`) plus all 7 Gemma-4 memory-file scenarios [P4 §2.1] | Adapter-by-adapter migration; per-arch revert is one file |
| R6+R7 (split big files) | Build-time include-graph regression; new circular dependencies | None needed (mechanical move) | Trivial — git revert |
| R8 (RecurrentState) | SSM/GDN state corruption on multi-sequence batches | `tests/test_gdn.cu` standalone covers single-sequence (P1 §8); multi-sequence GDN test does not exist | Branch park; SSM/GDN state classes remain independent until R8 ships green |

### 6.3 Streichliste removals — re-introduce-bug-class risk

The MEMORY ledger documents many shipped fixes. A naive deletion can
re-open a closed bug. Audit per item:

- **#1-#2 `mmq_q4k_v2.cu`:** memo `mmq_q4k_v2_v2_phase2_shipped_2026_05_16`
  documents the −4 % E2E regression. **Risk if deleted:** none —
  kernel was opt-in, not on default path. **Mitigation:** keep
  `git tag mmq_q4k_v2_all_phases` for restoration.
- **#3 smallM relocate:** the `IMP_NVFP4_SMALLM` knob exists. **Risk:**
  none — relocate, don't delete. The TMA build-out lives in tests/.
- **#4 `attention_tc.cu`:** P3 §10 #5 notes "still used (header
  included from attention_blackwell.cu:24)". **Risk:** breaking the
  Blackwell variant. **Mitigation:** verify the include is for
  typedefs only; `tests/test_attention_tc.cu` moves to
  `test_attention_blackwell.cu`.
- **#5 `gemm_moe_fused_tc.cu`:** P3 §10 #6 marks this "needs
  profiling whether routed under default." **Risk:** silent fused-
  MoE-GEMV throughput regression. **Mitigation:** add per-PR perf
  test before deletion.
- **#6 `gemm_capture_fp16_sm120.cu`:** P3 §3.7 notes "could be 600
  dead LOC or a working opt-in. Phase 4 should check." **Risk:**
  dropping a working FP16 capture path. **Mitigation:** verify
  dispatch frequency via single nsys trace before deletion.
- **#7-#10 (`__CUDA_ARCH__` guards, runtime flags, comments):**
  trivially safe. Documented in P1 §7.1-7.3.
- **#11 bench TUs relocate:** P1 §7.8 confirms they're already gated
  off in production Docker (`IMP_BUILD_BENCH=OFF`). **Risk:** none
  for runtime; CI must learn the new path.
- **#12-#13 (dead branch, dispatch table merge):** test-pinned. The
  merge needs the existing `tests/test_weight_dispatch.cu` plus a
  matrix coverage test (per §6.2 R5).
- **#14 dead `RuntimeConfig` fields:** P3 §9.5 estimates 8-10. **Risk:**
  re-introducing a knob someone runs in CI without the field. **Mitigation:**
  grep all CI scripts before each removal; mark fields `[[deprecated]]`
  for one release first.
- **#18-#19 (`throw → return`, `assert → log`):** P3 §6 is explicit.
  **Risk:** breaking error-propagation contract for callers that
  catch. Grep for `try { imp::` across `tools/` and `tests/` first.

---

## 7. Things the phase reports got wrong / left ambiguous

### 7.1 Disagreement: `mmq_q4k_v2.cu` — dead code, dormant strategic, or actively harmful?

- **P1 §7.5** says "1 870 LOC at risk pending Phase 2 dispatch-frequency
  check" (treats as soft-removable).
- **P2 §1, §3.2, §3.9** treats it as opt-in research, neither defending
  nor attacking.
- **P3 §3.3, §10 #1-#2** explicitly recommends monomorphizing 5 of 7
  templates (saves 700 LOC) OR deleting the entire 1 667 LOC TU
  (policy call).
- **MEMORY ledger** (`mmq_q4k_v2_v2_phase2_shipped_2026_05_16`):
  "End-to-end on Qwen3.6-35B Q4_K_M: −4 % pp (MoE keeps experts under
  MIN_M=64; fp16_cache hits skip v2; Phase 1 overhead per call). Real-
  world win pending a dense Q4_K_M model without fp16_cache."

**Resolution: needs measurement before deciding.** Ship the dense-
Q4_K_M release referenced in MEMORY first. If it materializes within
30 days and shows ≥ +5 % pp on a real model, monomorphize 5 of 7
templates (P3 §10 #1, 700 LOC) and keep the kernel. If it doesn't,
delete the entire TU (P3 §10 #2, 1 667 LOC). **Default in absence of
that measurement: monomorphize, don't delete** — it's the safer bet
and most of the cost is in the templates, not the kernel proper.

### 7.2 Disagreement: prefill graph capture status

- **P2 §6.8** ("Disagree" with the memo): "the device-args path at
  `executor_forward_moe.cu:508-665` is graph-capturable today (no
  D2H) — Phase 3 PR #164 confirmed +11–39 %. Phase 4 *for non-NVFP4
  MoE arches* is still blocked."
- **P2 §4.5** lists 5 surviving D2H sync sites in
  `executor_forward_moe.cu` that block capture in legacy/Gemma-4
  paths.

**Resolution: P2 §6.8 is right, but the residual 5 D2H sites are
real bugs in non-NVFP4 paths.** Ship M2 (per-token D2H removal in
Gemma-4 ggml) and M3 (Phase 4 prefill graph completion) as separate
PRs. The Qwen3-Coder NVFP4 path is the one already capturable;
Gemma-4 ggml + the legacy fallbacks are not.

### 7.3 Ambiguity: `gemm_capture_fp16_sm120.cu` and `gemm_moe_fused_tc.cu` — dead or alive?

Both flagged in P1 §7.5 and P3 §10 #6-#7 as "needs profiling under
default." Neither phase does the profile. **Punt to a Phase 4.5
measurement task:** single nsys trace on Qwen3-8B Q8_0 (uses FP16 GEMM
extensively) and Qwen3.6-35B-A3B (uses fused MoE). If neither file's
kernels appear in the SASS hot path, delete both; if they do, mark
them as keepers and update the comments.

### 7.4 Ambiguity: dead `RuntimeConfig` fields

P3 §9.5 estimates 8-10 dead fields in `RuntimeConfig` (Gemma-4
stabilization toggles), but doesn't grep CI scripts to confirm.
**Punt:** include this grep as part of R0 (mechanical sweep). Each
field that grep returns 0 hits for in `/scripts/`, `/tools/`,
`/tests/`, and `Dockerfile` is deletable.

### 7.5 Where I disagree with all three phase reports

**The `<500 LOC = new model` framing in P4 §5 is correct in spirit
but optimistic on the LLaMA-style dense case.** P4 §1.1 already shows
the loader plumbing alone is 30-50 LOC across enum + registry + parser
+ matcher rules. With chat template family already covered, the
"plugin file" itself can land in 50-80 LOC, but the surrounding
boilerplate (test fixture, perf-baseline row, docs entry) brings the
true total closer to ~150 LOC even after the refactor. **This doesn't
change the refactor decision** — the win is one-file-per-arch, not
specifically <500 LOC — but the marketing number is closer to
"one file per arch" than "<500 LOC per arch."

---

## 8. 30 / 60 / 90 day plan

A maintainer should be able to execute this without further synthesis.
Each period names concrete deliverables.

### Day 0-30 — Mechanical foundation, first quick wins, biggest perf bet

**PRs to ship (in order):**

1. **R0 — env-var + error-handling sweep** (4-5 days). Deliverables:
   - `RuntimeConfig` extended with all 16 IMP_* fields per
     `phase3_maint.md` §9.1 table.
   - `core/`, `memory/` `throw → return` sweep (~12 sites + 40 callers).
   - Hot-path `assert()` → logged-runtime-check (~55 sites).
   - LOC delta: -20 net.
2. **Perf quick wins #1, #2, #4, #7** (1 day batched). Deliverables:
   - mmvq scratch pre-warm (`executor_kernels.cu:2175-2181`).
   - Stale wgmma docstring deletion.
   - Cache `IMP_NVFP4_FORCE_DEQUANT` + `IMP_LOG_GEMM_ALGO` env reads.
   - Dead `mxfp4_act_sf` branch deletion.
3. **R1 — test parameterization + integration doc** (3 days).
   Deliverables:
   - `tests/test_e2e_models.cpp` rewritten with
     `INSTANTIATE_TEST_SUITE_P` over a model table.
   - `docs/integration.md` worked example.
4. **M1 — SwiGLU + activation-quant fusion** (5-7 days). Deliverables:
   - New fused kernel in `compute/quantize_fp16_nvfp4_moe_native.cu`
     with `apply_swiglu` integrated.
   - A/B harness comparing bit-exact output before/after.
   - Behind feature flag for one release.
5. **Streichliste hard-deletes** (1-2 days).
   - P3 §10 items #8-#11, #13, #15, #16, #17 (low-risk, mechanical).
   - LOC removed: ~700 hard delete + ~13 comment cleanup.

**Day 30 status:** ~700 LOC deleted, R0 + R1 in main, M1 shipped
behind flag, perf quick wins live. **Expected E2E perf delta: +5-10 %
pp on Qwen3-Coder NVFP4 from M1.** Test suite gains free per-arch
coverage for every future model.

### Day 31-60 — ArchPlugin landing, GEMMA4 if-ladder kill, GemmKernel registry

**PRs to ship:**

1. **R2 — ArchPlugin interface + Qwen3 migration** (1 week).
   Deliverables:
   - `model/arch_plugin.h`, `model/arch_registry.cpp`,
     `model/plugins/qwen3.cpp`.
   - All Qwen3 `kArchRegistry` rows + `chat_template.cpp` switch
     entries deleted.
2. **R3 — Migrate remaining 13 arches to plugins** (1.5 weeks).
   Deliverables:
   - 13 plugin files in `model/plugins/`.
   - Each `kArchRegistry` row deleted as its plugin lands.
3. **R4 — ModelArchAdapter / kill 40 GEMMA4 inline branches**
   (1.5 weeks, can start in parallel with R3 once R2 lands). Deliverables:
   - `arch_adapter.h` + `arch_adapter_gemma4.cu` files.
   - All 40 hot-path branches replaced with adapter calls.
   - LOC delta: -150 net.
4. **R5 — GemmKernel registry** (parallel with R4; 2-3 weeks total).
   Deliverables:
   - `compute/gemm_kernel_*.cu` per qtype.
   - `executor_kernels.cu` `gemm_dispatch_impl` deleted.
   - `weight_dispatch.cu` duplicate dispatch deleted.
   - `WeightCaches` god-struct collapsed.
   - LOC delta: -1 000 net.
5. **M2 — Gemma-4 ggml D2H sync removal** (1-2 days, can slot any time).

**Day 60 status:** plugin interface live, all 14 arches migrated,
GEMMA4 if-ladder dead, GemmKernel registry collapses dual dispatch
tables. **`graph/` shrinks from 15.8 KLOC to ~13.5 KLOC.** **Adding
a new arch is now a single 80-LOC file.** Net LOC removed since day
0: ~1 850 (700 + 150 + 1 000).

### Day 61-90 — Splits, MoE prefill graph completion, soft-delete confirmations, B1 setup

**PRs to ship:**

1. **R6 — Split `executor_forward_moe.cu` into 5 TUs** (1 week).
   Deliverables:
   - 5 new files in `src/graph/`.
   - `executor_forward_moe.cu` deleted.
   - Net cognitive load: -60 % per file.
2. **R7 — Split `engine.cpp` into 5 TUs** (1 week, parallel with R6).
   Deliverables:
   - `engine_init.cpp`, `engine_step_prefill.cpp`,
     `engine_step_decode.cpp`, `engine_mtp.cpp`, `engine.cpp`
     (residual).
3. **M3 — Complete MoE prefill graph capture Phase 4** (2-3 weeks,
   started day 61). Deliverables:
   - 5 D2H sync sites in `executor_forward_moe.cu` removed (now
     trivial after R6 split).
   - Multi-graph pool sized per chunk-shape bucket.
   - Expected +10-15 % pp on NVFP4 MoE.
4. **Soft-delete confirmations** (1 week scattered):
   - nsys trace on Qwen3-8B Q8_0 + Qwen3.6-35B-A3B confirms
     `gemm_capture_fp16_sm120.cu` and `gemm_moe_fused_tc.cu` dead-or-
     alive. Delete confirmed dead (~600-1 100 LOC).
   - `attention_tc.cu` deletion (411 LOC) once Blackwell typedef-only
     dependency confirmed.
   - smallM relocate (948 LOC moved out of `src/compute/`).
   - Bench TU relocate (1 772 LOC moved).
5. **B1 setup work** (start of multi-quarter big bet):
   - Per-bucket A/B harness for the NVFP4 prefill 14.7× gap.
   - First bucket-A pilot (CUTLASS scheduler tuning).

**Day 90 status:** **`graph/` shrinks from ~13.5 KLOC to ~10 KLOC.**
M3 ships with measurable +10-15 % pp on MoE prefill. ~3 200 LOC
removed from `src/compute/` (or relocated to `tests/bench/`). The
14.7× gap closes by ~25 % from M1 + M3 combined; the rest is
multi-quarter B1 work.

**Cumulative day-90 deliverables:**
- LOC removed (hard): ~2 850 (R0 -20, M1 0, streichliste hard ~1 700,
  R4 -150, R5 -1 000, soft-deletes confirmed 600-1 100, R7 0).
- LOC removed (soft, with profiling confirmation): up to ~3 800.
- LOC moved out of `src/compute/`: ~2 720.
- New plugin/adapter/kernel files: ~1 800 (mostly relocated, not
  added).
- Decode perf: unchanged at-roof.
- Prefill perf on Qwen3-Coder NVFP4: +15-25 % vs day-0 baseline (M1
  + M3 + half of M5 expected).
- New arch onboarding cost: 25 files / 2-4 days → 1 file / 0.5-1 day.

---

## 9. Open questions for the human

A short list of decisions the maintainer must make. Each has a
recommended default in brackets — these are my opinion, not a
committee's.

1. **Delete `mmq_q4k_v2.cu` (1 667 LOC) now or wait for the dense-
   Q4_K_M release referenced in MEMORY?**
   [Recommend: monomorphize to 2 templates (-700 LOC) now; defer full
   deletion 30 days; if no win on a real model by then, delete the
   remaining 967 LOC. Restore from `git tag mmq_q4k_v2_all_phases`
   if needed.]

2. **Ship B1 (close NVFP4 prefill 14.7× gap) or B2 (device-side
   LRU expert prefetch) as the quarter's big-bet?**
   [Recommend: B1 — the headline gap defines competitive perception.
   B2 only matters for >32GB MoE serving; reassess once B1 lands.]

3. **Move bench TUs from `src/compute/` to `tests/bench/` (1 772 LOC
   relocation)?**
   [Recommend: yes, do it as part of day 61-90. Already gated off in
   Docker; the move costs nothing structurally and cleans the kernel
   tree.]

4. **Delete `attention_tc.cu` (411 LOC) on the assumption that
   Blackwell variant subsumes it?**
   [Recommend: yes, after the typedef-only audit. Move
   `tests/test_attention_tc.cu` into `test_attention_blackwell.cu`.]

5. **Promote `IMP_USE_BITDECODING_QK` to a `RuntimeConfig` field
   AND keep the kernel, or delete the kernel since 0% gain across
   configs?**
   [Recommend: promote to config, keep the kernel. The 0% gain is on
   short ctx; the long-context BitDecoding paper claim (8.6×) is not
   yet reachable on imp because only 1.5 of 4 levers are in. Don't
   close the door.]

6. **Make CUDA Graphs default-ON for prefill once M3 ships (today
   opt-in via `IMP_PREFILL_GRAPH`)?**
   [Recommend: yes for NVFP4 MoE only; keep opt-in for other arches
   until each is verified clean.]

7. **Adopt the `ArchPlugin` design (P4 §5) verbatim, or modify the
   policy struct shape first?**
   [Recommend: adopt verbatim for the first 5 arches; iterate the
   struct shape after Qwen3 + Gemma-4 + Qwen3.6-MoE plugins are
   live. Don't bikeshed the interface before validating it.]

8. **Schedule the cross-chunk SSM state handoff fix (Nemotron-H
   cliff, 2-6 weeks LOW confidence) or defer Nemotron-H until a
   business reason emerges?**
   [Recommend: defer. Mark Nemotron-H as "experimental" in
   `docs/usage.md` until a customer asks. The bug is real but the
   user count is zero.]

---

End of Phase 5 synthesis.
