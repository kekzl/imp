# SETTLED.md — what previous passes already resolved

**Read this before generating audit hypotheses, not while verifying them.**

This file exists because the 2026-07-29 architecture audit spent most of its budget
refuting its own hypotheses. Its §15 names the cause: *"Several of these are the reason
hypotheses 1, 3, 4, 5, 7, 8, 9 and 11 came back REFUTED"* — eight of thirteen described
duplication that earlier campaigns had already collapsed. §19 names the other half:
*"Several dispatch priors were stale … 9 vs 16 architectures, C++20 vs C++23, `src/graph/`
vs `src/exec/`, missing histograms that exist"* — the brief the hypotheses were generated
from carried five verifiably wrong facts about the repo.

Both are generation-stage failures. The verification stage worked; it was fed too much.

## Convention

Same as the memory subsystem's running log (root `AUDIT.md`), and for the same stated
reason: *a suspected problem that turns out to be clean is worth exactly as much as one
that is real, and stops the next pass re-chasing it.*

- **REFUTED** — tested and found not to be true.
- **CONFIRMED** — verified against the code or measured.
- **SUPERSEDED BY S-nn** — was true when written; a later entry changed the thing it
  describes. The entry stays as the record of what was believed.

The **anchor** column is the point. It names the file that makes the entry settled, and
`scripts/check-release.sh` (section 1c, CI job `Release hygiene`) fails if an anchor stops
resolving — so an entry cannot quietly become the next stale prior. If an anchor moves,
re-open the entry; do not just fix the path.

## How to use it

1. Read this file **before** fanning out. Generate hypotheses *against* it.
2. A candidate that contradicts an entry is not reportable until you have disproven that
   entry's **anchor**. "The code changed since then" is the claim — show it.
3. Adding an entry is part of finishing an audit. A refutation you do not record here is a
   refutation the next pass pays for again.

---

## A — Already collapsed (do not re-flag as duplication)

| # | Claim that keeps coming back | Verdict | Anchor — what makes it settled | Since |
|---|---|---|---|---|
| S-1 | Architecture dispatch is duplicated across ~8 sites | **REFUTED** | `src/model/model_profile.h` — five booleans + one `AttnVariant`, decided once; the hot path reads those, not the enum. `layer_swa_window()` (`:73`) unifies four SWA variants so mask and KV sizing cannot drift. `ModelArch::` is in 15 files, 9 of them `src/model/`; arch #17 costs 6 files | 2026-07-29 (D1) |
| S-2 | Quant dispatch is duplicated across 8 formats × 5 concerns | **REFUTED as stated** — CONFIRMED narrowly for KV dtypes only | `src/exec/gemm_kernel_registry.cu` — an 85-LOC table replacing "~33 hand-written kernels", plus 8 `DequantTraits<>` in `src/compute/gemv_dp4a_traits.cuh`, pinned by `tests/test_gemm_kernel_registry.cu` (40 tests of the dispatch *contract*). A new quant format on the GEMM path costs 2 files | 2026-07-29 |
| S-3 | GGUF and SafeTensors loaders duplicate tensor mapping | **REFUTED** | Name normalisation and layer indexing live once in `src/model/weight_map.cpp` + `src/model/tensor_kind_table.cu`; the loader bodies differ only where the formats differ | 2026-07-29 |
| S-4 | The KV cache has more than one implementation | **REFUTED** | One `src/memory/kv_cache.h` / `src/memory/kv_cache.cu`, backed by one `src/memory/block_pool.h` | 2026-07-29 |
| S-5 | Execution paths multiply (eager × graph × batch × spec) | **REFUTED** | One `step()` → `step_decode()` → `step_decode_forward()` chain in `src/runtime/engine.cpp`; the variants are parameters on it, not parallel paths | 2026-07-29 |
| S-6 | Per-architecture layer primitives are re-pasted | **REFUTED** | RMSNorm, RoPE, SwiGLU and MoE top-k have one implementation each under `src/compute/` | 2026-07-29 |
| S-7 | Sampling is several overlapping chains and constrained decode bypasses part of it | **REFUTED** | `apply_constraint_mask()` at `src/exec/executor.cu:56` — nine lines called from every sampling path. **Keep the comment with the code**: it records that four copies of this chain is exactly how an unmasked path ships | 2026-07-29 |
| S-8 | The legacy materialised cuBLAS prefill is a vestige (~18 % of prefill) | **REFUTED, with one exception** | 0.0 % on hd=128/256. **Gemma-4's hd=512 global layers take it by design and by measurement** — `src/compute/attention_cublas.cu`, gated per layer at `src/exec/executor_attention_prefill.cu`. It is a deliberately retained tier, not a vestige | 2026-07-29 (F-16) |
| S-9 | Nemotron-H / SSM carries a private mini-runtime | **REFUTED for SSM**; CONFIRMED for vision | `src/compute/ssm.cu` and `src/memory/recurrent_snapshot_store.h` sit in the shared tree, not beside the model | 2026-07-29 |
| S-10 | `tools/`, `tests/` and the bench CLI duplicate benchmark logic | **REFUTED for benching** — CONFIRMED for arg parsing (27 flags, F-15, still open) | `tools/imp-bench/` has no overlap against `tests/`; the perf gate reads `tests/perf_baseline.json` through `scripts/bench_gate.sh` | 2026-07-29 |
| S-11 | NVFP4 grouped-GEMM has two competing paths | **CONFIRMED — but a designed 4-tier ladder, not a twin** | `src/exec/moe_prefill_decision.h` documents the tiers; each is selected by explicit preconditions and the bottom one serves host-offloaded experts. No death date exists for it | 2026-07-29 |

## B — Deliberate specialisation (consolidating this is the classic false positive)

Named in the 07-29 audit §15.12 as D6 — the delta is per-dtype control flow in the hottest
loop, and the shared part is already factored out.

- **Paged-decode kernels per KV dtype** — `src/compute/attention_paged.cu`,
  `attention_paged_fp8.cu`, `attention_paged_fp8_tile.cu`, `attention_paged_int4.cu`,
  `attention_paged_int8.cu`, `attention_paged_nvfp4.cu`, `attention_paged_nvfp4_tc.cu`.
  The ~35 % token overlap is the online-softmax rescale, which
  `src/compute/attention_paged_common.cuh` already holds.
- **The `mmq_q8_imma` family** — `src/compute/mmq_q8_imma.cu`, `mmq_q8_imma_q4k.cu`,
  `mmq_q8_imma_q6k.cu`, `mmq_q8_imma_q51.cu`. Same reasoning: the delta is the dequant
  inner step.
- **`src/compute/gdn_scan.cu` vs `src/compute/gdn_scan_tc.cu`** — scalar vs tensor-core
  scan, different arithmetic intensity.
- **Prefill vs decode kernels throughout** — different arithmetic intensity, by definition.
- **`src/vision/vision_encoder.cu` (SigLIP) vs `src/vision/qwen3vl_encoder.cu`** —
  genuinely different tower architectures.
- **Domain-cohesive files that are large but not conflated** — `src/compute/gdn.cu`,
  `src/compute/sampling.cu`. Split on conflation, never on size; the metric is recompile
  blast radius (`tools/filesize_thresholds.toml`).
- **`GraphExecutor` in `src/exec/`** — intrinsically forward-pass-coupled. The D2
  runner-extraction track ENDED in a cosmetic friend-backdoor; the header split is already
  done (1188 → 584 lines). Do not re-attempt runner classes. The `compute/` files are not
  god-files either.

## C — Hunted and absent (07-29 §6.2 — the sweep already ran)

- No `#if 0` blocks anywhere in `src/` or `tools/`.
- No `_v2` / `_new` / `_old` symbol pairs. The suffix hunt returned only algorithm-local
  variables (`m_new`, `h_old`) and legitimate API names (`inc_ref`, `mirostat_v2`).
- Exactly one file with `legacy` in its name — `src/exec/executor_forward_moe_legacy.cu` —
  and it is a reachable floor, not a twin.
- Zero stale-target *code*. All 24 mentions of `wgmma` / `tcgen05` / `TMEM` / `sm_100` in
  `src/` are comments stating the feature does **not** exist on sm_120a. They read like
  clutter and they are load-bearing — they stop the next reader reaching for a
  datacenter-Blackwell design.
- **A `VramOwned` type does not exist.** Past audits hallucinated it.
- **The `src/exec/executor_kernels.h` decl-only sweep ran on 2026-08-03.** Three kernels had
  a declaration and a definition and nothing else — no launch, no address-taken use, no
  test — and were removed: `add_fp16_bias_to_fp32_kernel`, `write_kv_cache_fp8_kernel`,
  `write_kv_cache_kernel`. **Every other kernel declared in that header has a caller**; do
  not re-run this sweep on it without a reason. Two of the three are worth remembering
  because they are different failure modes: the FP8 one was orphaned by `d5dd4bbd`, which
  replaced its four launch sites with one `write_kv_cache_fp8_fused_kernel`; the bias one
  was **never launched at all** — it entered the tree under the old `src/graph/` path with
  no caller, gained a header declaration it never needed, and was carried through the
  file-size split #784 by three commits that each moved it without noticing.
- **`write_kv_cache_fused_kernel` is the only FP16 KV-write path, deliberately.** The
  non-fused twin is gone (above), so its absence is not a gap. Its coverage moved onto the
  fused kernel in `tests/test_kv_cache_write.cu`: multi-token writes across a block
  boundary, and the 2-D `bt[seq * max_blocks_per_seq + block_idx]` indexing every paged
  write kernel shares — the only direct assertion on that indexing in the write path.

- **The non-kernel decl-only sweep ran on 2026-08-03** (the sibling of the kernel sweep
  above: `ccg coverage` only looks at `__global__`). Ten functions were declaration +
  definition and nothing else, and were removed: the four CuTe builders
  `build_tma_a/_b/_sfa/_sfb` (superseded by `build_tma_2d_u8` in the same file — their
  removal also dropped three `cute/` includes from that TU), the four **empty-bodied**
  `// Legacy stubs` `gdn_decode/_prefill/gdn_scan_decode/_prefill` (GDN really runs through
  `gdn_scan_chunkwise_*` / `gdn_scan_fused_*`), `GraphExecutor::forward_batch`, and
  `jinja::Template::render_string`.
  **Do not re-run this sweep raw.** It starts at 530 candidates and is dominated by four
  false-positive families the graph cannot see through: self-registering hooks bound by
  macro (`*_reset_static_cuda_state` — the R-11 mechanism, and it looks deadest of all),
  C-ABI exports under `include/` whose consumers are outside the repo, device-side inline
  helpers in `.cuh` used inside kernel bodies, and templates. 530 → 201 by occurrence count
  → 27 by the decl+def signature → 10 after reading each one. The other 17 were verified in
  a second pass the same day and removed too, so **the 27 are fully resolved** — do not
  re-open them.
- **`process_diag_set_mxfp4_blockscale` was a redundant mutator, NOT a dead knob** — worth
  keeping straight, because it looked like one. `process_diag.cpp` installs the flag from
  `cfg.attention.mxfp4_blockscale` and `attention_dispatch.cu` reads it, so the config path
  never depended on the setter. The setter is gone; the knob works.
- **Two whole modules were dead and are gone (2026-08-03):**
  - `src/compute/gemv_ggml_compat.h` + `src/compute/gemv_ggml_compat.cu` (174 lines). Its only export had no callers, its
    kernel was launched only by that dead wrapper, and the three
    `#include "compute/gemv_ggml_compat.h"` in the MoE executors were the sole mention in
    those files.
  - `src/core/threading.h` + `src/core/threading.cpp` (88 lines) — a `ThreadPool` class that appeared nowhere
    but its own declaration and definition. `core/threading.h` was included by exactly one
    file: its own `.cpp`. (Grepping `ThreadPool` across the repo hits Python files where
    the word is coincidental — restrict to C++ globs.)

**`ccg coverage` is a ONE-LEVEL check, not reachability.** It asks "does this kernel have a
launcher", not "is that launcher reachable from a live root". `gemv_q4k_ggml_compat_kernel`
counted toward its 420/423 *live* precisely because its launcher existed — and the launcher
was itself dead. So "every kernel has a caller" is a weaker statement than it sounds, and an
earlier note in this file that read as "all 423 kernels are explained" was too strong.
For the real question, BFS the `calls`/`references`/`instantiates` edges from roots in
`tools/`, `tests/` and `src/api/`: 3242 `src/` functions, 688 unreached. Cluster by file
before reading anything — a file where *every* symbol is unreached is the signal; single
symbols are mostly the blind spots below.

**Blind spots that dominate the unreached set — do not re-flag these:**
`src/exec/gemm_kernel_*.cu` (the registry leaves of S-2, bound through a table),
`src/core/logging.cpp` (reached only through the `IMP_LOG_*` macros),
`src/memory/alloc_interpose.cpp` (behind `IMP_ALLOC_INTERPOSE`, default OFF),
`src/quant/turboquant_fp4.cuh` (device inline helpers), and **destructors** —
`src/vision/vision_model.cpp` looked dead and is not: it holds `~VisionModel()` and
`free_gpu()`, and implicit destructor calls are invisible to the graph.

**Method note for "is this dead" findings.** A call-graph tool reporting *no callers* is
only evidence if the tool can see calls at all. Control it against a symbol you have
already proven live before you trust the negative — for the sweep above, `codegraph callers
write_kv_cache_fp8_fused_kernel` and `… write_kv_cache_nvfp4_kernel` both correctly returned
`write_kv_cache`, which is what made the two empty answers meaningful.

## D — Load-bearing; a "cleanup" here is a regression

| # | Thing | Why it must survive |
|---|---|---|
| S-20 | `src/memory/span.h` — `StableSpan` vs `DeviceSpan`, passkey-enforced | Encodes in the type system which memory a captured CUDA graph may bake an address into. Turns a bug class that actually happened (`AUDIT.md` B9/B13) into a compile error. Extend it; do not weaken it |
| S-21 | `src/memory/plan.h` — `plan_memory()` | Plans capacity without ever querying the device; pure function of a plain struct, runs in the CPU-only CI lane. Retired the #1103 class (free VRAM swinging 1.6 GB between identical invocations → different plan) |
| S-22 | The `throw` in `src/compute/attention_dispatch.cu` (#654) | "No tier accepted" is an error, not a degraded answer. It replaced a silent fallback that produced teacher-forced PPL ~1e10. Never soften it back into a fallback |
| S-23 | Zero virtual dispatch in `src/exec/` and `src/compute/` | Not one vtable call per token, layer or launch. Polymorphism is templates and traits |
| S-24 | `src/core/cuda_raii.h` | Move-only wrappers: deleted copy, `noexcept` moves, `[[nodiscard]] create()`, no exceptions in destructors |
| S-25 | `src/runtime/engine_init_resolver.cpp` | ~25 log lines stating each resolved policy **and its reason**. An operator can read what the engine decided and why |
| S-26 | `tools/filesize_thresholds.toml` `[allow]` with reason strings | Manages file size instead of silencing it; the gate rejects an empty reason |
| S-27 | `tools/alloc_allowlist.txt` as a two-way ratchet | Fails on a new allocating file *and* on a listed file that stopped allocating, so the list cannot go stale in either direction |
| S-28 | Property batteries in the CPU lane | `tests/test_json_constrain_property.cpp`, `tests/test_schema_constrain_property.cpp`, `tests/test_tokenizer_robustness.cpp`, `tests/test_gguf_fault_injection.cpp` — fuzzed in CI, no GPU |

## E — Findings that were themselves wrong (do not re-derive them)

From the 07-29 resolution log. These cost real time and the corrections matter more than
the fixes did.

- **F-16's "device sync per layer" is false.** The only `cudaDeviceSynchronize` in
  `src/compute/attention_cublas.cu` is inside `attention_cublas_prewarm()`, called once
  from `src/runtime/engine.cpp`. The prefill function contains none. The audit read one
  call site; grepping the file settles it without a profile.
- **F-17 does not reproduce.** The CUTLASS grouped GEMM genuinely never consults
  `process_diag_deterministic_gemm()` (`src/runtime/process_diag.h`), but the determinism
  E2E test is 3/3 green with bit-identical greedy output and perplexity.
- **F-4's count was wrong** — 3 files was actually 1 (#1206).
- **#1205's resolved-dispatch line never printed.** The call sat before the final `return`
  of `Engine::step()`, which the graphs-ON decode path never reaches (fixed in #1210). A
  gate that cannot be shown to fire has not been validated.

## F — Verify the brief before you generate from it

Every fact below was asserted by the 07-29 audit brief and was wrong. Where the brief and
the repo disagree, **the repo wins** — but only if someone checks. Each counter-check is
one command.

| Brief said | Reality | Counter-check |
|---|---|---|
| 9 architectures | **16** enumerators | count them in `src/model/model_arch.h` |
| 6 tested models | ~30 validated checkpoints | `docs/supported-models.md` |
| C++20 | **C++23** | the standard in `CMakeLists.txt` |
| `src/graph/` exists | renamed to `src/exec/`; `src/lora/` exists and was unlisted | `ls src/` |
| no p50/p99 histograms | Prometheus histograms exist | `grep _bucket` in `tools/imp-server/handlers_misc.cpp` |
| `/v1/messages` streaming is synthetic | real handler plus the shared stream driver | `tools/imp-server/handlers_messages.cpp` |

## G — Open: NOT settled, and not to be treated as closed

Still open from 07-29 with the reason each was not shipped blind — see
[`AUDIT_ARCH_2026_07_29.md`](AUDIT_ARCH_2026_07_29.md) for the full argument.

- **All 76 `IMP_LOG_DEBUG` sites are unreachable** (found 2026-08-03, not fixed). Every one
  is guarded by `log_get_level() <= LogLevel::DEBUG`; `g_log_level` in
  `src/core/logging.cpp` initialises to `INFO`, and its **only** writer is `log_set_level`,
  which nothing calls. There is no config key and no CLI flag for the level. So the engine
  has a debug-logging facility that cannot be switched on. `log_set_level` was left in
  place deliberately — it was on the decl-only removal list, and deleting it would have
  cemented the gap and taken away the obvious hook. The fix is a `RuntimeConfig` key that
  calls it (no ad-hoc env read), not a removal.
- **F-3 (rest)** — routing replica. **This entry understated the gap until 2026-08-03**: it
  described the residual limit of the *attention* half (a tier reordered ahead of the winner
  stays invisible, because the chain short-circuits and never asks it) as if that were all
  that was left. The **MoE** half had no runtime check at all — `select_moe_prefill_path`
  had zero production callers, ten test callers, and a comment in
  `src/exec/executor_forward_moe.cu` asking for the two predicates to be kept in sync by
  hand. Found with the call graph, confirmed by grep. Both halves now replay the model
  against what the chain observed (`verify_against_moe_routing_model` in
  `src/exec/executor_forward_moe_cutlass.cu`, mirroring
  `src/compute/attention_dispatch.cu`); the short-circuit limit above is what genuinely
  remains, for both. **Lesson for this ledger: an open entry that records one half of a
  symmetric problem reads as if the other half were closed.**
- **F-5 (rest)** — GPU CI lane: **declined by the repo owner, 2026-08-03.** The job and its
  nightly trigger stay dormant in `.github/workflows/ci.yml`. Consequence: `make verify-fast`
  locally is the only thing that ever runs a CUDA kernel against correctness or perf.
- **F-6 (rest)** — 20-39 % of steady-state VRAM unattributed.
- **F-9** — cuBLASLt algorithm selection unpinned (mechanism confirmed, magnitude refuted).
- **F-10** — `src/runtime/config.h` included by 22 files in `src/exec/`.
- **F-12** — `src/memory/vram_allocator.cu` still has 84 references.
- **F-24** — `src/runtime/engine.h` god-header.

## Per-area running logs

- **Memory subsystem** — root `AUDIT.md` (CONFIRMED / REFUTED / OPEN per finding, negative
  results included). Read it before sweeping `src/memory/`; design lives in
  `docs/MEMORY_ARCHITECTURE.md`.
- **Architecture / dispatch** — this file, plus
  [`AUDIT_ARCH_2026_07_29.md`](AUDIT_ARCH_2026_07_29.md) for the evidence behind each entry.
- **File size** — [`AUDIT_FILESIZE.md`](AUDIT_FILESIZE.md), per-file rationale.
