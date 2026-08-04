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

- **The debug-logging facility could not be switched on, and now can** (found and fixed
  2026-08-03). All 76 `IMP_LOG_DEBUG` sites are guarded by
  `log_get_level() <= LogLevel::DEBUG`; `g_log_level` initialises to `INFO` and its only
  writer, `log_set_level`, had no callers and no config key — so the guard never opened.
  `log_set_level` was on the decl-only removal list by signature; removing it would have
  cemented the gap and taken away the hook. Fixed with `diagnostics.log_level`, applied in
  `process_diag_install()` — the one function that runs from both tool mains *and*
  `Engine::init`, so a C-API consumer reaches it too. An unrecognised word warns and keeps
  the current level rather than falling back to `INFO`, which would have restored the same
  silent-default failure. **Measured, not assumed:** same model, same prompt, default level
  → 0 `[DEBUG]` lines, `--set diagnostics.log_level=debug` → 359 from 10 source files, both
  runs confirmed to have actually loaded the model (the first attempt compared two *failed*
  runs at 0 vs 0 because the model path was wrong — a control that proves nothing).

- **F-6 (VRAM attribution) is CLOSED — and it was closed before this ledger listed it as
  open.** The 07-29 audit measured 20-39 % of steady-state VRAM unattributed against a
  ≥95 % criterion. The memory campaign then fixed it (`AUDIT.md` B80/B81: the pool ledger
  is always on instead of gated behind `--mem-report`, and the library charge is measured
  over the whole init rather than the warmup-forward window). Re-measured 2026-08-03 on the
  three config families the audit named plus one more — dense GGUF (was 39 %), MoE (was
  20 %), and two NVFP4 dense — **all read 99.9-100.0 % accounted, residual 0-16 MiB**.
  `docs/MEMORY_ARCHITECTURE.md` still carried the old 61-80 % table, which is where the
  stale prior came from; corrected there too. **This is the ledger's own failure mode
  caught in the act**: an entry that was true when written, describing a subsystem whose
  running log (`AUDIT.md`) recorded the fix, in a section headed "NOT settled". Reading the
  per-area log at step 0 is what stopped a day of re-measuring something already done.

**Layering: two backward edges the 07-29 audit's §11.1 table does not list, both closed
(2026-08-03).** That table is otherwise complete and its verdicts stand — `model → vision`,
`exec → runtime`, `compute → runtime`, `core → compute` are all there with reasons. Two were
missing:

- **`compute → exec`** — `src/compute/weight_dispatch.h` included `exec/weight_handle.h`
  because its three signatures take `const WeightHandle&`. One backward edge against forty
  the other way. `WeightHandle` depended on nothing but `core/`, so it moved to
  `src/core/weight_handle.h`; `WeightRegistry` stays in `exec/`, being the executor's
  container rather than something the kernels need. Edge count now **0**.
- **`quant → compute`** — `src/quant/mxfp4_gemm.h` included
  `compute/gemm_cutlass_mxfp4_sm120.h` for `CutlassMxFP4Weight`, a POD with no project
  dependencies that describes a *quantised weight layout*. Moved to
  `src/quant/cutlass_mxfp4_weight.h`, which is where it belongs by meaning as well as by
  layer. **One `quant → compute` edge remains and is NOT an include artefact:**
  `src/quant/nvfp4_gemm.cu` calls `gemm()` because the NVFP4 fallback dequantises to FP16
  and hands off to the dense GEMM. Inverting that moves a dispatch decision on a GEMM path,
  which is not a thing to do unmeasured on a repo with no GPU CI lane.

Also closed: the two `compute → model` includes §11.1 called avoidable —
`embedding.cu` and `gemm_dp4a.cu` pulled the 800-LOC `model/model_config.h` with the comment
`// QType` while using neither `ModelConfig` nor `FFNActivation`. `QType` is in
`core/qtype.h`. **9 → 7**; the remaining seven are the real ones (tokenizer for the four
constrainers, `model.h` for `encoder_forward`, `mtp_head`, `FFNActivation`).

**Checked on the same sweep and NOT findings:** god-functions by callee count — the list is
led by `forward_logits` (216), which is the forward pass this ledger already records as
intrinsically coupled; and repeated function names — the top entries are `operator=` (27
files), `init` (19), `reset` (16), i.e. noise, not copy-paste. `kv_cache_dtype` "in 4
representations" (audit Track F) could not be confirmed either way: the name barely appears
in the tree today, and reconstructing the 07-29 vocabulary was out of scope.

**Build cost, quantified (2026-08-03).** CLAUDE.md says the metric is recompile blast
radius, not line count — this is what that actually costs. Transitive header fan-in across
the 450 translation units, multiplied by commits in the last six months:

| header | TUs rebuilt | commits/6mo | cost |
|---|---:|---:|---:|
| `src/runtime/config.h` | 85 | 130 | **11050** |
| `src/runtime/engine.h` | 41 | 129 | 5289 |
| `src/exec/executor.h` | 77 | 55 | 4235 |
| `src/model/model_config.h` | 133 | 31 | 4123 |

So the two open findings **F-10 and F-24 are the #1 and #2 build-cost items in the repo** —
the audit argued them on coupling and churn, and did not have the product. `core/qtype.h`
and `core/tensor.h` have the widest fan-in (254, 248) and are nowhere near the top: they
barely change, which is what a core type should do.

**Three cheap-win hypotheses died on measurement — do not re-run them:**
- *Trim the includes and forward-declare instead.* Dead on all three top headers: 31 of 33
  `config.h` includers read real members, 22 of 24 for `engine.h`, 42 of 43 for
  `executor.h`. This confirms the audit's wording that `config.h` in `exec/` is
  **algorithmic**, not incidental — the only real fix is the deferred `DispatchPolicy`
  extraction.
- *The file-size gate must be missing the big churny `.cu`s.* No: `check_filesize.py`
  reports `violations=0`, and `weight_upload.cu`, `executor_workspace_buffers.cu` and
  `cuda_graph.cu` are all allowlisted with reasons. A hand-rolled LOC grep reads ~11 %
  higher than the tool because it strips comments differently — use the tool.
- *Some `.cu` is a split candidate on cost alone.* The repo's rule is split on conflation,
  never on size, and the top of the churn×LOC ranking is allowlisted on cohesion grounds.

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
  **The MoE verifier is runtime-verified**, on Qwen3-30B-A3B-NVFP4-Modelopt, all three
  tiers, both directions — the real image silent and a build with
  `select_moe_prefill_path` forced to `LEGACY` firing and naming both answers, for
  `device_args`, `grouped` (`--set moe.nvfp4_device_args=false`) and `small_m`
  (`+ --set moe.nvfp4_smallM=true`). The silent run alone would not have shown this: a
  check that is never reached is silent too, which is exactly how #1205's resolved-dispatch
  line went unnoticed. The one-shot guard also holds — one error line although the tier
  fires on every layer.
- **F-5 (rest)** — GPU CI lane: **declined by the repo owner, 2026-08-03.** The job and its
  nightly trigger stay dormant in `.github/workflows/ci.yml`. Consequence: `make verify-fast`
  locally is the only thing that ever runs a CUDA kernel against correctness or perf.
- **F-9** — cuBLASLt algorithm selection unpinned. **Closed 2026-08-04 by fixing the
  estimator, not by persisting the result** (`src/compute/gemm.cu`). The magnitude the audit
  inherited was never measured; measuring it is what redirected the fix.
  **Measured on Qwen3-1.7B, 5 fresh processes, 8 shapes** (`diagnostics.log_gemm_algo`, and
  note the path is only reachable at all from dense BF16/FP16 SafeTensors and the vision
  encoders — a Q8_0 GGUF, including the perf-gate model, never enters it):
  the instability is **entirely in near-ties**. Every shape whose best candidate is genuinely
  ahead already picked the same one every time — M=512 N=6144 K=2048 spans 0.196-0.449 ms and
  chose cand[0] in 5/5, its cost reproducing to **0.3 %**, and it chose right even in a run
  where cold clocks inflated everything ~5x. Every unstable shape had its top candidates inside
  ~5-10 %, i.e. inside the measurement's own error, so what a flip costs is bounded by how close
  the tie is. Result 4/8 shapes stable before, **7/8 after**, the eighth alternating between two
  candidates that are equal within noise *and* both 8-45 % ahead of the heuristic — so it now
  takes a gain it previously threw away. No load-time cost (alternating A/B: 60.6 s vs 61.8 s
  median) and no throughput cost (311.0 vs 308.8 tok/s).
  **R-16 (persist the algo per shape+dtype) is REJECTED, not deferred.** Selection timed each
  candidate exactly once; a cache would have frozen whatever that noisy first run chose and
  handed it to every later process, converting a per-process mispick into a permanent one — on
  top of needing invalidation against driver and cuBLAS version, for an opaque blob cuBLAS does
  not promise is portable across library versions. Anchor: the timing loop in
  `benchmark_and_select_algo`, `src/compute/gemm.cu`.
  **Two designs were built and measured and did NOT work — do not re-try them blind:**
  (1) *k interleaved rounds, per-candidate minimum across rounds.* Assumes per-sample jitter.
  The noise is not jitter but **sustained slow windows**: a contaminated run inflates every
  candidate together (one run put all 8 candidates within 20 % where a quiet run spreads them
  over 40 %), so a minimum taken inside that window is just as wrong, and the longer selection
  window even made the previously-stable M=512 shapes flip.
  (2) *Requiring the same candidate to win every round.* Two challengers that are tied with each
  other but both clearly ahead of the heuristic split the round wins, unanimity fails, and the
  fallback keeps a base that is 8-45 % slower.
  What works is a **paired comparison inside each round** — every candidate against `base` timed
  in that same round, and it must win by `kAlgoMargin` in *every* round. Contamination scales
  both sides, so the ratio still carries signal where absolute times carry none. Plus sizing the
  timed window (`kTargetWindowMs`) instead of fixing the rep count: five reps at M=16 timed
  ~30-120 us, which is mostly launch overhead, and that is why small M was a coin flip while
  M=512 was not. **Mutation-validated**: with `base` forced to the *last* valid candidate, the
  switch fires on all four shapes with a real spread (M=512 N=6144 back to cand[0], 2.2x) and
  correctly stays put on the two where nothing is 10 % better.
  **A 3 % margin was tried first and was useless** — it sat below the residual noise. Set a
  threshold like this from the measured spread, not from what sounds tight.
- **F-10** — `src/runtime/config.h` included by 22 files in `src/exec/`. **Scoped by
  measurement 2026-08-03, and the audit's estimate is low by 2x.** It proposes extracting
  "the ~30 dispatch-relevant keys" into a `DispatchPolicy` POD; `src/exec/` actually reads
  **59 distinct RuntimeConfig leaves** across nine sections — gemm 15, attention 13, moe 12,
  diagnostics 7, gdn 6, generation 2, ffn 2, speculative 1, kv_cache 1 — at 91 read sites in
  21 of the 22 files. So the POD is 59 fields and every read site changes.
  **Access is per-object, not global**: `runtime_config()` in `exec/` is a member accessor
  on `GraphExecutor` (`src/exec/executor.h:422`) and `QuantPipeline`
  (`src/exec/quant_pipeline.h:86`), reached as implicit `this->`. `exec/` includes no
  `engine.h`. Anyone reading the 79 unqualified call sites as a process-global will
  mis-model the fix — there is still no process-global `RuntimeConfig`.
  **Both cheap alternatives are closed, tested:** forward-declaring instead of including
  fails because 21 of 22 files read real members and the owning classes hold the type by
  reference; and splitting `config.h` by section does not help for the same reason, even
  though **roughly half the churn (Runtime 67, Vram 19, Server 5, Rope 5 of ~194 hunks)
  lands in sections `exec/` never reads**. The saving is real but unreachable without
  changing what the owning classes hold — which is the POD extraction itself.
  Cost of leaving it: 85 TUs x 130 commits/6mo, the highest in the repo.
  **Design settled 2026-08-04, three facts the R-18 sketch does not mention:**
  (a) `engine_init_resolver.cpp` still *mutates* the config after load
  (`runtime.deterministic_gemm = true`, two sites), and `GraphExecutor` holds a
  live `const RuntimeConfig*`, not a copy — so a resolved-once POD is a semantic
  change, not only a mechanical one. It is safe **only** if filled after the
  resolvers: they run at `engine.cpp:851-856`, `set_runtime_config()` fires from
  `init_weights()` at `:972`, and nothing writes the config after that.
  (b) The cheaper shape is not the audit's 59-field POD but the **nine sections
  by value** — `gemm attention moe diagnostics gdn generation ffn speculative
  kv_cache` — which turns all 91 read sites into a prefix rename
  (`runtime_config().gemm.x` -> `dispatch_policy().gemm.x`) and removes the
  drift risk between POD and config that a hand-enumerated field list carries.
  (c) Those sections are nested inside `RuntimeConfig`, so they must be lifted
  out — and that is nearly free: **exactly one** explicit `RuntimeConfig::<Section>`
  reference exists across `src/ tools/ tests/`.
  What remains is volume, not uncertainty: lift nine structs into `core/`, fill
  the policy at `:972`, rename 91 accessors, drop the include from 22 files.
- **F-12** — `src/memory/vram_allocator.cu`: **56** live references (2026-08-04), down from
  67 a day earlier and from the audit's 84. Still open, but shrinking; count with the allocator's own files and comment
  lines excluded or you get 103 and read a regression that is not there — and count the type
  name alone, since adding `vram_alloc`/`vram_alloc_force` gives 104.
  **First consumer migrated 2026-08-04: the Qwen3-VL vision tower** (`qwen3vl_vision_upload.cpp`),
  which the audit named as the place the tier model was never applied. 792.2 MiB on
  Qwen3-VL-4B now comes from the T2 engine arena instead of `VRAMAllocator`.
  **The blocker that had to be solved first — and the reason to solve it the same way for the
  next consumer:** the arena opens at `engine.cpp:915`, *before* `init_weights()` and long
  before the vision warmup uploads anything, so a tenant whose bytes are only known at upload
  time cannot be charged. `qwen3vl_vision_tower_device_bytes()` answers it from shapes alone by
  walking `qwen3vl_visit_vision_tensors` — the same list the upload walks, so the reservation
  cannot drift from what is taken. Verified exact: predicted 792.2 MiB, uploaded 792.2 MiB.
  **Second increment, 2026-08-04: the pipeline and encoder scratch followed** (a further
  224.1 MiB on Qwen3-VL-4B at the default 4096-patch budget). The "`max_patches` is not known at
  arena-open time" objection this entry recorded an hour earlier was **wrong**: the budget is
  `runtime.vision_max_patches` (or 4096) rounded down to the merge unit, all config, none of it
  weights. It now lives in one place, `Qwen3VLPipeline::patch_budget()`, called by both
  `Engine::init` for sizing and the vision warmup for the actual init, so the two cannot drift.
  `src/vision/` references: **19 → 8**, repo-wide **67 → 56**.
  **The anti-drift device is the finding worth keeping.** Tower sizing could reuse the upload's
  own visitor; the scratch cannot, because the buffers are assigned to named members. So
  `demand_bytes()` duplicates the list and `taken_bytes()` accumulates what init actually took,
  and `Qwen3VLPipelineTest.ReservedBytesMatchTakenBytes` asserts they are equal. It earned its
  keep on the first run by failing: `taken_bytes()` counted only the pipeline's own 32.0 MiB and
  not the encoder's 192.1 MiB. **Mutation-validated**: dropping the FFN term from
  `Qwen3VLEncoder::demand_bytes` under-reserves by 32 MiB and the test fails.
  **Still NOT migrated:** Gemma's mmproj tower is a separate GGUF that genuinely is not loaded at
  arena-open time (`vision_pipeline.cpp`, `vision_encoder.cu`, 8 references). Migrating it needs
  the mmproj either loaded or stat-ed before the arena opens — unlike `max_patches`, that is a
  real ordering change, not an arithmetic one.
  **Ownership note:** the previous scheme documented the tower's blocks as caller-owned
  precisely because "a tower holding pointers into an allocator it does not own is a
  use-after-free the moment a teardown order puts the allocator first". The arena removes the
  per-block free entirely, so that hazard is gone rather than relocated.
  **Test trap, hit and fixed:** `Qwen3VLPipelineTest` builds its own `VRAMAllocator` and no
  arena, so all four cases failed the moment the tower became an arena tenant. Fixtures now open
  a `ScopedEngineArena` sized from `qwen3vl_vision_tower_device_bytes()` rather than a literal,
  so a changed fixture checkpoint cannot silently outgrow it. **Mutation-validated**: halving the
  fixture arena fails all four. Note these tests were *skipped*, not passing, until
  `IMP_TEST_MODEL_QWEN3VL` was set — the run that looked greenest tested nothing.
  **Second trap, found by setting that variable: `Qwen3VLPipelineTest` only runs from the source
  tree.** It loads `tests/fixtures/vision_test_64.png` by relative path and the `imp:test` image
  ships no `tests/` directory, so setting `IMP_TEST_MODEL_QWEN3VL` for an image run turns three
  silent skips into three `encode_file` failures that look like a code defect and are not. Run
  them with the repo mounted (`-v $PWD:/src -w /src ./build-dev/test-e2e`), or leave the variable
  unset for image runs — the documented full-suite recipe does the latter, which is why its
  reference state is 1 failure.

- **The memory plan is a SHADOW today — `PlanInput::features` is mostly unfilled, and that is
  not a bug.** `plan.cpp:111` sums `vision_tower_bytes` and `spec_decode_bytes` into
  `engine_persistent`, but production writes neither: `plan_shadow.cpp:29-31` fills only
  `ssm_state_bytes`, `n_swa_layers` and `swa_live_tokens`, and the only writers of
  `vision_tower_bytes` in the tree are `tests/test_memory_plan.cpp:172,279`. A field that only
  tests set looks exactly like the vacuous-pass pattern this ledger keeps recording, so it is
  worth stating plainly why it is not: `plan_shadow.cpp`'s own comment says the plan gets its
  real shape at **A7 step 6**, "where the plan runs BEFORE the upload and charges everything
  from a clean slate", and `AUDIT.md` B80 records what step 6 is waiting on (always-on pool
  bookkeeping). Until then the plan advises and does not allocate, so an unset feature field
  costs nothing at runtime. **The number that IS authoritative is the arena capacity at
  `engine.cpp:915`** — that is where a missing tenant actually under-reserves, and that is where
  the vision tower was added. Do not "fix" the plan fields in isolation and read it as closing a
  VRAM gap.
- **F-24** — `src/runtime/engine.h` god-header. **Its proposed fix is REFUTED by measurement,
  and a different lever was shipped instead (2026-08-04).**
  The audit proposes extracting the spec-decode member block into a `SpecDecodeState` struct,
  "the largest self-contained cluster". Largest by member count — **not where the churn is**.
  Of the 130 commits touching `engine.h` in six months, **3 (2 %) change spec/MTP lines only**;
  92 (71 %) do not touch spec at all. So the extraction removes ~2 % of the churn at best, and
  only if `Engine` held the struct by pointer — held by value, `engine.h` includes the new
  header and the same TUs rebuild for **zero** gain. Do not build it.
  **Churn is spread across every concern, which is the actual diagnosis:** request/sched 58
  commits, kv/cache 52, memory/vram 44, graph 40, spec/mtp 38, sampling 35, vision 11.
  `engine.h` changes because `Engine` is the coordination point for all of it, not because one
  extractable cluster is hot. **A full pimpl of the private state has a measured ceiling of
  42 %** (54 of 130 commits touch only the private section) — that is the honest number for
  anyone considering it, against an indirection on the decode path.
  **What was shipped attacks the other factor.** Cost is fan-in × churn, and fan-in was the
  cheap half: `src/api/imp_internal.h` had 17 includers and pulled `runtime/engine.h` in for all
  of them, while storing nothing but a `std::unique_ptr<imp::Engine>`. Forward-declaring
  `imp::Engine` there and moving `ImpContext_T`'s ctor/dtor out of line (the single point where
  `unique_ptr` needs the complete type) took `engine.h` from **33 to 23 rebuilt TUs, -30 %** —
  cost 4257 → 2967, better than the refuted extraction's 2 % and a fraction of the risk.
  **Method note, because a loose grep cost me a wrong answer here:** `rg 'engine\.|Engine::'`
  reported nine consumers needing the include, three of which were matches on
  `batching_engine.`. Deleting every candidate include and letting the compiler name the
  failures gave the true set — **two**. Measure fan-in with reverse BFS over the include graph;
  a forward walk with memoisation silently poisons its cache on include cycles and reported
  2 of 463 TUs.

## Per-area running logs

- **Memory subsystem** — root `AUDIT.md` (CONFIRMED / REFUTED / OPEN per finding, negative
  results included). Read it before sweeping `src/memory/`; design lives in
  `docs/MEMORY_ARCHITECTURE.md`.
- **Architecture / dispatch** — this file, plus
  [`AUDIT_ARCH_2026_07_29.md`](AUDIT_ARCH_2026_07_29.md) for the evidence behind each entry.
- **File size** — [`AUDIT_FILESIZE.md`](AUDIT_FILESIZE.md), per-file rationale.
