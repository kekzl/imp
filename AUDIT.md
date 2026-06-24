# AUDIT.md — imp structural & hardening audit (Phase 1, READ-ONLY)

**Date:** 2026-06-17 · **Scope:** whole repo (build · layers · CUDA · C++ · tests · bench · tooling · CI · docs · git).
**Method:** direct read of every build/config file (CMake, CI, Makefile, clang-format, editorconfig,
gitignore) + 3 verification sub-agents (CUDA hygiene, C++ hygiene, layer/include discipline), each
finding re-checked against source per the codebase-audit rule "never act on a raw finding". Counts are
measured, not estimated. No file other than this one was modified.

> **Supersedes the prior `AUDIT.md`** (a test-coverage-only deliverable dated 2026-06-15; recoverable via
> git history). Its verified conclusions and its two open bugs (F1/F2) are folded into §5 (Testing). The
> full prior methodology also lives in `docs/TEST_AUDIT.md`.

**Codebase size (measured):** 105,547 lines across `src/` + `include/` (151 `.h`, 115 `.cu`, 50 `.cpp`,
6 `.cuh`). Layers under `src/`: core · memory · model · quant · compute · exec · runtime · lora · vision · api.

**Headline:** the repo is **healthy and production-grade** — no blockers, build green, ~574 tests. The real
gaps are in **CI enforcement** (no format/tidy gate; GPU correctness + perf gates are dark because there is
no GPU runner), **build-system polish** (global flags instead of target-scoped; no presets; no package
export), and a few **hygiene nits** (a stale hard-coded version string in tracked markdown; CLAUDE.md is
gitignored; no `.clang-tidy`). Severity legend: **blocker / high / med / low**; effort **S/M/L**.

---

## 1. Build system (CMake)

**Current state.** `CMakeLists.txt` (719 lines) is modern and largely target-based: `target_include_directories`
with `BUILD_INTERFACE`/`INSTALL_INTERFACE` generator expressions (`:316`), `target_link_libraries` with
PUBLIC/PRIVATE (`:326`), `target_compile_definitions(imp PRIVATE IMP_USE_CUTLASS=1)` (`:341`).
`CMAKE_EXPORT_COMPILE_COMMANDS ON` (`:8`). Separable compilation + device-symbol resolve on the main lib and
every test/bench target (`:345`, `:438`). Single-arch is correctly enforced via raw gencode
(`arch=compute_120a,code=sm_120a` + `compute_120f` PTX fallback) with `CMAKE_CUDA_ARCHITECTURES OFF` (`:31-39`)
— the documented CMake-<3.31 workaround for the `a`/`f` suffix. Deps via `FetchContent` with pinned tags
(gtest v1.17.0, cutlass v4.5.2, httplib v0.46.1, json v3.12.0). Clean option matrix (`IMP_BUILD_TESTS/TOOLS/
BENCH/SERVER`, `IMP_SANITIZERS`, `IMP_COVERAGE`, `IMP_DISABLE_120F_FALLBACK`). `install(TARGETS imp …)` +
header install present.

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **B1** | Warnings/flags set **globally** via `set(CMAKE_CXX_FLAGS … -Wall -Wextra -Wpedantic)` and `set(CMAKE_CUDA_FLAGS …)` in `cmake/CompilerFlags.cmake:4,13` | Best practice = target-scoped `target_compile_options(imp PRIVATE …)` / INTERFACE warnings. Global flags also leak `-Wall -Wpedantic` onto FetchContent deps (gtest/cutlass) → warning noise the team can't fix. | med | M |
| **B2** | **No `CMakePresets.json`** (confirmed absent) | Configure args are duplicated by hand in `Makefile`, `ci.yml:79`, and docs. A `default`/`ci`/`debug` preset set would make `cmake --preset ci` the single source of truth. | med | S |
| **B3** | Install rules exist but **no `install(EXPORT)` / package-config / `imp::imp` ALIAS** (`CMakeLists.txt:699-707`) | Downstream `find_package(imp)` is impossible; the static lib isn't consumable as a package. Low impact *if* imp is only ever an app, but the install target implies otherwise. | low | M |
| **B4** | Ninja not enforced (CI `cmake -B build` uses the default Makefiles generator, `ci.yml:79`); ccache wired in CI only | Ninja + ccache as the default (via a preset) speeds local incremental builds; today ccache helps CI but not the local Docker build. | low | S |
| **B5** | **Dual dependency pinning** — versions live in both CMake `FetchContent` *and* the Dockerfile deps-clone (documented in `CLAUDE.md`) | A bump must touch two places or silently drift. A single pinned-versions include or build-arg would dedupe. | low | M |
| **B6** | `cmake_minimum_required(VERSION 3.25)` keeps the raw-gencode workaround alive | CMake ≥3.31 parses `CMAKE_CUDA_ARCHITECTURES "120a"` natively, dropping the `set(CMAKE_CUDA_FLAGS …gencode)` hack. Must stay **single-arch** (no multi-arch added). | low | S |

---

## 2. Directory & layer structure

**Current state — strong.** Public headers in `include/imp/` (`imp.h`, `config.h`, `error.h`, `types.h`) form a
coherent C surface (see §4). Layers are cohesive; no `../../`-style relative includes cross layers (verified:
includes are layer-rooted `#include "core/…"`, `"exec/…"`); no `.cu`/`.cuh` is `#include`d as a fragile sibling
TU. Largest files are **single-domain**, not god-files (per prior audits and re-checked): `model/jinja.cpp`
(~2.6k), `model/tokenizer.cpp` (~2.5k), `model/weight_upload.cu`, `compute/gdn.cu`,
`exec/pre_dequant_phase3_nvfp4_decode.cu`, `compute/attention_fmha_sm120.cu`, `model/gguf_loader.cpp`,
`compute/sampling.cu` — each one concern. `exec/`’s `GraphExecutor` is intrinsically forward-pass-coupled (prior
audit settled — **do not** split into runner classes).

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **D1** | A foundational layer reaches **up** into `runtime/`: `compute/*` includes `runtime/pdl.h` / config (~18 sites; sub-agent count), plus a few `quant/→runtime`, `memory/→runtime` for `process_diag.h`/`pdl.h` | These are instrumentation/diagnostic hooks, not algorithmic coupling — acceptable but they blur the layer DAG. A thin `core/diag.h` interface would let compute/quant depend down instead of up. Do **not** over-chase (prior audits refuted bigger versions of this). | low | M |

*No circular dependencies, no API-layer leakage, no conflated god-files.* Verdict: **healthy.**

---

## 3. CUDA specifics

**Current state — mostly good.** Centralized error macros in `src/core/logging.h:67-99`
(`IMP_CUDA_CHECK_LOG/_BOOL/_VOID`) + a throwing `IMP_CUDA_CHECK` in `src/memory/device_allocator.cu:13-22`.
Allocator strategy is a real **memory pool**: `DeviceAllocator` uses `cudaMallocAsync`/`cudaFreeAsync` over a
`cudaMemPool` (`device_allocator.cu:29,73,102`); scattered direct `cudaMalloc` exists only for scoped one-shot
scratch (quant conversion, spec-decode buffers) — no per-iteration bypass found. RAII wrappers exist:
`src/core/cuda_raii.h` (`CudaStream`, `CudaEvent`, non-copyable/moveable, dtor-destroy). Launch configs use
named constants (`gemm.cu:42 kGemvThreads=256`, `nvfp4_quant.cu:108 kMicroBlockSize=16`) + helpers
(`gemv_blocks`). `__host__ __device__` used purposefully (CUTLASS interop, shared device helpers), no misuse.
`-lineinfo` in RelWithDebInfo, stripped in Release (`cmake/CompilerFlags.cmake:15,22`). PTX fallback documented.

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **C1** | RAII wrappers exist but are **not universally adopted** — `memory/layer_offload.cu`, `exec/expert_cache.cu`, `runtime/green_ctx.cu`, `runtime/engine_weight_upload.cpp` create streams/events raw and destroy them manually in dtors | Works today (each has a matching destroy), but is not exception-safe across multi-step init and is inconsistent with `cuda_raii.h`. Migrating these to `CudaStream`/`CudaEvent` removes the manual-cleanup class of bug. | low-med | M |
| **C2** | Kernel launches are **mostly not** followed by an explicit `cudaGetLastError()` check (385 `<<<>>>` sites; only a handful checked, e.g. `dequant_fp16.cu:62`) | Launch-config errors (smem/reg over-budget) surface late and opaquely. A debug-only `IMP_LAUNCH_CHECK()` macro after launches (no-op in Release) would localize them without a hot-path cost. | low | M |

*Allocator, error macros, launch constants, profiling flags, `__host__/__device__`, PTX policy: **good**.*

---

## 4. C++ hygiene

**Current state — strong.** Everything is in `namespace imp {` (verified across implementation files); no
global-namespace leakage. The public ABI is a **clean PIMPL C API**: `include/imp/*.h` expose only opaque
handles (`typedef struct ImpModel_T* ImpModel`) + POD structs and include **only** `<stdint.h>/<stddef.h>` and
each other — zero CUDA/cutlass/internal leakage (verified by grepping the four headers’ includes). Smart
pointers dominate (151 `unique_ptr`/`shared_ptr` sites); the ~5 raw `new`/`delete` are justified (C-API
ownership bridge, static CUTLASS singletons). No `using namespace` in any header. Move semantics explicit
(`Buffer(Buffer&&) noexcept`). `.cuh` template files are large by necessity (device inlining).

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **H1** | `std::span` barely used (8 sites) vs the dominant host-side `T* ptr, size_t n` signature (e.g. `core/buffer.h:38`, `runtime/batch.h add_prefill_sequence`) | ptr+len is fine at kernel boundaries, but host-side APIs lose C++20 bounds/type safety. Incremental `std::span` adoption on host signatures is a safety win, not a rewrite. | low | M |
| **H2** | `noexcept` is sparse (~65 sites) on obviously-non-throwing getters | Cosmetic; marking trivial accessors `noexcept` documents the contract and can help the optimizer. | low | S |

*Namespacing, ABI/PIMPL, RAII-vs-raw, header discipline, casts: **good**.*

---

## 5. Testing

**Current state — large and, in places, exemplary.** ~574 GTest cases in 8 per-module binaries
(`CMakeLists.txt:449-609`); `ctest` registered via three label aggregates `unit`/`gpu`/`perf` (not
`gtest_discover_tests`, which double-ran — R5/#580), with `guard_e2e_lane_split` asserting the CPU/GPU filter
can’t silently drift. Best-in-class oracles: fp64 GGUF dequant re-derivation, fp64 attention crosspath +
numpy golden, GDN delta-rule CPU ref, GGUF fault-injection (19 cases), determinism/greedy locks. The prior
test-coverage audit (now §-folded) was **largely shipped** — Q4_0/Q5_K/FP8 oracles, mxfp4-attention ref,
tool-call + Bearer-auth unit tests, the 3-stage gate.

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **T1** | **CI runs only the CPU `unit` lane + the Python mock-API suite** — every GPU oracle and the real `handlers.cpp` (~4600 LOC) are local-only, because there is no GPU runner (`ci.yml:153-161`) | The strong correctness net protects against *local* regressions only; a GPU-path regression can reach `main` if the author skips the local gate. Structural (shared root cause with CI3/BM1). Fixable only with a self-hosted RTX 5090 runner. | high | L |
| **T2** | **Open bug F1** — `gemv_q4_0_q8_1` consumes Q4_0 nibbles *interleaved* (`gemv_dp4a_traits.cuh:251`) while ggml + imp’s own `dequant_q4_0_kernel` are *split* (`dequant_gpu.cu:299`); a fp64 oracle showed ~6× error. **Phase-3 call-graph trace CORRECTS the original "latent" framing:** it is NOT dead and NOT FP16-only — `gguf_q4_0_kernel` → `run_gguf_smallm` → `dispatch_dp4a_gemv` (`gemm_kernel_gguf.cu:129`, `executor_kernels.cu:67`) feeds it native split-layout Q4_0 weights, so it IS the live Q4_0 dense-decode backend (+ MoE batch path). It has simply never been exercised because **no Q4_0 model is in the local suite** — reachable-but-untested. **RESOLVED:** two bugs — interleaved→split nibble layout AND a mis-scaled passed `q8_sum` (now summed internally like every other type). Verified by the now-asserted fp64 oracle `GgufRef.Q4_0_GemvDp4a` (max_rel 6.27→1.07e-2, in band) + full `test-quant` 187 green. | ✅ fixed | — |
| **T3** | **Open bug F2** — Qwen3.5-4B GGUF tokenizer diverged from an HF golden (13/20 byte-exact, contractions/whitespace) but is **unconfirmed/confounded** (pretokenizer vs model-version vs test-parser) | Needs a clean repro (proper JSON golden parser + confirmed-identical HF/GGUF pair) before filing. The entire prior cross-engine PPL gap (#657) was exactly this blind spot. | med | M |
| **T4** | Tokenizer HF-parity + byte-exact chat-template goldens remain **env-gated / not committed** for most families (`test_tokenizer_compat.cpp` skips unless `IMP_TEST_MODEL` set, ≥80% bar) | Committing per-family goldens + a generator closes the highest-blast-radius dark spot. Infra-bound (needs a shipped tokenizer or HF in CI). | med | M |

*Detail: prior `AUDIT.md` (git history) + `docs/TEST_AUDIT.md` + `tests/README.md`.*

---

## 6. Benchmarks

**Current state — strong and reproducible.** `imp-bench` + a canonical regression gate
(`tests/perf_baseline.json`, 3% decode / 5% prefill), refreshed by `scripts/gen_perf_baseline.sh` under a
cold-median methodology (5 trials, median). Roofline pipeline (`tools/roofline/`, ncu+nsys) with pin/regress.
Methodology is documented in depth in `CLAUDE.md` (clock warmup, idle-downclock artifact, cuBLAS restart
variance, host-day ±8-15% drift, sample clocks during bench). `make check-gpu` refuses to bench on a busy GPU.

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **BM1** | The perf-regression gate runs **only in the GPU `test` CI job**, which is gated on a non-existent self-hosted runner (`ci.yml:161,185`) → in practice the gate is local-only (pre-push `verify-fast`) | A decode regression can reach `main` unflagged unless the author runs the local gate — the exact failure mode the gate’s own comment cites (tg128 284→146). Same root cause as T1/CI3. | med (high once a runner exists) | L |
| **BM2** | Warmup/iters/headline-metric policy is spread across `CLAUDE.md` + script comments, no single `BENCHMARKING.md` contract | A one-page methodology doc (which metric is headline, warmup discards, single-session-only comparison) makes the gate auditable by outsiders. | low | S |

---

## 7. Tooling

**Current state.** `.clang-format` (Google-based, tuned to the codebase, run via a throwaway `silkeh/clang:18`
container — clean-host policy) + `make format`/`format-check`. `.editorconfig` covers C/C++/CUDA/CMake/MD/
Python. Hand-rolled git hooks (`scripts/pre-commit.hook` = Stage-1 GPU suite, `pre-push.hook` = verify-fast)
installed by `make install-hooks`. `make sanitize` (compute-sanitizer memcheck) target exists. Nsight entry
points documented (`docs/nsys_profiling.md`, roofline ncu wrappers).

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **TL1** | **No `.clang-tidy`** (confirmed absent) | No static-analysis config; a CUDA-aware tidy (bugprone-*, performance-*, cppcoreguidelines-* subset, with `HeaderFilterRegex` scoped to `src/`+`include/`) is the single biggest tooling win. | med | M |
| **TL2** | **No `.pre-commit-config.yaml`** — hooks are hand-rolled shell instead of the `pre-commit` framework | Functional today, but the framework gives versioned, shareable hooks (clang-format, trailing-whitespace, EOF, large-file guard) that run identically for every contributor. | low | S |
| **TL3** | `compute-sanitizer` can’t run on this WSL2 host (WDDM, no debugger interface — documented in `Makefile:202`) | memcheck/racecheck/synccheck/initcheck can only live on a native-Linux GPU runner. Flagged, not fixable here. | low | — |
| **TL4** | No IWYU / cppcheck (optional) | Include-what-you-use would catch transitive-include reliance; cppcheck adds a second analyzer. Nice-to-have. | low | M |

---

## 8. CI/CD

**Current state.** `.github/workflows/ci.yml`: `Build` job on `nvidia/cuda:13.3.0-devel` with ccache (8G,
content check) + build-dir cache, asserts nvcc==13.3, configures Release, builds, **runs `ctest -L unit`**,
uploads artifacts. Separate `mock-api` job (Python schema/SSE/error contract vs `mock_server.py`). A `test`
job (GPU ctest + perf gate) is present but `if: vars.HAS_GPU_RUNNER == 'true'`. `auto-merge.yml`,
`release-docker.yml`, `roofline.yml` round it out. The required branch-ruleset check is the literal name
`Build` (do not rename).

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **CI1** | **No format-check gate** in CI (`make format-check` exists but no workflow runs it — verified) | Style can drift into `main`; the cheapest possible CI gate is missing. Add a tiny ubuntu job running `format-check`. | med | S |
| **CI2** | **No clang-tidy gate** (depends on TL1) | Static-analysis regressions land unflagged. | med | M |
| **CI3** | The GPU correctness job + perf gate are dark (no self-hosted runner) — `if: vars.HAS_GPU_RUNNER` is never true | CI verifies *compilation* + the CPU lane only; the real correctness/perf gates never run in CI. Root cause shared with T1/BM1; honest + documented, but it means "green CI" ≠ "GPU-correct". | high | L |

*Positives: pinned toolchain, ccache, unit lane runs, mock-API contract, artifact upload, protected check name.*

---

## 9. Docs & agent files

**Current state — rich.** `docs/` has `architecture.md` (canonical), `sm120.md`, `quantization.md`,
`determinism.md`, `nsys_profiling.md`, `supported-models.md`, `MISSION_JOURNAL.md`, plus `docs/audit/`.
`CONTRIBUTING.md`, `README.md`, `CHANGELOG.md`, `GOAL.md` present. A local `CLAUDE.md` (rich project
instructions) drives the agent workflow.

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **DOC1** | **Root `CLAUDE.md` is gitignored** (`.gitignore:99 CLAUDE.md`, verified `git check-ignore`) → it is **not in the repo** | The project’s primary agent-instruction file can’t be a committed/shared deliverable while that ignore rule stands. Phase-2 asks for a "populated CLAUDE.md" — must first decide: commit it (drop the ignore) or keep it local and make **AGENTS.md** the tracked equivalent. | med | S |
| **DOC2** | **No `AGENTS.md`** (the cross-tool standard many agents read; not blocked by `.gitignore`) | Phase-2 deliverable. A tracked AGENTS.md is the natural home for committed agent-role guardrails. | low-med | S |
| **DOC3** | `README.md`/`BENCHMARKS.md` carry hardcoded perf + version claims that drift (see G1) | Docs accuracy: tie benchmark headline numbers to `tests/perf_baseline.json` rather than re-typing them. | low | M |

---

## 10. Git hygiene

**Current state — good.** Comprehensive `.gitignore` (build, CUDA artifacts, models/weights, profiling
output, secrets `.env`/`*.key`/`*.pem`, agent state). Conventional Commits with PR refs throughout
(`fix:`, `test(server):`, `docs(readme):`, `release:`). No build artifacts tracked (`build/` ignored). Secrets
ignored.

| # | Finding | Gap vs best practice | Sev | Effort |
|---|---|---|---|---|
| **G1** | **Version strings in tracked markdown** violate the "versions live in CMake/lockfiles only" rule: `BENCHMARKS.md:6` hardcodes `v0.11.0` (**stale** — CMake is `0.11.2`), and `.claude/skills/shipping-prs/SKILL.md:71` hardcodes `v0.11.2` (tracked, `.gitignore` un-ignores `.claude/skills/`) | A re-typed version drifts on every release. CHANGELOG version headers are inherent and fine; free-text "current: vX.Y.Z" is not. | med | S |
| **G2** | `imp.conf` sits **untracked but not gitignored** (`?? imp.conf`) — only `imp.conf.example` is the tracked template | Easy to commit a local runtime config by accident. Add `imp.conf` to `.gitignore` (keep `.example`). | low | S |
| **G3** | **No `.gitattributes`** (no LFS, no enforced `eol`/linguist) | Committed binaries (`docs/architecture.png`, fixtures) are small so LFS isn’t urgent, but a `* text=auto eol=lf` + `linguist-vendored` for `third_party/` would harden the repo. | low | S |

---

## Prioritized findings (sorted by severity, then effort)

| Rank | ID | Area | Finding | Sev | Effort |
|---|---|---|---|---|---|
| 1 | **CI3** | CI/CD | GPU correctness + perf jobs dark (no self-hosted runner) — green CI ≠ GPU-correct | high | L |
| 2 | **T1** | Testing | GPU oracles + real `handlers.cpp` run local-only (same root cause as CI3) | high | L |
| 3 | **CI1** | CI/CD | No `format-check` gate in CI (cheapest missing gate) | med | S |
| 4 | **B2** | Build | No `CMakePresets.json`; configure args duplicated across Makefile/CI/docs | med | S |
| 5 | **G1** | Git | Stale/hardcoded version strings in tracked markdown (`BENCHMARKS.md`, skill) | med | S |
| 6 | **DOC1** | Docs | Root `CLAUDE.md` gitignored → can’t be a committed deliverable | med | S |
| 7 | **T2** | Testing | F1 `gemv_q4_0_q8_1` nibble-layout bug (quarantined) — fix or delete | med | S |
| 8 | **B1** | Build | Global `CMAKE_CXX/CUDA_FLAGS` instead of target-scoped INTERFACE warnings | med | M |
| 9 | **TL1** | Tooling | No `.clang-tidy` (CUDA-aware static analysis) | med | M |
| 10 | **CI2** | CI/CD | No clang-tidy gate (depends on TL1) | med | M |
| 11 | **T3** | Testing | F2 Qwen3.5 tokenizer divergence (unconfirmed/confounded) | med | M |
| 12 | **T4** | Testing | Tokenizer/chat-template goldens env-gated, not committed | med | M |
| 13 | **BM1** | Bench | Perf-regression gate effectively local-only (root cause = CI3) | med | L |
| 14 | **DOC2** | Docs | No `AGENTS.md` (Phase-2 deliverable) | low-med | S |
| 15 | **C1** | CUDA | RAII stream/event wrappers exist but not universally adopted | low-med | M |
| 16 | **B3** | Build | No package export / `find_package(imp)` config | low | M |
| 17 | **TL2** | Tooling | No `pre-commit` framework config (hand-rolled hooks instead) | low | S |
| 18 | **B6** | Build | CMake 3.25 min keeps raw-gencode workaround (3.31 drops it) | low | S |
| 19 | **B4** | Build | Ninja not enforced; ccache CI-only | low | S |
| 20 | **H2** | C++ | `noexcept` sparse on trivial getters | low | S |
| 21 | **BM2** | Bench | No single `BENCHMARKING.md` methodology contract | low | S |
| 22 | **G2** | Git | `imp.conf` untracked but not gitignored | low | S |
| 23 | **G3** | Git | No `.gitattributes` (eol/LFS/linguist) | low | S |
| 24 | **DOC3** | Docs | README/BENCHMARKS perf numbers drift from baseline json | low | M |
| 25 | **C2** | CUDA | Kernel launches mostly lack a post-launch error check (debug-only) | low | M |
| 26 | **B5** | Build | Dual dependency pinning (CMake + Dockerfile) | low | M |
| 27 | **D1** | Layers | `compute/quant/memory` reach up into `runtime/` for diag/instrumentation | low | M |
| 28 | **H1** | C++ | `std::span` underused vs host-side ptr+len | low | M |
| 29 | **TL4** | Tooling | No IWYU/cppcheck (optional) | low | M |
| 30 | **TL3** | Tooling | compute-sanitizer can’t run on WSL2 (infra, not fixable here) | low | — |

**No blockers.** Build is green, ~574 tests pass, architecture is sound. The high-value/low-effort cluster for
Phase 3 is **CI1, B2, G1, DOC1, T2** (all med/S) plus the structural **CI3/T1/BM1** trio that only a GPU
runner resolves. The single-arch target, the documented hardware constraints, and the deliberate two-config
split are **correct as-is** and must be preserved.

---

*Phase 1 complete. No implementation performed. Awaiting confirmation before Phase 2 (STRUCTURE.md).*

<!-- ===================================================================== -->

# AUDIT pass 2 — soundness & hardening (2026-06-24)

**Scope:** adversarial soundness audit over dimensions A–L (correctness/UB · fault
isolation · memory/VRAM · concurrency · **CUDA-graph↔allocator coupling** ·
ownership/lifetime · API fidelity · determinism · observability · dead-code ·
build · test-integrity). The pass-1 audit (above) covered build/CI/tooling/docs
hygiene; this pass goes after the soundness dimensions it underweighted.
**Method:** 4 parallel audit sub-agents (E+C / A+D / B+F / G+H+J), every finding
re-verified against source by call-graph trace before any edit (codebase-audit
hard rule). **compute-sanitizer cannot run on this WSL2/WDDM host** (TL3) — UB/race
claims are reasoned statically. Branch `audit/soundness-hardening-2026-06-24`.

## Phase-0 baseline — snapshot `8f2cc9c4` (immutable)

Build GREEN · CPU unit 37/37 · GPU suite exit 0 (model-E2E SKIPPED: `models/`
symlinks don't resolve under the `$(PWD)/models` mount; synthetic GPU tests pass).
Bench (Qwen3-Coder-30B-A3B NVFP4, 7 reps × 2 cold restarts) in `PERF_LOG.md`:
decode tg256 = **341.6 / 322.0 / 266.4 tok/s** @ pp512/2048/4096 (gate signal,
rock-stable across restarts); prefill pp medians ~20.8k / 48.5k / 42.4k tok/s.

## Findings ledger (§5 format)

### F-A1  evict_lru frees an in-flight decode-batch sequence → KV use-after-free
- dimension: C/F · severity: **high**
- evidence: `kv_cache_manager.cpp:370-384` (victim excludes only *pinned*, not the
  in-flight batch); `free_sequence` removes the seq from `lru_order_`
  (`:332-343`) so **every** live LRU entry is an active sequence; decode batch is
  `touch`ed only at step-end (`engine_scheduler.cpp:1496`), so a batch member is a
  valid victim when `append_block` fails mid-loop (`engine_scheduler.cpp:951`).
  `free_sequence(victim)` returns its blocks to the pool, the next `append_block`
  re-hands them, then `step_decode_forward(valid_decode)` runs the victim on
  freed/reused blocks.
- root cause: eviction victim selection has no "in current batch" exclusion; LRU
  `touch` happens at step-end, not admission.
- blast radius: multi-sequence (batch>1) decode under full-KV pressure → silent KV
  corruption / UAF. Single-seq already fails safe (cancel); multi-seq did not.
- fix: **applied (reject-newest, supersedes the initial batch-protect commit and
  also closes F-A1b).** `allocate_block_with_eviction` already reclaims *cached*
  (finished) blocks safely; the engine's last-resort `evict_lru` of a *live*
  sequence is the only unsafe step and has no recompute path. Removed it at ALL
  three engine sites (decode, prefill ×2, spec) → on true KV exhaustion the
  already-present cancel/rollback fallback runs (reject-newest: cancel the
  requesting sequence, leave in-flight ones intact). `evict_lru` kept as a manager
  primitive with a "do not re-wire into the engine" warning.
- validation: new unit `KVCacheManagerTest.AllocationNeverEvictsLiveSequenceUnderPressure`
  (pool full of live seqs, 0 reclaimable → append/allocate FAIL, no table stripped).
  Full suite + bench green, decode gate held; 3× deterministic A/B coherent.

### F-A3  poisoned CUDA context is cleared & ignored — engine serves garbage, no detect
- dimension: B · severity: **high**
- evidence: `executor_forward.cu:207-212` clears the sticky error each forward with
  only a WARN and proceeds; worker catch `batching_engine.cpp:165-201` cancels
  requests + invalidates graphs but never probes device health; no `cudaDeviceReset`
  / poisoned-context flag anywhere.
- root cause: faults are detected only via thrown C++ exceptions; a kernel illegal
  access does NOT throw — it sets a sticky error that forward() clears, so the
  process silently serves garbage on a dead context instead of failing.
- blast radius: one request's IMA corrupts the process-wide context → all later
  requests garbage until manual restart, with no signal.
- fix: **applied (conservative)** — worker catch now `cudaDeviceSynchronize()`s and
  classifies the sticky error; only genuinely context-poisoning classes (illegal/
  misaligned address, launch failure/timeout, HW-stack, ECC, …) set `stop_requested_`
  → worker fail-fasts with a loud "context poisoned, restart required" log. A plain
  host-side throw leaves the device clean and recovers exactly as before.
- validation: confined to the (already abnormal) exception path; normal path
  unaffected (full suite + bench green). Poison path is reasoned-correct, not
  IMA-injected (no fault-injection harness on this host). The deeper /health-gate
  + graceful drain is PARKED below.

### F-A4 / F-A7  Anthropic /v1/messages: no x-api-key auth; OpenAI-shaped 401
- dimension: G · severity: **high** (auth) / med (error shape)
- evidence: `main.cpp:164-172` only checked `Authorization: Bearer`; `x-api-key`
  (the canonical Anthropic SDK header) was absent → real Anthropic clients 401 on a
  secured server. 401 body was OpenAI-shaped on all routes.
- fix: **applied** — `api_key_matches()` (`utils.cpp`, constant-time) accepts both
  Bearer and `x-api-key`; `/v1/messages` now returns the Anthropic error envelope
  `{"type":"error","error":{"type":"authentication_error",…}}`.
- validation: new `ApiKeyAuth.*` unit tests (both headers, neither, empty-key guard).

### F-A8  No SSE `ping` on the Anthropic stream → long-prefill idle-timeout risk
- dimension: G · severity: med
- evidence: `run_anthropic_stream_` (`handlers.cpp:3811+`) never emits `event: ping`;
  a long prefill (large prompt, cold) can trip proxy/client idle timeouts.
- fix: **applied** — periodic `ping` during sustained idle in the pop loop (15 s;
  never fires while tokens flow). Disconnect on a failed ping emit cancels cleanly.

### F-A6  force_cublas_decode stack overrun `int32_t h_block_table[1024]` @64K ctx
- dimension: A · severity: med
- evidence: `executor_attention.cu:1101-1103` — fixed 1024-int stack array filled
  with `n_blocks=(ctx_len+15)/16`; at the branch's 64K context cap n_blocks=4096 →
  4× stack smash via `cudaMemcpy`. Debug arm (`attention.force_cublas_decode`,
  default-off) but config-triggerable + silent.
- fix: **applied** — heap `std::vector<int32_t>` sized to n_blocks.

### F-A13  engine_weight_upload raw stream/event leak on throw
- dimension: F · severity: low
- evidence: `engine_weight_upload.cpp:167-185` — raw `cudaStream_t`/`cudaEvent_t`
  held across `upload_weights_gpu()` (which can throw) leak on the unwind.
- fix: **applied** — `CudaStream`/`CudaEvent` RAII (`core/cuda_raii.h`).

### F-A15 / F-A16  stale comments
- dimension: J · severity: low
- `use_nvfp4_decode` "auto (sm_120→mode2, sm_90→mode1)" implied a compute-cap
  branch that does not exist (resolver picks by quant/MoE/GDN, `engine_init_resolver.cpp:287-332`).
  `anthropic.cpp:415` claimed "we reject n>1 upstream" — `n` is never parsed and
  Anthropic Messages has no `n`. **fix: applied** (comments corrected).

## Parked (verified real, not safely landable autonomously — migration plans)

### F-A2  Non-streaming unbounded conditional-graph burst ignores cancel/disconnect/timeout
- dimension: D/B · severity: **high** · fix: **parked**
- evidence: `engine_scheduler.cpp:1545-1560` + `cuda_graph.cu:1010-1023` — a lone
  non-streaming request with spec off launches the unbounded autonomous graph loop
  bounded only device-side by `max_tokens`; `is_cancelled()`/timeout are polled only
  *between* `step()`s, so a disconnect burns up to `max_tokens` (8192) dead tokens.
- migration: bound the device loop to a chunk and re-poll cancel/timeout per chunk —
  the spec-ngram `miss_burst` path is the existing template. Touches the
  conditional-graph loop (documented #683/#692 off-by-one history) → needs the
  multi-token-verify GPU coherence battery (byte-diff graphs-on vs `--no-cuda-graphs`)
  before it can ship. Streaming requests are already exempt.

### F-A1b  Prefill/spec eviction strips a live non-batch sequence → delayed zombie
- dimension: C/F · severity: **high** · fix: **applied (folded into the F-A1
  reject-newest fix above)**
- evidence: `engine_scheduler.cpp:401,442`, `engine_spec_ngram.cpp:238` — same
  `evict_lru` mechanism; `lru_order_` holds only live sequences, so a successful
  eviction stripped an active sequence's KV → blockless decode next step.
- resolution: all three sites converted to reject-newest (decode cancels the
  requester; prefill cancels the new request; spec rolls back to normal decode).
  The unsafe preemption path is deleted; the safe cached-block reclamation inside
  `allocate_block_with_eviction` is untouched.

### F-A9  NVFP4 grouped-MoE GEMM never consults the deterministic flag — REFUTED
- dimension: H · severity: med · fix: **refuted (no change)**
- evidence: `gemm_cutlass_grouped_3x.cu`, `gemm_grouped_nvfp4_smallM.cu` have 0
  `deterministic` references (gate only in `gemm.cu` + `moe_routing.cu`).
- experiment: 3 fresh-process greedy (temp=0, `IMP_DETERMINISTIC=1`) generations of
  Qwen3-Coder-30B NVFP4 → **byte-identical** output (A==B==C). The grouped GEMM is
  deterministic by construction (compile-time-fixed schedule, no atomic reduction),
  so the missing flag-check is harmless and `determinism.md` is NOT overstated. The
  doc was left unchanged. (Over-flag guard: a "0 references" smell did not survive
  the A/B — the value of the codebase-audit verify rule.)

### Other low / hardening (verified, parked with notes)
- **F-A5** vision request `stop()/start()` cancels all concurrent in-flight
  (`handlers.cpp:1004-1018`) — **parked, NOT a safe swap.** On inspection the vision
  path `stop()`s to get *exclusive* access while it mutates GLOBAL engine image state
  (`clear_image` + `set_image_from_memory`). The embeddings `pause()/resume()` works
  there because embeddings don't mutate shared image state; for vision, resuming a
  concurrent *text* request could make it process with the stale vision image. The
  real fix is per-request image binding (so vision needn't serialize the whole
  engine), not a stop→pause swap. (The agent's "mirror embeddings" was an over-flag.)
- **F-A10** boundary re-prefill re-quantizes a *shared* KV block with no COW
  (NVFP4/INT KV only; FP16 safe by idempotency) — `scheduler.cpp:80-83`.
- **F-A11** `size_t→int` truncation in FP8/NVFP4 migrate APIs
  (`pre_dequant_phase3_nvfp4_decode.cu:639`, `fp8_quant.cu:277`) — **parked**: genuinely
  unreachable (largest tensor ~778M ≪ 2³¹), and a correct widen touches the live FP8
  path's kernel grid math + linear indexing across 3 files — poor risk/reward autonomously.
- **F-A12** KV-write kernels lack an in-kernel `block_id>=0` guard — **parked**: the
  shared `kv_resolve_slot` chokepoint is also used by read paths and must not reject
  StreamingLLM's legitimate `-1` sentinels; LOW + unreachable (host caps) doesn't justify
  the per-kernel guard risk.
- **F-A14** copyable RAII-owning handle types (LayerOffload/ExpertLRUCache/GreenCtx)
  — **FIXED**: deleted copy ops → move-only; clean build confirms no copy/move sites.
- **F-A17** swallowed `cudaStreamSynchronize` return at the burst boundary
  (`cuda_graph.cu:1014`) — diagnostic hygiene.

## Verified SOUND (negatives — do not re-chase)
KV refcount / COW-teardown / cancellation-free correct; KV size math is `size_t`
end-to-end; packed sub-byte KV `/2` handled; DeviceAllocator pool sound;
conditional-graph allocator coupling stable (alloc-once + re-upload), #683/#692
off-by-one fixed; Model const-after-load + lock-free shareable; SequenceState
borrows KV correctly; C-ABI boundary catches all; HTTP exception_handler envelope;
streaming is REAL (incremental TTFT); constraint mask applied in the sampler;
logprobs descending; Anthropic SSE event ordering spec-correct; constant-time auth;
**no dead multi-arch scaffolding in the shipped binary** (sm_90/100/wgmma refs are
live `>=1200` guards or accurate comments); ffn-sparsity / mtp / int4-attn all live.
