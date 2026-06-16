# STRUCTURE.md — target state proposal (Phase 2, READ-ONLY)

**Date:** 2026-06-17 · **Status:** proposal awaiting approval. Nothing implemented.
**Companion:** `AUDIT.md` (findings) → this file (target) → `DISPATCH.md` (ordered tasks + gates).

**Guiding principle.** The audit found the layout, layer DAG, and file cohesion **already healthy** (no
god-files-by-conflation, no circular deps, clean PIMPL C-API). So this is **mostly ratification**: write down
the dependency rule that already holds, add the missing tooling/config, and fix a handful of hygiene nits.
**Explicitly NOT proposed:** moving/splitting any large file, splitting `GraphExecutor`, adding multi-arch
paths, or changing the two-config (`RuntimeConfig` vs `ModelConfig::Overrides`) split — all settled / correct.

---

## 1. Directory layout (target = current, formalized)

```
imp/
├─ include/imp/          PUBLIC C ABI surface — opaque handles + POD structs only (imp.h, config.h, error.h, types.h)
├─ src/
│  ├─ core/              Foundational types: Tensor, Buffer, QType, allocator base, logging, threading, cuda_raii
│  ├─ memory/            Device/VRAM/pinned allocators (cudaMallocAsync pool), KV / SSM / GDN state, layer offload
│  ├─ quant/             Dequant + quant GEMM kernels: GGUF Q*/K, NVFP4, FP8 E4M3, MXFP4, GPTQ
│  ├─ compute/           Stateless CUDA kernels: gemm, attention (FA2/paged), rope, norm, sampling, moe, gdn, ssm
│  ├─ model/             Loaders (GGUF / SafeTensors / HF / SPM), tokenizer, chat template, weight map, ModelProfile
│  ├─ exec/              GraphExecutor + forward-pass orchestration, gemm kernel registry/dispatch, pre-dequant phases
│  ├─ runtime/           Engine, scheduler, batching, RuntimeConfig, CUDA graph, spec/ngram, vram budget, storage planner
│  ├─ lora/              LoRA adapter load + apply
│  ├─ vision/            Vision pipeline / encoder / loader (SigLIP, gemma4v)
│  └─ api/               C-ABI boundary (imp_api.cpp) — internal throw → ImpError translation
├─ tools/                imp-cli, imp-bench, imp-server (not part of libimp; link against it)
├─ tests/                GTest binaries (8 modules) + tests/bench/ microbench TUs + tests/api/ (Python)
├─ cmake/                CompilerFlags.cmake (+ proposed: imp-deps.cmake for the single dep-pin source)
├─ scripts/              verify.sh, gen_perf_baseline.sh, test_server.sh, hooks, (+ proposed: bench_gate.sh)
├─ docs/                 architecture.md (canonical), sm120.md, quantization.md, determinism.md, audit/, …
└─ .github/workflows/    ci.yml (Build + Mock API + dark GPU), auto-merge, release-docker, roofline (+ proposed: lint)
```

### Layer dependency rule (the contract)

Foundational → high. **A layer may `#include` only from layers at or below it:**

```
core  ◄─ memory ◄─ quant ◄─ compute ◄─ exec ◄─ runtime ◄─ api
        (core)     (core,    (core,     (core,   (core, memory,
                    compute*) quant,     memory,   model, exec,
                              memory)    quant,    compute)
                                         compute,
                                         model)
        model ◄─ (core, quant)      lora ◄─ (core, exec)
        vision ◄─ (core, model, compute, runtime)
```

- `*` **One sanctioned upward edge:** `quant → compute` (CUTLASS GEMM dispatch) — architectural necessity.
- **Documented exception (D1, to be narrowed):** `compute/`, `quant/`, `memory/` currently reach up into
  `runtime/pdl.h` and `runtime/process_diag.h` for **diagnostics/instrumentation only** (~18+ sites). Target:
  expose those hooks through a `core/diag.h` interface so the DAG has no upward edges, OR explicitly whitelist
  them in `.clang-tidy`/docs as "diagnostic, non-algorithmic". **Low priority — do not over-chase.**
- **Enforcement (proposed):** a lightweight check (script or `clang-tidy` `misc-*` + a custom grep in the lint
  job) that fails on a *new* upward `#include` outside the whitelist. Advisory first, blocking later.

---

## 2. CLAUDE.md outline (to be committed — DOC1 decision: tracked)

> The repo-root `CLAUDE.md` is the **project** instruction file (distinct from personal `~/.claude/CLAUDE.md`).
> It will be un-ignored and committed. Outline below = the sections it must contain (most already exist; this
> ratifies + fills gaps). Must stay **English**, **no version strings** (G1 rule).

1. **What imp is** — from-scratch C++20/CUDA engine, single chip `sm_120a` (RTX 5090 / GB202). One-paragraph.
2. **Build & test commands** — `make build` (Docker, CUDA 13.3) / cmake preset; `make test-unit|test-gpu|
   verify-fast|verify`; `make install-hooks`; the 3-stage test gate (pre-commit GPU · CI CPU · server).
3. **Single-arch rule** — `sm_120a` SASS + `compute_120f` PTX fallback only. **Never add multi-arch paths.**
   Ignore datacenter-Blackwell (`sm_100`, tcgen05/TMEM/wgmma/TMA-WS) kernel designs.
4. **Performance gate + single-session rule** — `tests/perf_baseline.json` (3% decode / 5% prefill) is canonical;
   refresh via `scripts/gen_perf_baseline.sh` and say so in the PR. **Only compare benchmark numbers within one
   session/run** (compiler/cuBLAS autotuning makes cross-session numbers unreliable).
5. **Measurement methodology** — **decode is the A/B signal**; warm the clocks (discarded warmup >1s) before
   timed reps; sample `nvidia-smi` clocks during bench; never cooldown-wait (water-cooled, no throttle);
   prefill has 2.6× restart variance → don't A/B on it. Headline metric: `tg128` decode tok/s.
6. **Allocator / error-handling conventions** — device memory via the `DeviceAllocator` pool (`cudaMallocAsync`),
   not scattered `cudaMalloc`; CUDA errors via `IMP_CUDA_CHECK*` (`src/core/logging.h`); streams/events via
   `core/cuda_raii.h`; internal errors **throw**, translated to `ImpError` only at `src/api/imp_api.cpp`.
7. **Conventions** — English-only in the repo; branch off `main`, `gh pr create --base main`, **never stack PRs**;
   Conventional Commits (`fix:`/`test:`/`docs:`/`release:`) + PR number; match surrounding style, no speculative
   abstraction; runtime config in `RuntimeConfig` (`src/runtime/config.h`), no ad-hoc env reads.
8. **"Do not touch" zones** — single-arch gencode block in `CMakeLists.txt`; the `Build` CI check name
   (branch-ruleset required); `gdn.cu` and the other cohesive large TUs; `GraphExecutor` forward coupling;
   the two-config split; dual dep-pins must move together (CMake + Dockerfile).
9. **After hot-path changes** — coherence check (no repetition loop / token-stuck) after touching forward pass,
   MoE routing, KV cache, GDN state, or CUDA-graph capture.
10. **Hardware reality** — no FP4 cuBLASLt on sm_120 → CUTLASS primary; FP8 prefill unavailable; GGUF→NVFP4
    decode cache at init; Docker build cache must **not** use `--mount=type=cache`.

---

## 3. AGENTS.md outline (new, tracked)

> Focused subagent roles. Each role = **scope · allowed tools · MUST · MAY NOT**. The "MAY NOT" lines are
> load-bearing. Top of file: the global rules (single-arch, perf-gate single-session, no PR stacking, English-only,
> verify-every-finding) that apply to **all** roles.

| Role | Scope | Allowed tools | MUST | MAY NOT |
|---|---|---|---|---|
| **auditor** | Whole repo, **read-only** | Read, Grep, Glob, sub-agents | Verify every finding against source before reporting; write only `AUDIT.md` | Edit any code/config; act on an unverified sweep result |
| **build-engineer** | `CMakeLists.txt`, `cmake/`, presets, Dockerfile, deps, CI | Edit (build files), Bash (configure/build) | Keep build green; bump both dep-pin sites together; clean-reconfigure on build-system change | Touch kernel/algorithm logic; add multi-arch paths; rename the `Build` CI check |
| **kernel-optimizer** | `src/compute/**`, `src/quant/**` only | Edit (those dirs), Bash (build, bench) | **Benchmark before/after in the same session**; coherence-check after hot-path edits; keep single-arch | Edit build system, API, or runtime orchestration; commit a perf change without an in-session A/B |
| **test-writer** | `tests/**` | Edit (tests), Bash (build, run tests) | Add an **independent** oracle with a justified tolerance; tests adapt to the engine | Change `src/` to make a test pass; assert exact-equal on known-nondeterministic paths (MoE NVFP4) |
| **benchmark-runner** | `scripts/bench_gate.sh`, `tools/imp-bench`, baselines | Bash (bench), Edit (baseline json + bench scripts) | Warm clocks, ≥3 trials, single-session; enforce 3%/5% gate; sample clocks during run | Compare across sessions/days as if equal; refresh the baseline silently (must be stated in PR) |
| **docs** | `*.md`, `docs/`, `imp.conf.example` | Edit (docs only) | Keep docs in sync with code; **no version strings** (CMake/lockfiles only); English | Edit any `.cpp/.cu/.h/.cmake`; invent perf numbers (cite `perf_baseline.json`) |

**Global guardrails (header of AGENTS.md):** GPU must be free before any GPU job (`docker ps -q | wc -l == 0`);
never `sudo` on the host; `build/` is root-owned (remove via throwaway container); secrets via env/secret files
only.

---

## 4. Config set to be added

| File | Status | Purpose / key contents |
|---|---|---|
| `.clang-format` | **exists** ✓ | Keep as-is (Google-based, tuned). |
| `.editorconfig` | **exists** ✓ | Keep as-is. |
| `.clang-tidy` | **new** | CUDA-aware host-TU static analysis. Enabled: `bugprone-*`, `performance-*`, `cppcoreguidelines-pro-type-*` (subset), `readability-*` (subset), `misc-*`. `HeaderFilterRegex: '(src\|include)/imp'`. **Scoped to host `.cpp/.h`** (clang-tidy can't parse `.cu` without full CUDA flags — documented in the file header). `WarningsAsErrors: ''` initially (advisory), tighten later. |
| `.pre-commit-config.yaml` | **new** | Framework hooks: `clang-format` (style=file), `trailing-whitespace` (md excluded), `end-of-file-fixer`, `check-added-large-files`, `check-merge-conflict`, + a `local` hook running `make format-check`. The existing **GPU** `pre-commit.hook` stays a separate native hook (the framework can't gate a Docker GPU run cleanly). |
| `CMakePresets.json` | **new** | `configurePresets`: `default` (Release, tests+tools on, Ninja, ccache launchers), `ci` (mirrors `ci.yml`: Release, BENCH=OFF, ccache), `debug` (Debug + `IMP_SANITIZERS=ON`), `relwithdebinfo` (profiling, `-lineinfo`). Single source of truth for configure args. Single-arch gencode stays in `CMakeLists.txt`. |
| `cmake/imp-deps.cmake` | **new (B5)** | One place defining the pinned dep versions (gtest/cutlass/httplib/json) as variables, `include()`d by `CMakeLists.txt`; the Dockerfile reads the same values via build-args. Removes the dual-pin drift. |
| `.github/workflows/ci.yml` (lint job) | **edit** | Add a CPU-only `lint` job: `format-check` + `clang-tidy` on host TUs (uses `compile_commands.json`). **Do not** rename the `Build` job. |
| `scripts/bench_gate.sh` | **new** | Extract the inline perf-gate bash from `ci.yml:185-216` into a reusable script (warmup, parse, compare to `perf_baseline.json`, 3%/5% decision). Called by both the GPU CI job and local `verify`. |
| `.gitattributes` | **new (G3)** | `* text=auto eol=lf`, `third_party/** linguist-vendored`, mark binaries (`*.png *.svg` `binary`). |
| `BENCHMARKING.md` | **new (BM2)** | One-page methodology contract: headline metric (`tg128`), warmup discard, ≥3 trials, single-session-only, clock sampling — extracted from `CLAUDE.md` so it's auditable by outsiders. |

---

## 5. What changes vs. what's ratified

**Ratified (documented, not changed):** directory layout, layer DAG, the single sanctioned `quant→compute`
edge, PIMPL C-API, large cohesive TUs, `GraphExecutor`, two-config split, single-arch gencode, the existing
`.clang-format`/`.editorconfig`/Makefile targets/3-stage gate.

**Changed (small, gated — see DISPATCH.md):** commit CLAUDE.md (un-ignore); add AGENTS.md; remove version
strings from tracked markdown; target-scope the warning flags (B1); add the config set above; decide F1
(`gemv_q4_0_q8_1`); `.gitignore` `imp.conf`.

**Deferred (optional, listed with gates in DISPATCH.md, not in the core sequence):** C1 (RAII migration of the
4 raw stream/event sites), H1 (`std::span` on host APIs), B3 (package export), B6 (CMake 3.31 bump), D1
(`core/diag.h` to remove the upward diag edges), C2 (debug-only launch-check macro).

---

*Phase 2 (STRUCTURE.md) complete. See `DISPATCH.md` for the ordered, gated task list. Awaiting plan approval
before any Phase-3 implementation.*
