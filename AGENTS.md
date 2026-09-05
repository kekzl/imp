<!--
layer: L3
audience: agents
verified: 2026-09-06
commit: b5de0dd7
-->

# AGENTS.md - subagent roles and guardrails for imp

Focused agent roles for working on imp; the cross-tool companion to
[`CLAUDE.md`](CLAUDE.md). **Every rule in `CLAUDE.md` applies to every role**: single
architecture, gated performance, GPU-free check before a GPU job, `make dev` to iterate and
`make build` to measure, no bare `make format`, branch off `main` and never stack, no
busy-polling, the file-size gate, no `sudo`. This file adds only what is role-specific:
scope, tools, MUST and MAY NOT.

Rules with no other home:

- **Performance comparisons are single-session only.** Compiler/cuBLAS autotuning makes
  cross-session numbers unreliable; compare results captured within one run. Decode `tg128`
  is the headline signal; the gate is 8 % decode / 8 % prefill / 10 % `own_peak_mb`.
- **No version strings in markdown/configs**: versions live in CMake and lockfiles only.
- **Verify every finding against the real source** before acting on it (fan-out sweeps
  over-flag). Secrets via env only.

## Roles

### auditor
- **Scope:** whole repo, **read-only assessment**.
- **Allowed tools:** read/search tools, read-only sub-agents.
- **MUST:** read `docs/audit/SETTLED.md` **before forming hypotheses** and generate against it (eight of the 2026-07-29 audit's thirteen hypotheses described duplication that earlier campaigns had already collapsed); verify each finding against source before reporting; write only a dated report under `docs/audit/`, or append to an existing running findings log where one owns the area (`AUDIT.md` for the memory subsystem, which records REFUTED results too); append new refutations to `SETTLED.md` with their anchors; rank by severity+effort.
- **MAY NOT:** edit any code or config; act on an unverified sweep result; report a finding that contradicts a `SETTLED.md` entry without first disproving that entry's anchor; propose multi-arch or speculative rewrites.

### build-engineer
- **Scope:** `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, `Dockerfile`, `Makefile`, dependency pins, `.github/workflows/`.
- **Allowed tools:** edit (build/CI files), bash (configure/build).
- **MUST:** keep the build green; clean-reconfigure after a build-system change; bump both dep-pin sites
  (CMake + Dockerfile) together; keep the single-arch gencode block intact.
- **MAY NOT:** touch kernel/algorithm logic; add multi-arch paths; rename the `Build` CI job (branch-ruleset
  required check); introduce `--mount=type=cache` in the Docker build; collapse the Dockerfile's
  `toolchain`/`builder` split (`make dev` compiles in the `toolchain` stage) or let the two build paths
  diverge on compiler flags: a `-march` difference between them would silently confound every A/B.

### kernel-optimizer
- **Scope:** `src/compute/**` and `src/quant/**` only.
- **Allowed tools:** edit (those dirs), bash (build, benchmark).
- **MUST:** run a **before/after benchmark in the same session** for any perf-affecting change (warm clocks,
  >=3 trials, decode `tg128`); coherence-check after hot-path edits; stay single-arch.
- **MAY NOT:** edit the build system, public API, or runtime orchestration; commit a perf change without an
  in-session A/B; insert temperature cooldown waits (water-cooled GPU, no throttle).

### test-writer
- **Scope:** `tests/**`.
- **Allowed tools:** edit (tests), bash (build, run tests).
- **MUST:** add an **independent** oracle with a justified, inline-documented tolerance; tests adapt to the
  engine, not the reverse.
- **MAY NOT:** change `src/` to make a test pass; assert exact-equality on known-nondeterministic paths
  (e.g. NVFP4 MoE atomic-scatter); commit a conflated/unsound golden.

### benchmark-runner
- **Scope:** `scripts/bench_gate.sh`, `scripts/gen_perf_baseline.sh`, the `scripts/verify.sh` gates,
  `tools/imp-bench/`, `tests/perf_baseline*.json`.
- **Allowed tools:** bash (benchmark), edit (baselines + bench scripts).
- **MUST:** warm the clocks, run >=3 trials single-session, enforce the 8%/8%/10% gate, sample
  `nvidia-smi` clocks during the run to rule out depressed host state.
- **MAY NOT:** compare across sessions/days as if equal; refresh a baseline silently (must be stated in the PR).

### docs
- **Scope:** `*.md`, `docs/`, `imp.conf.example`.
- **Allowed tools:** edit (docs only).
- **MUST:** keep docs in sync with code; cite perf numbers from `tests/perf_baseline.json` rather than
  re-typing them; English.
- **MAY NOT:** edit any `.cpp/.cu/.h/.cmake`; introduce version strings; invent benchmark numbers.
