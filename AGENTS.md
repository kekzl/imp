<!--
layer: L3
audience: agents
verified: 2026-08-13
commit: 81ffa573
-->

# AGENTS.md — subagent roles & guardrails for imp

This file defines focused agent roles for working on imp. It is the cross-tool companion to
[`CLAUDE.md`](CLAUDE.md) (which holds the full build/test/benchmark conventions). Every role inherits the
global rules below; each role then narrows scope and lists explicit **MAY NOT** boundaries.

## Global rules (apply to every role)

- **English only in the repo.** All commits, PRs, code comments, docs and `.md` files are English.
- **Single architecture.** Target is exactly `sm_120a` (RTX 5090 / GB202) with a `compute_120f` PTX fallback.
  Never add speculative multi-arch paths or datacenter-Blackwell (`sm_100`, tcgen05/TMEM/wgmma) designs.
  TMA warp-specialised block-scaled GEMM is NOT in that list: it ships on sm_120a (#1543).
- **Performance is gated, single-session only.** `tests/perf_baseline.json` is canonical (8% decode / 8%
  prefill, plus 10% peak VRAM over the pinned `metrics.memory_mb.own_peak_mb`). Compiler/cuBLAS autotuning
  makes cross-session numbers unreliable — **only compare benchmark
  results captured within one run.** Decode `tg128` is the headline signal; refresh the baseline only via
  `scripts/gen_perf_baseline.sh`, and say so in the PR.
- **GPU must be free before any GPU job:** `docker ps -q | wc -l` MUST be `0` and `nvidia-smi` must show no
  active compute process. A busy GPU corrupts numbers and can OOM.
- **Iterate with `make dev`, gate with `make build`.** `make dev` is an incremental
  compile against a persistent `build-dev/` (seconds); `make build` recompiles the whole
  tree in a fresh image regardless of how little changed. Use `make dev` /
  `make dev-test` (= CI's `ctest -L unit`) for the edit-compile loop. **Benchmarks, the
  perf gate and anything pushed must be built by `make build`** — an incremental tree is
  where a stale object hides, and this repo re-pins baselines off measured numbers.
- **Never run bare `make format`.** The repo is not uniformly formatted; formatting whole
  files rewrites lines you did not touch, while CI checks only changed lines. Format files
  you *created*; hand-fix only your own added lines in files you edited.
- **Green gate before commit.** No commit on a failing build/test/gate. Branch off `main`,
  `gh pr create --base main`, never stack PRs. Conventional Commits + PR number.
- **Don't busy-poll a long job.** Builds, CI runs and merges take minutes; start them in the
  background and wait on one condition (a monitor with an until-loop, or a command that
  exits when the condition holds). Repeated status checks burn turns and buy nothing.
- **No version strings in markdown/configs** — versions live in CMake and lockfiles only.
- **Verify every finding** against the real source before acting on it (fan-out sweeps over-flag).
- Never `sudo` on the host; `build/` is root-owned (remove via a throwaway container); secrets via env only.

## File Layout & Size

The metric that matters is **recompile blast radius**, not line count. Each `.cu` is one
translation unit — editing one kernel in a 1.5k-LOC `.cu` re-`ptxas`es the whole TU (no
intra-file parallelism), and a fat header re-triggers every includer. Optimize files for
compile-time isolation:

- **One logical unit per file** (one kernel concept / one module). A `.cu` bundling several
  unrelated kernels is a split candidate.
- **Keep kernel definition, host launch-wrapper, and explicit template instantiations
  separable.** Push explicit instantiations into their own `.cu` when recompiles bite.
- **Thresholds are a proxy/smell, not the goal.** Gate: `tools/check_filesize.py` (config
  `tools/filesize_thresholds.toml`), on *code* LOC per category — kernel `.cu`
  warn>500/hard>600, normal TU warn>600/hard>800, header warn>500/hard>700. CI job
  `File size` = advisory warn step + blocking hard step.
- **Legitimately monolithic files belong in `[allow]` with a reason** (empty reason =
  gate failure). Don't split for splitting's sake. Baseline + per-file rationale:
  `docs/audit/AUDIT_FILESIZE.md`. This matches the "File Layout & Size" section in CLAUDE.md.

## Roles

### auditor
- **Scope:** whole repo, **read-only assessment**.
- **Allowed tools:** read/search tools, read-only sub-agents.
- **MUST:** read `docs/audit/SETTLED.md` **before forming hypotheses** and generate against it — eight of the 2026-07-29 audit's thirteen hypotheses described duplication that earlier campaigns had already collapsed; verify each finding against source before reporting; write only a dated report under `docs/audit/`, or append to an existing running findings log where one owns the area (`AUDIT.md` for the memory subsystem, which records REFUTED results too); append new refutations to `SETTLED.md` with their anchors; rank by severity+effort.
- **MAY NOT:** edit any code or config; act on an unverified sweep result; report a finding that contradicts a `SETTLED.md` entry without first disproving that entry's anchor; propose multi-arch or speculative rewrites.

### build-engineer
- **Scope:** `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, `Dockerfile`, `Makefile`, dependency pins, `.github/workflows/`.
- **Allowed tools:** edit (build/CI files), bash (configure/build).
- **MUST:** keep the build green; clean-reconfigure after a build-system change; bump both dep-pin sites
  (CMake + Dockerfile) together; keep the single-arch gencode block intact.
- **MAY NOT:** touch kernel/algorithm logic; add multi-arch paths; rename the `Build` CI job (branch-ruleset
  required check); introduce `--mount=type=cache` in the Docker build; collapse the Dockerfile's
  `toolchain`/`builder` split (`make dev` compiles in the `toolchain` stage) or let the two build paths
  diverge on compiler flags — a `-march` difference between them would silently confound every A/B.

### kernel-optimizer
- **Scope:** `src/compute/**` and `src/quant/**` only.
- **Allowed tools:** edit (those dirs), bash (build, benchmark).
- **MUST:** run a **before/after benchmark in the same session** for any perf-affecting change (warm clocks,
  ≥3 trials, decode `tg128`); coherence-check after hot-path edits; stay single-arch.
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
- **MUST:** warm the clocks, run ≥3 trials single-session, enforce the 8%/8%/10% gate, sample
  `nvidia-smi` clocks during the run to rule out depressed host state.
- **MAY NOT:** compare across sessions/days as if equal; refresh a baseline silently (must be stated in the PR).

### docs
- **Scope:** `*.md`, `docs/`, `imp.conf.example`.
- **Allowed tools:** edit (docs only).
- **MUST:** keep docs in sync with code; cite perf numbers from `tests/perf_baseline.json` rather than
  re-typing them; English.
- **MAY NOT:** edit any `.cpp/.cu/.h/.cmake`; introduce version strings; invent benchmark numbers.
