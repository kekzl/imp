# imp — Project Instructions

From-scratch C++23/CUDA LLM inference engine targeting **exactly one chip: NVIDIA Blackwell `sm_120a`** (RTX 5090 / GB202, 32 GB GDDR7, 1792 GB/s, native FP4 tensor cores). No portability layer, no FP16 dequant fallback in the hot path. ~100k LOC (src/ + include/). See [`docs/architecture.md`](docs/architecture.md) (canonical narrative) and [`docs/sm120.md`](docs/sm120.md).

**This file is the router, not the manual.** It holds what applies to every task; the playbooks live in the skills below and are loaded on demand. If something here is also in a skill, the skill is the copy that gets maintained.

## Where to start (task → entry point)

Match the task, invoke that skill first.

| Task | Start here |
|------|-----------|
| Build / run tests / CI red / dep bump | skill **building-and-testing** |
| Write/optimize a CUDA kernel (sm_120a) | skill **sm120-cuda-expert** |
| Benchmark, profile, refresh perf baseline | skill **benchmark-cuda** |
| Verify output coherence after hot-path change | skill **check-degeneration** |
| Quant formats / loaders / dequant (GGUF, NVFP4, FP8) | skill **quant-formats** |
| imp-server / OpenAI+Anthropic HTTP API | skill **server-api** |
| Add a new model architecture | skill **add-model-arch** |
| Open/merge a PR, cut a release | skill **shipping-prs** |
| Structure audit / dead code / god-files | skill **codebase-audit** |
| Keep docs in sync after a change | skill **docs-sync** |
| VRAM / ownership / lifetime / "where did the memory go" | read [`docs/MEMORY_ARCHITECTURE.md`](docs/MEMORY_ARCHITECTURE.md) **first** |

Canonical references: `docs/architecture.md` (narrative), `docs/sm120.md` (hardware), `docs/MEMORY_ARCHITECTURE.md` (memory subsystem: tiers, allocators, invariants I1-I7), `AGENTS.md` (subagent roles + guardrails), `docs/BENCHMARKING.md` (measurement contract).

## Two facts about this box that mislead rather than fail

- **A successful `cudaMalloc` proves nothing about free VRAM.** WDDM oversubscribes
  into host memory and returns `cudaSuccess` — 28 GiB succeeds with 22.6 GiB
  reported free. Measure *bandwidth* to tell resident from spilled: ~1530 vs
  ~237 GB/s. That 6.5x cliff is the mechanism behind #1103 (55 vs 391 tok/s), so
  "0 MiB free" is a correctness problem, not a tight fit.
- **Free VRAM only ever decreases within a process.** WSL2/WDDM never returns a
  process's peak commitment, however cleanly CUDA released it. Anything sized off
  `cudaMemGetInfo` is reading a moving floor — which is why capacity is planned,
  not discovered.

## Build & test

**Before any GPU job — tests, benchmarks, profiling, inference — check the card is free.** `nvidia-smi` must show no compute processes. A busy GPU corrupts numbers and can OOM. Re-check before *each* job, not once per session.

The host has **no CUDA toolkit** by design — build inside Docker. `build/` and `build-dev/` are root-owned by the container; remove them via `make dev-clean` or a throwaway container, never `sudo`.

```
make dev / make dev-test   # incremental (2-14 s) + the real CI lane — iterate here
make build                 # full image (~3.5 min) — the gate, benchmarks, pre-push
make verify-fast           # ~90 s pre-push gate    make verify   # ~5 min full
```

`make build` for anything you *measure* or push; `make dev` for everything else. `make test-unit` is a different binary from the CI lane — green there is not green in CI. Details, target list and CI job names: skill **building-and-testing**.

**Never run bare `make format`.** The repo is not uniformly clang-formatted, so formatting a whole file rewrites hundreds of lines you did not touch. CI checks *changed* lines only: format files you created, and for files you edited intersect the violation list with your own added lines rather than trusting a clean `format-check`.

## Conventions

- **The CHANGELOG is a changelog, not a journal.** One to three lines per entry:
  what changed for the reader, plus the number that makes it checkable. The
  investigation goes to `docs/` or `docs/MISSION_JOURNAL.md` and the entry links
  there.
- **English only in the repo.** PRs, commits, comments, docs, `.md` files. (Chat
  replies to the user stay German — this covers artifacts that land in the repo
  or on GitHub.)
- **Always branch off `main` and `gh pr create --base main`.** Never stack PRs
  (squash-merge + stacking caused recovery-PR cascades). Prefer fewer, batched PRs.
- **Performance is gated.** `tests/perf_baseline.json` is canonical (3% decode /
  5% prefill). Refresh via `scripts/gen_perf_baseline.sh` only when a change
  intentionally moves perf, and say so in the PR.
- Runtime config is `RuntimeConfig` in `src/runtime/config.h` (`imp.conf` +
  `--config` + `--set`). The only env vars still seeded are `IMP_DETERMINISTIC`
  and `IMP_FMHA_FA2`; don't reintroduce ad-hoc env reads.
- Internal errors throw and are translated to `ImpError` at the
  `src/api/imp_api.cpp` boundary — intentional, don't convert them to status returns.
- Match surrounding code style; simple and direct, no speculative abstraction.

## File size

The cost of an oversized file here is **recompile blast radius**, not line count: each `.cu` is one translation unit, so touching a kernel in a 1.5k-LOC `.cu` re-`ptxas`es the whole thing with no intra-file parallelism. One logical unit per file; split kernel / launch wrapper / explicit instantiations when recompiles bite.

`tools/check_filesize.py` gates *code* LOC (comments and blanks stripped) and runs in CI as `File size`. A legitimately monolithic file belongs in `[allow]` in `tools/filesize_thresholds.toml` **with a reason** — don't split for splitting's sake. Rationale per file: `docs/audit/AUDIT_FILESIZE.md`.

## Hardware reality (sm_120 ≠ datacenter Blackwell)

- **No `tcgen05` / TMEM / wgmma / TMA-WS grouped GEMM.** The FP4 path is
  `mma.sync` `mxf4nvf4` with FA2-style block-scaling. Ignore B200/`sm_100`
  (FlashAttention-4-style) kernel designs unless porting.
- **No FP4 cuBLASLt kernels on sm_120** → CUTLASS is the primary GEMM path; FP8
  prefill is unavailable. Dependency pins are single-sourced in
  `cmake/imp-deps.cmake` — bump only that file.
- For GGUF, weights become an NVFP4 decode cache at init (bandwidth win on the
  decode hot path); prefill stays full-precision via source dequant.
- Docker build cache must **not** use `--mount=type=cache` (silently invalidates
  test results).

## After hot-path changes

Verify the model still produces coherent output (no repetition loops / token-stuck / state corruption) after touching the forward pass, MoE routing, KV cache, GDN state, or CUDA-graph capture. Skill **check-degeneration**.
