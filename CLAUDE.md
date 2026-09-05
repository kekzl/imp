<!--
layer: L3
audience: agents
verified: 2026-09-06
commit: b5de0dd7
-->

# imp - Project Instructions

From-scratch C++23/CUDA LLM inference engine targeting **exactly one chip: NVIDIA Blackwell `sm_120a`** (RTX 5090 / GB202, 32 GB GDDR7, 1792 GB/s, native FP4 tensor cores). No portability layer, no FP16 dequant fallback in the hot path. Narrative: [`docs/internals/ARCHITECTURE.md`](docs/internals/ARCHITECTURE.md). Hardware, including what sm_120 lacks against datacenter Blackwell: [`docs/internals/SM120.md`](docs/internals/SM120.md).

**This file is the router, not the manual.** It holds what applies to every task; playbooks live in the skills below and load on demand. Anything here that a skill also covers is maintained in the skill.

## Where to start (task -> entry point)

Match the task, invoke that skill first.

| Task | Start here |
|------|-----------|
| Build / run tests / CI red / dep bump | skill **building-and-testing** |
| Write/optimize a CUDA kernel (sm_120a) | skill **sm120-cuda-expert** |
| Benchmark, profile, refresh perf baseline | skill **benchmark-cuda** |
| Touched the forward pass, MoE routing, KV cache, GDN state or graph capture: verify coherence | skill **check-degeneration** |
| Quant formats / loaders / dequant (GGUF, NVFP4, FP8) | skill **quant-formats** |
| imp-server / OpenAI+Anthropic HTTP API | skill **server-api** |
| Add a new model architecture | skill **add-model-arch** |
| Open/merge a PR, cut a release | skill **shipping-prs** |
| Who calls / launches X, blast radius, "is this still used" | skill **code-graph**: ask the index before grepping the tree |
| Structure audit / dead code / god-files / `File size` gate red | skill **codebase-audit**: read [`docs/audit/SETTLED.md`](docs/audit/SETTLED.md) **before** forming hypotheses |
| "Is this actually implemented?": stub, ignored request field, dead kernel, test that asserts nothing | skill **find-stubs** |
| Keep docs in sync after a change | skill **docs-sync** |
| Doc layer, header, PROV; `docs` / `citations` gate red | skill **docs-layers** |
| VRAM / ownership / lifetime / "where did the memory go" | read [`docs/internals/MEMORY.md`](docs/internals/MEMORY.md) **first** |

**Read the `CLAUDE.md` in the directory you are editing first.** `src/compute/`, `src/runtime/`, `src/model/`, `tools/imp-server/` and `tests/` each carry their invariants, entry points, directory-specific test binaries and pitfalls. Generic build and test commands live only here.

Docs are layered and gated by `scripts/docs_lint.py`: `README.md` L0, `docs/*.md` L1, `docs/internals/*.md` L2, the `CLAUDE.md` tree L3 (root budget 2000 tokens, directory files 800; an L3 `verified:` header more than 14 days behind the file's last commit is an error). Other canonical references: `AGENTS.md` (subagent roles), `docs/internals/BENCHMARKING.md` (measurement contract), `docs/internals/CPP23.md` (which C++23 the tree uses).

## Build & test

**Before any GPU job (tests, benchmarks, profiling, inference) check that the card is free, and re-check before each job.** The criterion is load (utilization plus used VRAM), not the process list: WDDM tenants never appear in `docker ps`, and a container holding the card may be the operator's own server. The check is the user-level `gpu-stats` skill (`gpu-busy-check.sh`, exit 0 = free). Busy: report the output and ask; never wait in a loop, never start anyway.

The host has **no CUDA toolkit** by design: build inside Docker. `build/` and `build-dev/` are root-owned by the container; remove them via `make dev-clean` or a throwaway container, never `sudo`.

```
make dev / make dev-test   # incremental (seconds) + the CI lane (ctest -L unit) - iterate here
make build                 # full image (minutes) - the gate, benchmarks, pre-push
make verify-fast           # pre-push gate (#1587), the only gate that runs a kernel against a check
make verify                # full
```

`make build` for anything you *measure* or push; `make dev` for everything else. `make test-unit` is a different binary from the CI lane: green there is not green in CI. CI has no GPU. Target list and CI job names: skill **building-and-testing**.

**Never run bare `make format`.** The repo is not uniformly clang-formatted and CI checks *changed* lines only: format files you created; in files you edited fix only your own added lines.

## Conventions

- **The CHANGELOG is a changelog, not a journal.** One to three lines per entry, plus the number that makes it checkable; the investigation goes to `docs/` and the entry links there.
- **English only in the repo** (PRs, commits, comments, docs). Chat replies to the user stay German. No em dashes in the repo (skill **shipping-prs**, rule 7).
- **Branch off `main`, `gh pr create --base main`, never stack PRs.** Fewer, batched PRs.
- **Performance is gated.** `tests/perf_baseline.json` is canonical (8 % decode / 8 % prefill). Refresh via `scripts/gen_perf_baseline.sh` only when a change intentionally moves perf, and say so in the PR.
- **Runtime config is `RuntimeConfig`** (`src/runtime/config.h`: `imp.conf` + `--config` + `--set`). The only env vars seeded into it: `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`, `IMP_SPEC_TRACE`, `IMP_JUMP_TRACE`, `IMP_PPL_DUMP`, `IMP_WORKER_TIMING` (the last four land in `diagnostics.*`). No ad-hoc env reads.
- **Dependency pins** are single-sourced in `cmake/imp-deps.cmake`. Docker build cache must **not** use `--mount=type=cache` (silently invalidates test results).
- **File size** is gated on recompile blast radius, not line count: per file, per function body (hard >500 code LOC) and per translation unit (an `#include`d `.cu` counts against its includer). Monolithic belongs in `[allow]` with a reason: `docs/audit/AUDIT_FILESIZE.md`.
- **VRAM misleads rather than fails on this box.** A successful `cudaMalloc` proves nothing (WDDM oversubscribes into host memory; bandwidth tells resident from spilled, 1530 vs 237 GB/s, #1103), and free VRAM only ever decreases within a process. Capacity is planned, not discovered: `docs/internals/MEMORY.md`.
- Match surrounding code style; simple and direct, no speculative abstraction.
