<!--
layer: L3
audience: agents
verified: 2026-09-13
commit: 81d22eff
-->

# imp - Project Instructions

C++23/CUDA LLM inference engine for exactly one chip: NVIDIA Blackwell `sm_120a` (RTX 5090 / GB202, 32 GB GDDR7, 1792 GB/s, native FP4 tensor cores). No portability layer, no FP16 dequant fallback in the hot path. Narrative: [`docs/internals/ARCHITECTURE.md`](docs/internals/ARCHITECTURE.md); what sm_120 lacks against datacenter Blackwell: [`docs/internals/SM120.md`](docs/internals/SM120.md).

This file is the router: rules for every task. Playbooks live in skills; a rule a skill also covers is maintained in the skill.

## Task -> entry point

| Task | Start here |
|---|---|
| Build, tests, CI red, dep bump | skill **building-and-testing** |
| Write/optimize a CUDA kernel | skill **sm120-cuda-expert** |
| Benchmark, profile, perf baseline | skill **benchmark-cuda** |
| Touched forward pass, MoE routing, KV cache, GDN state, graph capture | skill **check-degeneration** |
| Quant formats, loaders, dequant (GGUF, NVFP4, FP8) | skill **quant-formats** |
| imp-server, OpenAI/Anthropic HTTP API | skill **server-api** |
| New model architecture | skill **add-model-arch** |
| PR, merge, release | skill **shipping-prs** |
| Who calls/launches X, blast radius, still used | skill **code-graph**, before grepping |
| Structure audit, dead code, `File size` gate red | skill **codebase-audit**; read [`docs/audit/SETTLED.md`](docs/audit/SETTLED.md) before forming hypotheses |
| Stub, ignored request field, dead kernel, test asserting nothing | skill **find-stubs** |
| Docs after a change | skill **docs-sync** |
| Doc layer, header, PROV; `docs`/`citations` gate red | skill **docs-layers** |
| VRAM, ownership, lifetime | [`docs/internals/MEMORY.md`](docs/internals/MEMORY.md) first |

Editing under `src/compute/`, `src/runtime/`, `src/model/`, `tools/imp-server/` or `tests/`: read that directory's `CLAUDE.md` first. Generic build and test commands live only here.

Doc layers (L0 `README.md`, L1 `docs/*.md`, L2 `docs/internals/`, L3 `CLAUDE.md` tree) are gated by `scripts/docs_lint.py`: root <= 2000 tokens, directory files <= 800, an L3 `verified:` more than 14 days behind the file's last commit is an error. Also canonical: `AGENTS.md` (subagent roles), `docs/internals/BENCHMARKING.md` (measurement contract), `docs/internals/CPP23.md`.

## Build & test

- **Before each GPU job (tests, benchmarks, profiling, inference) check the card is free.** Criterion is load (utilization + used VRAM), not `docker ps`: WDDM tenants never show there. Check: user-level skill `gpu-stats` (`gpu-busy-check.sh`, exit 0 = free). Busy: report and ask; no wait loop, no start anyway.
- The host has no CUDA toolkit: build in Docker. `build/`, `build-dev/` are root-owned: `make dev-clean` or a throwaway container, never `sudo`.

```
make dev / make dev-test   # incremental (seconds) + the CI lane (ctest -L unit): iterate here
make build                 # full image (minutes): anything measured or pushed
make verify-fast           # pre-push gate, the only gate that runs a kernel against a check
make verify                # full
```

- `make test-unit` is a different binary from the CI lane: green there is not green in CI. CI has no GPU. Target list and CI job names: skill **building-and-testing**.
- **Never bare `make format`.** CI checks changed lines only: format files you created; in edited files only your added lines.

## Conventions

- English in the repo (PRs, commits, comments, docs); chat with the user in German. No em dashes in the repo.
- CHANGELOG: 1-3 lines per entry plus the number that makes it checkable; the investigation goes to `docs/`.
- Branch off `main`, `gh pr create --base main`, never stack PRs; batch.
- Perf is gated: `tests/perf_baseline.json` (8 % decode / 8 % prefill). Refresh via `scripts/gen_perf_baseline.sh` only for an intentional move, and say so in the PR.
- Runtime config is `RuntimeConfig` (`src/runtime/config.h`: `imp.conf` + `--config` + `--set`). Env vars seeded into it: `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`; into `diagnostics.*`: `IMP_SPEC_TRACE`, `IMP_JUMP_TRACE`, `IMP_PPL_DUMP`, `IMP_WORKER_TIMING`. No ad-hoc env reads.
- Dependency pins: only `cmake/imp-deps.cmake`. Dockerfile: no `--mount=type=cache` on the build dir (ninja reused stale objects, 03a2cc19); the content-addressed `ccache` mount stays.
- File size is gated on recompile blast radius: per file, per function body (> 500 code LOC hard), per translation unit (an `#include`d `.cu` counts against its includer). Exceptions go to `[allow]` with a reason: `docs/audit/AUDIT_FILESIZE.md`.
- VRAM misleads rather than fails (WSL2/WDDM): a successful `cudaMalloc` proves nothing, measured bandwidth tells resident from spilled (#1103), free VRAM only decreases within a process. Capacity is planned, not discovered: `docs/internals/MEMORY.md`.
- Match surrounding style; simple and direct, no speculative abstraction.
