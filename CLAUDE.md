<!--
layer: L3
audience: agents
verified: 2026-10-30
commit: 4967c34e
-->

# imp - Project Instructions

C++23/CUDA LLM inference engine for `sm_120a` only: every compute capability 12.0 card (RTX 50, RTX PRO Blackwell, 8-96 GB: [card table](docs/QUICKSTART.md#which-card)). Tuned and gated on the RTX 5090 (32 GB, 1792 GB/s). A VRAM or "fits" claim names the card size. No portability layer, no FP16 dequant fallback in the hot path. Architecture: [`docs/internals/ARCHITECTURE.md`](docs/internals/ARCHITECTURE.md); sm_120 vs datacenter: [`docs/internals/SM120.md`](docs/internals/SM120.md).

Router: rules for every task here, playbooks in skills. A rule a skill also covers lives in the skill.

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
| Structure audit, dead code, `File size` gate red | skill **codebase-audit**; first [`docs/audit/SETTLED.md`](docs/audit/SETTLED.md) |
| Stub, ignored request field, dead kernel, test asserting nothing | skill **find-stubs** |
| Docs after a change | skill **docs-sync** |
| Doc layer, header, PROV; `docs`/`citations` gate red | skill **docs-layers** |
| VRAM, ownership, lifetime | [`docs/internals/MEMORY.md`](docs/internals/MEMORY.md) first |

- Generic build/test commands only here; directory `CLAUDE.md` files hold area rules.
- `scripts/docs_lint.py` gates the doc layers (L0 `README.md`, L1 `docs/*.md`, L2 `docs/internals/`, L3 `CLAUDE.md` tree): root <= 2000 tokens, directory files <= 800, L3 `verified:` > 14 days behind last commit = error.
- Also canonical: `AGENTS.md` (subagent roles), `docs/internals/BENCHMARKING.md` (measurement contract), `docs/internals/CPP23.md`.

## Build & test

- **Before each GPU job: card free?** User skill `gpu-stats` (`gpu-busy-check.sh`, exit 0 = free). Criterion is load (utilization + VRAM), not `docker ps` (WDDM tenants are invisible there). Busy: report and ask, no wait loop, no start.
- No CUDA toolkit on the host: build in Docker. `build/`, `build-dev/` are root-owned: `make dev-clean` or a throwaway container, never `sudo`.

```
make dev / make dev-test   # incremental (seconds) + the CI lane (ctest -L unit): iterate here
make preflight             # CPU only, before every push: CI static gates, tidy + format on changed lines, actionlint
make build                 # full image (minutes): anything measured or pushed
make verify-fast           # pre-push gate, the only gate that runs a kernel against a check
make verify                # full
```

- **Commit and push WITHOUT `--no-verify`.** Both hooks (`make install-hooks`) run `make preflight` (~20 s); exit 0 = every row ok or skip. GPU tests only with `IMP_HOOK_GPU=1`.
- `make test-unit` != the CI lane binary. CI has no GPU. Targets and CI job names: skill **building-and-testing**.
- **Never bare `make format`.** CI checks changed lines only: format files you created, in edited files only your lines.

## Conventions

- English in the repo, German in chat. No em dashes in the repo.
- SafeTensors (NVFP4, FP8) first; GGUF is legacy (`docs/DESIGN_DECISIONS.md`): keep it loading and gated, no new GGUF kernel or perf work. Competitor reference: vLLM on the same SafeTensors checkpoint, not llama.cpp.
- CHANGELOG: 1-3 lines, <= 360 characters per entry (gated), plus the checkable number. Investigation goes to `docs/`.
- Branch off `main`, `gh pr create --base main`, never stack PRs; batch.
- Perf gate: `tests/perf_baseline.json` (8 % decode / 8 % prefill). Refresh via `scripts/gen_perf_baseline.sh` only for an intentional move, stated in the PR.
- Runtime config only via `RuntimeConfig` (`src/runtime/config.h`: `imp.conf` + `--config` + `--set`). Seeded env vars: `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`; `diagnostics.*`: `IMP_SPEC_TRACE`, `IMP_JUMP_TRACE`, `IMP_PPL_DUMP`, `IMP_WORKER_TIMING`. No ad-hoc env reads.
- Dependency pins only in `cmake/imp-deps.cmake`. Dockerfile: no `--mount=type=cache` on the build dir (stale ninja objects, 03a2cc19); the `ccache` mount stays.
- File size gate: per file, per function body (> 500 code LOC hard), per TU (an `#include`d `.cu` counts against its includer). Exceptions: `[allow]` with reason in `docs/audit/AUDIT_FILESIZE.md`.
- VRAM (WSL2/WDDM): a successful `cudaMalloc` proves nothing, bandwidth tells resident from spilled (#1103), free VRAM only decreases within a process. Capacity is planned: `docs/internals/MEMORY.md`.
