---
name: docs-sync
description: Use when keeping imp's docs and config examples coherent after a change - updating docs/internals/ARCHITECTURE.md / README / docs/GOAL.md / docs/MODELS.md / docs/roadmap.md / imp.conf.example / CHANGELOG / MISSION_JOURNAL, or "is this doc stale", "document this change", "the example config is out of date", "the README says X but the code does Y", "add a roadmap ledger row". Do NOT use for layer/frontmatter/lint questions (docs-layers), structural code audits (codebase-audit), measuring perf or refreshing the perf baseline (benchmark-cuda), or the agent's private memory (MEMORY.md).
---

# Docs sync - imp

## Hard rules

| # | Rule |
|---|---|
| 1 | English only in the repo. |
| 2 | `imp.conf.example` MUST match the parser: every key is a `B/I/F/S("...")` binder in `src/runtime/config.cpp` (surface in `src/core/config/*.h`); a dropped key logs `imp.conf: unknown key` (stale `q4k_imma_enabled` after #624). Add/remove/rename = `config.h`, `config.cpp`, `imp.conf.example` together, real default. A bogus `--set` key fails at start (`no such key`). |
| 3 | Numbers are commit-anchored; `docs/PERF.md` owns them. `tests/perf_baseline.json` is the gate (8%/8%/10% `own_peak_mb`, `scripts/verify.sh`); README perf block is GENERATED (`scripts/sync_docs.py`, `<!-- PERF:BEGIN -->`; `--check` blocks). `docs/BENCHMARKS.md` rows: date, commit SHA, CUDA, model, quant, metric, value, exact command. |
| 4 | Verify before you claim: grep the tree; half the staleness reports are already fixed (the "GOAL.md lists H100" flag was one). |
| 5 | Env vars are not config: the engine reads only `IMP_CONFIG`, `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`, `IMP_WORKER_TIMING`, `IMP_SPEC_TRACE`, `IMP_JUMP_TRACE`, `IMP_PPL_DUMP`; config is `imp.conf` / `--config` / `--set`. The container surface is `IMP_CONFIG` + `IMP_SET` (#1823, `docs/DEPLOYMENT.md` "From a container"); the 19 legacy `IMP_*` entrypoint names are frozen and get no new siblings. |
| 6 | Three blocking doc gates, in hooks and `Build`: `scripts/docs_lint.py`, `scripts/sync_docs.py --check`, `scripts/check_doc_citations.py .` (all living docs since #1783; a TU split or a de-prose compaction moves line numbers). |
| 7 | No prose: a paragraph without a number, path or decision is deleted (#1802/#1804 sweeps). No em dashes. |

## The doc set

| Doc | Owns | Touch when |
|---|---|---|
| `docs/internals/ARCHITECTURE.md` | canonical narrative | a refactor changes structure or data flow |
| `docs/internals/MEMORY.md`, root `AUDIT.md`, `docs/audit/AUDIT_ARCH_2026_07_29.md` | memory subsystem (tiers, allocators, invariants I1-I7), running findings log (incl. REFUTED), 07-29 architecture audit | ownership, lifetime, capacity, VRAM behaviour |
| `README.md` | pitch + quickstart; perf block generated | supported models or build steps change |
| `docs/PERF.md` | every number (partly generated) | any perf number changes |
| `docs/GOAL.md` | north star (Qwen3-14B Q6_K @ctx2048 -> 175 tok/s) + hardware scope | goal or scope changes |
| `docs/BENCHMARKS.md` | reproducible SHA-anchored table; `**Toolchain (current: \`vX.Y.Z\`):**` line parsed by `check-release.sh` | a publishable number; a release |
| `docs/FEATURES.md`, `docs/LIMITATIONS.md`, `docs/DESIGN_DECISIONS.md` | what exists / what does not / why | feature ships, limitation opens or closes, decision made or reversed |
| `docs/roadmap.md` | verdict ledgers (fact + number + decision per row), citations drift-gated by `check_doc_citations.py` (#1772) | a lever is measured, shipped, refuted or closed; investigation goes to `docs/plans/YYYY-MM-DD-<topic>.md` |
| `docs/plans/` | records with closure marks (docs-layers) | an item closes |
| `tests/perf_baseline*.json` | perf + peak-VRAM gate | intentional perf/VRAM move (`scripts/gen_perf_baseline.sh`) |
| `docs/MODELS.md`, `docs/DEPLOYMENT.md`, `docs/API.md`, `docs/QUICKSTART.md` | supported models; container/compose; HTTP contract (request tracing, priority); worked example (`kekzle/Qwen3.8-27B-NVFP4-vllm`) | new arch/model/quant; new flag or env; new endpoint field |
| `docs/MISSION_JOURNAL.md`, `docs/scoreboard.tsv` | mission log, competitive scoreboard | competitive result changes |
| `CHANGELOG.md` | 1-3 lines per entry, fact + number + (#PR), merged into the existing `### Added/Changed/Fixed` block (shipping-prs) | user-visible change |
| `docs/audit/` | audit reports and ledgers (codebase-audit) | an audit pass |
| `.claude/skills/*/SKILL.md`, `CLAUDE.md` tree | L3 agent routing; per-dir `CLAUDE.md` <= 800 tokens, no perf numbers | a workflow, gate or trap changes |

## When a change lands

| Change | Sync |
|---|---|
| Perf moved intentionally | `gen_perf_baseline.sh`, `BENCHMARKS.md` row, say so in the PR; measure per benchmark-cuda first |
| Config key added/removed/renamed | `config.h` + `config.cpp` + `imp.conf.example`; a new auto-default also needs an `imp-cli --bench` pin (`apply_config_pins` in `tools/imp-cli/args.cpp`) or the baseline bakes it in |
| New arch / model / quant | `docs/MODELS.md` (+ README if headline) |
| New endpoint field, header or env | `docs/API.md`, `docs/DEPLOYMENT.md`, `tests/api/mock_server.py` mirrors the contract |
| Structural refactor | `ARCHITECTURE.md` if the narrative broke, then `python3 scripts/check_doc_citations.py .` (the #1782 scheduler split cost a CI roundtrip on a `roadmap.md` citation) |
| A lever measured (shipped or refuted) | `docs/roadmap.md` ledger row; harness in `tools/analysis/`; record in `docs/plans/` if the investigation is longer than three lines |
| Any L0-L2 doc edited | bump `verified:`/`commit:` in its header (docs-layers) |

## Common mistakes

- A `BENCHMARKS.md` number from a cold single shot (benchmark-cuda STOP list).
- Hand-editing the README perf block.
- `imp.conf.example` key without a binder, or vice versa.
- German in repo docs.
- "Fixing" a doc from memory without grepping.
- Confusing repo docs with the agent's private `MEMORY.md`.
- A skill or `CLAUDE.md` carrying a count or runtime that nothing checks (docs-layers rule 3b).
