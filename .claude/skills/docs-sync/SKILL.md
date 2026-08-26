---
name: docs-sync
description: Use when keeping imp's docs and config examples coherent after a change - updating docs/internals/ARCHITECTURE.md / README / docs/GOAL.md / docs/MODELS.md / imp.conf.example / CHANGELOG / MISSION_JOURNAL, or "is this doc stale", "document this change", "the example config is out of date", "the README says X but the code does Y". Do NOT use for layer/frontmatter/lint questions (docs-layers), structural code audits (codebase-audit), measuring perf or refreshing the perf baseline (benchmark-cuda), or the agent's private memory (MEMORY.md).
---

# Docs sync — imp

## Hard rules

1. **English only in the repo.** Every committed artifact — PRs, commits, code
   comments, `.md` docs — is written entirely in English. (Chat replies to the user
   stay German; this rule is for things that land in the repo / on GitHub.)
2. **`imp.conf.example` MUST match the parser.** Every key in the example has to be
   a real `B/I/F/S("...")` binder in `src/runtime/config.cpp`; a key the parser
   dropped logs `imp.conf: unknown key` at load (e.g. stale `q4k_imma_enabled` after
   PR #624). When you add/remove/rename a config field: update `config.h`,
   `config.cpp`, AND `imp.conf.example` together, with the real default.
3. **Numbers are commit-anchored, and `docs/PERF.md` is the numbers SSoT.**
   `tests/perf_baseline.json` is the canonical gate (8% decode / 8% prefill, plus
   10% peak VRAM over `metrics.memory_mb.own_peak_mb` - evaluated by
   `scripts/verify.sh`). The README perf block is **generated**
   (`scripts/sync_docs.py`, `<!-- PERF:BEGIN -->` marker; `sync_docs.py --check`
   is a blocking gate) - never hand-edit it. `docs/BENCHMARKS.md` rows are
   `date | commit SHA | CUDA | model | quant | metric | value | exact command` —
   "the commit SHA is the version". Never paste a tok/s number without the commit +
   CUDA + reproducing command.
4. **Verify before you claim.** Docs drift; grep the tree before citing a doc fact.
   (Audit lore can be stale — e.g. the old "GOAL.md still lists H100/H200" flag was
   already fixed; `docs/GOAL.md` states sm_120a-exclusive.)
5. **Never document `IMP_*` env vars as config.** The legacy env surface was retired
   2026-07-07 - a handful of trace/diagnostic vars remain in `src/` (`IMP_CONFIG`,
   `IMP_DETERMINISTIC`, `IMP_FMHA_FA2`, `IMP_NO_WARMUP`, trace knobs), but config is
   `imp.conf` / `--config` / `--set` only. A doc or example suggesting an `IMP_*`
   var as configuration is a bug.
6. **Three blocking gates guard docs, and all run locally in pre-commit/pre-push
   (~2 s):** `scripts/docs_lint.py` (frontmatter/layers/provenance),
   `scripts/sync_docs.py --check` (generated perf block drift), and
   `scripts/check_doc_citations.py` (dead `file:line` citations - covers ALL living
   docs since #1783). A TU split or refactor that moves line numbers breaks
   citations in living docs; re-run the citation gate after any refactor
   (#1782→#1783 lesson). CI job is `Docs`, but the same gates also run blocking
   inside `Build`. For layer/frontmatter questions use skill **docs-layers** first.

## The doc set + what each owns

| Doc | Owns | Touch it when |
|---|---|---|
| `docs/internals/ARCHITECTURE.md` | Canonical narrative (the source of truth) | a refactor changes the high-level structure / data flow |
| `docs/internals/MEMORY.md` · `docs/audit/AUDIT_ARCH_2026_07_29.md` | Memory subsystem: lifetime tiers, allocators, invariants I1–I7, acceptance criteria · the running findings log (incl. REFUTED results) | ownership, lifetime, capacity or VRAM behaviour changes. `ARCHITECTURE.md` defers to it for anything memory-shaped, so don't re-narrate it there |
| `README.md` | User-facing pitch + quickstart; perf block is **generated** from the baseline (`sync_docs.py`) | supported models or build steps change (numbers regenerate themselves) |
| `docs/PERF.md` | **Single source of truth for every number about imp** (partly generated) | any perf number changes - other docs cite it, never fork it |
| `docs/GOAL.md` | North-star target (Qwen3-14B Q6_K @ctx2048 → 175 tok/s) + hardware scope | the goal or hardware scope changes |
| `docs/BENCHMARKS.md` | Reproducible perf table (SHA-anchored) | you measured a number worth publishing |
| `docs/FEATURES.md` · `docs/LIMITATIONS.md` · `docs/DESIGN_DECISIONS.md` | What exists (with status) · what is known not to work · why it is built this way (SSoT each) | a feature ships / a limitation is found or closed / a design decision is made or reversed |
| `docs/roadmap.md` | Current focus + open gaps; drift-gated by `check_doc_citations.py` (#1772) | an item opens, moves, or closes |
| `tests/perf_baseline*.json` | The perf **and peak-VRAM** gate (canonical) | a change intentionally moves perf or peak VRAM — refresh via `scripts/gen_perf_baseline.sh` (it re-pins `own_peak_mb` too) |
| `docs/MODELS.md` | Supported architectures/models | a new arch/model lands or a quant is dropped |
| `docs/MISSION_JOURNAL.md` · `docs/scoreboard.tsv` | Mission log · competitive scoreboard | a competitive/strategic result changes |
| `CHANGELOG.md` | Notable changes | user-visible behavior changes |
| `docs/audit/` | Audit reports (see codebase-audit) | an audit/cleanup pass |

## When a change lands, sync the matching doc

- **Perf moved** (intentionally) → refresh `perf_baseline.json` via
  `scripts/gen_perf_baseline.sh`, add a `docs/BENCHMARKS.md` row, and **say so in the PR**.
  Re-bench properly first (REQUIRED SUB-SKILL: benchmark-cuda — clock ramp + host
  drift make cold single shots lie).
- **Config flag added/removed** → `imp.conf.example` + the `config.h`/`config.cpp` pair.
- **New arch / model / quant** → `docs/MODELS.md` + README if headline.
- **Structural refactor** → `docs/internals/ARCHITECTURE.md` if the narrative no longer
  matches, AND `python3 scripts/check_doc_citations.py` - moved line numbers break
  `file:line` citations across all living docs.
- **Any L0-L2 doc edited** → bump its `verified:`/`commit:` frontmatter or
  `docs_lint.py` warns it stale (details: skill **docs-layers**).

## Common mistakes

- Editing `docs/BENCHMARKS.md` with a number from a cold/single-shot run (use benchmark-cuda).
- Hand-editing the README perf block - it is generated; `sync_docs.py --check` fails CI.
- Adding an `imp.conf.example` key without a parser binder (or vice versa) → silent drift.
- Writing repo docs/comments in German (English-only rule).
- "Fixing" a doc from memory without grepping — half the staleness reports are already fixed.
- Confusing repo docs with the agent's private `MEMORY.md` (that's the auto-memory system, not a repo doc).
