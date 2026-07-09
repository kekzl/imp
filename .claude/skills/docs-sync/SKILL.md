---
name: docs-sync
description: Use when keeping imp's docs and config examples coherent after a change — updating architecture.md / README / GOAL.md / supported-models.md / imp.conf.example / CHANGELOG / MISSION_JOURNAL, or "is this doc stale", "document this change", "the example config is out of date", "the README says X but the code does Y". Do NOT use for structural code audits (codebase-audit), measuring perf or refreshing the perf baseline (benchmark-cuda), or the agent's private memory (MEMORY.md).
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
3. **Numbers are commit-anchored.** `tests/perf_baseline.json` is the canonical gate
   (CI: 3% decode / 5% prefill). `BENCHMARKS.md` rows are
   `date | commit SHA | CUDA | model | quant | metric | value | exact command` —
   "the commit SHA is the version". Never paste a tok/s number without the commit +
   CUDA + reproducing command.
4. **Verify before you claim.** Docs drift; grep the tree before citing a doc fact.
   (Audit lore can be stale — e.g. the old "GOAL.md still lists H100/H200" flag was
   already fixed; GOAL.md:31 now states sm_120a-exclusive.)
5. **Never document `IMP_*` env vars as config.** The legacy env surface was retired
   2026-07-07 — the only live env vars are `IMP_DETERMINISTIC` and `IMP_FMHA_FA2`;
   everything else is `imp.conf` / `--config` / `--set`. A doc or example suggesting
   another `IMP_*` var is a bug.

## The doc set + what each owns

| Doc | Owns | Touch it when |
|---|---|---|
| `docs/architecture.md` | Canonical narrative (the source of truth) | a refactor changes the high-level structure / data flow |
| `README.md` | User-facing pitch + quickstart + headline numbers | supported models, build steps, or headline perf change |
| `GOAL.md` | North-star target (Qwen3-14B Q6_K @ctx2048 → 175 tok/s) + hardware scope | the goal or hardware scope changes |
| `BENCHMARKS.md` | Reproducible perf table (SHA-anchored) | you measured a number worth publishing |
| `tests/perf_baseline*.json` | The CI perf gate (canonical) | a change intentionally moves perf — refresh via `scripts/gen_perf_baseline.sh` |
| `docs/supported-models.md` | Supported architectures/models | a new arch/model lands or a quant is dropped |
| `docs/MISSION_JOURNAL.md` · `docs/scoreboard.tsv` | Mission log · competitive scoreboard | a competitive/strategic result changes |
| `CHANGELOG.md` | Notable changes | user-visible behavior changes |
| `docs/audit/` | Audit reports (see codebase-audit) | an audit/cleanup pass |

## When a change lands, sync the matching doc

- **Perf moved** (intentionally) → refresh `perf_baseline.json` via
  `scripts/gen_perf_baseline.sh`, add a `BENCHMARKS.md` row, and **say so in the PR**.
  Re-bench properly first (REQUIRED SUB-SKILL: benchmark-cuda — clock ramp + host
  drift make cold single shots lie).
- **Config flag added/removed** → `imp.conf.example` + the `config.h`/`config.cpp` pair.
- **New arch / model / quant** → `docs/supported-models.md` + README if headline.
- **Structural refactor** → `docs/architecture.md` if the narrative no longer matches.

## Common mistakes

- Editing `BENCHMARKS.md` with a number from a cold/single-shot run (use benchmark-cuda).
- Adding an `imp.conf.example` key without a parser binder (or vice versa) → silent drift.
- Writing repo docs/comments in German (English-only rule).
- "Fixing" a doc from memory without grepping — half the staleness reports are already fixed.
- Confusing repo docs with the agent's private `MEMORY.md` (that's the auto-memory system, not a repo doc).
