---
name: docs-layers
description: Use when writing, moving or auditing any .md in imp - deciding which file a paragraph belongs in, adding a doc, fixing a stale claim, or when docs_lint.py fails in CI. Covers the four reader layers (L0 README / L1 operators / L2 kernel devs / L3 agents), the HTML-comment metadata header, [PROV:] provenance, the single-source-of-truth map, generated perf blocks, which numbers may appear in prose, plan-doc closure. Triggers on "which doc does this go in", "docs lint failed", "add a doc", "this claim is stale", "update the README", "PROV block", "layer", "STALE.md". Do NOT use for the CHANGELOG or a release body (shipping-prs), or for code-comment accuracy (codebase-audit).
---

# Docs layers - imp

Docs are organised by READER, not topic; `scripts/docs_lint.py` gates it.

## Hard rules

| # | Rule |
|---|---|
| 1 | One layer per file, declared in the metadata header. |
| 2 | Link downward, never repeat: L0 links a kernel, L2 may assume L0 vocabulary. |
| 3 | A claim needs a code path; a number needs provenance. Unbacked = deleted, not softened ("experimental", "planned" are not rescue words). |
| 3b | A number also needs something that FAILS when it drifts: gate thresholds (8%/8%/10%, `verify.sh`), hardware constants (32 GB, 1792 GB/s), dated measurements (findings, true of the afternoon named). File/LOC/test counts and build/test runtimes qualify for none: write the COMMAND that prints the number (`python3 tools/check_test_lanes.py --report`) or the magnitude ("seconds" vs "minutes"). A restated gated number does not inherit the gate (`tests/CLAUDE.md` lost this twice, #1673; bulk-applied #1827). |
| 4 | Records are not documentation: `CHANGELOG.md`, `docs/MISSION_JOURNAL.md`, `docs/vram_audit.md`, root `AUDIT.md`, `docs/roadmap.md`, prefixes `docs/archive/`, `docs/audit/`, `docs/plans/` are lint-excluded, append-only. `docs/BENCHMARKS.md` IS linted (PROV-header allowlist); `docs/roadmap.md` is drift-gated by `check_doc_citations.py` (#1772). |
| 5 | English in the repo. |
| 6 | `file:line` citations in every living doc are gated (#1783): `scripts/check_doc_citations.py .` over 33 living docs (`docs/*.md`, `docs/internals/*.md`, root README/CONTRIBUTING/AGENTS/AUDIT); cite a path, not a bare basename (an ambiguous basename passes as `AMBIGUOUS`); it is the `citations` selection in `ci_static_gates.sh`, NOT the `docs` selection, and it checks the line EXISTS, not what it says. |
| 7 | No em dashes. |

## The layers

| Layer | Reader | Files | May assume |
|---|---|---|---|
| L0 | first contact, knows LLMs not CUDA | `README.md` | nothing |
| L1 | operators | `docs/*.md` | Docker, HTTP, quant basics |
| L2 | kernel work | `docs/internals/*.md` | PTX, MMA, occupancy, roofline |
| L3 | agents | per-directory `CLAUDE.md`, root `AGENTS.md` | only what the file says |

Smell: `mma.sync`, `TMA`, `splitk`, "NVFP4 block scaling" unexplained in L0/L1 belongs in L2.

## Metadata header (HTML comment, never YAML)

```markdown
<!--
layer: L1            # L0 | L1 | L2 | L3   (hard error if missing/invalid)
audience: operators  # newcomers | operators | kernel-devs | agents (hard error)
verified: 2026-08-13 # staleness warning only
commit: <sha8>       # hard error if missing; drift warning "edited Nx since" (#1683)
-->
```

Bump `verified:`/`commit:` on every content edit. Legacy `---` YAML still accepted. Do NOT convert `.claude/skills/*/SKILL.md` or `.github/ISSUE_TEMPLATE/*.md` (their YAML is functional).

## Single source of truth

| Information | Owner |
|---|---|
| any number | `docs/PERF.md` (README embeds a GENERATED extract) |
| what exists, with status | `docs/FEATURES.md` |
| what does not / untested | `docs/LIMITATIONS.md` |
| absent on purpose | `docs/DESIGN_DECISIONS.md` |
| what `sm_120a` has and lacks | `docs/internals/ARCHITECTURE.md` ("no tcgen05/TMEM/wgmma" once stood in eight files) |
| verdicts on levers | `docs/roadmap.md` ledgers (fact + number + decision per row; investigation in `docs/plans/` or the PR) |

Status legend in feature tables: verified (code path AND a gate test) / implemented (must also appear in `LIMITATIONS.md`) / not implemented (points at `DESIGN_DECISIONS.md`). "Verified" never means "green in CI" for anything GPU-shaped: CI has no GPU runner; the gate is `make verify-fast` locally.

## Provenance

L0/L1 throughput figures carry `[PROV: commit=<sha7> date=<YYYY-MM-DD> hw=RTX5090 model=<name> quant=<fmt> cuda=<ver> path=<dispatch> cmd=<command> n=<runs>]` within 12 lines (the quickstart is not exempt: a raw tok/s without PROV fails; write a percentage plus a link to the doc that carries the PROV). Generator emits `unknown` for missing fields (#1684); hand-written blocks fill real values. Severity: fails in L0/L1, warns in L2. `BENCHMARKS.md`, `GOAL.md`, `MODELS.md` declare per-row conventions in their headers. A harness that outlives an edit window puts its md5 next to the commit (`harness_md5=`).

## Generated blocks

`README.md` and `docs/PERF.md` `<!-- PERF:BEGIN -->` / `<!-- PERF:END -->`: hand edits fail CI. `python3 scripts/sync_docs.py` regenerates from `tests/perf_baseline.json`; `--check` is what CI runs.

## Before you push

`bash scripts/ci_static_gates.sh docs citations` (~2 s; the hooks run it). The `Docs` job runs `docs`; `Build` runs everything unfiltered, so a docs failure shows as `Build` red. Lint checks: forbidden tokens, unprovenanced numbers, header, generated drift, dead links, size budgets (README <= 400 lines, root `CLAUDE.md` <= 2000 tokens, per-directory <= 800), staleness > 180 days (warning -> `docs/audit/docs-rewrite/STALE.md`), refs-generator listing (`tests/refs/gen_*.py` rows in `tests/refs/README.md`, #1730). `.gitignore` respected (#1698). `STALE.md` is regenerated on every local run and blocks `git pull` until committed or `git checkout -- docs/audit/docs-rewrite/STALE.md`.

## Adding a doc

1. Layer from the reader. 2. Header. 3. Numbers: PROV or a link to `PERF.md`. 4. Features: code path `file.cu:123` + status. 5. Link from `docs/README.md` (and the README routing table if front-door). 6. Run both scripts.

## Per-directory `CLAUDE.md` (<= 800 tokens)

```
# <dir>: purpose (2 lines)
## Invariants     ## Entry points (3-7 files)   ## Build & test (this dir)
## Pitfalls       ## Do not touch               ## See also (links, no repetition)
```

No perf numbers in any `CLAUDE.md`; link `PERF.md`. L3 downgrades unprovenanced numbers to a warning; the rule is what you write to. Hook edits (`scripts/*.hook`) count as docs for the hooks (`.hook` skips the GPU gate since #1825).

## Plan docs (`docs/plans/`)

`YYYY-MM-DD-<topic>.md`, lint-exempt records, no header. Closure (#1786): `~~strikethrough~~` + `DONE <date> (#PR)` / `MEASURED` / `ANSWERED` / `REFUTED` / `CLOSED <date> (<reason>)`, acceptance cell rewritten to the standing evidence, terminal `## ROADMAP CLOSED (<date>)`. Items are closed, never deleted. Moving text out of `docs/roadmap.md` rewrites relative links (`](MODELS.md)` -> `](../MODELS.md)`); the hooks do not run `hygiene`, CI `Release hygiene` does.

## Audit trail

`docs/audit/docs-rewrite/`: `DOC_INVENTORY.md`, `CLAIM_VERIFICATION.md`, `PURGE_LOG.md` (append every removed claim with a reason), `OPEN_QUESTIONS.md`, `ONBOARDING_RUN.md`, `AGENT_EVAL.md`, `STALE.md`.

## Traps

- The brief can be stale: the rewrite's own brief carried six claims the tree had refuted, two of them perf figures that would have published a phantom regression.
- Writing an explanation a second time (PR body, then doc) means the second copy is wrong-sized.
- A stale code comment outranks nothing (`engine_init_resolver.cpp` says "prefill is never graph-captured" beside a `true` default). This skill file is outside the citation gate: prefer path-only citations here.
- Run the documented command before documenting it (`/v1/messages` returns a `thinking` block at `content[0]`; nothing said so until someone ran the curl).
- De-prose sweeps (#1802 roadmap 1292 -> 761 lines, #1804 23 files +2033/-3709) work as parallel worktree agents on their own branches; re-run the citation gate after compaction (`KERNELS.md:35` became `:20`).
