---
name: docs-layers
description: Use when writing, moving or auditing any .md in imp — deciding which file a paragraph belongs in, adding a doc, fixing a stale claim, or when docs_lint.py fails in CI. Covers the four reader layers (L0 README / L1 operators / L2 kernel devs / L3 agents), frontmatter, [PROV:] provenance, the single-source-of-truth map, and the generated perf blocks. Triggers on "which doc does this go in", "docs lint failed", "add a doc", "this claim is stale", "update the README", "PROV block", "layer". Do NOT use for the CHANGELOG or a release body (shipping-prs), or for code-comment accuracy (codebase-audit).
---

# Docs layers — imp

The documentation is organised by **who is reading**, not by topic. Getting a
paragraph into the wrong layer is the most common defect here, and it is what
`scripts/docs_lint.py` gates.

## Hard rules

1. **One layer per file**, declared in frontmatter. No mixed documents.
2. **Link downward, never repeat.** L0 does not explain a kernel, it links one.
   An L2 doc may assume L0 vocabulary; never the reverse.
3. **A claim needs a code path; a number needs provenance.** What cannot be
   backed is *deleted*, not softened. "experimental", "partial" and "planned"
   are not rescue words.
3b. **A number also needs something that FAILS when it goes wrong** - otherwise
   it is a maintenance bill with no reader, and it will be wrong. Three kinds
   qualify: gate thresholds (8%/8%/10%, `verify.sh` enforces exactly those),
   hardware constants (32 GB, 1792 GB/s - they cannot drift), and dated
   measurements, which are findings rather than inventory and are true of the
   afternoon they name. Counts of files, LOC or test cases and exact
   build/test runtimes qualify for none of it: **write the command that prints
   the number**, or the magnitude that changes a decision ("seconds" vs
   "minutes"), and stop. Restating a gated number in prose does not inherit its
   gate - the pin in `check_test_lanes.py` guards the tool's constant, not a
   copy of it in a doc. Applied in bulk 2026-08-31; `tests/CLAUDE.md` had lost
   this argument twice by then (#1673, then all four of its literals again).
4. **Records are not documentation.** Excluded from the linter: `CHANGELOG.md`,
   `docs/MISSION_JOURNAL.md`, `docs/vram_audit.md`, root `AUDIT.md`,
   `docs/roadmap.md`, plus the prefixes `docs/archive/`, `docs/audit/`,
   `docs/plans/`. Append-only, never rewritten; a record is a statement about
   one dated afternoon. Two nuances: `docs/BENCHMARKS.md` IS linted (it sits on
   the PROV-header allowlist, not the exclusion list), and `docs/roadmap.md`,
   though lint-excluded, is drift-gated by `check_doc_citations.py` (#1772)  - 
   dead `path:line` citations and renamed bare doc names fail there.
5. **English in the repo.** German only in chat.
6. **`file:line` citations in ANY living doc are gated** (#1783):
   `scripts/check_doc_citations.py` covers all 33 living docs (`docs/*.md`,
   `docs/internals/*.md`, root README/CONTRIBUTING/AGENTS/AUDIT). Cite a path,
   not a bare basename; after any refactor that moves line numbers, re-run it.
   It is the `citations` selection in `ci_static_gates.sh` - NOT part of the
   `docs` selection, so the `Docs` CI job does not run it; `Build`,
   pre-commit and pre-push do. Records are excluded.

## The layers

| layer | reader | files | may assume |
|---|---|---|---|
| L0 | first contact, knows LLMs not CUDA | `README.md` | nothing |
| L1 | operators: deploy, configure, diagnose | `docs/*.md` | Docker, HTTP, quant basics |
| L2 | kernel work | `docs/internals/*.md` | PTX, MMA, occupancy, roofline |
| L3 | AI agents | `CLAUDE.md` per directory (+ root `AGENTS.md`, same allowlist) | only what the file says |

**Smell test:** if a paragraph in L0 or L1 uses `mma.sync`, `TMA`, `splitk` or
"NVFP4 block scaling" without explaining it, it belongs in L2.

## Metadata header, on every in-scope file

**An HTML comment, never YAML frontmatter.** GitHub renders YAML front matter as
a visible table at the top of the page, so the first thing a visitor saw on the
README was `layer / audience / verified / commit` instead of what imp is. The
header is for the linter; the reader must not meet it.

```markdown
<!--
layer: L1            # L0 | L1 | L2 | L3   (hard error if missing/invalid)
audience: operators  # newcomers | operators | kernel-devs | agents (hard error)
verified: 2026-08-13 # last content verification (staleness warning only)
commit: <sha8>       # what it was verified against (hard error if missing)
-->
```

The `commit:` field also powers a drift warning ("edited Nx since the commit it
says it was verified against", #1683) - **bump `verified:`/`commit:` whenever
you edit the file's content**, or the lint flags it. The legacy `---` YAML form
is still ACCEPTED by the linter (so old files are never silently unchecked),
but new files use the HTML comment.

Do **not** convert `.claude/skills/*/SKILL.md` or `.github/ISSUE_TEMPLATE/*.md`:
their YAML frontmatter is functional, parsed by the skill loader and by GitHub.

## Single source of truth

Nothing outside the owning file states these. Everything else links.

| information | owner |
|---|---|
| any number | `docs/PERF.md` (README embeds a **generated** extract) |
| what exists, with status | `docs/FEATURES.md` |
| what does not, or is untested | `docs/LIMITATIONS.md` |
| what is absent on purpose | `docs/DESIGN_DECISIONS.md` |
| what `sm_120a` has and lacks | `docs/internals/ARCHITECTURE.md` (the linter allowlists the whole `docs/internals/` prefix; the ownership rule is this table's, stricter than the gate) |

The delimitation rule earns its own line: "no tcgen05 / TMEM / wgmma" once stood
in eight files, and a reader could not tell which was maintained. L2 docs may
*derive* from it ("no tcgen05, therefore the MMA blocks the issuing warp"); that
is rationale, not restatement. L0 and L1 link.

## Status legend, in every feature table

- ✅ **verified** — code path **and** a test that runs in a gate
- 🟡 **implemented** — code path, no test → **must also appear in `LIMITATIONS.md`**
- ⚪ **not implemented** — deliberate → **must point at `DESIGN_DECISIONS.md`**

**"Verified" never means "green in CI" for anything GPU-shaped.** CI has no GPU
runner; the CPU lane runs in under a second without launching one kernel
(`python3 tools/check_test_lanes.py --report` for the count - a literal here
went 248 stale in nine days, #1673).
The gate is `make verify-fast`, locally, before push. Never write that CI tests
the kernels.

## Provenance

Every throughput figure in L0/L1 carries:

```
[PROV: commit=<sha7> date=<YYYY-MM-DD> hw=RTX5090 model=<name> quant=<fmt>
       cuda=<ver> path=<dispatch> cmd=<command> n=<runs>]
```

(The generator emits `unknown` for fields the baseline lacks (#1684) - hand-
written blocks fill in real values, never copy `unknown`.)

Severity follows the layer, deliberately: **fails** the build in L0/L1, **warns**
in L2. In L2 a number is usually the result of the experiment the paragraph
describes, often a refuted one, and failing there would push the next author to
delete the figure rather than document it.

Three files (`BENCHMARKS.md`, `GOAL.md`, `MODELS.md`) carry provenance per row in
a convention declared in their own header. The rule is "no number without
provenance", not "without this syntax"; the linter checks the declaration is
still there.

## Generated blocks

`README.md` and `docs/PERF.md` contain `<!-- PERF:BEGIN -->` / `<!-- PERF:END -->`.
**Editing inside them by hand is a CI failure.**

```
python3 scripts/sync_docs.py           # regenerate from tests/perf_baseline.json
python3 scripts/sync_docs.py --check   # what CI runs
```

The source is `tests/perf_baseline.json` because that is what `verify-fast`
compares against; generating from anywhere else lets the README and the gate
drift apart.

## Before you push

```
bash scripts/ci_static_gates.sh docs citations   # what the hooks run (~2 s)
```

The pre-commit and pre-push hooks run this automatically (#1783). The `Docs` CI
job runs the `docs` selection; the SAME gates also run unfiltered as the first
step of the required `Build` check, so a docs-gate failure surfaces as `Build`
red. The eight lint checks: forbidden tokens, unprovenanced numbers,
frontmatter, generated drift, dead links, size budgets (README ≤ 400 lines,
root `CLAUDE.md` ≤ 2000 tokens, per-directory ≤ 800), staleness > 180 days
(warning → `docs/audit/docs-rewrite/STALE.md`), and refs-generator listing
(every `tests/refs/gen_*.py` needs a row in `tests/refs/README.md`, #1730).
The linter respects `.gitignore` (#1698), so scratch dirs do not flood it.

## Adding a doc

1. Decide the layer from the **reader**, not the topic.
2. Add frontmatter.
3. If it states a number, either a `[PROV:]` block or a link to `PERF.md`.
4. If it states a feature, a code path `file.cu:123` and a status.
5. Link it from `docs/README.md` and, if a reader would arrive from the front
   door, from the README routing table.
6. Run both scripts.

## Per-directory `CLAUDE.md`

Fixed section order, ≤ 800 tokens:

```
# <dir> — purpose (2 lines)
## Invariants     what must never break here
## Entry points   3-7 files, one line each
## Build & test   exact commands for THIS directory
## Pitfalls       what already went wrong here
## Do not touch   generated / vendored
## See also       links to docs/internals/, not repetition
```

**No perf numbers in any `CLAUDE.md`** — only a link to `PERF.md`, or there is a
second truth nobody maintains. (The gate is softer than the rule: L3 downgrades
unprovenanced numbers to a warning, and the root `CLAUDE.md` carries one such
number today - the rule is what you WRITE to, the gate is a floor.)

## Plan docs (`docs/plans/`)

`docs/plans/YYYY-MM-DD-<topic>.md` are lint-exempt RECORDS with no frontmatter.
Closure convention (#1786): an item in the work table is closed by
`~~strikethrough~~` plus one of `DONE <date> (#PR)` / `MEASURED` / `ANSWERED` /
`REFUTED` / `CLOSED <date> (<reason>)`, with the acceptance cell rewritten to
the standing evidence; a finished plan gets a terminal `## ROADMAP CLOSED
(<date>)` section. Items are closed, never deleted.

## The audit trail

`docs/audit/docs-rewrite/` holds the rewrite's own record: `DOC_INVENTORY.md`,
`CLAIM_VERIFICATION.md`, `PURGE_LOG.md` (append-only, every removal with a
reason), `OPEN_QUESTIONS.md`, `ONBOARDING_RUN.md`, `AGENT_EVAL.md`. Append to
`PURGE_LOG.md` when you delete a claim.

## Traps this repo has already hit

- **A dispatch or a doc can itself be stale.** The rewrite's own brief carried
  six claims already refuted by the tree, including two perf figures that would
  have published a regression that never happened. Verify the instruction against
  the code, the same as any other claim.
- **Duplication is the signal that an entry is too long.** If you just wrote the
  explanation in `docs/` or a PR body and are writing it again, the second copy
  is wrong-sized, not the first.
- **A stale code comment outranks nothing.** `engine_init_resolver.cpp` carries
  a "prefill is never graph-captured" comment while the default is `true`.
  Check the value, not the comment. (This skill file itself is outside the
  citation gate's scope - its own `file:line` pointers rot silently, prefer
  path-only citations here.)
- **Run the documented command before documenting it.** The onboarding run found
  that `/v1/messages` returns a `thinking` block at `content[0]`, so a client
  reading `content[0].text` sees nothing. No error, no test, and it was missing
  from the docs until someone ran the curl.
