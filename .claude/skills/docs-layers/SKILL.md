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
4. **Records are not documentation.** Archives, audit ledgers, the roadmap, the
   journal, BENCHMARKS: append-only, excluded from the linter, never rewritten.
   A record is a statement about one dated afternoon.
5. **English in the repo.** German only in chat.

## The layers

| layer | reader | files | may assume |
|---|---|---|---|
| L0 | first contact, knows LLMs not CUDA | `README.md` | nothing |
| L1 | operators: deploy, configure, diagnose | `docs/*.md` | Docker, HTTP, quant basics |
| L2 | kernel work | `docs/internals/*.md` | PTX, MMA, occupancy, roofline |
| L3 | AI agents | `CLAUDE.md` per directory | only what the file says |

**Smell test:** if a paragraph in L0 or L1 uses `mma.sync`, `TMA`, `splitk` or
"NVFP4 block scaling" without explaining it, it belongs in L2.

## Frontmatter, on every in-scope file

```yaml
---
layer: L1            # L0 | L1 | L2 | L3
audience: operators  # newcomers | operators | kernel-devs | agents
verified: 2026-08-13 # last content verification
commit: <sha8>       # what it was verified against
---
```

## Single source of truth

Nothing outside the owning file states these. Everything else links.

| information | owner |
|---|---|
| any number | `docs/PERF.md` (README embeds a **generated** extract) |
| what exists, with status | `docs/FEATURES.md` |
| what does not, or is untested | `docs/LIMITATIONS.md` |
| what is absent on purpose | `docs/DESIGN_DECISIONS.md` |
| what `sm_120a` has and lacks | `docs/internals/ARCHITECTURE.md`, **once** |

The delimitation rule earns its own line: "no tcgen05 / TMEM / wgmma" once stood
in eight files, and a reader could not tell which was maintained. L2 docs may
*derive* from it ("no tcgen05, therefore the MMA blocks the issuing warp"); that
is rationale, not restatement. L0 and L1 link.

## Status legend, in every feature table

- ✅ **verified** — code path **and** a test that runs in a gate
- 🟡 **implemented** — code path, no test → **must also appear in `LIMITATIONS.md`**
- ⚪ **not implemented** — deliberate → **must point at `DESIGN_DECISIONS.md`**

**"Verified" never means "green in CI" for anything GPU-shaped.** CI has no GPU
runner; the CPU lane runs ~1130 cases in 0.39 s without launching one kernel.
The gate is `make verify-fast`, locally, before push. Never write that CI tests
the kernels.

## Provenance

Every throughput figure in L0/L1 carries:

```
[PROV: commit=<sha7> date=<YYYY-MM-DD> hw=RTX5090 model=<name> quant=<fmt>
       cuda=13.3 path=<dispatch> cmd=<command> n=<runs>]
```

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
python3 scripts/sync_docs.py --check
python3 scripts/docs_lint.py
```

Both run in CI as the **Docs** job. The seven checks: forbidden tokens,
unprovenanced numbers, frontmatter, generated drift, dead links, size budgets
(README ≤ 400 lines, root `CLAUDE.md` ≤ 2000 tokens, per-directory ≤ 800),
staleness > 180 days (warning → `docs/audit/docs-rewrite/STALE.md`).

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
second truth nobody maintains.

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
- **A stale code comment outranks nothing.** `engine_init_resolver.cpp:565` says
  prefill is never graph-captured; the default is `true`. Check the value, not
  the comment.
- **Run the documented command before documenting it.** The onboarding run found
  that `/v1/messages` returns a `thinking` block at `content[0]`, so a client
  reading `content[0].text` sees nothing. No error, no test, and it was missing
  from the docs until someone ran the curl.
