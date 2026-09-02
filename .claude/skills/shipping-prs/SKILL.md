---
name: shipping-prs
description: Use when opening, merging, or releasing a PR for imp - branching off main, `gh pr create`, enabling auto-merge, writing the PR body or a CHANGELOG entry, cutting a tagged release (version bump + CHANGELOG + tag + GitHub release). Symptoms - "PR stuck BLOCKED", "my last commit didn't land on main", "which check is required", "how do I cut a release", "auto-merge merged too early", "check-release failed", "STALE.md blocks git pull". Do NOT use for build/test mechanics (building-and-testing) or perf measurement / baseline refresh (benchmark-cuda).
---

# Shipping PRs & Releases - imp

## Hard rules

| # | Rule | Detail |
|---|---|---|
| 1 | Branch off fresh `origin/main`, `gh pr create --base main`, NEVER stack | `git fetch origin && git switch -c <topic> origin/main`. Stacking on a squash repo caused recovery-PR cascades. Fewer, batched PRs. |
| 2 | English only in the repo | PR title/body, commits, comments, docs. Chat stays German. |
| 3 | `main` merges are SQUASH | PR title = final commit subject `... (#NNNN)`. |
| 4 | Required check = `Build` (ruleset 14716423) | Static gates block inside it since #1527 (`scripts/ci_static_gates.sh`: filesize, lanes, entrypoint, alloc, kernels, launchguards, docs, citations, hygiene). Advisory: `Lint`, `Mock API contract`, `Real API contract (model-less)`, `clang-tidy`, `Sanitizers`. `Test lanes` is its own check (#1770). Read `gh pr checks <n>` after the merge too. |
| 5 | One PR in flight at a time | Every merged PR dirties every open PR through `CHANGELOG.md`. Resolve, `git commit --no-verify` (the push hook gates the same tree), land serially. |
| 6 | Perf- or VRAM-moving change refreshes `tests/perf_baseline.json` IN THE SAME PR and says so | Gate 8% decode / 8% prefill / 10% `own_peak_mb`; `scripts/gen_perf_baseline.sh` (benchmark-cuda). |
| 7 | No em dashes anywhere in the repo | Colon, comma or full stop. All 43 releases were normalised 2026-08-13. |

## The auto-merge race

`auto-merge.yml` arms `gh pr merge --auto --squash --delete-branch` (the flag is what deletes the branch, #1534) the moment a non-draft owner PR is opened (opened / ready_for_review / reopened). The squash fires the instant `Build` is green.

- Push ALL commits before `gh pr create`. Draft PRs are not armed.
- After the merge: `git log -1 --stat origin/main`; when a late commit changed a NUMBER, grep `main` for the corrected value (#1081 shipped disproved figures; #1082 fixed them). Lost commit precedent: `a5403bd5` in #718.
- Late commit sequence: `gh pr merge --disable-auto <PR>` FIRST, edit, `make verify-fast`, `git push`, `gh pr merge --auto --squash --delete-branch`, verify the squash.
- Do not branch a new topic while a previous auto-merge is in flight (#1516 born conflicted 29 min after #1515; #1519 repeated it against #1518 an hour later).
- Never fix a red advisory check by pushing into an armed PR: disable, fix, push, re-arm (a red `Mock API contract` on #1803).

## Ship sequence

```bash
git fetch origin && git switch -c <topic> origin/main
# work; then:
make verify-fast                                   # measures imp:test; rebuild first (make build)
git push -u origin <topic>                         # scripts/pre-push.hook: static gates, require_free_gpu, verify-fast (perf gate only on PERF_RE)
gh pr create --base main --title "<squash subject>" --body-file <file>
git log -1 --stat origin/main                      # after merge
```

- `git push | tail` swallows the gate block (it prints BEFORE the git lines); read the full output.
- A push while your own `verify-fast` runs collides on the GPU (the hook runs the perf gate on `CMakeLists.txt`/kernel diffs).
- `docs_lint.py` regenerates `docs/audit/docs-rewrite/STALE.md` on every local run; commit it as an `.md`-only follow-up BEFORE `gh pr create` (hook skips `.md`), or it blocks `git pull` until `git checkout -- docs/audit/docs-rewrite/STALE.md`.
- Roofline history pushes (`.json`) trigger the full hook: push docs+history with `--no-verify`.
- Moving text from `docs/roadmap.md` to `docs/plans/` rewrites relative links (`](MODELS.md)` -> `](../MODELS.md)`); the hooks run no `hygiene`, CI `Release hygiene` catches it. Local: `docker run --rm -v $PWD:/src -w /src -e HOME=/tmp imp:toolchain bash -c 'git config --global --add safe.directory /src; bash scripts/ci_static_gates.sh hygiene docs citations'`.
- PR monitors: `pgrep -f "<string>"` matches the monitor's own shell; stop an old monitor before starting a second on the same PR.

## The PR body

Every paragraph carries a number, a path or a decision; reasoning goes to `docs/` and the PR links it.

```
## <change>        one section per topic, bullets under it
| | before | after |    a table wherever a count or a timing moved
## Gate            the measured block, pasted from the captured run, nothing wrapped around it
Not in here:       one line: what a reviewer would look for and not find
```

- Capture the gate output to a file, then paste. Never type gate numbers from memory (three wrong PR bodies in one day: #1664, #1666, #1689).
- Cut: the sentence that sets a finding up, the retelling of how a bug was found, reading instructions for the numbers. #1531 went 162 -> 57 lines with no fact lost. Same for commit messages.

## CHANGELOG entries

- One to three lines: what changed for the reader, the number that makes it checkable, `(#NNNN)`. v0.31.0's cut went 389 -> 93 lines for 35 entries.
- Write it short at PR time; before a release cut count lines per entry (>5 = journal).
- New entries merge into the EXISTING `### Added` / `### Changed` / `### Fixed` block of `[Unreleased]`: a second `### Added` fails `check-release.sh` ("repeats a '###' heading").
- Plain punctuation; no internal vocabulary without a greppable symbol; every number names model, quant, unit.

## Triage: a PR that will not merge

```bash
gh pr view <PR> --json mergeStateStatus,statusCheckRollup,reviewDecision
```

| State | Meaning | Action |
|---|---|---|
| `Build` green, still BLOCKED | required-check name != `Build` | realign job name or ruleset 14716423 |
| `reviewDecision` not APPROVED / unresolved thread | review action needed | |
| `gh pr checks` prints NOTHING and `mergeStateStatus=DIRTY` | conflict with `main`; GitHub runs no workflow on an unbuildable merge ref, so no CI, no auto-merge, no arming | rebase onto `origin/main`; force-push is gated, so push a fresh branch and reopen |
| `mergeStateStatus=UNKNOWN` | not computed yet | query again; never build a mechanism on it (#1516 cost an hour) |
| `Build` red on a refactor that moved lines | `citations` gate: dead `file:line` in a living doc (#1783; #1782 paid a CI roundtrip) | `python3 scripts/check_doc_citations.py .` |
| `File size` / `Test lanes` red after adding a GPU test | unlaned-test pin | raise `PINNED` in `tools/check_test_lanes.py` with a reason; allowlist `code_loc` drift: re-pin in `tools/filesize_thresholds.toml` |

## Cutting a tagged release

Version SSoT: `CMakeLists.txt` `project(imp ... VERSION X.Y.Z)`. A release is its own PR.

1. Bump `project(... VERSION X.Y.Z)`.
2. `CHANGELOG.md`: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `[Unreleased]`.
3. `docs/BENCHMARKS.md`: the `**Toolchain (current: \`vX.Y.Z\`):**` line (`check-release.sh` parses exactly that form).
4. `bash scripts/check-release.sh; echo $?` and read the EXIT CODE (an aborted gate prints no FAIL line, #1394). Known exit-1 cause: `test-spec-fidelity` "skipped, card not free enough" (needs ~26 GB free after the earlier stages); run `make test-spec-fidelity` separately, read exit 0. `check-release.sh` prints only `PASS make verify-fast`: run `make verify-fast` separately for the release-note figures; `bench-competitive` writes `/tmp/bench_competitive.tsv`.
5. Merge (squash), then `git tag vX.Y.Z <sha> && git push origin vX.Y.Z`.
6. `gh release create vX.Y.Z --title "vX.Y.Z: <what changed>" --notes-file <file> --verify-tag`. The tag alone is not the release.

Release notes form (three headings and a footer, nothing before them; no install block, no "what imp is" paragraph, measurement conditions ride inside the numbers):

```markdown
## Highlights          3-5 bullets, headline number inline, (#NNNN) for detail
## Also in here        one line each: deps, refusals, guards
## Gate                verify-fast on the tagged tree, model + quant + card named
No breaking changes. / Breaking: <what a user must change>
Full detail: CHANGELOG. N PRs since vX.Y.(Z-1).
```

- Lead with what a reader can now do (a checkpoint that runs, a modality), then speed.
- PR numbers in bullets come from `git log vPREV..HEAD --oneline`, not memory.
- Titles name the change, not the anecdote: `v0.25.0: Nemotron-3.5-Lightning runs; Qwen3.6-35B sees images`.
- Negative results are findings: verdict plus number.

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| Last commit missing from `main` | pushed after `gh pr create` | push all first; late additions via `--disable-auto` |
| PR stuck BLOCKED, `Build` green | required-check name mismatch | ruleset 14716423 |
| Recovery-PR cascade | stacked PRs | one branch per PR off fresh `origin/main` |
| Perf gate red in CI or hook | intentional perf change, stale baseline | refresh `perf_baseline.json` in the same PR |
| Open PR conflicted after another merged | CHANGELOG cycle | resolve, `--no-verify`, land serially |
| `git pull` refuses | regenerated `STALE.md` | `git checkout -- docs/audit/docs-rewrite/STALE.md` |
| German in PR/commit/docs | chat default leaked | English only |
| Release only bumped one of {CMakeLists, CHANGELOG, BENCHMARKS} | | bump all three |
