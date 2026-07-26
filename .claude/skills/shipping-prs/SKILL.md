---
name: shipping-prs
description: Use when opening, merging, or releasing a PR for imp — branching off main, `gh pr create`, enabling auto-merge, cutting a tagged release (version bump + CHANGELOG + tag). Symptoms — "PR stuck BLOCKED", "my last commit didn't land on main", "which check is required", "how do I cut a release", "auto-merge merged too early". Do NOT use for build/test mechanics (building-and-testing) or perf measurement / baseline refresh (benchmark-cuda).
---

# Shipping PRs & Releases — imp

## Hard rules (each has cost a recovery PR or lost work)

1. **Always branch off `main`; `gh pr create --base main`. NEVER stack PRs.** Squash-merge + stacking caused recovery-PR cascades. Branch from fresh `main` every time. Prefer fewer, **batched** PRs over one-per-fix.
2. **English only in the repo.** PR title + body, commits, code comments, docs, `.md` files — all English. (Chat to the user stays German; this rule is only for what lands on GitHub.)
3. **`main` merges are SQUASH** (each PR → one commit `… (#NNN)`). Write the PR title to be the final squash-commit subject.
4. **The required GitHub check is named exactly `Build`** (branch ruleset id `14716423`, "Require CI"). If a CI job is renamed without updating the ruleset, every PR hangs at `mergeStateStatus=BLOCKED`. CI has **no GPU runner** — `Build` only compiles + runs CPU/mock tests. GPU correctness/perf is **your job locally** (`make verify-fast` before push).
5. **Perf-moving change → refresh the baseline IN THE SAME PR and say so.** Regen `tests/perf_baseline.json` via `scripts/gen_perf_baseline.sh` (see `benchmark-cuda`), and state the intended delta in the PR body. CI gate is 3% decode / 5% prefill.

## The auto-merge race (this lost commit `a5403bd5` in #718 — read it)

**Auto-merge is armed AUTOMATICALLY the moment you open a non-draft PR** (workflow `auto-merge.yml` runs `gh pr merge --auto --squash` on owner PRs at opened/ready_for_review/reopened). You don't enable it — `gh pr create` IS the arming event. **Auto-merge squashes the PR the instant `Build` goes green**, so a commit pushed after opening can miss the merge and never land on `main`.

- Push **ALL** commits BEFORE `gh pr create`. Treat the PR as sealed once opened.
- After it merges, **verify** the squash on `main` actually contains your final work:
  `git log -1 --stat origin/main` (or diff the merged SHA against your branch head). Don't assume.
- **Never try to "beat" the race by pushing fast** — if `Build` goes green mid-push, you lose. Disable first.
- **It bit again on 2026-07-26 (#1081), and the failure mode is nastier than a lost commit: it published wrong data.** The PR shipped quality numbers that a follow-up commit had already corrected; the correction lost the race, so `main` documented figures that had been disproved. Nothing was red — CI passed, the PR merged, the branch looked done. It surfaced only by accident during cleanup, and needed a second PR (#1082) to fix. **The verify step above is not optional bookkeeping** — when the late commit changes *claims* (numbers, docs, a caveat), losing it means shipping something you know to be false. Grep `main` for the corrected value, don't just check that the merge happened.
- Opening a **draft** PR is the escape hatch when you know more commits are coming — the workflow skips drafts (arming fires on ready_for_review instead).

**Need another commit after the PR is open** (the common case — do this in order):

```bash
gh pr merge --disable-auto <PR#|branch|url>   # FIRST, before you even write the code — it can fire any second
# … add the change; verify GPU locally …
make verify-fast
git commit -am "…"   &&   git push            # land everything on the remote
gh pr merge --auto --squash                    # re-arm LAST
git log -1 --stat origin/main                  # after merge: confirm your new commit is in the squash
```

## Ship sequence

```bash
git switch main && git pull --ff-only          # always start from fresh main
git switch -c <topic-branch>                   # never reuse / stack
# … work; verify GPU locally …
make verify-fast                               # ~90s pre-push gate (build + filtered tests + perf + smoke)
git push -u origin <topic-branch>              # push EVERYTHING you intend to ship
gh pr create --base main --title "<squash subject>" --body "<what + why + perf note>"
# auto-merge is armed automatically on open (auto-merge.yml) — no manual step;
# it squashes as soon as `Build` is green. After merge:
git log -1 --stat origin/main                  # confirm your final commit is in the squash
```

**Don't branch a new topic off `main` while a previous PR's auto-merge is still in flight** — it squashes onto `main` any moment and your new branch misses it (conflict/rework later). Wait for the merge, `git pull --ff-only`, then branch.

### `mergeStateStatus=BLOCKED` — triage before assuming

Don't guess. Dump the real state first:

```bash
gh pr view <PR> --json mergeStateStatus,statusCheckRollup,reviewDecision
```

- **`Build` shows green but is not registering as satisfied** → the required-check name ≠ `Build` (hard rule 4). This is the imp-specific gotcha and the usual culprit, but confirm it's actually the only required check.
- `reviewDecision` not `APPROVED`, or an unresolved review thread → needs review action.
- Branch out-of-date with `main` → `git pull --no-rebase origin main` (or update via the PR), push.
- A *different* required status (not `Build`) still pending → wait for it.

## Cutting a tagged release (only when explicitly releasing)

Single source of truth for the version is **`CMakeLists.txt`** `project(imp … VERSION X.Y.Z)`. A release is its own PR:

1. Bump `project(... VERSION X.Y.Z)` in `CMakeLists.txt`.
2. `CHANGELOG.md`: rename the `## [Unreleased]` section to `## [X.Y.Z] - YYYY-MM-DD` (Keep-a-Changelog format; Added / Changed / Fixed). Leave a fresh empty `[Unreleased]`.
3. `docs/BENCHMARKS.md`: update the "current: **vX.Y.Z**" line — tagged releases snapshot a SHA, so published numbers must name the release they were taken on.
4. Merge that PR (squash) as usual, then tag the merged commit on `main`: `git tag vX.Y.Z <sha> && git push origin vX.Y.Z`. Tags are `vX.Y.Z` (e.g. `v0.18.0`). `scripts/check-release.sh` gates release-touching PRs in CI.

## Common mistakes → fix

| Symptom | Cause | Fix |
|---|---|---|
| Last commit missing from `main` | Pushed after opening the PR (auto-merge auto-arms on open) | Push all before `gh pr create`; late additions need `--disable-auto` first; verify the squash |
| PR stuck `BLOCKED`, `Build` green | Required-check name ≠ `Build` | Realign CI job name or ruleset 14716423 |
| Recovery-PR cascade | Stacked PRs on a squash repo | One branch per PR, always off fresh `main` |
| Perf gate red in CI | Intentional perf change, stale baseline | Refresh `perf_baseline.json` in the same PR + note it |
| German in PR/commit/docs | Global German default leaked into repo | English only in the repo |

## Red flags — STOP

- About to `gh pr create` but you still have unpushed/uncommitted work → **push first** (auto-merge arms itself on open).
- Branched off a feature branch instead of `main` → start over off `main`.
- Branching off `main` while a prior auto-merge is in flight → wait for it to land, pull, then branch.
- Releasing but only bumped one of {CMakeLists VERSION, CHANGELOG, BENCHMARKS} → bump all three.
