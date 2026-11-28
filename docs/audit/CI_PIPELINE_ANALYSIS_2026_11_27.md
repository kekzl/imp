<!--
layer: L2
audience: kernel-devs
verified: 2026-11-28
commit: e1a701b1
-->

# CI pipeline analysis kekzl/imp (as of 2026-11-27)

Phases 1-3, nothing changed. Every number comes from a gh output. Raw data and scripts are under `/mnt/data/scratch/pipeline_analysis/` (file name in parentheses). Durations are medians unless stated otherwise.

## Summary

- The merge path is a single job: `Build` is the only required check (`ruleset_14716423.json`). Code PRs wait a median of 435 s, p90 746 s, from push to green check. Docs-only PRs wait 94 s (`build_end.txt`).
- In the `Build` job, 278 s of 432 s go into the compile step. It depends on the number of ccache misses (0 misses: 19 s, 390 misses: 1116 s, `ccache_sample.tsv`) and on 4 vCPU (`nproc=4`, `build_mem_46064840098.txt`). The rest is overhead on the path: ccache save 38 s, container init 32 s, clang-tidy 26-59 s.
- The cache budget is structurally unfavorable. One main push saves about 4.2 GB (ccache 1.49 GB, ASan ccache 1.98 GB, build directory 0.45 GB, PTX 0.11 GB, CodeQL 0.16 GB). Each PR saves another 1.49 GB. The limit is 10 GB (`cache_storage_limit.json`). The oldest entry was 20 min old (`caches_t2.json`). Therefore the release layer cache and the fuzz corpus never come back: `Cache not found` in both nightlies (`fuzz_corpus_cache.txt`).
- Main pushes run serially through one concurrency group: 20 of 60 push builds waited more than 60 s, maximum 1843 s. 6 push runs were preempted (`build_queue_gt60.txt`, `conclusions.txt`). This contradicts the comment in ci.yml that main runs never cancel.
- The three measures with the largest effect per effort are: no ccache save on PRs (38 s less on the required check, 1.49 GB per PR freed), one concurrency group per SHA on main, and no `type=gha` export in the release (317-424 s per release). Next come clang-tidy in the CI image and `-j$(nproc)` instead of `-j2` in the PTX and sanitizer lanes. Everything else is structural: a larger runner, header fan-in, a sanitizer build without CUDA.
- Stability: CI had 0 failures in 162 runs. The 6 failures in the window (CodeQL and auto-merge) all fell within 10 minutes on 2026-11-25, between 15:55 and 16:05 UTC (`failures.txt`, `upload-sarif`). It was an external disturbance, not a flake.

## 1 Inventory

| Item | Finding |
|---|---|
| Language, build | C++23/CUDA (sm_120a), CMake, ccache. In CI the `Build` job uses the Make generator (`cmake -B build` without `-G`, output `[  0%] Building CXX object`, `build_objects_46064840098.txt`). The sanitizer lane uses Ninja. |
| Tests | GTest via ctest label `unit` (CPU), pytest in `tests/api` (mock and model-less). GPU tests run deliberately only locally. |
| Monorepo | no, one CMake project (library, server, tools, tests) |
| Workflows | `ci.yml` (13 jobs), `codeql.yml` (matrix c-cpp/python/actions), `auto-merge.yml`, `ci-image.yml` (only on changes to `.github/ci-image/**`), `release-docker.yml`, `roofline.yml` (only on changes to `tools/roofline/**`) |
| Composite actions, reusable workflows | none (`.github/actions/` is missing) |
| Images | CI: `ghcr.io/kekzl/imp-ci:cuda13.5@sha256:d9b6...` (`.github/ci-image/Dockerfile`, cuda-base plus minimal-build). Release: root `Dockerfile` (cuda-devel, stages toolchain/builder/bundle/runtime) |
| Runner | `ubuntu-26.04` GitHub-hosted, 4 vCPU (`nproc=4`), build peak 7.81 GB (`memory.peak=7809187840`) |
| Repo | public (`repo.json`), runner minutes are free. The bottleneck is time and cache budget, not money. |
| Volume | 400 runs in 2.5 days (`runs_all.json`): CI 162, CodeQL 160, auto-merge 76, roofline 2. Job minutes in the job window 2026-11-25T18:39 to 11-27T22:26: CI 2783, CodeQL 1062 (`runner_minutes.txt`) |
| PR pace | 80 merged PRs, creation to merge median 7.4 min, p90 17.1 min (`pr_wall.txt`). 72 of 77 PR branches had exactly 1 CI run (`pr_runs_per_branch.txt`). |

## 2 Job graph and critical path

```
CI (push main, PR, schedule 03:17, dispatch)
  Build [REQUIRED] ──needs──> Real API contract (model-less)
  Mock API · Lint · File size · Test lanes · Docs · Release hygiene · Alloc sites · Launch guards   (each without needs)
  PTX fallback · Sanitizers   (own CUDA build, without needs, path-controlled)
  Fuzz nightly   (only schedule/dispatch)
CodeQL: c-cpp | python | actions   (matrix, without needs)
Auto-merge: 1 job, arms squash on open
```

| Path | Length (median / p90) | Source |
|---|---|---|
| Merge (push until required green) = `Build`, code PR | 435 s / 746 s, max 1530 s | `build_end.txt` |
| Merge, docs-only PR | 94 s | `build_end.txt` |
| Entire CI run, PR | 502 s / 949 s, last job: Real API 48, Sanitizers 17, PTX 6 of 71 | `stats.txt` |
| Entire CI run, push main | 597 s / 1137 s | `stats.txt` |
| Queue (run created until job started), PR | 3 s, p90 4 s, max 31 s | `queue_by_wf.txt` |
| Queue, push main | 4 s, p90 370 s, max 1843 s, 219 of 669 jobs > 60 s | `queue_by_wf.txt` |

On PRs no runner limit slows things down (queue p90 4 s). The push queue arises solely from the concurrency group.

## 3 Measurements

### 3.1 `Build`, code PRs (n=58, job median 432 s, `steps_build_pr_code.txt`)

| Step | Median | p90 | Share of merge path (median step / 432 s) |
|---|---|---|---|
| Build (Compile + Link) | 278 s | 538 s | 64 % |
| Save ccache (53 of 58 runs) | 38 s | 47 s | 9 % |
| Initialize containers (pull 1.1 GB) | 32 s | 43 s | 7 % |
| clang-tidy (changed files) | 26 s (n=53), with header TUs 59 s (n=5) | 52 s / 118 s | 6-14 % |
| ctest -L unit | 15 s | 18 s | 3 % |
| Restore ccache | 13 s | 17 s | 3 % |
| Install clang-tidy (apt.llvm.org) | 12 s | 16 s, push p90 41 s | 3 % |
| Restore build directory | 10 s | 12 s | 2 % |
| Checkout (fetch-depth 0) | 5 s | 6 s | 1 % |
| Upload imp-server, Configure, Gates (background) | 3 s / 2 s / 0 s | | < 1 % |

Push builds (n=52, `steps_build_push_code.txt`) additionally need `Post Cache build directory`, with 38 s for the 452 MB save.

### 3.2 Compile time versus ccache misses (`ccache_sample.tsv`)

| Run | Event | Misses | Build step |
|---|---|---|---|
| 46036745206 | push | 0 / 1095 | 19 s |
| 46021383871 | PR | 2 / 1095 | 44 s |
| 46064840098 | PR | 26 / 1097 | 162 s |
| 46065445394 | push (same commit merged) | 26 / 1097 | 101 s |
| 46074976618 | PR | 187 / 1098 | 315 s |
| 46002249185 | PR | 390 / 1095 | 1116 s |

- All 1097 TUs go through ccache every time (`Cacheable calls: 1097 / 1097` in a run with 26 misses). The restored `build/` folder saves no compile calls, because the fresh checkout has newer mtimes than the objects. A run with only hits costs 19 s.
- The misses occur a second time on main after the merge (26 and 26). A PR cache under `refs/pull/N/merge` is not readable for `main`.
- Long single TUs (`gaps_pr_46064840098.txt`): `handlers_chat.cpp` 30.8 s, `imp_server_core` up to the link 19.1 s, `test_tool_call.cpp` 20.8 s, clang-tidy on 10 files 71.7 s.

### 3.3 Other jobs (PR, `stats.txt`, `steps_other_pr.txt`, `codeql_steps.txt`)

| Job | Median | p90 | Main step | Merge path |
|---|---|---|---|---|
| Sanitizers (runs in 37 of 71 PRs) | 151 s | 628 s | ASan build 387 s, p90 1944 s | no |
| PTX fallback (runs in 54 of 72) | 252 s | 463 s | device lib 204 s, assemble 78 s | no |
| Real API (model-less) | 40 s, queue 411 s (waits for build) | 46 s | apt + pip + cuBLAS wheel 17 s | no |
| Mock API | 53 s | 58 s | pytest 43 s | no |
| File size | 31 s | 34 s | gates 18 s | no |
| Lint | 22 s | 36 s | clang-format install 11 s | no |
| Alloc sites, Test lanes, Launch guards, Docs, Release hygiene | 17 / 12 / 11 / 10 / 8 s | | | no |
| CodeQL c-cpp PR / push | 199 s / 544 s | 344 s / 731 s | analyze 153 s / 510 s | no |
| CodeQL python, actions | 64 s, 52 s | | | no |
| Fuzz nightly | 2488 s | | 4 x 600 s | no |

### 3.4 Release (`release_steps*.txt`, `release_*.log`)

| Release | Build and push | of which compile (`builder 4/4`) | of which gha cache export | Tarball |
|---|---|---|---|---|
| v0.70.0 (44953879388) | 1637 s | 1069.5 s | 424.1 s | 44 s |
| v0.69.1 (44513955562) | 1236 s | 817.6 s | 317.3 s | 30 s |
| v0.69.0, v0.68.0, v0.67.1 | 1914 s, 1967 s, 1364 s | | | 30-38 s |

In the first build of v0.70.0, no toolchain stage is `CACHED`: the base image is pulled and apt runs (`#17`). The gha cache of the previous release was therefore gone. With a 10 GB limit and about 4.2 GB per main push, it cannot survive.

## 4 Analysis by check item

| Check item | Finding | Verdict |
|---|---|---|
| ccache build | Restore via restore-keys from the newest main entry, hit rate 64-100 % (`ccache_sample.tsv`). On PRs, build saves in 53 of 58 runs 1.49 GB in 38 s. At most 5 of 77 branches reuse it. | PR save is pure lost time on the required check |
| ccache PTX | `actions/cache` (with post-save) also on PRs, 113 MB per entry | small, same logic |
| ccache ASan | Save only on push, when the lane ran: 1.98 GB, the largest entry. One PR restored asan-813c226 while main was already at e9737f1 (`lanes_sample.tsv`) | budget eater |
| Build directory cache | 452 MB, restore 10 s, push save 38 s. Saves no compile calls (3.2), only FetchContent and configure | candidate for replacement: deps into the CI image |
| Cache budget | 10 GB limit, 15.8 GB active at the first snapshot (`cache_usage.json`). At the second snapshot 11 entries, oldest 20 min old (`caches_t2.json`, `cache_t2_time.txt`) | structurally overbooked |
| Fuzz corpus | `Cache not found for input keys: fuzz-corpus-...` in both nightlies | the corpus is never carried over, feature ineffective |
| Docker release | `type=gha,mode=max`: export 317-424 s, never reused. The compile stage depends on `COPY .` and is cold on every release anyway. The ccache mount does not survive the runner (comment in the workflow file) | remove export |
| Docker CI image | 1104 MB compressed, init 32 s per container job. clang-tidy and clang-format are missing from the image and come via apt on every run | clang-tidy into the image |
| paths / change detection | Docs-only skips the compile (14 of 72 PRs, 94 s). PTX and sanitizers are path-controlled. CodeQL and the 8 small gate jobs always run | good. Remainder: docs-only still pulls the 1.1 GB image (32 s of 94 s) |
| concurrency | PR: cancel-in-progress (1 PR run cancelled). Push: group `ci-push-refs/heads/main` without cancel, therefore serial and with pending preemption (6 push runs cancelled). CodeQL likewise (63 push jobs > 60 s queue) | group per SHA |
| Draft PRs | no draft filter in ci.yml, auto-merge skips drafts. No data on drafts in the window | open, low impact |
| Parallelization | Build: `-j$(nproc)` = 4. PTX and sanitizers: `-j$(bash scripts/build_jobs.sh)`, 6 GB per job. According to the script this yields 2 on a 16 GB runner (derived, not logged). The rule stems from CUTLASS TUs that have been gone since 2026-10-31 (script comment). Build at -j4 peaks at 7.81 GB | check `-j$(nproc)` |
| Sanitizer scope | The ASan build compiles 194 CUDA and 511 CXX objects (`sanitizer_objects.txt`). Device code is not sanitized (comment in ci.yml) | structural: CPU-only target like `IMP_FUZZERS_CPU_ONLY` |
| Fail-fast | Static gates run in the background from second 0. The own gate jobs are red after 8-31 s | good |
| Checkout | fetch-depth 0 on 7 jobs, 5 s | no lever |
| Artifacts | `imp-build` 15 MB per code run, read only within the same run, retention 90 days, 2570 artifacts total (`artifacts_summary.json`, `retention.json`) | `retention-days: 1` |
| Runner sizing | The compile scales with the misses on 4 vCPU. A larger runner would be the only lever on the compile core (p90 538 s), but it also costs money on public repos | owner decision |
| Self-hosted | The owner machine has the GPU, but the repo is public and fork approval is set to `first_time_contributors` (`fork_approval.json`) | not recommended |
| Flakiness | CI 0 failures, 0 re-runs (`reruns.json`). Failures only in the disturbance window on 2026-11-25 | stable |
| timeout-minutes | Missing in ci.yml, codeql.yml, roofline.yml and auto-merge.yml (`grep -c` = 0). The default is 6 h | add |
| Schedule | Cron 03:17 UTC, the runs start at 12:32, 12:42 and 13:02 UTC (`schedule_runs.txt`) | GitHub delay, nightly lands in daytime operation |
| permissions | Default `read` (`actions_perm_workflow.json`), ci.yml `contents: read`, auto-merge with job scope. ci-image sets `packages: write` at workflow level, thus also for PR runs (fork token stays read-only) | ok, tighten ci-image at job level |
| SHA pinning | All `uses:` are pinned to SHA. Repo setting `sha_pinning_required: false`, `allowed_actions: all` (`actions_perm.json`) | enable enforcement |
| Secrets for forks | No `pull_request_target`. `AUTO_MERGE_TOKEN` exists only in the owner PR path (`if: user.login == repository_owner`) | ok |
| Main push duplicate | One push run repeats all 13 jobs on the tree that the PR run already checked (same 26 misses). It is needed for the main ccache that every PR restores | optionally shorten the scope (question 2) |

## 5 Measures

The savings are measured where a source is given. "Estimate" means: derived from measured medians, not verified before phase 4.

### Quick wins (< 1 h)

| No. | Measure | Saving (basis) | Effort | Risk | Prio |
|---|---|---|---|---|---|
| M1 | ccache save only on push to `main`, also switch PTX to restore/save | -38 s median on the required check in 53 of 58 code PRs (`steps_build_pr_code.txt`), -1.49 GB cache per PR. Reuse occurred only in 5 of 77 branches | S | low: a re-push recompiles its own misses | 1 |
| M2 | Concurrency group per SHA for push (CI and CodeQL) | Push queue p90 370 s and max 1843 s drop away. 6 preempted main runs become 0. Main ccache is ready earlier | S | low: more parallel jobs at peak times. The PR queue is at p90 4 s, so there is room | 2 |
| M3 | Release: remove `cache-to/cache-from: type=gha` | -317 to -424 s per release (export in v0.69.1 and v0.70.0), no more budget pressure on CI | S | low to medium: the tarball build must hit the local builder cache (today 30-44 s) | 3 |
| M4 | `timeout-minutes` for all jobs (build 60, PTX 45, sanitizers 90, fuzz 60, CodeQL 30, gates and API jobs 10) | no time saved, but protection against 6-h hangs | S | low: each value is above the measured max (build 1526 s, PTX 1222 s, sanitizers 2441 s, fuzz 2488 s, CodeQL c-cpp 749 s, Real API 226 s, `job_max_dur.txt`). A cold cache after a CUDA bump was not measured | 4 |
| M5 | `retention-days: 1` for `imp-build`, ci-image `packages: write` at job level, repo setting SHA-pinning required | hygiene: 15 MB per run stays 90 days today | S | low | 6 |

### Medium term

| No. | Measure | Saving | Effort | Risk | Prio |
|---|---|---|---|---|---|
| M6 | clang-tidy-23 (and clang-format) into the `imp-ci` image, install step drops | -12 s median, push p90 41 s, on the required check. apt.llvm.org and the Ubuntu mirror drop out of the required path (mirror outages are noted in `install_llvm.sh`) | S-M (Dockerfile, ci-image run, digest) | medium: image gets larger, re-measure container init | 5 |
| M7 | PTX and sanitizers with `-j$(nproc)` instead of `build_jobs.sh` | Estimate: up to half of 204 s (PTX) and 387 s (ASan, p90 1944 s), with 4 instead of 2 jobs. Not on the merge path, but the last job in 23 of 71 PR runs | S | medium: ASan RAM unknown, so log `memory.peak` as in build | 7 |
| M8 | Pinned deps into the CI image (`/deps`, `FETCHCONTENT_SOURCE_DIR_*` as in the Dockerfile toolchain stage), remove build directory cache | -10 s restore per PR, -38 s post-save per push, -452 MB per push. Configure without cache is not measured and may partly eat the gain | M | medium | 8 |
| M9 | Fuzz corpus as artifact (retention 30 days) instead of cache, via `gh run download` from the last nightly | makes the corpus effective at all (today `Cache not found` 2 of 2) | M | low, needs `actions: read` | 9 |
| M10 | CodeQL c-cpp on push: paths filter (src, include, tools, scripts) or nightly only | -510 s analyze per main push in runner time, no wall gain for PRs | S | policy: alerts on main come later | 10 |
| M11 | Docs-only build without container (container via expression from an upstream change job) | -32 s of 94 s in 14 of 72 PRs | M | medium: required check name `Build` must stay. Check the behavior of an empty `container` beforehand | 11 |
| M12 | `-G Ninja` in the build job | not measured: a run with only hits takes 19 s under make, not much to gain there | S | low, needs a new build cache key | 12 |

### Structural

| No. | Measure | Saving | Effort | Risk | Prio |
|---|---|---|---|---|---|
| S1 | Sanitizer lane as CPU-only target (parser, tokenizer, schema, without libimp CUDA) | removes 194 of 705 objects from the ASan lane plus the nvcc share. Smaller ASan cache than 1.98 GB | L (CMake split, analogous to `IMP_FUZZERS_CPU_ONLY`) | medium: test coverage of the lane must stay the same | 13 |
| S2 | Lower header fan-in (misses per PR 2-390 determine build time 44-1116 s) | p90 of the merge path (746 s) depends on it | L | low, but much work | 14 |
| S3 | Larger runner (8-16 vCPU) only for `Build` | Estimate: compile core 278 s, p90 538 s, scales almost linearly with many misses. Single TUs such as `handlers_chat.cpp` (30.8 s) remain | S (YAML), but cost | cost, owner | question 1 |
| S4 | Shorten main push scope: on push only build with cache save and the path-controlled lanes. Gate jobs, mock and lint only on PR | runner time, push wall. No PR gain | S | policy: main status shows fewer checks | question 2 |

Not recommended: merging the 8 small gate jobs. They are not on the merge path, the PR queue is free (p90 4 s), and their value is the check names.

## 6 Diffs for the top 3

### M1: ccache save only on main (ci.yml)

```diff
       - name: ccache stats
         id: ccstats
         if: steps.changes.outputs.code == 'true'
         run: |
           ccache -s | tee "$RUNNER_TEMP/ccache-stats.txt"
           # Fail-open: an unparsable count saves.
           misses="$(awk '/^ *Misses:/ {print $2; exit}' "$RUNNER_TEMP/ccache-stats.txt")"
           echo "misses=${misses:-unknown}" >> "$GITHUB_OUTPUT"
-          # A PR saves from 30 misses on (a re-push recompiles less than that in under 90 s);
-          # main saves on any miss: its entry is the one every PR restores.
+          # main saves on any miss: its entry is the one every PR restores. PRs never save:
+          # 72 of 77 PR branches ran CI once, the save cost 38 s of `Build` and 1.49 GB of the 10 GB cache.
           save=true
           case "${misses:-unknown}" in
             *[!0-9]*) ;;
             0) save=false ;;
-            *) [ "$GITHUB_EVENT_NAME" != pull_request ] || [ "$misses" -ge 30 ] || save=false ;;
           esac
+          [ "$GITHUB_EVENT_NAME" != pull_request ] || save=false
           echo "save=$save" >> "$GITHUB_OUTPUT"
@@ ptx-fallback
-      - name: Restore ccache
-        if: steps.scope.outputs.run == 'true'
-        uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0
+      - name: Restore ccache
+        if: steps.scope.outputs.run == 'true' && github.event_name == 'push'
+        uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0
+        with:
+          path: /github/home/.ccache-ptx
+          key: ${{ runner.os }}-cuda13.5-ccache-ptx-${{ github.sha }}
+          restore-keys: |
+            ${{ runner.os }}-cuda13.5-ccache-ptx-
+      - name: Restore ccache (no save)
+        if: steps.scope.outputs.run == 'true' && github.event_name != 'push'
+        uses: actions/cache/restore@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0
         with:
           path: /github/home/.ccache-ptx
```

### M2: one concurrency group per main commit (ci.yml, codeql.yml)

```diff
 # ci.yml
 concurrency:
-  group: ci-${{ github.event_name }}-${{ github.ref }}
+  # push: one group per commit. A shared group queued 20 of 60 main Builds > 60 s (max 1843 s)
+  # and cancelled 6 pending main runs; GitHub keeps only one pending run per group.
+  group: ci-${{ github.event_name }}-${{ github.ref }}${{ github.event_name == 'push' && format('-{0}', github.sha) || '' }}
   cancel-in-progress: ${{ github.event_name == 'pull_request' }}

 # codeql.yml
 concurrency:
-  group: codeql-${{ github.ref }}
+  group: codeql-${{ github.ref }}${{ github.event_name == 'push' && format('-{0}', github.sha) || '' }}
   cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

### M3: no gha layer cache in the release (release-docker.yml)

```diff
       - name: Build and push
         id: build
         uses: docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc # v7.4.0
         with:
           ...
           tags: ${{ steps.meta.outputs.tags }}
           labels: ${{ steps.meta.outputs.labels }}
-          cache-from: type=gha
-          cache-to: type=gha,mode=max
+          # No type=gha: the export cost 317-424 s and CI's ccache entries evict it within hours
+          # (10 GB repo limit), so the next release built cold anyway. The tarball step below
+          # reuses this job's builder cache.
           sbom: true
           provenance: mode=max
@@
       - name: Build the release tarball
         uses: docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc # v7.4.0
         with:
           context: .
           target: bundle
           build-args: |
             IMP_BUILD_BENCH=ON
           outputs: type=local,dest=dist
-          cache-from: type=gha
```

## 7 Measurement before/after

Scripts are in `/mnt/data/scratch/pipeline_analysis/`: `fetch_jobs.sh`, `build_rows.sh`, `stats.jq`, `steps.jq`, `sample_logs.sh`, `sample_lanes.sh`.

| Measure | Metric | Before | Target |
|---|---|---|---|
| M1 | `Build` end_rel, code PRs, median / p90 (`build_end.txt`) | 435 s / 746 s | median about 38 s smaller |
| M1 | Step `Save ccache` on PR | success 53, skipped 5 (`pr_save_ccache_conclusions.txt`) | skipped on all |
| M1, M8 | `gh cache list`: oldest entry, number of entries | 20 min, 11 entries | older, more entries |
| M2 | Push jobs with queue > 60 s, cancelled push runs | 219 of 669 jobs, 6 runs | 0, 0 |
| M3 | Release step `Build and push`, step `Build the release tarball` | 1236-1967 s, 30-44 s | minus export (317-424 s), tarball unchanged |
| M6 | Steps `Install clang-tidy` and `Initialize containers` | 12 s, 32 s | 0 s, init about the same |
| M7 | PTX `Build the device library`, ASan build | 204 s, 387 s | smaller, plus `memory.peak` |
| M9 | Fuzz log `Cache not found` / corpus loaded | 2 of 2 empty | corpus loaded |

Procedure: collect at least 20 PR runs and 20 push runs after the merge, run the same jq programs, and compare medians and p90. Always report the compile core together with the misses, because the misses determine the spread.

### 7.1 Status after #3375 (2026-11-28)

Shipped in #3375 (owner decisions on section 8: S3 no, S4 path-driven, M10 nightly, nightly lag fixed): M1-M10, S4. M11 shipped and backed out in the same PR (saved 27 s, broke the container contract). Open: M12, S1, S2.

Early after-numbers, runs since the merge at 2026-11-28T00:35:22Z: 3 PR runs, 4 push runs, release v0.71.0 (`/mnt/data/scratch/pipeline_analysis/after/summary.txt`, script `after_3375.sh`). Medians and p90 wait for n >= 20 (BACKLOG-B-145).

| Measure | Metric | Before | After (early) |
|---|---|---|---|
| M1 | Step `Save ccache` on PR | success 53, skipped 5 | skipped 3 of 3 |
| M1 | `Build` end_rel, PRs | 435 s / 746 s (n=58) | 111, 533, 1999 s (n=3, misses not sampled) |
| M2 | Push jobs with queue > 60 s, cancelled push runs | 219 of 669 jobs, 6 runs | 0 of 52 jobs, 0 runs |
| M3 | Release step `Build and push`, step `Build the release tarball` | 1236-1967 s, 30-44 s | 1407 s, 37 s (n=1) |
| M6 | Step `Install clang-tidy`, `Initialize containers` | 12 s, 32 s | step gone, 27-39 s |

## 8 Open questions

1. Is a paid larger runner for `Build` wanted (S3)? Without it, the compile core stays at 278 s median and 538 s p90.
2. Must main push runs run all 13 jobs (S4), or is build with cache save plus the path-controlled lanes enough?
3. CodeQL c-cpp on every main push (510 s analyze), or a path filter or nightly (M10)?
4. Should the fuzz corpus persist across nights (M9)? Today it never does.
5. M7: Does `build_jobs.sh` really run with 2 on the runner? That is derived from the script and 16 GB, not logged. Add a log line before the switch.
6. The cron 03:17 UTC starts about 9 h later (12:32-13:02 UTC). Does that matter, or should the time move?
7. The comments on `ci_compile_slot.sh` in ci.yml describe a limit that the script no longer sets (it is now a pure `exec "$@"`). Should they be corrected in phase 4?
