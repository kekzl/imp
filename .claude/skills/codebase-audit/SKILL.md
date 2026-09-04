---
name: codebase-audit
description: Use when auditing the imp codebase for structural debt, dead code, god-objects/files, duplication, flag sprawl, or deciding whether a cleanup is worth shipping - "structure audit", "tech debt", "is this still used", "dead code", "refactor for clarity", "should we split this file", "remove this flag", "file size gate red", "alloc sites gate red". Do NOT use for build/test mechanics (building-and-testing), perf/kernel work (benchmark-cuda / sm120-cuda-expert), or output quality (check-degeneration).
---

# Codebase audit & structural debt - imp

## Two hard rules

1. **A raw finding is a candidate.** Fan-out sweeps over-flag: on 2026-06-09 findings C1/C2/C4 were all REFUTED on inspection and several "dead flags/files" were live. Verify each against the code before reporting; the value is in the verification.
2. **Read `docs/audit/SETTLED.md` BEFORE generating hypotheses.** The 2026-07-29 audit generated first: 8 of 13 hypotheses were already-collapsed duplication (its section 15) and 5 facts in its own brief were stale (9 vs 16 architectures, C++20 vs C++23, `src/graph/` vs `src/exec/`; section 19). A candidate that contradicts a ledger entry needs that entry's ANCHOR disproven. Closing a finding updates BOTH the report status line and the ledger (`scripts/check-release.sh` section 1d fails CI on a stale open entry; F-6, S-10 ("still open" two days after #1209) and section G each went stale for days in the 07-29 campaign).

## Verification recipes

| Claim | Check | Trap that fooled a past audit |
|---|---|---|
| flag/symbol X is dead | `rg` across `src/ tests/ tools/` | `dump_tokens`, `force_bos`, `bench.generate` looked dead in `src/`, read in `tools/imp-cli/main.cpp` |
| config field never read | grep the leaf for a value-read in a conditional, excluding decl/parse/copy | a local var with the field's name hid the read (`fp8_auto_legacy`) |
| kernel/function dead | trace the call graph, not `<<<` sites; code-graph has NO launch edges right now (`ccg enrich` broken) so `rg` the launcher | `q4k_imma` CACHE was dead; `mmq_q4k_imma_tile`/`_reorder` live via `q4k_imma_prefill -> mmq_q4k_imma_gemm`; only ~30 LOC dead, not "~1200" |
| env var dead | grep `docker-entrypoint.sh` too | `IMP_KV_FP8` read nowhere in `src/`, translated by the entrypoint into `--kv-fp8` |
| god-file, split it | split on conflation or compile-time isolation, not size | `gdn.cu`, `gemm.cu`, `json_schema.cpp` are one domain each; the 7 `attention_paged*` share `attention_paged_common.cuh`. Precedent: `engine_scheduler.cpp` 2230 -> 1291 + `engine_prefill.cpp` + `engine_decode_pipeline.cpp` (#1782); `gdn_scan_chunkpar.cu` -> K1 + `gdn_scan_chunkpar_pass.cu` + `.cuh` at the 600-LOC kernel line; sampling family -> `executor_sampling.cu` (#1790); smallm -> `executor_gemm_smallm.cu` (#1793) |
| rewrite into a registry/template | need a concrete bug | the `weight_map.cpp` `if (!matched)` ladder is readable; a mis-mapped name = garbage output |
| imp.conf key inert | read in `src/` AND `tools/`? C-API bridge at engine init? | `server.prefix_cache` was parsed, unread, bridged (#636) |
| a log line proves the code path | compare the printed number against the code that uses it | a CORRECT log line named a number the code did not use (#1746/#1705); the sparse startup line printed the 16-block arithmetic while ACTIVE printed the real one (#1819) |
| a number in a `.cu` is wrong | pull the arithmetic into a pure function + CPU tests at both operating points | `src/exec/sparse_attn_geometry.h`, 7 tests, 4 fail on the original defect (#1819) |

## Workflow

0. Ledger first: `docs/audit/SETTLED.md`; memory axis: root `AUDIT.md`; structural: `docs/audit/DEBT_LEDGER_2026_08_21.md` (stale `[allow]` reasons, the "every gate is advisory" prehistory of the blocking gates). Re-derive every brief fact (SETTLED section F lists the five that were wrong, each with a counter-check).
1. Fan out (Explore agents, grep) AGAINST the ledger.
2. Verify each with the recipes; demote refuted ones immediately.
3. Write up: dated ledger in `docs/audit/` (`DEBT_LEDGER_2026_08_21.md`, `AUDIT_ARCH_2026_07_29.md`; older `structural_debt_*` in `docs/archive/`), append refutations with anchors to `SETTLED.md` ("Refuted (do not re-chase)"); memory findings go to root `AUDIT.md` (CONFIRMED / REFUTED / OPEN). Counts are evidence, not estimates.
4. Ship as small PRs off `origin/main`, `make build` + `make test-unit` + hooks (skill **building-and-testing**, **shipping-prs**); behavior-sensitive removals also run **check-degeneration**.

## Gates that shape audit findings (`scripts/ci_static_gates.sh`; all block inside `Build` since #1527; hooks run them in ~2 s)

| Gate | Tool | Rule |
|---|---|---|
| File size | `tools/check_filesize.py`, `tools/filesize_thresholds.toml` | code LOC (comments/blanks stripped): kernel `.cu` warn 500 / hard 600, TU 600 / 800, header 500 / 700. `[allow]` entries `{ code_loc = N, reason = "..." }` are a TWO-WAY ceiling since #1526 (drift either way fails; `--update` re-pins). Cost metric = recompile blast radius (one `.cu` = one ptxas TU), so a `#include`d `.cu` is charged to its includer since #1905 — `executor_attention.cu` read 542 as a file and 1279 as the TU. Rationale per file: `docs/audit/AUDIT_FILESIZE.md`. The `filesize` group also runs `check_determinism_sites.py`, `check_dead_inline_accessors.py`, `check_log_fatal.py` |
| Function size | `tools/check_function_size.py`, `tools/function_size_thresholds.toml` | the largest *body* in a file, warn 200 / hard 500 code LOC (p99 of 5830 functions is 199). Same two-way `[allow]` ceiling, keyed `path::signature`. Exists because a file's cohesion reason was covering an 884-line function inside an allowlisted "(c) one concern" file (#1905) |
| Test lanes | `tools/check_test_lanes.py --report` | own check since #1770; pinned no-lane count fails on growth (both 2026-08-25 "File size" reds were this pin) |
| Alloc sites | `tools/check_alloc_sites.py` + `tools/alloc_allowlist.txt`, `tools/check_alloc_pairs.py` | invariant I1 (one module talks to the driver); fails on a new site AND a stale entry (#1479 left `gdn.cu`'s entry behind). Counts SOURCE sites: the T2 slot pool removed ~96% of runtime allocations with a flat site count (`AUDIT.md` B34); runtime number via `make check-alloc-interpose` (`steady_state_allocations()`, never benchmark that build, `AUDIT.md` G16) |
| Dead inline accessors | `tools/check_dead_inline_accessors.py` (`make check-dead-inline`), `tools/dead_inline_allowlist.txt` | header-inline definitions with no caller |
| Launch guards | `tools/check_launch_guards.py` | post-launch check present |
| Kernel resources | `make kernel-resources`, `tools/kernel_resource_baseline.txt` | register/local-frame ratchet (#1549) |
| Doc citations | `scripts/check_doc_citations.py .` | a file split moves line numbers and kills `file:line` in living docs (#1782 -> #1783); `docs/audit/` itself is exempt |

## Priors: settled, do not re-flag

Canonical: `docs/audit/SETTLED.md` (collapsed duplication S-1..S-11, deliberate specialisations, hunted-and-absent negatives, findings that were themselves wrong). Not yet in the ledger:

- Two config systems (`RuntimeConfig` vs `ModelConfig::Overrides`) are deliberate; keys use the typed binder ladder `B/I/F/S(...)` (#626); the config surface lives in `src/core/config/*.h`.
- Structural-debt chain D1-D4 / C1-C8 archived in `docs/archive/README.md`, verdicts in `docs/archive/housekeeping_2026_06_13.md`. Audit #5 (`docs/archive/structural_debt_2026_07_07.md`, companion `docs/archive/vram_audit_2026_07_07.md`): server layer is the debt source, issues #888-#897; read its NOT-flagged list first. `src/memory/vram_owned.h` EXISTS since #1530 (once a hallucinated finding).
- Memory (`AUDIT.md`, 2026-07-29): "engine teardowns leak ~15 GiB" is REFUTED as a leak: every CUDA release works, WSL2/WDDM never returns a process's peak commitment. Traps: `cudaMalloc` success proves nothing (28 GiB at 22.6 GiB free; ~1530 vs ~237 GB/s tells), `cudaMemPoolAttrReservedMemCurrent` drops to 0 on trim, assert on `UsedMemCurrent`.
- Serving-loop classes closed by measurement: host turnaround (#1791), elementwise/residual fusion (#1793), prefill||decode overlap (#1792), H2D consolidation (#1834). Roadmap item 0(c) complete.
- A refactor that touches `exec/executor.h` needs a forced rebuild (`grep -rl exec/executor.h src tools tests | xargs touch && make dev`): stale objects segfault and look like the refactor.

## Red flags

- Finding contradicts a `SETTLED.md` entry and you have not disproven its anchor.
- Hypotheses came from the brief, not the tree.
- Deleting on "no references" without grepping `tools/` and `docker-entrypoint.sh`.
- Splitting a file because it is big rather than conflated.
- Rewriting a working ladder for elegance without a concrete bug.
- LOC of "dead" kernels from `<<<` sites instead of the call graph.
- A log line taken as proof of the code path.
