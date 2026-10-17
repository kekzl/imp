# DISPATCH_backlog 2026-10-18

Append-only. Earlier entries are never rewritten; corrections are new dated entries. Items: [`BACKLOG_2026_10_18.md`](BACKLOG_2026_10_18.md).

Rules: commit per item `<type>(<scope>): <summary> [BACKLOG-<ID>]`; no test removed, skipped or loosened; no threshold, tolerance, filesize or lint relaxation; no warning suppression; DONE only with command + result.

## 2026-10-18 Phase 0: inventory

| Decision | Reason |
|---|---|
| Logs live in `docs/audit/` (`BACKLOG_*`, `DISPATCH_BACKLOG_*`, `REPORT_BACKLOG_*`), perf entries append to `docs/audit/PERF_LOG.md`, audit entries go into this file | root `AUDIT.md` is the memory-subsystem ledger; `docs/audit/` already holds inventory + dispatch pairs (`DEBT_INVENTORY_2026_10_15.md`, `DISPATCH_CLEANUP.md`) |
| Branch `auto/backlog-2026-10-18`, pushed as backup; merges to main only through a PR | mission: never push to main |
| R-054, R-055, R-080 BLOCKED (needs-human) | external account / release publishing / an owner decision the ROADMAP row names |
| W-01..W-05 WONTFIX | env-gated modes, debug-only aborts, on-demand benchmarks, C-017 measured |

Questions for the owner:
- R-054: register a self-hosted GPU runner for this repo (which machine, which labels)?
- R-055: which binary artifact (static `imp-server` tarball, .deb) and where is it published (GitHub release asset)?
- R-080: move model-specific block state out of `executor.h` against SETTLED B, yes or no?

Inventory counts: scout A todo-comment 1 (real markers 0), skipped-test 13, ci-failure 0 (15 newest main runs green), warning 17 rows; scout B roadmap 18, audit-open 38, issue 0, pr 0. verify-fast on main a6492284 before the run: all PASS (decode -7.80 %, prefill +7.81 %, PPL 0.000 %, server c8 -0.41 %).
