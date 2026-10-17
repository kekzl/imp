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

## 2026-10-18 Batch 1: tests, warnings, ledger reconciliation

Builder: B-05, B-06 (subagent); orchestrator: B-09, B-11, B-28. Validator (fresh subagent): every item PASS, no anti-cheat finding, no U+2013/U+2014 in the diff; `make dev-test` 13/13, `make build` exit 0. Reconciliation: second subagent against later ledger sections, git history and code.

| ID | Status | Evidence |
|---|---|---|
| B-01 | DONE, no change | `test-e2e --gtest_filter='PrefixCacheE2ETest.*'`, Qwen3-4B-Instruct-2507 Q8_0: 4 passed, 2 skipped (hybrid, needs `IMP_TEST_MODEL_GDN`) |
| B-02 | DONE, no change | skip unreachable: `convert_nvfp4_to_mxfp4_cutlass` sets `linear_scales` or throws |
| B-03 | DONE, no change | skip unreachable: tests use head_dim 64/128/256, all instantiated; a supported head_dim without an instance FAILs (#2244) |
| B-04 | WONTFIX | `docs/DETERMINISM.md:43` known limit 5, workaround one process per context; dense half fixed in #1337 |
| B-05, B-06 | DONE dc8e2a03 | `-Wdangling-else` 6 -> 0, `-Wmissing-field-initializers` 14 -> 0 (`make build` log) |
| B-07 | WONTFIX | GCC false positive: `common_prefix_tokens` returns at `snapshot_boundary.h:88` for one sequence before the loop the warning names |
| B-09 | DONE 26c13dad | H-002, H-011, H-012, H-022, H-030, C-012, C-016, C-018 closed in `DISPATCH_CLEANUP.md` 2026-10-16/17 sections; rest fixed here: H-013 `IMP_FMHA_FA2` strict, H-015 `gptq.cu` helper |
| B-10 | DONE, no change | persistent `residual_meta_d_buf_` sized for `max_batch_size`, capacity stride (#1648, `engine_scheduler.cpp:1153-1166`) |
| B-11 | DONE 47e1004c | fallback through `vram_alloc_force`; `tools/alloc_allowlist.txt` 362 -> 361 |
| B-12, B-17, B-18 | DONE, no change | #2746 (routing pool failure fails init), #2748 (cached SM count / capability) |
| B-13, B-14 | WONTFIX | measured, not pursued: C-023 269.87 vs 269.91 tok/s, C-024 sites not on default tiers (`DISPATCH_CLEANUP.md:952-953`) |
| B-15, B-16, B-19, B-20, B-21 | DONE, no change | `DISPATCH_CLEANUP.md:951` H-030, `:812-844` H-012, `:894` C-016, `:992` H-011, `:1024` H-022 |
| B-26 | WONTFIX | split refuted on measurement, SETTLED S-32; `run_attention` 1175 code LOC under its 1216 pin |
| B-27 | WONTFIX | `q4k_hmma_enabled` removed fb2ca627; the ledger rates the other three "not a defect" |
| B-28 | DONE 144408f6 | `check-release.sh` section added; run on this branch: PASS(A) 0, PASS(B) 3, PASS(C) 0 serving allocations |
| B-30 | DONE, no change | AUDIT.md B7 (WSL2 VMM probe 24/24), B84, B41/B49/B79/B81 (reserve measured and remembered); O1/O2 rows stale |
| B-33 | BLOCKED | KPAR path only at K/16 > 512: no local model reaches it |
| B-34 | BLOCKED | `static_assert(ELEMS == 4)` at `attention_paged_fp8_multitok.cu:43`; needs an hd=256 FP8-KV model and the owner's word that the opt-in has users |

Questions for the owner:
- B-33: download a model whose MoE NVFP4 decode GEMV has K/16 > 512 (which one) to re-measure MaxL1?
- B-34: is FP8 paged KV at head_dim 256 used by anyone, so the multitok port is worth its cost?
