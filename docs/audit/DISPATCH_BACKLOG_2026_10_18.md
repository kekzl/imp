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

## 2026-10-18 Batch 2: tidy sweep, kernel templates, encoder sources

Builders: B-23 (subagent, SASS gate with a control mutation that the compare caught), B-08 + B-22 (subagent, own build dir); orchestrator: B-29 item 8. Decision: one commit 06fdf269 for the four items, the parallel builders edited the same files and splitting hunks needs interactive staging.

| ID | Status | Evidence |
|---|---|---|
| B-08 | DONE 06fdf269 | 444 TUs swept: parentheses 4110, contains 244, min/max 110, string concat 10, redundant inline 4; 0 findings left for the five checks |
| B-22 | DONE 06fdf269 | 7 modernize checks enabled (170 fixes), 0 findings; 10 checks left off with counts in `.clang-tidy` |
| B-23 | DONE 06fdf269 | C-004..C-007 merged; 3 files 1732 -> 1586 LOC; validator SASS: 5 renamed instances identical to their old kernels |
| B-29 | IN_PROGRESS | split into B-29a..f; B-29d DONE |
| B-29d | DONE 06fdf269 | nomic-embed-text-v1.5 Q8_0: 84 sources freed, Q8 release pool 128.0 -> 0.0 MiB; `EncoderEmbedTest.NomicBertEmbedsMatchReferenceStructure` passed |
| B-29b, B-29c | BLOCKED | Nemotron MTP checkpoint and gemma-3 Q8_0/Q6_K not local |
| B-29f | BLOCKED | item 24 is a reserve-policy decision; item 14 partly covered by #2712 |

Validator (fresh agent, opus): SASS of 1123 kernels HEAD vs tree, 1114 identical, 9 differ only in the `std::max` select form (FSETP.GT + FSEL vs FSETP.GEU + !FSEL, same value for every input incl. NaN and signed zero): `mtp_attn_kv_scan(_rows)`, `apply_min_p`, `softmax_max`, `topk_partial(_rows)`, `topk_finalize(_rows)`, `absmax_kernel` (546 -> 498 SASS lines). About 230 host sites reviewed, 0 non-equivalent. Anti-cheat: `tests/` untouched, 0 NOLINT, 0 suppressions, alloc allowlist total 361 unchanged (one site moved into `Model::free_source_weight`), em dashes 15 added vs 19 removed on reflowed lines, none new.

Gates on the tree: `make dev-test` 13/13, `make preflight` 0 failures, `make test-gpu` 3749 ran 0 failed, `make verify-fast` all PASS (decode tg128 -0.50 %, prefill pp512 +17.53 %, PPL 0.000 %, server c8 +2.95 %, peak VRAM -0.08 %).

Question for the owner:
- B-29f: keep the five stacked reserves (library 3900 + 4 % + 256 + 10 % floor + kv_fraction 0.8) or collapse them into one measured margin?

## 2026-10-18 Batch 3: LM-head SfAtom, MXFP4 scales, topk, FFN graph churn

Orchestrator: B-29a, B-24; validator (fresh agent): both PASS, `make dev-test` 13/13, `make preflight` 0 failures, anti-cheat clean.

| ID | Status | Evidence |
|---|---|---|
| B-29a | DONE 14b805e7 | Qwen3-14B-NVFP4 `gemm.nvfp4_lm_head=on`, `max_batch_size` 4: base "+47480 KiB SfAtom scales", fix borrows them; two concurrent temp-0 chats: base 1 == base 2 == fix 2 byte for byte, fix 1 differed by batching (n>1 CUTLASS head vs n==1 GEMV per step) |
| B-29e | BLOCKED | no dense MXFP4 GGUF local; Qwen3.5-4B-mxfp4 is GDN and decodes through the FP16 fallback (8020 MiB), gpt-oss-20b is MoE |
| B-24 | DONE (this commit) | `docs/audit/PERF_LOG.md` 2026-10-18: instantiates 396 -> 324, 72 updates, -53.7 ms; `FfnGraphCacheGpu.EvictedExecIsUpdatedForTheNextBucket` passes |
| B-25 | OPEN, measured | topk_partial + topk_finalize 67.6 us per decode step, 2.3 % of GPU time at top-k 40 (PERF_LOG); the frame is a dynamically indexed array, not spills. Lever: radix-select partial, at most about 1.9 % decode |

Validator notes kept: updates cost 0.12 ms each before and 0.02 ms after (unexplained, not gated); `spare_` keeps one exec per layer beyond eviction (graph execs without memory nodes).

PR #2777 merged batches 1-3 (809ad716). Continued on `auto/backlog-2026-10-18b`.

## 2026-10-18 Batch 4: verify ratio, InternVL video

| ID | Status | Evidence |
|---|---|---|
| B-31 | DONE, no change | re-measured `imp-cli --bench --bench-pp 512 --bench-reps 5`: Qwen3-14B Q6_K 14.14 ms/verify at 4.12 tok/verify vs 5.8 ms decode step (2.4x), NVFP4 12.43 ms at 2.00 tok/verify vs 5.95 ms (2.1x); the ledger's 51.97 ms / 8.4x and the 3.4x quant gap are gone |
| B-32 | WONTFIX | the ledger closes it itself: "Not chased", the question dissolved when the drop turned out a priced trade (`DEBT_LEDGER_2026_08_21.md:705-707`) |
| B-35 | DONE (this commit) | server: InternVL `video` part = one image per frame, `Frame{i}: ` lines; golden rows written from HF `replace_video_token`, not from imp output; InternVL3.5-2B e2e: "A cat and red double decker buses.", 1077 prompt tokens; validator PASS (logic), gates fixed after its run (API_FEATURES 301 -> 300 lines, format, stale 400 text) |

Decision (B-35): frames use the image path (448, ImageNet mean/std) as OpenGVLab's `load_video(..., max_num=1)`, not the checkpoint's `video_preprocessor_config.json` (384 x 384, CLIP stats), which cannot give the 256 tokens `replace_video_token` writes per frame. `imp-cli --video-frames` stays Qwen3-VL only.
