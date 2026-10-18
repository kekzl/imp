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

PR #2778 merged B-35 (6d1c883d). Continued on `auto/backlog-2026-10-18c`.

## 2026-10-18 Batch 5: roadmap measurements (R-077, R-065, R-049)

| ID | Status | Evidence |
|---|---|---|
| R-077 | DONE (this commit) | Gemma-4-26B-A4B Q4_K_M, `imp-cli --bench --bench-pp 512 --bench-reps 3`, ngram off: overlay tg 256.48 / pp512 16221.60 tok/s, PPL 231.1655; MMVQ (`diagnostics.no_nvfp4_decode_cache=true`) 116.85 / 13904.67, PPL 223.7162. Overlay stays; every kernel the row names is reachable through a knob or MTP, none retired (deleting a knob's path is a user-facing removal); resolver log corrected |
| R-065 | OPEN, re-measured | `tools/roofline/inventory/ncu_cell.sh q8-8b pp512` (clocks not locked on this host): `mmq_imma_q8k128_kernel` grid 32x4 117.3 us / tensor pipe 30.9 %, grid 96x4 215.4 us / 31.6 %; `mmq_imma_kernel<32>` grid 8x16 34.8 us / 12.6 %; nsys pp512: K128 kernel 51.2 % of prefill GPU time |
| R-049 | OPEN, verify cost measured | Qwen3-14B NVFP4 11.87-12.72 ms per 2-token verify (k 1/3/7/15 all draft 1 on the bench history) vs 5.94 ms decode (168.23 tok/s); Q6_K 14.14 ms per 4.12-token verify; next sub-item: forced-width verify harness for widths 8..16 |

Finding beside R-049: on Qwen3-14B-NVFP4 a 2-token n-gram verify costs 2.0x a decode step, so n-gram at width 2 gains nothing there (6.1 vs 5.94 ms per token).

## 2026-10-18 Batch 6: radix top-k (B-25), speculation under offload (R-119.1, R-119.3), L-row splits

Builder (subagent): B-25; it paused twice for this session's GPU lock (holder `imp-05m`) as the rules require. Validator (fresh agent, opus): B-25 PASS (dispatch, tie-break proof against legacy, smem 5176/5680 B static, stack 0, launch checks, test registered in `test-compute`); offload give-up PARTIAL: the post-verify give-up (acceptance-poor / draft-poor) was still live under offload, now gated too and re-measured.

| ID | Status | Evidence |
|---|---|---|
| B-25 | DONE (this batch) | topk partial + finalize 66.77 -> 18.76 us per decode step; `TopkRadixSampler.*` 3 tests, 588 configs x 12 seeds byte-compared vs legacy; mutant (tie-break `r = ix[p]`) fails it; 45/45 sampling tests on the build |
| R-119.1 | DONE | Qwen3-14B Q6_K `--gpu-layers 0`: tg 188.6 ms/token, pp 4/16/64 = 189.70/193.53/189.71 ms; `--gpu-layers 20`: 97.1 ms, 116.93/109.40/111.76; resident: tg 5.7 ms, pp 64 44.86 ms. t(64)/t(1) = 1.0 at full streaming: GO |
| R-119.3 | DONE (this batch) | n-gram ran under offload already (k=16 default; bench history 3.99 tok/verify at 190.54 ms vs 188.6 ms decode); both give-up rules gated with `!offload_mgr_`; code rewrite 5.30 -> 6.83 tok/s, prose 5.28 -> 5.29 |
| R-119.2 | OPEN | real-text acceptance: code rewrite 5 verifies x 10.00 tok/verify, prose 0 drafts (suffix drafter finds nothing); `speculative.k` 3/7/15 changes nothing on the bench history (the suffix drafter sizes the draft) |
| R-112 | OPEN | split below; 112.1 (decision measurement) next |

Plan agent split (verified anchors), kept here as the work queue:

| Sub-item | Goal | Anchor |
|---|---|---|
| 112.1 | share of decode time cold-page fetches would cost at `sparse_topk=4096` (32 KiB slots, mapped pinned gather) | `src/exec/executor_attention_decode.cpp:142`, `tools/analysis/h2d_gather_probe.cu:50` |
| 112.2 | spill carries key min/max so spill and sparse decode can coexist | `src/memory/kv_host_spill.cpp:11`, `src/runtime/engine_init_resolver.cpp:910-912` |
| 112.3 | mapped spill buffer, GPU-only gather (no mailbox) | `src/memory/kv_host_spill.cpp:116` |
| 112.4 | KVCacheManager owns cold pages of live sequences | `src/memory/kv_cache_manager.h:358` |
| 112.5 | selection scores cold pages, gather into a VRAM ring | `src/exec/sparse_attn_select.cu:654` |
| 112.6 | full-KV readers restore cold pages in bulk | `src/exec/executor_attention_decode.cpp:119` |
| 112.7 | end to end: more concurrent long sessions than VRAM holds | `tools/analysis/serving_sparse_ab.sh` |
| 119.4 | lift multi-candidate verify caps (width 8, 17 rows) under offload | `src/runtime/engine_spec_ngram.cpp:418-421`, `src/runtime/engine_spec_capture.cpp:63` |
| 119.5 | best-first tree drafter from suffix votes and recycling | `src/runtime/engine_spec_ngram.cpp:424` |
| 119.6 | shared-prefix tree mask, only if compute limits before 64 rows | `src/runtime/engine_spec_ngram.cpp:727-760` |

No dense model > 32 GB is local; `Qwen3-14B-Q6_K.gguf` at `--gpu-layers 0` stands in (streams about the overflow of a 44 GB dense model).

PR #2779 merged batches 5-6 (5e10de3c). Continued on `auto/backlog-2026-10-18d`.

## 2026-10-18 Batch 7: token recycling under offload (R-119.2)

| ID | Status | Evidence |
|---|---|---|
| R-119.2 | DONE (this commit) | recycle_width sweep under `--gpu-layers 0` (PERF_LOG 2026-10-18): width 4 best, prose 5.30 -> 6.13, code 6.85 -> 8.87 tok/s; `apply_offload_recycling_default` (`src/runtime/offload_spec_defaults.h`, called first in `init_resolve_sparse_attention_`) turns recycling on (width 4) under offload unless set; `OffloadSpecDefaults.*` 3 CPU tests; resident model: no auto line; lossless: server byte A/B identical (1193 bytes, 19 verifies) with sparse pinned off in both arms |

Decision: under offload recycling outranks sparse decode (recycling refuses the sparse key min/max pool, `sparse_minmax_refusal`); a streamed pass is PCIe-bound, attention cost is not the limit there. An operator who wants sparse sets `speculative.token_recycling=false`.

PR #2780 merged batch 7 (09797f27).

## 2026-10-18 Phase 3: final sweep and report

| ID | Status | Evidence |
|---|---|---|
| R-119.4 | OPEN, measured | recycling width 8 below width 4 under offload (prose 6.09 vs 6.13, code 8.43 vs 8.87 tok/s): wider candidate rows wait for a better drafter (119.5) |
| B-29 | BLOCKED | a and d DONE; b, c, e, f blocked on models or the reserve decision |

Final gates on main 09797f27: `make build` exit 0, `make test-gpu` 3756 ran 0 failed, `make dev-test` 13/13, `make preflight` 0 failures; inventory re-run: 0 comment markers, 0 open issues / PRs. Report: [`REPORT_BACKLOG_2026_10_18.md`](REPORT_BACKLOG_2026_10_18.md).

## 2026-10-18 Batch 8: cold KV pages decision (R-112.1)

Measurement only, no builder: selection ids dumped by a scratch patch (never committed), gather cost from `tools/analysis/h2d_gather_probe.cu` (now takes a slot size, picks distinct destination slots). Validation: the first probe run picked duplicate slots, concurrent copies into one slot mixed and failed verify; rerun with distinct slots, 0 verify failures.

| ID | Status | Evidence |
|---|---|---|
| R-112.1 | DONE | churn 110.28 / 110.71 / 116.03 of 256 blocks per layer per step (Qwen3-8B 32k, Qwen3-4B-2507 32k / 64k); ring 4x at 32k: 760.6 misses = 24.9 MB per token, +10.4 % (49.7 GB/s floor) to +22.4 % (per-layer gather) of the 4.820 ms sparse step |
| R-112.2-112.7 | WONTFIX | NO-GO from 112.1: `--kv-nvfp4` saves the same 50 % without PCIe, PCIe bytes scale with sessions; reopen on a sticky selection that passes the NIAH A/B |

Record: [`docs/plans/2026-10-18-kv-cold-pages-decision.md`](../plans/2026-10-18-kv-cold-pages-decision.md).

## 2026-10-18 Batch 9: owner answers R-054, B-29f

Orchestrator only (one subagent mapped the reserve chain read-only). Validation: base image from origin/main vs this tree, 10-model sweep at batch 1 and 16, greedy identity, 10 min soak, mutant on the ceiling fix.

| ID | Status | Evidence |
|---|---|---|
| R-054 | WONTFIX | owner 2026-10-18: no self-hosted GPU runner; roadmap row 54 closed |
| B-29f | DONE | `vram.reserve_floor_pct` 10 -> 0 (floor = 4 % headroom), `vram.kv_fraction` 0.8 -> 1.0; 9 of 10 models unchanged, Qwen3.6-35B-A3B-NVFP4 ceiling 118128 -> 143184 tokens; soak showed the warmup re-plan never raised a pool with key min/max (8396 kept, 19063 computed): `raise_ceiling` rebuilds it now, soak 19063 blocks, growth 14 MiB, 0 errors |

Test change: `RaiseCeilingKeepsTheCommitAndGrowsPastTheOldCeiling` asserted the refusal with key min/max on; it now asserts the rebuilt pool (moved pointer, stride, backed grown block), mutant without the swap fails it. `VramKnobDefaultsArePinned`, `KvFractionScalesKvPool`, `ReserveFloorPctScalesReserve`, `VramKnobsAreClamped`, `test_config.cpp` defaults: pinned to the new defaults, same contracts; new `VramReserveFloor.*` in test-core.

Record: [`docs/plans/2026-10-18-one-measured-vram-margin.md`](../plans/2026-10-18-one-measured-vram-margin.md).

PR #2783 merged batch 9 (3bb3864d).

## 2026-10-18 Batch 10: 64 verify rows under offload (R-119.4)

Orchestrator only. The batch-7 width-8 arm ran at depth 1: `width * (1 + depth) <= 17` capped it, so 119.4 had not been measured at full rows.

| ID | Status | Evidence |
|---|---|---|
| R-119.4 | DONE | `recycle_mc_shape` (`src/runtime/offload_spec_defaults.h`): 64 rows under offload, 17 resident (unchanged); `kOffloadRecycleWidth` 4 -> 8; PERF_LOG 2026-10-18 row 119.4: prose 6.12 -> 7.22, code 8.90 -> 9.80 tok/s; server `--gpu-layers 0` greedy, default vs speculation off: 1169 bytes identical; log `auto -> on, width 8 (layer offload)` |

Tests: `McShapeResidentKeepsTheSeventeenRowBudget`, `McShapeOffloadUsesSixtyFourRows` (test-core); `OffloadTurnsRecyclingOnAtWidthEight` pins the constant; `OperatorKeysWin` moved its explicit width 8 -> 2 (8 is now the default and would not show the operator key winning).

## 2026-10-18 Batch 11: owner answer R-055 (A: tarball, CUDA libraries bundled)

| ID | Status | Evidence |
|---|---|---|
| R-055 | DONE | Dockerfile `bundle` stage + `scripts/bundle_release.sh`: libexec/ binaries, every ldd-resolved library except `libcuda.so.1` (glibc and loader included, the binaries need GLIBC_2.43 via `sqrtf`), bin/ launchers run the bundled loader; 824 MB, 46 libs; `release-docker.yml` checks, cosign-signs and attaches it; `debian:12` and `ubuntu:22.04`: no library outside lib/; `debian:12 --gpus all` Llama-3.2-3B Q8_0 greedy 713 bytes identical to `imp:test`, request 0.286-0.293 vs 0.285-0.289 s |
