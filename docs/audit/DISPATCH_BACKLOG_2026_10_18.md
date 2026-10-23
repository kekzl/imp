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

## 2026-10-18 Batch 12: owner question B-34 answered from the config

| ID | Status | Evidence |
|---|---|---|
| B-34 | WONTFIX | FP8 KV at head_dim 256 is never the default: `kv_cache.dtype = "auto"` (`imp.conf.example:210`) puts the hd256 families (Qwen3.5/3.6/3.8) on NVFP4 KV, the FP8 lists are hd128 families (SETTLED A2-4); no local container sets `kv_cache.dtype=fp8` (`maria-imp`: `--max-batch 8` only). Reopen when an operator runs `kv_cache.dtype=fp8` on an hd256 model |

## 2026-10-18 Batch 13: owner go R-080, measured scope

| ID | Status | Evidence |
|---|---|---|
| R-080 | DONE | per-family state structs dropped by measurement: of 47 `executor.h` commits since 2026-09-18, 4 changed fields only, 42 also method declarations; done instead: `engine.h` includes `exec/executor_fwd.h` (Engine ctor out of line), 11 `pre_dequant_*` TUs and `pre_dequant_internal.h` include `quant_pipeline.h` / `weight_caches.h` / `quant_scratch.h`, 5 TUs that dereference the executor include it directly; `make dev` after one real `executor.h` edit: 125 compiles / 150 steps / 68.8 s -> 78 / 104 / 47.4 s; `make dev-test` 13/13 |

## 2026-10-18 Batch 14: R-119.5 tree drafter, R-119.6 dropped

| ID | Status | Evidence |
|---|---|---|
| R-119.5 | DONE (opt-in) | `TokenRecycleTable::draft_tree` + `observe_rank` (slot hit rates from verify-row argmax), `speculative.recycle_tree`; PERF_LOG 2026-10-18 119.5: deterministic sum over prose / code / list 24.70 -> 24.99 (w8 d3) / 25.10 (w8 d7) tok/s, per prompt -6 .. +6 %; lossless 1193 bytes identical with 21 verifies; tests `TreeBranchesBelowTheRootByRankSum`, `TreeLearnsSlotHitRates`, `TreeKeepsWidthAndDepth` (mutants: rank cost dropped, hit counter dropped, both fail) |
| R-119.6 | DROPPED | shared-prefix mask only pays when compute limits before 64 rows: 196.6 ms/verify at 32 rows vs 204.1 at 64 |

## 2026-10-18 Batch 15: R-065 Q8_0 K128 kernel

| ID | Status | Evidence |
|---|---|---|
| R-065 | WONTFIX (NO-GO) | PERF_LOG 2026-10-18 row 65, gate-up M=512 under ncu: base 215.8 us; 512 threads 219.7; IMMA onto magic bits instead of I2FP 216.9; BM=32 column tail 165.0 + 56.4; each arm `MmqQ8Imma.*` 30/30 (tail arm 31/31 with its own bit-identity test); code reverted, ROADMAP row 65 closed with a reopen condition |

## 2026-10-18 Batch 16: R-049 verify cost, captured verify on small-M, B-36 found

| ID | Status | Evidence |
|---|---|---|
| R-049 | OPEN, economics measured | PERF_LOG 2026-10-18 row 49: batch 5's 2.0x was `imp-cli --bench` running verify eager (`tools/imp-cli/args.cpp` turns `speculative.capture` off); captured Qwen3-14B NVFP4 16-row verify 9.72 ms = 1.65x decode on the small-M route; next step is a DFlash drafter checkpoint (model download, owner) |
| R-049.1 | DONE (this PR) | `speculative.verify_smallm_captured` (default on): same image, NVFP4 n-gram 8.59 / 8.53 -> 7.73 / 7.75 ms/verify, tg 226.32 / 228.66 -> 249.27 tok/s, Q6_K 16 rows 31.47 -> 12.17; `make test-spec-fidelity` red with a per-stream rule (graph vs eager on different kernels), green with the per-step rule; `make verify-fast` OK |
| B-36 | OPEN (new) | Qwen3-14B Q6_K pp2..pp32 39.0-40.7 ms vs 5.83 ms decode: `gguf_source_fallback` dequantizes every weight per prompt chunk |

## 2026-10-19 Batch 17: R-049 DSpark block drafter

| ID | Status | Evidence |
|---|---|---|
| R-049 | DONE (opt-in) | PERF_LOG 2026-10-19 row 49: `speculative.dspark_model` + deepseek-ai/dflash_qwen3_8b_block7; Qwen3-8B-NVFP4 code 280.70 -> 457.96, list 281.63 -> 349.25, prose 282.25 -> 260.43 tok/s; FP8 code 170.89 -> 329.47 (vLLM 320.00); `DSparkDrafterTest.*` 2/2, `MtpGreedyIdentityTest.DSparkDoesNotChangeGreedyTokens` margins 0.686 / 0.394 nats (feature-tap mutant red), verify-fast OK, test-spec-fidelity OK |
| R-049.2 | OPEN (new) | propose 3.25-3.81 ms host wall per draft vs 5.8 ms NVFP4 verify; levers: target NVFP4 LM head, captured propose |

## 2026-10-20 Batch 18: R-049.2 DSpark propose

| ID | Status | Evidence |
|---|---|---|
| R-049.2 | DONE | PERF_LOG 2026-10-20: per-row FP8 drafter weights (own lm_head, not the target's) + captured propose, 3.07 -> 1.76 ms per draft; Qwen3-8B-NVFP4 prose at parity with nospec (no gain), code 531-591 vs 276-282 tok/s; `DSparkDrafterTest.*` 4/4 (graph test red under a stale-context mutant), greedy identity margins unchanged |

## 2026-10-22 Batch 19: B-36 dense Q6_K short prompts

| ID | Status | Evidence |
|---|---|---|
| B-36 | DONE | `gemm.q6k_imma_smallm`: Qwen3-14B-Q6_K pp2/8/16/32 39.4-41.4 -> 19.7-21.2 ms (A B B A, 2 runs per arm), pp512 74.8-75.9 ms both arms, PPL `ppl_4k.txt` chunk 32 11.4532 -> 11.4519 (chunk 0: 11.4543) |
| B-36b | OPEN (new) | nsys pp16: `mmq_imma_q6k_raw_kernel<32,false,true>` 18.5 ms of the forward; ncu down_proj 135 us = 540 GB/s, long scoreboard 2891 of ~6300 stall samples. Tried: full-step register prefetch 19.85 -> 19.55 ms at 146 regs (1 CTA/SM), with `__launch_bounds__(256, 2)` 22-23 ms; neither shipped |

## 2026-10-22 Batch 20: B-36b Q6_K small-M kernel

| ID | Status | Evidence |
|---|---|---|
| B-36b | DONE | `mmq_imma_q6k_k256_kernel` (one super-block per step, 2 cp.async stages): Qwen3-14B-Q6_K pp2/8/16/32 9.78/9.65/9.95/11.89 ms (was 19.7-21.2), tg 172 tok/s both; nsys pp16 Q6_K GEMM avg 65.9 -> 29.3 us. Steps: 4-B cp.async windows pp16 13.05 ms, 16-B windows 11.43, skip rows 16..31 at M <= 16 9.95. Refuted: 8 warps along N 10.44, 1 super-block per split 10.08. ncu after 4-B step: issue-bound (MIO throttle top, DRAM 48 %). PPL chunk 32 11.4477; 2 mutants red (window shift, M>16 half) |

## 2026-10-22 Batch 21: B-37 dense Q4_K and uncached Q8_0 short prompts

| ID | Status | Evidence |
|---|---|---|
| B-37 | DONE | Q4_K M <= 32 on BM=32 tiles + K-split by N and K (not under runtime.deterministic), warps with no live row skip their MMAs; uncached Q8_0 prompt rows M <= 32 via dp4a in place. Same window main vs branch, Qwen3.8-27B-Q4_K_M pp2/16/32/64 53.62/59.35/58.26/63.10 -> 23.03/29.46/38.39/54.52 ms, gemma-3-12b-it-Q4_K_M pp2/16/32 26.11/25.87/26.76 -> 9.31/8.68/10.74 ms; pp512 within main spread (Qwen 136.8-144.7 main, 144.8-145.4 branch). PPL chunk 32 Qwen 5.0758 -> 5.0841 (Q4_K alone 5.0705, Q8_0 dp4a = int8 activations +0.27 %), gemma 11.9341 -> 11.9285; chunk 0 bit-equal. 3 mutants red |

## 2026-10-23 Batch 22: B-38 Q8_0 raw IMMA for GDN projections

| ID | Status | Evidence |
|---|---|---|
| B-38 | DONE | `mmq_imma_q8raw_kernel` (8 Q8_0 blocks = 272 B per 256-K step, 16-B cp.async, 2 stages, K-split by N and K) for uncached Q8_0 rows 4..32; GDN gate/alpha/beta reuse one activation quantize. A/B alternating, 3 rounds x 5 reps, Qwen3.8-27B-Q4_K_M pp2/8/16/32/512 off 23.19/24.84/27.82/36.93/133.28 -> on 22.60/23.31/23.54/26.09/132.90 ms. M 2..3 stays dp4a (launch-bound, one launch less). PPL `ppl_4k.txt` chunk 32 5.0841 -> 5.0753, chunk 0 5.0744 both. 8 concurrent greedy chats coherent both arms, 0 error lines. Mutant (even blocks read word-aligned) red in 2 of 4 Q8Raw tests |

## 2026-10-23 Batch 23: B-39 UD GGUF decode

| ID | Status | Evidence |
|---|---|---|
| B-39 | DONE | Root cause: `fused_ffn_norm` gated on `is_dp4a_qtype(w_gate)` only, `gemv_gate_up_fused` read up_proj in the gate format; UD-Q4_K_M mixes gate/up types in 28 of 65 blocks (64 layers + MTP). Release v0.57.0 image: PPL `ppl_4k.txt` chunk 1 248320, stream 0 of 8 greedy chats `We<\|endoftext\|><\|im_start\|>...`; fixed: chunk 1 5.0855 (chunk 32 5.0424), 8 chats coherent, degen_suite --quick 52 checks 0 FAIL. IQ4_NL/IQ4_XS dp4a traits (`iq4nl_lut8`, `__byte_perm` codebook) in every dp4a family, `nvfp4_beneficial` IQ4 like Q4_K: NVFP4 decode cache 6240.9 + 2164.2 -> 2415.9 + 1351.4 MiB. Uncached dp4a types at M=1 (beta 0 / residual beta 1) via `dp4a_token_gemv` instead of a 170 MiB dequant per token. tg 45.13 -> 87.38 tok/s (Q4_K_M 75.40 -> 76.65). GgufRef IQ4 tests in the dp4a band (rms_rel 3.7e-3 - 4.2e-3), 2 mutants red (LUT halves swapped, scale bit offset) |

## 2026-10-23 Batch 24: B-40 UD GGUF prefill on IMMA

| ID | Status | Evidence |
|---|---|---|
| B-40 | DONE | `mmq_imma_gguf_smallm_kernel<Fmt>` (the B-38 Q8_0 kernel as a format template, Q8_0 bit-identical: Qwen3.8-27B-Q4_K_M PPL chunk 32 5.0753 both) for IQ4_NL (`iq4nl_lut8` into both B registers), IQ4_XS (8-B cp.async, 144-B rows), Q5_K (min term from the activation row sums); 32-row blocks on gridDim.x above 32 rows, Q5_K above 32 rows on the BM=128 raw kernel. Q6_K uncached rows 2..32 on the k256 kernel, FP16-tier Q4_K/Q5_K at M <= 32 on IMMA instead of dp4a dense (137 us avg). A/B `gemm.gguf_imma_smallm`, 2 rounds x 5 reps, UD-Q4_K_M pp2/16/32/512 82.7/84.1/88.0/161.5 -> 21.5/22.9/25.9/140.7 ms (IQ4_XS above 32 rows on `mmq_imma_iq4xs_bm128_kernel`: 464/536 -> 373/347 us per call, Q4_K BM=128 283 us). PPL chunk 32 5.0472 -> 5.0409, chunk 0 5.0401 -> 5.0417 (off arm keeps the Q6_K and FP16-tier routes). Large-M IMMA at 25-32 % of the int8 peak (IQ4_XS 373 us for 91 GFLOP): per-MMA scale FMAs; 128-K steps + ldmatrix are the next lever. Left on dequant: IQ3_S (4 tensors). 4 mutants red (Q5_K min sign, IQ4_XS scale bit, IQ4_NL even-block read, BM=128 sub-block scale swap) |
| B-40b | DONE | Q8_0 above 32 rows on `mmq_imma_gguf_bm128_kernel<BigQ8_0>` (68-B segments, 4-B cp.async) for N >= 1024: A/B 2 rounds, Qwen3.8-27B-Q4_K_M pp512 132.8 -> 127.9 ms, PPL chunk 0 5.0744 -> 5.0699. Without the N guard 133.3 -> 135.1 ms: GDN alpha/beta (N=48) as 4 BM=128 CTAs over 80 K-steps; Q8_0 on 32-row blocks 133.3 -> 137.1 ms. Mutant (block 0 read word-aligned) red |
| B-40c | DONE | Q5_K above 32 rows on `mmq_imma_gguf_bm128_kernel<BigQ5K>` (header + qh + 32 B qs per 64-K step): nsys 328 / 377 -> 298 / 341 us per call (beta 0 / 1). q - 16 enters the MMA in both Q5_K formats: with q = 0..31 the scale and min terms cancel in fp32, UD-Q4_K_M PPL chunk 0 5.0508 uncentered vs 5.0393 centered (old BM=128 kernel 5.0417, dequant 5.0401). Q4_K on the same template refuted: 283 / 348 -> 304 / 373 us, Q4_K_M pp512 +5.4 ms. Mutant (qh bit j ^ 1) red |

## 2026-10-23 Batch 25: B-41 GGUF decode rows, token GEMV, nextn MTP head

| ID | Status | Evidence |
|---|---|---|
| B-41 | DONE | nsys imp-server at 32 streams, Qwen3.8-27B-Q4_K_M: `gemm_qk_dp4a_dense_kernel<Q4_K>` 74 % (392 / 560 us per call: K=17408 fits 4 rows per weight pass). Q4_K decode rows on `mmq_imma_q4k_raw_kernel<32>` (K-split): 32-stream aggregate 288 -> 938 tok/s, UD-Q4_K_M 533 -> 875. Uncached Q8_0 / Q4_K M=1 on `dp4a_token_gemv`: tg32 76.65 -> 77.44, PPL chunk 1 5.0863 -> 5.0798. UD nextn block as MTP head (FP16 on upload, 928 MiB): 1 stream mtp_k 0 86.1 -> auto 95.5 tok/s, 69.6 % accepted, 22.2 ms per verify vs 11.7 ms per plain step. 8 concurrent chats coherent on both GGUFs. Measurement trap: `tools/analysis/conc_client.py` defaults to model Qwen3.8-27B-NVFP4-vllm and the server swaps to it: pass `MODEL_NAME`. Refuted: Q4_K BM=64 tiles for prompts above 32 rows, pp512 129.5 -> 144.4 ms |

## 2026-10-23 Batch 26: B-43 external MTP head for GGUF

| ID | Status | Evidence |
|---|---|---|
| B-43 | DONE | `speculative.mtp_head` -> `attach_mtp_head_file` (Qwen layout, fc [d_model, 2 d_model] checked) between load and mtp_auto in imp-server and imp-cli. Qwen3.8-27B-Q4_K_M + `Qwen3.8-27B-NVFP4-vllm/mtp.safetensors`, server 1 stream, 2 rounds: 77.2 / 77.5 -> 92.9 / 124.6 tok/s (acceptance 67.5-96.5 % by prompt set), 22 ms per verify; degen_suite --quick 51 of 52 (ple-isolation needs >1 concurrent request), 79 % of 1558 drafts accepted, 0 error lines. Mutant (d_model guard off) red in MtpExternalHead.AttachesAMatchingQwenHeadOnly. Not shipped from this batch: per-warp Q4_K header tables (q4k_raw<32> at M=31 53-58 -> 57-61 us), K-split target 768 CTAs (31 streams 1249 -> 1219 tok/s). 32 streams clamp to batch 31 at upload (KV reserve cap total/5 = 6516 MiB vs 890 MiB planned KV): 31 streams 1267 vs 32 streams 851-938 tok/s |

## 2026-10-23 Batch 27: B-44 multi-row GEMV for MTP verify

| ID | Status | Evidence |
|---|---|---|
| B-44 | DONE | `RowsOnce<QT>` (Q4_K, Q5_K, Q8_0, IQ4_XS): one block decode per thread, one dot per activation row, same arithmetic order; 4 weight rows per CTA at K >= 8192. Qwen3.8-27B-Q4_K_M + `speculative.mtp_head`, imp-cli tg600 greedy, nsys: Q4_K rows gate/up 40.0 -> 33.6 us (1.49 TB/s), down 45.8 -> 37.2 us, Q8_0 rows 21.6 -> 16.7 us, verify 22.9 -> 20.1 ms, tg 98.9 -> 112.6 tok/s; UD-Q4_K_M verify 22.1 -> 19.7 ms. Mutant (Q4_K min sign) red, 24 failures. Refuted: 4 weight rows per CTA at K=5120 (Q4_K 35.8 -> 40.0 us); FP16-tier GGUF weights at 2..4 rows through the rows kernel instead of cuBLAS (nvjet 0.747 -> 0.016 ms but verify GPU 19.99 -> 20.27 ms); NVFP4 copies of the MTP body (fc, q/k/v/o, MLP; 227.8 MiB): verify 19.74-19.92 -> 19.35-19.38 ms but acceptance 74.6 -> 65.7 %, tg 113.8-114.9 -> 105.4-105.6 tok/s; MLP only: 63.3 %, 101.9 tok/s |

## 2026-10-23 Batch 28: B-45 MTP draft attention

| ID | Status | Evidence |
|---|---|---|
| B-45 | DONE | `src/compute/mtp_attn_scan.cu`: warp per K row, warps split P*V, 512 threads; split-K (<= 32 CTAs per head) past 512 rows; smem opt-in once at the device maximum. `mtp_feed_batch`: Q, attention, o_proj, MLP for the last row only. Qwen3.8-27B-Q4_K_M + `speculative.mtp_head`, imp-cli greedy, nsys per call: ~670 rows 107 -> 17.7 us, 6.4k rows 280 -> 34 us (0.75 -> 0.09 ms per verify), drafts unchanged (334 of 448). `AttnScanMatchesSoftmaxReference`: FP64 reference, 1..20003 rows; mutants 16 / 6 red. Found on the way: a 64-row test workspace lowered the process-wide smem limit for later scans. Final tree, imp-cli greedy, 6.4k-token prompt, 2 runs per arm: prefill 1860-1893 -> 1552-1554 ms, decode 62.2-64.2 -> 74.4 tok/s, verify 30.5-31.4 -> 26.3 ms, 145 accepted both arms; short prompt 113.8-114.9 -> 114.5-115.4 tok/s; degen_suite --quick 52 of 52, 1758 of 1975 drafts accepted, 0 error lines. Refuted on the way: 1024 threads per CTA (register limit, launch refused) |
| B-45 note | REFUTED (CPU measurement) | Draft LM head over token ids < N (prefix cut of the 248077-row head): Qwen3.8 tokenizer, share of tokens with id < 98304 / < 151643: ppl_corpus_45k 99.62 / 99.62 %, mtp_forward.cu 99.92 / 99.93 %, README 99.47 / 99.47 %, German chat sample (407 tokens) 81.57 / 81.57 %; specials `<\|im_end\|>` 248046, `</think>` 248069 sit above every cut. A German chat loses 18 % of draftable tokens: only a frequency-selected row subset (FR-Spec style, multilingual corpus) stays open, ceiling ~0.44 ms per verify |

## 2026-10-23 Batch 29: B-46 IMMA wave tail

| ID | Status | Evidence |
|---|---|---|
| B-46 | DONE | `imma_wave_tail` (`mmq_q8_imma_internal.cuh`): dense BM=128 grid with a last wave under half full runs the leading column tiles plain (same bits) and the rest K-split into one more wave (split-K scratch capped at the planned 8 MiB), finalize per column block. Q4_K raw kernel + `gguf_bm128` (Q8_0, IQ4_XS, Q5_K). ncu Q4_K gate/up M=512: 544 CTAs, 3.20 waves, 1 CTA per SM (187 regs, 70 KB smem), issue slots 48.8 %. imp-cli pp512, 2 runs per arm: Q4_K_M 129.35-129.38 -> 124.16-124.26 ms (Q4_K tail alone 125.32-125.47), UD-Q4_K_M 140.42-140.48 -> 134.02-134.09 ms; pp4096 957.6 vs 959.7 / 1030.8 vs 1032.7 ms; PPL chunk 512 Q4_K_M 5.0727 / 5.0807 vs 5.0801, UD 5.0413 vs 5.0386. Tests `Q4KDenseWaveTailSplit` (head columns bit-identical, tail moves), `GgufLargeMWaveTailVsReference` (IQ4_XS, Q5_K, Q8_0); mutant (tail weight offset dropped) red in all 3. Found on the way: Q6_K above 32 rows dequantizes per call by design (half-MMA split measured slower), 16 calls x 74 us per pp512 forward |

## 2026-10-23 Batch 30: B-47 MTP draft LM head

| ID | Status | Evidence |
|---|---|---|
| B-47 | DONE | `mtp_draft_lm_head_nvfp4_`: a 16-bit head served as FP8 gets an NVFP4 copy (682 MiB) while MTP is on (`speculative.mtp_nvfp4_head`, `mtp_k != 0`), built before the source is freed and priced into the release; `lm_head_nvfp4_view` returns it first, the draft skips the FP8 view when an NVFP4 view exists. Qwen3.8-27B-NVFP4 imp-cli 600 tokens, 2 runs per arm: 102.31-102.90 -> 105.48-105.68 tok/s, verify 18.95-19.06 -> 18.27-18.31 ms, 291 -> 289 accepted. degen_suite --quick at max_batch_size=1: 52 of 52, 2874 of 3529 drafts accepted, 0 error lines. Refuted: NVFP4 for the main head too (118.7-118.8 tok/s, but PPL 5.1778 -> 5.2309 deterministic chunk 512) |
