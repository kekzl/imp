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
## 2026-10-23 Batch 31: B-48 GDN alpha/beta in verify

| ID | Status | Evidence |
|---|---|---|
| B-48 | DONE | `try_gdn_alpha_beta_narrow_` drops the `cur_spec_verify_` exclusion (#1977 measured 10 of 3072 outputs 1 FP16 ulp from cuBLAS; verify already ran cuBLAS, not the M=1 decode kernels). Qwen3.8-27B-NVFP4-vllm imp-cli 600 tokens, main vs branch: 102.31-102.90 -> 105.35-106.08 tok/s, verify 18.95-19.06 -> 18.52-18.65 ms; nsys: `gemm_f16_narrow_smallm_kernel` 8256 launches over 149 verifies, the 64x8 nvjet split-K kernel gone. degen_suite --quick at max_batch_size=1: 52 checks, 0 FAIL, 1 skip (prefix-snapshot resend), 2331 of 2809 drafts accepted, 0 error lines |

## 2026-10-23 Batch 32: B-49 producer quantize in verify

| ID | Status | Evidence |
|---|---|---|
| B-49 | DONE | `smallm_producer_xq_` drops the `cur_spec_verify_` exclusion. Qwen3.8-27B-NVFP4-vllm imp-cli 600 tokens, main vs branch: 102.31-102.90 -> 104.82-105.18 tok/s, verify 18.95-19.06 -> 18.54-18.60 ms, accepted 291 / emitted 597 in both arms (same text). degen_suite --quick at max_batch_size=1: 52 of 52, 2832 of 3491 drafts accepted, 0 error lines. Refuted on the way: `residual_beta1_nvfp4_ok_` in verify as well (104.54-105.05 tok/s, flat; matches #1793) |
| B-49 note | REFUTED | Per-row FP8 copies of the MTP body (fc, q/k/v/o, MLP; 405.2 MiB, `gemv_fp8_rowscale`), 2 runs per arm: Q4_K_M + external head 114.93-115.55 vs 116.03-116.35 tok/s (accepted 330 vs 334), NVFP4 106.11-106.36 vs 105.25-105.77 tok/s (288 vs 294); verify only 0.1-0.5 ms shorter, acceptance -0.8..-2 pp. Superseded at depth 7 on the bound verify lane by B-76 (the mtp_k=2 chain stays FP16) |

## 2026-10-23 Batch 33: B-50 one LM head pass per verify

| ID | Status | Evidence |
|---|---|---|
| B-50 | DONE | `InferenceState::skip_lm_head`: the single-stream and batched verify set it (off under `diagnostics.spec_capture_fidelity`, which compares the forward's logits); forward_logits leaves `logits_out` empty, softcap and the final-logit dump skip an empty tensor. nsys before: 709 `fp8_head_mma` launches over 325 verifies, 326 pairs 6 us apart. imp-cli 500 tokens, 4 prompts, NVFP4: 104.77 / 105.32 / 107.39 / 150.82 -> 111.81 / 111.91 / 114.11 / 159.80 tok/s (mean 117.08 -> 124.41), drafted/accepted identical per prompt, verify 16.4 ms. 600-token prompt: Q4_K_M + external head 116.74 -> 119.52, UD 104.28 -> 107.24 tok/s. degen_suite --quick max_batch_size=1 and mtp_k=2 multi-batch: 52 checks, 0 FAIL each; 4 streams with batch_verify=true: 330.9 vs 329.1 tok/s on main (accepted 0 on both arms, pre-existing). Refuted on the way: `speculative.verify_smallm_captured=false` (batched W4A16 GEMV) mean 109.33 vs 117.08 tok/s |

## 2026-10-24 Batch 34: B-51 upload KV reserve before a slot

| ID | Status | Evidence |
|---|---|---|
| B-51 | DONE | #2860: `upload_fitting_batch` judges each batch with KV at the batch-1 floor, `upload_fit_reserve_bytes` gives KV the rest up to the cap. Qwen3.8-27B-Q4_K_M, max_seq_len 131072, 32 slots: "clamped 32 -> 31" -> "trimmed 6516 -> 6477 MiB"; 32 streams x 300 tok 861.2 / 859.5 (31 slots) -> 1296.3 / 1294.4 tok/s. Qwen3.8-27B-NVFP4 64 slots vs 51: 1722.1 / 1720.3 -> 1797.1 / 1800.9 tok/s. `UploadReserve.*` 8 of 8, mutant 3 red |

## 2026-10-24 Batch 35: B-52 verify GEMMs at the roofline

| ID | Status | Evidence |
|---|---|---|
| B-52 | WONTFIX | #2861: PDL-inflated nsys durations; exclusive frontier time 1.34-1.67 TB/s per shape (gate/up 17408 x 5120: 41.9 us duration, 30.4 us exclusive). Verify forward 12116 us wall, 11999 us busy; between verifies 836 us idle under nsys = the profiler's +0.85 ms/verify (16.7 native vs 17.6) |
| B-52 note | REFUTED | smallm v2 split-K at N=5120 with partial planes bounded to M rows: 4-prompt greedy ms/verify +0.1..+0.6 vs one stripe; the M bound alone identical to main |

## 2026-10-24 Batch 36: B-53 Q5_K 128-K steps

| ID | Status | Evidence |
|---|---|---|
| B-53 | DONE | #2863: step width S is a format-policy template parameter; Q5_K S=4. Qwen3.8-27B-UD-Q4_K_M pp512 133.5-133.6 -> 128.0-128.2 ms, pp2048 509.4 / 510.2 -> 487.7 / 488.1 ms, PPL `ppl_slice_12k.txt` 5.0068 both arms; Q4_K_M and gemma-3-12b-it-Q4_K_M flat |
| B-53 note | REFUTED | IQ4_XS S=4 +1.9 % UD pp512, Q8_0 S=4 +1.6 % Q4_K_M pp512, `#pragma unroll 2` on the sub-block loop loses the gain, s32 MMA seeded with 0x4B400000 (FADD instead of I2FP) +0.8 % UD / +0.4 % Q4_K_M |

## 2026-10-24 Batch 37: B-54 IQ4_NL on the BM=128 kernel

| ID | Status | Evidence |
|---|---|---|
| B-54 | DONE | #2864: `BigIQ4NL<2>`. Per launch at pp512 gate/up 524.0 -> 355.1 + 39.5 us, down 497.8 -> 371.5 us; UD-Q4_K_M pp512 127.64 / 127.77 / 127.89 -> 126.85 / 127.47 / 127.14 ms (S=4 127.25-127.40), PPL 5.0068 both arms |

## 2026-10-24 Batch 38: B-55 IQ3_S native, Q3_K word loads, IQ4_NL K-parallel

| ID | Status | Evidence |
|---|---|---|
| B-55 | DONE | IQ3_S `DequantTraits` + `FmtIQ3S` / `BigIQ3S<2>` (K % 512 == 0), NVFP4 decode copy off (4746 -> 4555 MiB, KV 7167 -> 7922 blocks). Per launch: IQ3_S prompt GEMM 290 + 274 us (dequant + cuBLAS) -> 406 gate / 457 down at pp512, M <= 32 48.8 us; verify rows Q3_K 95.4 -> 38.6 us, IQ3_S 37.1 / 40.6 us. UD-Q4_K_M tg128 87.97 / 87.97 / 87.96 -> 88.66 / 88.65 / 88.55, pp512 +0.42 % (noise), PPL `ppl_slice_12k.txt` 5.0068 -> 5.0059, degen_suite 52 checks 0 FAIL |
| B-55 verify | DONE | FP16-tier GGUF weights at 2..32 rows run the in-place kernels for every source qtype (was Q4_K / Q5_K only): UD-Q4_K_M MTP verify had 3-4 cuBLAS FP16 GEMMs per step (116 us per 17408 x 5120), kernel sum per step 16.67 -> 16.02 ms; the freed NVFP4 budget had grown the FP16 cache by one tensor (2600 -> 2770 MiB). MTP imp-cli math prompt (202 verifies, 297 accepted both arms) 136.69 / 136.01 -> 142.48 / 142.79 tok/s, 4-prompt mean 112.27 -> 115.72; Qwen3-8B-Q8_0 tg128 +0.08 %, Qwen3.8-27B-Q4_K_M -0.13 % (nsys kernel sum 6523.1 -> 6515.4 ms) |
| B-55 note | MEASURED | IQ4_NL `kPreferKpar` alone: tg128 +0.30 % (mtp on) / +0.37 % (mtp_k=0), 6 of 6 pairs. `BigIQ3S` steps: per-warp decode 563 / 630 us, per-CTA predecode 511 / 564, one thread per sub-block 464 / 550, smem codebook 429 / 472 (cp.async.ca shares L1), 12 pieces + LDS.64 tile 406 / 457; IQ4_XS 294 / 360 for the same shapes |
| B-55 open | OPEN | `BigIQ3S` (smem-codebook build) vs IQ4_XS: ncu wait 0.99, mio_throttle 0.58, barrier 0.43 stalls per issue (IQ4_XS 0.36 / 0.17 / 0.31), 204 vs 177 regs; Q3_K has no IMMA (dequant 528 us + cuBLAS per prompt GEMM, 7 tensors in UD-Q4_K_M) |

## 2026-10-24 Batch 39: B-56 warp-per-block K-quant dequant

| ID | Status | Evidence |
|---|---|---|
| B-56 | DONE | `dequant_q3k_v2_kernel` (smem-staged u16 loads) and `dequant_q45k_v2_kernel` (uint2 loads), one warp per super-block, uint4 stores, element kernels kept for unaligned buffers; GgufDequant max_rel 0 both paths, 3 mutations caught. Qwen3.8-27B-UD-Q4_K_M `--vram-budget 24000` (FP16 cache 0 tensors) per launch Q3_K 482.6 -> 146.8 us (105), Q5_K 204.7 -> 116.7 us (74), Q4_K embedding 3659.9 -> 2182.5 us; pp512 3889.18 / 3886.57 / 3888.93 -> 3962.62 / 3957.78 / 3955.86 tok/s; greedy tokens identical; default budget 4038.05 / 4032.85 -> 4038.94 / 4042.58 |
| B-55 open (Q3_K) | REFUTED | Premise wrong at 32 GB: the 7 Q3_K tensors sit in the FP16 cache (44 tensors, 4170 MiB), prompt GEMMs run cuBLAS, Q3_K dequant 7 launches at load only. Uncached the dequant now costs 146.8 us, so a Q3_K IMMA races ~150 us + cuBLAS (`BigIQ3S` reached 406-457 us vs 564): not pursued |

## 2026-10-24 Batch 40: B-57 per-CTA predecode in the GGUF IMMA prompt kernel

| ID | Status | Evidence |
|---|---|---|
| B-57 | DONE | `mmq_imma_gguf_bm128_kernel`: IQ4_XS / IQ4_NL / Q5_K B tile decoded once per CTA (predecode tile, as IQ3_S), A by ldmatrix.x4, B scales as [sub-block][128] (one 8-B / 16-B load per lane). ncu IQ4_XS gate: 315.2M -> 234.0M instructions, 378.8 -> 350.8 us. UD-Q4_K_M per launch: IQ4_XS 298.3 -> 270.6 / 356.4 -> 319.7 us, Q5_K 226.2 -> 211.6 / 307.2 -> 288.5, IQ4_NL 367.2 -> 328.0 / 377.0 -> 334.0, IQ3_S 466.2 -> 441.8; IMMA kernel sum 1261.40 -> 1183.41 ms; pp512 4030.51 / 4016.23 / 4002.96 / 4012.50 / 4014.72 -> 4189.83 / 4199.52 / 4202.88 / 4197.97 / 4177.97 tok/s; PPL 5.0059 both; Qwen3-8B-Q8_0 (q8k128 kernels) unchanged |
| B-57 levers | REFUTED | MMA accumulator bias 0x4B400000 instead of I2FP: kernel sum 951.89 -> 964.37 ms (FADD lands on the FMA pipe). `fma.rn.f32x2` lowers to two FFMA on sm_120a. `__launch_bounds__(256, 2)` on IQ4 (128 regs, 2 CTAs/SM): 299.2 -> 375.5 us. Q5_K S=2 + predecode 242.9 / 335.9 us vs S=4 226.2 / 307.2. IQ4_XS S=4 + predecode 292.6 vs S=2 280.0 us |
| B-57 open | OPEN | ncu after B-57: stall_math 15.9 % of samples (12.2 on IMMA), wait 13.6, barrier 10.7, mio 9.8; tensor pipe 33 % active, issue 47 %. Next: fewer barriers per K-step (B predecode one step ahead), dense Q4_K raw kernel (229 us) unchanged |

## 2026-10-25 Batch 41: B-58 two barriers per K-step in the GGUF IMMA prompt kernels

| ID | Status | Evidence |
|---|---|---|
| B-58 | DONE | `mmq_imma_gguf_bm128_kernel`, `mmq_imma_q4k_raw_kernel`: the next stage loads after the first barrier, the trailing `__syncthreads` is gone (3 -> 2 per K-step). UD-Q4_K_M nsys, 2 pairs: IMMA kernel sum 1201.47 / 1203.29 -> 1194.37 / 1194.46 ms; IQ4_XS 269.1 -> 267.4 us, Q5_K 211.5 -> 209.8 / 288.3 -> 286.1, IQ3_S 430.8 -> 425.4, Q4_K raw 226.7 -> 226.0; pp512 4201.10 / 4219.33 / 4237.19 -> 4229.37 / 4235.49 / 4258.01 tok/s; ncu IQ4_XS stall_barrier 10.7 -> 7.2 %; PPL ppl_slice_12k 5.0059 both |
| B-58 levers | REFUTED | Q4_K raw (alpha, beta) as fp32 [kb][col] (one LDS.128, no HADD2): 254-255 regs, 227.1 -> 237.1 us, 350.1 -> 373.7 us |
| B-58 open | OPEN | One barrier per K-step (B decoded one step ahead into a second tile) bounded by stall_barrier 7.2 % on IQ4 (~1.5 % of the IMMA kernel sum); Q5_K S=4 is at 98 KB smem, a second tile does not fit, and the TU has 526 of 600 kernel code LOC. Q4_K raw after B-58: issue 56 %, 8 FFMA + 4 FMUL + 4 I2FP per IMMA |

## 2026-10-25 Batch 42: B-59 Qwen3.8-Flash-Next decode

| ID | Status | Evidence |
|---|---|---|
| B-59 | DONE | `gemm.fp8_attn_proj=auto` covers Qwen4Exp (`fp8_attn_proj_full`): 120 sidecar projections, 1988.4 -> 2558.4 MiB. imp-server, 6 sequential prompts x 400 tokens: decode 143.50 -> 149.87 tok/s, every prompt +3.4..+6.2 %, hits 87.0 / 87.3 %, TTFT unchanged (310.9 / 310.0 ms). Bench tg128 106.29-106.46 -> 120.19-120.70 tok/s (hits 90.7 -> 93.8 %: greedy text differs), pp512 1225-1244 -> 1211-1237. Decode-path PPL (`--prefill-chunk-size 1`, deterministic, ppl_4k) 5.0246 -> 5.0285 |
| B-59 levers | REFUTED | Expert cache LFU victims (decayed routing count, `moe.expert_cache_decay`): bench tg128 106.2 -> 109.9 (0.98) / 113.9 (0.99) / 115.2 (0.999), but imp-server over 6 different prompts 146.24 -> 142.71 (0.99) / 136.71 (0.995) / 124.74 (0.999) tok/s, hits 86.8 -> 79.5 % at 0.999: the count holds the previous topic's experts. Offline replay (`tools/analysis/expert_cache_sim.py` on `diagnostics.moe_expert_trace`) matched LRU hits (88.08 vs 88.1 %) but the trace records n = 1 steps only. Gather grid x 4 / 8 / 32 / 64 blocks: tg128 105.83-106.35 / 105.52-106.51 / 105.45-105.47 / 103.87-103.95 vs 16: 105.89-106.46 (PCIe-bound). `hc_read_decode` loops unrolled x4 (bit-identical, PPL 5.0285): tg128 120.58-120.70 -> 118.96-119.34 |
| B-59 open | OPEN | Per token (nsys, 396 slots): gather 3.96 ms on the critical path (~3.6 keys of 0.88 MiB per layer at ~40 GB/s), router -> topk -> resolve chain ~25 us per layer (resolve 11-17 us, one block), `hc_read_decode` 2 x 11 us per layer with 2 of 8 warps in its GEMV; the F16 attention sources stay resident for prefill (no fp8rows path for q/k/v/o, closed by B-60) |

## 2026-10-25 Batch 43: B-60 Qwen3.8-Flash-Next F16 attention sources

| ID | Status | Evidence |
|---|---|---|
| B-60 | DONE | Phase 4b frees F16 q/k/v/o with an FP8 row-scale sidecar when experts are host-resident (`release_f16_attn_sources_`, pool `ReleasePool::AttnSources`): 48 projections, 1140.0 MiB, expert cache 396 -> 423 slots/layer (server), 475 -> 500 (bench). imp-server, 6 sequential prompts x 400 tokens, A/B/A/B: decode 147.98 / 148.87 -> 150.20 / 150.38 tok/s, hits 87.3 -> 88.1 %, TTFT mean 316.4 -> 292.2 ms; the one prompt with identical text 137.27 -> 140.19. PPL ppl_4k deterministic: prefill 5.0131 -> 5.0057, decode path 5.0285 / 5.0285. Bench pp512 1204.13-1213.18 -> 1270.68-1274.01, tg128 118.97-119.28 -> 115.19-115.26 (hits 93.8 -> 92.5 %: greedy text differs). 4 concurrent requests 143.4 / 144.4 -> 143.2 / 143.2 tok/s |
| B-60 open | OPEN | Gather (PCIe) stays the largest per-token block; next VRAM for the cache: the server's KV ceiling reserve, the vision arena (`runtime.no_vision`, 1079 MiB) |

## 2026-10-25 Batch 44: B-61 Qwen3.8-Flash-Next TTFT

| ID | Status | Evidence |
|---|---|---|
| B-61 | DONE | Staging tables uploaded once (`stage_ptrs_`, `stage_alpha_row_`), scale conversion only for routed experts, 16 grid-stride blocks per expert, no memset without SfAtom padding. Bit-identical (PPL ppl_4k 5.0057 / 5.0057). Bench pp32 cold cache (`--max-tokens 4`) 272.61-275.56 -> 255.12 ms, warm (`--max-tokens 128`) 157.32-176.48 -> 138.68-138.85 ms, pp512 396.04 / 399.22 -> 385.76-392.95 ms. imp-server, 6 prompts: TTFT mean 301.7 -> 295.0 ms (without the cold first request 292.8 -> 281.6), decode 150.95 / 150.46 -> 150.08 / 150.27 tok/s |
| B-61 open | OPEN | A cold pp32 prefill (nsys) is 81 % `expert_stage_touched_range_kernel`: 14967 touched projections, 16.9 % resident in the expert cache, 10.9 GB over PCIe at ~50 GB/s. Left: GEMMs 12 ms, scale conversion 10 ms, GPU idle 25 ms per prefill. Below the PCIe floor only with fewer bytes: CPU computing part of the missing experts from host RAM in parallel, or prefetching the next layer's experts from a pre-gate prediction |

## 2026-10-29 Batch 45: B-62 IQ3_S piece offsets without branches

| ID | Status | Evidence |
|---|---|---|
| B-62 | DONE | `BigIQ3S::piece_off` from packed immediates (base offset per piece, j0 factor in 5-bit fields) instead of a ternary chain: BSSY / BRA 35 / 95 -> 16 / 22. Qwen3.8-27B-UD-Q4_K_M pp512 nsys, 4 alternating pairs: BigIQ3S kernel sum 15.38 / 15.79 / 15.35 / 15.42 -> 12.03 / 12.39 / 12.71 / 12.05 ms (45 launches), pp512 3903-4019 -> 4000-4023 tok/s. PPL 45k deterministic 4.4968 both (bit-identical offsets). |

## 2026-11-04 Batch 46: B-63 Qwen3.8-Flash-Next next-layer expert prefetch

| ID | Status | Evidence |
|---|---|---|
| B-63 | DONE | `moe.expert_prefetch` (default 2): layer L's MoE input through layer L+1's router, top-10, up to N absent keys get LRU victims and are gathered behind L's misses on the gather stream; a hit on a pending slot waits for L+1's gather event. imp-server, `tools/analysis/flashnext_prefetch_ab.sh`, 6 prompts x 400, medians: N=0 91.69 / 91.42 / 91.62 / 91.32, N=1 92.37 / 92.29, N=2 93.72 / 93.32 / 93.48 / 94.17, N=3 91.69 / 92.09, N=4 90.19 / 90.36 tok/s. N=2: hits 87.8 -> 89.5 %, misses 490487 -> 421538, 151606 prefetched, 80116 routed (52.8 %). Decode-path PPL ppl_4k deterministic 5.0285 both (235571 keys prefetched) |
| B-63 levers | REFUTED | N >= 3: N=4 cuts misses to 368303 (-24.9 %) and decodes -1.3 %; half the prefetches are wrong and their PCIe time lands before L+1's own gather. Offline price (`tools/analysis/expert_prefetch_sim.py` on a trace with prediction records, 424 slots): predictor recall 66.0 % of top-10, LRU replay 87.0 % vs live 87.8 % |
| B-63 open | OPEN | Wrong prefetches cost the next layer's critical gather: a better predictor (hc-mixed L+1 input instead of L's MoE input) or issuing from layer L-1 widens the window; the gather of L+1 is FIFO behind the prefetch on one stream |

## 2026-11-04 Batch 47: B-63 open, next-layer predictor and the gather FIFO

| ID | Status | Evidence |
|---|---|---|
| B-64 | DONE | `moe.expert_prefetch_hc_norm` (default on, Qwen4Exp): layer L+1's router reads the hc streams normed with L+1's mlp norm and summed (`hc_norm_sum`, one block) instead of L's MoE input, and L's gather no longer waits on the predictor read (`pf_read` joins behind the gather). Offline, fresh trace with prefetch off, `expert_prefetch_sim.py --tag`, 424 slots, N=2: recall 66.0 -> 71.4 % of top-10, prefetches routed 52.9 -> 57.8 %, misses 161.1 -> 154.8 per token. imp-server `flashnext_prefetch_ab.sh`, 6 prompts x 400, 3 alternating pairs: main 94.74 / 95.00 / 94.60, decoupled read alone 94.66 / 94.74 / 94.84, both 95.57 / 96.02 / 95.98 tok/s; misses 421538 -> 405125, routed 52.8 -> 57.7 %; the 6 answers byte-identical to main. Shipped tree vs main 43c8d8d3, 3 pairs: 94.42 / 94.88 / 94.82 -> 96.11 / 95.84 / 95.88 tok/s. Decode-path PPL ppl_4k deterministic 5.0285 both (branch 251527 keys prefetched; the main run fell back to the host expert tier, 0 prefetched). degen_suite --quick 52 checks, 0 FAIL |
| B-63 FIFO | REFUTED | The next layer's gather never waits on the prefetch: completion counters read by that layer's resolve, N=2 0 of 151606 keys still copying, N=4 0 of 258043, N=6 31 of 331635. Abort of unrouted copies plus a rank-serial copy kernel, 3 pairs: main 90.94 / 89.72 / 89.90, N=4 89.59 / 87.98 / 87.65, N=6 89.61 / 89.48 / 89.72, N=4 serial without abort 88.11 / 87.85 / 88.31 tok/s |
| B-63 predictors | REFUTED | Offline recall of top-10 (same trace): L's MoE input to L+2 56.8 %, L's attention input to L+1 52.7 %, L+1's attention input 58.5 %, L's mix weights with L+1's norm 69.6 %, L+1's full mlp hc read (+13 MB reads per layer) 78.8 % at L's MoE, 81.9 % after it, L+2 full read 68.1 %. hc input with the gather still waiting on the read: main 91.53 / 94.09 / 94.61, N=2 90.08 / 93.34 / 93.49, N=3 88.97 / 92.58 / 92.32 tok/s. With both: N=3 94.52 / 94.31 vs N=2 96.18 / 95.50 |
| B-64 open | OPEN | The full L+1 mlp read (78.8 %) needs a mix without the 13 MB of hc down / up reads. The N > 2 loss is closed by B-65 |
| B-65 | DONE | `moe.expert_prefetch` 2 -> 6 at `moe.expert_prefetch_blocks` 1 (new key, copy blocks per prefetched key): own copy kernel `expert_prefetch_gather_kernel` (4 loads in flight per thread), a landed prefetch joins the hit group, the next resolve waits on `pf_ready`. Qwen3.8-Flash-Next-NVFP4 server, `flashnext_prefetch_ab.sh`, 6 prompts x 400, 3 pairs: main (N=2, 16-block gather) 96.38 / 95.98 / 95.74, N=2 blocks 16 88.94 / 89.07 / 88.88, N=4 blocks 2 100.20 / 100.06 / 100.06, N=6 blocks 1 102.96 / 102.74 / 102.56, N=6 blocks 2 100.26 / 99.83 / 99.85, N=8 blocks 1 102.49 / 102.61 / 102.47, N=10 blocks 1 101.48 / 101.82 / 101.50 tok/s. Shipped defaults, 2 pairs, blocks 1: main 95.84 / 95.66, N=0 92.15 / 91.88, N=2 96.77 / 96.38, N=4 100.23 / 99.80, N=5 101.95 / 101.69, N=6 102.84 / 102.63, N=7 101.83 / 102.96; hit rate 89.9 -> 92.5 %, misses 405125 -> 301472. Other host-expert models, main vs N=6, 2 pairs: Kimi-Linear-48B-A3B 322.46 / 322.46 vs 322.62 / 323.32, Qwen3-30B-A3B with `moe.force_host_experts=48` 306.36 / 307.61 vs 306.87 / 307.53. The 6 answers byte-identical across all arms per model (47 runs), 0 stderr matches. Decode-path PPL ppl_4k deterministic 5.0285 both (main 251527, branch 542149 keys prefetched), degen_suite --quick 52 checks 0 FAIL, verify-fast OK |
| B-65 cause | DONE | Cause of B-64 open measured (nsys `--cuda-graph-trace=node`, imp-cli 400 tokens): per layer A = resolve start to miss gather end, B = miss gather end to next resolve. main N=0 / 2 / 4: A 80.0 / 66.7 / 56.6 us, B 152.8 / 155.8 / 172.6 us. Kernels beside a prefetch gather slow down (gemv_fp8_e4m3 12.3 -> 24.3 us); the cost scales with PCIe loads in flight, not copy time: branch N=4 with 4 loads per thread and `moe.expert_prefetch_blocks` 16 / 4 / 2 / 1: B 192.2 / 170.0 / 157.5 / 159.6 us at an unchanged 45.7 us copy. Prefetch resolve 9.6 us on its side stream, not the cost. Server numbers: row B-65 |

## 2026-11-10 Batch 48: B-66 MTP multi-slot head scan, split-K

| ID | Status | Evidence |
|---|---|---|
| B-66 | DONE | `mtp_attn_scan_slots` (bound multi-slot draft pass, `src/compute/mtp_attn_scan.cu`): `mtp_attn_kv_scan_rows_split_kernel`, grid (24 heads, `kMtpScanSlotSplits` 16, rows <= `kMtpScanSlotSplitRows` 16), then `mtp_attn_kv_combine_kernel` per (head, row); before, one CTA per (head, row) walked the 16384-row ring serially. Qwen3.8-27B-NVFP4-vllm server defaults, nsys `--cuda-graph-trace=node` (`tools/analysis/mtp_longctx_nsys.sh`, 80k prompt, 1 chat): scan 886.09 -> 140.43 us per call, step 15.350 -> 14.426 ms. `mtp_longctx_ab.sh`, main 2257a441 vs branch, alternating pairs, per-stream decode tok/s at 20k / 40k / 80k: defaults 1 chat (3 pairs) 130.06 / 126.64 / 121.61 -> 137.47 / 133.68 / 128.21, 2 chats 123.68 / 119.85 / 116.45 -> 128.42 / 123.52 / 121.04; `kv_cache.host_spill_mb=16384` (2 pairs) 1 chat 137.22 / 141.57 / 135.88 -> 150.16 / 154.87 / 148.24, 2 chats 141.28 / 130.28 / 129.69 -> 151.61 / 141.31 / 139.15; 4 chats at 8k / 20k: defaults 125.57 / 111.67 -> 127.73 / 116.04, spill 141.69 / 122.32 -> 142.89 / 132.29. Head feed 1.91 -> 1.16 ms per call (defaults), 3.70 -> 2.22 (spill). 1 chat: drafted, accepted and output md5 equal per context in both arms. Cost: workspace row 359 -> 367 MiB (partials in one 8 MiB allocation), KV pool 15452 -> 15431 blocks at 2 chats. `AttnScanSlotsSplitMatchesDirect` (rows 1 / 3 / 16, ring wrap, rel. RMS < 1e-3), degen_suite 52 checks 0 FAIL, 8 concurrent 652.3 tok/s with 0 stderr matches, verify-fast OK |
| B-66 cause | DONE | 1 vs 2 long chats on main (same A/B, main arms): head feed per call flat (defaults 1.91 vs 1.90 ms, spill 3.70 vs 3.72 ms), verify step 14.43 -> 15.02 ms (defaults) and 16.93 -> 17.75 ms (spill) in forward + argmax. The rest of the per-stream gap is acceptance per prompt trajectory (66.0-75.2 % at depth 2, 2 chats), not contention on the head |
| B-66 default | DONE | `speculative.mtp_batch_verify_factor_row` (default true): at 2 drafts draft 1 leaves the scan as a GDN factor row + conv tap (3.22 MiB per recurrent slot) instead of a second spare slot per row (80.0 MiB each), folded into the live slot when 1 of 2 drafts is accepted; the concurrent GDN-hybrid default keeps `mtp_batch_verify_k=2` (`tools/common/mtp_auto.cpp`). Qwen3.8-27B-NVFP4-vllm server defaults, `mtp_longctx_ab.sh`, 2 pairs, k 1 -> factor row, all streams: 1 chat at 20k / 40k / 80k 137.08-137.79 / 133.81-133.86 / 128.41-128.80 -> 165.83-166.20 / 156.86-156.98 / 154.30-154.34 tok/s; 2 chats 248.94-249.24 / 243.31-244.16 / 240.15-240.44 -> 287.77-288.06 / 288.09-292.06 / 284.98-286.72; 4 chats at 8k / 20k 476.19-482.75 / 450.19-451.67 -> 575.17-583.27 / 509.62-509.81; 8 chats at 8k 569.98-571.44 -> 570.43-573.75; KV plan 11239 -> 10901 blocks (second spare: 10313). The NVFP4 draft head on top (`speculative.mtp_nvfp4_head=true`, 682 MiB) moves nothing consistent: 1 chat 20k / 80k 164.42-166.44 / 153.40-154.23 -> 162.98-163.21 / 153.61-154.91, 2 chats 287.68-293.26 / 283.81-285.02 -> 298.29-301.96 / 273.70-274.48; it stays off. Not measured: SWE-bench dev 50 on the 338 blocks. Evidence `docs/audit/qwen38_mtp_verify_factor_row_2026_11_11/`, [plan](../plans/2026-09-11-batched-mtp-verify.md) section "Draft 1 as a factor row" |
| B-66 default, before | SUPERSEDED | Concurrent GDN-hybrid default (no NVFP4 draft head, `mtp_batch_verify_k=1`, `tools/common/mtp_auto.cpp`) on main, 1 chat at 20k / 40k / 80k: 130.2-130.7 / 126.5-127.3 / 120.7-121.9 tok/s; full pair (`speculative.mtp_nvfp4_head=true`, `mtp_batch_verify_k=2`) 149.00-149.09 / 141.56-141.76 / 147.38-147.47; `kv_cache.host_spill_mb=16384` 135.73-137.66 / 140.61-141.55 / 134.63-135.80. KV pool 15452 / 12537 / 14527 blocks. The default trades 10-17 % at 1 chat for 2915 blocks; owner decision |
| B-67 | DONE | Bound verify at 3 drafts (`speculative.mtp_batch_verify_k` default 3): factor rows + conv taps indexed by verify group x level (48 layers x 4 groups x 2 drafts = 18.3 + 7.5 MiB; by recurrent slot at 2 drafts 82.3 + 33.8 MiB), the fold applies 1 or 2 rows (`kGdnFactorFoldMaxRows`, `src/compute/gdn_factor.cu`), `mtp_k=auto` on a concurrent GDN hybrid follows the verify depth, an unset depth follows a smaller `mtp_k`. Qwen3.8-27B-NVFP4-vllm server defaults, `mtp_longctx_ab.sh`, one image, D 2 -> D 3, 3 sessions x 768 tokens, tok/s per stream: 1 chat at 20k / 80k 174.52 / 164.89 -> 181.83 / 181.18; 2 chats 156.36 / 150.90 -> 164.40 / 160.96; 4 chats at 8k / 20k 151.03 / 136.78 -> 163.02 / 144.64; all streams 4 x 20k 522.60 -> 544.05. Step at 4 requests 14.44 + 2.91 ms for 9.56 tokens -> 15.23 + 4.31 ms for 11.27. KV plan 10901 -> 11164 blocks. D 2 on the branch against main: 16 of 16 streams md5-equal. Worst session 1 x 20k: 165.64 -> 158.17 (464 of 903 drafts accepted). Evidence `docs/audit/qwen38_mtp_verify_depth3_2026_11_11/`, [plan](../plans/2026-09-11-batched-mtp-verify.md) section "Up to 3 drafts per bound request" |
| B-67.1 | NOT BUILT (2026-11-11, replay only) | Verify depth per request by its acceptance: the third draft costs 0.79 ms forward + 1.40 ms head per step at 4 requests and loses on a low-acceptance text (B-67 worst session). Replay of traced depth-2 and depth-3 steps through 50 policies (`tools/analysis/mtp_depth_replay.py`, a calculation, no A/B): best policy without the step's outcome +0.01 % / -0.02 % / -1.62 % against fixed depth 3 at 1 / 2 / 4 chats, with the outcome known +3.95 % / +2.50 % / +1.34 %; a full accept repeats with P 0.645 / 0.648 / 0.665. Depth by requests in the step shipped instead (B-69). Evidence `docs/audit/qwen38_mtp_verify_depth4_2026_11_11/depth_replay.txt` |
| B-68 | DONE | A draft that matched `</think>` under a think budget ended the accept loop after `matched++`: emitted == matched, the kept rows already held the token and the next step forwarded it again at the same position. The attention KV row is rewritten with the same values, the GDN recurrent state advances through the token twice. Fix: one accept loop for the single-stream verify and the bound batched verify (`spec_accept_rows`, `src/runtime/spec_accept.h`), the crossing token stays the chunk's bonus (emitted == matched + 1 unless the request finished). Qwen3.8-27B-NVFP4-vllm dev build, 12 thinking chats x 2, main -> fix, forwards of `</think>` at an already kept position: single-stream (`--max-batch 1`, 2 drafts) 8 of 24 requests -> 0; bound verify with `speculative.mtp_draft_vocab=0` (3 drafts) 16 of 24 -> 0; 2 of 12 answers change per path, every arm 12 of 12 reproducible. Bound verify at server defaults: 0 of 24 on main, `mtp_draft_vocab=98304` (`tools/common/mtp_auto.cpp`) scores LM-head rows below 98304 and `</think>` is id 248069. Which answer is right is not measured (the MTP text differs from the eager text inside the think block already). CPU test `SpecAcceptRows.*` (`tests/test_spec_gates.cpp`, 18708 boundary crossings swept), 6 of 6 mutants red. Evidence: `docs/audit/qwen38_spec_think_boundary_2026_11_11/` |
| B-69 | DONE | Verify depth per step: 4 drafts below 4 requests (`speculative.mtp_batch_verify_k` default 4, `kMtpBoundVerifyMaxDepth`), 3 from 4 requests up (new `speculative.mtp_batch_verify_k_wide`, default 3, `mtp_bound_verify_step_depth`); factor rows 48 layers x 4 groups x 3 drafts = 27.4 + 11.2 MiB, the fold applies 1 to 3 rows (`kGdnFactorFoldMaxRows`), a fourth head chain level (`chain_pos3`). Qwen3.8-27B-NVFP4-vllm server defaults, `mtp_longctx_ab.sh`, one image, 3 drafts -> default, 2 pairs x 2 sessions x 768 tokens, tok/s per stream (min-max): 1 chat at 20k 194.25 (193.21-195.18) -> 211.32 (209.64-212.82), at 80k 176.56 (176.01-177.08) -> 188.66 (170.44-199.89); all streams: 2 chats at 20k / 80k 315.45 / 311.60 -> 324.78 / 320.01, 3 chats at 8k / 20k 493.42 / 427.93 -> 500.39 / 444.28, 4 chats at 8k / 20k 594.82 / 530.84 -> 589.75 / 530.61 (same depth, inside min-max). Opposite of the wide key (`_k_wide=4`): 4 x 20k 517.18 against 530.61, 4 x 8k 588.57 against 589.75. Step at 4 requests 14.709 + 4.010 ms for 11.52 tokens at 3 drafts, 15.506 + 5.333 ms for 12.77 at 4. KV plan 11164 -> 11126 blocks. 3 drafts on the branch against main: 14 of 14 streams md5-equal. Worst session of B-67 (1 x 20k, third session): 158.72 -> 169.60. `diagnostics.spec_trace` prints one `spec-batch-trace` line per bound step (`tools/analysis/mtp_trace_hist.py`). Evidence `docs/audit/qwen38_mtp_verify_depth4_2026_11_11/`, [plan](../plans/2026-09-11-batched-mtp-verify.md) section "4 drafts per bound request, 3 from 4 requests up" |
| B-69.1 | DONE | Think budget under MTP: a verify chunk landed up to `mtp_k` tokens past the budget before the closer was forced, so the greedy text at a budget cut followed `mtp_k`. Fix: the accept loop ends a chunk once it emitted the reasoning tokens the budget has left (`think_logic::think_tokens_left`, `src/runtime/think_stop_logic.h`; `spec_accept_rows`, `src/runtime/spec_accept.h`), bound batched verify and single-stream verify. Qwen3.8-27B-NVFP4-vllm server, 1 greedy chat x 20k, 384 tokens, default budget, reasoning tokens before the closer at `mtp_k` 0 / 1 / 2 / 3 / 4, main e2906af1 -> fix: bound verify 193 / 193 / 194 / 195 / 196 -> 193 x 5; single-stream (`runtime.max_batch_size=1`, `mtp_k` 0 to 3) 193 / 193 / 193 / 194 -> 193 x 4. Bound verify texts for `mtp_k` 1 to 4: 4 -> 1; `mtp_factor_row_equiv.sh` depth check runs with the default budget, 4 of 4 md5-equal (main 1 of 4). Single-stream texts still follow `mtp_k` (3 texts for 1 to 3 on both images, cause not measured). CPU test `SpecAcceptRows.ChunkEndsAtTheThinkBudget`, red with the break removed. Evidence `docs/audit/qwen38_think_budget_exact_2026_11_11/`, [plan](../plans/2026-09-11-batched-mtp-verify.md) section "A verify chunk ends at the think budget" |
| B-69.2 | PROFILED (2026-11-11, no A/B) | Head chain level at 4 requests: 1.32 ms per level (5.333 against 4.010 ms head at 4 against 3 drafts), 1.02 ms at 1 request (4.227 against 3.204 ms). nsys of the bound verify step, imp:m3091, 20k-token chats, wall per step: 1 chat at 3 / 4 drafts 16.828 / 18.008 ms, 4 chats at 3 / 4 drafts 20.458 / 22.644 ms (`head_level_wall.txt`). Exclusive kernel time (end minus max(start, previous end on the stream): the CUPTI duration of a PDL kernel includes the wait on its predecessor, `elementwise_add_store_fp16_kernel` 34.94 us raw, 0.47 us exclusive), 1 chat at 4 drafts, ms per step: `gemm_nvfp4_smallm_v2_kernel` 8.609, `fp8_head_cpasync_kernel` 2.062, `gemv_fp16_kernel` 1.705, every other kernel below 0.7 (`head_level_excl_c1_d4.txt`). Host gap argmax -> embedding at 4 chats x 3 drafts: 0.973 ms per step in the mean (994.0 us), median 325.7 us; the mean carries `cudaGraphInstantiate` (`head_level_gaps_c4_d3.txt`, `head_level_gapapi_c4_d3.txt`). The 3 matrix kernels hold 12.376 of 17.043 ms exclusive; no lever identified in the rest: a reading of one profile, no arm built. Evidence `docs/audit/qwen38_suffix_shadow_2026_11_11/` |
| B-70 | NOT BUILT (2026-11-11, replay only) | Suffix drafts (copy spans) for MTP-bound requests: `Engine::batch_verify_draft_` asks the suffix index only when MTP has no draft, so a bound agent chat never drafts a copy. Instrument: `diagnostics.spec_trace` logs `spec-sfx-shadow id pos n T mtp=nd:m:e sfx=len:match:agree` per request step (`src/runtime/suffix_shadow.h`): the 32-token suffix draft at that position, scored against the tokens the request emitted afterwards. Workload: recorded familiar SWE-bench sessions replayed turn by turn against `/v1/messages` (`tools/analysis/agent_turn_replay.py`), Qwen3.8-27B-NVFP4-vllm server, `speculative.max_match=32`, `max_tokens` 4096, think budget 2048. Serial run, 6 sessions x 60 turns: 166353 output tokens, reasoning 75.2 % (recorded mix 75.2 % thinking chars), 165.47 tok/s decode, 60599 steps; 86.0 % of the steps have no suffix match; at match >= 32 (2.7 % of the steps) MTP already emits 4.69 of 5 tokens per step and the suffix draft cut to MTP depth 4.58. Policy replay (`tools/analysis/sfx_shadow.py`, a calculation, no A/B), suffix precedence at match >= M with K draft rows, 1 request: K 4 loses at every M (-0.02 to -1.04 %), best point M 8 / K 16 +3.08 %. Mixed run (1 chat, then 4 chats, 190118 tokens): 1 request best +1.31 % (M 8, K 16); 4 requests under one row count per step lose at 23 of 24 points (M 12: K 8 -1.77 %, K 16 -5.00 %, K 32 -11.96 %; M 32 / K 4 +0.04 %). Deep suffix steps need folds past 3 factor rows, cap rows and graph shapes up to 17 rows; not built for a replayed +3 % at 1 request. Gates: `SuffixShadow.*` 5 tests, 6 of 6 mutants red (`mutants.txt`). Evidence `docs/audit/qwen38_suffix_shadow_2026_11_11/` |
| B-71 | DONE | Coupled MTP drafts: a sampled bound request's draft at level lv is drawn from the head's logits with the request's `top_k` / `top_p` / temperature and the seed of output index `size() + lv`, the seed of the verify row that checks it (`speculative.mtp_draft_coupled`, default true; `engine_internal::draft_row_draws`, `GraphExecutor::stage_draft_draws`, drawn inside the head-pass graph, key field `draw`, `kMtpHeadGraphMax` 48 -> 96, scratch 32 rows 2.02 MiB). The top-k / top-p samplers draw by an exponential race keyed on (seed, token id) (`sampler_race_pick`, `src/compute/sampling_internal.cuh`) instead of one uniform through the sorted CDF: seeded sampled texts change once. An unseeded request keeps one sampler base (`Request::draw_base`). Qwen3.8-27B-NVFP4-vllm server, recorded familiar SWE-bench sessions (`tools/analysis/agent_turn_replay.py --serial --seed 1000`), 3 chats x 30 turns in turn. Sampler choice on experiment builds, deterministic, tokens per step argmax drafts -> coupled: CDF sampler 2.787 -> 2.833 (draft temperature x 0.7 the best of 5), race sampler 2.847 -> 3.340 (x 1.0 the best of 3: 3.276 / 3.340 / 3.253). Shipped tree, deterministic, texts md5-equal: 2.981 -> 3.461 tokens per step, steps accepting all 4 drafts 33.1 % -> 43.7 %, 176.13 -> 203.81 tok/s decode per chat; default mode off / on / off / on 168.43 / 194.79 / 168.71 / 193.70 tok/s; 2 chats overlapping 157.23 -> 179.08, 4 chats 123.91 -> 138.71 tok/s decode per chat (unpaired). Default mode, 2 server starts: argmax drafts 3 of 3 texts md5-equal, coupled 2 of 3 (deterministic mode, coupled, 3 starts: 13917 of 13917 steps equal), row in `docs/DETERMINISM.md`. Tests: `SampleVerifyRows.CoupledDraftFollowsTheTokenNotTheRank` (503 of 512 rows), `CoupledDraftMatchesTheVerifyDrawMoreOftenThanTheArgmax` (511 against 259 of 512), `DraftRowDraws.*`, `StepSeed.UnseededRequestKeepsOneBase`; 5 of 5 mutants red (`mutants.txt`). Evidence `docs/audit/qwen38_coupled_draft_2026_11_11/`, [plan](../plans/2026-09-11-batched-mtp-verify.md) section "Coupled MTP drafts" |
| B-72 | DONE | Verify depth ladder: below 4 requests a step in which a request accepted all N drafts is followed by a step at N + 2 drafts, capped at new `speculative.mtp_batch_verify_k_streak` (default 7: 4 -> 6 -> 7), any other step returns to `speculative.mtp_batch_verify_k` (`mtp_bound_verify_streak_next`, `kMtpBoundVerifyStreakStep`, `src/runtime/spec_batch_verify_state.h`). A streak never moves the n x (depth + 1) rows of a step into the next 8-row tile of the small-M GEMM (`mtp_bound_verify_tile_depth`, `kMtpBoundVerifyRowTile`): 1 or 2 requests reach 7 drafts (8, 16 rows), 3 requests stay at 4 (15 rows), 4 requests keep `_k_wide`. `kMtpBoundVerifyMaxDepth` 4 -> 8, `kGdnFactorFoldMaxRows` 3 -> 7, `kMtpPassChainLevels` 7, `kMtpHeadGraphMax` 96 -> 384, `mtp_k=auto` -> 7; factor rows 48 layers x 4 groups x 6 drafts = 54.8 + 22.5 MiB (27.4 + 11.2 at 4 drafts), KV plan 17089 -> 16977 blocks. New `diagnostics.mtp_verify_pin=N` pads every step below 4 requests to N draft rows, so depth policies emit md5-equal texts. Qwen3.8-27B-NVFP4-vllm server, recorded familiar SWE-bench sessions (`tools/analysis/agent_turn_replay.py --serial --seed 1000`), 3 chats x 30 turns in turn. Survival of draft 5 / 6 after a step that accepted 4 of 4: 35.2 % / 29.1 % (2223 steps), draft 4 after a partial step: 28.4 % (5154 steps). Paired (pin 8, 40259 output tokens, 3 of 3 texts md5-equal, step counts pinned x class cost unpinned), tokens per ms: 4 drafts 0.19893, ladder to 6 0.20288 (+1.99 %), to 7 0.20457 (+2.84 %), to 8 on 9 rows 0.20314 (+2.12 %); tokens per step 3.3620 -> 3.6281. Unpaired: default mode `k_streak` 0 / 7 193.79 -> 200.95 tok/s decode per chat; deterministic 203.79 -> 206.92; 2 chats overlapping, tokens per ms of the steps at 2 requests 0.3718 -> 0.3807, 0.3892 (2 runs); 3 chats overlapping 159.18 -> 157.66 tok/s decode per chat with 3958 of 3958 steps at 3 requests on 4 drafts (same policy, spread). Without the tile rule the steps at 3 requests cost 15.366 / 16.862 / 17.220 ms forward at 15 / 21 / 24 rows for 0.5055 -> 0.5035 tokens per ms. Tests `BatchVerifyPredicates.StepDepthFollowsTheRequestsInTheStep`, `MtpAuto.*`, `GdnVerifyTwoDraftsTest.ScanFactorRowsFoldOneToSevenDraftsOfEight` (+ `Bf16`), `ConvFactorRowsFoldFourToSevenDraftsOfEight`; 13 of 13 predicate mutants and 2 of 2 fold kernel mutants red (`mutants.txt`). Evidence `docs/audit/qwen38_verify_ladder_2026_11_11/`, [plan](../plans/2026-09-11-batched-mtp-verify.md) section "Verify depth ladder after a full accept" |
| B-72.1 | NOT BUILT (2026-11-11, survival only) | A second row tile on a long streak (9 or more rows at 1 request): after a step that accepted 8 of 8 the next accepts 8 of 8 again in 58.8 % of 551 steps (`survival.txt`), a 9th row costs 1.04 ms forward (13.461 -> 14.501 ms, `row_sweep.txt`) plus about 1.0 ms head per chain level; the ladder to 8 on 9 rows measured +2.12 % against +2.84 % at 7. A calculation from 551 steps, no A/B of a 16-row step after 8 of 8. 2026-11-12: `speculative.mtp_batch_verify_k_streak=8` alone does not open the second tile: 1 chat paired (pin 8, `mtp_k` 8, deterministic, 40259 output tokens, 3 of 3 session md5 equal) 208.21 (7) vs 208.05 (8) tok/s per chat, 11203 steps and 4.79 drafted/step in both, because `mtp_bound_verify_tile_depth` (`src/runtime/spec_batch_verify_state.h`) clamps 1 chat to 7 drafts (8 rows) before the streak cap (`docs/audit/qwen38_mtp_body_fp8_2026_11_12/k_streak8.txt`). With the E4M3 body the 7th draft after a full accept is break-even (B-78: tok/ms 0.2198 vs 0.2199), so deeper rows stay NOT BUILT |
| B-73 | OPEN (lead, 2026-11-11) | Draft head format on a concurrent server: FP8 over LM-head rows [0, 98304) (default with `kv_cache.host_spill_mb`) 4.084 ms head per step, 203.59 tok/s decode per chat; the full NVFP4 copy (`speculative.mtp_nvfp4_head=true`, 682 MiB) 4.553 ms, 199.31 tok/s (`draft_head_fp8_vs_nvfp4.txt`, pin 4, texts md5-equal, 48176 output tokens): the full copy stays off. Not measured: an NVFP4 copy over the first `speculative.mtp_draft_vocab` rows only (`QuantPipeline::mtp_draft_lm_head_nvfp4_`, `src/exec/pre_dequant_phase3_nvfp4_decode.cpp`; `GraphExecutor::draft_rows_vocab_`, `src/exec/executor_draft_draws.cpp`) |
| B-73 | DONE | NVFP4 draft copy over the LM-head row prefix: a concurrent GDN-hybrid server drafts on an NVFP4 copy of rows [0, `speculative.mtp_draft_vocab`) instead of the FP8 head's rows [0, 98304); verification stays on the full FP8 head. `mtp_concurrent_pool_defaults` (`tools/common/mtp_auto.cpp`) sets `speculative.mtp_draft_vocab=98304` unless set and turns `speculative.mtp_nvfp4_head` off only when `mtp_draft_vocab=0` (full copy) was set. New `src/compute/mtp_draft_rows.h`: `mtp_draft_copy_rows` sizes the copy (`QuantPipeline::mtp_draft_lm_head_nvfp4_`, `src/exec/pre_dequant_phase3_nvfp4_decode.cpp`), `mtp_draft_pass_rows` the rows every draft pass reads (`GraphExecutor::draft_rows_vocab_`, `src/exec/executor_draft_draws.cpp`; `mtp_emit_token`, `src/compute/mtp_forward.cu`). Qwen3.8-27B-NVFP4-vllm: copy 270.0 MiB at 98304 of 248320 rows against 682.0 MiB at all rows; KV ceiling after the library-reserve replan 21166 -> 20380 blocks in default mode (deterministic arms: FP8 18853, NVFP4 prefix 18067, NVFP4 full 16863, `kv_ceiling.txt`). Server, recorded familiar SWE-bench sessions (`tools/analysis/agent_turn_replay.py --serial --seed 1000`), 3 chats x 30 turns in turn. Paired (`diagnostics.mtp_verify_pin=7`, `speculative.mtp_k=7`, `runtime.deterministic=true`, 52107 output tokens, 3 of 3 texts md5-equal across the 4 arms): FP8 prefix 198.04 / 198.03 tok/s decode per chat at 4.822 / 4.824 ms head per step, NVFP4 prefix 204.26 at 4.148, NVFP4 full 192.68 at 5.405. Unpaired, default mode, FP8 -> NVFP4 prefix, tok/s decode per chat: 1 chat 200.89 -> 205.35 (head 4.688 -> 3.995 ms per step); 2 chats overlapping 185.08 -> 197.14 (wall 299.40 -> 291.19 with 30692 vs 37141 output tokens, head ms per step lower in every state); 4 chats overlapping 147.12 -> 149.95 (wall 402.78 -> 427.63); 8 chats x 20 turns, 3 runs each, 79.00 / 73.68 / 74.30 -> 75.47 / 77.61 / 75.82, wall 359.94 / 402.00 / 376.36 -> 432.93 / 417.59 / 448.68 tok/s: the wall gain has no isolated mechanism (output token counts differ per run), no loss measured. KV-saturated (8 agents x 8 turns, prompt median 99550 to 102062 tokens, `kv_cache.host_spill_mb=16384`, 3 runs per arm, `kv_saturated_agents.txt`): 132.5 -> 140.5 output tok per wall s, uncached prompt tok 9074420 vs 8937938, wall per run 601.5 vs 619.6 s against an FP8 spread of 572.8 to 629.8 s, texts not paired: no loss resolved, no gain claimed. Not measured: GDN hybrids with `gemm.nvfp4_lm_head_gdn` or a natively NVFP4 head on a concurrent server (their drafts now read rows [0, 98304) of the main-head view, no copy). Tests `MtpDraftRows.*` (4, `tests/test_mtp_draft_rows.cpp`), `MtpAuto.*`; 8 of 8 CPU mutants red (`mutants.txt`). Evidence `docs/audit/qwen38_draft_head_prefix_2026_11_12/` (final-tree gates `gates.txt`), [plan](../plans/2026-09-11-batched-mtp-verify.md) section "NVFP4 draft copy over the row prefix" |
| B-74 | DONE | One-chat bound verify lane: a GDN-hybrid `imp-server --max-batch 1` with `speculative.mtp_k` auto ran the single-stream chain (`mtp_k` 2, full 682.0 MiB NVFP4 draft copy); it now takes the bound batched verify of the concurrent server (auto `mtp_k` 7, ladder 4 -> 6 -> 7, draft copy rows [0, 98304) 270.0 MiB). `mtp_auto_engine_batch` (`tools/common/mtp_auto.cpp`, called from `load_model_into_state`, `tools/imp-server/handlers.cpp`) returns engine batch 2 (`kMtpBoundVerifyEngineBatch`) and pins `runtime.max_batch_size=1`. Admission, decode and KV pool stay at one chat: `admitted_batch` (`src/exec/expert_batch_policy.h`) feeds the scheduler (`src/runtime/engine.cpp`), `config_.kv_ceiling_seqs` and `ShadowPlanProbe::kv_seqs` -> `ConcurrencyLimits::kv_seqs` (`src/runtime/engine_kv_cache_init.cpp`, `src/runtime/plan_shadow.cpp`, `src/memory/plan.cpp`). Declines (chain kept): `speculative.mtp_concurrent=false`, explicit `speculative.mtp_k`, `runtime.deterministic=true`, `speculative.batch_verify=true`, `speculative.mtp_batch_verify=0`, no head, not a GDN hybrid, configured batch != 1. Qwen3.8-27B-NVFP4-vllm, `--max-batch 1`, `kv_cache.host_spill_mb=16384`, agent replay (`tools/analysis/agent_turn_replay.py --serial --seed 1000`, 3 chats x 30 turns in turn): lane 205.58 tok/s decode per chat (198.50 wall, 65378 output tok, 90 requests, 3.443 tok per step, stderr hits 0) vs chain 69.28 (68.46 wall, 63300 output tok); MTP off on the B-73 tree 97.36 (`one_chat_default.txt`). Plain prompts 4 x 1024 tokens, lane vs chain: greedy 179.72 vs 129.49, sampled 187.43 vs 126.23 tok/s, lane repeat 179.85 / 187.55; 2 requests at once run serially (req_s 11.78 / 5.64, queue_s 5.64, `plain_probe.txt`). KV plan 8192 blocks 2304 MiB in both arms ("admission 1 of engine batch 2"); before `kv_seqs` the lane planned 16384 blocks 4608 MiB, GPU used 21806 -> 21102 MiB, chain 21178 (`kv_plan.txt`). Tests `MtpAuto.OneChatHybridTakesTheBoundVerifyBatch`, `MtpAuto.BoundVerifyBatchIsForAOneChatAutoHybridOnly` (`tests/test_mtp_auto.cpp`), `ExpertBatchPolicy.AdmittedBatchIsTheSmallerOfEngineBatchAndDecodeCap`, `ShadowPlan.CappedAdmissionSizesKvForTheAdmittedSequences` (`tests/test_memory_plan.cpp`); Reload: `load_model_into_state` wrote the resolved keys (`speculative.mtp_k`, `ngram`, `mtp_draft_vocab`, `runtime.max_batch_size`) into `state.runtime_config`, the next load read them as explicit and declined the lane; `mtp_auto_restore_startup` (`tools/common/mtp_auto.cpp`, `ServerState::startup_runtime_config`) resolves every load against the startup config: `--max-batch 1` after `/admin/suspend` + `/admin/resume`, unfixed -> fixed, greedy 133.63 -> 179.80, sampled 123.66 -> 187.72 tok/s (before the resume 179.66 / 187.34 and 179.93 / 187.75), default server 179.22 -> 178.95 / 186.71 -> 187.56 unfixed (`reload.txt`); swap from a head-less model (auto -> `mtp_k=0` kept) is code reading, not measured. Tests `MtpAuto.ReloadResolvesAgainstTheStartupConfig`. 8 of 8 CPU mutants red (`mutants.txt`). Not built / open: imp-cli keeps the chain (batch 1 hardcoded in `tools/imp-cli/main.cpp`, `mtp_auto_after_load` has no serving parameter): closed by B-75; the weight-upload reserve is still sized at engine batch 2 (`src/runtime/engine_weight_upload.cpp:278-286`), not measured on a card below 32 GB. Evidence `docs/audit/qwen38_single_chat_bound_verify_2026_11_12/` (final-tree gates `gates.txt`) |
| B-75 | DONE | imp-cli chat runs (`--prompt`, `--prompt-file`, `--interactive`) on a GDN hybrid take the B-74 lane: `tools/imp-cli/main.cpp` sets `chat_run` (no `--bench`, no `--perplexity`, no `--mtp-spec-decode`), `engine_batch = mtp_auto_engine_batch(...)` and `config.max_batch_size = engine_batch`; `mtp_auto_after_load` (`tools/common/mtp_auto.cpp`) takes the `MtpAutoServing` of that batch (default `SingleStream`). Opt-out `speculative.mtp_concurrent=false`; `--bench`, `--perplexity`, `--mtp-spec-decode` stay engine batch 1. Qwen3.8-27B-NVFP4-vllm, imp-cli one-shot, 4 plain prompts x 1024 tokens, one process per prompt, rate = sum(tokens - 1) / sum(decode ms), chain -> lane: run a (lane first) greedy 126.13 -> 167.61, sampled (temperature 0.7) 123.68 -> 182.11 tok/s; run b (chain first) greedy 126.01 -> 167.43, sampled 123.39 -> 182.60. Lane log: "engine batch 2", `mtp_k` auto -> 7, draft head rows [0, 98304) 270.0 MiB, KV 8192 blocks, "admission 1 of engine batch 2"; chain: auto -> 2, 682.0 MiB, KV 8192 blocks; stderr hits 0 in all arms. `--interactive`, 4 turns, greedy: 7421 / Brimble counts 4 / 4 in both arms; shutdown VRAM audit peak 21521.8 MiB lane vs 21401.6 chain (`cli.txt`). Test `MtpAuto.AfterLoadTakesTheServingOfTheEngineBatch`; mutant `noserving2` red (`cli.txt`). Not measured: `--json`, image / video prompts, `bench.generate` and long prompts on the lane (`make test-niah` drives imp-server, not imp-cli); the `main.cpp` wiring has no unit test, its evidence is the log lines. B-74 open item "default server KV plan 11014 -> 14321 blocks after a resume": library reserve, not a defect (first load plans with the 1536 MiB constant and replans to 14414 after measuring 367 MiB; the resume plans with 367 MiB from the cache, `reload.txt`). Evidence `docs/audit/qwen38_single_chat_bound_verify_2026_11_12/` (`cli.txt`, final-tree gates `gates_cli.txt`) |
| B-76 | DONE | MTP head body in E4M3 on the bound verify lane: nsys of the imp-cli lane (FP16 body) puts the body GEMV `gemv_fp16_kernel` at 5677 launches, avg 65404.7 ns, 7.5 % of GPU time; the draft LM head `gemv_nvfp4_kpar_mb_fp32_kernel` 850 launches (`profile.txt`). New key `speculative.mtp_body_fp8` (default true): `mtp_body_fp8_allocate` (`src/compute/mtp_forward.cu`, `quantize_fp8_rows_async`) keeps a per-row E4M3 copy of a dense head body (fc, q/k/v/o, MLP), `mtp_body_proj` reads it through `gemv_fp8_rowscale_f16` at <= 32 rows, else FP16; priced as `MtpWorkspaceShape::body_fp8_bytes` (`src/compute/mtp_workspace_bytes.h`) only when `mtp_bound_verify_rows(...) > 0` (`src/runtime/engine_spec_mtp.cpp`). Draft-only, verification stays exact. Qwen3.8-27B-NVFP4-vllm, default server, agent replay 1 chat (3 sessions x 30 turns, `--serial --seed 1000`). Paired (`diagnostics.mtp_verify_pin=7`, `speculative.mtp_k=7`, `runtime.deterministic=true`, prototype key opt-in): FP16 204.28 vs E4M3 218.78 tok/s decode per chat, 52107 output tokens each, 3 of 3 session md5 equal; tok/step 3.561 vs 3.555, head 4.151 vs 2.968 ms, fwd 13.220 vs 13.219, tok/ms 0.2050 vs 0.2196 (`paired.txt`). Default mode, unpaired: prototype 205.89 vs 221.12; default-on image 204.18 vs 220.32, head 3.966 vs 2.875 ms (`default_mode.txt`). Copy 406.0 MiB, MTP workspace 366.72 -> 772.72 MiB = the plan row, KV ceiling after the reserve replan 20380 -> 19199 blocks (live pass 14602 -> 13158, `kv_plan.txt`). The mtp_k=2 chain measured neutral with the copy (imp-cli 125.85 / 123.47 vs 128.64 / 120.75 tok/s greedy / sampled), so only the bound lane takes it; imp-cli lane, dev prototype, unpaired: 167.48 -> 198.75 greedy, 182.28 -> 189.62 sampled (`cli.txt`). Final image (code tree 5247f09a): imp-cli lane 204.63 / 189.45 with the copy line in 9 of 9 runs, chain 126.11 / 123.80 with 0 of 9, 0 plan-row WARN; default server 219.95 tok/s per chat, head 2.886 ms (`activity.txt`). Trap: `runtime.deterministic=true` on imp-cli leaves the head unloaded under mtp_k=auto (both arms MTP off). Test `MtpWorkspaceBytes.BodyFp8CopyIsPricedForADenseHead`; 4 of 4 CPU mutants red (`mutants.txt`). Not built: MoE heads keep the FP16 body (`mtp_body_fp8_bytes` 0 with a router), the chain keeps FP16; no GPU unit test of `mtp_body_proj`. Supersedes the B-49 note at depth 7. Evidence `docs/audit/qwen38_mtp_body_fp8_2026_11_12/` (final-tree gates `gates.txt`). 2 and 4 overlapping chats (#3100 image, no `--serial`, `--seed 1000`, unpaired: texts differ), off -> on: 2 chats x 30 turns 190.00 -> 200.20 tok/s decode per chat, wall 307.75 -> 309.04 tok/s; 4 chats x 20 turns 148.22 -> 158.87 per chat, wall 463.57 -> 469.17; step rows `n=4 T=4` tok/step 12.329 -> 12.523, head 3.526 -> 2.713 ms; stderr hits 0 in all arms (`multi_chat.txt`) |
| B-77 | REFUTED | Fewer draft LM-head rows (`speculative.mtp_draft_vocab`) with the E4M3 body (B-76): Qwen3.8-27B-NVFP4-vllm, default server, image imp:bfp8d (#3100 code), paired (`diagnostics.mtp_verify_pin=7`, `speculative.mtp_k=7`, `runtime.deterministic=true`, 1 chat, 3 sessions x 30 turns, `--serial --seed 1000`, 52107 output tokens, 3 of 3 session md5 equal in all arms). 98304 / 65536 / 49152 rows: NVFP4 copy 270.0 / 180.0 / 135.0 MiB, 219.08 / 216.48 / 212.74 tok/s decode per chat, tok/step 3.556 / 3.453 / 3.361, head 2.971 / 2.684 / 2.529 ms, tok/ms 0.2199 / 0.2173 / 0.2136. Fewer rows lose more acceptance than they save head time; 98304 stays (`docs/audit/qwen38_mtp_body_fp8_2026_11_12/draft_vocab.txt`) |
| B-78 | REFUTED | Verify ladder step `kMtpBoundVerifyStreakStep` 2 -> 3 (`src/runtime/spec_batch_verify_state.h`; 4 -> 7 after a full accept instead of 4 -> 6 -> 7), same paired setup: step 2 on imp:bfp8d (#3100 code) vs step 3 on prototype image imp:lad3 (one-line change, not shipped), 52107 output tokens, 3 of 3 session md5 equal: 218.95 vs 219.05 tok/s decode per chat, tok/step 3.556 vs 3.581, head 2.979 vs 3.073 ms, drafted/step 4.76 vs 4.92, tok/ms 0.2198 vs 0.2199: neutral, step 2 stays (`docs/audit/qwen38_mtp_body_fp8_2026_11_12/ladder_step3.txt`) |
| B-79 | MEASURED (default 16384 stays) | MTP head KV window `speculative.mtp_kv_window` 4096 to 32768 with the E4M3 body (B-76): Qwen3.8-27B-NVFP4-vllm, default server, image imp:bfp8d (#3100 code), paired (`diagnostics.mtp_verify_pin=7`, `speculative.mtp_k=7`, `runtime.deterministic=true`, 1 chat, 3 sessions x 30 turns, `--serial --seed 1000`, 52107 output tokens, 3 of 3 session md5 equal in all arms; longest prompt 31573 tokens). 4096 / 8192 / 16384 / 32768: 207.09 / 215.50 / 218.96 / 216.75 tok/s decode per chat, steps 15797 / 15048 / 14649 / 14649, tok/step 3.298 / 3.462 / 3.556 / 3.556, head 2.631 / 2.792 / 2.970 / 3.150 ms, tok/ms 0.2079 / 0.2163 / 0.2198 / 0.2176, MTP workspace 580.72 / 644.72 / 772.72 / 1028.72 MiB. Smaller windows lose acceptance faster than they save head time; 32768 drafts the same steps as 16384 at a higher head cost. Attribution only, 50k-context step profile (`runtime.no_pdl=true`): `mtp_attn_kv_scan_rows_split_kernel` 0.137 ms per step, `sparse_score_blocks_kernel` 0.156 ms per step of a 4.874 ms step (`docs/audit/qwen38_mtp_body_fp8_2026_11_12/kv_window.txt`) |
| B-80 | DONE | GQA-grouped MTP head KV scan (`src/compute/mtp_attn_scan.cu`): (a) `scan_scores` / `scan_pv` load `kScanUnroll`=4 rows per warp before reducing, per-row sum order unchanged; (b) `mtp_attn_kv_scan_rows_group_kernel<G>` (CTA per kv head x split x row, the G query heads of a kv head share each K/V row load, 256 threads, `kGroupSplits`=64 splits in the existing partial buffer for <= `kScanGroupRows`=4 rows, G in {2, 4, 6, 8}, head_dim <= 256, smem <= 48 KiB at the ring size), dispatched first in `mtp_attn_scan_slots`; captured head passes patch its smem through `mtp_scan_node_smem` (`src/compute/mtp_pass_tables.cpp`). ncu, imp-cli lane, Qwen3.8-27B-NVFP4-vllm, 27686-token prompt, 1 row: split kernel 132.96 us (DRAM 28.62 %, L2 47.02 %, 18.53 cycles per issued instruction), unroll only 108.32 us, grouped 53.54 us (DRAM 71.12 %, 9.09 cycles, grid 256); 2 rows 177.92 -> 165.50 us with unroll only (`ncu.txt`). Agent replay 1 chat, paired (pin 7, `mtp_k` 7, deterministic, 52107 output tokens, 3 of 3 session md5 equal): 218.63 -> 220.97 tok/s decode per chat, head 2.981 -> 2.814 ms, tok/step 3.556 both, steps 14649 vs 14653, tok/ms 0.2195 -> 0.2218; default mode, unpaired 219.78 -> 222.53 (`replay.txt`). 50k context, attribution only (`runtime.no_pdl=true`): scan 136.62 -> 51.26 us per call (`profile.txt`; the profile tool's default anchor misses the grouped kernel, re-run with `--anchor '^mtp_attn_kv_scan_rows(_split|_group)?_kernel$'`). Mutants (`mutants.txt`): head map, `scan_group` off, `mtp_scan_node_smem` ignoring groups, grouped PV score index red on the existing `MtpFeedBatchTest.*`; the unroll guard `<` -> `<=` survived and is red on the new `MtpFeedBatchTest.AttnScanIgnoresTheRowPastTheWindow`. Preflight forced two code changes after the measurements: P.V and the partial store moved into `group_pv_store` (kernel CCN 29 > 25) and the head index widened to int64 (clang-tidy `bugprone-implicit-widening-of-multiplication-result`); head-map and PV-index mutants red again on that code, final-image replay in `gates.txt`. Coordinator job name "B-77"; ID B-80 because B-77..B-79 were taken. Evidence `docs/audit/qwen38_mtp_head_scan_2026_11_12/` (final-tree gates `gates.txt`) |
