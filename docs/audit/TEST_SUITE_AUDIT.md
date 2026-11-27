<!--
layer: L2
audience: kernel-devs
verified: 2026-11-27
commit: 86e5cf30
-->

# imp - test-suite audit

Owner assignment 2026-11-27: every test justifies its existence, the suite runs faster, regression detection does not drop. Phase 1 (this section) is read-only: no test, build file or CI file changed. Tree: `86e5cf30` (origin/main after #3347). Raw data: `/mnt/data/scratch/tsa/` (`logs/`, `junit/`, `bld/ninja_log_*.txt`, `scripts/`, `tables/`). Prior work reused, not redone: [`SETTLED.md`](SETTLED.md) S-2, S-28; [`MUTATION_BASELINE.md`](MUTATION_BASELINE.md); [`BASELINE_TESTLIST_2026_10_15.txt`](BASELINE_TESTLIST_2026_10_15.txt).

## Phase 1: inventory and baseline

Status 2026-11-27 00:06Z: CPU part done. **GPU part pending**: `gpu-busy-check.sh` rc 1 (60 % median util), `/tmp/gpu-lock` held by `imp-59` (`imp-int10 validation`, ends ~02:42Z); no acquire, no GPU job run. Runner for the GPU part: `/mnt/data/scratch/tsa/scripts/gpu_runs.sh` (8 binaries + the 3 extra `test-gpu-suite` lines + `make verify-fast`, 3x each, XML to `gpu/`), then `analyze.py` + `suite_table.py` regenerate every table below.

### Counts

| Item | Count | Command |
|---|---:|---|
| ctest entries | 23 (13 `unit`, 6 `gpu`, 4 `perf`) | `ctest -N -L <label>` in a configured build |
| gtest binaries | 9 (8 in `imp-tests`, plus opt-in `test-hf-network`) | `CMakeLists.txt` `imp_add_test_module` |
| gtest cases | 4476 (core 2513, text 313, compute 377, attention 214, quant 367, kv 145, moe-gdn 224, e2e 320, hf-network 3) | `<bin> --gtest_list_tests` (`logs/list_*.txt`) |
| gtest suites | 859 | `logs/suite_map.tsv` |
| `DISABLED_` cases | 17 (compute 6, attention 2, quant 5, e2e 4) | `grep DISABLED_ logs/list_*.txt` |
| cases in the CI lane | 2905 (core 2513, text 313, e2e subset 79), 0 skipped in 3 runs | `junit/gtest_*/*.json` |
| cases in no CI lane | 1571 (1396 macros by `tools/check_test_lanes.py --report`) | |
| test sources registered | 491 under `tests/` (incl. `tests/bench/`) + 25 from `src/`, `tools/`, `fuzz/` | `logs/src_module_map.tsv` |
| suites reading a checkpoint env var | 41 sure + 25 via a shared helper (file level) | column `weights` below |
| pytest functions in `tests/api/` | 279 (before parametrize) | `grep -c "def test_" tests/api/test_*.py` |
| shell/python test scripts in `tests/` | 4 static-gate shell tests, 10 server batteries, 2 manual shell tools | below |

### Lanes

| Lane | Trigger | What runs |
|---|---|---|
| CI `Build`: `ctest -L unit -j4 --timeout 120` | every PR, push to main, nightly 03:17 UTC | `unit_core`, `unit_text`, `unit_e2e_subset` (`_unit_e2e_filter`, 79 cases), 10 `guard_*` scripts |
| CI `Build` + hooks: `scripts/ci_static_gates.sh` | every PR; `make preflight` in both hooks | `tests/test_entrypoint.sh`, `tests/test_gpu_lock.sh`, `tests/test_image_tag.sh`, `tests/test_verify_ab_sha_once.sh`, 31 gate `--selftest` lines |
| CI `Sanitizers` | path filter (`tools/untrusted_input_sources.txt`) + nightly | `test-core`, `test-text` under ASan/UBSan |
| CI `Mock API contract` | every PR | `pytest tests/api -m "not perf and not tools"` against `tests/api/mock_server.py` |
| CI `Real API contract (model-less)` | every PR | `pytest tests/api -m nomodel` against a model-less `imp-server` |
| CI `Fuzz (nightly, libFuzzer)` | nightly, dispatch | `fuzz_wav`, `fuzz_json_schema`, `fuzz_regex`, `fuzz_gbnf`, 10 min each |
| local `make test-gpu` | by hand; pre-commit with `IMP_HOOK_GPU=1` (modules by path, full on `include/ src/core src/api CMakeLists cmake/`) | `imp-tests` (8 binaries, unfiltered, no model env) + `*DetEvalE2ETest*` (Qwen3-8B-NVFP4, Gemma-4-26B-A4B NVFP4) + `*Qwen38TokenizerParity*` + `KimiTiktokenParityTest.*` |
| local `make test-e2e` | by hand | 7 docker groups with model env (model battery, locks/degeneration/prefix, hybrid restore, batch invariance, Inkling, MTP identity, drafters) |
| local `make verify-fast` | by hand; pre-push with `IMP_HOOK_GPU=1` | fast gtest filter (17 patterns, `scripts/verify.sh:285`), perf, peak VRAM, graphs, PPL drift, server, smoke, Inkling serve smoke, I2 interpose row |
| local `make verify` | by hand | full `imp-tests`, lane split, 2 smokes, 6 decode-route parity rows + the fast rows |
| local, other Make targets | by hand, some in `scripts/check-release.sh` | `test-hybrid-restore`, `test-spec-fidelity`, `test-vision`, `test-server` (13 python/shell batteries), `test-quantize`, `test-quantize-repro`, `test-rerank`, `test-embed`, `test-niah`, `test-agents`, `test-agents-external`, `test-perf` (bench), `check-alloc-interpose`, `asan` |
| `ctest -L gpu` (6), `ctest -L perf` (4) | none: no target, script or workflow invokes them | registration only (`tools/check_test_lanes.py` reads them) |
| no lane | none | `test-hf-network` (3 cases, `IMP_TEST_NETWORK=1`), `tests/test_engine_chat.sh`, `tests/test_server_concurrent.sh` (both "MANUAL tool" headers), `tests/api/elchtest.sh`, `tests/api/run_all_models.sh`, the model-backed half of `tests/api` |

### Baseline: build (test targets, 16 cores)

Throwaway dir `/mnt/data/scratch/tsa/bld/clean` in `imp:toolchain` (dev flags, Release, `-DIMP_BUILD_TESTS=ON`). `libimp` + `libimp_core` prebuilt with ccache (untimed); timed part with `CCACHE_DISABLE=1`, targets `test-core test-text test-compute test-attention test-quant test-kv test-moe-gdn test-e2e test-hf-network` (546 steps clean). Before every run the script waited for no foreign `imp:toolchain` container and no host `docker build`; a run that saw one was repeated. Waits: clean run 1 repeated 3x (peer `docker build` / dev containers at 23:18Z, 23:22Z, 23:26Z), then waited 934 s (23:27:03Z-23:42:37Z), 4th try accepted although a peer `docker build` started again 23:43:18Z (flagged). Runs 2 and 3 clean. `logs/build_timing.log`.

| Measurement | run 1 s | run 2 s | run 3 s | median s |
|---|---:|---:|---:|---:|
| clean (546 steps) | 243.4 (contended) | 227.4 | 206.9 | 227.4 |
| incremental: `touch tests/test_models.h` (35 steps: 30 TUs + relinks) | 26.9 | 18.2 | 16.9 | 18.2 |
| incremental: `touch tests/test_anthropic_effort.cpp` (4 steps, test-core) | 3.7 | 3.8 | 3.7 | 3.7 |

Per-TU times come from the `.ninja_log` slices of clean runs 1, 3 and an extra clean run 4 (222.1 s, uncontended, `logs/build_timing_clean4.log`): ninja recompacted the log at the start of run 2, its slice is empty. TU times are wall seconds under `-j16` load. Sum of TU medians: 3372 CPU-s; links are 0.2-1.9 s per binary, compile dominates.

### Baseline: CI lane (`ctest -L unit -j4 --timeout 120`, no GPU)

Same command as `.github/workflows/ci.yml:296`, against the prep build `bld/main`, in `imp:toolchain` without `--gpus`. JUnit: `junit/ctest_unit_<i>.xml`; per-case gtest JSON: `junit/gtest_<i>/`. All 3 runs uncontended, rc 0, 0 failures, 0 skips. `logs/ci_lane.log`.

| ctest -L unit wall | run 1 s | run 2 s | run 3 s | median s |
|---|---:|---:|---:|---:|
| 13 entries, `-j4` | 10.72 | 11.07 | 10.77 | 10.77 |

The critical path is `guard_makefile_filters` (10.76 s), not a gtest binary: the 2905 CI-lane cases sum to 4.41 s (medians), `unit_core` runs 4.74 s.

| ctest entry | run 1 s | run 2 s | run 3 s | median s |
|---|---:|---:|---:|---:|
| guard_makefile_filters | 10.41 | 11.06 | 10.76 | 10.76 |
| guard_verify_filter | 7.43 | 7.71 | 7.59 | 7.59 |
| unit_core | 4.74 | 4.66 | 4.81 | 4.74 |
| guard_degen_thresholds | 0.35 | 0.35 | 0.34 | 0.35 |
| guard_precommit_filter | 0.09 | 0.08 | 0.08 | 0.08 |
| unit_e2e_subset | 0.28 | 0.08 | 0.08 | 0.08 |
| guard_e2e_lane_split | 0.28 | 0.08 | 0.08 | 0.08 |
| guard_det_suite_filter | 0.23 | 0.07 | 0.07 | 0.07 |
| unit_text | 0.05 | 0.04 | 0.04 | 0.04 |
| guard_unit_skips | 0.03 | 0.02 | 0.02 | 0.02 |
| guard_spec_rr_wiring | 0.01 | 0.01 | 0.01 | 0.01 |
| guard_cancel_teardown | 0.01 | 0.01 | 0.01 | 0.01 |
| guard_lane_filter_copy | 0.01 | 0.01 | 0.01 | 0.01 |

### Baseline: local GPU

**Pending** (GPU held by `imp-59`, see top). Planned rows: `test-core`, `test-text`, `test-compute`, `test-attention`, `test-quant`, `test-kv`, `test-moe-gdn`, `test-e2e` each with `--gtest_output=xml`, `deteval`, `qwen38_tok`, `kimi_tok`, `verify_fast` (timestamped per line for the per-component split), 3 runs each.

### Top 20 by runtime (per case)

CI lane only until the GPU part runs; the GPU cases will displace all of these.

| # | case | binary | lane | median s | 3 runs s |
|---:|---|---|---|---:|---|
| 1 | `FpBitsLegacyParity.Fp16EncodeFloatSweep` | test-core | CI | 2.38 | 2.27 / 2.38 / 2.50 |
| 2 | `FuzzCorpus.Regex` | test-core | CI | 0.57 | 0.57 / 0.58 / 0.57 |
| 3 | `CheckpointLimits.HugeLoopsAreRefusedRatherThanExhaustingMemory` | test-core | CI | 0.25 | 0.25 / 0.23 / 0.26 |
| 4 | `FpBitsLegacyParity.NgramGatherElementAll16M` | test-core | CI | 0.08 | 0.08 / 0.08 / 0.08 |
| 5 | `WeightUploadTraits.MatchesPerFormatUploadersByteForByte` | test-core | CI | 0.08 | 0.08 / 0.08 / 0.08 |
| 6 | `Fp8LmHead.EncodeIsRoundToNearestEvenWithSaturation` | test-core | CI | 0.07 | 0.06 / 0.08 / 0.07 |
| 7 | `FuzzCorpus.GgufLoader` | test-core | CI | 0.06 | 0.06 / 0.08 / 0.06 |
| 8 | `FuzzCorpus.TokenizerJson` | test-core | CI | 0.06 | 0.06 / 0.09 / 0.06 |
| 9 | `FuzzCorpus.MmprojLoader` | test-core | CI | 0.06 | 0.06 / 0.06 / 0.06 |
| 10 | `FuzzCorpus.SafeTensorsLoader` | test-core | CI | 0.06 | 0.06 / 0.06 / 0.05 |
| 11 | `ModelLifetimeGate.ExclusiveWaitsUntilReadersLeave` | test-core | CI | 0.05 | 0.05 / 0.05 / 0.05 |
| 12 | `StrictCliNumbersDeathTest.MalformedValueExitsNamingTheFlag` | test-core | CI | 0.05 | 0.05 / 0.05 / 0.05 |
| 13 | `ModelSwapState.SwapWhileAHandlerHoldsALeaseAnswers503AndFreesNothing` | test-core | CI | 0.05 | 0.05 / 0.05 / 0.05 |
| 14 | `ModelSwapState.SuspendWhileAHandlerHoldsALeaseAnswers503AndFreesNothing` | test-core | CI | 0.05 | 0.05 / 0.05 / 0.05 |
| 15 | `Gemma4AudioFeatures.SoftTokenCountMatchesProcessorArithmetic` | test-core | CI | 0.04 | 0.04 / 0.04 / 0.04 |
| 16 | `ModelLifetimeGate.TimesOutWithReadersInsideAndStaysUsable` | test-core | CI | 0.03 | 0.03 / 0.03 / 0.03 |
| 17 | `ServerArgsDeathTest.ResponsesStoreFlagsRejectGarbage` | test-core | CI | 0.03 | 0.15 / 0.03 / 0.03 |
| 18 | `HFConfigGolden.EveryFixtureMatchesItsGolden` | test-core | CI | 0.03 | 0.05 / 0.03 / 0.03 |
| 19 | `ServerArgsDeathTest.MaxQueuedTokensRejectsGarbage` | test-core | CI | 0.03 | 0.03 / 0.03 / 0.02 |
| 20 | `ServerArgsDeathTest.IdleUnloadSecondsRejectsGarbage` | test-core | CI | 0.03 | 0.03 / 0.02 / 0.03 |

### Top 10 by compile cost

Per TU (median of 3 clean runs, wall s under -j16), per link step, and per binary.

| # | TU | binary | median s | 3 runs s |
|---:|---|---|---:|---|
| 1 | `tests/test_sampling_params_golden.cpp` | test-core | 33.6 | 33.6 / 29.5 / 40.2 |
| 2 | `tests/test_constraint_validation.cpp` | test-core | 33.1 | 33.6 / 28.8 / 33.1 |
| 3 | `tests/test_sse_stream_utils.cpp` | test-core | 31.4 | 40.8 / 31.4 / 30.0 |
| 4 | `tests/test_server_request_limits.cpp` | test-core | 26.5 | 29.6 / 26.5 / 24.5 |
| 5 | `tests/test_quant_integration.cu` | test-quant | 26.0 | 26.0 / 23.1 / 32.9 |
| 6 | `tests/test_hf_fetch.cpp` | test-core | 23.2 | 27.4 / 22.3 / 23.2 |
| 7 | `tests/test_spec_usage.cpp` | test-core | 23.2 | 31.3 / 23.2 / 22.2 |
| 8 | `tests/test_capability_refusal.cpp` | test-core | 22.9 | 26.0 / 22.9 / 22.7 |
| 9 | `tests/test_logprobs_shapes.cpp` | test-core | 22.7 | 31.2 / 22.7 / 20.6 |
| 10 | `tests/test_moe_executor.cu` | test-moe-gdn | 21.9 | 35.9 / 21.9 / 20.1 |

| # | link step | median s | 3 runs s |
|---:|---|---:|---|
| 1 | test-core (host link/archive) | 1.7 | 4.7 / 1.6 / 1.7 |
| 2 | test-kv (host link/archive) | 0.6 | 0.9 / 0.6 / 0.5 |
| 3 | test-quant (host link/archive) | 0.6 | 1.2 / 0.6 / 0.6 |
| 4 | test-moe-gdn (host link/archive) | 0.6 | 1.3 / 0.6 / 0.4 |
| 5 | test-compute (host link/archive) | 0.5 | 1.0 / 0.5 / 0.5 |
| 6 | test-compute (device link) | 0.5 | 0.6 / 0.5 / 0.4 |
| 7 | test-quant (device link) | 0.5 | 0.5 / 0.5 / 0.4 |
| 8 | test-e2e (host link/archive) | 0.5 | 0.7 / 0.5 / 0.3 |
| 9 | test-attention (host link/archive) | 0.3 | 0.3 / 0.3 / 0.2 |
| 10 | test-attention (device link) | 0.2 | 0.2 / 0.3 / 0.2 |

| binary | TUs | TU compile s (sum of medians, CPU) | link s (median) |
|---|---:|---:|---:|
| test-core | 256 | 1142.7 | 1.9 |
| test-compute | 57 | 592.3 | 1.1 |
| test-quant | 65 | 589.5 | 1.0 |
| test-e2e | 62 | 438.5 | 0.6 |
| test-moe-gdn | 25 | 224.5 | 0.8 |
| test-attention | 15 | 150.9 | 0.5 |
| test-kv | 17 | 120.2 | 0.8 |
| test-text | 18 | 87.5 | 0.3 |
| imp_test_kernels | 10 | 23.5 | 0.0 |
| test-hf-network | 1 | 2.6 | 0.2 |

### Anomalies

| # | Finding | Evidence |
|---:|---|---|
| A1 | `ctest -L gpu` (6 entries) and `ctest -L perf` (4) are invoked by no Make target, script or workflow | `git grep -nE "\-L (gpu|perf)"`: only the `CMakeLists.txt:1767` comment |
| A2 | `perf_e2e` filter `*Perf*:*Bench*:*Throughput*` matches 0 cases in `test-e2e` (gtest reports PASSED) | `test-e2e --gtest_list_tests --gtest_filter=...` = 0; compute 13, attention 4, quant 22 (12 of these 39 are `DISABLED_`) |
| A3 | `make test-all` and `scripts/smoke_test.sh:43` run `test-gdn`, a binary no CMake target builds | `Makefile:228`, `Dockerfile:134` copies it only `[ -f build/test-gdn ]` |
| A4 | `make test-golden` runs nothing (one `echo`) | `Makefile:539` |
| A5 | 17 `DISABLED_` cases; the skill says 7 | `logs/list_*.txt`; `.claude/skills/building-and-testing/SKILL.md:118` |
| A6 | Never run in any lane: `test-hf-network` (3 cases, opt-in), `tests/test_engine_chat.sh`, `tests/test_server_concurrent.sh`, `tests/api/elchtest.sh`, `tests/api/run_all_models.sh` | `git grep` finds no invocation site outside the files themselves |
| A7 | The model-backed half of `tests/api` (not `nomodel`, not `mock`) runs only by hand | `ci.yml:477` runs `-m nomodel`; `ci.yml:484` "State the uncovered half" |
| A8 | CI lane wall = one guard script: `guard_makefile_filters` 10.76 s and `guard_verify_filter` 7.59 s vs `unit_core` 4.74 s | `tables/ci_lane.md` |
| A9 | 22 suites are benchmarks (time kernels, print numbers): attention 3, compute 7, quant 12, 5 of them marked DISABLED; assignment rule: own target, not deleted | `BENCH:` rows below |
| A10 | 2 suites without `EXPECT_`/`ASSERT_`: `FuzzCorpus` (a crash is the failure), `MoeRouterFusedBench` (DISABLED bench) | `NO-ASSERT:` rows below |
| A11 | 14 suites use sleeps or wall-clock thresholds (flake candidates for Phase 2): test-core `EngineMetricsCell`, `GgufFaultInjection`, `InflightGate`, `ModelContext`, `ModelLifetimeGate`, `QueuedTokenGate`, `RateLimiter`; test-kv `KVCacheGrowAdmissionTest`, `KVCacheGrowTest`, `RecurrentSnapshotStoreTest`; test-e2e `DeviceFaultSignalTest`, `LoraHotSwap`, `ServingSignalsTest`; test-compute `SamplingAdvancedTest` | `TIMING:` rows below |
| A12 | Flakes in the 3 CI-lane runs: none (rc 0, 0 failures x3). GPU flakes: pending | `junit/ctest_unit_*.xml` |
| A13 | Build cost is in compilation, not linking: test-core 256 TUs = 1142.7 CPU-s, its link 1.7 s; 8 of the top 10 TUs are test-core `.cpp` files (22.7-33.6 s each) | `tables/top_compile.md` |

### Per-suite table

Columns: runtime = sum over the suite's cases of the per-case median of 3 runs, suffix `C` = CI lane, `G` = local GPU, "GPU pending" = not measured yet. build = median clean-build seconds of the TU(s) the suite lives in (a TU shared by several suites is counted in each). what it checks = one line per suite from the TEST bodies (names + first assertions, read by two Haiku agents, spot-checked); prefixes `BENCH:`, `NO-ASSERT:`, `TIMING:` are their flags. GPU = the binary is a GPU module. weights: `n` for every suite that ran in the CI lane (no model mount, no skips); `y: <VAR>` = the suite's file reads that checkpoint/reference variable; `y?` = the file includes `tests/test_models.h` or `tests/inkling_fixture.h` (file level, resolved by the GPU run's skip column).

| Test (suite) | module | cases | file | runtime s | build s | what it checks | GPU | weights (env) | skipped |
|---|---|---:|---|---:|---:|---|---|---|---:|
| ApaPrefillTest | test-attention | 8 | test_attention_ra2.cu | GPU pending | 9.3 | APA prefill: rerun bit-identity, all-hot exact pass, paged KV equals flat, workspace carve | y | n |  |
| AttentionChunkedTest | test-attention | 5 | test_attention_chunked.cu | GPU pending | 8.5 | cuBLAS chunked prefill attention: square, offset causal mask, GQA, sliding window vs naive | y | n |  |
| AttentionCrossPathTest | test-attention | 11 | test_attention_crosspath.cu | GPU pending | 11.1 | Attention golden configs: MHA/GQA, softcap, sliding window, 1-2 token edge cases | y | n |  |
| Fa2Hd512Test | test-attention | 4 | test_attention_fa2_hd512.cu | GPU pending | 10.6 | FA2 hd=512 causal GQA prefill attention vs CPU reference | y | n |  |
| FmhaFA2Dense2CtaTest | test-attention | 5 | test_fmha_fp8.cu | GPU pending | 10.1 | FA2 dense 2-CTA FP8-KV prefill vs reference over causal, odd, window, chunked shapes | y | n |  |
| FmhaFA2Hd256Test | test-attention | 14 | test_fmha_fp8.cu | GPU pending | 10.1 | FA2 hd=256 PV-f16 prefill vs reference over causal, GQA, window, softcap; WMMA micro-bench | y | n |  |
| FmhaFA2HeavyFirstTest | test-attention | 3 | test_fmha_fp8.cu | GPU pending | 10.1 | FA2 heavy-first CTA order: outputs bit-identical across causal, chunked, odd lengths | y | n |  |
| FmhaFA2PvF16Test | test-attention | 13 | test_fmha_fp8.cu | GPU pending | 10.1 | FA2 PV-f16 accumulate prefill vs reference: causal, GQA, window, softcap, chunk shapes | y | n |  |
| FmhaFA2Test | test-attention | 21 | test_fmha_fp8.cu | GPU pending | 10.1 | FA2 FP16-QK prefill vs reference: causal, GQA, window, chunk offsets, occupancy bands | y | n |  |
| FmhaHd512Test | test-attention | 9 | test_attention_fmha_hd512.cu | GPU pending | 10.8 | WMMA FMHA hd=512 vs reference at Gemma-4 geometry; sliced cuBLAS parity; disabled benches | y | n |  |
| FmhaSm120Test | test-attention | 24 | test_attention_fmha_sm120.cu | GPU pending | 9.5 | sm_120 FMHA hd 64/96/128/256 vs reference: GQA, sliding window, softcap, chunked q_offset | y | n |  |
| GptOssSinkRef | test-attention | 9 | test_gpt_oss_sinks_ref.cu | GPU pending | 9.9 | Sink logits vs fp64 reference: no-sink softmax, FMHA, FA2 hd64, paged decode, streaming | y | n |  |
| INT4QuantTest | test-attention | 1 | test_paged_attention.cu | GPU pending | 11.7 | INT4 KV nibble pack/unpack roundtrip over signed range [-8, 7] | y | n |  |
| PagedAttentionINT4Test | test-attention | 1 | test_paged_attention.cu | GPU pending | 11.7 | Paged decode with INT4 KV vs FP32 CPU reference, single head | y | n |  |
| PagedAttentionReduce | test-attention | 3 | test_attention_paged_reduce.cu | GPU pending | 10.1 | Paged split-K reduce vs fp64 merge; staged and unstaged paths bit-identical; Q8 epilogue | y | n |  |
| PagedAttentionTest | test-attention | 24 | test_paged_attention.cu | GPU pending | 11.7 | Paged decode FP16 KV vs CPU reference: blocks, GQA, batch lengths, split-K, sinks, HD512 | y | n |  |
| PagedF16Decode | test-attention | 1 | test_attention_paged_oracle.cu | GPU pending | 15.9 | BENCH: F16 paged decode serving-shape microbench (times kernels, prints numbers) | y | n |  |
| PagedF16Multitok | test-attention | 1 | test_attention_paged_oracle.cu | GPU pending | 15.9 | F16 multitok paged attention vs reference on split-K and direct routes | y | n |  |
| PagedFp8Decode | test-attention | 1 | test_attention_paged_oracle.cu | GPU pending | 15.9 | BENCH: FP8 paged decode serving-shape microbench (times kernels, prints numbers) | y | n |  |
| PagedFp8Multitok | test-attention | 2 | test_attention_paged_oracle.cu | GPU pending | 15.9 | FP8 multitok paged decode vs reference: single split and grouped split rows | y | n |  |
| PagedNvfp4Decode | test-attention | 1 | test_attention_paged_oracle.cu | GPU pending | 15.9 | BENCH: NVFP4 paged decode long-context microbench (times kernels, prints numbers) | y | n |  |
| PagedNvfp4Mma | test-attention | 1 | test_attention_paged_oracle.cu | GPU pending | 15.9 | NVFP4 MMA split-K decode vs fp64 reference over GQA ratios and HD 128/256 | y | n |  |
| PagedNvfp4Multitok | test-attention | 1 | test_attention_paged_oracle.cu | GPU pending | 15.9 | NVFP4 four-token paged kernels vs reference on both split-K routes | y | n |  |
| PagedOracle | test-attention | 18 | test_attention_paged_oracle.cu | GPU pending | 15.9 | Paged oracle: HD128/HD256 sweeps; sinks and window over evicted blocks, all KV dtypes | y | n |  |
| PagedSplitKFallback | test-attention | 1 | test_attention_paged_oracle.cu | GPU pending | 15.9 | Split-K phase-1 launch failure falls back; long context matches split-K and reference | y | n |  |
| QsaIndexer | test-attention | 5 | test_qsa_indexer.cu | GPU pending | 10.1 | QSA indexer: dense-paged equivalence, CPU top-k selection, query prep, split-K, GQA paths | y | n |  |
| Ra2PrefillTest | test-attention | 6 | test_attention_ra2.cu | GPU pending | 9.3 | RA2 FP4 prefill attention vs FP32 CPU oracle: hd 64/128/256, chunk continuation, declines | y | n |  |
| SparseAttnE2E | test-attention | 3 | test_sparse_attn_select.cu | GPU pending | 11.9 | Sparse attention select end to end: identity table bit-identical; MLA budget keeps needle | y | n |  |
| SparseMinMaxTest | test-attention | 8 | test_sparse_attn_select.cu | GPU pending | 11.9 | Sparse key min/max and mean/std block metadata: prefill, decode, rollback, FP8, NVFP4 | y | n |  |
| SparsePrefillGeometry | test-attention | 1 | test_sparse_prefill.cu | GPU pending | 9.6 | Sparse prefill past-token tail geometry (namespace helper) keeps the tail tokens | y | n |  |
| SparsePrefillTest | test-attention | 2 | test_sparse_prefill.cu | GPU pending | 9.6 | Sparse prefill compacted block table: full budget bit-identical, sink/recent/needle order | y | n |  |
| SparseScoreMeanStd | test-attention | 1 | test_sparse_attn_select.cu | GPU pending | 11.9 | Sparse block selection: outlier page loses its budget slot to the consistent page | y | n |  |
| SparseScoreSplit | test-attention | 1 | test_sparse_attn_select.cu | GPU pending | 11.9 | Sparse block scores at Qwen3.8-27B shape vs reference (split score path) | y | n |  |
| SparseSelectTest | test-attention | 5 | test_sparse_attn_select.cu | GPU pending | 11.9 | Sparse block top-k: identity under budget, ascending order, negative scores, spec rows | y | n |  |
| ActivationTest | test-compute | 14 | test_activation.cu | GPU pending | 8.6 | SwiGLU, GELU, GeGLU vs CPU reference: FP32/FP16, vector/tail paths, Inf input, NVFP4 | y | n |  |
| CudaFaultSignalTest | test-compute | 2 | test_cuda_fault_signal.cu | GPU pending | 8.2 | Sticky vs non-sticky CUDA illegal-address faults: death-test exit, clear-and-return path | y | n |  |
| DrafterNorms | test-compute | 2 | test_drafter_norm_primitives.cu | GPU pending | 8.5 | Drafter head-norm with RoPE and row RMSNorm: per-row positions, weight types vs reference | y | n |  |
| EmbeddingTest | test-compute | 9 | test_embedding.cu | GPU pending | 7.4 | Embedding lookup FP32/FP16 vs host table: boundary ids, large vocab, device-side token ids | y | n |  |
| ExecutorKernelsTest | test-compute | 13 | test_executor_kernels.cu | GPU pending | 10.2 | Elementwise add, bias, scale, residual-RMSNorm kernels vs host values; decode pipeline | y | n |  |
| FP8GemmTest | test-compute | 13 | test_fp8_gemm.cu | GPU pending | 14.4 | FP8 and cuBLASLt GEMM/GEMV vs fp64 reference: rowscale heads, quant roundtrip, M-invariant | y | n |  |
| FP8ProjStreamTest | test-compute | 2 | test_fp8_proj_stream.cu | GPU pending | 9.4 | Streamed FP16 FP8 projection bit-equal to split-K reference; Sweep prints timings | y | n |  |
| FfnGraphCacheGpu | test-compute | 2 | test_ffn_graph_cache_gpu.cu | GPU pending | 8.0 | FFN graph cache: capture refusal runs eager and disables graphs; evicted exec updated | y | n |  |
| Fp32ResidualNorm | test-compute | 5 | test_fp32_residual_norm.cu | GPU pending | 8.0 | Fused fp32 residual+RMSNorm vs norm-then-quant chain; row bits M-independent; refusals | y | n |  |
| GatedResidualDecode | test-compute | 5 | test_gated_residual_decode.cu | GPU pending | 8.3 | Gated residual decode read/write vs unfused reference; refuses unsupported shapes | y | n |  |
| GemmCaptureProbe | test-compute | 2 | test_gemm_capture_probe.cu | GPU pending | 11.5 | GEMM workspace retaken after arena reset; cold shape inside graph capture stays valid | y | n |  |
| GemmCaptureSm120 | test-compute | 5 | test_gemm_capture_fp16_sm120.cu | GPU pending | 10.3 | sm_120 FP16 GEMM vs reference at several shapes incl narrow N; cuBLASLt compare | y | n |  |
| GemmDP4ATest | test-compute | 3 | test_gemm_dp4a.cu | GPU pending | 11.8 | DP4A GEMV Q8_0 and Q6_K vs dequant reference; fused SwiGLU Q8_1 quantize | y | n |  |
| GemmF16NarrowPrefill | test-compute | 7 | test_gemm_f16_narrow_prefill.cu | GPU pending | 12.4 | Narrow-N FP16 prefill GEMM vs reference, launch determinism, refusals, cuBLASLt bench | y | n |  |
| GemmF16NarrowSmallM | test-compute | 5 | test_gemm_f16_narrow_smallm.cu | GPU pending | 10.9 | Narrow-N small-M FP16 GEMM vs reference, determinism, refusals; graph-replay bench | y | n |  |
| GemmF16SplitK | test-compute | 1 | test_gemm_f16_tile.cu | GPU pending | 13.0 | Cluster split-K FP16 GEMM vs cuBLASLt over shapes | y | n |  |
| GemmF16TileBench | test-compute | 1 | test_gemm_f16_tile.cu | GPU pending | 13.0 | BENCH: FP16 tile GEMM vs cuBLASLt wall-time bench (times kernels, prints numbers) | y | n |  |
| GemmF16TileTest | test-compute | 6 | test_gemm_f16_tile.cu | GPU pending | 13.0 | FP16 tile GEMM vs cuBLASLt over shape/variant params; fp32 accumulate within rounding | y | n |  |
| GemmGroupedNvfp4Tile | test-compute | 1 | test_gemm_nvfp4_f16_tile.cu | GPU pending | 14.1 | Grouped NVFP4 tile GEMM: each expert's rows bit-equal to the single-expert GEMM | y | n |  |
| GemmKernelRegistryTest | test-compute | 17 | test_gemm_kernel_registry.cu | GPU pending | 12.2 | GEMM registry: NoMatch keys, GGUF Q8_0/Q6_K/dequant paths vs direct, scratch refusal | y | n |  |
| GemmMxfp4RepeatTest | test-compute | 2 | test_gemm_mxfp4_f16_tile.cu | GPU pending | 14.2 | MXFP4 FP16 tile GEMM bit-identical across repeated runs | y | n |  |
| GemmMxfp4TileBench | test-compute | 1 | test_gemm_mxfp4_f16_tile.cu | GPU pending | 14.2 | BENCH: MXFP4 tile GEMM vs dequant+GEMM wall-time bench | y | n |  |
| GemmMxfp4TileTest | test-compute | 8 | test_gemm_mxfp4_f16_tile.cu | GPU pending | 14.2 | MXFP4 tile GEMM bit-equal to dequant plus FP16 tile GEMM over variants | y | n |  |
| GemmNvfp4TileBench | test-compute | 1 | test_gemm_nvfp4_f16_tile.cu | GPU pending | 14.1 | BENCH: NVFP4 tile GEMM wall-time bench on Flint attention shapes | y | n |  |
| GemmNvfp4TileQuant | test-compute | 1 | test_gemm_nvfp4_f16_tile.cu | GPU pending | 14.1 | NVFP4 tile GEMM vs dequantize_nvfp4 reference after quantize roundtrip | y | n |  |
| GemmNvfp4TileTest | test-compute | 7 | test_gemm_nvfp4_f16_tile.cu | GPU pending | 14.1 | NVFP4 tile GEMM bit-equal to exact-F16-weight tile; scales cover E4M3 subnormals | y | n |  |
| GemmQ45kRepeat | test-compute | 1 | test_gemm_q45k_f16_tile.cu | GPU pending | 14.0 | Q4_K-to-F16 tile GEMM: repeated runs bit-identical on a multi-wave shape | y | n |  |
| GemmQ45kTileBench | test-compute | 1 | test_gemm_q45k_f16_tile.cu | GPU pending | 14.0 | BENCH: Q4_K tile GEMM vs IMMA with flushed cache (times kernels, prints numbers) | y | n |  |
| GemmQ45kTileTest | test-compute | 10 | test_gemm_q45k_f16_tile.cu | GPU pending | 14.0 | Q4_K-to-F16 tile GEMM bit-equal to dequant plus F16 tile GEMM over variants | y | n |  |
| GemmQ6kRepeatTest | test-compute | 2 | test_gemm_q6k_f16_tile.cu | GPU pending | 14.6 | Q6_K-to-F16 tile GEMM: repeated runs bit-identical on a multi-wave shape | y | n |  |
| GemmQ6kTileBench | test-compute | 1 | test_gemm_q6k_f16_tile.cu | GPU pending | 14.6 | BENCH: Q6_K tile GEMM vs dequant plus GEMM wall-time bench | y | n |  |
| GemmQ6kTileTest | test-compute | 8 | test_gemm_q6k_f16_tile.cu | GPU pending | 14.6 | Q6_K-to-F16 tile GEMM bit-equal to dequant plus F16 tile GEMM over variants | y | n |  |
| GemmStrategy | test-compute | 1 | test_gemm_kernel_registry.cu | GPU pending | 12.2 | GemmStrategy operator== compares by value (linear-scan registry lookup) | y | n |  |
| GemmTest | test-compute | 11 | test_gemm.cu | GPU pending | 7.3 | FP32/FP16 GEMM, GEMV and GEMV dispatch vs CPU reference, alpha/beta, zero matrix | y | n |  |
| GemvDp4aMultiTest | test-compute | 3 | test_gemv_dp4a_multi.cu | GPU pending | 13.4 | Multi-weight DP4A GEMV matches the single-weight GEMVs per weight | y | n |  |
| GemvFp16HotCold | test-compute | 1 | test_gemv_hot_cold_bench.cu | GPU pending | 8.8 | BENCH: FP16 GEMV hot vs cold cache timing (DISABLED, prints numbers) | y | n |  |
| GptOssYarnRef | test-compute | 2 | test_gpt_oss_yarn_ref.cu | GPU pending | 8.1 | gpt-oss YaRN RoPE: fp64 reference vs golden; kernel parity at long positions | y | n |  |
| HadamardTest | test-compute | 12 | test_hadamard.cu | GPU pending | 10.0 | Hadamard transform FP16 vs CPU reference: in-place path, involution property | y | n |  |
| InklingAttention | test-compute | 1 | test_inkling_kernels.cu | GPU pending | 9.9 | Inkling attention kernel: few-row form equals row form over 64 x 128 past keys | y | n |  |
| InklingAttnPaged | test-compute | 2 | test_inkling_attn_paged.cu | GPU pending | 10.0 | Inkling paged attention bit-equal to contiguous on shuffled blocks; hot/cold bench off | y | n |  |
| InklingCrossSlot | test-compute | 2 | test_inkling_cross_slot.cu | GPU pending | 13.7 | Inkling prefix restore across slots: restored prefix and draft match a fresh run | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingDecode | test-compute | 2 | test_inkling_kernels.cu | GPU pending | 9.9 | Inkling decode: small-tile expert GEMM vs slot GEMVs; I8B32 projection vs dequant GEMM | y | n |  |
| InklingDraftModel | test-compute | 4 | test_inkling_draft.cu | GPU pending | 11.7 | Inkling draft model: drafts follow reference continuation; local ring and slab rings | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingLookahead | test-compute | 2 | test_inkling_parity.cu | GPU pending | 21.3 | Inkling lookahead: rejected draft rows leave no trace; checkpoint inside a call | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingLora | test-compute | 3 | test_inkling_lora.cu | GPU pending | 11.8 | Inkling LoRA: zero delta leaves forward unchanged; delta equals merged weights | y | y: MODEL_INKLING |  |
| InklingPagedKv | test-compute | 1 | test_inkling_parity.cu | GPU pending | 21.3 | Inkling full-attention K/V in pool: shuffled pool blocks bit-equal to contiguous | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingParity | test-compute | 4 | test_inkling_parity.cu | GPU pending | 21.3 | Inkling vs reference: residual stream, logits, greedy continuation, grouped experts | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingPhraseRewind | test-compute | 2 | test_inkling_phrase_rewind.cu | GPU pending | 11.8 | Inkling phrase rewind: seek replays first-run rows; banned token resamples text | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingPrefill | test-compute | 4 | test_inkling_kernels.cu,test_inkling_parity.cu | GPU pending | 31.2 | Inkling grouped MXFP8 and NVFP4 expert GEMM vs slot GEMV; timing benches disabled | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingQuality | test-compute | 1 | test_inkling_quality.cu | GPU pending | 10.5 | Inkling quality: KLD against reference logits on served call forms | y | y: INKLING_KLD INKLING_KLD_CHUNKS MODEL_INKLING |  |
| InklingRewind | test-compute | 6 | test_inkling_parity.cu | GPU pending | 21.3 | Inkling rewind: resumed prefix equals fresh run; request binds to longest-prefix slot | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingServePath | test-compute | 1 | test_inkling_parity.cu | GPU pending | 21.3 | Inkling pool runtime bit-equal to contiguous forward on the serve path | y | y: INKLING_ORACLE MODEL_INKLING |  |
| InklingVisionParity | test-compute | 1 | test_inkling_parity.cu | GPU pending | 21.3 | Inkling vision tower merged features vs reference | y | y: INKLING_ORACLE MODEL_INKLING |  |
| LayerNormTest | test-compute | 9 | test_layernorm.cu | GPU pending | 9.4 | RMSNorm (FP32/FP16, residual, row-block, NVFP4 producer) vs CPU reference | y | n |  |
| LtGemmEx | test-compute | 5 | test_gemm_lt_ex.cu | GPU pending | 9.4 | cuBLASLt ex GEMM: strided batch, MoE grouped and FP8 per-expert scale, KV batched | y | n |  |
| MLAAbsorb | test-compute | 1 | test_mla.cu | GPU pending | 10.0 | MLA absorbed decode equals materialized-KV reference | y | n |  |
| MLAAttnOutput | test-compute | 6 | test_mla.cu | GPU pending | 10.0 | MLA attention output: v_head_dim != head_dim, prefill compaction, head gate, padded V | y | n |  |
| MLAProjection | test-compute | 2 | test_mla.cu | GPU pending | 10.0 | MLA two-step KV projection vs CPU reference; latent RMSNorm | y | n |  |
| MMVQ | test-compute | 6 | test_mmvq.cu | GPU pending | 8.6 | MMVQ vs DP4A GEMV for Q4_K, Q5_K, Q8_0 at small and large dims | y | n |  |
| MRope | test-compute | 5 | test_mrope.cu | GPU pending | 7.4 | Multi-axis RoPE: text positions equal single axis; interleaved and sectioned layouts | y | n |  |
| MmqQ4kImmaLayout | test-compute | 3 | test_mmq_q4k_imma_layout.cu | GPU pending | 8.7 | Q4_K IMMA weight reorder round trip: small, multi-superblock and FFN shapes | y | n |  |
| MoeRouterDecode | test-compute | 5 | test_moe_router_decode.cu | GPU pending | 11.2 | MoE router decode vs reference at Flint, bias, wide and odd-row shapes; refusal | y | n |  |
| MoeRouterFusedBench | test-compute | 1 | test_moe_router_fused.cu | GPU pending | 14.3 | NO-ASSERT: BENCH: MoE fused router tile/split sweep (DISABLED, prints timings) | y | n |  |
| MoeRouterFusedRefusal | test-compute | 1 | test_moe_router_fused.cu | GPU pending | 14.3 | MoE fused router refuses shapes outside its envelope (experts > 128) | y | n |  |
| MoeRouterFusedTest | test-compute | 8 | test_moe_router_fused.cu | GPU pending | 14.3 | MoE fused router ids and weights vs reference; rows independent of M and split | y | n |  |
| Nvfp4PromptStreamK | test-compute | 2 | test_nvfp4_prompt_streamk.cu | GPU pending | 10.7 | NVFP4 prompt stream-K GEMM: row invariance vs M; narrow tile matches wide tile | y | n |  |
| Nvfp4TileSwigluQuant | test-compute | 7 | test_nvfp4_tile_swiglu_quant.cu | GPU pending | 9.8 | NVFP4 tile SwiGLU plus quant fused: bytes equal the unfused path | y | n |  |
| PenaltyHistAppendTest | test-compute | 1 | test_penalty_hist_append.cu | GPU pending | 7.6 | Penalty history append: skips, bounds, bytes between sample slots | y | n |  |
| PromptLogprobsRows | test-compute | 4 | test_prompt_logprobs_rows.cu | GPU pending | 9.1 | Prompt logprob rows: row-major and slab-major outputs vs reference; entropy buffer | y | n |  |
| QkNormRopeFusedTest | test-compute | 3 | test_qknorm_rope_fused.cu | GPU pending | 11.6 | Fused QK-norm plus RoPE vs norm then standalone RoPE | y | n |  |
| QkNormRopeWarpTest | test-compute | 7 | test_qknorm_rope_fused.cu | GPU pending | 11.6 | Warp QK-norm plus RoPE bit-identical to the CTA kernel | y | n |  |
| RoPETest | test-compute | 10 | test_rope.cu | GPU pending | 9.4 | RoPE FP32/FP16, YaRN, long context, partial, MTP M-RoPE, query temperature vs refs | y | n |  |
| RowwiseTopM | test-compute | 4 | test_rowwise_topm.cu | GPU pending | 8.0 | Row-wise top-m: CPU reference, lowest-index tie break, M=1 equals argmax, arena exhaustion | y | n |  |
| SampleVerifyRows | test-compute | 9 | test_sampling_verify_rows.cu | GPU pending | 9.1 | Speculative verify sampling: row draws match eager sampler; min-p; coupled draft | y | n |  |
| SamplerCubPath | test-compute | 1 | test_sampling.cu | GPU pending | 12.5 | Sampler CUB path: each call samples from its own logits, not previous calls | y | n |  |
| SamplerSeeding | test-compute | 2 | test_sampling.cu | GPU pending | 12.5 | Sampler seeding: consecutive seeds differ; device salt draws like host step seed | y | n |  |
| SamplingAdvancedTest | test-compute | 7 | test_sampling_advanced.cu | GPU pending | 10.1 | TIMING: Logit bias, DRY penalty and Mirostat v2: token moves and seed stability | y | n |  |
| SamplingTest | test-compute | 30 | test_sampling.cu | GPU pending | 12.5 | Greedy, top-k, top-p (multiblock and CUB), NaN/inf logits, async rows, penalty rows | y | n |  |
| ShortConv | test-compute | 1 | test_shortconv.cu | GPU pending | 8.3 | Short conv chunked prefill then batched decode vs reference | y | n |  |
| SoftmaxTest | test-compute | 9 | test_activation.cu,test_softmax.cu | GPU pending | 17.8 | Softmax FP32/FP16 vs CPU reference: stability, large row, NVFP4 producer | y | n |  |
| SwigluQuantizeCutlass | test-compute | 1 | test_swiglu_quantize_cutlass.cu | GPU pending | 7.7 | SwiGLU plus quantize (CUTLASS) byte-identical to SwiGLU then quantize | y | n |  |
| TopkRadixSampler | test-compute | 3 | test_sampling_topk_radix.cu | GPU pending | 9.4 | Top-k radix sampler: bytes and tokens equal legacy path; unsupported shapes stay legacy | y | n |  |
| AnswerLostToReasoning | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | answer_lost_to_reasoning fires only when thinking ate the whole reply, bool cases | n | n |  |
| AnswerReserveKey | test-core | 4 | test_think_stop_logic.cpp | 0.00C | 4.9 | answer reserve tokens: default constant, floor, zero and negative clamp, EXPECT_EQ | n | n |  |
| AnthropicCacheControl | test-core | 11 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic cache_control markers map to cache-prompt flags across system/message/tool | n | n |  |
| AnthropicCacheUsage | test-core | 6 | test_anthropic_transform.cpp,test_anthropic_usage.cpp | 0.00C | 22.0 | Anthropic usage: spec counters and cache read/creation fields pass through, EXPECT_EQ | n | n |  |
| AnthropicContentBlocks | test-core | 9 | test_constraint_validation.cpp | 0.00C | 33.1 | Anthropic content-block validator accepts convertible blocks, refuses unknown/malformed | n | n |  |
| AnthropicEffort | test-core | 2 | test_anthropic_effort.cpp | 0.00C | 4.2 | Anthropic effort maps to reasoning_effort, nearest template level, EXPECT_EQ | n | n |  |
| AnthropicEnvelope | test-core | 5 | test_server_request_limits.cpp | 0.00C | 26.5 | Anthropic envelope: path detection, error wrapper per dialect, request-id header/body | n | n |  |
| AnthropicGuidedPassthrough | test-core | 3 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic shim passes guided-decoding and response_format fields through unchanged | n | n |  |
| AnthropicMessages | test-core | 12 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic messages to OpenAI shape: system, images, tool_use, tool_result, reasoning | n | n |  |
| AnthropicPrefill | test-core | 2 | test_assistant_prefill_transform.cpp | 0.00C | 4.5 | Anthropic assistant prefill sets continue_final_message only with trailing text | n | n |  |
| AnthropicReasoningReport | test-core | 2 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic usage reports reasoning tokens as output details, finish-detail copy rule | n | n |  |
| AnthropicResponse | test-core | 12 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic response shape: error envelope, text/thinking/tool_use blocks, bad args | n | n |  |
| AnthropicSampling | test-core | 1 | test_anthropic_effort.cpp | 0.00C | 4.2 | Anthropic sampling fields seed, top_k, temperature reach the chat body, EXPECT_EQ | n | n |  |
| AnthropicStopReason | test-core | 4 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic stop_reason: end_turn vs stop_sequence, engine reasons never leak verbatim | n | n |  |
| AnthropicSystemRole | test-core | 6 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic system role: leading system messages fold into the system prompt | n | n |  |
| AnthropicThinking | test-core | 15 | test_anthropic_thinking_blocks.cpp,test_anthropic_transform.cpp | 0.00C | 21.9 | Anthropic thinking config maps to enable_thinking and budget, clamp/signature rules | n | n |  |
| AnthropicToolChoice | test-core | 6 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic tool_choice maps to OpenAI tool_choice and parallel_tool_calls flags | n | n |  |
| AnthropicToolResult | test-core | 2 | test_anthropic_transform.cpp | 0.00C | 19.4 | Anthropic tool_result is_error labels the tool output for the model | n | n |  |
| ApiKeyAuth | test-core | 5 | test_server_auth.cpp | 0.00C | 19.3 | API key auth: Bearer and x-api-key accepted, mismatch and empty key refused | n | n |  |
| ArchIdentity | test-core | 4 | test_arch_identity.cpp | 0.00C | 2.9 | Arch identity: every arch string keeps its enum, one table, HF/GGUF loaders agree | n | n |  |
| Arena | test-core | 8 | test_memory_allocators.cpp | 0.00C | 4.8 | Arena allocator: aligned takes, usage, growth, exhaustion, reset, close, typed errors | n | n |  |
| ArenaLazy | test-core | 3 | test_memory_allocators.cpp | 0.00C | 4.8 | Lazy arena commits nothing until a take reaches it, fixed-region fallback, exhaustion | n | n |  |
| AttentionDispatchRules | test-core | 6 | test_attention_dispatch_rules.cpp | 0.00C | 1.9 | Attention dispatch: FMHA/FA2 head-dim coverage, MLA refusal, MXFP4 decode refusal | n | n |  |
| AttnDispatchTable | test-core | 9 | test_routing_decision.cpp | 0.00C | 3.4 | Attention dispatch table: default FA2, decline fallbacks, sinks routing, all-decline throw | n | n |  |
| AttnRoutingModelCoupling | test-core | 3 | test_routing_decision.cpp | 0.00C | 3.4 | Attention path replay: every tier replays to itself, sinks/no-accept map to none | n | n |  |
| AttnTierPrecedence | test-core | 2 | test_routing_decision.cpp | 0.00C | 3.4 | Attention tier precedence: FA2 beats FMHA sm120, FMHA beats Blackwell when both accept | n | n |  |
| AudioConfigTest | test-core | 3 | test_hf_config_loader.cpp | 0.00C | 6.6 | Audio config: modality flag set by object only, not null or absent, HF config load | n | n |  |
| AudioWav | test-core | 3 | test_audio_features.cpp | 0.00C | 3.7 | WAV reader: PCM16 mono, stereo averaging, extensible float, malformed files refused | n | n |  |
| AwqArch | test-core | 4 | test_awq_plan_sites.cpp | 0.00C | 6.5 | AWQ arch table: offset convention per arch, layer prefix resolution, plain-norm archs | n | n |  |
| AwqCalibration | test-core | 8 | test_awq_calibration.cpp | 0.00C | 4.5 | AWQ calibration file round-trip, foreign/truncated refusal, scale and fold algebra | n | n |  |
| AwqNormFold | test-core | 6 | test_awq_calibration.cpp | 0.00C | 4.5 | AWQ norm fold: unit-offset gain division, BF16 near-zero loss, clamp bounds, vector fold | n | n |  |
| AwqSites | test-core | 19 | test_awq_plan_sites.cpp | 0.00C | 6.5 | AWQ site selection: attention/GDN fold groups, group selector, tie map, default groups | n | n |  |
| BailingHybridConfig | test-core | 1 | test_bailing_hybrid.cpp | 0.00C | 3.4 | Bailing hybrid config: Ling3Tiny maps KDA, MLA and router fields, EXPECT_EQ | n | n |  |
| BailingHybridTensors | test-core | 3 | test_bailing_hybrid.cpp | 0.00C | 3.4 | Bailing hybrid tensors: KDA layer fused and MLA renamed, incomplete layer refused | n | n |  |
| BannedStringsParams | test-core | 1 | test_sampling_params_golden.cpp | 0.00C | 33.6 | Banned-strings param parse: list rules, null/absent handling, status codes | n | n |  |
| Base64 | test-core | 5 | test_base64.cpp | 0.00C | 20.0 | Base64: RFC 4648 vectors, padding at each residue, decode inverse, all byte values | n | n |  |
| BatchVerifyPredicates | test-core | 18 | test_batch_verify_predicates.cpp | 0.00C | 5.4 | Batched verify predicates: head slots, economics guard, carry order, history bindings | n | n |  |
| BearerAuth | test-core | 8 | test_server_auth.cpp | 0.00C | 19.3 | Bearer auth: token match accepted, wrong, missing prefix, empty, prefix and long tokens | n | n |  |
| BlockPool | test-core | 7 | test_memory_allocators.cpp | 0.00C | 4.8 | BlockPool: acquire/release balance, distinct stable blocks, exhaustion ref, churn | n | n |  |
| BlockPoolSlots | test-core | 2 | test_memory_allocators.cpp | 0.00C | 4.8 | BlockPool slots: lowest free id first, shrink drops only the free tail | n | n |  |
| BlockRef | test-core | 1 | test_memory_allocators.cpp | 0.00C | 4.8 | BlockRef move transfers the reference without changing the referent count | n | n |  |
| BudgetRecount | test-core | 4 | test_think_stop_logic.cpp | 0.00C | 4.9 | Budget recount: thinking opener counted between markers, prompt-seeded and no-think | n | n |  |
| BufferStamp | test-core | 3 | test_buffer_stamp.cpp | 0.00C | 1.7 | BufferStamp: live until a view retires it, matches buffer/weight/rows, clear drops it | n | n |  |
| CApiContract | test-core | 4 | test_c_api_contract.cpp | 0.00C | 2.0 | C API contract: null args invalid, lora unload, kv dtype validation, six dtypes pass | n | n |  |
| CApiEnumBinding | test-core | 6 | test_c_api_enum_binding.cpp | 0.00C | 1.8 | C API enum binding: arch ids match public header, unique, round-trip, dtype stable | n | n |  |
| CalibrationOutPath | test-core | 4 | test_server_args.cpp | 0.00C | 3.2 | Calibration output path: flag wins over config key, empty key does not clobber | n | n |  |
| CapabilityRefusal | test-core | 6 | test_capability_refusal.cpp | 0.00C | 22.9 | Capability refusal: Inkling refuses loop features, OpenAI/Anthropic envelopes carry param | n | n |  |
| CfgParams | test-core | 1 | test_sampling_params_golden.cpp | 0.00C | 33.6 | cfg_scale and cfg_negative_prompt parse rules, status codes, defaults | n | n |  |
| CgroupHeadroomTest | test-core | 1 | test_weight_snapshot.cpp | 0.00C | 4.0 | cgroup headroom: limit minus non-reclaimable memory, parse cases, EXPECT_EQ | n | n |  |
| ChannelSplit | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Channel split: thought, content, unknown and analysis routing of reasoning output | n | n |  |
| CheckpointLimits | test-core | 20 | test_checkpoint_limits.cpp | 0.25C | 6.1 | Checkpoint limits: huge/overflow layer and expert indices, dims, shard names, traversal | n | n |  |
| CliConfigPins | test-core | 1 | test_mtp_auto.cpp | 0.00C | 4.3 | CLI config pins: --perplexity commits KV pool up front, other paths stay growable | n | n |  |
| ClientErrorLogLine | test-core | 2 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Client error log line: status, route, reason on one line, control bytes stripped | n | n |  |
| CombineImageHash | test-core | 1 | test_image_placeholders.cpp | 0.00C | 3.1 | combine_image_hash: order-sensitive, never zero, single image hashes to itself | n | n |  |
| ConfigBadValue | test-core | 5 | test_config.cpp | 0.00C | 4.7 | Config bad value: bool, int, float and tri-state keys reject unreadable values | n | n |  |
| ConfigBinding | test-core | 1 | test_config.cpp | 0.00C | 4.7 | Config binding: prefill_chunk_size default -1, override reaches the field | n | n |  |
| ConfigBindings | test-core | 1 | test_config_bindings.cpp | 0.00C | 3.7 | Config bindings: every example key moves only its own field, count floor | n | n |  |
| ConfigExplicitKeys | test-core | 1 | test_mtp_auto.cpp | 0.00C | 4.3 | Config explicit keys: was_set true only for keys the operator set | n | n |  |
| ConstraintValidation | test-core | 17 | test_constraint_validation.cpp | 0.00C | 33.1 | Constraint validation: regex/grammar accepted or refused with parser error, all spellings | n | n |  |
| ContentParts | test-core | 8 | test_constraint_validation.cpp | 0.00C | 33.1 | Content parts: text, image, video, input_audio accepted, unknown and malformed refused | n | n |  |
| ControlVector | test-core | 4 | test_control_vector.cpp | 0.00C | 3.7 | Control-vector GGUF load: direction per layer, layer/width/type/truncated refusals | n | n |  |
| CorsOrigins | test-core | 3 | test_server_auth.cpp | 0.00C | 19.3 | CORS allow-origin: no origin by default, exact echo, wildcard allows any origin | n | n |  |
| CudaErrorsTest | test-core | 6 | test_cuda_errors.cpp | 0.00C | 2.4 | CUDA error classes: sticky faults unrecoverable, sync throws naming site, EXPECT_THROW | n | n |  |
| CudaStaticReset | test-core | 1 | test_cuda_static_reset.cpp | 0.00C | 1.7 | CUDA static-reset hook registry is populated, EXPECT_GT | n | n |  |
| DecodeGraphCtx | test-core | 6 | test_decode_graph_ctx.cpp | 0.00C | 2.0 | Decode graph capture: pow2 buckets, deterministic capture contexts, batch key order | n | n |  |
| DequantAwq | test-core | 3 | test_dequant_awq.cpp | 0.00C | 2.2 | AWQ dequant bit-exact vs reference per group/column, GEMM layout shape checks | n | n |  |
| DequantCap | test-core | 3 | test_spec_request.cpp | 0.00C | 2.5 | Dequant graph-capture cap: lm_head exempt, eligible plane over cap disables capture | n | n |  |
| DequantGptq | test-core | 7 | test_dequant_gptq.cpp | 0.00C | 3.0 | GPTQ dequant bit-exact (asym zeros v1/v2), g_idx group select, zero-format parse | n | n |  |
| DeviceInfoTest | test-core | 4 | test_device_info.cpp | 0.00C | 1.8 | Device info: SM count fallback 170, sm120 and PDL cc gates, cached values stable | n | n |  |
| DeviceOwnedTest | test-core | 17 | test_device_owned.cpp | 0.00C | 2.9 | DeviceOwned handles: empty/zero-byte, adopt allocator releases once, async stream release | n | n |  |
| DispatchPaths | test-core | 2 | test_dispatch_record.cpp | 0.00C | 2.7 | Dispatch path enum: every value named, names distinct within each family | n | n |  |
| DispatchPolicyFwd | test-core | 1 | test_config.cpp | 0.00C | 4.7 | Dispatch policy: null reads the default, wired policy returned as itself | n | n |  |
| DispatchRecord | test-core | 2 | test_dispatch_record.cpp | 0.00C | 2.7 | Dispatch record starts unset, tier flags track their setters, dump gate | n | n |  |
| DraftRowDraws | test-core | 2 | test_step_seed.cpp | 0.00C | 4.9 | Draft row draws: level rows take verify seed, greedy top_k 1, empty without sampled row | n | n |  |
| DropNullFields | test-core | 1 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Null JSON fields dropped at top level only, nested nulls kept | n | n |  |
| EffectiveCapabilities | test-core | 7 | test_tensor_kind_table.cpp | 0.00C | 3.0 | Effective capabilities: NVFP4 stripped for sub-5-bit or F16 sources, floor supported | n | n |  |
| EndThinking | test-core | 7 | test_think_stop_logic.cpp | 0.00C | 4.9 | End-thinking forcing: ChatML newline then closer, Harmony final opener, no-op outside | n | n |  |
| EndThinkingRegistry | test-core | 5 | test_end_thinking.cpp | 0.00C | 3.6 | End-thinking registry: unknown id NotFound, pending flagged once, phase sets status | n | n |  |
| EngineCapabilities | test-core | 8 | test_engine_capabilities.cpp | 0.00C | 2.1 | Engine capability gates: dense serves all, Inkling/gated forwards refuse unsupported | n | n |  |
| EngineConfigBridge | test-core | 4 | test_engine_config_bridge.cpp | 0.00C | 2.5 | Engine config bridge: explicit default beats imp.conf, unset falls through, EXPECT_EQ | n | n |  |
| EngineMetricsCell | test-core | 4 | test_engine_metrics_snapshot.cpp | 0.03C | 2.9 | TIMING: Metrics cell: zero before publish, copies, no torn concurrent reads, bounded | n | n |  |
| EngineStreamOpen | test-core | 2 | test_engine_stream_open.cpp | 0.00C | 2.9 | Engine stream open: result matches the stream, no device fails the open | n | n |  |
| ErrorEnvelope | test-core | 1 | test_server_request_limits.cpp | 0.00C | 26.5 | Error envelope: param and code fields present only when supplied | n | n |  |
| ExecApaWorkspaceBytes | test-core | 1 | test_workspace_sizes.cpp | 0.00C | 4.1 | APA workspace size follows the carve layout, exact byte value | n | n |  |
| ExecGdnDtBytes | test-core | 2 | test_workspace_sizes.cpp | 0.00C | 4.1 | GDN dt buffer bound holds for every row count up to max tokens, ASSERT_LE | n | n |  |
| ExecKdaKimiScratch | test-core | 1 | test_workspace_sizes.cpp | 0.00C | 4.1 | KDA scratch: split-QKV and LoRA buffers fit, offsets exact, EXPECT_EQ | n | n |  |
| ExecMaxTokens | test-core | 3 | test_workspace_sizes.cpp | 0.00C | 4.1 | exec max_tokens clamps to 4096, config fallback, replicates as-built GDN cap | n | n |  |
| ExecMaxWeightK | test-core | 1 | test_workspace_sizes.cpp | 0.00C | 4.1 | Max logical weight K over tensors equals 12288, EXPECT_EQ | n | n |  |
| ExecRa2WorkspaceBytes | test-core | 1 | test_workspace_sizes.cpp | 0.00C | 4.1 | RA2 workspace size follows the carve layout, exact byte value | n | n |  |
| ExecSsmConvF32 | test-core | 1 | test_workspace_sizes.cpp | 0.00C | 4.1 | SSM conv f32 region ends inside proj buffer for GDN and KDA, EXPECT_LE | n | n |  |
| ExecSsmZCols | test-core | 4 | test_workspace_sizes.cpp | 0.00C | 4.1 | SSM z-columns cover the attention gate or the recurrent inner, whichever is larger | n | n |  |
| ExecT2Demand | test-core | 25 | test_workspace_sizes.cpp | 0.00C | 4.1 | T2 workspace demand: GEMM scratch formula, cap handling, largest fitting target | n | n |  |
| ExitCodes | test-core | 4 | test_exit_codes.cpp | 0.00C | 1.9 | Exit codes: success 0, taxonomy maps one-to-one, distinct bytes, unknown error to 1 | n | n |  |
| ExpertBatchPolicy | test-core | 9 | test_expert_batch_policy.cpp | 0.00C | 4.0 | Expert batch policy: auto cap at device rows, host-expert tier resolves to one | n | n |  |
| ExpertCacheReinitCpu | test-core | 2 | test_expert_cache_reinit.cpp | 0.00C | 2.5 | Expert cache reinit: refused budgets leave cache disabled, empty layer mask refused | n | n |  |
| ExpertDestack | test-core | 9 | test_quantize_expert_destack.cpp | 0.00C | 2.9 | Expert destack: layout by model type, GPT-OSS/Gemma4 plans, gate-up and down reads | n | n |  |
| ExpertPack | test-core | 5 | test_expert_pack.cpp | 0.00C | 2.3 | Expert pack: uniform F16 shape, quantized/mixed refused, failed alloc leaves experts | n | n |  |
| ExpertPlacement | test-core | 6 | test_expert_placement.cpp | 0.00C | 1.6 | Expert placement: NVFP4 host path needed only with host-resident expert layers | n | n |  |
| ExtractReasoning | test-core | 6 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Reasoning extraction: think block split, unclosed think, last close tag wins | n | n |  |
| FallbackSite | test-core | 2 | test_log_once.cpp | 0.00C | 2.5 | Fallback counters: every hit counted, logged once per load, same-name sites sum | n | n |  |
| FfnGraphCache | test-core | 2 | test_ffn_graph_cache.cpp | 0.00C | 2.0 | FFN graph cache: row bucket steps and bounds, empty cache runs eager | n | n |  |
| Fim | test-core | 5 | test_fim.cpp | 0.00C | 2.3 | FIM token discovery: synthetic markers, PSM order, GGUF metadata ids win | n | n |  |
| FinishReason | test-core | 2 | test_server_request_limits.cpp | 0.00C | 26.5 | finish_reason: engine-only values map to the OpenAI enum, enum members pass through | n | n |  |
| FlintConfigTest | test-core | 3 | test_flint.cpp | 0.00C | 3.7 | Flint config load: layer map, router and shared expert, unsupported builds refused | n | n |  |
| FlintNames | test-core | 1 | test_flint.cpp | 0.00C | 3.7 | Flint tensor names: sink and expert-bias canonical names, EXPECT_EQ | n | n |  |
| ForceThinkEnd | test-core | 9 | test_think_stop_logic.cpp | 0.00C | 4.9 | Forced think-end: fires at budget while thinking, reserve cap grows answer room | n | n |  |
| ForcedToolEnvelope | test-core | 3 | test_tool_call.cpp | 0.00C | 16.5 | Forced tool-call envelope literals per family, Harmony and Gemma4 parse as a call | n | n |  |
| ForwardProfile | test-core | 2 | test_forward_profile.cpp | 0.00C | 1.6 | Forward profile never runs on a capturing stream, eager only when flagged | n | n |  |
| Fp8Keep | test-core | 9 | test_quantize_fp8_keep.cpp | 0.01C | 3.6 | FP8 keep-list: fused group selection, rank parse, byte budget by score, ModelOpt output | n | n |  |
| Fp8LmHead | test-core | 5 | test_fp8_lm_head.cpp | 0.09C | 3.4 | FP8 lm_head: decode matches bit-field reference for all codes, RNE encode, saturation | n | n |  |
| Fp8Source | test-core | 13 | test_quantize_fp8_source.cpp | 0.00C | 3.4 | FP8 E4M3 source: grid decode, block-edge derivation, per-block scale application | n | n |  |
| Fp8SourceAction | test-core | 4 | test_quantize_policy.cpp | 0.00C | 4.5 | FP8 source action: refused weights widened, gated flag from caller, lm_head flag | n | n |  |
| FpBits | test-core | 9 | test_fp_bits.cpp | 0.00C | 2.7 | Half/bf16 bit conversions: known values, round-trip, rescale, subnormals, RNE | n | n |  |
| FpBitsLegacyParity | test-core | 6 | test_fp_bits_legacy_parity.cpp | 2.46C | 2.4 | Exhaustive FP bit-parity vs legacy: all 65536 fp16/bf16 codes, E4M3, MXFP8, ngram | n | n |  |
| FusedGateQProj | test-core | 4 | test_quantize_policy.cpp | 0.00C | 4.5 | Fused gated q_proj finder: gated found, dense left alone, per layer, nested prefix | n | n |  |
| FuzzCorpus | test-core | 9 | test_fuzz_corpus.cpp | 0.81C | 2.8 | NO-ASSERT: Fuzz corpus runs each parser target over seeded corpora via run_target helper | n | n |  |
| GbnfDepthCap | test-core | 2 | test_gbnf_grammar.cpp | 0.00C | 4.9 | GBNF nesting depth cap: deep groups refused, ordinary nesting parses | n | n |  |
| GbnfGrammar | test-core | 23 | test_gbnf_grammar.cpp | 0.00C | 4.9 | GBNF grammar parser refusals: no root, undefined rule, left recursion, bad bounds | n | n |  |
| GbnfGrammarTest | test-core | 1 | test_gbnf_grammar.cpp | 0.00C | 4.9 | GBNF matcher stays bounded under deep self-recursion, would_accept battery | n | n |  |
| GdnChunkparArgs | test-core | 1 | test_launcher_args.cpp | 0.00C | 3.3 | GDN chunk-parallel args struct: fields carried in declaration order, EXPECT_EQ | n | n |  |
| GdnFusedSplit | test-core | 3 | test_gdn_fused_split.cpp | 0.00C | 1.8 | GDN fused qkvz/ba split regroups per key-head group, refuses bad shape or quantized | n | n |  |
| GdnProjection | test-core | 2 | test_quantize_policy.cpp | 0.00C | 4.5 | GDN projection name matcher: Qwen3.6 recipe names only, role flag vocabulary | n | n |  |
| GdnScanArgs | test-core | 2 | test_launcher_args.cpp | 0.00C | 3.3 | GDN scan args struct: fields in declaration order, omitted fields take defaults | n | n |  |
| GdnScanBatchedArgs | test-core | 2 | test_launcher_args.cpp | 0.00C | 3.3 | Batched GDN scan args struct: field order and strides, omitted fields default | n | n |  |
| GemmAlgoCapture | test-core | 3 | test_gemm_algo_capture.cpp | 0.00C | 1.5 | cuBLASLt algo probe runs only when query ok and not capturing, first fitting algo | n | n |  |
| GemmFallbackFailsLoud | test-core | 6 | test_gemm_fallback_fails_loud.cpp | 0.00C | 2.6 | GEMM fallback paths throw loudly on missing scratch, residual, cache or source | n | n |  |
| GemmKernelRegistryLink | test-core | 1 | test_gemm_kernel_registry_link.cpp | 0.00C | 1.5 | GEMM kernel registry holds exactly the produced tier/shape keys, EXPECT_EQ | n | n |  |
| GemmMode | test-core | 4 | test_gemm_mode.cpp | 0.00C | 2.0 | GEMM mode flags default off, decode-row and verify-small-M gating, EXPECT_FALSE | n | n |  |
| Gemma4AudioFeatures | test-core | 5 | test_audio_features.cpp | 0.04C | 3.7 | Gemma4 audio mel features match HF extractor, clip length refusals, HTK filters | n | n |  |
| Gemma4AudioLoad | test-core | 5 | test_gemma4_audio_load.cpp | 0.00C | 4.1 | Gemma4 audio config and tensor load: slots, tables, missing or wrong-shape refused | n | n |  |
| Gemma4AudioPlaceholders | test-core | 2 | test_audio_features.cpp | 0.00C | 3.7 | Gemma4 audio placeholder expansion per clip, count mismatch refused and untouched | n | n |  |
| Gemma4EConfig | test-core | 4 | test_gemma4_e_variant.cpp | 0.00C | 4.6 | Gemma4 E4B config: per-layer inputs, KV sources, PLI pad, double-wide MLP refused | n | n |  |
| Gemma4EWeights | test-core | 1 | test_gemma4_e_variant.cpp | 0.00C | 4.6 | Gemma4 E weights: per-layer input table and shared-KV mapping non-null, EXPECT_NE | n | n |  |
| Gemma4KvSources | test-core | 1 | test_gemma4_e_variant.cpp | 0.00C | 4.6 | Gemma4 KV source layers edge cases: empty and nothing-to-share, EXPECT | n | n |  |
| Gemma4RopeConfigTest | test-core | 2 | test_hf_config_loader.cpp | 0.00C | 6.6 | Gemma4 RoPE config: proportional full-attention table, absent without partial factor | n | n |  |
| GenerationConfigTest | test-core | 4 | test_hf_config_loader.cpp | 0.00C | 6.6 | generation_config.json: partial fields leave sentinels, do_sample false forces greedy | n | n |  |
| GgufFaultInjection | test-core | 27 | test_gguf_fault_injection.cpp | 0.00C | 3.5 | TIMING: GGUF fault injection: truncated, bad magic/version, huge counts refused, valid | n | n |  |
| GgufLlama3Bos | test-core | 1 | test_gguf_llama3.cpp | 0.00C | 2.9 | Llama-3 BPE GGUF defaults add_bos true, EXPECT on tokenizer flag | n | n |  |
| GgufLoaderTest | test-core | 9 | test_gguf_loader.cpp | 0.00C | 3.3 | GGUF loader: missing file, bad magic, truncation, version, zero tensors, arch refusal | n | n |  |
| GgufRopeFreqs | test-core | 3 | test_gguf_llama3.cpp | 0.00C | 2.9 | GGUF rope_freqs tensor: absent leaves tables empty, Llama3 closed form, bad length ignored | n | n |  |
| GgufSilentDrops | test-core | 8 | test_gguf_silent_drops.cpp | 0.00C | 4.6 | GGUF tensor-name mapping: bias and NextN blocks not silently dropped or clobbered | n | n |  |
| GptqHessianFile | test-core | 1 | test_awq_calibration.cpp | 0.00C | 4.5 | GPTQ Hessian file round-trip of entries, truncation rejected, ASSERT_EQ | n | n |  |
| GracePeriod | test-core | 6 | test_think_stop_logic.cpp | 0.00C | 4.9 | Grace-period stop: blocks exit until content seen, hard cap releases, whitespace not | n | n |  |
| GraphEligibility | test-core | 16 | test_graph_eligibility.cpp | 0.00C | 3.2 | Graph eligibility reasons: names distinct, KV pressure only mid-run, demote records first | n | n |  |
| GraphEpoch | test-core | 7 | test_graph_epoch.cpp | 0.00C | 2.5 | Graph epoch bump monotonic, stale graph refused and recaptured once, regrow bump rules | n | n |  |
| GraphSlotPool | test-core | 12 | test_graph_slots.cpp | 0.00C | 2.9 | Graph slot pool: open shape, bad capacities, double open, non-overlapping slot buffers | n | n |  |
| GrowRows | test-core | 3 | test_grow_rows.cpp | 0.00C | 2.2 | Growable rows commit in steps, refused commit keeps row count, fixed without backend | n | n |  |
| HFConfigGolden | test-core | 1 | test_hf_config_golden.cpp | 0.03C | 3.5 | HF config golden: all 98 fixtures match stored goldens, mismatch count zero | n | n |  |
| HarmonySplit | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Harmony channel split: analysis to reasoning, final to content, commentary as reasoning | n | n |  |
| HealthUnservable | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Health verdict: floored pool unservable and named, healthy and unknown cases | n | n |  |
| HfFetchInfo | test-core | 1 | test_hf_fetch.cpp | 0.00C | 23.2 | HF repo info parse: siblings, commit sha, gated flag, EXPECT_EQ | n | n |  |
| HfFetchMock | test-core | 10 | test_hf_fetch.cpp | 0.04C | 23.2 | HF fetch on mock server: verify sha, resume with Range, retry, sha mismatch fails | n | n |  |
| HfFetchParse | test-core | 2 | test_hf_fetch.cpp | 0.00C | 23.2 | hf:// URI parse: repo and file accepted, malformed rejected | n | n |  |
| HfFetchSelect | test-core | 3 | test_hf_fetch.cpp | 0.00C | 23.2 | HF file selection: single GGUF auto-picked, several need selector, safetensors set | n | n |  |
| HfFetchSha | test-core | 1 | test_hf_fetch.cpp | 0.00C | 23.2 | SHA-256 of files vs known vectors, EXPECT_EQ | n | n |  |
| HfFetchUrl | test-core | 1 | test_hf_fetch.cpp | 0.00C | 23.2 | HF API and resolve URL builders for repo and revision refs, EXPECT_EQ | n | n |  |
| Holdback | test-core | 10 | test_stream_pipeline.cpp | 0.00C | 4.0 | Stop-sequence holdback: flush without overrun, earliest match wins, empty stop skipped | n | n |  |
| HoldbackDecision | test-core | 2 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Holdback decision: flush cut on UTF-8 boundary, stop match cuts exactly at the match | n | n |  |
| HostEmbedding | test-core | 3 | test_host_embedding.cpp | 0.00C | 2.1 | Host embedding lookup bits match upload staging for bf16, f32, f16 | n | n |  |
| HostExpertTier | test-core | 6 | test_host_expert_tier.cpp | 0.01C | 3.7 | Host expert tier: LRU eviction, transient scratch, exchange, aligned and unaligned loads | n | n |  |
| HostFileMap | test-core | 1 | test_host_expert_tier.cpp | 0.00C | 3.7 | Host file map resolves pointers to fd-buffered offsets from maps text, EXPECT_LT | n | n |  |
| HostPinnedSites | test-core | 1 | test_host_pinned.cpp | 0.00C | 3.5 | Host pinned-memory sites: acquire, move, reset update byte and count totals | n | n |  |
| HostRamHeadroomTest | test-core | 1 | test_weight_snapshot.cpp | 0.00C | 4.0 | Host RAM headroom: 6 GiB floor or 10 percent, unknown total, EXPECT_EQ | n | n |  |
| HostRegistration | test-core | 8 | test_host_pinned.cpp | 0.00C | 3.5 | Host registration RAII: register once, unregister once, move and reset semantics | n | n |  |
| HostSpillAuto | test-core | 2 | test_snapshot_host_default.cpp | 0.00C | 2.2 | Host spill auto size: two-thirds of spare RAM in whole GiB, only unset key resolved | n | n |  |
| HostTaskPool | test-core | 3 | test_host_task_pool.cpp | 0.01C | 1.9 | Host task pool: wait sees every task, destructor joins, zero threads still runs | n | n |  |
| ImageDecode | test-core | 9 | test_image_decode.cpp | 0.00C | 3.8 | JPEG decode bit-matches Pillow, file and memory agree, GIF/BMP refused, corrupt fails | n | n |  |
| ImagePlaceholders | test-core | 8 | test_image_placeholders.cpp | 0.00C | 3.1 | Image placeholder expansion: per-image count, mismatch refused, sequence unchanged | n | n |  |
| ImageTokensBefore | test-core | 3 | test_image_placeholders.cpp | 0.00C | 3.1 | Image token offset before a chunk: counts prefix only, clamps, text-only starts at zero | n | n |  |
| InflightGate | test-core | 2 | test_server_request_limits.cpp | 0.01C | 26.5 | TIMING: In-flight gate: refuses limit+1, readmits after leave, concurrent entries capped | n | n |  |
| InklingDense8 | test-core | 1 | test_inkling_kv_layout.cpp | 0.00C | 3.2 | Inkling dense8 fit check: VRAM budget arithmetic, KV pool not charged, EXPECT | n | n |  |
| InklingGuardsFailLoud | test-core | 5 | test_inkling_guards_fail_loud.cpp | 0.00C | 1.6 | Inkling kernel guards throw loudly: head dim, missing state, router expert range | n | n |  |
| InklingImageParts | test-core | 4 | test_vision_family.cpp | 0.00C | 2.9 | Inkling image parts: picture inside text split out, header role kept, expansion count | n | n |  |
| InklingKvLayout | test-core | 2 | test_inkling_kv_layout.cpp | 0.00C | 3.2 | Inkling KV layout: global apps own one layer each, prefix-cache snapshot key | n | n |  |
| InklingLoraTargets | test-core | 3 | test_inkling_registry.cpp | 0.00C | 3.3 | Inkling LoRA targets: module names map to qkvo, other tensors refused, dims | n | n |  |
| InklingPrefixCap | test-core | 2 | test_inkling_kv_layout.cpp | 0.00C | 3.2 | Inkling prefix cap: never past slot match, pool holds max-seq plus lookahead rows | n | n |  |
| InklingRegistry | test-core | 1 | test_inkling_registry.cpp | 0.00C | 3.3 | Inkling weight registry: five borrowed fp16 handles per layer, 24-layer count | n | n |  |
| InklingSlabLayout | test-core | 6 | test_inkling_slab_layout.cpp | 0.00C | 3.9 | Inkling slab layout: slot conv state plus sliding rings, aligned disjoint regions | n | n |  |
| InklingSlotTable | test-core | 4 | test_inkling_slot_table.cpp | 0.00C | 3.0 | Inkling slot table: rows find slot by request id, unknown id throws, release unbinds | n | n |  |
| InklingSnapLayout | test-core | 3 | test_inkling_slab_layout.cpp | 0.00C | 3.9 | Inkling snapshot layout: entry bytes and sliding windows, restore onto another slot | n | n |  |
| InternVLLmNames | test-core | 3 | test_internvl_lm_names.cpp | 0.00C | 4.1 | InternVL LM tensor names: every LM tensor assigned, vision counted apart, Qwen3-VL | n | n |  |
| InternVLPreprocess | test-core | 2 | test_internvl_preprocess.cpp | 0.01C | 2.9 | InternVL tile preprocess matches HF within one step, prompt ids match HF processor | n | n |  |
| InternVLVision | test-core | 4 | test_internvl_vision.cpp | 0.00C | 3.8 | InternVL vision config parse, refusals, loader reads every tensor, HF per-stage match | n | n |  |
| JsonConstrainFsm | test-core | 14 | test_json_constrain_property.cpp | 0.00C | 6.9 | JSON-constrain FSM rejects a second decimal point in numbers, sim_token_valid battery | n | n |  |
| JsonConstrainPropertyTest | test-core | 9 | test_json_constrain_property.cpp | 0.01C | 6.9 | JSON constrainer property battery: valid docs and prefixes accepted, bad syntax rejected | n | n |  |
| JsonEscape | test-core | 1 | test_sse_stream_utils.cpp | 0.00C | 31.4 | JSON escaping: control chars, quotes, UTF-8 round-trip, EXPECT_EQ | n | n |  |
| JsonNestingDepth | test-core | 5 | test_sse_stream_utils.cpp | 0.00C | 31.4 | JSON nesting depth counter: plain nesting, braces inside strings ignored | n | n |  |
| JsonNumberBounds | test-core | 6 | test_json_number_bounds.cpp | 0.01C | 3.0 | JSON number bounds: exact compare, exclusivity, tighten, prefix reachability, brute force | n | n |  |
| JsonOutTest | test-core | 9 | test_json_out.cpp | 0.00C | 2.7 | JSON writer: empty object, key order, precision, escapes, UTF-8 pass-through | n | n |  |
| KVAccounting | test-core | 5 | test_kv_accounting.cpp | 0.00C | 2.7 | KV accounting: for-accounting geometry, salt seeds lookup, probe count, reuse stops at | n | n |  |
| KVCacheManagerTest | test-core | 47 | test_kv_cache.cpp | 0.00C | 7.0 | KV cache manager: allocate/rollback/append/free, LRU eviction, live seq never evicted | n | n |  |
| KVCacheTest | test-core | 8 | test_kv_cache.cpp | 0.00C | 7.0 | KV cache: construction, block alloc/free, refcount, NVFP4 head-dim reject, bytes per block | n | n |  |
| KVHostSpillTest | test-core | 8 | test_kv_host_spill.cpp | 0.00C | 3.0 | KV host spill tier: budget off, committed round-trip, LRU replace, chunk pin, bindings | n | n |  |
| KVSessionPins | test-core | 9 | test_kv_session_pins.cpp | 0.00C | 2.5 | KV session pins: prefix hit under pressure, LRU unpinned first, TTL and budget unpin | n | n |  |
| KdaReference | test-core | 4 | test_kda_reference.cpp | 0.00C | 3.1 | KDA reference recurrence matches FLA naive, gate bound and softplus closed form | n | n |  |
| KimiLinearConfig | test-core | 2 | test_kimi_linear.cpp | 0.00C | 4.5 | Kimi-Linear config: KDA, MLA nope and router mapping, unsupported refused | n | n |  |
| KimiLinearTensors | test-core | 4 | test_kimi_linear.cpp | 0.00C | 4.5 | Kimi-Linear tensors: dense QKV fuse, LoRA rename, NVFP4 scales kept, broken layout refused | n | n |  |
| KvAdmission | test-core | 5 | test_kv_admission.cpp | 0.00C | 4.2 | KV admission estimate: P90 of finished outputs with floor, expected length admits more | n | n |  |
| KvBlockSize | test-core | 2 | test_kv_residual_sizing.cpp | 0.00C | 2.5 | KV block size: multiples of 16 up to 256 served, all others refused with a reason | n | n |  |
| KvDtypeGolden | test-core | 5 | test_kv_dtype.cpp | 0.00C | 3.7 | KV dtype golden: block bytes and planner byte math per layer, FP4 head-dim multiple | n | n |  |
| KvDtypeInfo | test-core | 2 | test_kv_dtype.cpp | 0.00C | 3.7 | KV dtype info flags and block bytes match golden table, EXPECT_EQ | n | n |  |
| KvFp8Scale | test-core | 10 | test_kv_fp8_scale.cpp | 0.01C | 3.8 | KV FP8 scale: mode parse, pow2 ceil, amax/checkpoint sources, histogram bins, EXPECT | n | n |  |
| KvPoolVerdict | test-core | 6 | test_kv_residual_sizing.cpp | 0.00C | 2.5 | KV pool verdict: full sequence sufficient, clamped or floored messages, partial block | n | n |  |
| KvReplan | test-core | 10 | test_kv_replan.cpp | 0.00C | 2.1 | KV replan from cached measurement: ceiling gains freed reserve, live reserve floor | n | n |  |
| KvResidualSizing | test-core | 8 | test_kv_residual_sizing.cpp | 0.00C | 2.5 | KV residual sizing: plan fit, residual clamp vs floor, headroom boundary exact | n | n |  |
| KvScaleTensor | test-core | 3 | test_kv_fp8_scale.cpp | 0.00C | 3.8 | KV scale tensor names parse to K/V slot, host scalar read, matched-step routing | n | n |  |
| KvWriteRopeArgs | test-core | 2 | test_launcher_args.cpp | 0.00C | 3.3 | KV write RoPE args struct: fields in declaration order, omitted fields default | n | n |  |
| LaneNeeds | test-core | 4 | test_lane_needs.cpp | 0.01C | 3.7 | Lane needs: field combinations map to old lanes, logprobs not forced, mid-run bits | n | n |  |
| LatencyLadder | test-core | 5 | test_server_request_limits.cpp | 0.00C | 26.5 | Latency histogram ladders: seconds vs ITL buckets, cumulative sums, negatives clamp | n | n |  |
| LayerHostKeep | test-core | 8 | test_layer_host_keep.cpp | 0.00C | 5.9 | Layer host-keep: Qwen3-8B GPU-layers 20 keeps 16 layers on host, dense-only gate | n | n |  |
| LayerOffloadPlan | test-core | 5 | test_layer_offload_plan.cpp | 0.00C | 3.6 | Layer offload plan: Q8_0 block-sized bytes, 16 host layers planned, init refusals | n | n |  |
| LeakedStaticTest | test-core | 6 | test_leaked_static.cpp | 0.00C | 2.6 | Leaked-static holder: destructor never runs, owners release once, reset idempotent | n | n |  |
| Lfm2Names | test-core | 1 | test_lfm2_names.cpp | 0.00C | 1.7 | LFM2 canonical tensor names: modules and suffixes map, EXPECT_EQ | n | n |  |
| LibraryForwardWindow | test-core | 2 | test_library_reserve_measurement.cpp | 0.00C | 3.2 | Library forward window: tracked growth is no library claim, bytes never wrap | n | n |  |
| LibraryReserveCache | test-core | 9 | test_library_reserve_cache.cpp | 0.00C | 3.5 | Library reserve cache: key stability, file round-trip, missing key, remeasure replaces | n | n |  |
| LibraryReserveMeasurement | test-core | 3 | test_library_reserve_measurement.cpp | 0.00C | 3.2 | Library reserve measurement: no window means none, larger of window and residual | n | n |  |
| LlmCompressorFormatDetect | test-core | 8 | test_llm_compressor_loader.cpp | 0.00C | 9.3 | llm-compressor format detect: ModelOpt preferred, recipe YAML and compressed-tensors | n | n |  |
| LlmCompressorIgnoreList | test-core | 3 | test_llm_compressor_loader.cpp | 0.00C | 9.3 | llm-compressor ignore list: config.json names replace recipe patterns, router-proj rule | n | n |  |
| LlmCompressorKeepVision | test-core | 6 | test_llm_compressor_loader.cpp | 0.00C | 9.3 | llm-compressor keep-vision: visual prefix verbatim, vision suffixes kept, MTP skipped | n | n |  |
| LlmCompressorRecipe | test-core | 6 | test_llm_compressor_loader.cpp | 0.00C | 9.3 | llm-compressor recipe parse: Gemma4 and Qwen3.6 recipes ok, non-NVFP4 scheme rejected | n | n |  |
| LlmCompressorTranslate | test-core | 17 | test_llm_compressor_loader.cpp | 0.00C | 9.3 | llm-compressor tensor-name translate: packed and global-scale renames, vision skipped | n | n |  |
| LlmCompressorUnused | test-core | 4 | test_llm_compressor_loader.cpp | 0.00C | 9.3 | llm-compressor unused-tensor rule: MTP skipped not unused, drop only when no consumer | n | n |  |
| LmHeadRoute | test-core | 3 | test_lm_head_route.cpp | 0.00C | 1.6 | LM-head route decision table, NVFP4 not FP8-cached, draft head route, EXPECT_EQ | n | n |  |
| LoaderRopeParity | test-core | 7 | test_loader_rope_parity.cpp | 0.01C | 3.8 | Loader RoPE parity: Gemma4, Llama3, Granite, Qwen3, Gemma3 params match reference | n | n |  |
| LogLevel | test-core | 2 | test_process_diag.cpp | 0.00C | 2.5 | Log level parse: every word case-insensitive, anything else rejected | n | n |  |
| LogOnce | test-core | 5 | test_log_once.cpp | 0.00C | 2.5 | LogOnce latch: fires once then after rearm, exactly one thread wins a race | n | n |  |
| LogitBiasVocabError | test-core | 1 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Logit-bias vocab check: id at or above vocab size named in the error | n | n |  |
| LogitLensParams | test-core | 1 | test_sampling_params_golden.cpp | 0.00C | 33.6 | Logit-lens params parse: lens count, logprobs flag, EXPECT_EQ | n | n |  |
| LogprobShapes | test-core | 6 | test_logprobs_shapes.cpp | 0.00C | 22.7 | Logprob response shapes: chat array, completions parallel arrays, offsets, streaming | n | n |  |
| LogprobTokens | test-core | 3 | test_logprobs_shapes.cpp | 0.00C | 22.7 | Logprob token JSON: valid UTF-8 unchanged, half codepoint serialises, raw bytes | n | n |  |
| LoraDimsTest | test-core | 5 | test_lora_dims.cpp | 0.00C | 2.5 | LoRA dims: projection widths follow base, GQA kv narrower, FFN targets refused without | n | n |  |
| LtGemmArgs | test-core | 3 | test_gemm_lt_args.cpp | 0.00C | 3.2 | cuBLASLt GEMM args struct: fields in declaration order, GemmEx unstrided batch of one | n | n |  |
| LtGemmBatchedArgs | test-core | 2 | test_gemm_lt_args.cpp | 0.00C | 3.2 | Batched cuBLASLt GEMM args struct: field order and defaults, EXPECT_EQ | n | n |  |
| LtRouteTrace | test-core | 3 | test_lt_route_trace.cpp | 0.00C | 1.9 | cuBLASLt route trace labels: gemm precedence, FP8 label, note and dump follow flag | n | n |  |
| MLAConfig | test-core | 6 | test_mla.cpp | 0.00C | 3.4 | MLA config: YaRN mscale, DeepSeek-V2-Lite fields parse, MLA profile selection | n | n |  |
| MLATensorKindMatcher | test-core | 1 | test_mla.cpp | 0.00C | 3.4 | MLA tensor kind matcher: kv_a and kv_a_layernorm projection kinds, EXPECT_EQ | n | n |  |
| MLAWeightMap | test-core | 5 | test_mla.cpp | 0.00C | 3.4 | MLA weight map: DeepSeek dense, router, routed and shared experts, attention names | n | n |  |
| MRopeConfig | test-core | 4 | test_mrope_config.cpp | 0.00C | 2.8 | M-RoPE config: split read from rope params or scaling, text-only off, malformed ignored | n | n |  |
| MRopePositions | test-core | 11 | test_mrope_positions.cpp | 0.00C | 2.8 | M-RoPE positions: text ascending on 3 axes, image row/col, continuation start | n | n |  |
| MemAccountKeyed | test-core | 2 | test_mem_account_keyed.cpp | 0.00C | 1.7 | Keyed memory accounting: free uncharges exactly charged bytes, unknown pointer no-op | n | n |  |
| MemAccountRanges | test-core | 1 | test_memory_allocators.cpp | 0.00C | 4.8 | Memory account ranges: listed until their pool drops them, counts and bytes, EXPECT | n | n |  |
| MeminfoParseTest | test-core | 2 | test_weight_snapshot.cpp | 0.00C | 4.0 | meminfo parse: MemAvailable, MemTotal and cgroup limit, EXPECT_EQ on parsed values | n | n |  |
| MemoryPlan | test-core | 25 | test_memory_plan.cpp | 0.00C | 9.5 | Memory plan battery: dense reference fits, library reserve line item, KV residual, tight | n | n |  |
| ModelContext | test-core | 5 | test_model_context.cpp | 0.00C | 3.3 | TIMING: Model context length: configured wins, declared capped at auto ceiling, config | n | n |  |
| ModelIdForPath | test-core | 1 | test_model_name_policy.cpp | 0.00C | 5.1 | Model id from path: HF snapshot dir maps to repo id, EXPECT_EQ | n | n |  |
| ModelLifetimeGate | test-core | 8 | test_model_lifetime_gate.cpp | 0.08C | 3.1 | TIMING: Model lifetime gate: exclusive waits for readers, timeout stays usable, stale | n | n |  |
| ModelNamePolicy | test-core | 5 | test_model_name_policy.cpp | 0.00C | 5.1 | Model name policy: basename looked up in models dir, repo-id shape, absolute/traversal | n | n |  |
| ModelProfileTable | test-core | 4 | test_model_profile_table.cpp | 0.00C | 2.7 | Model profile table: one row per arch, SWA windows, loader trait lists, no-SWA mask | n | n |  |
| ModelScan | test-core | 4 | test_model_name_policy.cpp | 0.00C | 5.1 | Model scan lists GGUF and safetensors dirs, symlink loops and unreadable dirs tolerated | n | n |  |
| ModelSwapState | test-core | 4 | test_model_swap_state.cpp | 0.10C | 20.6 | Model swap state: swap or suspend under a lease answers 503 and frees nothing | n | n |  |
| ModeloptMixedPrecision | test-core | 5 | test_llm_compressor_loader.cpp | 0.00C | 9.3 | ModelOpt mixed-precision table: NVFP4 table accepted, missing or non-NVFP4 refused | n | n |  |
| MoeGroupLimit | test-core | 4 | test_moe_group_limit.cpp | 0.00C | 2.5 | MoE group-limited routing matches Ling gate golden, stable under fp16 bias | n | n |  |
| MoeLaunchRows | test-core | 4 | test_routing_decision.cpp | 0.00C | 3.4 | MoE launch rows: one-hot takes all rows, skew ratio, empty inputs zero | n | n |  |
| MoeNvfp4CacheRule | test-core | 4 | test_moe_nvfp4_cache_rule.cpp | 0.00C | 1.3 | MoE NVFP4 whole-layer cache rule: lone eligible down not cached, dense not cached | n | n |  |
| MoePrefillSelect | test-core | 2 | test_moe_prefill_select.cpp | 0.00C | 2.2 | MoE prefill: fused dp4a and IMMA qkind refuse unsupported qtypes, EXPECT_THROW | n | n |  |
| MoePrefillTable | test-core | 7 | test_routing_decision.cpp | 0.00C | 3.4 | MoE prefill path table: device args, GPT-OSS staged/unstaged, grouped and legacy fallback | n | n |  |
| MoeRoutingModelCoupling | test-core | 2 | test_routing_decision.cpp | 0.00C | 3.4 | MoE prefill path replay: every tier replays to itself, GPT-OSS to grouped | n | n |  |
| MoeTierPrecedence | test-core | 3 | test_routing_decision.cpp | 0.00C | 3.4 | MoE prefill tier precedence: device args over grouped over legacy, entry gate first | n | n |  |
| MtpAttnGuard | test-core | 1 | test_mtp_attn_guard.cpp | 0.00C | 2.8 | MTP attention guard: zero KV heads with weights returns false with a log message | n | n |  |
| MtpAuto | test-core | 28 | test_mtp_auto.cpp | 0.00C | 4.3 | MTP auto sentinel: default auto, single-stream and concurrent pair/pool choices, 28 cases | n | n |  |
| MtpDraftRows | test-core | 4 | test_mtp_draft_rows.cpp | 0.00C | 1.5 | MTP draft rows: copy holds prefix or every row, FP8 head and cutlass head vocab | n | n |  |
| MtpExternalHead | test-core | 1 | test_mtp_presence.cpp | 0.00C | 4.4 | MTP external head attaches only a matching Qwen head file | n | n |  |
| MtpHeadGraph | test-core | 7 | test_mtp_head_graph.cpp | 0.00C | 2.2 | MTP head graph: eager first use, capture then replay, ring window and resume rules | n | n |  |
| MtpKvSlots | test-core | 2 | test_mtp_kv_slots.cpp | 0.00C | 2.7 | MTP KV slots: slot committed at first use, refused commit reported, ASSERT_NE | n | n |  |
| MtpLayoutDispatch | test-core | 5 | test_mtp_layout_dispatch.cpp | 0.00C | 3.6 | MTP layout dispatch: Qwen4-Exp 310 names mapped, missing expert scale incomplete | n | n |  |
| MtpNameRule | test-core | 4 | test_mtp_presence.cpp | 0.00C | 4.4 | MTP name rule: both checkpoint spellings accepted, main-model tensors rejected | n | n |  |
| MtpProbe | test-core | 5 | test_mtp_presence.cpp | 0.00C | 4.4 | MTP probe: sidecar, sharded and single-file layouts found, non-head and no-head quiet | n | n |  |
| MtpSiblingHead | test-core | 1 | test_mtp_presence.cpp | 0.00C | 4.4 | MTP sibling head completes only from a byte-equal neighbour file | n | n |  |
| MtpWorkspaceBytes | test-core | 2 | test_memory_plan.cpp | 0.00C | 9.5 | MTP workspace bytes: Qwen3.8 head at server defaults, NVFP4 body copy priced for dense | n | n |  |
| Mxfp8Widen | test-core | 4 | test_mxfp8_widen.cpp | 0.00C | 2.7 | MXFP8 widen: element decode exact, pair to bf16, NaN and NVFP4 scales skipped, E4M3 | n | n |  |
| NGramContext | test-core | 4 | test_ngram_context.cpp | 0.00C | 1.5 | N-gram context: EOS at sequence start, spans prompt and output, sequences isolated | n | n |  |
| NGramHash | test-core | 4 | test_ngram_table.cpp | 0.00C | 2.1 | N-gram hash: EOS-filled start, shift blocked by EOS, carried equals one-shot, Python | n | n |  |
| NgramDraft | test-core | 8 | test_ngram_draft.cpp | 0.00C | 2.1 | N-gram draft: empty without repeat, simple repeat, truncates at K, prefers longer and | n | n |  |
| NonStreamReasoningTokens | test-core | 5 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Non-stream reasoning token count: between markers, prompt opener from token zero, fallback | n | n |  |
| NpyWriterTest | test-core | 4 | test_npy_writer.cpp | 0.00C | 4.5 | NPY writer: parseable header, spec-padded header, row-major payload, shape for one row | n | n |  |
| NvFP4ExpertKey | test-core | 1 | test_nvfp4_expert_offload.cpp | 0.00C | 2.4 | NVFP4 expert key matcher: per-expert MoE weights only, EXPECT_TRUE | n | n |  |
| NvFP4HostExpertsServable | test-core | 2 | test_nvfp4_expert_offload.cpp | 0.00C | 2.4 | NVFP4 host experts servable: promoted host projection accepted, staging-unusable refused | n | n |  |
| NvFP4IgnoreList | test-core | 5 | test_nvfp4_quant_contract.cpp | 0.00C | 9.4 | NVFP4 ignore list enforced, missing global scale refused, inventory counts match | n | n |  |
| NvFP4MergedScaleProvenance | test-core | 7 | test_nvfp4_quant_contract.cpp | 0.00C | 9.4 | NVFP4 merged-scale provenance: fused split recorded, separate tensors no split, slice | n | n |  |
| NvFP4SlotLayout | test-core | 5 | test_nvfp4_expert_offload.cpp | 0.00C | 2.4 | NVFP4 slot layout: expert ranges, uint2 load alignment, scale inside slot, no overlap | n | n |  |
| Nvfp4DequantWs | test-core | 4 | test_nvfp4_dequant_ws.cpp | 0.00C | 1.7 | NVFP4 dequant workspace: none when absent or too small, deferred take runs once | n | n |  |
| Nvfp4OutputBytes | test-core | 2 | test_quantize_policy.cpp | 0.00C | 4.5 | NVFP4 output bytes: packed nibbles, micro scales, tensor scale, empty matrix refused | n | n |  |
| Nvfp4ScratchKey | test-core | 1 | test_nvfp4_quant_contract.cpp | 0.00C | 9.4 | NVFP4 scratch key: malformed index resolves to nothing, EXPECT_EQ | n | n |  |
| Nvfp4Undeclared | test-core | 1 | test_flint.cpp | 0.00C | 3.7 | NVFP4 undeclared module: packed codes without declaration refused and named | n | n |  |
| OffloadSpecDefaults | test-core | 5 | test_offload_spec_defaults.cpp | 0.00C | 2.0 | Offload spec defaults: offload turns recycling on at width 8, operator keys win, MC rows | n | n |  |
| OutOfRangeMessage | test-core | 2 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Out-of-range message: penalty kernel bounds refused, bounds and defaults accepted | n | n |  |
| OutputMarkersInkling | test-core | 8 | test_output_markers.cpp | 0.00C | 2.3 | Inkling output markers: think tags kept only, tool-call header dropped, args renamed | n | n |  |
| PagedDecodeArgs | test-core | 2 | test_paged_q8_epilogue.cpp | 0.00C | 2.5 | Paged decode args struct: fields in declaration order, omitted fields default, EXPECT_EQ | n | n |  |
| PagedDecodeTensorArgs | test-core | 1 | test_paged_q8_epilogue.cpp | 0.00C | 2.5 | Paged decode tensor args: omitted fields take defaults, kv_scale 1.0, EXPECT_EQ | n | n |  |
| PagedFp8Decode | test-core | 2 | test_spec_request.cpp | 0.00C | 2.5 | Paged FP8 decode fast kernel only at head dim 128, serving vs fast question | n | n |  |
| PagedQ8Epilogue | test-core | 3 | test_paged_q8_epilogue.cpp | 0.00C | 2.5 | Paged Q8 epilogue taken once per call, not outside its shape, nothing armed never taken | n | n |  |
| PagedSplitKScratch | test-core | 3 | test_paged_q8_epilogue.cpp | 0.00C | 2.5 | Paged split-K scratch bytes per split, null scratch never splits, small scratch caps | n | n |  |
| ParseCompletionPrompt | test-core | 2 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Completion prompt parse: string, ids and one-element lists accepted, batches refused by | n | n |  |
| ParseGemmaToolCallBody | test-core | 1 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Gemma tool-call body parser rejects malformed bodies, EXPECT_FALSE | n | n |  |
| ParseLogitBias | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Logit-bias parse: absent empty, valid kept, malformed entries error not drop, cap refuses | n | n |  |
| ParseMaxTokensField | test-core | 5 | test_sse_stream_utils.cpp | 0.00C | 31.4 | max_tokens parse: default, max_tokens and max_completion_tokens precedence, nulls ignored | n | n |  |
| ParseStopField | test-core | 5 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Stop field parse: string or array up to 16, truncation reported, non-strings skipped | n | n |  |
| ParseStreamToolBody | test-core | 2 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Stream tool-call body parse: Gemma body, JSON then XML fallback, failure | n | n |  |
| PathWithin | test-core | 4 | test_model_name_policy.cpp | 0.00C | 5.1 | path_within: inside vs outside, sibling prefix outside, traversal resolved before compare | n | n |  |
| PhaseFixture | test-core | 26 | test_memory_backend.cpp | 0.09C | 3.5 | Phase fixture: region release on destruction, move ownership once, reset idempotent | n | n |  |
| PhraseBan | test-core | 6 | test_phrase_ban.cpp | 0.00C | 3.1 | Phrase ban: match ends in new text only, token-at-byte, snapshot pick, regex battery | n | n |  |
| PinnedBuffer | test-core | 13 | test_host_pinned.cpp | 0.00C | 3.5 | Pinned buffer RAII: frees exactly once, failed acquire is empty, zero bytes refused, move | n | n |  |
| PreDequantBudget | test-core | 4 | test_pre_dequant_budget.cpp | 0.00C | 2.0 | Pre-dequant budget: NVFP4 reservation charged to early phases only, KV growth once, clamp | n | n |  |
| PrefillGraphCache | test-core | 3 | test_prefill_graph_cache.cpp | 0.00C | 2.2 | Prefill graph cache: short keys keep a runner each, long keys share one on consecutive | n | n |  |
| PrefillPacing | test-core | 7 | test_prefill_pacing.cpp | 0.00C | 1.7 | Prefill pacing: uncapped with no decoder, cap off, weight arithmetic, protection scenario | n | n |  |
| ProcessDiag | test-core | 6 | test_process_diag.cpp | 0.00C | 2.5 | Process diagnostics install: mirrored fields transferred, defaults, log level applied | n | n |  |
| PromptLogprobs | test-core | 1 | test_prompt_logprobs.cpp | 0.00C | 10.7 | Prompt logprobs chunk rows formula, EXPECT_EQ against max chunk rows | n | n |  |
| PromptLogprobsAssembly | test-core | 7 | test_prompt_logprobs.cpp | 0.00C | 10.7 | Prompt logprobs assembly: vLLM and echo shapes, limit cuts alternatives not the target | n | n |  |
| PromptLogprobsParse | test-core | 5 | test_prompt_logprobs.cpp | 0.00C | 10.7 | Prompt logprobs parse: off when absent, range 0-20, stream refused, engine top-N | n | n |  |
| PromptTail | test-core | 7 | test_prompt_tail.cpp | 0.00C | 1.8 | Prompt tail split: issue split keeps 33 rows, reuse cap leaves minimum rows, short prompt | n | n |  |
| QknormRopeArgs | test-core | 2 | test_launcher_args.cpp | 0.00C | 3.3 | QK-norm RoPE args struct: fields in declaration order, omitted fields default | n | n |  |
| QuantReport | test-core | 6 | test_quantize_report.cpp | 0.00C | 2.8 | Quant error report: exact values zero error, off-grid distance, worst tensors ordered | n | n |  |
| QuantizeCheckpointOut | test-core | 28 | test_quantize_checkpoint_out.cpp | 0.00C | 6.1 | Checkpoint writer: format names, per-format tensor names, global scale, fused group match | n | n |  |
| QuantizeFp8Head | test-core | 6 | test_quantize_checkpoint_out.cpp | 0.00C | 6.1 | FP8 head writer: modes and defaults, writes codes and row scales under loader names | n | n |  |
| QuantizeKvHint | test-core | 2 | test_quantize_checkpoint_out.cpp | 0.00C | 6.1 | KV FP8 hint allowlist covers measured families only, loader reads the writer hint | n | n |  |
| QuantizePolicy | test-core | 7 | test_quantize_policy.cpp | 0.00C | 4.5 | Quantize policy: linear weights taken, MLA latent and router refused, embeddings kept | n | n |  |
| QuantizeRecipe | test-core | 5 | test_quantize_recipe.cpp | 0.00C | 3.3 | Quantize recipe JSON: fixed field order, uncalibrated null, same recipe same bytes | n | n |  |
| QueuedTokenGate | test-core | 7 | test_server_request_limits.cpp | 0.01C | 26.5 | TIMING: Queued-token gate: admits up to cap, refuses over, lease releases once, concurrent | n | n |  |
| Qwen3VLVisionConfig | test-core | 7 | test_qwen3vl_vision_config.cpp | 0.00C | 2.5 | Qwen3-VL vision config: real checkpoint parse, inconsistent geometry and deepstack refused | n | n |  |
| Qwen3VLVisionGrid | test-core | 10 | test_qwen3vl_vision_grid.cpp | 0.00C | 2.5 | Qwen3-VL vision grid: merge-block token order, patch coverage, interpolation weights | n | n |  |
| Qwen3VLVisionLoad | test-core | 9 | test_qwen3vl_vision_load.cpp | 0.00C | 3.0 | Qwen3-VL vision load: real checkpoint loads, null slot or shape mismatch refused | n | n |  |
| Qwen3VLVisionMap | test-core | 7 | test_qwen3vl_vision_map.cpp | 0.00C | 2.8 | Qwen3-VL vision tensor map: every real tensor recognised once, malformed index unknown | n | n |  |
| QwenPatchify | test-core | 4 | test_vision_preprocess.cpp | 0.00C | 4.7 | Qwen patchify: grid and shape, merge-block grouping, channel-major temporal repeat | n | n |  |
| QwenSmartResize | test-core | 7 | test_vision_preprocess.cpp | 0.00C | 4.7 | Qwen smart-resize: aligned unchanged, min/max pixel scaling, ties round to even | n | n |  |
| QwenVideoFold | test-core | 1 | test_qwen3vl_video.cpp | 0.00C | 5.4 | Qwen video fold: pads rewritten and token types kept, EXPECT_EQ | n | n |  |
| QwenVideoLayout | test-core | 7 | test_qwen3vl_video.cpp | 0.00C | 5.4 | Qwen video layout: grids, timestamps, placeholder ids and rope positions match HF | n | n |  |
| QwenVideoPatchify | test-core | 4 | test_qwen3vl_video.cpp | 0.00C | 5.4 | Qwen video patchify matches HF video processor within one fp16 ulp, bad input refused | n | n |  |
| QwenVideoSmartResize | test-core | 1 | test_qwen3vl_video.cpp | 0.00C | 5.4 | Qwen video smart-resize matches HF for stored vectors, EXPECT_EQ | n | n |  |
| RateLimitKey | test-core | 3 | test_server_request_limits.cpp | 0.00C | 26.5 | Rate-limit key: forwarded-for ignored from untrusted peer, trusted proxy honoured | n | n |  |
| RateLimiter | test-core | 3 | test_server_request_limits.cpp | 0.00C | 26.5 | TIMING: Rate limiter: quiet buckets evicted, window slides, refuses past limit, admits | n | n |  |
| ReasoningFinishDetail | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Reasoning finish detail: fires only when answer lost to reasoning, field written only then | n | n |  |
| RecurrentSlotForReset | test-core | 3 | test_recurrent_slot.cpp | 0.00C | 2.0 | Recurrent slot for reset: known id returns its slot, unknown id resets nothing | n | n |  |
| RegexConstrain | test-core | 15 | test_regex_constrain.cpp | 0.00C | 3.0 | Regex constrainer: accepts only matching continuation, multi-char tokens, alternation | n | n |  |
| RegexNfaCostBounds | test-core | 4 | test_regex_constrain.cpp | 0.01C | 3.0 | Regex NFA cost bounds: huge repeats, nested repeats, deep groups refused, ordinary | n | n |  |
| RequestTables | test-core | 7 | test_request_tables.cpp | 0.00C | 6.1 | Request tables: eight tables in finish order, cancel leaves no entry, release idempotent | n | n |  |
| ResponseStore | test-core | 8 | test_responses_store.cpp | 0.00C | 7.1 | Response store: put/get round-trip, TTL, purge, entry and byte caps evict LRU | n | n |  |
| ResponsesContinue | test-core | 2 | test_responses_store.cpp | 0.00C | 7.1 | Responses continue: merged input equals stateless resend, chaining, string input | n | n |  |
| ResponsesPrefill | test-core | 1 | test_assistant_prefill_transform.cpp | 0.00C | 4.5 | Responses prefill: continue_final_message forwarded, EXPECT_TRUE/EXPECT_FALSE | n | n |  |
| ResponsesTransform | test-core | 34 | test_responses_transform.cpp | 0.00C | 16.7 | Responses-to-chat transform: input items, tool round-trip, tool nesting, format knobs | n | n |  |
| RopeOverrideTest | test-core | 10 | test_rope_override.cpp | 0.00C | 2.8 | RoPE override: off by default, YaRN and linear injection, explicit origin ctx, refusals | n | n |  |
| RopeScalingConfigTest | test-core | 18 | test_hf_config_loader.cpp | 0.00C | 6.6 | RoPE scaling config: Llama3 factor table, tie-embeddings tri-state, per-arch layer maps | n | n |  |
| RoutingMtpOwnersTest | test-core | 2 | test_routing_mtp_owners.cpp | 0.00C | 2.4 | Routing and MTP owners: buffers released once, move leaves no view, free clears views | n | n |  |
| RuntimeConfigTest | test-core | 24 | test_config.cpp | 0.00C | 4.7 | Runtime config defaults and bindings: KV block size, lm-dequant bisect, SWA tri-state | n | n |  |
| SSMStateLazy | test-core | 3 | test_memory_allocators.cpp | 0.00C | 4.8 | SSM state lazy: reserves every slot commits none, ensure commits only that slot | n | n |  |
| Sae | test-core | 1 | test_sae.cpp | 0.00C | 3.1 | SAE token info: top by activation then id, relative error, EXPECT_FLOAT_EQ | n | n |  |
| SaeDir | test-core | 2 | test_sae.cpp | 0.00C | 3.1 | SAE directory load: layer and tensors load, mismatches refused | n | n |  |
| SafeTensorsAwq | test-core | 3 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors AWQ: unsupported variants refused with detected config, 4-bit zero-point load | n | n |  |
| SafeTensorsDtypeTable | test-core | 2 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors dtype table: servable rows have matching widths, mismatched wire types refused | n | n |  |
| SafeTensorsGptq | test-core | 5 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors GPTQ: unknown format refused, pre-fix qzeros refused, auto-GPTQ v1/v2 offsets | n | n |  |
| SafeTensorsHighDimTensors | test-core | 1 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors high-dim tensors flattened not dropped, conv5d in the captured log | n | n |  |
| SafeTensorsHostileHeader | test-core | 6 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors hostile header: bad dtype, huge offset, negative/overflow shapes dropped | n | n |  |
| SafeTensorsMalformedEntryWarnings | test-core | 2 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors malformed entries: missing dtype/shape and byte-count mismatch warn | n | n |  |
| SafeTensorsRaw | test-core | 2 | test_safetensors_writer.cpp | 0.00C | 3.2 | SafeTensors raw writer/reader round-trip, truncated and bogus files rejected | n | n |  |
| SafeTensorsValidateHeaderSize | test-core | 6 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors header size validation: below 8 bytes, exceeds file, uint64 overflow, soft cap | n | n |  |
| SafeTensorsValidateTensorOffsets | test-core | 7 | test_safetensors_loader.cpp | 0.00C | 4.8 | SafeTensors tensor offset validation: start/end past file, byte mismatch, zero-size valid | n | n |  |
| SafeTensorsWriter | test-core | 5 | test_safetensors_writer.cpp | 0.00C | 3.2 | SafeTensors writer: round-trip of bytes and header, contiguous offsets, escaped names | n | n |  |
| SamplingDefaultsParity | test-core | 4 | test_sampling_defaults_parity.cpp | 0.00C | 4.0 | Sampling defaults parity: CLI equals server per arch, Qwen preset, generation_config | n | n |  |
| SamplingParamsGolden | test-core | 1 | test_sampling_params_golden.cpp | 0.02C | 33.6 | Sampling params golden: every dialect matches the recorded parse, golden file loads | n | n |  |
| SanitizeForEcho | test-core | 3 | test_server_request_limits.cpp | 0.00C | 26.5 | sanitize_for_echo: non-printable replaced, truncation marker, result serialises | n | n |  |
| SchemaAdditionalProperties | test-core | 10 | test_schema_constrain_property.cpp | 0.00C | 6.4 | JSON-schema additionalProperties: explicit true, free-form objects, malformed JSON | n | n |  |
| SchemaConstrainPropertyTest | test-core | 8 | test_schema_constrain_property.cpp | 0.02C | 6.4 | JSON-schema constrainer property battery: conforming docs and prefixes accepted, missing | n | n |  |
| SchemaDepthCap | test-core | 2 | test_schema_constrain_property.cpp | 0.00C | 6.4 | JSON-schema depth cap: deep nesting refused, moderate nesting parses | n | n |  |
| SchemaEnumLiteral | test-core | 2 | test_schema_constrain_property.cpp | 0.00C | 6.4 | JSON-schema enum literals: numbers, booleans, null and mixed legal-token checks | n | n |  |
| SchemaEnvelope | test-core | 1 | test_schema_constrain_property.cpp | 0.00C | 6.4 | Schema envelope: forced open bounds leading whitespace, token_legal battery | n | n |  |
| SchemaIntegerBound | test-core | 3 | test_schema_constrain_property.cpp | 0.00C | 6.4 | Schema integer bound: digits stop at int64 width, short ints unaffected, number unbounded | n | n |  |
| SchemaKeywords | test-core | 12 | test_schema_keywords.cpp | 0.00C | 4.0 | Schema allOf/anyOf/optional keywords: value parsed not a string, EXPECT_EQ | n | n |  |
| SchemaKeywordsProperty | test-core | 2 | test_schema_keywords.cpp | 0.01C | 4.0 | Schema keywords property: conforming docs complete, every prefix legal, no dead end | n | n |  |
| SchemaParserDesync | test-core | 6 | test_schema_constrain_property.cpp | 0.00C | 6.4 | Schema parser desync: additionalProperties variants, enum literals, trailing input error | n | n |  |
| SchemaUnenforceableKeywords | test-core | 3 | test_schema_constrain_property.cpp | 0.00C | 6.4 | Schema unenforceable assertion keywords refused, annotations ignored, const as enum | n | n |  |
| SchemaUnicodeEscape | test-core | 4 | test_schema_constrain_property.cpp | 0.00C | 6.4 | Schema JSON unicode escapes: BMP, surrogate pairs, 3-byte escapes decode to UTF-8 | n | n |  |
| SentencePieceLoader | test-core | 10 | test_sentencepiece_loader.cpp | 0.00C | 3.1 | SentencePiece model parse: pieces, scores, trainer spec, truncated protobuf refused | n | n |  |
| ServableContext | test-core | 5 | test_server_request_limits.cpp | 0.00C | 26.5 | Servable context: pool wins when smaller, plan stands when pool larger, growable ceiling | n | n |  |
| ServerArgs | test-core | 7 | test_server_args.cpp | 0.00C | 3.2 | Server args defaults and parses: idle unload, responses store, VRAM budget, limits | n | n |  |
| ServerArgsDeathTest | test-core | 3 | test_server_args.cpp | 0.08C | 3.2 | Server args death test: garbage values exit via EXPECT_EXIT | n | n |  |
| ServerJsonKeyOrder | test-core | 3 | test_server_json_key_order.cpp | 0.00C | 7.2 | Server JSON key order kept through parse/dump for tool schemas and tool_use input | n | n |  |
| ServerVisionInline | test-core | 5 | test_server_vision_inline.cpp | 0.00C | 3.8 | Vision inline placement: blocks land where parts sat, count mismatch strips and reports | n | n |  |
| ServingMetadataLayout | test-core | 4 | test_serving_metadata_layout.cpp | 0.00C | 2.1 | Serving metadata layout: 256-byte aligned disjoint regions, size bounds, SWA off adds | n | n |  |
| ShadowPlan | test-core | 16 | test_memory_plan.cpp | 0.00C | 9.5 | Shadow plan: reports both pool ceilings, capped admission sizes KV, zero-KV names context | n | n |  |
| ShouldQuantize | test-core | 1 | test_quantize_policy.cpp | 0.00C | 4.5 | Quantize filter leaves the MTP draft head alone, EXPECT_FALSE | n | n |  |
| Sm120FmhaTileSelect | test-core | 3 | test_fmha_sm120_tile_select.cpp | 0.00C | 2.2 | SM120 FMHA tile selection: every selected tile has an instance, old holes gone | n | n |  |
| SmallMXqTag | test-core | 3 | test_smallm_xq_tag.cpp | 0.00C | 1.4 | Small-M MXFP quantize tag: producer stamp live until buffer touched, consumer adopts own | n | n |  |
| SnapshotHostDefault | test-core | 4 | test_snapshot_host_default.cpp | 0.00C | 2.2 | Snapshot host default: spill off keeps default, spill raises host tier to half budget | n | n |  |
| Span | test-core | 1 | test_memory_allocators.cpp | 0.00C | 4.8 | Span: stable widens to device span, subspans inherit the guarantee, arena-backed | n | n |  |
| SparseGeometry | test-core | 7 | test_sparse_attn_geometry.cpp | 0.00C | 2.1 | Sparse attention geometry: budget, engage threshold, sink and recent windows per block | n | n |  |
| SpecAcceptRows | test-core | 4 | test_spec_gates.cpp | 0.00C | 2.9 | Speculative accept rows: unfinished chunk keeps bonus outside rows, think edge, stop ends | n | n |  |
| SpecBatchRr | test-core | 3 | test_spec_request.cpp | 0.00C | 2.5 | Spec batch round-robin verify active only with a batch and a non-recurrent model | n | n |  |
| SpecField | test-core | 7 | test_spec_usage.cpp | 0.00C | 23.2 | Speculative request field: absent leaves unset, bool meaning kept, depth object, range 400 | n | n |  |
| SpecGates | test-core | 12 | test_spec_gates.cpp | 0.00C | 2.9 | Speculative gates: MTP, n-gram, token recycling, DSpark enable independently, nothing | n | n |  |
| SpecRequest | test-core | 10 | test_spec_request.cpp | 0.00C | 2.5 | Speculative request resolve: unset takes server default, request lowers depth, false wins | n | n |  |
| SpecUsage | test-core | 4 | test_spec_usage.cpp | 0.00C | 23.2 | Spec usage block: absent without verify step, counters beside details, decline reported | n | n |  |
| SpecUsageKeys | test-core | 2 | test_spec_usage.cpp | 0.00C | 23.2 | Spec usage keys: table matches chat shape, copy moves present keys only, EXPECT_EQ | n | n |  |
| SpecialTokensMapTest | test-core | 3 | test_hf_config_loader.cpp | 0.00C | 6.6 | special_tokens_map.json: Mistral object form and Qwen string form parse, missing file | n | n |  |
| SseChunk | test-core | 5 | test_sse_stream_utils.cpp | 0.00C | 31.4 | SSE chunk JSON: envelope, finish reason and logprobs, completion shape, finish detail | n | n |  |
| SseWriter | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | SSE writer frames: content and reasoning keys, id/model escaped, no NUL byte in frame | n | n |  |
| SsmScanPrefillArgs | test-core | 1 | test_launcher_args.cpp | 0.00C | 3.3 | SSM scan prefill args struct: fields in declaration order, dtype fields, EXPECT_EQ | n | n |  |
| SsmStatePool | test-core | 4 | test_memory_plan.cpp | 0.00C | 9.5 | SSM state pool: Qwen3.8 geometry sized for allocator, tail alignment, failure fatal only | n | n |  |
| SsrfClassifier | test-core | 9 | test_image_fetch.cpp | 0.00C | 2.6 | SSRF classifier: loopback, cloud metadata, private ranges, v4-mapped and unspecified | n | n |  |
| SsrfFetch | test-core | 5 | test_image_fetch.cpp | 0.00C | 2.6 | SSRF fetch: remote fetch off by default, private hosts refused even when enabled, scheme | n | n |  |
| StackedExperts | test-core | 3 | test_quantize_policy.cpp | 0.00C | 4.5 | Stacked experts found under each naming convention, 3-D and non-float tensors ignored | n | n |  |
| StepSeed | test-core | 2 | test_step_seed.cpp | 0.00C | 4.9 | Step seed: seeded request adds output count, unseeded keeps one base, EXPECT_EQ 10 | n | n |  |
| StopMask | test-core | 7 | test_think_stop_logic.cpp | 0.00C | 4.9 | Stop mask: masks in think with budget and during grace, released on content or hard cap | n | n |  |
| StreamEquality | test-core | 2 | test_stream_pipeline.cpp | 0.00C | 4.0 | Streaming equals non-stream: concat equals full decode, stop truncation matches non-stream | n | n |  |
| StreamMultibyteSim | test-core | 1 | test_stream_pipeline.cpp | 0.00C | 4.0 | Stream multibyte: split codepoint never emitted as half, UTF-8 buffer empty at end | n | n |  |
| StreamReasoningSplit | test-core | 21 | test_sse_stream_utils.cpp,test_stream_reasoning_split.cpp | 0.00C | 34.7 | Stream reasoning split: second think pass stays out of content, reentry across pieces | n | n |  |
| StreamSim | test-core | 1 | test_stream_pipeline.cpp | 0.00C | 4.0 | Stream sim: stop match split across chunks is caught, emitted text and stopped flag | n | n |  |
| StrictCliNumbers | test-core | 2 | test_server_args.cpp | 0.00C | 3.2 | Strict CLI numbers: whole value or refused, valid values still parse | n | n |  |
| StrictCliNumbersDeathTest | test-core | 1 | test_server_args.cpp | 0.05C | 3.2 | Strict CLI numbers death test: malformed value exits 1 naming the flag | n | n |  |
| StrictToolConstraint | test-core | 5 | test_tool_call.cpp | 0.00C | 16.5 | Strict tool constraint: loose tool does not disable strict one, unnamed tool refused | n | n |  |
| StripChannelHeaders | test-core | 1 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Strip channel headers: header tokens removed, body kept, EXPECT_EQ | n | n |  |
| StripThinkBlock | test-core | 4 | test_reasoning_reconcile.cpp,test_sse_stream_utils.cpp | 0.00C | 51.6 | Strip think block: unclosed trailing or leading think cleared, block removed, answer kept | n | n |  |
| StructuredOutputThinking | test-core | 5 | test_reasoning_reconcile.cpp | 0.00C | 20.2 | Structured output thinking: whole-reply constraint excludes thinking, unconstrained may | n | n |  |
| SuffixDraft | test-core | 17 | test_suffix_draft.cpp | 0.00C | 2.5 | Suffix draft: empty without repeat, repeat found, majority vote beats recency, truncation | n | n |  |
| SuffixShadow | test-core | 5 | test_suffix_shadow.cpp | 0.00C | 2.4 | Suffix shadow: line waits for outcome, full agreement no draft, declined step replaced | n | n |  |
| SystemFingerprint | test-core | 2 | test_server_request_limits.cpp | 0.00C | 26.5 | system_fingerprint stable per model, differs across models, empty name still yields one | n | n |  |
| TensorKindMatcher | test-core | 8 | test_tensor_kind_matcher.cpp | 0.00C | 2.1 | Tensor kind matcher: attention, FFN, MoE, GDN, norms, embeddings, unknown names | n | n |  |
| TensorKindTable | test-core | 7 | test_tensor_kind_table.cpp | 0.00C | 3.0 | Tensor kind table: entry per kind, GDN and norms restricted to FP16/FP32 tiers | n | n |  |
| TensorTest | test-core | 10 | test_tensor.cpp | 0.00C | 2.7 | Tensor basics: default, numel, nbytes, strides, reshape, dtype names, scalar | n | n |  |
| TextThink | test-core | 5 | test_think_stop_logic.cpp | 0.00C | 4.9 | Text think-state machine: enter/exit across split tokens, marker survives window eviction | n | n |  |
| ThinkCloser | test-core | 1 | test_think_stop_logic.cpp | 0.00C | 4.9 | Think closer: message end after answer does not mask stop, grace holds stop back | n | n |  |
| ThinkLimit | test-core | 5 | test_think_stop_logic.cpp | 0.00C | 4.9 | Think limit: graph and host paths agree, reserve follows config, cap bounds default | n | n |  |
| ThinkSeedFromPromptTail | test-core | 4 | test_think_stop_logic.cpp | 0.00C | 4.9 | Think seed from prompt tail: trailing opener seeds both flags, closed block seeds neither | n | n |  |
| ThinkTokenAccept | test-core | 6 | test_think_stop_logic.cpp | 0.00C | 4.9 | Think token accept: control and user-defined types accepted, normal and absent rejected | n | n |  |
| ThinkingReconcile | test-core | 7 | test_reasoning_reconcile.cpp | 0.00C | 20.2 | Thinking reconcile: open prefix turns thinking on, closed block defaults off, explicit | n | n |  |
| TokenCategory | test-core | 2 | test_json_constrain_property.cpp | 0.00C | 6.9 | Token category: non-ASCII counts as string content, string-ending tokens carry quote | n | n |  |
| TokenExtras | test-core | 2 | test_token_extras.cpp | 0.00C | 4.0 | Token extras: handler keeps worker copy of logprobs as request grows, no copy without them | n | n |  |
| TokenRecycle | test-core | 17 | test_token_recycle_draft.cpp | 0.00C | 2.5 | Token recycling draft: pair chain, most recent pair wins, eviction, top-k rank order, 17 | n | n |  |
| TokenSpansTest | test-core | 5 | test_logprobs_shapes.cpp | 0.00C | 22.7 | Token spans: flush attributes pieces on token boundary, remainder rebased, untracked bytes | n | n |  |
| TokenizerFlagsTest | test-core | 5 | test_hf_config_loader.cpp | 0.00C | 6.6 | Tokenizer flags from tokenizer_config: add_bos, Mistral BOS/EOS, default system prompt | n | n |  |
| ToolArgCoerce | test-core | 1 | test_tool_call_glm.cpp | 0.00C | 10.6 | Tool-argument coercion matches vLLM coerce-to-schema-type table, golden rows | n | n |  |
| ToolCallChatML | test-core | 6 | test_tool_call.cpp | 0.00C | 16.5 | ChatML tool-call parse: single and sequential calls, arguments key, malformed body, drift | n | n |  |
| ToolCallCohere | test-core | 1 | test_tool_call.cpp | 0.00C | 16.5 | Cohere action array parses through the ChatML parser into two named calls | n | n |  |
| ToolCallDialectGolden | test-core | 1 | test_tool_call_dialect_golden.cpp | 0.01C | 8.8 | Tool-call dialect golden: all families match the recorded fixture | n | n |  |
| ToolCallDialectRegistry | test-core | 2 | test_tool_call_dialect_registry.cpp | 0.00C | 3.9 | Tool-call dialect registry: every family matches pre-registry switches, ChatML default | n | n |  |
| ToolCallDispatch | test-core | 1 | test_tool_call.cpp | 0.00C | 16.5 | Tool-call dispatch routes each family to its parser, EXPECT_EQ on name | n | n |  |
| ToolCallFault | test-core | 4 | test_tool_call_xml_params.cpp | 0.00C | 10.3 | Tool-call fault line: clean call has none, raw text on request, raw capped at 2000 bytes | n | n |  |
| ToolCallGemma | test-core | 3 | test_tool_call.cpp | 0.00C | 16.5 | Gemma tool-call parse: single call, no call returns content, digit-leading token kept | n | n |  |
| ToolCallGlm | test-core | 5 | test_tool_call_glm.cpp | 0.00C | 10.6 | GLM tool-call parse: arg-key body, fault line raw body, pairs match vLLM finditer | n | n |  |
| ToolCallHarmony | test-core | 8 | test_tool_call.cpp | 0.00C | 16.5 | Harmony tool-call parse: recipient channel becomes call, truncated call parses, final not | n | n |  |
| ToolCallLlama3 | test-core | 8 | test_tool_call.cpp | 0.00C | 16.5 | Llama-3 tool-call parse: single, bare JSON object, plain JSON answer not a call | n | n |  |
| ToolCallQwen36Xml | test-core | 21 | test_tool_call.cpp,test_tool_call_xml_params.cpp | 0.00C | 26.8 | Qwen3.6 XML tool-call parse: parameter coercion, string params, no-arg call, 21 cases | n | n |  |
| ToolCallReconstruct | test-core | 7 | test_tool_call.cpp | 0.00C | 16.5 | Tool-call reconstruct: content kept, each dialect re-wraps parsed call, no-function entry | n | n |  |
| ToolCallValidate | test-core | 6 | test_tool_call.cpp | 0.00C | 16.5 | Tool-call args validate against schema: required, type, non-object, unknown tool no-op | n | n |  |
| ToolChoice | test-core | 5 | test_constraint_validation.cpp | 0.00C | 33.1 | tool_choice validation: named tool must exist, required without tools refused | n | n |  |
| ToolChoiceEnforcement | test-core | 10 | test_tool_call.cpp | 0.00C | 16.5 | tool_choice enforcement per family: named and required enforced only where supported | n | n |  |
| ToolNamesFromRequest | test-core | 1 | test_tool_call.cpp | 0.00C | 16.5 | Tool names from request: OpenAI and bare shapes extracted, EXPECT_EQ | n | n |  |
| ToolResponseFormat | test-core | 6 | test_tool_call.cpp | 0.00C | 16.5 | Tool response format: content passthrough, non-string serialised, text parts joined | n | n |  |
| ToolResultImages | test-core | 4 | test_tool_result_images.cpp | 0.00C | 6.4 | Tool-result images: picture moves to a new user turn behind the tool run, ordering rules | n | n |  |
| ToolStreamFilterChatML | test-core | 18 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Tool stream filter ChatML: chunked JSON body, args streamed before close, escapes | n | n |  |
| ToolStreamFilterCohere | test-core | 1 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Tool stream filter Cohere: action array yields one call per element across chunk sizes | n | n |  |
| ToolStreamFilterGemma | test-core | 5 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Tool stream filter Gemma: char-by-char, whole string, random chunks, two calls in one feed | n | n |  |
| ToolStreamFilterLlama3 | test-core | 2 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Tool stream filter Llama-3: args stream incrementally, chunked input yields same call | n | n |  |
| ToolStreamFilterUtf8 | test-core | 4 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Tool stream filter UTF-8: argument deltas never end mid-codepoint, 4-byte chars survive | n | n |  |
| ToolTagScan | test-core | 2 | test_tool_stream_filter.cpp | 0.00C | 15.4 | Tool tag scan: Gemma marker prefixes held back, ChatML and Llama-3 unchanged | n | n |  |
| TracingTest | test-core | 7 | test_tracing.cpp | 0.00C | 8.0 | Tracing: traceparent accepted only in well-formed version 00, random ids width and | n | n |  |
| TypeTablesGolden | test-core | 2 | test_type_tables_golden.cpp | 0.00C | 2.1 | Wire and QType type tables match golden for every id, element bytes and names | n | n |  |
| UploadExceedsCheckpoint | test-core | 4 | test_kv_residual_sizing.cpp | 0.00C | 2.5 | Upload exceeds checkpoint: measured pair, quarter boundary inclusive, unknown size not a | n | n |  |
| UploadReserve | test-core | 7 | test_upload_reserve.cpp | 0.00C | 4.5 | Upload reserve reproduces logged 64-slot value, Qwen3.8 batch kept by trimming KV, KV | n | n |  |
| Utf8 | test-core | 3 | test_stream_pipeline.cpp | 0.00C | 4.0 | utf8_complete_len: complete ASCII and multibyte, truncated trailing sequence held back | n | n |  |
| Utf8ChunkLen | test-core | 4 | test_sse_stream_utils.cpp | 0.00C | 31.4 | UTF-8 chunk length never splits a multibyte character, chunks reassemble, always | n | n |  |
| Utf8Stitch | test-core | 5 | test_sse_stream_utils.cpp | 0.00C | 31.4 | UTF-8 stitch rejoins split characters across pieces, invalid lead not held back forever | n | n |  |
| VideoEvs | test-core | 12 | test_video_evs.cpp | 0.00C | 4.2 | Video EVS pruning: retained count floors at one pair, ties keep lower index, pad layout | n | n |  |
| ViewOrOwnedTest | test-core | 10 | test_view_or_owned.cpp | 0.00C | 3.4 | View-or-owned: view set/reset never releases, owned replace releases once, idempotent | n | n |  |
| VisionCacheKeys | test-core | 5 | test_vision_cache_keys.cpp | 0.00C | 2.0 | Vision cache keys: text before first picture keys as text, new picture leaves old keys | n | n |  |
| VisionFamily | test-core | 4 | test_vision_family.cpp | 0.00C | 2.9 | Vision family registry: exact names map, loadable only with encoder, probe follows | n | n |  |
| VisionLoaderCheck | test-core | 12 | test_vision_loader_check.cpp | 0.00C | 2.4 | Vision loader check: Qwen3-VL merger refused, Gemma3/4v tower complete, fused QKV caught | n | n |  |
| VisionPreprocess | test-core | 6 | test_vision_preprocess.cpp | 0.00C | 4.7 | Vision preprocess: CHW layout, channel order, mean/std, resize to square, garbage fails | n | n |  |
| VisionPromptLayout | test-core | 1 | test_vision_prompt_layout.cpp | 0.00C | 2.7 | Vision prompt layout matches golden recorded on main, EXPECT_EQ over model types and cases | n | n |  |
| VramReserveFloor | test-core | 2 | test_kv_residual_sizing.cpp | 0.00C | 2.5 | VRAM reserve floor: default equals allocator headroom, policy percent only raises it | n | n |  |
| WeightCacheEntriesTest | test-core | 12 | test_weight_cache_entries.cpp | 0.00C | 4.7 | Weight cache entries: FP16 move leaves source empty, map erase releases once, NVFP4 owner | n | n |  |
| WeightKeyTest | test-core | 1 | test_weight_snapshot.cpp | 0.00C | 4.0 | Weight key canonical formats: L<layer>.<name> strings, EXPECT_STREQ | n | n |  |
| WeightMapExpertGuard | test-core | 5 | test_weight_map_expert_gate.cpp | 0.00C | 3.4 | Weight map expert guard: up/down-only MoE accepted, refused without any expert, dense | n | n |  |
| WeightMapModalitySkips | test-core | 7 | test_weight_map_modality_skips.cpp | 0.00C | 4.2 | Weight map modality skips: audio and vision counted apart, text-only load skips nothing | n | n |  |
| WeightMapOutNormWarn | test-core | 3 | test_weight_map_out_norm_warn.cpp | 0.00C | 3.5 | Weight map out-norm warning: mixer without final norm not warned, missing norm warns | n | n |  |
| WeightOwnedTest | test-core | 10 | test_weight_owned.cpp | 0.00C | 2.4 | Weight owner: scope end frees once on adoption stream, reset idempotent, move transfers | n | n |  |
| WeightPageRelease | test-core | 4 | test_weight_page_release.cpp | 0.00C | 2.8 | Weight page release: data still reads back after release, split shards released | n | n |  |
| WeightRecordIndicesTest | test-core | 1 | test_weight_snapshot.cpp | 0.00C | 4.0 | Weight record indices: file-supplied indices bounds-checked, EXPECT_TRUE | n | n |  |
| WeightSnapshotArmSlotTest | test-core | 1 | test_weight_snapshot.cpp | 0.00C | 4.0 | Weight snapshot arm slot: arm, take, disarm sequence on the global slot, EXPECT_EQ | n | n |  |
| WeightTierEntries | test-core | 2 | test_weight_tier_entries.cpp | 0.00C | 2.9 | Weight tier entries: bind takes source map nodes, every read tier needs its entry | n | n |  |
| WeightUploadLogTest | test-core | 6 | test_weight_snapshot.cpp | 0.00C | 4.0 | Weight upload log: record resolves sizes and offsets, evict marks dead and drops live | n | n |  |
| WeightUploadTraits | test-core | 2 | test_weight_upload_staging.cpp | 0.08C | 4.6 | Weight upload traits: per-format uploaders byte-for-byte equal, one-byte change detected | n | n |  |
| WrongFieldTypeMessage | test-core | 3 | test_sse_stream_utils.cpp | 0.00C | 31.4 | Wrong-field-type message names the first wrong field, valid bodies and nulls not reported | n | n |  |
| AdmissionCapacityTest | test-e2e | 1 | test_api_generate.cpp | GPU pending | 2.7 | E2E: prompt larger than the KV pool returns a typed error, not cancellation | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| ApiGenerateParityTest | test-e2e | 3 | test_api_generate.cpp | GPU pending | 2.7 | E2E generate API: non-empty output, streaming delivers all tokens, manual prefill loop | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| BatchBuilderTest | test-e2e | 7 | test_continuous_batching.cpp | 0.00C | 5.2 | Continuous batching builder: decode, prefill, mixed, reset, 16-sequence batches | y | n |  |
| BatchInvarianceTest | test-e2e | 4 | test_batch_invariance.cu | GPU pending | 17.8 | E2E batch invariance: NVFP4 and FP8 decode solo vs batched, neighbour independence | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| CancelLifecycleHybridTest | test-e2e | 1 | test_cancel_lifecycle.cpp | GPU pending | 5.0 | E2E hybrid model: cancelling a queued request leaves the running sequence state intact | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| CancelLifecycleTest | test-e2e | 2 | test_cancel_lifecycle.cpp | GPU pending | 5.0 | E2E cancel during a graph burst: waits it out, releases every table, no token after cancel | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| CandidateTokenGuardTest | test-e2e | 8 | test_candidate_tokens.cpp | GPU pending | 3.2 | Candidate-token guard: single-token letters; refuses merged, duplicate, multi-token | y | y: TOKENIZER_QWEN3 |  |
| ChunkedPrefillTest | test-e2e | 7 | test_chunked_prefill.cu | GPU pending | 8.0 | E2E chunked prefill logits equal unchunked: Qwen3-4B, Llama-3.2-3B, long context | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| DFlashDrafterTest | test-e2e | 1 | test_dspark_drafter.cu | GPU pending | 9.0 | DFlash drafter block output vs reference (e2e, needs model) | y | y: MODEL_DFLASH MODEL_DSPARK MODEL_DSPARK_TARGET |  |
| DSparkDrafterTest | test-e2e | 4 | test_dspark_drafter.cu | GPU pending | 9.0 | DSpark drafter: block vs reference, context rollback, FP8 weights, graph replay length | y | y: MODEL_DFLASH MODEL_DSPARK MODEL_DSPARK_TARGET |  |
| DecodeGraphCtxGrowthTest | test-e2e | 1 | test_e2e_models.cpp | GPU pending | 3.4 | E2E decode graph: long prompt after a short request stays alive (context growth) | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 |  |
| DeepStackInject | test-e2e | 5 | test_deepstack_inject.cu | GPU pending | 6.7 | DeepStack injection adds at image tokens only; repeats accumulate; surplus kept | y | n |  |
| DegenerationTest | test-e2e | 6 | test_degeneration.cpp | GPU pending | 3.1 | E2E degeneration: repetition, greedy determinism, graph replay vs eager, no special tokens | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| DetEvalE2ETest | test-e2e | 10 | test_determinism_e2e.cpp | GPU pending | 3.1 | E2E determinism: greedy and perplexity bit-identical in one context; fresh cases off | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| DeviceFaultSignalTest | test-e2e | 1 | test_batching_engine_fault.cu | GPU pending | 13.4 | TIMING: Fault inside a batching step cancels requests and refuses later ones | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| EncoderEmbedTest | test-e2e | 1 | test_encoder_embed.cpp | GPU pending | 3.9 | NomicBert encoder embeddings match reference structure (e2e, GGUF) | y | n |  |
| EndToEndModelTest | test-e2e | 8 | test_e2e.cpp | GPU pending | 6.4 | E2E model: load, tokenize, generate, prefill/decode, 8k/16k context, isolation, think-stop | y | y: MODEL MODEL_GDN |  |
| EndToEndTest | test-e2e | 7 | test_e2e.cpp | 0.00C | 6.4 | C API smoke: version, config and params defaults, error strings, null arguments | y | n |  |
| EngineIntegrationTest | test-e2e | 12 | test_engine_integration.cu | GPU pending | 16.7 | Engine integration: init, prefill/decode steps, FP8/NVFP4 paths, pipelined batched decode | y | n |  |
| EngineRelaunchTest | test-e2e | 2 | test_engine_relaunch.cpp | GPU pending | 3.2 | Model reload frees VRAM; second engine on the same model handle runs clean | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| FimRealTokenizer | test-e2e | 2 | test_fim_tokenizer.cpp | GPU pending | 2.7 | FIM tokenizer: Qwen-Coder ids and PSM prompt; non-FIM tokenizer refused | y | y: FIM_TOKENIZER NOFIM_TOKENIZER |  |
| ForwardPassTest | test-e2e | 8 | test_forward_pass.cu | GPU pending | 18.9 | Forward-pass logits on a synthetic model: prefill, decode, GQA, batch composition | y | n |  |
| GDNModelTest | test-e2e | 2 | test_e2e_models.cpp | GPU pending | 3.4 | GDN model e2e: coherent output and multi-turn GDN state across turns | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 |  |
| GPUBatchTest | test-e2e | 6 | test_continuous_batching.cpp | GPU pending | 5.2 | GPU batch upload and free: sequence uploads, double free, empty batch, varying contexts | y | n |  |
| GdnGraphBucketTest | test-e2e | 2 | test_gdn_graph_buckets.cu | GPU pending | 14.3 | GDN graph buckets: each batch bucket uses its own recurrent slot; recycled slot intact | y | y: MODEL_GDN |  |
| Gemma4AudioEncoderTest | test-e2e | 2 | test_gemma4_audio_encoder.cu | GPU pending | 9.3 | Gemma-4 audio encoder: WAV front end and tower/embedder match HF features | y | y: MODEL_GEMMA4_AUDIO |  |
| Gemma4E4BRef | test-e2e | 2 | test_e2e_gemma4_e4b.cpp | GPU pending | 2.4 | Gemma-4 E4B greedy 20-token text matches HF reference (capital, sky) | y | y: MODEL_GEMMA4_E4B |  |
| Gemma4GraphsTest | test-e2e | 2 | test_e2e_models.cpp | GPU pending | 3.4 | Gemma-4 e2e with CUDA graphs: capital answer and 256-token coherent decode | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 |  |
| Gemma4ModelTest | test-e2e | 4 | test_e2e_models.cpp | GPU pending | 3.4 | Gemma-4 e2e: chat answer, raw completion, SWA prefill route, no repetition | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 |  |
| Gemma4SafeTensorsTest | test-e2e | 1 | test_e2e_models.cpp | GPU pending | 3.4 | Gemma-4 SafeTensors e2e: needle beyond the sliding window is retrieved | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 |  |
| GgufDeadWeightForms | test-e2e | 2 | test_gguf_dead_weight_forms.cpp | GPU pending | 6.0 | GGUF dead weight: dense GGUF carries no scale atom; NVFP4 served head frees its source | y | n |  |
| GreedyLockTest | test-e2e | 3 | test_e2e_greedy_lock.cpp | GPU pending | 3.0 | E2E greedy lock: frozen sequences; teacher-forced decode feeds and scores forced tokens | y | y: MODEL |  |
| HybridBatchedDecodeTest | test-e2e | 3 | test_hybrid_batched_decode.cu | GPU pending | 15.2 | Hybrid batched decode: row independence, batched vs solo logit delta, think-close repeats | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| HybridRestoreChainTest | test-e2e | 8 | test_hybrid_restore_chain.cu | GPU pending | 17.3 | Hybrid snapshot restore chain: restored state stays close to a fresh run (e2e) | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| HybridSharedScoreTest | test-e2e | 1 | test_hybrid_shared_score.cu | GPU pending | 15.1 | Hybrid shared-score snapshots: rows restoring one snapshot are independent across waves | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| InternVLEncoder | test-e2e | 2 | test_internvl_encoder.cu | GPU pending | 10.7 | InternVL vision encoder per-stage and full tower output vs HF FP32 reference | y | y: MODEL_INTERNVL |  |
| KimiTiktokenParityTest | test-e2e | 4 | test_tokenizer_kimi.cpp | GPU pending | 2.7 | Kimi tiktoken tokenizer: fingerprint, encode equals HF on all cases, round trip, vocab | y | y: MODEL_KIMI |  |
| Ling3Ref | test-e2e | 2 | test_e2e_ling3.cpp | GPU pending | 2.3 | Ling-3 e2e: greedy capital answer and chunked prefill match HF reference | y | y: MODEL_LING3 |  |
| LlmCompressorE2E | test-e2e | 4 | test_e2e_llm_compressor.cpp | GPU pending | 3.1 | LLM-compressor/ModelOpt checkpoints (Gemma-4, Mistral Small, Qwen-Coder): load, generate | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| LoraHotSwap | test-e2e | 3 | test_lora.cpp | GPU pending | 5.0 | TIMING: LoRA hot swap: zero adapter is identity, shapes vs model, prefix-cache key | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| Mamba2BatchedDecodeTest | test-e2e | 1 | test_hybrid_batched_decode.cu | GPU pending | 15.2 | Mamba2 batched decode: rows decode independently of their order (hybrid e2e) | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| MoePrefixHitE2ETest | test-e2e | 2 | test_prefix_cache_e2e.cpp | GPU pending | 7.2 | MoE prefix-cache hit at two offsets is bit-equal to a fresh run (e2e) | y | y: MODEL MODEL_GDN |  |
| MtpGreedyIdentityTest | test-e2e | 6 | test_mtp_greedy_identity.cpp | GPU pending | 6.0 | Speculative decoding (MTP, n-gram, DSpark, DFlash) keeps constrained greedy tokens | y | y: MODEL_DFLASH MODEL_DSPARK MODEL_DSPARK_TARGET MODEL_MTP |  |
| MtpProbe | test-e2e | 1 | test_real_checkpoints.cpp | GPU pending | 2.3 | MTP head is found in a real checkpoint (env-gated probe) | y | y: IMP_MTP_MODEL |  |
| MtpQwen4ExpReference | test-e2e | 1 | test_mtp_qwen4exp_reference.cpp | GPU pending | 7.8 | MTP two draft steps match vLLM math on bf16 rows (reference test) | y | y: MODEL_MTP_QWEN4EXP |  |
| Nvfp4PrefillCacheKnobTest | test-e2e | 1 | test_nvfp4_prefill_cache_knob.cpp | GPU pending | 3.7 | Decode knob keeps the CUTLASS NVFP4 prefill cache | y | n |  |
| PrefixCacheE2ETest | test-e2e | 8 | test_prefix_cache_e2e.cpp | GPU pending | 7.2 | Prefix cache e2e: hit equals fresh run; hybrid snapshot and FP8-KV restores | y | y: MODEL MODEL_GDN |  |
| PrimaryModelTest | test-e2e | 4 | test_e2e_models.cpp | GPU pending | 3.4 | Primary model e2e: coherent generation, multi-turn, tokenize roundtrip, prefill/decode | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 |  |
| PromptLogprobsE2ETest | test-e2e | 1 | test_prompt_logprobs_e2e.cpp | GPU pending | 4.2 | Prompt logprobs e2e: chunk logits released after the pass; VRAM allocator | y | y: MODEL |  |
| Q8DropSourceE2E | test-e2e | 1 | test_q8_drop_source_e2e.cpp | GPU pending | 5.9 | Q8 weight source dropped after load: greedy decode still runs; VRAM checked | y | n |  |
| QuantPipelineTest | test-e2e | 1 | test_quant_pipeline.cpp | GPU pending | 2.1 | Quant pipeline build populates decode caches for coherent decode | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| Qwen38TokenizerParityTest | test-e2e | 4 | test_tokenizer_qwen38.cpp | GPU pending | 2.6 | Qwen3.8 tokenizer: fingerprint, encode equals HF on all cases, round trip, vocab size | y | y: MODEL_QWEN38 |  |
| Qwen3VLEncoder | test-e2e | 2 | test_qwen3vl_encoder.cu | GPU pending | 9.5 | Qwen3-VL vision encoder vs independent CPU reference, including attention chunks | y | n |  |
| Qwen3VLPipelineTest | test-e2e | 6 | test_qwen3vl_pipeline.cu | GPU pending | 10.2 | Qwen3-VL image pipeline: reserved bytes, finite deterministic encodings, HF video frames | y | y: MODEL_QWEN3VL |  |
| Qwen3VLPipelineVideoStamps | test-e2e | 1 | test_qwen3vl_pipeline.cu | GPU pending | 10.2 | Qwen3-VL video timestamp ids match HF | y | y: MODEL_QWEN3VL |  |
| RaggedPrefillTest | test-e2e | 6 | test_prefill_ragged.cu | GPU pending | 18.8 | Ragged prefill: matches serial dense, per-layer KV heads, repeats, chunked, mixed decode | y | n |  |
| RequestTest | test-e2e | 5 | test_continuous_batching.cpp,test_scheduler_prefix_bypass.cpp | 0.00C | 9.6 | Request context_len, status transitions, default state; prefix and ragged admission rules | y | n |  |
| SamplerRowBansTest | test-e2e | 1 | test_sampler_row_bans.cu | GPU pending | 16.1 | Stashed sampler rows keep their own ban list | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| SchedulerGpuTest | test-e2e | 1 | test_scheduler.cpp | GPU pending | 6.6 | GPU scheduler: pool grows under aggregate memory pressure | y | n |  |
| SchedulerTest | test-e2e | 43 | test_scheduler.cpp,test_scheduler_embedding_prefix.cpp,test_scheduler_kv_decode.cpp,test_scheduler_prefix_bypass.cpp | 0.01C | 18.3 | Scheduler: admission gate, memory-aware admission, prefill/decode priority, cancel, aging | y | n |  |
| SentencePieceLoader | test-e2e | 1 | test_real_checkpoints.cpp | GPU pending | 2.3 | Real HF-cached SentencePiece model loads cleanly (env-gated) | y | y: IMP_MTP_MODEL |  |
| ServingSignalsTest | test-e2e | 8 | test_serving_signals.cpp | GPU pending | 6.1 | TIMING: Serving signals e2e: KV exhaustion, valve arming, host swap, queue time | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| SpecCaptureFidelityTest | test-e2e | 2 | test_spec_capture_fidelity.cpp | GPU pending | 4.2 | Cached-graph speculation forward equals eager; hybrid speculation does not collapse | y | y: MODEL_SPEC_FIDELITY MODEL_SSM |  |
| StoragePlanner | test-e2e | 14 | test_weight_registry_preservation.cpp | 0.00C | 9.7 | Weight storage planner: tier plans under budget, NVFP4 tiers, dual-path hints, VRAM sums | y | n |  |
| StubModelTest | test-e2e | 5 | test_e2e.cpp | 0.00C | 6.4 | Stub GGUF model e2e: load, tokenize, create context and infer, VRAM leak detection | y | n |  |
| SuspendResumeTest | test-e2e | 3 | test_suspend_resume.cu | GPU pending | 9.1 | Suspend/resume: snapshot round trip is token-identical; armed mismatch is ignored | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| TensorKindCoverage | test-e2e | 1 | test_tensor_kind_coverage.cpp | GPU pending | 2.7 | Small Qwen GGUF: no unknown tensor kinds | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| TokenizerCompatTest | test-e2e | 1 | test_tokenizer_compat.cpp | GPU pending | 2.9 | Tokenizer matches HF on the loaded GGUF and SafeTensors model | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| VideoEvsKernels | test-e2e | 2 | test_video_evs_kernels.cu | GPU pending | 6.9 | Video EVS kernels: dissimilarity matches host cosine; gather compacts kept rows in order | y | n |  |
| VisionChunkOffset | test-e2e | 5 | test_vision_chunk_offset.cu | GPU pending | 7.0 | Vision embeddings chunk offset: second chunk, DeepStack offset, chunked equals unchunked | y | n |  |
| VisionGolden | test-e2e | 2 | test_vision_golden.cu | GPU pending | 7.3 | Vision encoder golden outputs for Gemma3 SigLIP and Gemma4v | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| VramBudgetReserve | test-e2e | 21 | test_vram_budget_reserve.cpp | GPU pending | 10.1 | VRAM budget planner: KV pool, weight-cache room, reserve floor, IMMA planes, budget | y | n |  |
| VramLedgerE2E | test-e2e | 1 | test_vram_ledger_e2e.cpp | GPU pending | 5.8 | E2E VRAM ledger: MXFP4 GDN native holds no FP16 cache | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| WarmCacheTest | test-e2e | 3 | test_warm_cache.cu | GPU pending | 8.1 | Warm weight cache: cold load writes it, warm boot token-identical; corrupt cache ignored | y | y?: test_models.h / inkling_fixture.h (file level) |  |
| WeightRegistryPreservation | test-e2e | 1 | test_weight_registry_preservation.cpp | 0.00C | 9.7 | Weight registry: NVFP4 mode keeps SSM in/out tensors; tier plans and VRAM sums | y | n |  |
| HfFetchNetwork | test-hf-network | 3 | test_hf_fetch_network.cpp | no lane | 2.6 | HF fetch (network): GGUF fetch verifies SHA then hits cache; gated repo fails cleanly | n | n |  |
| CaptureAbort | test-kv | 3 | test_capture_abort.cu | GPU pending | 7.9 | Graph capture abort: throw leaves stream usable; first replay failure still runs step | y | n |  |
| DeviceAllocAllGpu | test-kv | 3 | test_device_alloc_all.cu | GPU pending | 8.4 | Device alloc-all on GPU: async takes all buffers, release returns them, failure is clean | y | n |  |
| DeviceOwnedGpu | test-kv | 3 | test_device_alloc_all.cu | GPU pending | 8.4 | DeviceOwned ledger: tracked bytes per allocator family (plain, async, allocator) | y | n |  |
| FP8KVCache | test-kv | 9 | test_fp8_kv_cache.cu | GPU pending | 10.8 | FP8 KV cache: construction, scale calibration, block bytes, split-K and HD256 vs FP16 | y | n |  |
| Fp8KvStaticScale | test-kv | 6 | test_fp8_kv_static_scale.cu | GPU pending | 10.7 | FP8 KV static scale: E4M3 bound, power-of-two grid, in-place round trip, split scales | y | n |  |
| INT8KVCache | test-kv | 2 | test_fp8_kv_cache.cu | GPU pending | 10.8 | INT8 KV cache: construction and split-K consistency | y | n |  |
| KVBlockCopy | test-kv | 4 | test_kv_block_copy.cu | GPU pending | 8.0 | KV block copy: FP16/FP8 fan-out, key min/max metadata copied, untouched block intact | y | n |  |
| KVCacheGrowAdmissionTest | test-kv | 2 | test_kv_cache_grow_admission_gpu.cpp | GPU pending | 3.5 | TIMING: Growable KV pool: capped reclaim without per-block VRAM read; grow-ahead | y | n |  |
| KVCacheGrowTest | test-kv | 7 | test_kv_cache_gpu.cpp | GPU pending | 6.9 | TIMING: Growable KV pool: growth stops at headroom, ceiling raise, grow-ahead, scale plane | y | n |  |
| KVCacheIdleReleaseTest | test-kv | 2 | test_kv_cache_grow_admission_gpu.cpp | GPU pending | 3.5 | KV idle release: free tail goes back, cached prefix and referenced tail stay | y | n |  |
| KVCacheManagerGrowTest | test-kv | 1 | test_kv_cache_gpu.cpp | GPU pending | 6.9 | KV manager grows before reclaiming cached blocks | y | n |  |
| KVCacheManagerTest | test-kv | 7 | test_kv_cache_gpu.cpp | GPU pending | 6.9 | KV manager: fingerprint gate, reclaim count, SWA rollback, snapshot pack/restore, clone | y | n |  |
| KVCacheTest | test-kv | 10 | test_kv_cache_gpu.cpp | GPU pending | 6.9 | KV cache pointers, read/write, NVFP4 layout, SWA block pointers, residency probe, swap | y | n |  |
| KVCacheWriteTest | test-kv | 4 | test_kv_cache_write.cu | GPU pending | 10.1 | KV cache write: paged write, fused K/V write, batched block table, INT8 per-head scale | y | n |  |
| KVGatherTest | test-kv | 5 | test_kv_gather.cu | GPU pending | 9.8 | KV gather paged to flat: FP16, FP8 dequant, INT8 roundtrip, sentinel zero fill | y | n |  |
| KVHostSpillGpuTest | test-kv | 27 | test_kv_host_spill_gpu.cu | GPU pending | 11.7 | Host KV spill: reclaimed prefix restores bit-exact incl key min/max; snapshot chain | y | n |  |
| MtpUploadBudgetTest | test-kv | 3 | test_mtp_upload_budget.cpp | GPU pending | 2.4 | MTP upload peak bytes: packed layout equals file size; per-expert slabs; uncounted experts | y | n |  |
| PrefixEquivTest | test-kv | 13 | test_prefix_cache_equiv.cpp | GPU pending | 4.0 | Prefix cache equivalence: reuse counts, shared blocks, eviction, chain holes, salt, probe | y | n |  |
| PrefixPersistTest | test-kv | 1 | test_prefix_cache_equiv.cpp | GPU pending | 4.0 | Prefix cache persistence: quantized KV restores its scales | y | n |  |
| RecurrentSnapshotStoreTest | test-kv | 13 | test_recurrent_snapshot_store.cpp | GPU pending | 5.3 | TIMING: Recurrent snapshot store: host-tier eviction and restore, LRU, protected entries | y | n |  |
| SSMStateLazyGpu | test-kv | 2 | test_kv_cache_gpu.cpp | GPU pending | 6.9 | SSM state lazy GPU: slot commit backs one stride; decommitted slots return to the driver | y | n |  |
| TouchAuditGpu | test-kv | 1 | test_touch_audit.cu | GPU pending | 8.0 | Touch audit: pages written since the snapshot counted exactly | y | n |  |
| VmmBackendTest | test-kv | 8 | test_vmm_backend.cu | GPU pending | 8.8 | VMM backend: reservation costs no memory, growth into ceiling, zeroed blocks, fixed pools | y | n |  |
| VramFits | test-kv | 1 | test_vram_query.cpp | GPU pending | 2.8 | VRAM fit check: size plus margin compared with free plus credit | y | n |  |
| VramLedger | test-kv | 2 | test_vram_query.cpp | GPU pending | 2.8 | VRAM ledger: plain allocation moves live bytes and free; unknown-pointer free is a no-op | y | n |  |
| VramQuerySnapshot | test-kv | 2 | test_vram_query.cpp | GPU pending | 2.8 | VRAM query snapshot: free at install capped at budget; pending reservations not counted | y | n |  |
| VramQueryTest | test-kv | 4 | test_vram_query.cpp | GPU pending | 2.8 | VRAM budget query: uncapped passthrough, total capped by budget, exhaustion clamps to zero | y | n |  |
| ActivationCalibratorExperts | test-moe-gdn | 1 | test_activation_calibrator_experts.cu | GPU pending | 7.1 | Activation calibrator: one entry per routed expert segment | y | n |  |
| ExpertCacheLayerMask | test-moe-gdn | 1 | test_expert_cache.cu | GPU pending | 7.8 | Expert cache layer mask: only cached layers get rows; setup and teardown | y | n |  |
| ExpertCachePhase3Test | test-moe-gdn | 11 | test_expert_cache.cu | GPU pending | 7.8 | Expert cache LRU: slot mirror, same-layer eviction order, per-layer isolation, hit moves | y | n |  |
| ExpertCacheReinitTest | test-moe-gdn | 3 | test_expert_cache.cu | GPU pending | 7.8 | Expert cache reinit: grown budget takes; failure falls back; both fail disables | y | n |  |
| GDNScanTest | test-moe-gdn | 14 | test_gdn.cu | GPU pending | 8.9 | GDN scan CPU vs GPU: single/multi token, fused vs legacy, chunk handoff, zero state | y | n |  |
| GdnBatchedScanTest | test-moe-gdn | 12 | test_gdn_batched.cu | GPU pending | 9.9 | Batched GDN scan bit-identical at batch 1, 8, 32, non-contiguous slots; conv1d decode | y | n |  |
| GdnRaggedRidersTest | test-moe-gdn | 1 | test_gdn_ragged_riders.cu | GPU pending | 9.4 | GDN ragged riders: decode riders and prompts match separate solo runs | y | n |  |
| GdnVerifySnapshotTest | test-moe-gdn | 7 | test_gdn_verify_snapshot.cu | GPU pending | 8.1 | GDN verify snapshot writes: fused and chunkwise F32/BF16; second slab one row later | y | n |  |
| GdnVerifyTwoDraftsTest | test-moe-gdn | 12 | test_gdn_verify_two_drafts.cu | GPU pending | 10.2 | GDN verify two-draft scan and conv factor rows fold draft states | y | n |  |
| GptOssMoeFusedBias | test-moe-gdn | 4 | test_gpt_oss_moe_fused_bias.cu | GPU pending | 8.5 | gpt-oss MoE fused bias: GLU quantize and scatter-down bias byte-identical to launches | y | n |  |
| GrammarConstrainMaskTest | test-moe-gdn | 1 | test_grammar_regex_mask.cu | GPU pending | 7.2 | Grammar regex mask: model vocab larger than tokenizer masks padding | y | n |  |
| JsonConstrainFsmTest | test-moe-gdn | 5 | test_json_constrain.cu | GPU pending | 10.2 | JSON constrained FSM: nested docs accepted; mismatched closers, trailing content rejected | y | n |  |
| JsonConstrainTest | test-moe-gdn | 12 | test_json_constrain.cu | GPU pending | 10.2 | JSON constrain: token classes, valid-token mask, control chars masked, allow-list init | y | n |  |
| KdaScan | test-moe-gdn | 7 | test_kda.cu | GPU pending | 9.3 | KDA scan (Kimi-Linear): recurrent and chunk vs reference, softplus gate, ragged, bf16 | y | n |  |
| MoEDeterministicPermute | test-moe-gdn | 1 | test_moe.cu | GPU pending | 9.9 | MoE deterministic permute: expert layout matches the serial rule | y | n |  |
| MoEExecutorTest | test-moe-gdn | 12 | test_moe_executor.cu | GPU pending | 21.9 | MoE executor e2e: init, valid forward, deterministic, bounded logit drift, mixed MoE/dense | y | n |  |
| MoEMultiCtaPermute | test-moe-gdn | 1 | test_moe.cu | GPU pending | 9.9 | MoE multi-CTA permute bit-exact vs the single-CTA kernel | y | n |  |
| MoERoutingEdgeTest | test-moe-gdn | 2 | test_moe.cu | GPU pending | 9.9 | MoE routing edges: empty expert, all tokens routed to one expert | y | n |  |
| MoERoutingTest | test-moe-gdn | 7 | test_moe.cu | GPU pending | 9.9 | MoE routing: top-k, weight normalization, expert offsets, sorted ids, gather/scatter | y | n |  |
| MoERoutingWideTest | test-moe-gdn | 2 | test_moe.cu | GPU pending | 9.9 | MoE routing with more than 256 experts: candidate set and softmax top-k per launch | y | n |  |
| MoeGroupTopk | test-moe-gdn | 1 | test_moe_group_topk.cu | GPU pending | 6.9 | MoE group top-k kernel matches the Ling gate reference | y | n |  |
| MoeImbalanceKernel | test-moe-gdn | 3 | test_moe.cu | GPU pending | 9.9 | MoE imbalance kernel: one-hot expert takes all rows; skewed routing vs host; many experts | y | n |  |
| MtpBodyNvfp4Test | test-moe-gdn | 2 | test_mtp_body_nvfp4.cu | GPU pending | 8.7 | MTP body NVFP4 copy: direct quantize per weight; one-row and multi-row feeds read the copy | y | n |  |
| MtpFeedBatchTest | test-moe-gdn | 11 | test_mtp_feed_batch.cu | GPU pending | 11.0 | MTP feed batch: batched vs per-pair and multi-slot feeds, packed tables, captured head | y | n |  |
| PLEBatched | test-moe-gdn | 3 | test_ple_batched.cu | GPU pending | 7.7 | PLE batched decode: rows match per-sequence bitwise; neighbours do not leak; prefill chunk | y | n |  |
| PreambleGateTest | test-moe-gdn | 29 | test_json_constrain.cu | GPU pending | 10.2 | JSON preamble gate: off by default; budget-only mode; thinking absorb; tool transitions | y | n |  |
| RegexConstrainMaskTest | test-moe-gdn | 1 | test_grammar_regex_mask.cu | GPU pending | 7.2 | Regex constrain mask: model vocab larger than tokenizer masks the padding | y | n |  |
| RegexNfaTest | test-moe-gdn | 4 | test_schema_constrain.cu | GPU pending | 10.6 | Regex NFA: literals, char-class ranges, counted repeats, anchors | y | n |  |
| SSMConv1dTest | test-moe-gdn | 10 | test_ssm.cu | GPU pending | 8.3 | SSM conv1d: decode shift-convolve, vectorised bit-exact, prefill causal, chunked prefill | y | n |  |
| SSMConvSplitTest | test-moe-gdn | 2 | test_ssm_conv_split.cu | GPU pending | 6.8 | SSM conv split: bit-identical to separate kernels; uncovered shapes refused | y | n |  |
| SSMScanSsdTest | test-moe-gdn | 2 | test_ssm_scan_ssd.cu | GPU pending | 7.2 | SSM scan SSD: matches double reference; uncovered shapes refused | y | n |  |
| SSMScanTest | test-moe-gdn | 5 | test_ssm.cu,test_ssm_scan_reg.cu | GPU pending | 16.1 | SSM scan: padded chunk length, prefill snapshot state, register scan vs legacy and double | y | n |  |
| SchemaConstrainTest | test-moe-gdn | 32 | test_schema_constrain.cu,test_tool_call_constrain.cu | GPU pending | 23.0 | JSON schema constrain: patterns, enums, item bounds, refs, recursion; tool calls, XML | y | n |  |
| SsmDecodeBatchedKernelTest | test-moe-gdn | 3 | test_ssm_batched_decode.cu | GPU pending | 7.0 | SSM batched decode: conv and scan rows match single-sequence on their slot (F16/F32) | y | n |  |
| BatchedSmallM | test-quant | 3 | test_nvfp4_batched_smallm_equiv.cu | GPU pending | 9.8 | NVFP4 batched small-M GEMM: rows match the single-row call; matches dequantised reference | y | n |  |
| DequantAwqGpu | test-quant | 1 | test_dequant_awq_gpu.cu | GPU pending | 7.4 | AWQ GPU dequant kernel bit-equal to host reference | y | n |  |
| DequantGptqGpu | test-quant | 1 | test_dequant_gptq_gpu.cu | GPU pending | 7.5 | GPTQ GPU dequant kernel bit-equal to host reference | y | n |  |
| DualPathQuant | test-quant | 2 | test_quant_integration.cu | GPU pending | 26.0 | Dual-path quant: attention weights excluded from NVFP4; weight-cache flag | y | n |  |
| FmhaVLoadBench | test-quant | 4 | test_fmha_v_load_bench.cu | GPU pending | 6.0 | BENCH: cp.async vs TMA bulk V-load attention timing (times kernels, prints numbers) | y | n |  |
| Fp4PvBenchTest | test-quant | 4 | test_fp4_pv_bench.cu | GPU pending | 7.6 | BENCH: FP4 PV attention accuracy and blockscale vs HMMA throughput (times kernels) | y | n |  |
| Fp8ActRowsTest | test-quant | 3 | test_fp8_act_rows.cu | GPU pending | 8.6 | FP8 activation row quantize matches host rule; row bits independent of neighbours | y | n |  |
| Fp8GemmSmallMBench | test-quant | 1 | test_fp8_gemm_smallm.cu | GPU pending | 10.1 | BENCH: FP8 small-M GEMM vs cuBLASLt (DISABLED, times kernels, prints numbers) | y | n |  |
| Fp8GemmSmallMTest | test-quant | 6 | test_fp8_gemm_smallm.cu | GPU pending | 10.1 | FP8 small-M GEMM vs fp64 per M, beta, row-bit invariance, cuBLASLt match, refusals | y | n |  |
| Fp8GemmTileBench | test-quant | 1 | test_gemm_fp8_tile.cu | GPU pending | 10.7 | BENCH: FP8 tile GEMM vs cuBLASLt (DISABLED, times kernels, prints numbers) | y | n |  |
| Fp8GemmTileRefusal | test-quant | 1 | test_gemm_fp8_tile.cu | GPU pending | 10.7 | FP8 tile GEMM refuses illegal shapes and leaves the output untouched | y | n |  |
| Fp8GemmTileTest | test-quant | 14 | test_gemm_fp8_tile.cu | GPU pending | 10.7 | FP8 tile GEMM vs cuBLASLt; beta=1 accumulates | y | n |  |
| Fp8GemvFold | test-quant | 2 | test_fp8_gemv_fold.cu | GPU pending | 9.4 | FP8 GEMV with folded input and residual emit matches gemv_fp8 and host norm | y | n |  |
| GemmNvfp4Batched | test-quant | 5 | test_nvfp4_gemm_batched.cu | GPU pending | 7.9 | NVFP4 batched GEMM vs dequant reference at attn, FFN, down, narrow dims; accumulate | y | n |  |
| GemvKparLoopRepro | test-quant | 2 | test_nvfp4_gemv_kpar_loop.cu | GPU pending | 7.9 | NVFP4 KPAR GEMV: per-row loop equals single call at Gemma-4 gate/up/down dims | y | n |  |
| GgufDequant | test-quant | 14 | test_gguf_dequant_ref.cu | GPU pending | 14.1 | GGUF dequant, all scale modes, vs reference (K=512 coverage) | y | n |  |
| GgufDequantCoverage | test-quant | 1 | test_gguf_dequant_ref.cu | GPU pending | 14.1 | Every supported GGUF format has a dequant reference | y | n |  |
| GgufDequantKQuants | test-quant | 1 | test_gguf_dequant_ref.cu | GPU pending | 14.1 | GGUF K-quant dequant: tail, CTA and unaligned destination vs reference | y | n |  |
| GgufRef | test-quant | 23 | test_gguf_dequant_ref.cu | GPU pending | 14.1 | GGUF GEMV references: Q8_0/Q6_K/Q5_1 dp4a, MoE decode dispatch, MMVQ, dense dp4a rows | y | n |  |
| GptOssMxfp4ConvertRef | test-quant | 6 | test_gpt_oss_mxfp4_convert_ref.cu | GPU pending | 9.7 | gpt-oss MXFP4 convert: down, gate, up slices, extra scale, clamp outlier, zero nibbles | y | n |  |
| GptqNvfp4 | test-quant | 4 | test_gptq_nvfp4.cu | GPU pending | 11.2 | GPTQ-to-NVFP4: beats round-to-nearest on correlated inputs; identity Hessian; Cholesky | y | n |  |
| GroupedNvfp4ArgsK | test-quant | 2 | test_grouped_nvfp4_args.cu | GPU pending | 10.7 | Grouped NVFP4 device args equal host args | y | n |  |
| GroupedNvfp4ArgsTest | test-quant | 1 | test_grouped_nvfp4_args.cu | GPU pending | 10.7 | Grouped NVFP4 host args match per-expert dense GEMM | y | n |  |
| GroupedNvfp4DecodeTest | test-quant | 3 | test_gemm_grouped_nvfp4_decode.cu | GPU pending | 11.1 | Grouped NVFP4 decode: F16 bit-equal to tile and fp64; act epilogue equal to unfused; bench | y | n |  |
| GroupedNvfp4RefManyExperts | test-quant | 2 | test_grouped_nvfp4_ref.cu | GPU pending | 10.3 | Grouped NVFP4 reference with many experts vs fp64 | y | n |  |
| GroupedNvfp4RefSmallK | test-quant | 4 | test_grouped_nvfp4_ref.cu | GPU pending | 10.3 | Grouped NVFP4 reference at small K vs fp64 | y | n |  |
| GroupedNvfp4RefTest | test-quant | 3 | test_grouped_nvfp4_ref.cu | GPU pending | 10.3 | Grouped NVFP4 reference: even, empty-expert boundary, single active expert distributions | y | n |  |
| GroupedNvfp4TileK | test-quant | 6 | test_gemm_grouped_nvfp4_tile.cu | GPU pending | 11.2 | Grouped NVFP4 tile GEMM vs fp64 reference | y | n |  |
| GroupedNvfp4TileTest | test-quant | 1 | test_gemm_grouped_nvfp4_tile.cu | GPU pending | 11.2 | BENCH: grouped NVFP4 tile GEMM wall-time bench (times kernels, prints numbers) | y | n |  |
| LmHeadDraftRows | test-quant | 1 | test_lm_head_draft_rows.cu | GPU pending | 7.2 | NVFP4 multi-row LM-head draft: row-count invariant, cost priced (timed) | y | n |  |
| MmaPeakSaturatedTest | test-quant | 1 | test_mma_peak_saturated.cu | GPU pending | 7.2 | MMA peak-saturated kernel calibration invariants (times kernels) | y | n |  |
| MmaVariantsBench | test-quant | 1 | test_mxf4nvf4_mma_variants_bench.cu | GPU pending | 6.8 | BENCH: mxf4nvf4 MMA variant comparison (times kernels, prints numbers) | y | n |  |
| MmqQ4kImmaBench | test-quant | 1 | test_mmq_q4k_imma_bench.cu | GPU pending | 6.9 | BENCH: Q4_K IMMA MMQ variant timing (times kernels, prints numbers) | y | n |  |
| MmqQ4kImmaGemm | test-quant | 3 | test_mmq_q4k_imma_gemm.cu | GPU pending | 10.6 | Q4_K IMMA GEMM end to end: smallest, multi-tile and FFN-like shapes | y | n |  |
| MmqQ4kImmaTile | test-quant | 5 | test_mmq_q4k_imma_tile_bench.cu | GPU pending | 10.7 | Q4_K IMMA tile correctness: tiny, multi-sub, multi-tile, FFN shapes; BenchSweep timing | y | n |  |
| MmqQ8Imma | test-quant | 51 | test_mmq_imma_gguf_smallm.cu,test_mmq_q8_imma.cu,test_mmq_q8_imma_q4k_moe.cu,test_mmq_q8_imma_q5k.cu,test_mmq_q8_imma_q6k.cu,test_mmq_q8_imma_reuse.cu,test_mmq_q8_imma_tail.cu | GPU pending | 79.8 | Q8 IMMA MMQ: square, tails, small-M split-K and MoE grouped GEMM vs reference | y | n |  |
| Mxf4nvf4MmaBenchTest | test-quant | 1 | test_mxf4nvf4_mma_bench.cu | GPU pending | 6.8 | BENCH: block-scale vs legacy mxf4nvf4 MMA timing (times kernels, prints numbers) | y | n |  |
| Mxf4nvf4ProbeTest | test-quant | 2 | test_mxf4nvf4_probe.cu | GPU pending | 6.7 | mxf4nvf4 block-scale MMA probe: clean launch; zero A zeroes the output | y | n |  |
| Mxf4nvf4QkTest | test-quant | 4 | test_mxf4nvf4_qkt_validate.cu | GPU pending | 7.9 | mxf4nvf4 Q*K^T validation: uniform, row/col indicator, random E2M1 regression | y | n |  |
| Mxfp4GemmTile | test-quant | 1 | test_gemm_mxfp4_tile.cu | GPU pending | 9.9 | MXFP4 tile GEMM refuses a partial atom K | y | n |  |
| Mxfp4GemmTileTest | test-quant | 6 | test_gemm_mxfp4_tile.cu | GPU pending | 9.9 | MXFP4 tile GEMM vs decoded reference over dims | y | n |  |
| Mxfp4Unpacked | test-quant | 2 | test_gemm_mxfp4_tile.cu | GPU pending | 9.9 | MXFP4 unpacked dequant and embedding lookup match host decode | y | n |  |
| Mxfp8Gemm | test-quant | 6 | test_mxfp8_gemm.cu | GPU pending | 8.2 | MXFP8 GEMM vs fp32 reference within E4M3 budget; partial scale tile; refusals; dequant | y | n |  |
| Mxfp8GemmTileBench | test-quant | 1 | test_gemm_mxfp8_tile.cu | GPU pending | 9.0 | BENCH: MXFP8 tile GEMM variants timing (times kernels, prints numbers) | y | n |  |
| Mxfp8GemmTileTest | test-quant | 8 | test_gemm_mxfp8_tile.cu | GPU pending | 9.0 | MXFP8 tile GEMM vs host reference | y | n |  |
| NVFP4OutlierTest | test-quant | 2 | test_nvfp4_outlier_ref.cu | GPU pending | 8.1 | NVFP4 outlier quant-dequant GEMV vs independent reference; extreme outlier floor finite | y | n |  |
| NVFP4QuantTest | test-quant | 2 | test_nvfp4_quant.cu | GPU pending | 6.8 | NVFP4 quant roundtrip and basic GEMV | y | n |  |
| NvFP4CompressedTensorsRef | test-quant | 4 | test_nvfp4_compressed_tensors_ref.cu | GPU pending | 9.5 | NVFP4 compressed-tensors ref: unity scales, two-level scaling, zero tensor scale, sign | y | n |  |
| NvFP4ExportRoundTrip | test-quant | 2 | test_quantize_nvfp4_roundtrip.cu | GPU pending | 8.7 | NVFP4 export roundtrip: tiny model within grid bound; offset-folded norm keeps gain | y | n |  |
| NvFP4GemmGraphCapture | test-quant | 5 | test_nvfp4_gemm_graph_capture.cu | GPU pending | 9.3 | NVFP4 GEMM graph capture: workspace preferred over lazy alloc; deferred workspace replays | y | n |  |
| NvFP4GemvGdnInput | test-quant | 2 | test_nvfp4_gemv_gdn_input.cu | GPU pending | 8.6 | NVFP4 GEMV GDN input: four launches equal fused at Qwen3.8 dims; refuses unservable shapes | y | n |  |
| NvFP4GemvGemma3Dims | test-quant | 3 | test_nvfp4_gemv_gemma3_dims.cu | GPU pending | 8.1 | NVFP4 GEMV at Gemma-3 dims: KPAR per projection, fused QKV, gate/up, GeGLU down | y | n |  |
| NvFP4MergedScaleSpread | test-quant | 1 | test_nvfp4_merged_scale_spread.cu | GPU pending | 8.2 | NVFP4 merged sibling projections each keep their own global scale | y | n |  |
| NvFP4NormFold | test-quant | 2 | test_nvfp4_norm_fold.cu | GPU pending | 9.0 | NVFP4 norm fold: RMSNorm then GEMV matches the folded GEMV, with and without weight offset | y | n |  |
| NvFP4PromoteWeightScale2 | test-quant | 9 | test_nvfp4_compressed_tensors_ref.cu | GPU pending | 9.5 | NVFP4 weight_scale_2 promotion: NaN, inf and zero handling for ModelOpt and LLM-compressor | y | n |  |
| NvFP4SmallMTest | test-quant | 4 | test_nvfp4_smallm.cu | GPU pending | 8.6 | NVFP4 small-M GEMM vs host reference; A4 variant; bandwidth floor checks (timed) | y | n |  |
| NvFP4SmallMV2Test | test-quant | 12 | test_nvfp4_smallm_v2.cu | GPU pending | 10.8 | NVFP4 small-M v2 GEMM: reference, striped tiles, accumulate, bandwidth survey, ring sweep | y | n |  |
| NvFP4ValidatePackedScaleShapes | test-quant | 7 | test_nvfp4_compressed_tensors_ref.cu | GPU pending | 9.5 | NVFP4 packed-scale shape validator: accepts Qwen3/Gemma-4 shapes, rejects bad groups | y | n |  |
| NvFP4ValidateWeightScaleDtype | test-quant | 5 | test_nvfp4_compressed_tensors_ref.cu | GPU pending | 9.5 | NVFP4 weight-scale dtype validator: accepts FP8 E4M3, rejects uint8, E5M2, F16, none | y | n |  |
| Nvfp4GemmAlphaTest | test-quant | 4 | test_nvfp4_gemm_alpha.cu | GPU pending | 8.2 | NVFP4 GEMM alpha: FP8 E4M3 clamp boundary, alpha applied, Mistral L0 reproducer and layout | y | n |  |
| Nvfp4GemmTile | test-quant | 1 | test_nvfp4_gemm_tile.cu | GPU pending | 9.2 | NVFP4 GEMM tile: data-parallel rows invariant to M | y | n |  |
| Nvfp4GemmTileBench | test-quant | 1 | test_nvfp4_gemm_tile.cu | GPU pending | 9.2 | BENCH: NVFP4 GEMM tile vs cuBLASLt wall-time bench (times kernels, prints numbers) | y | n |  |
| Nvfp4GemmTileF32 | test-quant | 1 | test_nvfp4_gemm_tile.cu | GPU pending | 9.2 | NVFP4 GEMM tile with FP32 output vs reference | y | n |  |
| Nvfp4GemmTileTest | test-quant | 8 | test_nvfp4_gemm_tile.cu | GPU pending | 9.2 | NVFP4 GEMM tile vs reference over shape and variant params | y | n |  |
| Nvfp4QuantHwTest | test-quant | 4 | test_nvfp4_quant_hw.cu | GPU pending | 7.6 | NVFP4 quant roundtrip on hardware for head_dim 64 and 128 blocks | y | n |  |
| Nvfp4QuantRefTest | test-quant | 2 | test_nvfp4_quant_ref.cu | GPU pending | 8.2 | NVFP4 quant reference: gaussian roundtrip; representable values bit-exact | y | n |  |
| Nvfp4SmallMWsBench | test-quant | 1 | test_nvfp4_smallm_ws.cu | GPU pending | 9.4 | BENCH: NVFP4 small-M workspace GEMM vs small-M v2 (DISABLED, prints timings) | y | n |  |
| Nvfp4SmallMWsTest | test-quant | 6 | test_nvfp4_smallm_ws.cu | GPU pending | 9.4 | NVFP4 small-M workspace GEMM vs double reference at every M; row bits; refusals | y | n |  |
| Nvfp4StaticActScaleTest | test-quant | 3 | test_nvfp4_static_act_scale.cu | GPU pending | 8.4 | NVFP4 static activation scale: small-M and prompt tile vs double reference; producers | y | n |  |
| Nvfp4TinyBlockScaleTest | test-quant | 2 | test_nvfp4_static_act_scale.cu | GPU pending | 8.4 | NVFP4 tiny block-scale: SF atom bytes equal plain quantize; prompt tile vs reference | y | n |  |
| PrefillRowInvariance | test-quant | 3 | test_prefill_row_invariance.cu | GPU pending | 8.5 | Prefill row invariance: router logits, MoE act quant, HD512 attention vs batch and chunk | y | n |  |
| Q8PlanesOnly | test-quant | 3 | test_q8_planes_only.cu | GPU pending | 11.3 | Q8 IMMA planes: planes-to-F16 matches dequant; build declines; fallback from planes | y | n |  |
| QuantIntegrationTest | test-quant | 24 | test_quant_integration.cu | GPU pending | 26.0 | Quant integration: Q4_0, Q8_0, Q4_K, Q5_K upload, forward pass, determinism, multi-layer | y | n |  |
| QuantTest | test-quant | 2 | test_quant.cu | GPU pending | 7.7 | Quant config defaults and INT4 GEMM basic vs reference | y | n |  |
| QuantizeMoeNative | test-quant | 10 | test_quantize_fp16_nvfp4_moe_native.cu | GPU pending | 7.3 | MoE native NVFP4 quantize: device M offsets, compact alpha and SFA offsets, empty experts | y | n |  |
| SmallMV2Pair | test-quant | 3 | test_nvfp4_batched_smallm_equiv.cu | GPU pending | 9.8 | NVFP4 small-M v2 pair bit-exact to two single calls; M1 bench disabled; striped refusals | y | n |  |
| TmaBlockScaleBench | test-quant | 1 | test_tma_block_scale_bench.cu | GPU pending | 6.6 | BENCH: TMA vs block-scale descriptor launch timing (times kernels, prints numbers) | y | n |  |
| WeightDispatchTest | test-quant | 13 | test_weight_dispatch.cu | GPU pending | 13.1 | Weight dispatch: FP16, FP8, NVFP4, CUTLASS, MXFP4 each match direct kernel; refusals | y | n |  |
| ChatTemplateApplyTest | test-text | 20 | test_chat_template.cpp | 0.00C | 9.2 | Chat template apply: ChatML, Llama 2/3, Nemotron, Gemma, DeepSeek-R1, Phi structure | n | n |  |
| ChatTemplateArchDefaultTest | test-text | 1 | test_chat_template.cpp | 0.00C | 9.2 | Chat template default family for each architecture | n | n |  |
| ChatTemplateContinue | test-text | 6 | test_chat_template_continue.cpp | 0.00C | 3.6 | Chat template continue: Qwen3 trimmed and verbatim content match HF; cut rule | n | n |  |
| ChatTemplateDefaultSysTest | test-text | 5 | test_chat_template.cpp | 0.00C | 9.2 | Chat template default system prompt: Jinja extraction; injected only without user system | n | n |  |
| ChatTemplateDetectTest | test-text | 14 | test_chat_template.cpp | 0.00C | 9.2 | Chat template family detection from template text (ChatML, Llama, Gemma, Mistral, Phi) | n | n |  |
| ChatTemplateFamilyNameTest | test-text | 1 | test_chat_template.cpp | 0.00C | 9.2 | Chat template family names for every family | n | n |  |
| ChatTemplateGolden | test-text | 13 | test_chat_template_goldens.cpp | 0.00C | 4.2 | Chat template goldens: per-family rendered prompts vs committed golden; generation prompt | n | n |  |
| ChatTemplateInitTest | test-text | 10 | test_chat_template.cpp | 0.00C | 9.2 | Chat template init: per-family success, XML tool dialect detection, missing-token fallback | n | n |  |
| ChatTemplateParseFamilyTest | test-text | 1 | test_chat_template.cpp | 0.00C | 9.2 | Chat template parse_family: every name maps to its family | n | n |  |
| ChatTemplateTokenize | test-text | 1 | test_chat_template.cpp | 0.00C | 9.2 | Chat template tokenize: control-token strip flags swallow template whitespace | n | n |  |
| ChatTemplateTools | test-text | 1 | test_chat_template.cpp | 0.00C | 9.2 | Chat template tools: parameter past int64 max renders instead of throwing | n | n |  |
| ChatTemplateVisionTest | test-text | 6 | test_chat_template.cpp | 0.00C | 9.2 | Chat template vision: Gemma image tokens per turn; text-only fallback for other families | n | n |  |
| Cl100kPreTokenizeTest | test-text | 1 | test_tokenizer.cpp | 0.00C | 8.2 | cl100k pre-tokenizer: digit triples and case-blind standalone contractions | n | n |  |
| DigitTriplesRightSplitTest | test-text | 1 | test_tokenizer_llama_bpe.cpp | 0.00C | 2.7 | Llama-BPE digit triples: right-aligned split only before a word boundary | n | n |  |
| HFChatTemplateTest | test-text | 5 | test_chat_template.cpp | 0.00C | 9.2 | HF tokenizer_config chat_template: string, array default/fallback/empty, missing field | n | n |  |
| HarmonyGoldenTest | test-text | 4 | test_gpt_oss_harmony_golden.cpp | 0.00C | 6.0 | gpt-oss Harmony golden: user-only, system+user, multi-turn final channel, channel markers | n | n |  |
| InklingChatTemplate | test-text | 4 | test_inkling_chat_template.cpp | 0.00C | 3.5 | Inkling chat template: reasoning levels advertised, named and numeric levels, refusals | n | n |  |
| InklingDraftPolicy | test-text | 3 | test_inkling_draft_policy.cpp | 0.00C | 2.0 | Inkling draft policy: rows follow measured chance; out-of-vocab token ends the draft | n | n |  |
| InklingEffortSpec | test-text | 1 | test_inkling_chat_template.cpp | 0.00C | 3.5 | Inkling effort spec: other templates accept any value | n | n |  |
| JinjaChatTest | test-text | 7 | test_jinja.cpp | 0.00C | 9.5 | Jinja chat templates: ChatML, Gemma, Qwen3 and multi-turn render vs expected text | n | n |  |
| JinjaFilterTest | test-text | 1 | test_jinja.cpp | 0.00C | 9.5 | Jinja selectattr filter | n | n |  |
| JinjaParity | test-text | 8 | test_jinja_parity.cpp | 0.00C | 11.5 | Jinja parity: real chat templates (Qwen3, Nemotron, Qwen3.8, GLM, Flint, Kimi, gpt-oss) | n | n |  |
| JinjaTest | test-text | 65 | test_jinja.cpp | 0.00C | 9.5 | Jinja engine: expressions, control flow, filters, slices, macros, block set, whitespace | n | n |  |
| JinjaUndefinedTest | test-text | 6 | test_jinja_undefined.cpp | 0.00C | 2.2 | Jinja undefined: unset vs false/true variables, is defined, Qwen thinking guard | n | n |  |
| JinjaValueTest | test-text | 6 | test_jinja.cpp | 0.00C | 9.5 | Jinja value types: none, bool, int, string, array, object conversions | n | n |  |
| KimiChatTemplate | test-text | 3 | test_kimi_chat_template.cpp | 0.00C | 2.8 | Kimi chat template render: one turn, multi-turn with system, tool-call history | n | n |  |
| Ling3ChatTemplate | test-text | 3 | test_ling3_chat_template.cpp | 0.00C | 2.9 | Ling-3 chat template: thinking default, thinking off, thinking forced | n | n |  |
| NfcNormalize | test-text | 2 | test_tokenizer.cpp | 0.00C | 8.2 | NFC normalization: Latin combining and Hangul jamo compose; decompose, reorder, recompose | n | n |  |
| O200kPreTokenizeTest | test-text | 8 | test_tokenizer.cpp | 0.00C | 8.2 | o200k pre-tokenizer: code lines, digit triples, URLs, contractions, case runs, symbols | n | n |  |
| PreTokenizeSplitSequenceTest | test-text | 4 | test_tokenizer_pretok_json.cpp | 0.00C | 4.1 | HF-json pre-tokenizer split sequences match HF on DeepSeek-shaped steps; regex refusals | n | n |  |
| PreTokenizeUnicodeTest | test-text | 4 | test_tokenizer.cpp | 0.00C | 8.2 | Unicode pre-tokenize: emoji symbols, whitespace/digits/CJK punctuation, marks, cased runs | n | n |  |
| Qwen2PreTokenizeTest | test-text | 10 | test_tokenizer.cpp | 0.00C | 8.2 | Qwen2 pre-tokenizer: symbol runs, digits, contractions, newline and trailing whitespace | n | n |  |
| Qwen38ReasoningEffort | test-text | 9 | test_qwen38_chat_template.cpp | 0.00C | 3.8 | Qwen3.8 chat template reasoning effort: advertised values, default xhigh, preamble | n | n |  |
| TiktokenLoader | test-text | 5 | test_tokenizer_tiktoken.cpp | 0.00C | 3.1 | tiktoken loader: lowest-rank merges, atomic specials, Kimi tool names, malformed ranks | n | n |  |
| TokenizerAddedTokens | test-text | 3 | test_tokenizer.cpp | 0.00C | 8.2 | Added tokens: non-special atomic, normalized flags, lstrip/rstrip whitespace swallowing | n | n |  |
| TokenizerControlTest | test-text | 4 | test_tokenizer.cpp | 0.00C | 8.2 | SPM control tokens: mark-as-control allocates, idempotent, ignores bad ids, keeps types | n | n |  |
| TokenizerDispatchTest | test-text | 3 | test_tokenizer.cpp | 0.00C | 8.2 | Tokenizer dispatch: SPM default, set_type, max-length input | n | n |  |
| TokenizerEos | test-text | 2 | test_tokenizer_robustness.cpp | 0.00C | 4.0 | EOS: first added EOS replaces placeholder; a named EOS keeps first place | n | n |  |
| TokenizerGPT2Test | test-text | 11 | test_tokenizer.cpp | 0.00C | 8.2 | GPT-2 byte-BPE: encode, merges, decode roundtrip, digits, spaces, UTF-8, stale ranks | n | n |  |
| TokenizerGemma4Test | test-text | 4 | test_tokenizer.cpp | 0.00C | 8.2 | Gemma-4 tokenizer: merge skipped when result not in vocab; truncated UTF-8; byte fallback | n | n |  |
| TokenizerJsonHostileIds | test-text | 4 | test_tokenizer_robustness.cpp | 0.00C | 4.0 | tokenizer.json robustness: negative or huge vocab ids dropped; well-formed vocab loads | n | n |  |
| TokenizerJsonPretok | test-text | 5 | test_tokenizer_pretok_json.cpp | 0.00C | 4.1 | tokenizer.json pre-tokenizer rewrite: digit runs, splits, char and Onig classes vs HF | n | n |  |
| TokenizerLlamaBpeTest | test-text | 2 | test_tokenizer_llama_bpe.cpp | 0.00C | 2.7 | Llama-BPE pre-tokenizer: cl100k chunks for Llama, Nemotron chunks for Tekken | n | n |  |
| TokenizerNormalizer | test-text | 2 | test_tokenizer.cpp | 0.00C | 8.2 | Normalizer: NFC only when tokenizer JSON declares it; Gemma-4 JSON routes to Gemma-4 | n | n |  |
| TokenizerRobustness | test-text | 16 | test_tokenizer_robustness.cpp | 0.00C | 4.0 | Byte tokenizer roundtrip for ASCII, UTF-8, NUL and surrogate bytes; decode ids never crash | n | n |  |
| TokenizerSPMTest | test-text | 16 | test_tokenizer.cpp | 0.00C | 8.2 | SPM tokenizer: vocab load, encode words and subwords, byte fallback, decode, UTF-8 and CJK | n | n |  |
| TranscriptIdStoreTest | test-text | 2 | test_transcript_ids.cpp | 0.00C | 4.0 | Transcript id store: longest prefix wins with FIFO cap; resent reply keeps model split | n | n |  |
