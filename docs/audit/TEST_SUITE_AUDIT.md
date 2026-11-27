<!--
layer: L2
audience: kernel-devs
verified: 2026-11-27
commit: 86e5cf30
-->

# imp - test-suite audit

Owner assignment 2026-11-27: every test justifies its existence, the suite runs faster, regression detection does not drop. Phase 1 (this section) is read-only: no test, build file or CI file changed. Tree: `86e5cf30` (origin/main after #3347). Raw data: `/mnt/data/scratch/tsa/` (`logs/`, `junit/`, `bld/ninja_log_*.txt`, `scripts/`, `tables/`). Prior work reused, not redone: [`SETTLED.md`](SETTLED.md) S-2, S-28; [`MUTATION_BASELINE.md`](MUTATION_BASELINE.md); [`BASELINE_TESTLIST_2026_10_15.txt`](BASELINE_TESTLIST_2026_10_15.txt).

## Phase 1: inventory and baseline

Status 2026-11-27: CPU part done 00:06Z; GPU part done 03:24Z under GPU lock `tsa` (acquired 01:15:59Z). Runner `/mnt/data/scratch/tsa/scripts/gpu_runs3.sh`: 8 binaries + the 3 extra `test-gpu-suite` lines, 3x each, image `imp:test` (tree of `86e5cf30`), `docker run` as in `make test-gpu` (image user, no `-u`), XML to `gpu/`; then `analyze.py` + `suite_table.py` regenerate every table below. Load rule: 1-min load sampled every 5 s, a timing with max > 4 is repeated; peer builds were held off by `/mnt/data/scratch/tsa/QUIET_GPU_BASELINE` from 02:26Z; all 33 kept timings ran at load max <= 3.82, none repeated. An earlier run 1 at load 3.5-27.7 is kept in `gpu/contended/`, not used. `make verify-fast` not timed (TSA-3 scope: binaries).

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
| suites reading a checkpoint env var | 41 sure + 25 via a shared helper (file level; GPU run: 14 skip all cases without a model env, 2 skip some, 9 run on a hard-coded `/models/...` default) | column `weights` below |
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

Wall per `docker run` (container start included), kept attempt per run, `logs/gpu_runs3.log`. The 11 rows are what `make test-gpu` runs: sum of medians 1022.51 s, the 6 GPU binaries 967.34 s.

| run | run 1 s (load max) | run 2 s (load max) | run 3 s (load max) | median s | rc |
|---|---:|---:|---:|---:|---|
| test-core | 6.07 (3.46) | 6.08 (1.73) | 6.06 (1.22) | 6.07 | rc=0 / rc=0 / rc=0 |
| test-text | 0.91 (3.26) | 1.01 (1.59) | 1.04 (1.20) | 1.01 | rc=0 / rc=0 / rc=0 |
| test-compute | 38.56 (3.26) | 38.81 (1.62) | 38.87 (1.20) | 38.81 | rc=0 / rc=0 / rc=0 |
| test-attention | 239.50 (2.25) | 246.18 (1.35) | 241.97 (1.17) | 241.97 | rc=0 / rc=0 / rc=0 |
| test-quant | 168.06 (1.13) | 168.05 (1.23) | 169.06 (1.08) | 168.06 | rc=0 / rc=0 / rc=0 |
| test-kv | 3.93 (1.05) | 3.96 (1.03) | 3.88 (2.42) | 3.93 | rc=0 / rc=0 / rc=0 |
| test-moe-gdn | 9.37 (1.05) | 9.59 (1.10) | 9.48 (2.31) | 9.48 | rc=0 / rc=0 / rc=0 |
| test-e2e | 553.92 (2.46) | 505.09 (1.95) | 499.31 (3.82) | 505.09 | rc=0 / rc=0 / rc=0 |
| deteval | 51.13 (1.80) | 45.51 (1.30) | 40.28 (1.12) | 45.51 | rc=0 / rc=0 / rc=0 |
| qwen38_tok | 1.41 (1.70) | 1.42 (1.24) | 1.39 (1.09) | 1.41 | rc=0 / rc=0 / rc=0 |
| kimi_tok | 1.17 (1.70) | 1.21 (1.24) | 1.16 (1.08) | 1.17 | rc=0 / rc=0 / rc=0 |

gtest wall from the XML (`tables/gpu_binaries.md`); skipped = cases skipped in any run (no model env, `/models` mounted):

| binary | run 1 s | run 2 s | run 3 s | median s | cases | skipped | failed (runs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| test-attention | 238.34 | 245.06 | 240.83 | 240.83 | 214 | 0 | 0 |
| test-compute | 37.29 | 37.61 | 37.64 | 37.61 | 377 | 31 | 0 |
| test-core | 4.86 | 4.96 | 4.87 | 4.87 | 2513 | 1 | 0 |
| test-e2e | 549.69 | 503.67 | 497.81 | 503.67 | 320 | 116 | 0 |
| test-kv | 2.77 | 2.83 | 2.76 | 2.77 | 145 | 0 | 0 |
| test-moe-gdn | 8.17 | 8.40 | 8.30 | 8.30 | 224 | 2 | 0 |
| test-quant | 166.84 | 166.81 | 167.88 | 166.84 | 367 | 4 | 0 |
| test-text | 0.03 | 0.03 | 0.03 | 0.03 | 313 | 0 | 0 |

Flakes in the 3 quiet runs: none (33 of 33 rc 0). Contended run (`gpu/contended/`): `NvFP4SmallMTest.A4BandwidthStableWithoutWindow` red 2 of 2 at load 16.3-16.6 (`tests/test_nvfp4_smallm.cu:315`, 113.5 and 148.8 us vs `< 70.0`), green 3 of 3 here; `CudaFaultSignalTest.IllegalAddressIsStickyAndThrowsAtTheNextForward` red under `docker run -u 1000:1000` (death test: `chdir("/home/imp")` denied), green as the image user.

### Top 20 by runtime (per case)

CI lane and local GPU together (medians of 3; the CI-lane top case `FpBitsLegacyParity.Fp16EncodeFloatSweep` 2.38 s does not reach the list).

| # | case | binary | lane | median s | 3 runs s |
|---:|---|---|---|---:|---|
| 1 | `ServingSignalsTest.FirstShortPromptsReplayPrewarmedPrefillGraphs` | test-e2e | local GPU | 153.85 | 154.64 / 153.84 / 153.85 |
| 2 | `Nvfp4GemmTileBench.VsCublasLt` | test-quant | local GPU | 113.47 | 113.47 / 113.49 / 113.47 |
| 3 | `SpecCaptureFidelityTest.CachedGraphMatchesEagerForward` | test-e2e | local GPU | 48.11 | 46.76 / 48.11 / 48.44 |
| 4 | `FmhaSm120Test.HD256_LongSeq_SlidingWindow1024` | test-attention | local GPU | 43.84 | 43.84 / 43.60 / 45.31 |
| 5 | `FmhaSm120Test.HD256_LongSeq` | test-attention | local GPU | 43.27 | 43.62 / 41.94 / 43.27 |
| 6 | `MtpQwen4ExpReference.TwoDraftStepsMatchVllmMath` | test-e2e | local GPU | 33.98 | 33.83 / 33.98 / 34.48 |
| 7 | `MtpGreedyIdentityTest.MtpDoesNotChangeGreedyTokens` | test-e2e | local GPU | 29.64 | 33.77 / 29.64 / 28.59 |
| 8 | `MtpGreedyIdentityTest.NgramDoesNotChangeGreedyTokensHybrid` | test-e2e | local GPU | 25.27 | 30.98 / 25.21 / 25.27 |
| 9 | `FmhaSm120Test.HD128_LongSeq_SlidingWindow1024` | test-attention | local GPU | 24.05 | 24.05 / 24.14 / 23.80 |
| 10 | `GptOssSinkRef.Fa2Hd64SinkMatchesReference` | test-attention | local GPU | 21.77 | 21.93 / 21.77 / 21.77 |
| 11 | `MtpGreedyIdentityTest.MtpDoesNotChangeConstrainedGreedyTokens` | test-e2e | local GPU | 21.08 | 26.85 / 21.08 / 20.68 |
| 12 | `BatchInvarianceTest.Fp8DenseDecodeOwnSmallMVsCublasLt` | test-e2e | local GPU | 16.91 | 16.91 / 17.07 / 16.02 |
| 13 | `FmhaFA2Dense2CtaTest.Chunked` | test-attention | local GPU | 16.37 | 16.72 / 16.37 / 14.59 |
| 14 | `MtpGreedyIdentityTest.DFlashDoesNotChangeGreedyTokens` | test-e2e | local GPU | 12.58 | 14.38 / 12.58 / 11.87 |
| 15 | `FmhaSm120Test.ChunkedLargeContext` | test-attention | local GPU | 12.45 | 12.36 / 12.50 / 12.45 |
| 16 | `MtpGreedyIdentityTest.DSparkDoesNotChangeGreedyTokens` | test-e2e | local GPU | 12.19 | 14.38 / 12.19 / 12.06 |
| 17 | `BatchInvarianceTest.NativeNvfp4DecodeSoloVsBatched` | test-e2e | local GPU | 10.73 | 10.73 / 10.88 / 10.03 |
| 18 | `MtpGreedyIdentityTest.NgramDoesNotChangeGreedyTokensDense` | test-e2e | local GPU | 10.72 | 13.68 / 10.72 / 10.30 |
| 19 | `FmhaFA2Hd256Test.PvF16_LongCtx1536` | test-attention | local GPU | 10.50 | 10.50 / 11.03 / 10.42 |
| 20 | `FmhaFA2Test.FP16QK_Bkv32Band_Chunked` | test-attention | local GPU | 9.48 | 10.19 / 9.20 / 9.48 |

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
| A12 | Flakes in the 3 CI-lane runs: none (rc 0, 0 failures x3). GPU: none in the 3 quiet runs; `NvFP4SmallMTest.A4BandwidthStableWithoutWindow` red 2 of 2 at load 16 (contended run, wall-clock floor) | `junit/ctest_unit_*.xml`, `gpu/`, `gpu/contended/` |
| A13 | Build cost is in compilation, not linking: test-core 256 TUs = 1142.7 CPU-s, its link 1.7 s; 8 of the top 10 TUs are test-core `.cpp` files (22.7-33.6 s each) | `tables/top_compile.md` |

### Per-suite table

Columns: runtime = sum over the suite's cases of the per-case median of 3 runs, suffix `C` = CI lane, `G` = local GPU (quiet baseline), `n/r` = no case ran (test-hf-network, opt-in). build = median clean-build seconds of the TU(s) the suite lives in (a TU shared by several suites is counted in each). what it checks = one line per suite from the TEST bodies (names + first assertions, read by two Haiku agents, spot-checked); prefixes `BENCH:`, `NO-ASSERT:`, `TIMING:` are their flags. GPU = the binary is a GPU module. weights: `n` for every suite that ran in the CI lane (no model mount, no skips); `y: <VAR>` = the suite's file reads that checkpoint/reference variable; suites that include `tests/test_models.h` or `tests/inkling_fixture.h` are settled by the GPU run (`/models` mounted, no model env): `y: all N skip`, `y: k of N skip`, or `y: ran on a hard-coded /models default`.

| Test (suite) | module | cases | file | runtime s | build s | what it checks | GPU | weights (env) | skipped |
|---|---|---:|---|---:|---:|---|---|---|---:|
| ApaPrefillTest | test-attention | 8 | test_attention_ra2.cu | 0.75G | 9.3 | APA prefill: rerun bit-identity, all-hot exact pass, paged KV equals flat, workspace carve | y | n |  |
| AttentionChunkedTest | test-attention | 5 | test_attention_chunked.cu | 0.00G | 8.5 | cuBLAS chunked prefill attention: square, offset causal mask, GQA, sliding window vs naive | y | n |  |
| AttentionCrossPathTest | test-attention | 11 | test_attention_crosspath.cu | 0.59G | 11.1 | Attention golden configs: MHA/GQA, softcap, sliding window, 1-2 token edge cases | y | n |  |
| Fa2Hd512Test | test-attention | 4 | test_attention_fa2_hd512.cu | 0.10G | 10.6 | FA2 hd=512 causal GQA prefill attention vs CPU reference | y | n |  |
| FmhaFA2Dense2CtaTest | test-attention | 5 | test_fmha_fp8.cu | 38.44G | 10.1 | FA2 dense 2-CTA FP8-KV prefill vs reference over causal, odd, window, chunked shapes | y | n |  |
| FmhaFA2Hd256Test | test-attention | 14 | test_fmha_fp8.cu | 11.99G | 10.1 | FA2 hd=256 PV-f16 prefill vs reference over causal, GQA, window, softcap; WMMA micro-bench | y | n |  |
| FmhaFA2HeavyFirstTest | test-attention | 3 | test_fmha_fp8.cu | 0.08G | 10.1 | FA2 heavy-first CTA order: outputs bit-identical across causal, chunked, odd lengths | y | n |  |
| FmhaFA2PvF16Test | test-attention | 13 | test_fmha_fp8.cu | 6.20G | 10.1 | FA2 PV-f16 accumulate prefill vs reference: causal, GQA, window, softcap, chunk shapes | y | n |  |
| FmhaFA2Test | test-attention | 21 | test_fmha_fp8.cu | 19.25G | 10.1 | FA2 FP16-QK prefill vs reference: causal, GQA, window, chunk offsets, occupancy bands | y | n |  |
| FmhaHd512Test | test-attention | 9 | test_attention_fmha_hd512.cu | 0.80G | 10.8 | WMMA FMHA hd=512 vs reference at Gemma-4 geometry; sliced cuBLAS parity; disabled benches | y | n |  |
| FmhaSm120Test | test-attention | 24 | test_attention_fmha_sm120.cu | 125.77G | 9.5 | sm_120 FMHA hd 64/96/128/256 vs reference: GQA, sliding window, softcap, chunked q_offset | y | n |  |
| GptOssSinkRef | test-attention | 9 | test_gpt_oss_sinks_ref.cu | 22.08G | 9.9 | Sink logits vs fp64 reference: no-sink softmax, FMHA, FA2 hd64, paged decode, streaming | y | n |  |
| INT4QuantTest | test-attention | 1 | test_paged_attention.cu | 0.00G | 11.7 | INT4 KV nibble pack/unpack roundtrip over signed range [-8, 7] | y | n |  |
| PagedAttentionINT4Test | test-attention | 1 | test_paged_attention.cu | 0.00G | 11.7 | Paged decode with INT4 KV vs FP32 CPU reference, single head | y | n |  |
| PagedAttentionReduce | test-attention | 3 | test_attention_paged_reduce.cu | 0.02G | 10.1 | Paged split-K reduce vs fp64 merge; staged and unstaged paths bit-identical; Q8 epilogue | y | n |  |
| PagedAttentionTest | test-attention | 24 | test_paged_attention.cu | 0.04G | 11.7 | Paged decode FP16 KV vs CPU reference: blocks, GQA, batch lengths, split-K, sinks, HD512 | y | n |  |
| PagedF16Decode | test-attention | 1 | test_attention_paged_oracle.cu | 1.57G | 15.9 | BENCH: F16 paged decode serving-shape microbench (times kernels, prints numbers) | y | n |  |
| PagedF16Multitok | test-attention | 1 | test_attention_paged_oracle.cu | 0.69G | 15.9 | F16 multitok paged attention vs reference on split-K and direct routes | y | n |  |
| PagedFp8Decode | test-attention | 1 | test_attention_paged_oracle.cu | 1.11G | 15.9 | BENCH: FP8 paged decode serving-shape microbench (times kernels, prints numbers) | y | n |  |
| PagedFp8Multitok | test-attention | 2 | test_attention_paged_oracle.cu | 1.51G | 15.9 | FP8 multitok paged decode vs reference: single split and grouped split rows | y | n |  |
| PagedNvfp4Decode | test-attention | 1 | test_attention_paged_oracle.cu | 1.19G | 15.9 | BENCH: NVFP4 paged decode long-context microbench (times kernels, prints numbers) | y | n |  |
| PagedNvfp4Mma | test-attention | 1 | test_attention_paged_oracle.cu | 6.21G | 15.9 | NVFP4 MMA split-K decode vs fp64 reference over GQA ratios and HD 128/256 | y | n |  |
| PagedNvfp4Multitok | test-attention | 1 | test_attention_paged_oracle.cu | 0.59G | 15.9 | NVFP4 four-token paged kernels vs reference on both split-K routes | y | n |  |
| PagedOracle | test-attention | 18 | test_attention_paged_oracle.cu | 1.58G | 15.9 | Paged oracle: HD128/HD256 sweeps; sinks and window over evicted blocks, all KV dtypes | y | n |  |
| PagedSplitKFallback | test-attention | 1 | test_attention_paged_oracle.cu | 0.03G | 15.9 | Split-K phase-1 launch failure falls back; long context matches split-K and reference | y | n |  |
| QsaIndexer | test-attention | 5 | test_qsa_indexer.cu | 0.31G | 10.1 | QSA indexer: dense-paged equivalence, CPU top-k selection, query prep, split-K, GQA paths | y | n |  |
| Ra2PrefillTest | test-attention | 6 | test_attention_ra2.cu | 0.79G | 9.3 | RA2 FP4 prefill attention vs FP32 CPU oracle: hd 64/128/256, chunk continuation, declines | y | n |  |
| SparseAttnE2E | test-attention | 3 | test_sparse_attn_select.cu | 0.01G | 11.9 | Sparse attention select end to end: identity table bit-identical; MLA budget keeps needle | y | n |  |
| SparseMinMaxTest | test-attention | 8 | test_sparse_attn_select.cu | 0.00G | 11.9 | Sparse key min/max and mean/std block metadata: prefill, decode, rollback, FP8, NVFP4 | y | n |  |
| SparsePrefillGeometry | test-attention | 1 | test_sparse_prefill.cu | 0.00G | 9.6 | Sparse prefill past-token tail geometry (namespace helper) keeps the tail tokens | y | n |  |
| SparsePrefillTest | test-attention | 2 | test_sparse_prefill.cu | 0.00G | 9.6 | Sparse prefill compacted block table: full budget bit-identical, sink/recent/needle order | y | n |  |
| SparseScoreMeanStd | test-attention | 1 | test_sparse_attn_select.cu | 0.00G | 11.9 | Sparse block selection: outlier page loses its budget slot to the consistent page | y | n |  |
| SparseScoreSplit | test-attention | 1 | test_sparse_attn_select.cu | 0.01G | 11.9 | Sparse block scores at Qwen3.8-27B shape vs reference (split score path) | y | n |  |
| SparseSelectTest | test-attention | 5 | test_sparse_attn_select.cu | 0.00G | 11.9 | Sparse block top-k: identity under budget, ascending order, negative scores, spec rows | y | n |  |
| ActivationTest | test-compute | 14 | test_activation.cu | 0.03G | 8.6 | SwiGLU, GELU, GeGLU vs CPU reference: FP32/FP16, vector/tail paths, Inf input, NVFP4 | y | n |  |
| CudaFaultSignalTest | test-compute | 2 | test_cuda_fault_signal.cu | 0.28G | 8.2 | Sticky vs non-sticky CUDA illegal-address faults: death-test exit, clear-and-return path | y | n |  |
| DrafterNorms | test-compute | 2 | test_drafter_norm_primitives.cu | 0.00G | 8.5 | Drafter head-norm with RoPE and row RMSNorm: per-row positions, weight types vs reference | y | n |  |
| EmbeddingTest | test-compute | 9 | test_embedding.cu | 0.00G | 7.4 | Embedding lookup FP32/FP16 vs host table: boundary ids, large vocab, device-side token ids | y | n |  |
| ExecutorKernelsTest | test-compute | 13 | test_executor_kernels.cu | 0.00G | 10.2 | Elementwise add, bias, scale, residual-RMSNorm kernels vs host values; decode pipeline | y | n |  |
| FP8GemmTest | test-compute | 13 | test_fp8_gemm.cu | 5.23G | 14.4 | FP8 and cuBLASLt GEMM/GEMV vs fp64 reference: rowscale heads, quant roundtrip, M-invariant | y | n | 2 |
| FP8ProjStreamTest | test-compute | 2 | test_fp8_proj_stream.cu | 0.54G | 9.4 | Streamed FP16 FP8 projection bit-equal to split-K reference; Sweep prints timings | y | n | 1 |
| FfnGraphCacheGpu | test-compute | 2 | test_ffn_graph_cache_gpu.cu | 0.00G | 8.0 | FFN graph cache: capture refusal runs eager and disables graphs; evicted exec updated | y | n |  |
| Fp32ResidualNorm | test-compute | 5 | test_fp32_residual_norm.cu | 0.06G | 8.0 | Fused fp32 residual+RMSNorm vs norm-then-quant chain; row bits M-independent; refusals | y | n |  |
| GatedResidualDecode | test-compute | 5 | test_gated_residual_decode.cu | 0.06G | 8.3 | Gated residual decode read/write vs unfused reference; refuses unsupported shapes | y | n |  |
| GemmCaptureProbe | test-compute | 2 | test_gemm_capture_probe.cu | 0.01G | 11.5 | GEMM workspace retaken after arena reset; cold shape inside graph capture stays valid | y | n |  |
| GemmCaptureSm120 | test-compute | 5 | test_gemm_capture_fp16_sm120.cu | 0.15G | 10.3 | sm_120 FP16 GEMM vs reference at several shapes incl narrow N; cuBLASLt compare | y | n |  |
| GemmDP4ATest | test-compute | 3 | test_gemm_dp4a.cu | 0.00G | 11.8 | DP4A GEMV Q8_0 and Q6_K vs dequant reference; fused SwiGLU Q8_1 quantize | y | n |  |
| GemmF16NarrowPrefill | test-compute | 7 | test_gemm_f16_narrow_prefill.cu | 7.12G | 12.4 | Narrow-N FP16 prefill GEMM vs reference, launch determinism, refusals, cuBLASLt bench | y | n |  |
| GemmF16NarrowSmallM | test-compute | 5 | test_gemm_f16_narrow_smallm.cu | 0.47G | 10.9 | Narrow-N small-M FP16 GEMM vs reference, determinism, refusals; graph-replay bench | y | n |  |
| GemmF16SplitK | test-compute | 1 | test_gemm_f16_tile.cu | 0.03G | 13.0 | Cluster split-K FP16 GEMM vs cuBLASLt over shapes | y | n |  |
| GemmF16TileBench | test-compute | 1 | test_gemm_f16_tile.cu | 1.77G | 13.0 | BENCH: FP16 tile GEMM vs cuBLASLt wall-time bench (times kernels, prints numbers) | y | n |  |
| GemmF16TileTest | test-compute | 6 | test_gemm_f16_tile.cu | 0.03G | 13.0 | FP16 tile GEMM vs cuBLASLt over shape/variant params; fp32 accumulate within rounding | y | n |  |
| GemmGroupedNvfp4Tile | test-compute | 1 | test_gemm_nvfp4_f16_tile.cu | 0.02G | 14.1 | Grouped NVFP4 tile GEMM: each expert's rows bit-equal to the single-expert GEMM | y | n |  |
| GemmKernelRegistryTest | test-compute | 17 | test_gemm_kernel_registry.cu | 0.00G | 12.2 | GEMM registry: NoMatch keys, GGUF Q8_0/Q6_K/dequant paths vs direct, scratch refusal | y | n |  |
| GemmMxfp4RepeatTest | test-compute | 2 | test_gemm_mxfp4_f16_tile.cu | 0.23G | 14.2 | MXFP4 FP16 tile GEMM bit-identical across repeated runs | y | n |  |
| GemmMxfp4TileBench | test-compute | 1 | test_gemm_mxfp4_f16_tile.cu | 0.41G | 14.2 | BENCH: MXFP4 tile GEMM vs dequant+GEMM wall-time bench | y | n |  |
| GemmMxfp4TileTest | test-compute | 8 | test_gemm_mxfp4_f16_tile.cu | 0.02G | 14.2 | MXFP4 tile GEMM bit-equal to dequant plus FP16 tile GEMM over variants | y | n |  |
| GemmNvfp4TileBench | test-compute | 1 | test_gemm_nvfp4_f16_tile.cu | 0.35G | 14.1 | BENCH: NVFP4 tile GEMM wall-time bench on Flint attention shapes | y | n |  |
| GemmNvfp4TileQuant | test-compute | 1 | test_gemm_nvfp4_f16_tile.cu | 0.00G | 14.1 | NVFP4 tile GEMM vs dequantize_nvfp4 reference after quantize roundtrip | y | n |  |
| GemmNvfp4TileTest | test-compute | 7 | test_gemm_nvfp4_f16_tile.cu | 0.14G | 14.1 | NVFP4 tile GEMM bit-equal to exact-F16-weight tile; scales cover E4M3 subnormals | y | n |  |
| GemmQ45kRepeat | test-compute | 1 | test_gemm_q45k_f16_tile.cu | 0.15G | 14.0 | Q4_K-to-F16 tile GEMM: repeated runs bit-identical on a multi-wave shape | y | n |  |
| GemmQ45kTileBench | test-compute | 1 | test_gemm_q45k_f16_tile.cu | 6.05G | 14.0 | BENCH: Q4_K tile GEMM vs IMMA with flushed cache (times kernels, prints numbers) | y | n |  |
| GemmQ45kTileTest | test-compute | 10 | test_gemm_q45k_f16_tile.cu | 0.02G | 14.0 | Q4_K-to-F16 tile GEMM bit-equal to dequant plus F16 tile GEMM over variants | y | n |  |
| GemmQ6kRepeatTest | test-compute | 2 | test_gemm_q6k_f16_tile.cu | 0.21G | 14.6 | Q6_K-to-F16 tile GEMM: repeated runs bit-identical on a multi-wave shape | y | n |  |
| GemmQ6kTileBench | test-compute | 1 | test_gemm_q6k_f16_tile.cu | 2.11G | 14.6 | BENCH: Q6_K tile GEMM vs dequant plus GEMM wall-time bench | y | n |  |
| GemmQ6kTileTest | test-compute | 8 | test_gemm_q6k_f16_tile.cu | 0.03G | 14.6 | Q6_K-to-F16 tile GEMM bit-equal to dequant plus F16 tile GEMM over variants | y | n |  |
| GemmStrategy | test-compute | 1 | test_gemm_kernel_registry.cu | 0.00G | 12.2 | GemmStrategy operator== compares by value (linear-scan registry lookup) | y | n |  |
| GemmTest | test-compute | 11 | test_gemm.cu | 0.00G | 7.3 | FP32/FP16 GEMM, GEMV and GEMV dispatch vs CPU reference, alpha/beta, zero matrix | y | n |  |
| GemvDp4aMultiTest | test-compute | 3 | test_gemv_dp4a_multi.cu | 0.25G | 13.4 | Multi-weight DP4A GEMV matches the single-weight GEMVs per weight | y | n |  |
| GemvFp16HotCold | test-compute | 1 | test_gemv_hot_cold_bench.cu | 0.00G | 8.8 | BENCH: FP16 GEMV hot vs cold cache timing (DISABLED, prints numbers) | y | n |  |
| GptOssYarnRef | test-compute | 2 | test_gpt_oss_yarn_ref.cu | 0.00G | 8.1 | gpt-oss YaRN RoPE: fp64 reference vs golden; kernel parity at long positions | y | n |  |
| HadamardTest | test-compute | 12 | test_hadamard.cu | 0.00G | 10.0 | Hadamard transform FP16 vs CPU reference: in-place path, involution property | y | n |  |
| InklingAttention | test-compute | 1 | test_inkling_kernels.cu | 0.50G | 9.9 | Inkling attention kernel: few-row form equals row form over 64 x 128 past keys | y | n |  |
| InklingAttnPaged | test-compute | 2 | test_inkling_attn_paged.cu | 1.28G | 10.0 | Inkling paged attention bit-equal to contiguous on shuffled blocks; hot/cold bench off | y | n |  |
| InklingCrossSlot | test-compute | 2 | test_inkling_cross_slot.cu | 0.00G | 13.7 | Inkling prefix restore across slots: restored prefix and draft match a fresh run | y | y: INKLING_ORACLE MODEL_INKLING | 2 |
| InklingDecode | test-compute | 2 | test_inkling_kernels.cu | 0.04G | 9.9 | Inkling decode: small-tile expert GEMM vs slot GEMVs; I8B32 projection vs dequant GEMM | y | n |  |
| InklingDraftModel | test-compute | 4 | test_inkling_draft.cu | 0.00G | 11.7 | Inkling draft model: drafts follow reference continuation; local ring and slab rings | y | y: INKLING_ORACLE MODEL_INKLING | 4 |
| InklingLookahead | test-compute | 2 | test_inkling_parity.cu | 0.00G | 21.3 | Inkling lookahead: rejected draft rows leave no trace; checkpoint inside a call | y | y: INKLING_ORACLE MODEL_INKLING | 2 |
| InklingLora | test-compute | 3 | test_inkling_lora.cu | 0.00G | 11.8 | Inkling LoRA: zero delta leaves forward unchanged; delta equals merged weights | y | y: MODEL_INKLING | 3 |
| InklingPagedKv | test-compute | 1 | test_inkling_parity.cu | 0.00G | 21.3 | Inkling full-attention K/V in pool: shuffled pool blocks bit-equal to contiguous | y | y: INKLING_ORACLE MODEL_INKLING | 1 |
| InklingParity | test-compute | 4 | test_inkling_parity.cu | 0.00G | 21.3 | Inkling vs reference: residual stream, logits, greedy continuation, grouped experts | y | y: INKLING_ORACLE MODEL_INKLING | 4 |
| InklingPhraseRewind | test-compute | 2 | test_inkling_phrase_rewind.cu | 0.00G | 11.8 | Inkling phrase rewind: seek replays first-run rows; banned token resamples text | y | y: INKLING_ORACLE MODEL_INKLING | 2 |
| InklingPrefill | test-compute | 4 | test_inkling_kernels.cu,test_inkling_parity.cu | 0.00G | 31.2 | Inkling grouped MXFP8 and NVFP4 expert GEMM vs slot GEMV; timing benches disabled | y | y: INKLING_ORACLE MODEL_INKLING | 1 |
| InklingQuality | test-compute | 1 | test_inkling_quality.cu | 0.00G | 10.5 | Inkling quality: KLD against reference logits on served call forms | y | y: INKLING_KLD INKLING_KLD_CHUNKS MODEL_INKLING | 1 |
| InklingRewind | test-compute | 6 | test_inkling_parity.cu | 0.00G | 21.3 | Inkling rewind: resumed prefix equals fresh run; request binds to longest-prefix slot | y | y: INKLING_ORACLE MODEL_INKLING | 6 |
| InklingServePath | test-compute | 1 | test_inkling_parity.cu | 0.00G | 21.3 | Inkling pool runtime bit-equal to contiguous forward on the serve path | y | y: INKLING_ORACLE MODEL_INKLING | 1 |
| InklingVisionParity | test-compute | 1 | test_inkling_parity.cu | 0.00G | 21.3 | Inkling vision tower merged features vs reference | y | y: INKLING_ORACLE MODEL_INKLING | 1 |
| LayerNormTest | test-compute | 9 | test_layernorm.cu | 0.02G | 9.4 | RMSNorm (FP32/FP16, residual, row-block, NVFP4 producer) vs CPU reference | y | n |  |
| LtGemmEx | test-compute | 5 | test_gemm_lt_ex.cu | 0.26G | 9.4 | cuBLASLt ex GEMM: strided batch, MoE grouped and FP8 per-expert scale, KV batched | y | n |  |
| MLAAbsorb | test-compute | 1 | test_mla.cu | 0.00G | 10.0 | MLA absorbed decode equals materialized-KV reference | y | n |  |
| MLAAttnOutput | test-compute | 6 | test_mla.cu | 0.04G | 10.0 | MLA attention output: v_head_dim != head_dim, prefill compaction, head gate, padded V | y | n |  |
| MLAProjection | test-compute | 2 | test_mla.cu | 0.00G | 10.0 | MLA two-step KV projection vs CPU reference; latent RMSNorm | y | n |  |
| MMVQ | test-compute | 6 | test_mmvq.cu | 0.17G | 8.6 | MMVQ vs DP4A GEMV for Q4_K, Q5_K, Q8_0 at small and large dims | y | n |  |
| MRope | test-compute | 5 | test_mrope.cu | 0.00G | 7.4 | Multi-axis RoPE: text positions equal single axis; interleaved and sectioned layouts | y | n |  |
| MmqQ4kImmaLayout | test-compute | 3 | test_mmq_q4k_imma_layout.cu | 0.00G | 8.7 | Q4_K IMMA weight reorder round trip: small, multi-superblock and FFN shapes | y | n |  |
| MoeRouterDecode | test-compute | 5 | test_moe_router_decode.cu | 0.06G | 11.2 | MoE router decode vs reference at Flint, bias, wide and odd-row shapes; refusal | y | n |  |
| MoeRouterFusedBench | test-compute | 1 | test_moe_router_fused.cu | 0.00G | 14.3 | NO-ASSERT: BENCH: MoE fused router tile/split sweep (DISABLED, prints timings) | y | n |  |
| MoeRouterFusedRefusal | test-compute | 1 | test_moe_router_fused.cu | 0.00G | 14.3 | MoE fused router refuses shapes outside its envelope (experts > 128) | y | n |  |
| MoeRouterFusedTest | test-compute | 8 | test_moe_router_fused.cu | 4.96G | 14.3 | MoE fused router ids and weights vs reference; rows independent of M and split | y | n |  |
| Nvfp4PromptStreamK | test-compute | 2 | test_nvfp4_prompt_streamk.cu | 0.26G | 10.7 | NVFP4 prompt stream-K GEMM: row invariance vs M; narrow tile matches wide tile | y | n |  |
| Nvfp4TileSwigluQuant | test-compute | 7 | test_nvfp4_tile_swiglu_quant.cu | 0.06G | 9.8 | NVFP4 tile SwiGLU plus quant fused: bytes equal the unfused path | y | n |  |
| PenaltyHistAppendTest | test-compute | 1 | test_penalty_hist_append.cu | 0.00G | 7.6 | Penalty history append: skips, bounds, bytes between sample slots | y | n |  |
| PromptLogprobsRows | test-compute | 4 | test_prompt_logprobs_rows.cu | 0.07G | 9.1 | Prompt logprob rows: row-major and slab-major outputs vs reference; entropy buffer | y | n |  |
| QkNormRopeFusedTest | test-compute | 3 | test_qknorm_rope_fused.cu | 0.01G | 11.6 | Fused QK-norm plus RoPE vs norm then standalone RoPE | y | n |  |
| QkNormRopeWarpTest | test-compute | 7 | test_qknorm_rope_fused.cu | 0.14G | 11.6 | Warp QK-norm plus RoPE bit-identical to the CTA kernel | y | n |  |
| RoPETest | test-compute | 10 | test_rope.cu | 0.00G | 9.4 | RoPE FP32/FP16, YaRN, long context, partial, MTP M-RoPE, query temperature vs refs | y | n |  |
| RowwiseTopM | test-compute | 4 | test_rowwise_topm.cu | 0.00G | 8.0 | Row-wise top-m: CPU reference, lowest-index tie break, M=1 equals argmax, arena exhaustion | y | n |  |
| SampleVerifyRows | test-compute | 9 | test_sampling_verify_rows.cu | 0.10G | 9.1 | Speculative verify sampling: row draws match eager sampler; min-p; coupled draft | y | n |  |
| SamplerCubPath | test-compute | 1 | test_sampling.cu | 0.00G | 12.5 | Sampler CUB path: each call samples from its own logits, not previous calls | y | n |  |
| SamplerSeeding | test-compute | 2 | test_sampling.cu | 0.11G | 12.5 | Sampler seeding: consecutive seeds differ; device salt draws like host step seed | y | n |  |
| SamplingAdvancedTest | test-compute | 7 | test_sampling_advanced.cu | 0.02G | 10.1 | TIMING: Logit bias, DRY penalty and Mirostat v2: token moves and seed stability | y | n |  |
| SamplingTest | test-compute | 30 | test_sampling.cu | 0.41G | 12.5 | Greedy, top-k, top-p (multiblock and CUB), NaN/inf logits, async rows, penalty rows | y | n |  |
| ShortConv | test-compute | 1 | test_shortconv.cu | 0.01G | 8.3 | Short conv chunked prefill then batched decode vs reference | y | n |  |
| SoftmaxTest | test-compute | 9 | test_activation.cu,test_softmax.cu | 0.00G | 17.8 | Softmax FP32/FP16 vs CPU reference: stability, large row, NVFP4 producer | y | n |  |
| SwigluQuantizeCutlass | test-compute | 1 | test_swiglu_quantize_cutlass.cu | 0.06G | 7.7 | SwiGLU plus quantize (CUTLASS) byte-identical to SwiGLU then quantize | y | n |  |
| TopkRadixSampler | test-compute | 3 | test_sampling_topk_radix.cu | 2.30G | 9.4 | Top-k radix sampler: bytes and tokens equal legacy path; unsupported shapes stay legacy | y | n |  |
| AnswerLostToReasoning | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | answer_lost_to_reasoning fires only when thinking ate the whole reply, bool cases | n | n |  |
| AnswerReserveKey | test-core | 4 | test_think_stop_logic.cpp | 0.00G | 4.9 | answer reserve tokens: default constant, floor, zero and negative clamp, EXPECT_EQ | n | n |  |
| AnthropicCacheControl | test-core | 11 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic cache_control markers map to cache-prompt flags across system/message/tool | n | n |  |
| AnthropicCacheUsage | test-core | 6 | test_anthropic_transform.cpp,test_anthropic_usage.cpp | 0.00G | 22.0 | Anthropic usage: spec counters and cache read/creation fields pass through, EXPECT_EQ | n | n |  |
| AnthropicContentBlocks | test-core | 9 | test_constraint_validation.cpp | 0.00G | 33.1 | Anthropic content-block validator accepts convertible blocks, refuses unknown/malformed | n | n |  |
| AnthropicEffort | test-core | 2 | test_anthropic_effort.cpp | 0.00G | 4.2 | Anthropic effort maps to reasoning_effort, nearest template level, EXPECT_EQ | n | n |  |
| AnthropicEnvelope | test-core | 5 | test_server_request_limits.cpp | 0.00G | 26.5 | Anthropic envelope: path detection, error wrapper per dialect, request-id header/body | n | n |  |
| AnthropicGuidedPassthrough | test-core | 3 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic shim passes guided-decoding and response_format fields through unchanged | n | n |  |
| AnthropicMessages | test-core | 12 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic messages to OpenAI shape: system, images, tool_use, tool_result, reasoning | n | n |  |
| AnthropicPrefill | test-core | 2 | test_assistant_prefill_transform.cpp | 0.00G | 4.5 | Anthropic assistant prefill sets continue_final_message only with trailing text | n | n |  |
| AnthropicReasoningReport | test-core | 2 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic usage reports reasoning tokens as output details, finish-detail copy rule | n | n |  |
| AnthropicResponse | test-core | 12 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic response shape: error envelope, text/thinking/tool_use blocks, bad args | n | n |  |
| AnthropicSampling | test-core | 1 | test_anthropic_effort.cpp | 0.00G | 4.2 | Anthropic sampling fields seed, top_k, temperature reach the chat body, EXPECT_EQ | n | n |  |
| AnthropicStopReason | test-core | 4 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic stop_reason: end_turn vs stop_sequence, engine reasons never leak verbatim | n | n |  |
| AnthropicSystemRole | test-core | 6 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic system role: leading system messages fold into the system prompt | n | n |  |
| AnthropicThinking | test-core | 15 | test_anthropic_thinking_blocks.cpp,test_anthropic_transform.cpp | 0.00G | 21.9 | Anthropic thinking config maps to enable_thinking and budget, clamp/signature rules | n | n |  |
| AnthropicToolChoice | test-core | 6 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic tool_choice maps to OpenAI tool_choice and parallel_tool_calls flags | n | n |  |
| AnthropicToolResult | test-core | 2 | test_anthropic_transform.cpp | 0.00G | 19.4 | Anthropic tool_result is_error labels the tool output for the model | n | n |  |
| ApiKeyAuth | test-core | 5 | test_server_auth.cpp | 0.00G | 19.3 | API key auth: Bearer and x-api-key accepted, mismatch and empty key refused | n | n |  |
| ArchIdentity | test-core | 4 | test_arch_identity.cpp | 0.01G | 2.9 | Arch identity: every arch string keeps its enum, one table, HF/GGUF loaders agree | n | n |  |
| Arena | test-core | 8 | test_memory_allocators.cpp | 0.00G | 4.8 | Arena allocator: aligned takes, usage, growth, exhaustion, reset, close, typed errors | n | n |  |
| ArenaLazy | test-core | 3 | test_memory_allocators.cpp | 0.00G | 4.8 | Lazy arena commits nothing until a take reaches it, fixed-region fallback, exhaustion | n | n |  |
| AttentionDispatchRules | test-core | 6 | test_attention_dispatch_rules.cpp | 0.00G | 1.9 | Attention dispatch: FMHA/FA2 head-dim coverage, MLA refusal, MXFP4 decode refusal | n | n |  |
| AttnDispatchTable | test-core | 9 | test_routing_decision.cpp | 0.00G | 3.4 | Attention dispatch table: default FA2, decline fallbacks, sinks routing, all-decline throw | n | n |  |
| AttnRoutingModelCoupling | test-core | 3 | test_routing_decision.cpp | 0.00G | 3.4 | Attention path replay: every tier replays to itself, sinks/no-accept map to none | n | n |  |
| AttnTierPrecedence | test-core | 2 | test_routing_decision.cpp | 0.00G | 3.4 | Attention tier precedence: FA2 beats FMHA sm120, FMHA beats Blackwell when both accept | n | n |  |
| AudioConfigTest | test-core | 3 | test_hf_config_loader.cpp | 0.00G | 6.6 | Audio config: modality flag set by object only, not null or absent, HF config load | n | n |  |
| AudioWav | test-core | 3 | test_audio_features.cpp | 0.00G | 3.7 | WAV reader: PCM16 mono, stereo averaging, extensible float, malformed files refused | n | n |  |
| AwqArch | test-core | 4 | test_awq_plan_sites.cpp | 0.00G | 6.5 | AWQ arch table: offset convention per arch, layer prefix resolution, plain-norm archs | n | n |  |
| AwqCalibration | test-core | 8 | test_awq_calibration.cpp | 0.00G | 4.5 | AWQ calibration file round-trip, foreign/truncated refusal, scale and fold algebra | n | n |  |
| AwqNormFold | test-core | 6 | test_awq_calibration.cpp | 0.00G | 4.5 | AWQ norm fold: unit-offset gain division, BF16 near-zero loss, clamp bounds, vector fold | n | n |  |
| AwqSites | test-core | 19 | test_awq_plan_sites.cpp | 0.00G | 6.5 | AWQ site selection: attention/GDN fold groups, group selector, tie map, default groups | n | n |  |
| BailingHybridConfig | test-core | 1 | test_bailing_hybrid.cpp | 0.00G | 3.4 | Bailing hybrid config: Ling3Tiny maps KDA, MLA and router fields, EXPECT_EQ | n | n |  |
| BailingHybridTensors | test-core | 3 | test_bailing_hybrid.cpp | 0.00G | 3.4 | Bailing hybrid tensors: KDA layer fused and MLA renamed, incomplete layer refused | n | n |  |
| BannedStringsParams | test-core | 1 | test_sampling_params_golden.cpp | 0.00G | 33.6 | Banned-strings param parse: list rules, null/absent handling, status codes | n | n |  |
| Base64 | test-core | 5 | test_base64.cpp | 0.00G | 20.0 | Base64: RFC 4648 vectors, padding at each residue, decode inverse, all byte values | n | n |  |
| BatchVerifyPredicates | test-core | 18 | test_batch_verify_predicates.cpp | 0.00G | 5.4 | Batched verify predicates: head slots, economics guard, carry order, history bindings | n | n |  |
| BearerAuth | test-core | 8 | test_server_auth.cpp | 0.00G | 19.3 | Bearer auth: token match accepted, wrong, missing prefix, empty, prefix and long tokens | n | n |  |
| BlockPool | test-core | 7 | test_memory_allocators.cpp | 0.00G | 4.8 | BlockPool: acquire/release balance, distinct stable blocks, exhaustion ref, churn | n | n |  |
| BlockPoolSlots | test-core | 2 | test_memory_allocators.cpp | 0.00G | 4.8 | BlockPool slots: lowest free id first, shrink drops only the free tail | n | n |  |
| BlockRef | test-core | 1 | test_memory_allocators.cpp | 0.00G | 4.8 | BlockRef move transfers the reference without changing the referent count | n | n |  |
| BudgetRecount | test-core | 4 | test_think_stop_logic.cpp | 0.00G | 4.9 | Budget recount: thinking opener counted between markers, prompt-seeded and no-think | n | n |  |
| BufferStamp | test-core | 3 | test_buffer_stamp.cpp | 0.00G | 1.7 | BufferStamp: live until a view retires it, matches buffer/weight/rows, clear drops it | n | n |  |
| CApiContract | test-core | 4 | test_c_api_contract.cpp | 0.00G | 2.0 | C API contract: null args invalid, lora unload, kv dtype validation, six dtypes pass | n | n |  |
| CApiEnumBinding | test-core | 6 | test_c_api_enum_binding.cpp | 0.00G | 1.8 | C API enum binding: arch ids match public header, unique, round-trip, dtype stable | n | n |  |
| CalibrationOutPath | test-core | 4 | test_server_args.cpp | 0.00G | 3.2 | Calibration output path: flag wins over config key, empty key does not clobber | n | n |  |
| CapabilityRefusal | test-core | 6 | test_capability_refusal.cpp | 0.00G | 22.9 | Capability refusal: Inkling refuses loop features, OpenAI/Anthropic envelopes carry param | n | n |  |
| CfgParams | test-core | 1 | test_sampling_params_golden.cpp | 0.00G | 33.6 | cfg_scale and cfg_negative_prompt parse rules, status codes, defaults | n | n |  |
| CgroupHeadroomTest | test-core | 1 | test_weight_snapshot.cpp | 0.00G | 4.0 | cgroup headroom: limit minus non-reclaimable memory, parse cases, EXPECT_EQ | n | n |  |
| ChannelSplit | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Channel split: thought, content, unknown and analysis routing of reasoning output | n | n |  |
| CheckpointLimits | test-core | 20 | test_checkpoint_limits.cpp | 0.24G | 6.1 | Checkpoint limits: huge/overflow layer and expert indices, dims, shard names, traversal | n | n |  |
| CliConfigPins | test-core | 1 | test_mtp_auto.cpp | 0.00G | 4.3 | CLI config pins: --perplexity commits KV pool up front, other paths stay growable | n | n |  |
| ClientErrorLogLine | test-core | 2 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Client error log line: status, route, reason on one line, control bytes stripped | n | n |  |
| CombineImageHash | test-core | 1 | test_image_placeholders.cpp | 0.00G | 3.1 | combine_image_hash: order-sensitive, never zero, single image hashes to itself | n | n |  |
| ConfigBadValue | test-core | 5 | test_config.cpp | 0.00G | 4.7 | Config bad value: bool, int, float and tri-state keys reject unreadable values | n | n |  |
| ConfigBinding | test-core | 1 | test_config.cpp | 0.00G | 4.7 | Config binding: prefill_chunk_size default -1, override reaches the field | n | n |  |
| ConfigBindings | test-core | 1 | test_config_bindings.cpp | 0.00G | 3.7 | Config bindings: every example key moves only its own field, count floor | n | n |  |
| ConfigExplicitKeys | test-core | 1 | test_mtp_auto.cpp | 0.00G | 4.3 | Config explicit keys: was_set true only for keys the operator set | n | n |  |
| ConstraintValidation | test-core | 17 | test_constraint_validation.cpp | 0.00G | 33.1 | Constraint validation: regex/grammar accepted or refused with parser error, all spellings | n | n |  |
| ContentParts | test-core | 8 | test_constraint_validation.cpp | 0.00G | 33.1 | Content parts: text, image, video, input_audio accepted, unknown and malformed refused | n | n |  |
| ControlVector | test-core | 4 | test_control_vector.cpp | 0.00G | 3.7 | Control-vector GGUF load: direction per layer, layer/width/type/truncated refusals | n | n |  |
| CorsOrigins | test-core | 3 | test_server_auth.cpp | 0.00G | 19.3 | CORS allow-origin: no origin by default, exact echo, wildcard allows any origin | n | n |  |
| CudaErrorsTest | test-core | 6 | test_cuda_errors.cpp | 0.33G | 2.4 | CUDA error classes: sticky faults unrecoverable, sync throws naming site, EXPECT_THROW | n | n |  |
| CudaStaticReset | test-core | 1 | test_cuda_static_reset.cpp | 0.00G | 1.7 | CUDA static-reset hook registry is populated, EXPECT_GT | n | n |  |
| DecodeGraphCtx | test-core | 6 | test_decode_graph_ctx.cpp | 0.00G | 2.0 | Decode graph capture: pow2 buckets, deterministic capture contexts, batch key order | n | n |  |
| DequantAwq | test-core | 3 | test_dequant_awq.cpp | 0.00G | 2.2 | AWQ dequant bit-exact vs reference per group/column, GEMM layout shape checks | n | n |  |
| DequantCap | test-core | 3 | test_spec_request.cpp | 0.00G | 2.5 | Dequant graph-capture cap: lm_head exempt, eligible plane over cap disables capture | n | n |  |
| DequantGptq | test-core | 7 | test_dequant_gptq.cpp | 0.00G | 3.0 | GPTQ dequant bit-exact (asym zeros v1/v2), g_idx group select, zero-format parse | n | n |  |
| DeviceInfoTest | test-core | 4 | test_device_info.cpp | 0.00G | 1.8 | Device info: SM count fallback 170, sm120 and PDL cc gates, cached values stable | n | n |  |
| DeviceOwnedTest | test-core | 17 | test_device_owned.cpp | 0.00G | 2.9 | DeviceOwned handles: empty/zero-byte, adopt allocator releases once, async stream release | n | n |  |
| DispatchPaths | test-core | 2 | test_dispatch_record.cpp | 0.00G | 2.7 | Dispatch path enum: every value named, names distinct within each family | n | n |  |
| DispatchPolicyFwd | test-core | 1 | test_config.cpp | 0.00G | 4.7 | Dispatch policy: null reads the default, wired policy returned as itself | n | n |  |
| DispatchRecord | test-core | 2 | test_dispatch_record.cpp | 0.00G | 2.7 | Dispatch record starts unset, tier flags track their setters, dump gate | n | n |  |
| DraftRowDraws | test-core | 2 | test_step_seed.cpp | 0.00G | 4.9 | Draft row draws: level rows take verify seed, greedy top_k 1, empty without sampled row | n | n |  |
| DropNullFields | test-core | 1 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Null JSON fields dropped at top level only, nested nulls kept | n | n |  |
| EffectiveCapabilities | test-core | 7 | test_tensor_kind_table.cpp | 0.00G | 3.0 | Effective capabilities: NVFP4 stripped for sub-5-bit or F16 sources, floor supported | n | n |  |
| EndThinking | test-core | 7 | test_think_stop_logic.cpp | 0.00G | 4.9 | End-thinking forcing: ChatML newline then closer, Harmony final opener, no-op outside | n | n |  |
| EndThinkingRegistry | test-core | 5 | test_end_thinking.cpp | 0.00G | 3.6 | End-thinking registry: unknown id NotFound, pending flagged once, phase sets status | n | n |  |
| EngineCapabilities | test-core | 8 | test_engine_capabilities.cpp | 0.00G | 2.1 | Engine capability gates: dense serves all, Inkling/gated forwards refuse unsupported | n | n |  |
| EngineConfigBridge | test-core | 4 | test_engine_config_bridge.cpp | 0.00G | 2.5 | Engine config bridge: explicit default beats imp.conf, unset falls through, EXPECT_EQ | n | n |  |
| EngineMetricsCell | test-core | 4 | test_engine_metrics_snapshot.cpp | 0.03G | 2.9 | TIMING: Metrics cell: zero before publish, copies, no torn concurrent reads, bounded | n | n |  |
| EngineStreamOpen | test-core | 2 | test_engine_stream_open.cpp | 0.15G | 2.9 | Engine stream open: result matches the stream, no device fails the open | n | n | 1 |
| ErrorEnvelope | test-core | 1 | test_server_request_limits.cpp | 0.00G | 26.5 | Error envelope: param and code fields present only when supplied | n | n |  |
| ExecApaWorkspaceBytes | test-core | 1 | test_workspace_sizes.cpp | 0.00G | 4.1 | APA workspace size follows the carve layout, exact byte value | n | n |  |
| ExecGdnDtBytes | test-core | 2 | test_workspace_sizes.cpp | 0.00G | 4.1 | GDN dt buffer bound holds for every row count up to max tokens, ASSERT_LE | n | n |  |
| ExecKdaKimiScratch | test-core | 1 | test_workspace_sizes.cpp | 0.00G | 4.1 | KDA scratch: split-QKV and LoRA buffers fit, offsets exact, EXPECT_EQ | n | n |  |
| ExecMaxTokens | test-core | 3 | test_workspace_sizes.cpp | 0.00G | 4.1 | exec max_tokens clamps to 4096, config fallback, replicates as-built GDN cap | n | n |  |
| ExecMaxWeightK | test-core | 1 | test_workspace_sizes.cpp | 0.00G | 4.1 | Max logical weight K over tensors equals 12288, EXPECT_EQ | n | n |  |
| ExecRa2WorkspaceBytes | test-core | 1 | test_workspace_sizes.cpp | 0.00G | 4.1 | RA2 workspace size follows the carve layout, exact byte value | n | n |  |
| ExecSsmConvF32 | test-core | 1 | test_workspace_sizes.cpp | 0.00G | 4.1 | SSM conv f32 region ends inside proj buffer for GDN and KDA, EXPECT_LE | n | n |  |
| ExecSsmZCols | test-core | 4 | test_workspace_sizes.cpp | 0.00G | 4.1 | SSM z-columns cover the attention gate or the recurrent inner, whichever is larger | n | n |  |
| ExecT2Demand | test-core | 25 | test_workspace_sizes.cpp | 0.00G | 4.1 | T2 workspace demand: GEMM scratch formula, cap handling, largest fitting target | n | n |  |
| ExitCodes | test-core | 4 | test_exit_codes.cpp | 0.00G | 1.9 | Exit codes: success 0, taxonomy maps one-to-one, distinct bytes, unknown error to 1 | n | n |  |
| ExpertBatchPolicy | test-core | 9 | test_expert_batch_policy.cpp | 0.00G | 4.0 | Expert batch policy: auto cap at device rows, host-expert tier resolves to one | n | n |  |
| ExpertCacheReinitCpu | test-core | 2 | test_expert_cache_reinit.cpp | 0.00G | 2.5 | Expert cache reinit: refused budgets leave cache disabled, empty layer mask refused | n | n |  |
| ExpertDestack | test-core | 9 | test_quantize_expert_destack.cpp | 0.00G | 2.9 | Expert destack: layout by model type, GPT-OSS/Gemma4 plans, gate-up and down reads | n | n |  |
| ExpertPack | test-core | 5 | test_expert_pack.cpp | 0.00G | 2.3 | Expert pack: uniform F16 shape, quantized/mixed refused, failed alloc leaves experts | n | n |  |
| ExpertPlacement | test-core | 6 | test_expert_placement.cpp | 0.00G | 1.6 | Expert placement: NVFP4 host path needed only with host-resident expert layers | n | n |  |
| ExtractReasoning | test-core | 6 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Reasoning extraction: think block split, unclosed think, last close tag wins | n | n |  |
| FallbackSite | test-core | 2 | test_log_once.cpp | 0.00G | 2.5 | Fallback counters: every hit counted, logged once per load, same-name sites sum | n | n |  |
| FfnGraphCache | test-core | 2 | test_ffn_graph_cache.cpp | 0.00G | 2.0 | FFN graph cache: row bucket steps and bounds, empty cache runs eager | n | n |  |
| Fim | test-core | 5 | test_fim.cpp | 0.00G | 2.3 | FIM token discovery: synthetic markers, PSM order, GGUF metadata ids win | n | n |  |
| FinishReason | test-core | 2 | test_server_request_limits.cpp | 0.00G | 26.5 | finish_reason: engine-only values map to the OpenAI enum, enum members pass through | n | n |  |
| FlintConfigTest | test-core | 3 | test_flint.cpp | 0.00G | 3.7 | Flint config load: layer map, router and shared expert, unsupported builds refused | n | n |  |
| FlintNames | test-core | 1 | test_flint.cpp | 0.00G | 3.7 | Flint tensor names: sink and expert-bias canonical names, EXPECT_EQ | n | n |  |
| ForceThinkEnd | test-core | 9 | test_think_stop_logic.cpp | 0.00G | 4.9 | Forced think-end: fires at budget while thinking, reserve cap grows answer room | n | n |  |
| ForcedToolEnvelope | test-core | 3 | test_tool_call.cpp | 0.00G | 16.5 | Forced tool-call envelope literals per family, Harmony and Gemma4 parse as a call | n | n |  |
| ForwardProfile | test-core | 2 | test_forward_profile.cpp | 0.00G | 1.6 | Forward profile never runs on a capturing stream, eager only when flagged | n | n |  |
| Fp8Keep | test-core | 9 | test_quantize_fp8_keep.cpp | 0.01G | 3.6 | FP8 keep-list: fused group selection, rank parse, byte budget by score, ModelOpt output | n | n |  |
| Fp8LmHead | test-core | 5 | test_fp8_lm_head.cpp | 0.09G | 3.4 | FP8 lm_head: decode matches bit-field reference for all codes, RNE encode, saturation | n | n |  |
| Fp8Source | test-core | 13 | test_quantize_fp8_source.cpp | 0.00G | 3.4 | FP8 E4M3 source: grid decode, block-edge derivation, per-block scale application | n | n |  |
| Fp8SourceAction | test-core | 4 | test_quantize_policy.cpp | 0.00G | 4.5 | FP8 source action: refused weights widened, gated flag from caller, lm_head flag | n | n |  |
| FpBits | test-core | 9 | test_fp_bits.cpp | 0.00G | 2.7 | Half/bf16 bit conversions: known values, round-trip, rescale, subnormals, RNE | n | n |  |
| FpBitsLegacyParity | test-core | 6 | test_fp_bits_legacy_parity.cpp | 2.29G | 2.4 | Exhaustive FP bit-parity vs legacy: all 65536 fp16/bf16 codes, E4M3, MXFP8, ngram | n | n |  |
| FusedGateQProj | test-core | 4 | test_quantize_policy.cpp | 0.00G | 4.5 | Fused gated q_proj finder: gated found, dense left alone, per layer, nested prefix | n | n |  |
| FuzzCorpus | test-core | 9 | test_fuzz_corpus.cpp | 0.77G | 2.8 | NO-ASSERT: Fuzz corpus runs each parser target over seeded corpora via run_target helper | n | n |  |
| GbnfDepthCap | test-core | 2 | test_gbnf_grammar.cpp | 0.00G | 4.9 | GBNF nesting depth cap: deep groups refused, ordinary nesting parses | n | n |  |
| GbnfGrammar | test-core | 23 | test_gbnf_grammar.cpp | 0.00G | 4.9 | GBNF grammar parser refusals: no root, undefined rule, left recursion, bad bounds | n | n |  |
| GbnfGrammarTest | test-core | 1 | test_gbnf_grammar.cpp | 0.00G | 4.9 | GBNF matcher stays bounded under deep self-recursion, would_accept battery | n | n |  |
| GdnChunkparArgs | test-core | 1 | test_launcher_args.cpp | 0.00G | 3.3 | GDN chunk-parallel args struct: fields carried in declaration order, EXPECT_EQ | n | n |  |
| GdnFusedSplit | test-core | 3 | test_gdn_fused_split.cpp | 0.00G | 1.8 | GDN fused qkvz/ba split regroups per key-head group, refuses bad shape or quantized | n | n |  |
| GdnProjection | test-core | 2 | test_quantize_policy.cpp | 0.00G | 4.5 | GDN projection name matcher: Qwen3.6 recipe names only, role flag vocabulary | n | n |  |
| GdnScanArgs | test-core | 2 | test_launcher_args.cpp | 0.00G | 3.3 | GDN scan args struct: fields in declaration order, omitted fields take defaults | n | n |  |
| GdnScanBatchedArgs | test-core | 2 | test_launcher_args.cpp | 0.00G | 3.3 | Batched GDN scan args struct: field order and strides, omitted fields default | n | n |  |
| GemmAlgoCapture | test-core | 3 | test_gemm_algo_capture.cpp | 0.00G | 1.5 | cuBLASLt algo probe runs only when query ok and not capturing, first fitting algo | n | n |  |
| GemmFallbackFailsLoud | test-core | 6 | test_gemm_fallback_fails_loud.cpp | 0.00G | 2.6 | GEMM fallback paths throw loudly on missing scratch, residual, cache or source | n | n |  |
| GemmKernelRegistryLink | test-core | 1 | test_gemm_kernel_registry_link.cpp | 0.00G | 1.5 | GEMM kernel registry holds exactly the produced tier/shape keys, EXPECT_EQ | n | n |  |
| GemmMode | test-core | 4 | test_gemm_mode.cpp | 0.00G | 2.0 | GEMM mode flags default off, decode-row and verify-small-M gating, EXPECT_FALSE | n | n |  |
| Gemma4AudioFeatures | test-core | 5 | test_audio_features.cpp | 0.04G | 3.7 | Gemma4 audio mel features match HF extractor, clip length refusals, HTK filters | n | n |  |
| Gemma4AudioLoad | test-core | 5 | test_gemma4_audio_load.cpp | 0.00G | 4.1 | Gemma4 audio config and tensor load: slots, tables, missing or wrong-shape refused | n | n |  |
| Gemma4AudioPlaceholders | test-core | 2 | test_audio_features.cpp | 0.00G | 3.7 | Gemma4 audio placeholder expansion per clip, count mismatch refused and untouched | n | n |  |
| Gemma4EConfig | test-core | 4 | test_gemma4_e_variant.cpp | 0.00G | 4.6 | Gemma4 E4B config: per-layer inputs, KV sources, PLI pad, double-wide MLP refused | n | n |  |
| Gemma4EWeights | test-core | 1 | test_gemma4_e_variant.cpp | 0.00G | 4.6 | Gemma4 E weights: per-layer input table and shared-KV mapping non-null, EXPECT_NE | n | n |  |
| Gemma4KvSources | test-core | 1 | test_gemma4_e_variant.cpp | 0.00G | 4.6 | Gemma4 KV source layers edge cases: empty and nothing-to-share, EXPECT | n | n |  |
| Gemma4RopeConfigTest | test-core | 2 | test_hf_config_loader.cpp | 0.00G | 6.6 | Gemma4 RoPE config: proportional full-attention table, absent without partial factor | n | n |  |
| GenerationConfigTest | test-core | 4 | test_hf_config_loader.cpp | 0.00G | 6.6 | generation_config.json: partial fields leave sentinels, do_sample false forces greedy | n | n |  |
| GgufFaultInjection | test-core | 27 | test_gguf_fault_injection.cpp | 0.00G | 3.5 | TIMING: GGUF fault injection: truncated, bad magic/version, huge counts refused, valid | n | n |  |
| GgufLlama3Bos | test-core | 1 | test_gguf_llama3.cpp | 0.00G | 2.9 | Llama-3 BPE GGUF defaults add_bos true, EXPECT on tokenizer flag | n | n |  |
| GgufLoaderTest | test-core | 9 | test_gguf_loader.cpp | 0.00G | 3.3 | GGUF loader: missing file, bad magic, truncation, version, zero tensors, arch refusal | n | n |  |
| GgufRopeFreqs | test-core | 3 | test_gguf_llama3.cpp | 0.00G | 2.9 | GGUF rope_freqs tensor: absent leaves tables empty, Llama3 closed form, bad length ignored | n | n |  |
| GgufSilentDrops | test-core | 8 | test_gguf_silent_drops.cpp | 0.00G | 4.6 | GGUF tensor-name mapping: bias and NextN blocks not silently dropped or clobbered | n | n |  |
| GptqHessianFile | test-core | 1 | test_awq_calibration.cpp | 0.00G | 4.5 | GPTQ Hessian file round-trip of entries, truncation rejected, ASSERT_EQ | n | n |  |
| GracePeriod | test-core | 6 | test_think_stop_logic.cpp | 0.00G | 4.9 | Grace-period stop: blocks exit until content seen, hard cap releases, whitespace not | n | n |  |
| GraphEligibility | test-core | 16 | test_graph_eligibility.cpp | 0.00G | 3.2 | Graph eligibility reasons: names distinct, KV pressure only mid-run, demote records first | n | n |  |
| GraphEpoch | test-core | 7 | test_graph_epoch.cpp | 0.00G | 2.5 | Graph epoch bump monotonic, stale graph refused and recaptured once, regrow bump rules | n | n |  |
| GraphSlotPool | test-core | 12 | test_graph_slots.cpp | 0.00G | 2.9 | Graph slot pool: open shape, bad capacities, double open, non-overlapping slot buffers | n | n |  |
| GrowRows | test-core | 3 | test_grow_rows.cpp | 0.00G | 2.2 | Growable rows commit in steps, refused commit keeps row count, fixed without backend | n | n |  |
| HFConfigGolden | test-core | 1 | test_hf_config_golden.cpp | 0.06G | 3.5 | HF config golden: all 98 fixtures match stored goldens, mismatch count zero | n | n |  |
| HarmonySplit | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Harmony channel split: analysis to reasoning, final to content, commentary as reasoning | n | n |  |
| HealthUnservable | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Health verdict: floored pool unservable and named, healthy and unknown cases | n | n |  |
| HfFetchInfo | test-core | 1 | test_hf_fetch.cpp | 0.00G | 23.2 | HF repo info parse: siblings, commit sha, gated flag, EXPECT_EQ | n | n |  |
| HfFetchMock | test-core | 10 | test_hf_fetch.cpp | 0.04G | 23.2 | HF fetch on mock server: verify sha, resume with Range, retry, sha mismatch fails | n | n |  |
| HfFetchParse | test-core | 2 | test_hf_fetch.cpp | 0.00G | 23.2 | hf:// URI parse: repo and file accepted, malformed rejected | n | n |  |
| HfFetchSelect | test-core | 3 | test_hf_fetch.cpp | 0.00G | 23.2 | HF file selection: single GGUF auto-picked, several need selector, safetensors set | n | n |  |
| HfFetchSha | test-core | 1 | test_hf_fetch.cpp | 0.00G | 23.2 | SHA-256 of files vs known vectors, EXPECT_EQ | n | n |  |
| HfFetchUrl | test-core | 1 | test_hf_fetch.cpp | 0.00G | 23.2 | HF API and resolve URL builders for repo and revision refs, EXPECT_EQ | n | n |  |
| Holdback | test-core | 10 | test_stream_pipeline.cpp | 0.00G | 4.0 | Stop-sequence holdback: flush without overrun, earliest match wins, empty stop skipped | n | n |  |
| HoldbackDecision | test-core | 2 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Holdback decision: flush cut on UTF-8 boundary, stop match cuts exactly at the match | n | n |  |
| HostEmbedding | test-core | 3 | test_host_embedding.cpp | 0.00G | 2.1 | Host embedding lookup bits match upload staging for bf16, f32, f16 | n | n |  |
| HostExpertTier | test-core | 6 | test_host_expert_tier.cpp | 0.01G | 3.7 | Host expert tier: LRU eviction, transient scratch, exchange, aligned and unaligned loads | n | n |  |
| HostFileMap | test-core | 1 | test_host_expert_tier.cpp | 0.00G | 3.7 | Host file map resolves pointers to fd-buffered offsets from maps text, EXPECT_LT | n | n |  |
| HostPinnedSites | test-core | 1 | test_host_pinned.cpp | 0.00G | 3.5 | Host pinned-memory sites: acquire, move, reset update byte and count totals | n | n |  |
| HostRamHeadroomTest | test-core | 1 | test_weight_snapshot.cpp | 0.00G | 4.0 | Host RAM headroom: 6 GiB floor or 10 percent, unknown total, EXPECT_EQ | n | n |  |
| HostRegistration | test-core | 8 | test_host_pinned.cpp | 0.00G | 3.5 | Host registration RAII: register once, unregister once, move and reset semantics | n | n |  |
| HostSpillAuto | test-core | 2 | test_snapshot_host_default.cpp | 0.00G | 2.2 | Host spill auto size: two-thirds of spare RAM in whole GiB, only unset key resolved | n | n |  |
| HostTaskPool | test-core | 3 | test_host_task_pool.cpp | 0.02G | 1.9 | Host task pool: wait sees every task, destructor joins, zero threads still runs | n | n |  |
| ImageDecode | test-core | 9 | test_image_decode.cpp | 0.00G | 3.8 | JPEG decode bit-matches Pillow, file and memory agree, GIF/BMP refused, corrupt fails | n | n |  |
| ImagePlaceholders | test-core | 8 | test_image_placeholders.cpp | 0.00G | 3.1 | Image placeholder expansion: per-image count, mismatch refused, sequence unchanged | n | n |  |
| ImageTokensBefore | test-core | 3 | test_image_placeholders.cpp | 0.00G | 3.1 | Image token offset before a chunk: counts prefix only, clamps, text-only starts at zero | n | n |  |
| InflightGate | test-core | 2 | test_server_request_limits.cpp | 0.01G | 26.5 | TIMING: In-flight gate: refuses limit+1, readmits after leave, concurrent entries capped | n | n |  |
| InklingDense8 | test-core | 1 | test_inkling_kv_layout.cpp | 0.00G | 3.2 | Inkling dense8 fit check: VRAM budget arithmetic, KV pool not charged, EXPECT | n | n |  |
| InklingGuardsFailLoud | test-core | 5 | test_inkling_guards_fail_loud.cpp | 0.00G | 1.6 | Inkling kernel guards throw loudly: head dim, missing state, router expert range | n | n |  |
| InklingImageParts | test-core | 4 | test_vision_family.cpp | 0.00G | 2.9 | Inkling image parts: picture inside text split out, header role kept, expansion count | n | n |  |
| InklingKvLayout | test-core | 2 | test_inkling_kv_layout.cpp | 0.00G | 3.2 | Inkling KV layout: global apps own one layer each, prefix-cache snapshot key | n | n |  |
| InklingLoraTargets | test-core | 3 | test_inkling_registry.cpp | 0.00G | 3.3 | Inkling LoRA targets: module names map to qkvo, other tensors refused, dims | n | n |  |
| InklingPrefixCap | test-core | 2 | test_inkling_kv_layout.cpp | 0.00G | 3.2 | Inkling prefix cap: never past slot match, pool holds max-seq plus lookahead rows | n | n |  |
| InklingRegistry | test-core | 1 | test_inkling_registry.cpp | 0.00G | 3.3 | Inkling weight registry: five borrowed fp16 handles per layer, 24-layer count | n | n |  |
| InklingSlabLayout | test-core | 6 | test_inkling_slab_layout.cpp | 0.00G | 3.9 | Inkling slab layout: slot conv state plus sliding rings, aligned disjoint regions | n | n |  |
| InklingSlotTable | test-core | 4 | test_inkling_slot_table.cpp | 0.00G | 3.0 | Inkling slot table: rows find slot by request id, unknown id throws, release unbinds | n | n |  |
| InklingSnapLayout | test-core | 3 | test_inkling_slab_layout.cpp | 0.00G | 3.9 | Inkling snapshot layout: entry bytes and sliding windows, restore onto another slot | n | n |  |
| InternVLLmNames | test-core | 3 | test_internvl_lm_names.cpp | 0.00G | 4.1 | InternVL LM tensor names: every LM tensor assigned, vision counted apart, Qwen3-VL | n | n |  |
| InternVLPreprocess | test-core | 2 | test_internvl_preprocess.cpp | 0.01G | 2.9 | InternVL tile preprocess matches HF within one step, prompt ids match HF processor | n | n |  |
| InternVLVision | test-core | 4 | test_internvl_vision.cpp | 0.00G | 3.8 | InternVL vision config parse, refusals, loader reads every tensor, HF per-stage match | n | n |  |
| JsonConstrainFsm | test-core | 14 | test_json_constrain_property.cpp | 0.00G | 6.9 | JSON-constrain FSM rejects a second decimal point in numbers, sim_token_valid battery | n | n |  |
| JsonConstrainPropertyTest | test-core | 9 | test_json_constrain_property.cpp | 0.01G | 6.9 | JSON constrainer property battery: valid docs and prefixes accepted, bad syntax rejected | n | n |  |
| JsonEscape | test-core | 1 | test_sse_stream_utils.cpp | 0.00G | 31.4 | JSON escaping: control chars, quotes, UTF-8 round-trip, EXPECT_EQ | n | n |  |
| JsonNestingDepth | test-core | 5 | test_sse_stream_utils.cpp | 0.00G | 31.4 | JSON nesting depth counter: plain nesting, braces inside strings ignored | n | n |  |
| JsonNumberBounds | test-core | 6 | test_json_number_bounds.cpp | 0.01G | 3.0 | JSON number bounds: exact compare, exclusivity, tighten, prefix reachability, brute force | n | n |  |
| JsonOutTest | test-core | 9 | test_json_out.cpp | 0.00G | 2.7 | JSON writer: empty object, key order, precision, escapes, UTF-8 pass-through | n | n |  |
| KVAccounting | test-core | 5 | test_kv_accounting.cpp | 0.00G | 2.7 | KV accounting: for-accounting geometry, salt seeds lookup, probe count, reuse stops at | n | n |  |
| KVCacheManagerTest | test-core | 47 | test_kv_cache.cpp | 0.00G | 7.0 | KV cache manager: allocate/rollback/append/free, LRU eviction, live seq never evicted | n | n |  |
| KVCacheTest | test-core | 8 | test_kv_cache.cpp | 0.00G | 7.0 | KV cache: construction, block alloc/free, refcount, NVFP4 head-dim reject, bytes per block | n | n |  |
| KVHostSpillTest | test-core | 8 | test_kv_host_spill.cpp | 0.00G | 3.0 | KV host spill tier: budget off, committed round-trip, LRU replace, chunk pin, bindings | n | n |  |
| KVSessionPins | test-core | 9 | test_kv_session_pins.cpp | 0.00G | 2.5 | KV session pins: prefix hit under pressure, LRU unpinned first, TTL and budget unpin | n | n |  |
| KdaReference | test-core | 4 | test_kda_reference.cpp | 0.00G | 3.1 | KDA reference recurrence matches FLA naive, gate bound and softplus closed form | n | n |  |
| KimiLinearConfig | test-core | 2 | test_kimi_linear.cpp | 0.00G | 4.5 | Kimi-Linear config: KDA, MLA nope and router mapping, unsupported refused | n | n |  |
| KimiLinearTensors | test-core | 4 | test_kimi_linear.cpp | 0.00G | 4.5 | Kimi-Linear tensors: dense QKV fuse, LoRA rename, NVFP4 scales kept, broken layout refused | n | n |  |
| KvAdmission | test-core | 5 | test_kv_admission.cpp | 0.00G | 4.2 | KV admission estimate: P90 of finished outputs with floor, expected length admits more | n | n |  |
| KvBlockSize | test-core | 2 | test_kv_residual_sizing.cpp | 0.00G | 2.5 | KV block size: multiples of 16 up to 256 served, all others refused with a reason | n | n |  |
| KvDtypeGolden | test-core | 5 | test_kv_dtype.cpp | 0.00G | 3.7 | KV dtype golden: block bytes and planner byte math per layer, FP4 head-dim multiple | n | n |  |
| KvDtypeInfo | test-core | 2 | test_kv_dtype.cpp | 0.00G | 3.7 | KV dtype info flags and block bytes match golden table, EXPECT_EQ | n | n |  |
| KvFp8Scale | test-core | 10 | test_kv_fp8_scale.cpp | 0.01G | 3.8 | KV FP8 scale: mode parse, pow2 ceil, amax/checkpoint sources, histogram bins, EXPECT | n | n |  |
| KvPoolVerdict | test-core | 6 | test_kv_residual_sizing.cpp | 0.00G | 2.5 | KV pool verdict: full sequence sufficient, clamped or floored messages, partial block | n | n |  |
| KvReplan | test-core | 10 | test_kv_replan.cpp | 0.00G | 2.1 | KV replan from cached measurement: ceiling gains freed reserve, live reserve floor | n | n |  |
| KvResidualSizing | test-core | 8 | test_kv_residual_sizing.cpp | 0.00G | 2.5 | KV residual sizing: plan fit, residual clamp vs floor, headroom boundary exact | n | n |  |
| KvScaleTensor | test-core | 3 | test_kv_fp8_scale.cpp | 0.00G | 3.8 | KV scale tensor names parse to K/V slot, host scalar read, matched-step routing | n | n |  |
| KvWriteRopeArgs | test-core | 2 | test_launcher_args.cpp | 0.00G | 3.3 | KV write RoPE args struct: fields in declaration order, omitted fields default | n | n |  |
| LaneNeeds | test-core | 4 | test_lane_needs.cpp | 0.01G | 3.7 | Lane needs: field combinations map to old lanes, logprobs not forced, mid-run bits | n | n |  |
| LatencyLadder | test-core | 5 | test_server_request_limits.cpp | 0.00G | 26.5 | Latency histogram ladders: seconds vs ITL buckets, cumulative sums, negatives clamp | n | n |  |
| LayerHostKeep | test-core | 8 | test_layer_host_keep.cpp | 0.00G | 5.9 | Layer host-keep: Qwen3-8B GPU-layers 20 keeps 16 layers on host, dense-only gate | n | n |  |
| LayerOffloadPlan | test-core | 5 | test_layer_offload_plan.cpp | 0.00G | 3.6 | Layer offload plan: Q8_0 block-sized bytes, 16 host layers planned, init refusals | n | n |  |
| LeakedStaticTest | test-core | 6 | test_leaked_static.cpp | 0.00G | 2.6 | Leaked-static holder: destructor never runs, owners release once, reset idempotent | n | n |  |
| Lfm2Names | test-core | 1 | test_lfm2_names.cpp | 0.00G | 1.7 | LFM2 canonical tensor names: modules and suffixes map, EXPECT_EQ | n | n |  |
| LibraryForwardWindow | test-core | 2 | test_library_reserve_measurement.cpp | 0.00G | 3.2 | Library forward window: tracked growth is no library claim, bytes never wrap | n | n |  |
| LibraryReserveCache | test-core | 9 | test_library_reserve_cache.cpp | 0.00G | 3.5 | Library reserve cache: key stability, file round-trip, missing key, remeasure replaces | n | n |  |
| LibraryReserveMeasurement | test-core | 3 | test_library_reserve_measurement.cpp | 0.00G | 3.2 | Library reserve measurement: no window means none, larger of window and residual | n | n |  |
| LlmCompressorFormatDetect | test-core | 8 | test_llm_compressor_loader.cpp | 0.00G | 9.3 | llm-compressor format detect: ModelOpt preferred, recipe YAML and compressed-tensors | n | n |  |
| LlmCompressorIgnoreList | test-core | 3 | test_llm_compressor_loader.cpp | 0.00G | 9.3 | llm-compressor ignore list: config.json names replace recipe patterns, router-proj rule | n | n |  |
| LlmCompressorKeepVision | test-core | 6 | test_llm_compressor_loader.cpp | 0.00G | 9.3 | llm-compressor keep-vision: visual prefix verbatim, vision suffixes kept, MTP skipped | n | n |  |
| LlmCompressorRecipe | test-core | 6 | test_llm_compressor_loader.cpp | 0.00G | 9.3 | llm-compressor recipe parse: Gemma4 and Qwen3.6 recipes ok, non-NVFP4 scheme rejected | n | n |  |
| LlmCompressorTranslate | test-core | 17 | test_llm_compressor_loader.cpp | 0.00G | 9.3 | llm-compressor tensor-name translate: packed and global-scale renames, vision skipped | n | n |  |
| LlmCompressorUnused | test-core | 4 | test_llm_compressor_loader.cpp | 0.00G | 9.3 | llm-compressor unused-tensor rule: MTP skipped not unused, drop only when no consumer | n | n |  |
| LmHeadRoute | test-core | 3 | test_lm_head_route.cpp | 0.00G | 1.6 | LM-head route decision table, NVFP4 not FP8-cached, draft head route, EXPECT_EQ | n | n |  |
| LoaderRopeParity | test-core | 7 | test_loader_rope_parity.cpp | 0.01G | 3.8 | Loader RoPE parity: Gemma4, Llama3, Granite, Qwen3, Gemma3 params match reference | n | n |  |
| LogLevel | test-core | 2 | test_process_diag.cpp | 0.00G | 2.5 | Log level parse: every word case-insensitive, anything else rejected | n | n |  |
| LogOnce | test-core | 5 | test_log_once.cpp | 0.00G | 2.5 | LogOnce latch: fires once then after rearm, exactly one thread wins a race | n | n |  |
| LogitBiasVocabError | test-core | 1 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Logit-bias vocab check: id at or above vocab size named in the error | n | n |  |
| LogitLensParams | test-core | 1 | test_sampling_params_golden.cpp | 0.00G | 33.6 | Logit-lens params parse: lens count, logprobs flag, EXPECT_EQ | n | n |  |
| LogprobShapes | test-core | 6 | test_logprobs_shapes.cpp | 0.00G | 22.7 | Logprob response shapes: chat array, completions parallel arrays, offsets, streaming | n | n |  |
| LogprobTokens | test-core | 3 | test_logprobs_shapes.cpp | 0.00G | 22.7 | Logprob token JSON: valid UTF-8 unchanged, half codepoint serialises, raw bytes | n | n |  |
| LoraDimsTest | test-core | 5 | test_lora_dims.cpp | 0.00G | 2.5 | LoRA dims: projection widths follow base, GQA kv narrower, FFN targets refused without | n | n |  |
| LtGemmArgs | test-core | 3 | test_gemm_lt_args.cpp | 0.00G | 3.2 | cuBLASLt GEMM args struct: fields in declaration order, GemmEx unstrided batch of one | n | n |  |
| LtGemmBatchedArgs | test-core | 2 | test_gemm_lt_args.cpp | 0.00G | 3.2 | Batched cuBLASLt GEMM args struct: field order and defaults, EXPECT_EQ | n | n |  |
| LtRouteTrace | test-core | 3 | test_lt_route_trace.cpp | 0.00G | 1.9 | cuBLASLt route trace labels: gemm precedence, FP8 label, note and dump follow flag | n | n |  |
| MLAConfig | test-core | 6 | test_mla.cpp | 0.00G | 3.4 | MLA config: YaRN mscale, DeepSeek-V2-Lite fields parse, MLA profile selection | n | n |  |
| MLATensorKindMatcher | test-core | 1 | test_mla.cpp | 0.00G | 3.4 | MLA tensor kind matcher: kv_a and kv_a_layernorm projection kinds, EXPECT_EQ | n | n |  |
| MLAWeightMap | test-core | 5 | test_mla.cpp | 0.00G | 3.4 | MLA weight map: DeepSeek dense, router, routed and shared experts, attention names | n | n |  |
| MRopeConfig | test-core | 4 | test_mrope_config.cpp | 0.00G | 2.8 | M-RoPE config: split read from rope params or scaling, text-only off, malformed ignored | n | n |  |
| MRopePositions | test-core | 11 | test_mrope_positions.cpp | 0.00G | 2.8 | M-RoPE positions: text ascending on 3 axes, image row/col, continuation start | n | n |  |
| MemAccountKeyed | test-core | 2 | test_mem_account_keyed.cpp | 0.00G | 1.7 | Keyed memory accounting: free uncharges exactly charged bytes, unknown pointer no-op | n | n |  |
| MemAccountRanges | test-core | 1 | test_memory_allocators.cpp | 0.00G | 4.8 | Memory account ranges: listed until their pool drops them, counts and bytes, EXPECT | n | n |  |
| MeminfoParseTest | test-core | 2 | test_weight_snapshot.cpp | 0.00G | 4.0 | meminfo parse: MemAvailable, MemTotal and cgroup limit, EXPECT_EQ on parsed values | n | n |  |
| MemoryPlan | test-core | 25 | test_memory_plan.cpp | 0.00G | 9.5 | Memory plan battery: dense reference fits, library reserve line item, KV residual, tight | n | n |  |
| ModelContext | test-core | 5 | test_model_context.cpp | 0.00G | 3.3 | TIMING: Model context length: configured wins, declared capped at auto ceiling, config | n | n |  |
| ModelIdForPath | test-core | 1 | test_model_name_policy.cpp | 0.00G | 5.1 | Model id from path: HF snapshot dir maps to repo id, EXPECT_EQ | n | n |  |
| ModelLifetimeGate | test-core | 8 | test_model_lifetime_gate.cpp | 0.08G | 3.1 | TIMING: Model lifetime gate: exclusive waits for readers, timeout stays usable, stale | n | n |  |
| ModelNamePolicy | test-core | 5 | test_model_name_policy.cpp | 0.00G | 5.1 | Model name policy: basename looked up in models dir, repo-id shape, absolute/traversal | n | n |  |
| ModelProfileTable | test-core | 4 | test_model_profile_table.cpp | 0.00G | 2.7 | Model profile table: one row per arch, SWA windows, loader trait lists, no-SWA mask | n | n |  |
| ModelScan | test-core | 4 | test_model_name_policy.cpp | 0.00G | 5.1 | Model scan lists GGUF and safetensors dirs, symlink loops and unreadable dirs tolerated | n | n |  |
| ModelSwapState | test-core | 4 | test_model_swap_state.cpp | 0.10G | 20.6 | Model swap state: swap or suspend under a lease answers 503 and frees nothing | n | n |  |
| ModeloptMixedPrecision | test-core | 5 | test_llm_compressor_loader.cpp | 0.00G | 9.3 | ModelOpt mixed-precision table: NVFP4 table accepted, missing or non-NVFP4 refused | n | n |  |
| MoeGroupLimit | test-core | 4 | test_moe_group_limit.cpp | 0.00G | 2.5 | MoE group-limited routing matches Ling gate golden, stable under fp16 bias | n | n |  |
| MoeLaunchRows | test-core | 4 | test_routing_decision.cpp | 0.00G | 3.4 | MoE launch rows: one-hot takes all rows, skew ratio, empty inputs zero | n | n |  |
| MoeNvfp4CacheRule | test-core | 4 | test_moe_nvfp4_cache_rule.cpp | 0.00G | 1.3 | MoE NVFP4 whole-layer cache rule: lone eligible down not cached, dense not cached | n | n |  |
| MoePrefillSelect | test-core | 2 | test_moe_prefill_select.cpp | 0.00G | 2.2 | MoE prefill: fused dp4a and IMMA qkind refuse unsupported qtypes, EXPECT_THROW | n | n |  |
| MoePrefillTable | test-core | 7 | test_routing_decision.cpp | 0.00G | 3.4 | MoE prefill path table: device args, GPT-OSS staged/unstaged, grouped and legacy fallback | n | n |  |
| MoeRoutingModelCoupling | test-core | 2 | test_routing_decision.cpp | 0.00G | 3.4 | MoE prefill path replay: every tier replays to itself, GPT-OSS to grouped | n | n |  |
| MoeTierPrecedence | test-core | 3 | test_routing_decision.cpp | 0.00G | 3.4 | MoE prefill tier precedence: device args over grouped over legacy, entry gate first | n | n |  |
| MtpAttnGuard | test-core | 1 | test_mtp_attn_guard.cpp | 0.00G | 2.8 | MTP attention guard: zero KV heads with weights returns false with a log message | n | n |  |
| MtpAuto | test-core | 28 | test_mtp_auto.cpp | 0.00G | 4.3 | MTP auto sentinel: default auto, single-stream and concurrent pair/pool choices, 28 cases | n | n |  |
| MtpDraftRows | test-core | 4 | test_mtp_draft_rows.cpp | 0.00G | 1.5 | MTP draft rows: copy holds prefix or every row, FP8 head and cutlass head vocab | n | n |  |
| MtpExternalHead | test-core | 1 | test_mtp_presence.cpp | 0.00G | 4.4 | MTP external head attaches only a matching Qwen head file | n | n |  |
| MtpHeadGraph | test-core | 7 | test_mtp_head_graph.cpp | 0.00G | 2.2 | MTP head graph: eager first use, capture then replay, ring window and resume rules | n | n |  |
| MtpKvSlots | test-core | 2 | test_mtp_kv_slots.cpp | 0.00G | 2.7 | MTP KV slots: slot committed at first use, refused commit reported, ASSERT_NE | n | n |  |
| MtpLayoutDispatch | test-core | 5 | test_mtp_layout_dispatch.cpp | 0.00G | 3.6 | MTP layout dispatch: Qwen4-Exp 310 names mapped, missing expert scale incomplete | n | n |  |
| MtpNameRule | test-core | 4 | test_mtp_presence.cpp | 0.00G | 4.4 | MTP name rule: both checkpoint spellings accepted, main-model tensors rejected | n | n |  |
| MtpProbe | test-core | 5 | test_mtp_presence.cpp | 0.00G | 4.4 | MTP probe: sidecar, sharded and single-file layouts found, non-head and no-head quiet | n | n |  |
| MtpSiblingHead | test-core | 1 | test_mtp_presence.cpp | 0.00G | 4.4 | MTP sibling head completes only from a byte-equal neighbour file | n | n |  |
| MtpWorkspaceBytes | test-core | 2 | test_memory_plan.cpp | 0.00G | 9.5 | MTP workspace bytes: Qwen3.8 head at server defaults, NVFP4 body copy priced for dense | n | n |  |
| Mxfp8Widen | test-core | 4 | test_mxfp8_widen.cpp | 0.00G | 2.7 | MXFP8 widen: element decode exact, pair to bf16, NaN and NVFP4 scales skipped, E4M3 | n | n |  |
| NGramContext | test-core | 4 | test_ngram_context.cpp | 0.00G | 1.5 | N-gram context: EOS at sequence start, spans prompt and output, sequences isolated | n | n |  |
| NGramHash | test-core | 4 | test_ngram_table.cpp | 0.00G | 2.1 | N-gram hash: EOS-filled start, shift blocked by EOS, carried equals one-shot, Python | n | n |  |
| NgramDraft | test-core | 8 | test_ngram_draft.cpp | 0.00G | 2.1 | N-gram draft: empty without repeat, simple repeat, truncates at K, prefers longer and | n | n |  |
| NonStreamReasoningTokens | test-core | 5 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Non-stream reasoning token count: between markers, prompt opener from token zero, fallback | n | n |  |
| NpyWriterTest | test-core | 4 | test_npy_writer.cpp | 0.00G | 4.5 | NPY writer: parseable header, spec-padded header, row-major payload, shape for one row | n | n |  |
| NvFP4ExpertKey | test-core | 1 | test_nvfp4_expert_offload.cpp | 0.00G | 2.4 | NVFP4 expert key matcher: per-expert MoE weights only, EXPECT_TRUE | n | n |  |
| NvFP4HostExpertsServable | test-core | 2 | test_nvfp4_expert_offload.cpp | 0.00G | 2.4 | NVFP4 host experts servable: promoted host projection accepted, staging-unusable refused | n | n |  |
| NvFP4IgnoreList | test-core | 5 | test_nvfp4_quant_contract.cpp | 0.00G | 9.4 | NVFP4 ignore list enforced, missing global scale refused, inventory counts match | n | n |  |
| NvFP4MergedScaleProvenance | test-core | 7 | test_nvfp4_quant_contract.cpp | 0.00G | 9.4 | NVFP4 merged-scale provenance: fused split recorded, separate tensors no split, slice | n | n |  |
| NvFP4SlotLayout | test-core | 5 | test_nvfp4_expert_offload.cpp | 0.00G | 2.4 | NVFP4 slot layout: expert ranges, uint2 load alignment, scale inside slot, no overlap | n | n |  |
| Nvfp4DequantWs | test-core | 4 | test_nvfp4_dequant_ws.cpp | 0.00G | 1.7 | NVFP4 dequant workspace: none when absent or too small, deferred take runs once | n | n |  |
| Nvfp4OutputBytes | test-core | 2 | test_quantize_policy.cpp | 0.00G | 4.5 | NVFP4 output bytes: packed nibbles, micro scales, tensor scale, empty matrix refused | n | n |  |
| Nvfp4ScratchKey | test-core | 1 | test_nvfp4_quant_contract.cpp | 0.00G | 9.4 | NVFP4 scratch key: malformed index resolves to nothing, EXPECT_EQ | n | n |  |
| Nvfp4Undeclared | test-core | 1 | test_flint.cpp | 0.00G | 3.7 | NVFP4 undeclared module: packed codes without declaration refused and named | n | n |  |
| OffloadSpecDefaults | test-core | 5 | test_offload_spec_defaults.cpp | 0.00G | 2.0 | Offload spec defaults: offload turns recycling on at width 8, operator keys win, MC rows | n | n |  |
| OutOfRangeMessage | test-core | 2 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Out-of-range message: penalty kernel bounds refused, bounds and defaults accepted | n | n |  |
| OutputMarkersInkling | test-core | 8 | test_output_markers.cpp | 0.00G | 2.3 | Inkling output markers: think tags kept only, tool-call header dropped, args renamed | n | n |  |
| PagedDecodeArgs | test-core | 2 | test_paged_q8_epilogue.cpp | 0.00G | 2.5 | Paged decode args struct: fields in declaration order, omitted fields default, EXPECT_EQ | n | n |  |
| PagedDecodeTensorArgs | test-core | 1 | test_paged_q8_epilogue.cpp | 0.00G | 2.5 | Paged decode tensor args: omitted fields take defaults, kv_scale 1.0, EXPECT_EQ | n | n |  |
| PagedFp8Decode | test-core | 2 | test_spec_request.cpp | 0.00G | 2.5 | Paged FP8 decode fast kernel only at head dim 128, serving vs fast question | n | n |  |
| PagedQ8Epilogue | test-core | 3 | test_paged_q8_epilogue.cpp | 0.00G | 2.5 | Paged Q8 epilogue taken once per call, not outside its shape, nothing armed never taken | n | n |  |
| PagedSplitKScratch | test-core | 3 | test_paged_q8_epilogue.cpp | 0.00G | 2.5 | Paged split-K scratch bytes per split, null scratch never splits, small scratch caps | n | n |  |
| ParseCompletionPrompt | test-core | 2 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Completion prompt parse: string, ids and one-element lists accepted, batches refused by | n | n |  |
| ParseGemmaToolCallBody | test-core | 1 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Gemma tool-call body parser rejects malformed bodies, EXPECT_FALSE | n | n |  |
| ParseLogitBias | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Logit-bias parse: absent empty, valid kept, malformed entries error not drop, cap refuses | n | n |  |
| ParseMaxTokensField | test-core | 5 | test_sse_stream_utils.cpp | 0.00G | 31.4 | max_tokens parse: default, max_tokens and max_completion_tokens precedence, nulls ignored | n | n |  |
| ParseStopField | test-core | 5 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Stop field parse: string or array up to 16, truncation reported, non-strings skipped | n | n |  |
| ParseStreamToolBody | test-core | 2 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Stream tool-call body parse: Gemma body, JSON then XML fallback, failure | n | n |  |
| PathWithin | test-core | 4 | test_model_name_policy.cpp | 0.00G | 5.1 | path_within: inside vs outside, sibling prefix outside, traversal resolved before compare | n | n |  |
| PhaseFixture | test-core | 26 | test_memory_backend.cpp | 0.09G | 3.5 | Phase fixture: region release on destruction, move ownership once, reset idempotent | n | n |  |
| PhraseBan | test-core | 6 | test_phrase_ban.cpp | 0.00G | 3.1 | Phrase ban: match ends in new text only, token-at-byte, snapshot pick, regex battery | n | n |  |
| PinnedBuffer | test-core | 13 | test_host_pinned.cpp | 0.00G | 3.5 | Pinned buffer RAII: frees exactly once, failed acquire is empty, zero bytes refused, move | n | n |  |
| PreDequantBudget | test-core | 4 | test_pre_dequant_budget.cpp | 0.00G | 2.0 | Pre-dequant budget: NVFP4 reservation charged to early phases only, KV growth once, clamp | n | n |  |
| PrefillGraphCache | test-core | 3 | test_prefill_graph_cache.cpp | 0.00G | 2.2 | Prefill graph cache: short keys keep a runner each, long keys share one on consecutive | n | n |  |
| PrefillPacing | test-core | 7 | test_prefill_pacing.cpp | 0.00G | 1.7 | Prefill pacing: uncapped with no decoder, cap off, weight arithmetic, protection scenario | n | n |  |
| ProcessDiag | test-core | 6 | test_process_diag.cpp | 0.00G | 2.5 | Process diagnostics install: mirrored fields transferred, defaults, log level applied | n | n |  |
| PromptLogprobs | test-core | 1 | test_prompt_logprobs.cpp | 0.00G | 10.7 | Prompt logprobs chunk rows formula, EXPECT_EQ against max chunk rows | n | n |  |
| PromptLogprobsAssembly | test-core | 7 | test_prompt_logprobs.cpp | 0.00G | 10.7 | Prompt logprobs assembly: vLLM and echo shapes, limit cuts alternatives not the target | n | n |  |
| PromptLogprobsParse | test-core | 5 | test_prompt_logprobs.cpp | 0.00G | 10.7 | Prompt logprobs parse: off when absent, range 0-20, stream refused, engine top-N | n | n |  |
| PromptTail | test-core | 7 | test_prompt_tail.cpp | 0.00G | 1.8 | Prompt tail split: issue split keeps 33 rows, reuse cap leaves minimum rows, short prompt | n | n |  |
| QknormRopeArgs | test-core | 2 | test_launcher_args.cpp | 0.00G | 3.3 | QK-norm RoPE args struct: fields in declaration order, omitted fields default | n | n |  |
| QuantReport | test-core | 6 | test_quantize_report.cpp | 0.00G | 2.8 | Quant error report: exact values zero error, off-grid distance, worst tensors ordered | n | n |  |
| QuantizeCheckpointOut | test-core | 28 | test_quantize_checkpoint_out.cpp | 0.00G | 6.1 | Checkpoint writer: format names, per-format tensor names, global scale, fused group match | n | n |  |
| QuantizeFp8Head | test-core | 6 | test_quantize_checkpoint_out.cpp | 0.00G | 6.1 | FP8 head writer: modes and defaults, writes codes and row scales under loader names | n | n |  |
| QuantizeKvHint | test-core | 2 | test_quantize_checkpoint_out.cpp | 0.00G | 6.1 | KV FP8 hint allowlist covers measured families only, loader reads the writer hint | n | n |  |
| QuantizePolicy | test-core | 7 | test_quantize_policy.cpp | 0.00G | 4.5 | Quantize policy: linear weights taken, MLA latent and router refused, embeddings kept | n | n |  |
| QuantizeRecipe | test-core | 5 | test_quantize_recipe.cpp | 0.00G | 3.3 | Quantize recipe JSON: fixed field order, uncalibrated null, same recipe same bytes | n | n |  |
| QueuedTokenGate | test-core | 7 | test_server_request_limits.cpp | 0.00G | 26.5 | TIMING: Queued-token gate: admits up to cap, refuses over, lease releases once, concurrent | n | n |  |
| Qwen3VLVisionConfig | test-core | 7 | test_qwen3vl_vision_config.cpp | 0.00G | 2.5 | Qwen3-VL vision config: real checkpoint parse, inconsistent geometry and deepstack refused | n | n |  |
| Qwen3VLVisionGrid | test-core | 10 | test_qwen3vl_vision_grid.cpp | 0.00G | 2.5 | Qwen3-VL vision grid: merge-block token order, patch coverage, interpolation weights | n | n |  |
| Qwen3VLVisionLoad | test-core | 9 | test_qwen3vl_vision_load.cpp | 0.00G | 3.0 | Qwen3-VL vision load: real checkpoint loads, null slot or shape mismatch refused | n | n |  |
| Qwen3VLVisionMap | test-core | 7 | test_qwen3vl_vision_map.cpp | 0.00G | 2.8 | Qwen3-VL vision tensor map: every real tensor recognised once, malformed index unknown | n | n |  |
| QwenPatchify | test-core | 4 | test_vision_preprocess.cpp | 0.00G | 4.7 | Qwen patchify: grid and shape, merge-block grouping, channel-major temporal repeat | n | n |  |
| QwenSmartResize | test-core | 7 | test_vision_preprocess.cpp | 0.00G | 4.7 | Qwen smart-resize: aligned unchanged, min/max pixel scaling, ties round to even | n | n |  |
| QwenVideoFold | test-core | 1 | test_qwen3vl_video.cpp | 0.00G | 5.4 | Qwen video fold: pads rewritten and token types kept, EXPECT_EQ | n | n |  |
| QwenVideoLayout | test-core | 7 | test_qwen3vl_video.cpp | 0.00G | 5.4 | Qwen video layout: grids, timestamps, placeholder ids and rope positions match HF | n | n |  |
| QwenVideoPatchify | test-core | 4 | test_qwen3vl_video.cpp | 0.00G | 5.4 | Qwen video patchify matches HF video processor within one fp16 ulp, bad input refused | n | n |  |
| QwenVideoSmartResize | test-core | 1 | test_qwen3vl_video.cpp | 0.00G | 5.4 | Qwen video smart-resize matches HF for stored vectors, EXPECT_EQ | n | n |  |
| RateLimitKey | test-core | 3 | test_server_request_limits.cpp | 0.00G | 26.5 | Rate-limit key: forwarded-for ignored from untrusted peer, trusted proxy honoured | n | n |  |
| RateLimiter | test-core | 3 | test_server_request_limits.cpp | 0.00G | 26.5 | TIMING: Rate limiter: quiet buckets evicted, window slides, refuses past limit, admits | n | n |  |
| ReasoningFinishDetail | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Reasoning finish detail: fires only when answer lost to reasoning, field written only then | n | n |  |
| RecurrentSlotForReset | test-core | 3 | test_recurrent_slot.cpp | 0.00G | 2.0 | Recurrent slot for reset: known id returns its slot, unknown id resets nothing | n | n |  |
| RegexConstrain | test-core | 15 | test_regex_constrain.cpp | 0.00G | 3.0 | Regex constrainer: accepts only matching continuation, multi-char tokens, alternation | n | n |  |
| RegexNfaCostBounds | test-core | 4 | test_regex_constrain.cpp | 0.00G | 3.0 | Regex NFA cost bounds: huge repeats, nested repeats, deep groups refused, ordinary | n | n |  |
| RequestTables | test-core | 7 | test_request_tables.cpp | 0.00G | 6.1 | Request tables: eight tables in finish order, cancel leaves no entry, release idempotent | n | n |  |
| ResponseStore | test-core | 8 | test_responses_store.cpp | 0.00G | 7.1 | Response store: put/get round-trip, TTL, purge, entry and byte caps evict LRU | n | n |  |
| ResponsesContinue | test-core | 2 | test_responses_store.cpp | 0.00G | 7.1 | Responses continue: merged input equals stateless resend, chaining, string input | n | n |  |
| ResponsesPrefill | test-core | 1 | test_assistant_prefill_transform.cpp | 0.00G | 4.5 | Responses prefill: continue_final_message forwarded, EXPECT_TRUE/EXPECT_FALSE | n | n |  |
| ResponsesTransform | test-core | 34 | test_responses_transform.cpp | 0.00G | 16.7 | Responses-to-chat transform: input items, tool round-trip, tool nesting, format knobs | n | n |  |
| RopeOverrideTest | test-core | 10 | test_rope_override.cpp | 0.00G | 2.8 | RoPE override: off by default, YaRN and linear injection, explicit origin ctx, refusals | n | n |  |
| RopeScalingConfigTest | test-core | 18 | test_hf_config_loader.cpp | 0.00G | 6.6 | RoPE scaling config: Llama3 factor table, tie-embeddings tri-state, per-arch layer maps | n | n |  |
| RoutingMtpOwnersTest | test-core | 2 | test_routing_mtp_owners.cpp | 0.00G | 2.4 | Routing and MTP owners: buffers released once, move leaves no view, free clears views | n | n |  |
| RuntimeConfigTest | test-core | 24 | test_config.cpp | 0.00G | 4.7 | Runtime config defaults and bindings: KV block size, lm-dequant bisect, SWA tri-state | n | n |  |
| SSMStateLazy | test-core | 3 | test_memory_allocators.cpp | 0.00G | 4.8 | SSM state lazy: reserves every slot commits none, ensure commits only that slot | n | n |  |
| Sae | test-core | 1 | test_sae.cpp | 0.00G | 3.1 | SAE token info: top by activation then id, relative error, EXPECT_FLOAT_EQ | n | n |  |
| SaeDir | test-core | 2 | test_sae.cpp | 0.00G | 3.1 | SAE directory load: layer and tensors load, mismatches refused | n | n |  |
| SafeTensorsAwq | test-core | 3 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors AWQ: unsupported variants refused with detected config, 4-bit zero-point load | n | n |  |
| SafeTensorsDtypeTable | test-core | 2 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors dtype table: servable rows have matching widths, mismatched wire types refused | n | n |  |
| SafeTensorsGptq | test-core | 5 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors GPTQ: unknown format refused, pre-fix qzeros refused, auto-GPTQ v1/v2 offsets | n | n |  |
| SafeTensorsHighDimTensors | test-core | 1 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors high-dim tensors flattened not dropped, conv5d in the captured log | n | n |  |
| SafeTensorsHostileHeader | test-core | 6 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors hostile header: bad dtype, huge offset, negative/overflow shapes dropped | n | n |  |
| SafeTensorsMalformedEntryWarnings | test-core | 2 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors malformed entries: missing dtype/shape and byte-count mismatch warn | n | n |  |
| SafeTensorsRaw | test-core | 2 | test_safetensors_writer.cpp | 0.00G | 3.2 | SafeTensors raw writer/reader round-trip, truncated and bogus files rejected | n | n |  |
| SafeTensorsValidateHeaderSize | test-core | 6 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors header size validation: below 8 bytes, exceeds file, uint64 overflow, soft cap | n | n |  |
| SafeTensorsValidateTensorOffsets | test-core | 7 | test_safetensors_loader.cpp | 0.00G | 4.8 | SafeTensors tensor offset validation: start/end past file, byte mismatch, zero-size valid | n | n |  |
| SafeTensorsWriter | test-core | 5 | test_safetensors_writer.cpp | 0.00G | 3.2 | SafeTensors writer: round-trip of bytes and header, contiguous offsets, escaped names | n | n |  |
| SamplingDefaultsParity | test-core | 4 | test_sampling_defaults_parity.cpp | 0.00G | 4.0 | Sampling defaults parity: CLI equals server per arch, Qwen preset, generation_config | n | n |  |
| SamplingParamsGolden | test-core | 1 | test_sampling_params_golden.cpp | 0.02G | 33.6 | Sampling params golden: every dialect matches the recorded parse, golden file loads | n | n |  |
| SanitizeForEcho | test-core | 3 | test_server_request_limits.cpp | 0.00G | 26.5 | sanitize_for_echo: non-printable replaced, truncation marker, result serialises | n | n |  |
| SchemaAdditionalProperties | test-core | 10 | test_schema_constrain_property.cpp | 0.00G | 6.4 | JSON-schema additionalProperties: explicit true, free-form objects, malformed JSON | n | n |  |
| SchemaConstrainPropertyTest | test-core | 8 | test_schema_constrain_property.cpp | 0.02G | 6.4 | JSON-schema constrainer property battery: conforming docs and prefixes accepted, missing | n | n |  |
| SchemaDepthCap | test-core | 2 | test_schema_constrain_property.cpp | 0.00G | 6.4 | JSON-schema depth cap: deep nesting refused, moderate nesting parses | n | n |  |
| SchemaEnumLiteral | test-core | 2 | test_schema_constrain_property.cpp | 0.00G | 6.4 | JSON-schema enum literals: numbers, booleans, null and mixed legal-token checks | n | n |  |
| SchemaEnvelope | test-core | 1 | test_schema_constrain_property.cpp | 0.00G | 6.4 | Schema envelope: forced open bounds leading whitespace, token_legal battery | n | n |  |
| SchemaIntegerBound | test-core | 3 | test_schema_constrain_property.cpp | 0.00G | 6.4 | Schema integer bound: digits stop at int64 width, short ints unaffected, number unbounded | n | n |  |
| SchemaKeywords | test-core | 12 | test_schema_keywords.cpp | 0.00G | 4.0 | Schema allOf/anyOf/optional keywords: value parsed not a string, EXPECT_EQ | n | n |  |
| SchemaKeywordsProperty | test-core | 2 | test_schema_keywords.cpp | 0.01G | 4.0 | Schema keywords property: conforming docs complete, every prefix legal, no dead end | n | n |  |
| SchemaParserDesync | test-core | 6 | test_schema_constrain_property.cpp | 0.00G | 6.4 | Schema parser desync: additionalProperties variants, enum literals, trailing input error | n | n |  |
| SchemaUnenforceableKeywords | test-core | 3 | test_schema_constrain_property.cpp | 0.00G | 6.4 | Schema unenforceable assertion keywords refused, annotations ignored, const as enum | n | n |  |
| SchemaUnicodeEscape | test-core | 4 | test_schema_constrain_property.cpp | 0.00G | 6.4 | Schema JSON unicode escapes: BMP, surrogate pairs, 3-byte escapes decode to UTF-8 | n | n |  |
| SentencePieceLoader | test-core | 10 | test_sentencepiece_loader.cpp | 0.00G | 3.1 | SentencePiece model parse: pieces, scores, trainer spec, truncated protobuf refused | n | n |  |
| ServableContext | test-core | 5 | test_server_request_limits.cpp | 0.00G | 26.5 | Servable context: pool wins when smaller, plan stands when pool larger, growable ceiling | n | n |  |
| ServerArgs | test-core | 7 | test_server_args.cpp | 0.00G | 3.2 | Server args defaults and parses: idle unload, responses store, VRAM budget, limits | n | n |  |
| ServerArgsDeathTest | test-core | 3 | test_server_args.cpp | 0.08G | 3.2 | Server args death test: garbage values exit via EXPECT_EXIT | n | n |  |
| ServerJsonKeyOrder | test-core | 3 | test_server_json_key_order.cpp | 0.00G | 7.2 | Server JSON key order kept through parse/dump for tool schemas and tool_use input | n | n |  |
| ServerVisionInline | test-core | 5 | test_server_vision_inline.cpp | 0.00G | 3.8 | Vision inline placement: blocks land where parts sat, count mismatch strips and reports | n | n |  |
| ServingMetadataLayout | test-core | 4 | test_serving_metadata_layout.cpp | 0.00G | 2.1 | Serving metadata layout: 256-byte aligned disjoint regions, size bounds, SWA off adds | n | n |  |
| ShadowPlan | test-core | 16 | test_memory_plan.cpp | 0.00G | 9.5 | Shadow plan: reports both pool ceilings, capped admission sizes KV, zero-KV names context | n | n |  |
| ShouldQuantize | test-core | 1 | test_quantize_policy.cpp | 0.00G | 4.5 | Quantize filter leaves the MTP draft head alone, EXPECT_FALSE | n | n |  |
| Sm120FmhaTileSelect | test-core | 3 | test_fmha_sm120_tile_select.cpp | 0.00G | 2.2 | SM120 FMHA tile selection: every selected tile has an instance, old holes gone | n | n |  |
| SmallMXqTag | test-core | 3 | test_smallm_xq_tag.cpp | 0.00G | 1.4 | Small-M MXFP quantize tag: producer stamp live until buffer touched, consumer adopts own | n | n |  |
| SnapshotHostDefault | test-core | 4 | test_snapshot_host_default.cpp | 0.00G | 2.2 | Snapshot host default: spill off keeps default, spill raises host tier to half budget | n | n |  |
| Span | test-core | 1 | test_memory_allocators.cpp | 0.00G | 4.8 | Span: stable widens to device span, subspans inherit the guarantee, arena-backed | n | n |  |
| SparseGeometry | test-core | 7 | test_sparse_attn_geometry.cpp | 0.00G | 2.1 | Sparse attention geometry: budget, engage threshold, sink and recent windows per block | n | n |  |
| SpecAcceptRows | test-core | 4 | test_spec_gates.cpp | 0.00G | 2.9 | Speculative accept rows: unfinished chunk keeps bonus outside rows, think edge, stop ends | n | n |  |
| SpecBatchRr | test-core | 3 | test_spec_request.cpp | 0.00G | 2.5 | Spec batch round-robin verify active only with a batch and a non-recurrent model | n | n |  |
| SpecField | test-core | 7 | test_spec_usage.cpp | 0.00G | 23.2 | Speculative request field: absent leaves unset, bool meaning kept, depth object, range 400 | n | n |  |
| SpecGates | test-core | 12 | test_spec_gates.cpp | 0.00G | 2.9 | Speculative gates: MTP, n-gram, token recycling, DSpark enable independently, nothing | n | n |  |
| SpecRequest | test-core | 10 | test_spec_request.cpp | 0.00G | 2.5 | Speculative request resolve: unset takes server default, request lowers depth, false wins | n | n |  |
| SpecUsage | test-core | 4 | test_spec_usage.cpp | 0.00G | 23.2 | Spec usage block: absent without verify step, counters beside details, decline reported | n | n |  |
| SpecUsageKeys | test-core | 2 | test_spec_usage.cpp | 0.00G | 23.2 | Spec usage keys: table matches chat shape, copy moves present keys only, EXPECT_EQ | n | n |  |
| SpecialTokensMapTest | test-core | 3 | test_hf_config_loader.cpp | 0.00G | 6.6 | special_tokens_map.json: Mistral object form and Qwen string form parse, missing file | n | n |  |
| SseChunk | test-core | 5 | test_sse_stream_utils.cpp | 0.00G | 31.4 | SSE chunk JSON: envelope, finish reason and logprobs, completion shape, finish detail | n | n |  |
| SseWriter | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | SSE writer frames: content and reasoning keys, id/model escaped, no NUL byte in frame | n | n |  |
| SsmScanPrefillArgs | test-core | 1 | test_launcher_args.cpp | 0.00G | 3.3 | SSM scan prefill args struct: fields in declaration order, dtype fields, EXPECT_EQ | n | n |  |
| SsmStatePool | test-core | 4 | test_memory_plan.cpp | 0.00G | 9.5 | SSM state pool: Qwen3.8 geometry sized for allocator, tail alignment, failure fatal only | n | n |  |
| SsrfClassifier | test-core | 9 | test_image_fetch.cpp | 0.00G | 2.6 | SSRF classifier: loopback, cloud metadata, private ranges, v4-mapped and unspecified | n | n |  |
| SsrfFetch | test-core | 5 | test_image_fetch.cpp | 0.00G | 2.6 | SSRF fetch: remote fetch off by default, private hosts refused even when enabled, scheme | n | n |  |
| StackedExperts | test-core | 3 | test_quantize_policy.cpp | 0.00G | 4.5 | Stacked experts found under each naming convention, 3-D and non-float tensors ignored | n | n |  |
| StepSeed | test-core | 2 | test_step_seed.cpp | 0.00G | 4.9 | Step seed: seeded request adds output count, unseeded keeps one base, EXPECT_EQ 10 | n | n |  |
| StopMask | test-core | 7 | test_think_stop_logic.cpp | 0.00G | 4.9 | Stop mask: masks in think with budget and during grace, released on content or hard cap | n | n |  |
| StreamEquality | test-core | 2 | test_stream_pipeline.cpp | 0.00G | 4.0 | Streaming equals non-stream: concat equals full decode, stop truncation matches non-stream | n | n |  |
| StreamMultibyteSim | test-core | 1 | test_stream_pipeline.cpp | 0.00G | 4.0 | Stream multibyte: split codepoint never emitted as half, UTF-8 buffer empty at end | n | n |  |
| StreamReasoningSplit | test-core | 21 | test_sse_stream_utils.cpp,test_stream_reasoning_split.cpp | 0.00G | 34.7 | Stream reasoning split: second think pass stays out of content, reentry across pieces | n | n |  |
| StreamSim | test-core | 1 | test_stream_pipeline.cpp | 0.00G | 4.0 | Stream sim: stop match split across chunks is caught, emitted text and stopped flag | n | n |  |
| StrictCliNumbers | test-core | 2 | test_server_args.cpp | 0.00G | 3.2 | Strict CLI numbers: whole value or refused, valid values still parse | n | n |  |
| StrictCliNumbersDeathTest | test-core | 1 | test_server_args.cpp | 0.05G | 3.2 | Strict CLI numbers death test: malformed value exits 1 naming the flag | n | n |  |
| StrictToolConstraint | test-core | 5 | test_tool_call.cpp | 0.00G | 16.5 | Strict tool constraint: loose tool does not disable strict one, unnamed tool refused | n | n |  |
| StripChannelHeaders | test-core | 1 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Strip channel headers: header tokens removed, body kept, EXPECT_EQ | n | n |  |
| StripThinkBlock | test-core | 4 | test_reasoning_reconcile.cpp,test_sse_stream_utils.cpp | 0.00G | 51.6 | Strip think block: unclosed trailing or leading think cleared, block removed, answer kept | n | n |  |
| StructuredOutputThinking | test-core | 5 | test_reasoning_reconcile.cpp | 0.00G | 20.2 | Structured output thinking: whole-reply constraint excludes thinking, unconstrained may | n | n |  |
| SuffixDraft | test-core | 17 | test_suffix_draft.cpp | 0.00G | 2.5 | Suffix draft: empty without repeat, repeat found, majority vote beats recency, truncation | n | n |  |
| SuffixShadow | test-core | 5 | test_suffix_shadow.cpp | 0.00G | 2.4 | Suffix shadow: line waits for outcome, full agreement no draft, declined step replaced | n | n |  |
| SystemFingerprint | test-core | 2 | test_server_request_limits.cpp | 0.00G | 26.5 | system_fingerprint stable per model, differs across models, empty name still yields one | n | n |  |
| TensorKindMatcher | test-core | 8 | test_tensor_kind_matcher.cpp | 0.00G | 2.1 | Tensor kind matcher: attention, FFN, MoE, GDN, norms, embeddings, unknown names | n | n |  |
| TensorKindTable | test-core | 7 | test_tensor_kind_table.cpp | 0.00G | 3.0 | Tensor kind table: entry per kind, GDN and norms restricted to FP16/FP32 tiers | n | n |  |
| TensorTest | test-core | 10 | test_tensor.cpp | 0.00G | 2.7 | Tensor basics: default, numel, nbytes, strides, reshape, dtype names, scalar | n | n |  |
| TextThink | test-core | 5 | test_think_stop_logic.cpp | 0.00G | 4.9 | Text think-state machine: enter/exit across split tokens, marker survives window eviction | n | n |  |
| ThinkCloser | test-core | 1 | test_think_stop_logic.cpp | 0.00G | 4.9 | Think closer: message end after answer does not mask stop, grace holds stop back | n | n |  |
| ThinkLimit | test-core | 5 | test_think_stop_logic.cpp | 0.00G | 4.9 | Think limit: graph and host paths agree, reserve follows config, cap bounds default | n | n |  |
| ThinkSeedFromPromptTail | test-core | 4 | test_think_stop_logic.cpp | 0.00G | 4.9 | Think seed from prompt tail: trailing opener seeds both flags, closed block seeds neither | n | n |  |
| ThinkTokenAccept | test-core | 6 | test_think_stop_logic.cpp | 0.00G | 4.9 | Think token accept: control and user-defined types accepted, normal and absent rejected | n | n |  |
| ThinkingReconcile | test-core | 7 | test_reasoning_reconcile.cpp | 0.00G | 20.2 | Thinking reconcile: open prefix turns thinking on, closed block defaults off, explicit | n | n |  |
| TokenCategory | test-core | 2 | test_json_constrain_property.cpp | 0.00G | 6.9 | Token category: non-ASCII counts as string content, string-ending tokens carry quote | n | n |  |
| TokenExtras | test-core | 2 | test_token_extras.cpp | 0.00G | 4.0 | Token extras: handler keeps worker copy of logprobs as request grows, no copy without them | n | n |  |
| TokenRecycle | test-core | 17 | test_token_recycle_draft.cpp | 0.00G | 2.5 | Token recycling draft: pair chain, most recent pair wins, eviction, top-k rank order, 17 | n | n |  |
| TokenSpansTest | test-core | 5 | test_logprobs_shapes.cpp | 0.00G | 22.7 | Token spans: flush attributes pieces on token boundary, remainder rebased, untracked bytes | n | n |  |
| TokenizerFlagsTest | test-core | 5 | test_hf_config_loader.cpp | 0.00G | 6.6 | Tokenizer flags from tokenizer_config: add_bos, Mistral BOS/EOS, default system prompt | n | n |  |
| ToolArgCoerce | test-core | 1 | test_tool_call_glm.cpp | 0.00G | 10.6 | Tool-argument coercion matches vLLM coerce-to-schema-type table, golden rows | n | n |  |
| ToolCallChatML | test-core | 6 | test_tool_call.cpp | 0.00G | 16.5 | ChatML tool-call parse: single and sequential calls, arguments key, malformed body, drift | n | n |  |
| ToolCallCohere | test-core | 1 | test_tool_call.cpp | 0.00G | 16.5 | Cohere action array parses through the ChatML parser into two named calls | n | n |  |
| ToolCallDialectGolden | test-core | 1 | test_tool_call_dialect_golden.cpp | 0.01G | 8.8 | Tool-call dialect golden: all families match the recorded fixture | n | n |  |
| ToolCallDialectRegistry | test-core | 2 | test_tool_call_dialect_registry.cpp | 0.00G | 3.9 | Tool-call dialect registry: every family matches pre-registry switches, ChatML default | n | n |  |
| ToolCallDispatch | test-core | 1 | test_tool_call.cpp | 0.00G | 16.5 | Tool-call dispatch routes each family to its parser, EXPECT_EQ on name | n | n |  |
| ToolCallFault | test-core | 4 | test_tool_call_xml_params.cpp | 0.00G | 10.3 | Tool-call fault line: clean call has none, raw text on request, raw capped at 2000 bytes | n | n |  |
| ToolCallGemma | test-core | 3 | test_tool_call.cpp | 0.00G | 16.5 | Gemma tool-call parse: single call, no call returns content, digit-leading token kept | n | n |  |
| ToolCallGlm | test-core | 5 | test_tool_call_glm.cpp | 0.00G | 10.6 | GLM tool-call parse: arg-key body, fault line raw body, pairs match vLLM finditer | n | n |  |
| ToolCallHarmony | test-core | 8 | test_tool_call.cpp | 0.00G | 16.5 | Harmony tool-call parse: recipient channel becomes call, truncated call parses, final not | n | n |  |
| ToolCallLlama3 | test-core | 8 | test_tool_call.cpp | 0.00G | 16.5 | Llama-3 tool-call parse: single, bare JSON object, plain JSON answer not a call | n | n |  |
| ToolCallQwen36Xml | test-core | 21 | test_tool_call.cpp,test_tool_call_xml_params.cpp | 0.00G | 26.8 | Qwen3.6 XML tool-call parse: parameter coercion, string params, no-arg call, 21 cases | n | n |  |
| ToolCallReconstruct | test-core | 7 | test_tool_call.cpp | 0.00G | 16.5 | Tool-call reconstruct: content kept, each dialect re-wraps parsed call, no-function entry | n | n |  |
| ToolCallValidate | test-core | 6 | test_tool_call.cpp | 0.00G | 16.5 | Tool-call args validate against schema: required, type, non-object, unknown tool no-op | n | n |  |
| ToolChoice | test-core | 5 | test_constraint_validation.cpp | 0.00G | 33.1 | tool_choice validation: named tool must exist, required without tools refused | n | n |  |
| ToolChoiceEnforcement | test-core | 10 | test_tool_call.cpp | 0.00G | 16.5 | tool_choice enforcement per family: named and required enforced only where supported | n | n |  |
| ToolNamesFromRequest | test-core | 1 | test_tool_call.cpp | 0.00G | 16.5 | Tool names from request: OpenAI and bare shapes extracted, EXPECT_EQ | n | n |  |
| ToolResponseFormat | test-core | 6 | test_tool_call.cpp | 0.00G | 16.5 | Tool response format: content passthrough, non-string serialised, text parts joined | n | n |  |
| ToolResultImages | test-core | 4 | test_tool_result_images.cpp | 0.00G | 6.4 | Tool-result images: picture moves to a new user turn behind the tool run, ordering rules | n | n |  |
| ToolStreamFilterChatML | test-core | 18 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Tool stream filter ChatML: chunked JSON body, args streamed before close, escapes | n | n |  |
| ToolStreamFilterCohere | test-core | 1 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Tool stream filter Cohere: action array yields one call per element across chunk sizes | n | n |  |
| ToolStreamFilterGemma | test-core | 5 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Tool stream filter Gemma: char-by-char, whole string, random chunks, two calls in one feed | n | n |  |
| ToolStreamFilterLlama3 | test-core | 2 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Tool stream filter Llama-3: args stream incrementally, chunked input yields same call | n | n |  |
| ToolStreamFilterUtf8 | test-core | 4 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Tool stream filter UTF-8: argument deltas never end mid-codepoint, 4-byte chars survive | n | n |  |
| ToolTagScan | test-core | 2 | test_tool_stream_filter.cpp | 0.00G | 15.4 | Tool tag scan: Gemma marker prefixes held back, ChatML and Llama-3 unchanged | n | n |  |
| TracingTest | test-core | 7 | test_tracing.cpp | 0.00G | 8.0 | Tracing: traceparent accepted only in well-formed version 00, random ids width and | n | n |  |
| TypeTablesGolden | test-core | 2 | test_type_tables_golden.cpp | 0.00G | 2.1 | Wire and QType type tables match golden for every id, element bytes and names | n | n |  |
| UploadExceedsCheckpoint | test-core | 4 | test_kv_residual_sizing.cpp | 0.00G | 2.5 | Upload exceeds checkpoint: measured pair, quarter boundary inclusive, unknown size not a | n | n |  |
| UploadReserve | test-core | 7 | test_upload_reserve.cpp | 0.00G | 4.5 | Upload reserve reproduces logged 64-slot value, Qwen3.8 batch kept by trimming KV, KV | n | n |  |
| Utf8 | test-core | 3 | test_stream_pipeline.cpp | 0.00G | 4.0 | utf8_complete_len: complete ASCII and multibyte, truncated trailing sequence held back | n | n |  |
| Utf8ChunkLen | test-core | 4 | test_sse_stream_utils.cpp | 0.00G | 31.4 | UTF-8 chunk length never splits a multibyte character, chunks reassemble, always | n | n |  |
| Utf8Stitch | test-core | 5 | test_sse_stream_utils.cpp | 0.00G | 31.4 | UTF-8 stitch rejoins split characters across pieces, invalid lead not held back forever | n | n |  |
| VideoEvs | test-core | 12 | test_video_evs.cpp | 0.00G | 4.2 | Video EVS pruning: retained count floors at one pair, ties keep lower index, pad layout | n | n |  |
| ViewOrOwnedTest | test-core | 10 | test_view_or_owned.cpp | 0.00G | 3.4 | View-or-owned: view set/reset never releases, owned replace releases once, idempotent | n | n |  |
| VisionCacheKeys | test-core | 5 | test_vision_cache_keys.cpp | 0.00G | 2.0 | Vision cache keys: text before first picture keys as text, new picture leaves old keys | n | n |  |
| VisionFamily | test-core | 4 | test_vision_family.cpp | 0.00G | 2.9 | Vision family registry: exact names map, loadable only with encoder, probe follows | n | n |  |
| VisionLoaderCheck | test-core | 12 | test_vision_loader_check.cpp | 0.00G | 2.4 | Vision loader check: Qwen3-VL merger refused, Gemma3/4v tower complete, fused QKV caught | n | n |  |
| VisionPreprocess | test-core | 6 | test_vision_preprocess.cpp | 0.00G | 4.7 | Vision preprocess: CHW layout, channel order, mean/std, resize to square, garbage fails | n | n |  |
| VisionPromptLayout | test-core | 1 | test_vision_prompt_layout.cpp | 0.00G | 2.7 | Vision prompt layout matches golden recorded on main, EXPECT_EQ over model types and cases | n | n |  |
| VramReserveFloor | test-core | 2 | test_kv_residual_sizing.cpp | 0.00G | 2.5 | VRAM reserve floor: default equals allocator headroom, policy percent only raises it | n | n |  |
| WeightCacheEntriesTest | test-core | 12 | test_weight_cache_entries.cpp | 0.00G | 4.7 | Weight cache entries: FP16 move leaves source empty, map erase releases once, NVFP4 owner | n | n |  |
| WeightKeyTest | test-core | 1 | test_weight_snapshot.cpp | 0.00G | 4.0 | Weight key canonical formats: L<layer>.<name> strings, EXPECT_STREQ | n | n |  |
| WeightMapExpertGuard | test-core | 5 | test_weight_map_expert_gate.cpp | 0.00G | 3.4 | Weight map expert guard: up/down-only MoE accepted, refused without any expert, dense | n | n |  |
| WeightMapModalitySkips | test-core | 7 | test_weight_map_modality_skips.cpp | 0.00G | 4.2 | Weight map modality skips: audio and vision counted apart, text-only load skips nothing | n | n |  |
| WeightMapOutNormWarn | test-core | 3 | test_weight_map_out_norm_warn.cpp | 0.00G | 3.5 | Weight map out-norm warning: mixer without final norm not warned, missing norm warns | n | n |  |
| WeightOwnedTest | test-core | 10 | test_weight_owned.cpp | 0.00G | 2.4 | Weight owner: scope end frees once on adoption stream, reset idempotent, move transfers | n | n |  |
| WeightPageRelease | test-core | 4 | test_weight_page_release.cpp | 0.00G | 2.8 | Weight page release: data still reads back after release, split shards released | n | n |  |
| WeightRecordIndicesTest | test-core | 1 | test_weight_snapshot.cpp | 0.00G | 4.0 | Weight record indices: file-supplied indices bounds-checked, EXPECT_TRUE | n | n |  |
| WeightSnapshotArmSlotTest | test-core | 1 | test_weight_snapshot.cpp | 0.00G | 4.0 | Weight snapshot arm slot: arm, take, disarm sequence on the global slot, EXPECT_EQ | n | n |  |
| WeightTierEntries | test-core | 2 | test_weight_tier_entries.cpp | 0.00G | 2.9 | Weight tier entries: bind takes source map nodes, every read tier needs its entry | n | n |  |
| WeightUploadLogTest | test-core | 6 | test_weight_snapshot.cpp | 0.00G | 4.0 | Weight upload log: record resolves sizes and offsets, evict marks dead and drops live | n | n |  |
| WeightUploadTraits | test-core | 2 | test_weight_upload_staging.cpp | 0.07G | 4.6 | Weight upload traits: per-format uploaders byte-for-byte equal, one-byte change detected | n | n |  |
| WrongFieldTypeMessage | test-core | 3 | test_sse_stream_utils.cpp | 0.00G | 31.4 | Wrong-field-type message names the first wrong field, valid bodies and nulls not reported | n | n |  |
| AdmissionCapacityTest | test-e2e | 1 | test_api_generate.cpp | 1.28G | 2.7 | E2E: prompt larger than the KV pool returns a typed error, not cancellation | y | y: ran on a hard-coded /models default |  |
| ApiGenerateParityTest | test-e2e | 3 | test_api_generate.cpp | 4.44G | 2.7 | E2E generate API: non-empty output, streaming delivers all tokens, manual prefill loop | y | y: ran on a hard-coded /models default |  |
| BatchBuilderTest | test-e2e | 7 | test_continuous_batching.cpp | 0.00G | 5.2 | Continuous batching builder: decode, prefill, mixed, reset, 16-sequence batches | y | n |  |
| BatchInvarianceTest | test-e2e | 4 | test_batch_invariance.cu | 43.16G | 17.8 | E2E batch invariance: NVFP4 and FP8 decode solo vs batched, neighbour independence | y | y: ran on a hard-coded /models default |  |
| CancelLifecycleHybridTest | test-e2e | 1 | test_cancel_lifecycle.cpp | 0.00G | 5.0 | E2E hybrid model: cancelling a queued request leaves the running sequence state intact | y | y: all 1 skip without a model env | 1 |
| CancelLifecycleTest | test-e2e | 2 | test_cancel_lifecycle.cpp | 0.00G | 5.0 | E2E cancel during a graph burst: waits it out, releases every table, no token after cancel | y | y: all 2 skip without a model env | 2 |
| CandidateTokenGuardTest | test-e2e | 8 | test_candidate_tokens.cpp | 0.00G | 3.2 | Candidate-token guard: single-token letters; refuses merged, duplicate, multi-token | y | y: TOKENIZER_QWEN3 | 8 |
| ChunkedPrefillTest | test-e2e | 7 | test_chunked_prefill.cu | 0.00G | 8.0 | E2E chunked prefill logits equal unchunked: Qwen3-4B, Llama-3.2-3B, long context | y | y: all 7 skip without a model env | 7 |
| DFlashDrafterTest | test-e2e | 1 | test_dspark_drafter.cu | 6.19G | 9.0 | DFlash drafter block output vs reference (e2e, needs model) | y | y: MODEL_DFLASH MODEL_DSPARK MODEL_DSPARK_TARGET |  |
| DSparkDrafterTest | test-e2e | 4 | test_dspark_drafter.cu | 11.90G | 9.0 | DSpark drafter: block vs reference, context rollback, FP8 weights, graph replay length | y | y: MODEL_DFLASH MODEL_DSPARK MODEL_DSPARK_TARGET |  |
| DecodeGraphCtxGrowthTest | test-e2e | 1 | test_e2e_models.cpp | 0.00G | 3.4 | E2E decode graph: long prompt after a short request stays alive (context growth) | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 | 1 |
| DeepStackInject | test-e2e | 5 | test_deepstack_inject.cu | 0.00G | 6.7 | DeepStack injection adds at image tokens only; repeats accumulate; surplus kept | y | n |  |
| DegenerationTest | test-e2e | 6 | test_degeneration.cpp | 13.01G | 3.1 | E2E degeneration: repetition, greedy determinism, graph replay vs eager, no special tokens | y | y: ran on a hard-coded /models default |  |
| DetEvalE2ETest | test-e2e | 10 | test_determinism_e2e.cpp | 0.00G | 3.1 | E2E determinism: greedy and perplexity bit-identical in one context; fresh cases off | y | y: 6 of 10 skip without a model env | 6 |
| DeviceFaultSignalTest | test-e2e | 1 | test_batching_engine_fault.cu | 2.35G | 13.4 | TIMING: Fault inside a batching step cancels requests and refuses later ones | y | y: ran on a hard-coded /models default |  |
| EncoderEmbedTest | test-e2e | 1 | test_encoder_embed.cpp | 0.00G | 3.9 | NomicBert encoder embeddings match reference structure (e2e, GGUF) | y | n | 1 |
| EndToEndModelTest | test-e2e | 8 | test_e2e.cpp | 0.00G | 6.4 | E2E model: load, tokenize, generate, prefill/decode, 8k/16k context, isolation, think-stop | y | y: MODEL MODEL_GDN | 8 |
| EndToEndTest | test-e2e | 7 | test_e2e.cpp | 0.00G | 6.4 | C API smoke: version, config and params defaults, error strings, null arguments | y | n |  |
| EngineIntegrationTest | test-e2e | 12 | test_engine_integration.cu | 0.66G | 16.7 | Engine integration: init, prefill/decode steps, FP8/NVFP4 paths, pipelined batched decode | y | n |  |
| EngineRelaunchTest | test-e2e | 2 | test_engine_relaunch.cpp | 5.70G | 3.2 | Model reload frees VRAM; second engine on the same model handle runs clean | y | y: ran on a hard-coded /models default |  |
| FimRealTokenizer | test-e2e | 2 | test_fim_tokenizer.cpp | 0.00G | 2.7 | FIM tokenizer: Qwen-Coder ids and PSM prompt; non-FIM tokenizer refused | y | y: FIM_TOKENIZER NOFIM_TOKENIZER | 2 |
| ForwardPassTest | test-e2e | 8 | test_forward_pass.cu | 0.80G | 18.9 | Forward-pass logits on a synthetic model: prefill, decode, GQA, batch composition | y | n |  |
| GDNModelTest | test-e2e | 2 | test_e2e_models.cpp | 0.00G | 3.4 | GDN model e2e: coherent output and multi-turn GDN state across turns | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 | 2 |
| GPUBatchTest | test-e2e | 6 | test_continuous_batching.cpp | 0.00G | 5.2 | GPU batch upload and free: sequence uploads, double free, empty batch, varying contexts | y | n |  |
| GdnGraphBucketTest | test-e2e | 2 | test_gdn_graph_buckets.cu | 0.00G | 14.3 | GDN graph buckets: each batch bucket uses its own recurrent slot; recycled slot intact | y | y: MODEL_GDN | 2 |
| Gemma4AudioEncoderTest | test-e2e | 2 | test_gemma4_audio_encoder.cu | 0.00G | 9.3 | Gemma-4 audio encoder: WAV front end and tower/embedder match HF features | y | y: MODEL_GEMMA4_AUDIO | 2 |
| Gemma4E4BRef | test-e2e | 2 | test_e2e_gemma4_e4b.cpp | 0.00G | 2.4 | Gemma-4 E4B greedy 20-token text matches HF reference (capital, sky) | y | y: MODEL_GEMMA4_E4B | 2 |
| Gemma4GraphsTest | test-e2e | 2 | test_e2e_models.cpp | 0.00G | 3.4 | Gemma-4 e2e with CUDA graphs: capital answer and 256-token coherent decode | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 | 2 |
| Gemma4ModelTest | test-e2e | 4 | test_e2e_models.cpp | 0.00G | 3.4 | Gemma-4 e2e: chat answer, raw completion, SWA prefill route, no repetition | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 | 4 |
| Gemma4SafeTensorsTest | test-e2e | 1 | test_e2e_models.cpp | 0.00G | 3.4 | Gemma-4 SafeTensors e2e: needle beyond the sliding window is retrieved | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 | 1 |
| GgufDeadWeightForms | test-e2e | 2 | test_gguf_dead_weight_forms.cpp | 0.00G | 6.0 | GGUF dead weight: dense GGUF carries no scale atom; NVFP4 served head frees its source | y | n | 2 |
| GreedyLockTest | test-e2e | 3 | test_e2e_greedy_lock.cpp | 0.00G | 3.0 | E2E greedy lock: frozen sequences; teacher-forced decode feeds and scores forced tokens | y | y: MODEL | 3 |
| HybridBatchedDecodeTest | test-e2e | 3 | test_hybrid_batched_decode.cu | 0.00G | 15.2 | Hybrid batched decode: row independence, batched vs solo logit delta, think-close repeats | y | y: all 3 skip without a model env | 3 |
| HybridRestoreChainTest | test-e2e | 8 | test_hybrid_restore_chain.cu | 0.00G | 17.3 | Hybrid snapshot restore chain: restored state stays close to a fresh run (e2e) | y | y: all 8 skip without a model env | 8 |
| HybridSharedScoreTest | test-e2e | 1 | test_hybrid_shared_score.cu | 0.00G | 15.1 | Hybrid shared-score snapshots: rows restoring one snapshot are independent across waves | y | y: all 1 skip without a model env | 1 |
| InternVLEncoder | test-e2e | 2 | test_internvl_encoder.cu | 0.01G | 10.7 | InternVL vision encoder per-stage and full tower output vs HF FP32 reference | y | y: MODEL_INTERNVL | 1 |
| KimiTiktokenParityTest | test-e2e | 4 | test_tokenizer_kimi.cpp | 0.00G | 2.7 | Kimi tiktoken tokenizer: fingerprint, encode equals HF on all cases, round trip, vocab | y | y: MODEL_KIMI | 4 |
| Ling3Ref | test-e2e | 2 | test_e2e_ling3.cpp | 0.00G | 2.3 | Ling-3 e2e: greedy capital answer and chunked prefill match HF reference | y | y: MODEL_LING3 | 2 |
| LlmCompressorE2E | test-e2e | 4 | test_e2e_llm_compressor.cpp | 8.85G | 3.1 | LLM-compressor/ModelOpt checkpoints (Gemma-4, Mistral Small, Qwen-Coder): load, generate | y | y: 2 of 4 skip without a model env | 2 |
| LoraHotSwap | test-e2e | 3 | test_lora.cpp | 0.00G | 5.0 | TIMING: LoRA hot swap: zero adapter is identity, shapes vs model, prefix-cache key | y | y: all 3 skip without a model env | 3 |
| Mamba2BatchedDecodeTest | test-e2e | 1 | test_hybrid_batched_decode.cu | 0.00G | 15.2 | Mamba2 batched decode: rows decode independently of their order (hybrid e2e) | y | y: all 1 skip without a model env | 1 |
| MoePrefixHitE2ETest | test-e2e | 2 | test_prefix_cache_e2e.cpp | 0.00G | 7.2 | MoE prefix-cache hit at two offsets is bit-equal to a fresh run (e2e) | y | y: MODEL MODEL_GDN | 2 |
| MtpGreedyIdentityTest | test-e2e | 6 | test_mtp_greedy_identity.cpp | 111.48G | 6.0 | Speculative decoding (MTP, n-gram, DSpark, DFlash) keeps constrained greedy tokens | y | y: MODEL_DFLASH MODEL_DSPARK MODEL_DSPARK_TARGET MODEL_MTP |  |
| MtpProbe | test-e2e | 1 | test_real_checkpoints.cpp | 0.00G | 2.3 | MTP head is found in a real checkpoint (env-gated probe) | y | y: IMP_MTP_MODEL |  |
| MtpQwen4ExpReference | test-e2e | 1 | test_mtp_qwen4exp_reference.cpp | 33.98G | 7.8 | MTP two draft steps match vLLM math on bf16 rows (reference test) | y | y: MODEL_MTP_QWEN4EXP |  |
| Nvfp4PrefillCacheKnobTest | test-e2e | 1 | test_nvfp4_prefill_cache_knob.cpp | 2.08G | 3.7 | Decode knob keeps the CUTLASS NVFP4 prefill cache | y | n |  |
| PrefixCacheE2ETest | test-e2e | 8 | test_prefix_cache_e2e.cpp | 0.00G | 7.2 | Prefix cache e2e: hit equals fresh run; hybrid snapshot and FP8-KV restores | y | y: MODEL MODEL_GDN | 8 |
| PrimaryModelTest | test-e2e | 4 | test_e2e_models.cpp | 0.00G | 3.4 | Primary model e2e: coherent generation, multi-turn, tokenize roundtrip, prefill/decode | y | y: MODEL MODEL_GDN MODEL_GEMMA4 MODEL_GEMMA4_NVFP4 | 4 |
| PromptLogprobsE2ETest | test-e2e | 1 | test_prompt_logprobs_e2e.cpp | 0.00G | 4.2 | Prompt logprobs e2e: chunk logits released after the pass; VRAM allocator | y | y: MODEL | 1 |
| Q8DropSourceE2E | test-e2e | 1 | test_q8_drop_source_e2e.cpp | 0.00G | 5.9 | Q8 weight source dropped after load: greedy decode still runs; VRAM checked | y | n | 1 |
| QuantPipelineTest | test-e2e | 1 | test_quant_pipeline.cpp | 0.00G | 2.1 | Quant pipeline build populates decode caches for coherent decode | y | y: all 1 skip without a model env | 1 |
| Qwen38TokenizerParityTest | test-e2e | 4 | test_tokenizer_qwen38.cpp | 0.00G | 2.6 | Qwen3.8 tokenizer: fingerprint, encode equals HF on all cases, round trip, vocab size | y | y: MODEL_QWEN38 | 4 |
| Qwen3VLEncoder | test-e2e | 2 | test_qwen3vl_encoder.cu | 0.13G | 9.5 | Qwen3-VL vision encoder vs independent CPU reference, including attention chunks | y | n |  |
| Qwen3VLPipelineTest | test-e2e | 6 | test_qwen3vl_pipeline.cu | 0.00G | 10.2 | Qwen3-VL image pipeline: reserved bytes, finite deterministic encodings, HF video frames | y | y: MODEL_QWEN3VL | 6 |
| Qwen3VLPipelineVideoStamps | test-e2e | 1 | test_qwen3vl_pipeline.cu | 0.00G | 10.2 | Qwen3-VL video timestamp ids match HF | y | y: MODEL_QWEN3VL | 1 |
| RaggedPrefillTest | test-e2e | 6 | test_prefill_ragged.cu | 0.35G | 18.8 | Ragged prefill: matches serial dense, per-layer KV heads, repeats, chunked, mixed decode | y | n |  |
| RequestTest | test-e2e | 5 | test_continuous_batching.cpp,test_scheduler_prefix_bypass.cpp | 0.00G | 9.6 | Request context_len, status transitions, default state; prefix and ragged admission rules | y | n |  |
| SamplerRowBansTest | test-e2e | 1 | test_sampler_row_bans.cu | 0.00G | 16.1 | Stashed sampler rows keep their own ban list | y | y: all 1 skip without a model env | 1 |
| SchedulerGpuTest | test-e2e | 1 | test_scheduler.cpp | 0.00G | 6.6 | GPU scheduler: pool grows under aggregate memory pressure | y | n |  |
| SchedulerTest | test-e2e | 43 | test_scheduler.cpp,test_scheduler_embedding_prefix.cpp,test_scheduler_kv_decode.cpp,test_scheduler_prefix_bypass.cpp | 0.02G | 18.3 | Scheduler: admission gate, memory-aware admission, prefill/decode priority, cancel, aging | y | n |  |
| SentencePieceLoader | test-e2e | 1 | test_real_checkpoints.cpp | 0.00G | 2.3 | Real HF-cached SentencePiece model loads cleanly (env-gated) | y | y: IMP_MTP_MODEL | 1 |
| ServingSignalsTest | test-e2e | 8 | test_serving_signals.cpp | 172.22G | 6.1 | TIMING: Serving signals e2e: KV exhaustion, valve arming, host swap, queue time | y | y: ran on a hard-coded /models default |  |
| SpecCaptureFidelityTest | test-e2e | 2 | test_spec_capture_fidelity.cpp | 54.46G | 4.2 | Cached-graph speculation forward equals eager; hybrid speculation does not collapse | y | y: MODEL_SPEC_FIDELITY MODEL_SSM |  |
| StoragePlanner | test-e2e | 14 | test_weight_registry_preservation.cpp | 0.00G | 9.7 | Weight storage planner: tier plans under budget, NVFP4 tiers, dual-path hints, VRAM sums | y | n |  |
| StubModelTest | test-e2e | 5 | test_e2e.cpp | 0.13G | 6.4 | Stub GGUF model e2e: load, tokenize, create context and infer, VRAM leak detection | y | n |  |
| SuspendResumeTest | test-e2e | 3 | test_suspend_resume.cu | 14.74G | 9.1 | Suspend/resume: snapshot round trip is token-identical; armed mismatch is ignored | y | y: ran on a hard-coded /models default |  |
| TensorKindCoverage | test-e2e | 1 | test_tensor_kind_coverage.cpp | 0.00G | 2.7 | Small Qwen GGUF: no unknown tensor kinds | y | y: all 1 skip without a model env | 1 |
| TokenizerCompatTest | test-e2e | 1 | test_tokenizer_compat.cpp | 0.00G | 2.9 | Tokenizer matches HF on the loaded GGUF and SafeTensors model | y | y: all 1 skip without a model env | 1 |
| VideoEvsKernels | test-e2e | 2 | test_video_evs_kernels.cu | 0.00G | 6.9 | Video EVS kernels: dissimilarity matches host cosine; gather compacts kept rows in order | y | n |  |
| VisionChunkOffset | test-e2e | 5 | test_vision_chunk_offset.cu | 0.00G | 7.0 | Vision embeddings chunk offset: second chunk, DeepStack offset, chunked equals unchunked | y | n |  |
| VisionGolden | test-e2e | 2 | test_vision_golden.cu | 0.00G | 7.3 | Vision encoder golden outputs for Gemma3 SigLIP and Gemma4v | y | y: all 2 skip without a model env | 2 |
| VramBudgetReserve | test-e2e | 21 | test_vram_budget_reserve.cpp | 0.00G | 10.1 | VRAM budget planner: KV pool, weight-cache room, reserve floor, IMMA planes, budget | y | n |  |
| VramLedgerE2E | test-e2e | 1 | test_vram_ledger_e2e.cpp | 0.00G | 5.8 | E2E VRAM ledger: MXFP4 GDN native holds no FP16 cache | y | y: all 1 skip without a model env | 1 |
| WarmCacheTest | test-e2e | 3 | test_warm_cache.cu | 16.20G | 8.1 | Warm weight cache: cold load writes it, warm boot token-identical; corrupt cache ignored | y | y: ran on a hard-coded /models default |  |
| WeightRegistryPreservation | test-e2e | 1 | test_weight_registry_preservation.cpp | 0.00G | 9.7 | Weight registry: NVFP4 mode keeps SSM in/out tensors; tier plans and VRAM sums | y | n |  |
| HfFetchNetwork | test-hf-network | 3 | test_hf_fetch_network.cpp | n/r | 2.6 | HF fetch (network): GGUF fetch verifies SHA then hits cache; gated repo fails cleanly | n | n |  |
| CaptureAbort | test-kv | 3 | test_capture_abort.cu | 0.00G | 7.9 | Graph capture abort: throw leaves stream usable; first replay failure still runs step | y | n |  |
| DeviceAllocAllGpu | test-kv | 3 | test_device_alloc_all.cu | 0.01G | 8.4 | Device alloc-all on GPU: async takes all buffers, release returns them, failure is clean | y | n |  |
| DeviceOwnedGpu | test-kv | 3 | test_device_alloc_all.cu | 0.00G | 8.4 | DeviceOwned ledger: tracked bytes per allocator family (plain, async, allocator) | y | n |  |
| FP8KVCache | test-kv | 9 | test_fp8_kv_cache.cu | 0.49G | 10.8 | FP8 KV cache: construction, scale calibration, block bytes, split-K and HD256 vs FP16 | y | n |  |
| Fp8KvStaticScale | test-kv | 6 | test_fp8_kv_static_scale.cu | 0.00G | 10.7 | FP8 KV static scale: E4M3 bound, power-of-two grid, in-place round trip, split scales | y | n |  |
| INT8KVCache | test-kv | 2 | test_fp8_kv_cache.cu | 0.01G | 10.8 | INT8 KV cache: construction and split-K consistency | y | n |  |
| KVBlockCopy | test-kv | 4 | test_kv_block_copy.cu | 0.00G | 8.0 | KV block copy: FP16/FP8 fan-out, key min/max metadata copied, untouched block intact | y | n |  |
| KVCacheGrowAdmissionTest | test-kv | 2 | test_kv_cache_grow_admission_gpu.cpp | 0.09G | 3.5 | TIMING: Growable KV pool: capped reclaim without per-block VRAM read; grow-ahead | y | n |  |
| KVCacheGrowTest | test-kv | 7 | test_kv_cache_gpu.cpp | 0.07G | 6.9 | TIMING: Growable KV pool: growth stops at headroom, ceiling raise, grow-ahead, scale plane | y | n |  |
| KVCacheIdleReleaseTest | test-kv | 2 | test_kv_cache_grow_admission_gpu.cpp | 0.14G | 3.5 | KV idle release: free tail goes back, cached prefix and referenced tail stay | y | n |  |
| KVCacheManagerGrowTest | test-kv | 1 | test_kv_cache_gpu.cpp | 0.00G | 6.9 | KV manager grows before reclaiming cached blocks | y | n |  |
| KVCacheManagerTest | test-kv | 7 | test_kv_cache_gpu.cpp | 0.01G | 6.9 | KV manager: fingerprint gate, reclaim count, SWA rollback, snapshot pack/restore, clone | y | n |  |
| KVCacheTest | test-kv | 10 | test_kv_cache_gpu.cpp | 1.36G | 6.9 | KV cache pointers, read/write, NVFP4 layout, SWA block pointers, residency probe, swap | y | n |  |
| KVCacheWriteTest | test-kv | 4 | test_kv_cache_write.cu | 0.00G | 10.1 | KV cache write: paged write, fused K/V write, batched block table, INT8 per-head scale | y | n |  |
| KVGatherTest | test-kv | 5 | test_kv_gather.cu | 0.00G | 9.8 | KV gather paged to flat: FP16, FP8 dequant, INT8 roundtrip, sentinel zero fill | y | n |  |
| KVHostSpillGpuTest | test-kv | 27 | test_kv_host_spill_gpu.cu | 0.09G | 11.7 | Host KV spill: reclaimed prefix restores bit-exact incl key min/max; snapshot chain | y | n |  |
| MtpUploadBudgetTest | test-kv | 3 | test_mtp_upload_budget.cpp | 0.00G | 2.4 | MTP upload peak bytes: packed layout equals file size; per-expert slabs; uncounted experts | y | n |  |
| PrefixEquivTest | test-kv | 13 | test_prefix_cache_equiv.cpp | 0.00G | 4.0 | Prefix cache equivalence: reuse counts, shared blocks, eviction, chain holes, salt, probe | y | n |  |
| PrefixPersistTest | test-kv | 1 | test_prefix_cache_equiv.cpp | 0.00G | 4.0 | Prefix cache persistence: quantized KV restores its scales | y | n |  |
| RecurrentSnapshotStoreTest | test-kv | 13 | test_recurrent_snapshot_store.cpp | 0.34G | 5.3 | TIMING: Recurrent snapshot store: host-tier eviction and restore, LRU, protected entries | y | n |  |
| SSMStateLazyGpu | test-kv | 2 | test_kv_cache_gpu.cpp | 0.03G | 6.9 | SSM state lazy GPU: slot commit backs one stride; decommitted slots return to the driver | y | n |  |
| TouchAuditGpu | test-kv | 1 | test_touch_audit.cu | 0.00G | 8.0 | Touch audit: pages written since the snapshot counted exactly | y | n |  |
| VmmBackendTest | test-kv | 8 | test_vmm_backend.cu | 0.07G | 8.8 | VMM backend: reservation costs no memory, growth into ceiling, zeroed blocks, fixed pools | y | n |  |
| VramFits | test-kv | 1 | test_vram_query.cpp | 0.00G | 2.8 | VRAM fit check: size plus margin compared with free plus credit | y | n |  |
| VramLedger | test-kv | 2 | test_vram_query.cpp | 0.00G | 2.8 | VRAM ledger: plain allocation moves live bytes and free; unknown-pointer free is a no-op | y | n |  |
| VramQuerySnapshot | test-kv | 2 | test_vram_query.cpp | 0.00G | 2.8 | VRAM query snapshot: free at install capped at budget; pending reservations not counted | y | n |  |
| VramQueryTest | test-kv | 4 | test_vram_query.cpp | 0.01G | 2.8 | VRAM budget query: uncapped passthrough, total capped by budget, exhaustion clamps to zero | y | n |  |
| ActivationCalibratorExperts | test-moe-gdn | 1 | test_activation_calibrator_experts.cu | 0.00G | 7.1 | Activation calibrator: one entry per routed expert segment | y | n |  |
| ExpertCacheLayerMask | test-moe-gdn | 1 | test_expert_cache.cu | 0.00G | 7.8 | Expert cache layer mask: only cached layers get rows; setup and teardown | y | n |  |
| ExpertCachePhase3Test | test-moe-gdn | 11 | test_expert_cache.cu | 0.03G | 7.8 | Expert cache LRU: slot mirror, same-layer eviction order, per-layer isolation, hit moves | y | n |  |
| ExpertCacheReinitTest | test-moe-gdn | 3 | test_expert_cache.cu | 0.00G | 7.8 | Expert cache reinit: grown budget takes; failure falls back; both fail disables | y | n |  |
| GDNScanTest | test-moe-gdn | 14 | test_gdn.cu | 0.23G | 8.9 | GDN scan CPU vs GPU: single/multi token, fused vs legacy, chunk handoff, zero state | y | n | 2 |
| GdnBatchedScanTest | test-moe-gdn | 12 | test_gdn_batched.cu | 0.08G | 9.9 | Batched GDN scan bit-identical at batch 1, 8, 32, non-contiguous slots; conv1d decode | y | n |  |
| GdnRaggedRidersTest | test-moe-gdn | 1 | test_gdn_ragged_riders.cu | 0.05G | 9.4 | GDN ragged riders: decode riders and prompts match separate solo runs | y | n |  |
| GdnVerifySnapshotTest | test-moe-gdn | 7 | test_gdn_verify_snapshot.cu | 0.01G | 8.1 | GDN verify snapshot writes: fused and chunkwise F32/BF16; second slab one row later | y | n |  |
| GdnVerifyTwoDraftsTest | test-moe-gdn | 12 | test_gdn_verify_two_drafts.cu | 0.98G | 10.2 | GDN verify two-draft scan and conv factor rows fold draft states | y | n |  |
| GptOssMoeFusedBias | test-moe-gdn | 4 | test_gpt_oss_moe_fused_bias.cu | 0.04G | 8.5 | gpt-oss MoE fused bias: GLU quantize and scatter-down bias byte-identical to launches | y | n |  |
| GrammarConstrainMaskTest | test-moe-gdn | 1 | test_grammar_regex_mask.cu | 0.00G | 7.2 | Grammar regex mask: model vocab larger than tokenizer masks padding | y | n |  |
| JsonConstrainFsmTest | test-moe-gdn | 5 | test_json_constrain.cu | 0.00G | 10.2 | JSON constrained FSM: nested docs accepted; mismatched closers, trailing content rejected | y | n |  |
| JsonConstrainTest | test-moe-gdn | 12 | test_json_constrain.cu | 0.00G | 10.2 | JSON constrain: token classes, valid-token mask, control chars masked, allow-list init | y | n |  |
| KdaScan | test-moe-gdn | 7 | test_kda.cu | 0.14G | 9.3 | KDA scan (Kimi-Linear): recurrent and chunk vs reference, softplus gate, ragged, bf16 | y | n |  |
| MoEDeterministicPermute | test-moe-gdn | 1 | test_moe.cu | 0.00G | 9.9 | MoE deterministic permute: expert layout matches the serial rule | y | n |  |
| MoEExecutorTest | test-moe-gdn | 12 | test_moe_executor.cu | 0.33G | 21.9 | MoE executor e2e: init, valid forward, deterministic, bounded logit drift, mixed MoE/dense | y | n |  |
| MoEMultiCtaPermute | test-moe-gdn | 1 | test_moe.cu | 0.05G | 9.9 | MoE multi-CTA permute bit-exact vs the single-CTA kernel | y | n |  |
| MoERoutingEdgeTest | test-moe-gdn | 2 | test_moe.cu | 0.00G | 9.9 | MoE routing edges: empty expert, all tokens routed to one expert | y | n |  |
| MoERoutingTest | test-moe-gdn | 7 | test_moe.cu | 0.51G | 9.9 | MoE routing: top-k, weight normalization, expert offsets, sorted ids, gather/scatter | y | n |  |
| MoERoutingWideTest | test-moe-gdn | 2 | test_moe.cu | 0.12G | 9.9 | MoE routing with more than 256 experts: candidate set and softmax top-k per launch | y | n |  |
| MoeGroupTopk | test-moe-gdn | 1 | test_moe_group_topk.cu | 0.00G | 6.9 | MoE group top-k kernel matches the Ling gate reference | y | n |  |
| MoeImbalanceKernel | test-moe-gdn | 3 | test_moe.cu | 0.00G | 9.9 | MoE imbalance kernel: one-hot expert takes all rows; skewed routing vs host; many experts | y | n |  |
| MtpBodyNvfp4Test | test-moe-gdn | 2 | test_mtp_body_nvfp4.cu | 0.03G | 8.7 | MTP body NVFP4 copy: direct quantize per weight; one-row and multi-row feeds read the copy | y | n |  |
| MtpFeedBatchTest | test-moe-gdn | 11 | test_mtp_feed_batch.cu | 4.59G | 11.0 | MTP feed batch: batched vs per-pair and multi-slot feeds, packed tables, captured head | y | n |  |
| PLEBatched | test-moe-gdn | 3 | test_ple_batched.cu | 0.00G | 7.7 | PLE batched decode: rows match per-sequence bitwise; neighbours do not leak; prefill chunk | y | n |  |
| PreambleGateTest | test-moe-gdn | 29 | test_json_constrain.cu | 0.00G | 10.2 | JSON preamble gate: off by default; budget-only mode; thinking absorb; tool transitions | y | n |  |
| RegexConstrainMaskTest | test-moe-gdn | 1 | test_grammar_regex_mask.cu | 0.00G | 7.2 | Regex constrain mask: model vocab larger than tokenizer masks the padding | y | n |  |
| RegexNfaTest | test-moe-gdn | 4 | test_schema_constrain.cu | 0.00G | 10.6 | Regex NFA: literals, char-class ranges, counted repeats, anchors | y | n |  |
| SSMConv1dTest | test-moe-gdn | 10 | test_ssm.cu | 0.08G | 8.3 | SSM conv1d: decode shift-convolve, vectorised bit-exact, prefill causal, chunked prefill | y | n |  |
| SSMConvSplitTest | test-moe-gdn | 2 | test_ssm_conv_split.cu | 0.06G | 6.8 | SSM conv split: bit-identical to separate kernels; uncovered shapes refused | y | n |  |
| SSMScanSsdTest | test-moe-gdn | 2 | test_ssm_scan_ssd.cu | 0.44G | 7.2 | SSM scan SSD: matches double reference; uncovered shapes refused | y | n |  |
| SSMScanTest | test-moe-gdn | 5 | test_ssm.cu,test_ssm_scan_reg.cu | 0.37G | 16.1 | SSM scan: padded chunk length, prefill snapshot state, register scan vs legacy and double | y | n |  |
| SchemaConstrainTest | test-moe-gdn | 32 | test_schema_constrain.cu,test_tool_call_constrain.cu | 0.00G | 23.0 | JSON schema constrain: patterns, enums, item bounds, refs, recursion; tool calls, XML | y | n |  |
| SsmDecodeBatchedKernelTest | test-moe-gdn | 3 | test_ssm_batched_decode.cu | 0.01G | 7.0 | SSM batched decode: conv and scan rows match single-sequence on their slot (F16/F32) | y | n |  |
| BatchedSmallM | test-quant | 3 | test_nvfp4_batched_smallm_equiv.cu | 0.10G | 9.8 | NVFP4 batched small-M GEMM: rows match the single-row call; matches dequantised reference | y | n |  |
| DequantAwqGpu | test-quant | 1 | test_dequant_awq_gpu.cu | 0.04G | 7.4 | AWQ GPU dequant kernel bit-equal to host reference | y | n |  |
| DequantGptqGpu | test-quant | 1 | test_dequant_gptq_gpu.cu | 0.13G | 7.5 | GPTQ GPU dequant kernel bit-equal to host reference | y | n |  |
| DualPathQuant | test-quant | 2 | test_quant_integration.cu | 0.00G | 26.0 | Dual-path quant: attention weights excluded from NVFP4; weight-cache flag | y | n |  |
| FmhaVLoadBench | test-quant | 4 | test_fmha_v_load_bench.cu | 0.16G | 6.0 | BENCH: cp.async vs TMA bulk V-load attention timing (times kernels, prints numbers) | y | n |  |
| Fp4PvBenchTest | test-quant | 4 | test_fp4_pv_bench.cu | 0.25G | 7.6 | BENCH: FP4 PV attention accuracy and blockscale vs HMMA throughput (times kernels) | y | n |  |
| Fp8ActRowsTest | test-quant | 3 | test_fp8_act_rows.cu | 0.17G | 8.6 | FP8 activation row quantize matches host rule; row bits independent of neighbours | y | n |  |
| Fp8GemmSmallMBench | test-quant | 1 | test_fp8_gemm_smallm.cu | 0.00G | 10.1 | BENCH: FP8 small-M GEMM vs cuBLASLt (DISABLED, times kernels, prints numbers) | y | n |  |
| Fp8GemmSmallMTest | test-quant | 6 | test_fp8_gemm_smallm.cu | 4.40G | 10.1 | FP8 small-M GEMM vs fp64 per M, beta, row-bit invariance, cuBLASLt match, refusals | y | n |  |
| Fp8GemmTileBench | test-quant | 1 | test_gemm_fp8_tile.cu | 0.00G | 10.7 | BENCH: FP8 tile GEMM vs cuBLASLt (DISABLED, times kernels, prints numbers) | y | n |  |
| Fp8GemmTileRefusal | test-quant | 1 | test_gemm_fp8_tile.cu | 0.00G | 10.7 | FP8 tile GEMM refuses illegal shapes and leaves the output untouched | y | n |  |
| Fp8GemmTileTest | test-quant | 14 | test_gemm_fp8_tile.cu | 3.18G | 10.7 | FP8 tile GEMM vs cuBLASLt; beta=1 accumulates | y | n |  |
| Fp8GemvFold | test-quant | 2 | test_fp8_gemv_fold.cu | 1.02G | 9.4 | FP8 GEMV with folded input and residual emit matches gemv_fp8 and host norm | y | n |  |
| GemmNvfp4Batched | test-quant | 5 | test_nvfp4_gemm_batched.cu | 1.45G | 7.9 | NVFP4 batched GEMM vs dequant reference at attn, FFN, down, narrow dims; accumulate | y | n |  |
| GemvKparLoopRepro | test-quant | 2 | test_nvfp4_gemv_kpar_loop.cu | 0.01G | 7.9 | NVFP4 KPAR GEMV: per-row loop equals single call at Gemma-4 gate/up/down dims | y | n |  |
| GgufDequant | test-quant | 14 | test_gguf_dequant_ref.cu | 0.00G | 14.1 | GGUF dequant, all scale modes, vs reference (K=512 coverage) | y | n |  |
| GgufDequantCoverage | test-quant | 1 | test_gguf_dequant_ref.cu | 0.00G | 14.1 | Every supported GGUF format has a dequant reference | y | n |  |
| GgufDequantKQuants | test-quant | 1 | test_gguf_dequant_ref.cu | 0.00G | 14.1 | GGUF K-quant dequant: tail, CTA and unaligned destination vs reference | y | n |  |
| GgufRef | test-quant | 23 | test_gguf_dequant_ref.cu | 0.78G | 14.1 | GGUF GEMV references: Q8_0/Q6_K/Q5_1 dp4a, MoE decode dispatch, MMVQ, dense dp4a rows | y | n |  |
| GptOssMxfp4ConvertRef | test-quant | 6 | test_gpt_oss_mxfp4_convert_ref.cu | 0.00G | 9.7 | gpt-oss MXFP4 convert: down, gate, up slices, extra scale, clamp outlier, zero nibbles | y | n |  |
| GptqNvfp4 | test-quant | 4 | test_gptq_nvfp4.cu | 0.20G | 11.2 | GPTQ-to-NVFP4: beats round-to-nearest on correlated inputs; identity Hessian; Cholesky | y | n |  |
| GroupedNvfp4ArgsK | test-quant | 2 | test_grouped_nvfp4_args.cu | 0.01G | 10.7 | Grouped NVFP4 device args equal host args | y | n |  |
| GroupedNvfp4ArgsTest | test-quant | 1 | test_grouped_nvfp4_args.cu | 0.01G | 10.7 | Grouped NVFP4 host args match per-expert dense GEMM | y | n |  |
| GroupedNvfp4DecodeTest | test-quant | 3 | test_gemm_grouped_nvfp4_decode.cu | 5.61G | 11.1 | Grouped NVFP4 decode: F16 bit-equal to tile and fp64; act epilogue equal to unfused; bench | y | n |  |
| GroupedNvfp4RefManyExperts | test-quant | 2 | test_grouped_nvfp4_ref.cu | 0.71G | 10.3 | Grouped NVFP4 reference with many experts vs fp64 | y | n |  |
| GroupedNvfp4RefSmallK | test-quant | 4 | test_grouped_nvfp4_ref.cu | 0.02G | 10.3 | Grouped NVFP4 reference at small K vs fp64 | y | n |  |
| GroupedNvfp4RefTest | test-quant | 3 | test_grouped_nvfp4_ref.cu | 0.02G | 10.3 | Grouped NVFP4 reference: even, empty-expert boundary, single active expert distributions | y | n |  |
| GroupedNvfp4TileK | test-quant | 6 | test_gemm_grouped_nvfp4_tile.cu | 0.79G | 11.2 | Grouped NVFP4 tile GEMM vs fp64 reference | y | n |  |
| GroupedNvfp4TileTest | test-quant | 1 | test_gemm_grouped_nvfp4_tile.cu | 0.16G | 11.2 | BENCH: grouped NVFP4 tile GEMM wall-time bench (times kernels, prints numbers) | y | n |  |
| LmHeadDraftRows | test-quant | 1 | test_lm_head_draft_rows.cu | 0.19G | 7.2 | NVFP4 multi-row LM-head draft: row-count invariant, cost priced (timed) | y | n |  |
| MmaPeakSaturatedTest | test-quant | 1 | test_mma_peak_saturated.cu | 0.02G | 7.2 | MMA peak-saturated kernel calibration invariants (times kernels) | y | n |  |
| MmaVariantsBench | test-quant | 1 | test_mxf4nvf4_mma_variants_bench.cu | 0.47G | 6.8 | BENCH: mxf4nvf4 MMA variant comparison (times kernels, prints numbers) | y | n |  |
| MmqQ4kImmaBench | test-quant | 1 | test_mmq_q4k_imma_bench.cu | 0.01G | 6.9 | BENCH: Q4_K IMMA MMQ variant timing (times kernels, prints numbers) | y | n |  |
| MmqQ4kImmaGemm | test-quant | 3 | test_mmq_q4k_imma_gemm.cu | 0.02G | 10.6 | Q4_K IMMA GEMM end to end: smallest, multi-tile and FFN-like shapes | y | n |  |
| MmqQ4kImmaTile | test-quant | 5 | test_mmq_q4k_imma_tile_bench.cu | 0.57G | 10.7 | Q4_K IMMA tile correctness: tiny, multi-sub, multi-tile, FFN shapes; BenchSweep timing | y | n |  |
| MmqQ8Imma | test-quant | 51 | test_mmq_imma_gguf_smallm.cu,test_mmq_q8_imma.cu,test_mmq_q8_imma_q4k_moe.cu,test_mmq_q8_imma_q5k.cu,test_mmq_q8_imma_q6k.cu,test_mmq_q8_imma_reuse.cu,test_mmq_q8_imma_tail.cu | 11.94G | 79.8 | Q8 IMMA MMQ: square, tails, small-M split-K and MoE grouped GEMM vs reference | y | n |  |
| Mxf4nvf4MmaBenchTest | test-quant | 1 | test_mxf4nvf4_mma_bench.cu | 0.13G | 6.8 | BENCH: block-scale vs legacy mxf4nvf4 MMA timing (times kernels, prints numbers) | y | n |  |
| Mxf4nvf4ProbeTest | test-quant | 2 | test_mxf4nvf4_probe.cu | 0.00G | 6.7 | mxf4nvf4 block-scale MMA probe: clean launch; zero A zeroes the output | y | n |  |
| Mxf4nvf4QkTest | test-quant | 4 | test_mxf4nvf4_qkt_validate.cu | 0.00G | 7.9 | mxf4nvf4 Q*K^T validation: uniform, row/col indicator, random E2M1 regression | y | n |  |
| Mxfp4GemmTile | test-quant | 1 | test_gemm_mxfp4_tile.cu | 0.00G | 9.9 | MXFP4 tile GEMM refuses a partial atom K | y | n |  |
| Mxfp4GemmTileTest | test-quant | 6 | test_gemm_mxfp4_tile.cu | 0.15G | 9.9 | MXFP4 tile GEMM vs decoded reference over dims | y | n |  |
| Mxfp4Unpacked | test-quant | 2 | test_gemm_mxfp4_tile.cu | 0.00G | 9.9 | MXFP4 unpacked dequant and embedding lookup match host decode | y | n |  |
| Mxfp8Gemm | test-quant | 6 | test_mxfp8_gemm.cu | 0.43G | 8.2 | MXFP8 GEMM vs fp32 reference within E4M3 budget; partial scale tile; refusals; dequant | y | n |  |
| Mxfp8GemmTileBench | test-quant | 1 | test_gemm_mxfp8_tile.cu | 0.43G | 9.0 | BENCH: MXFP8 tile GEMM variants timing (times kernels, prints numbers) | y | n |  |
| Mxfp8GemmTileTest | test-quant | 8 | test_gemm_mxfp8_tile.cu | 1.66G | 9.0 | MXFP8 tile GEMM vs host reference | y | n |  |
| NVFP4OutlierTest | test-quant | 2 | test_nvfp4_outlier_ref.cu | 0.00G | 8.1 | NVFP4 outlier quant-dequant GEMV vs independent reference; extreme outlier floor finite | y | n |  |
| NVFP4QuantTest | test-quant | 2 | test_nvfp4_quant.cu | 0.00G | 6.8 | NVFP4 quant roundtrip and basic GEMV | y | n |  |
| NvFP4CompressedTensorsRef | test-quant | 4 | test_nvfp4_compressed_tensors_ref.cu | 0.00G | 9.5 | NVFP4 compressed-tensors ref: unity scales, two-level scaling, zero tensor scale, sign | y | n |  |
| NvFP4ExportRoundTrip | test-quant | 2 | test_quantize_nvfp4_roundtrip.cu | 0.00G | 8.7 | NVFP4 export roundtrip: tiny model within grid bound; offset-folded norm keeps gain | y | n |  |
| NvFP4GemmGraphCapture | test-quant | 5 | test_nvfp4_gemm_graph_capture.cu | 0.00G | 9.3 | NVFP4 GEMM graph capture: workspace preferred over lazy alloc; deferred workspace replays | y | n |  |
| NvFP4GemvGdnInput | test-quant | 2 | test_nvfp4_gemv_gdn_input.cu | 0.49G | 8.6 | NVFP4 GEMV GDN input: four launches equal fused at Qwen3.8 dims; refuses unservable shapes | y | n |  |
| NvFP4GemvGemma3Dims | test-quant | 3 | test_nvfp4_gemv_gemma3_dims.cu | 0.88G | 8.1 | NVFP4 GEMV at Gemma-3 dims: KPAR per projection, fused QKV, gate/up, GeGLU down | y | n |  |
| NvFP4MergedScaleSpread | test-quant | 1 | test_nvfp4_merged_scale_spread.cu | 0.13G | 8.2 | NVFP4 merged sibling projections each keep their own global scale | y | n |  |
| NvFP4NormFold | test-quant | 2 | test_nvfp4_norm_fold.cu | 0.39G | 9.0 | NVFP4 norm fold: RMSNorm then GEMV matches the folded GEMV, with and without weight offset | y | n |  |
| NvFP4PromoteWeightScale2 | test-quant | 9 | test_nvfp4_compressed_tensors_ref.cu | 0.00G | 9.5 | NVFP4 weight_scale_2 promotion: NaN, inf and zero handling for ModelOpt and LLM-compressor | y | n |  |
| NvFP4SmallMTest | test-quant | 4 | test_nvfp4_smallm.cu | 2.05G | 8.6 | NVFP4 small-M GEMM vs host reference; A4 variant; bandwidth floor checks (timed) | y | n |  |
| NvFP4SmallMV2Test | test-quant | 12 | test_nvfp4_smallm_v2.cu | 9.38G | 10.8 | NVFP4 small-M v2 GEMM: reference, striped tiles, accumulate, bandwidth survey, ring sweep | y | n | 4 |
| NvFP4ValidatePackedScaleShapes | test-quant | 7 | test_nvfp4_compressed_tensors_ref.cu | 0.00G | 9.5 | NVFP4 packed-scale shape validator: accepts Qwen3/Gemma-4 shapes, rejects bad groups | y | n |  |
| NvFP4ValidateWeightScaleDtype | test-quant | 5 | test_nvfp4_compressed_tensors_ref.cu | 0.00G | 9.5 | NVFP4 weight-scale dtype validator: accepts FP8 E4M3, rejects uint8, E5M2, F16, none | y | n |  |
| Nvfp4GemmAlphaTest | test-quant | 4 | test_nvfp4_gemm_alpha.cu | 0.16G | 8.2 | NVFP4 GEMM alpha: FP8 E4M3 clamp boundary, alpha applied, Mistral L0 reproducer and layout | y | n |  |
| Nvfp4GemmTile | test-quant | 1 | test_nvfp4_gemm_tile.cu | 0.28G | 9.2 | NVFP4 GEMM tile: data-parallel rows invariant to M | y | n |  |
| Nvfp4GemmTileBench | test-quant | 1 | test_nvfp4_gemm_tile.cu | 113.47G | 9.2 | BENCH: NVFP4 GEMM tile vs cuBLASLt wall-time bench (times kernels, prints numbers) | y | n |  |
| Nvfp4GemmTileF32 | test-quant | 1 | test_nvfp4_gemm_tile.cu | 0.40G | 9.2 | NVFP4 GEMM tile with FP32 output vs reference | y | n |  |
| Nvfp4GemmTileTest | test-quant | 8 | test_nvfp4_gemm_tile.cu | 0.50G | 9.2 | NVFP4 GEMM tile vs reference over shape and variant params | y | n |  |
| Nvfp4QuantHwTest | test-quant | 4 | test_nvfp4_quant_hw.cu | 0.00G | 7.6 | NVFP4 quant roundtrip on hardware for head_dim 64 and 128 blocks | y | n |  |
| Nvfp4QuantRefTest | test-quant | 2 | test_nvfp4_quant_ref.cu | 0.00G | 8.2 | NVFP4 quant reference: gaussian roundtrip; representable values bit-exact | y | n |  |
| Nvfp4SmallMWsBench | test-quant | 1 | test_nvfp4_smallm_ws.cu | 0.00G | 9.4 | BENCH: NVFP4 small-M workspace GEMM vs small-M v2 (DISABLED, prints timings) | y | n |  |
| Nvfp4SmallMWsTest | test-quant | 6 | test_nvfp4_smallm_ws.cu | 0.72G | 9.4 | NVFP4 small-M workspace GEMM vs double reference at every M; row bits; refusals | y | n |  |
| Nvfp4StaticActScaleTest | test-quant | 3 | test_nvfp4_static_act_scale.cu | 0.02G | 8.4 | NVFP4 static activation scale: small-M and prompt tile vs double reference; producers | y | n |  |
| Nvfp4TinyBlockScaleTest | test-quant | 2 | test_nvfp4_static_act_scale.cu | 0.01G | 8.4 | NVFP4 tiny block-scale: SF atom bytes equal plain quantize; prompt tile vs reference | y | n |  |
| PrefillRowInvariance | test-quant | 3 | test_prefill_row_invariance.cu | 0.02G | 8.5 | Prefill row invariance: router logits, MoE act quant, HD512 attention vs batch and chunk | y | n |  |
| Q8PlanesOnly | test-quant | 3 | test_q8_planes_only.cu | 0.00G | 11.3 | Q8 IMMA planes: planes-to-F16 matches dequant; build declines; fallback from planes | y | n |  |
| QuantIntegrationTest | test-quant | 24 | test_quant_integration.cu | 0.93G | 26.0 | Quant integration: Q4_0, Q8_0, Q4_K, Q5_K upload, forward pass, determinism, multi-layer | y | n |  |
| QuantTest | test-quant | 2 | test_quant.cu | 0.00G | 7.7 | Quant config defaults and INT4 GEMM basic vs reference | y | n |  |
| QuantizeMoeNative | test-quant | 10 | test_quantize_fp16_nvfp4_moe_native.cu | 0.00G | 7.3 | MoE native NVFP4 quantize: device M offsets, compact alpha and SFA offsets, empty experts | y | n |  |
| SmallMV2Pair | test-quant | 3 | test_nvfp4_batched_smallm_equiv.cu | 1.06G | 9.8 | NVFP4 small-M v2 pair bit-exact to two single calls; M1 bench disabled; striped refusals | y | n |  |
| TmaBlockScaleBench | test-quant | 1 | test_tma_block_scale_bench.cu | 0.09G | 6.6 | BENCH: TMA vs block-scale descriptor launch timing (times kernels, prints numbers) | y | n |  |
| WeightDispatchTest | test-quant | 13 | test_weight_dispatch.cu | 0.02G | 13.1 | Weight dispatch: FP16, FP8, NVFP4, CUTLASS, MXFP4 each match direct kernel; refusals | y | n |  |
| ChatTemplateApplyTest | test-text | 20 | test_chat_template.cpp | 0.00G | 9.2 | Chat template apply: ChatML, Llama 2/3, Nemotron, Gemma, DeepSeek-R1, Phi structure | n | n |  |
| ChatTemplateArchDefaultTest | test-text | 1 | test_chat_template.cpp | 0.00G | 9.2 | Chat template default family for each architecture | n | n |  |
| ChatTemplateContinue | test-text | 6 | test_chat_template_continue.cpp | 0.00G | 3.6 | Chat template continue: Qwen3 trimmed and verbatim content match HF; cut rule | n | n |  |
| ChatTemplateDefaultSysTest | test-text | 5 | test_chat_template.cpp | 0.00G | 9.2 | Chat template default system prompt: Jinja extraction; injected only without user system | n | n |  |
| ChatTemplateDetectTest | test-text | 14 | test_chat_template.cpp | 0.00G | 9.2 | Chat template family detection from template text (ChatML, Llama, Gemma, Mistral, Phi) | n | n |  |
| ChatTemplateFamilyNameTest | test-text | 1 | test_chat_template.cpp | 0.00G | 9.2 | Chat template family names for every family | n | n |  |
| ChatTemplateGolden | test-text | 13 | test_chat_template_goldens.cpp | 0.00G | 4.2 | Chat template goldens: per-family rendered prompts vs committed golden; generation prompt | n | n |  |
| ChatTemplateInitTest | test-text | 10 | test_chat_template.cpp | 0.00G | 9.2 | Chat template init: per-family success, XML tool dialect detection, missing-token fallback | n | n |  |
| ChatTemplateParseFamilyTest | test-text | 1 | test_chat_template.cpp | 0.00G | 9.2 | Chat template parse_family: every name maps to its family | n | n |  |
| ChatTemplateTokenize | test-text | 1 | test_chat_template.cpp | 0.00G | 9.2 | Chat template tokenize: control-token strip flags swallow template whitespace | n | n |  |
| ChatTemplateTools | test-text | 1 | test_chat_template.cpp | 0.00G | 9.2 | Chat template tools: parameter past int64 max renders instead of throwing | n | n |  |
| ChatTemplateVisionTest | test-text | 6 | test_chat_template.cpp | 0.00G | 9.2 | Chat template vision: Gemma image tokens per turn; text-only fallback for other families | n | n |  |
| Cl100kPreTokenizeTest | test-text | 1 | test_tokenizer.cpp | 0.00G | 8.2 | cl100k pre-tokenizer: digit triples and case-blind standalone contractions | n | n |  |
| DigitTriplesRightSplitTest | test-text | 1 | test_tokenizer_llama_bpe.cpp | 0.00G | 2.7 | Llama-BPE digit triples: right-aligned split only before a word boundary | n | n |  |
| HFChatTemplateTest | test-text | 5 | test_chat_template.cpp | 0.00G | 9.2 | HF tokenizer_config chat_template: string, array default/fallback/empty, missing field | n | n |  |
| HarmonyGoldenTest | test-text | 4 | test_gpt_oss_harmony_golden.cpp | 0.00G | 6.0 | gpt-oss Harmony golden: user-only, system+user, multi-turn final channel, channel markers | n | n |  |
| InklingChatTemplate | test-text | 4 | test_inkling_chat_template.cpp | 0.00G | 3.5 | Inkling chat template: reasoning levels advertised, named and numeric levels, refusals | n | n |  |
| InklingDraftPolicy | test-text | 3 | test_inkling_draft_policy.cpp | 0.00G | 2.0 | Inkling draft policy: rows follow measured chance; out-of-vocab token ends the draft | n | n |  |
| InklingEffortSpec | test-text | 1 | test_inkling_chat_template.cpp | 0.00G | 3.5 | Inkling effort spec: other templates accept any value | n | n |  |
| JinjaChatTest | test-text | 7 | test_jinja.cpp | 0.00G | 9.5 | Jinja chat templates: ChatML, Gemma, Qwen3 and multi-turn render vs expected text | n | n |  |
| JinjaFilterTest | test-text | 1 | test_jinja.cpp | 0.00G | 9.5 | Jinja selectattr filter | n | n |  |
| JinjaParity | test-text | 8 | test_jinja_parity.cpp | 0.00G | 11.5 | Jinja parity: real chat templates (Qwen3, Nemotron, Qwen3.8, GLM, Flint, Kimi, gpt-oss) | n | n |  |
| JinjaTest | test-text | 65 | test_jinja.cpp | 0.00G | 9.5 | Jinja engine: expressions, control flow, filters, slices, macros, block set, whitespace | n | n |  |
| JinjaUndefinedTest | test-text | 6 | test_jinja_undefined.cpp | 0.00G | 2.2 | Jinja undefined: unset vs false/true variables, is defined, Qwen thinking guard | n | n |  |
| JinjaValueTest | test-text | 6 | test_jinja.cpp | 0.00G | 9.5 | Jinja value types: none, bool, int, string, array, object conversions | n | n |  |
| KimiChatTemplate | test-text | 3 | test_kimi_chat_template.cpp | 0.00G | 2.8 | Kimi chat template render: one turn, multi-turn with system, tool-call history | n | n |  |
| Ling3ChatTemplate | test-text | 3 | test_ling3_chat_template.cpp | 0.00G | 2.9 | Ling-3 chat template: thinking default, thinking off, thinking forced | n | n |  |
| NfcNormalize | test-text | 2 | test_tokenizer.cpp | 0.00G | 8.2 | NFC normalization: Latin combining and Hangul jamo compose; decompose, reorder, recompose | n | n |  |
| O200kPreTokenizeTest | test-text | 8 | test_tokenizer.cpp | 0.00G | 8.2 | o200k pre-tokenizer: code lines, digit triples, URLs, contractions, case runs, symbols | n | n |  |
| PreTokenizeSplitSequenceTest | test-text | 4 | test_tokenizer_pretok_json.cpp | 0.00G | 4.1 | HF-json pre-tokenizer split sequences match HF on DeepSeek-shaped steps; regex refusals | n | n |  |
| PreTokenizeUnicodeTest | test-text | 4 | test_tokenizer.cpp | 0.00G | 8.2 | Unicode pre-tokenize: emoji symbols, whitespace/digits/CJK punctuation, marks, cased runs | n | n |  |
| Qwen2PreTokenizeTest | test-text | 10 | test_tokenizer.cpp | 0.00G | 8.2 | Qwen2 pre-tokenizer: symbol runs, digits, contractions, newline and trailing whitespace | n | n |  |
| Qwen38ReasoningEffort | test-text | 9 | test_qwen38_chat_template.cpp | 0.00G | 3.8 | Qwen3.8 chat template reasoning effort: advertised values, default xhigh, preamble | n | n |  |
| TiktokenLoader | test-text | 5 | test_tokenizer_tiktoken.cpp | 0.00G | 3.1 | tiktoken loader: lowest-rank merges, atomic specials, Kimi tool names, malformed ranks | n | n |  |
| TokenizerAddedTokens | test-text | 3 | test_tokenizer.cpp | 0.00G | 8.2 | Added tokens: non-special atomic, normalized flags, lstrip/rstrip whitespace swallowing | n | n |  |
| TokenizerControlTest | test-text | 4 | test_tokenizer.cpp | 0.00G | 8.2 | SPM control tokens: mark-as-control allocates, idempotent, ignores bad ids, keeps types | n | n |  |
| TokenizerDispatchTest | test-text | 3 | test_tokenizer.cpp | 0.00G | 8.2 | Tokenizer dispatch: SPM default, set_type, max-length input | n | n |  |
| TokenizerEos | test-text | 2 | test_tokenizer_robustness.cpp | 0.00G | 4.0 | EOS: first added EOS replaces placeholder; a named EOS keeps first place | n | n |  |
| TokenizerGPT2Test | test-text | 11 | test_tokenizer.cpp | 0.00G | 8.2 | GPT-2 byte-BPE: encode, merges, decode roundtrip, digits, spaces, UTF-8, stale ranks | n | n |  |
| TokenizerGemma4Test | test-text | 4 | test_tokenizer.cpp | 0.00G | 8.2 | Gemma-4 tokenizer: merge skipped when result not in vocab; truncated UTF-8; byte fallback | n | n |  |
| TokenizerJsonHostileIds | test-text | 4 | test_tokenizer_robustness.cpp | 0.00G | 4.0 | tokenizer.json robustness: negative or huge vocab ids dropped; well-formed vocab loads | n | n |  |
| TokenizerJsonPretok | test-text | 5 | test_tokenizer_pretok_json.cpp | 0.00G | 4.1 | tokenizer.json pre-tokenizer rewrite: digit runs, splits, char and Onig classes vs HF | n | n |  |
| TokenizerLlamaBpeTest | test-text | 2 | test_tokenizer_llama_bpe.cpp | 0.00G | 2.7 | Llama-BPE pre-tokenizer: cl100k chunks for Llama, Nemotron chunks for Tekken | n | n |  |
| TokenizerNormalizer | test-text | 2 | test_tokenizer.cpp | 0.00G | 8.2 | Normalizer: NFC only when tokenizer JSON declares it; Gemma-4 JSON routes to Gemma-4 | n | n |  |
| TokenizerRobustness | test-text | 16 | test_tokenizer_robustness.cpp | 0.00G | 4.0 | Byte tokenizer roundtrip for ASCII, UTF-8, NUL and surrogate bytes; decode ids never crash | n | n |  |
| TokenizerSPMTest | test-text | 16 | test_tokenizer.cpp | 0.00G | 8.2 | SPM tokenizer: vocab load, encode words and subwords, byte fallback, decode, UTF-8 and CJK | n | n |  |
| TranscriptIdStoreTest | test-text | 2 | test_transcript_ids.cpp | 0.00G | 4.0 | Transcript id store: longest prefix wins with FIFO cap; resent reply keeps model split | n | n |  |

Cases > 1 s inside suites > 5 s (median of 3):

| case | module | median s |
|---|---|---:|
| `ServingSignalsTest.FirstShortPromptsReplayPrewarmedPrefillGraphs` | test-e2e | 153.85 |
| `Nvfp4GemmTileBench.VsCublasLt` | test-quant | 113.47 |
| `SpecCaptureFidelityTest.CachedGraphMatchesEagerForward` | test-e2e | 48.11 |
| `FmhaSm120Test.HD256_LongSeq_SlidingWindow1024` | test-attention | 43.84 |
| `FmhaSm120Test.HD256_LongSeq` | test-attention | 43.27 |
| `MtpQwen4ExpReference.TwoDraftStepsMatchVllmMath` | test-e2e | 33.98 |
| `MtpGreedyIdentityTest.MtpDoesNotChangeGreedyTokens` | test-e2e | 29.64 |
| `MtpGreedyIdentityTest.NgramDoesNotChangeGreedyTokensHybrid` | test-e2e | 25.27 |
| `FmhaSm120Test.HD128_LongSeq_SlidingWindow1024` | test-attention | 24.05 |
| `GptOssSinkRef.Fa2Hd64SinkMatchesReference` | test-attention | 21.77 |
| `MtpGreedyIdentityTest.MtpDoesNotChangeConstrainedGreedyTokens` | test-e2e | 21.08 |
| `BatchInvarianceTest.Fp8DenseDecodeOwnSmallMVsCublasLt` | test-e2e | 16.91 |
| `FmhaFA2Dense2CtaTest.Chunked` | test-attention | 16.37 |
| `MtpGreedyIdentityTest.DFlashDoesNotChangeGreedyTokens` | test-e2e | 12.58 |
| `FmhaSm120Test.ChunkedLargeContext` | test-attention | 12.45 |
| `MtpGreedyIdentityTest.DSparkDoesNotChangeGreedyTokens` | test-e2e | 12.19 |
| `BatchInvarianceTest.NativeNvfp4DecodeSoloVsBatched` | test-e2e | 10.73 |
| `MtpGreedyIdentityTest.NgramDoesNotChangeGreedyTokensDense` | test-e2e | 10.72 |
| `FmhaFA2Hd256Test.PvF16_LongCtx1536` | test-attention | 10.50 |
| `FmhaFA2Test.FP16QK_Bkv32Band_Chunked` | test-attention | 9.48 |
| `BatchInvarianceTest.Fp8DenseDecodeRowIndependentOfNeighbours` | test-e2e | 9.42 |
| `WarmCacheTest.HostileAllocIndexFallsBackToColdLoad` | test-e2e | 7.50 |
| `NvFP4SmallMV2Test.RepeatedLaunchesBitwiseStable` | test-quant | 6.99 |
| `SpecCaptureFidelityTest.HybridSpeculationDoesNotCollapse` | test-e2e | 6.35 |
| `SuspendResumeTest.SnapshotRoundtripTokenIdentical` | test-e2e | 6.33 |
| `PagedNvfp4Mma.MatchesReferenceAndMultitok` | test-attention | 6.21 |
| `DFlashDrafterTest.BlockMatchesReference` | test-e2e | 6.19 |
| `BatchInvarianceTest.Fp8KvDecodeRowIndependentOfNeighbours` | test-e2e | 6.09 |
| `FmhaFA2Test.FP16QK_Chunked_LongCtx` | test-attention | 6.05 |
| `GemmQ45kTileBench.FlushedVsImma` | test-compute | 6.05 |
| `SuspendResumeTest.DroppedKeyFallsBackColdAndStaysIdentical` | test-e2e | 5.74 |
| `FmhaFA2Dense2CtaTest.RealisticMagnitude` | test-attention | 5.73 |
| `FmhaFA2Dense2CtaTest.Causal1024_24Q4KV` | test-attention | 5.69 |
| `LlmCompressorE2E.Gemma4_LoadsAndGeneratesCoherent` | test-e2e | 5.58 |
| `FmhaFA2Dense2CtaTest.SlidingWindow` | test-attention | 5.46 |
| `FmhaFA2Dense2CtaTest.OddSeq1000` | test-attention | 5.18 |
| `WarmCacheTest.ColdLoadWritesCacheWarmBootTokenIdentical` | test-e2e | 5.12 |
| `FmhaFA2PvF16Test.Bq128Band_LongCtx` | test-attention | 4.96 |
| `GemmF16NarrowPrefill.BenchVsCublas` | test-compute | 4.86 |
| `DSparkDrafterTest.GraphReplayFollowsContextLength` | test-e2e | 4.42 |
| `DegenerationTest.ShortPromptNoRepetition` | test-e2e | 4.19 |
| `ServingSignalsTest.QuantizedKvPressureArmsTheValveAndTheRequestFinishes` | test-e2e | 3.96 |
| `ServingSignalsTest.AdmissionDoesNotWaitForKvGrowth` | test-e2e | 3.87 |
| `MmqQ8Imma.Q4KDenseWaveTailSplit` | test-quant | 3.67 |
| `WarmCacheTest.CorruptCacheIsIgnored` | test-e2e | 3.58 |
| `MmqQ8Imma.GgufLargeMWaveTailVsReference` | test-quant | 3.35 |
| `DSparkDrafterTest.Fp8WeightsTrackReference` | test-e2e | 3.29 |
| `LlmCompressorE2E.Gemma4_LoadsWithoutIMA` | test-e2e | 3.27 |
| `GroupedNvfp4DecodeTest.ActEpilogueBitEqualToUnfusedPath` | test-quant | 3.26 |
| `EngineRelaunchTest.SecondEngineOnSameModelHandleNeverIMAs` | test-e2e | 2.87 |
| `EngineRelaunchTest.ReloadAfterInferenceReleasesVramAndDoesNotCrash` | test-e2e | 2.83 |
| `FP8GemmTest.RowscaleFp32StreamedHeadMatchesSplitKAndReference` | test-compute | 2.83 |
| `SuspendResumeTest.ArmedMismatchIsIgnored` | test-e2e | 2.67 |
| `ServingSignalsTest.KvOverrunSwapsToHostInsteadOfCancelling` | test-e2e | 2.64 |
| `ServingSignalsTest.ARequestInsideItsPromiseNeverArmsTheValve` | test-e2e | 2.26 |
| `ServingSignalsTest.QueueTimeIsTheSchedulerWaitAndTheSplitIsVisible` | test-e2e | 2.24 |
| `DSparkDrafterTest.BlockMatchesReference` | test-e2e | 2.22 |
| `DegenerationTest.LongGenerationStability` | test-e2e | 2.09 |
| `DSparkDrafterTest.ContextInTwoPartsAndRollback` | test-e2e | 1.97 |
| `ServingSignalsTest.MidDecodeKvExhaustionFinishesAsCapacity` | test-e2e | 1.94 |
| `GroupedNvfp4DecodeTest.F16BitEqualToTileAndFp64` | test-quant | 1.91 |
| `DegenerationTest.NoLeakedSpecialTokens` | test-e2e | 1.75 |
| `NvFP4SmallMV2Test.BandwidthAboveStarvationFloor` | test-quant | 1.73 |
| `DegenerationTest.PrefillGraphReplayMatchesEager` | test-e2e | 1.72 |
| `GemmF16NarrowPrefill.MatchesReferenceAtQwen38Shape` | test-compute | 1.71 |
| `DegenerationTest.SecondRequestNotCorrupt` | test-e2e | 1.65 |
| `DegenerationTest.GreedyDeterminism` | test-e2e | 1.61 |
| `ServingSignalsTest.SpeculationCountersReachTheRequest` | test-e2e | 1.46 |
| `MmqQ8Imma.Q4KDenseLargeMNRMSE` | test-quant | 1.32 |
| `FmhaFA2Test.FP16QK_Bkv32Band_CausalMultiTile` | test-attention | 1.05 |
| `FmhaFA2PvF16Test.Bkv32Band_CausalMultiTile` | test-attention | 1.03 |
| `FmhaFA2Test.FP16QK_Bkv32Band_SlidingWindow` | test-attention | 1.03 |

## Phase 2: classification

Read-only like Phase 1: no test, build or CI file changed. Part A = test-core, test-text, test-hf-network (529 suites, `/mnt/data/scratch/tsa/phase2_A.md`); part B = the 6 GPU modules (330 suites, `/mnt/data/scratch/tsa/phase2_B.md`); merged per-suite table below (`scripts/tsa3_merge_p2.sh`, flips from `p3/flips.tsv`). Origin column: part A `REG` = first commit is a fix or names a regression, `REF?` = body cites issues; part B `REG` = subject `fix`/`revert`, `(body)` = body names a bug or reproducer, `fN` = fix commits on the suite's files. `REGRESSION-ORIGIN` suites are KEEP unless a mutant proves another catcher.

### Verdicts per module (suites; cases in the last column)

| verdict | test-core | test-text | test-hf-network | test-compute | test-attention | test-quant | test-kv | test-moe-gdn | test-e2e | suites | cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KEEP | 465 | 47 | 0 | 61 | 22 | 52 | 25 | 28 | 53 | 753 | 3971 |
| SHRINK | 8 | 0 | 0 | 9 | 6 | 6 | 0 | 0 | 3 | 32 | 229 |
| MERGE | 5 | 0 | 0 | 3 | 1 | 0 | 1 | 3 | 0 | 13 | 101 |
| MOVE | 0 | 0 | 0 | 8 | 4 | 17 | 1 | 3 | 18 | 51 | 154 |
| DELETE | 3 | 0 | 1 | 1 | 1 | 3 | 0 | 0 | 1 | 10 | 21 |
| all | 481 | 47 | 1 | 82 | 34 | 78 | 27 | 34 | 75 | 859 | 4476 |

### Mutants

Part A: 11 mutants on the CPU, `logs/p2a_mutants.log`. Part B: 23 mutants on the GPU, worktree `imp-tsa3m` (branch `audit/tsa3-mut`, no commit), one binary set per mutant from `bld/tsa3m` (`imp:toolchain`, ccache, `scripts/tsa3_build_mutants.sh`), file backed up, restored with `git checkout`, `git status --short` empty after each (`logs/tsa3_build_mut*.log`). Every changed binary differs from base except `test-compute` for M-Q4K3 (TU not linked). Control: every catcher filter on the unmutated base set, 25 of 25 rc 0. Runs: `scripts/tsa3_run_mutants.sh`, logs `mut/runs/`. Totals: 34 mutants, 30 red (5 of them `BUILD_RED`), 4 survived.

Part A:

| id | candidate | mutant (file:line, edit) | expected catcher | result |
|---|---|---|---|---|
| LA-1 | LtGemmArgs, LtGemmBatchedArgs (declaration-order case) | src/compute/gemm_lt_ex.h:25-26 swap `int m`/`int n` | build (designated-init call sites) | BUILD_RED (g++ 'designator order ... does not match' in gemm_grouped.cpp:232, gemm_batched.cpp:60, gemm.cpp:786, qwen3vl_encoder.cpp:30, gemma4_audio_encoder.cpp:21; nvcc inkling_attn_gemm.cu:213 'out-of-order initializers') |
| HF-1 | HfFetchNetwork.GatedRepoWithoutTokenFailsClean | src/model/hf_fetch.cpp:374 gated check -> `false &&` | HfFetchMock | RED by: HfFetchMock.GatedRepoWithoutTokenIsAClearErrorBeforeAnyDownload |
| HF-2 | HfFetchNetwork.TinyGgufFetchVerifiesShaThenHitsCache | src/model/hf_fetch.cpp:399 sha compare -> `false &&` | HfFetchMock | RED by: HfFetchMock.ShaMismatchFailsAndLeavesNoFile |
| HF-3 | HfFetchNetwork.TinyGgufFetchVerifiesShaThenHitsCache | src/model/hf_fetch.cpp:453 cache hit -> `false &&` | HfFetchMock | RED by: HfFetchMock.DownloadsVerifiesThenServesFromCacheWithoutRequests HfFetchMock.SafetensorsRepoFetchesTheSetAndReturnsTheSnapshotDir |
| FP-1 | FpBitsLegacyParity.Fp16EncodeFloatSweep (SHRINK) | src/core/fp_bits.h:69 normal-range RNE tie -> `if (true)` | FpBits (expected), sweep | RED by: FpBitsLegacyParity.Fp16EncodeFloatSweep FpBitsLegacyParity.NgramGatherElementAll16M WeightUploadTraits.MatchesPerFormatUploadersByteForByte |
| FP-2 | FpBitsLegacyParity.Fp16EncodeFloatSweep (SHRINK) | src/core/fp_bits.h:62 subnormal RNE tie -> `rem >= halfway` | FpBits / static_assert | BUILD_RED (static_assert src/core/fp_bits.h:122 failed) |
| LA-2 | PagedDecodeArgs, KvWriteRopeArgs, QknormRopeArgs, GdnScanBatchedArgs (declaration-order case) | src/compute/attention_paged.h:43-44 swap `n_heads`/`n_kv_heads` | build (nvcc call sites) | BUILD_RED (nvcc "out-of-order initializers" attention_paged_fp8.cu:436,506,542,627) |
| TD-1 | ToolCallDispatch | tools/imp-server/tool_call_dialect.cpp:23 LLAMA3 -> kChatmlToolDialect | ToolCallDialectGolden | RED by: ForcedToolEnvelope.LiteralsPerFamily StrictToolConstraint.NonChatMLFamilyStillOptsOut ToolCallDialectGolden.AllFamiliesMatchRecordedGolden ToolCallDispatch.RoutesByFamily ToolCallReconstruct.Llama3WrapsCall ToolChoiceEnforcement.BareArgsFamiliesEnforceTheNamedCaseButNotRequired ToolChoiceEnforcement.RequiredIsEnforcedOnChatMLOnly ToolStreamFilterLlama3.ArgsStreamIncrementally ToolStreamFilterLlama3.Chunked ToolTagScan.ChatmlAndLlama3Unchanged |
| TD-2 | ToolCallDispatch | tools/imp-server/tool_call_dialect.cpp:24 GEMMA -> kChatmlToolDialect | ToolCallDialectGolden | RED by: ForcedToolEnvelope.Gemma4JsonBodyParsesAsACall ForcedToolEnvelope.LiteralsPerFamily ToolCallDispatch.RoutesByFamily ToolCallReconstruct.GemmaWrapsCall ToolChoiceEnforcement.BareArgsFamiliesEnforceTheNamedCaseButNotRequired ToolChoiceEnforcement.NamedFunctionIsEnforceableOnGemmaOnlyWithTheNativeToken ToolChoiceEnforcement.NamedFunctionIsEnforcedOnChatMLLlama3AndHarmony ToolChoiceEnforcement.RequiredIsEnforcedOnChatMLOnly ToolResponseFormat.GemmaDefaultsNameToTool ToolResponseFormat.GemmaWrapsWithNameAndQuotes ToolStreamFilterGemma.CharByChar ToolStreamFilterGemma.WholeString |
| LA-3 | SsmScanPrefillArgs | src/compute/ssm.h:132-133 swap `n_tokens`/`n_heads` | build | BUILD_RED (nvcc executor_ssm_gdn.cu:290; g++ test_launcher_args.cpp:186) |
| LA-4 | GdnChunkparArgs, GdnScanArgs | src/compute/gdn.h:117-118 swap `n_tokens`/`n_heads` | build | BUILD_RED (nvcc gdn_ragged.cu:97; g++ test_launcher_args.cpp:176) |

Part B:

| id | candidate | mutant (file:line, edit) | expected catcher | result (red cases, first failing assertion) | effect |
|---|---|---|---|---|---|
| M-NV1 | NVFP4QuantTest DELETE | `src/quant/nvfp4_quant.cu:344` `h_absmax / kFP4E2M1Max` -> `/ 4.0f` | NvFP4GemvGemma3Dims | SURVIVED: 7 of 7 green (NvFP4GemvGemma3Dims, NVFP4QuantTest, Nvfp4QuantRefTest) | no flip: the candidate is blind too; gap or equivalent mutant (global scale cancels in quantize + GEMV) |
| M-NV2 | NVFP4QuantTest DELETE | `src/compute/nvfp4_quant_ref.cu:75` `vec_max / 6.0f` -> `/ 4.0f` | Nvfp4QuantRefTest | RED: `Nvfp4QuantRefTest.RepresentableValuesAreBitExact` (`tests/test_nvfp4_quant_ref.cu:119`, diff 0.25 > 0.01); NVFP4QuantTest green | DELETE holds |
| M-INT4 | INT4QuantTest DELETE | `src/compute/attention_paged_int4.cu:20` sign extension -> `return val;` | PagedAttentionINT4Test | RED: `PagedAttentionINT4Test.DecodeSingleHead`, `PagedOracle/3` HD128_Sweep, HD256_Sweep, SinksAndWindowAcrossEvictedBlocks (`tests/test_paged_attention.cu:1611`, 0.289 > 0.15); INT4QuantTest green | DELETE holds |
| M-Q4K1 | MmqQ4kImma{Tile,Gemm,Layout} DELETE | `src/compute/mmq_q8_imma_q4k.cu:50` `(8.0f * a)` -> `(7.0f * a)` | MmqQ8Imma Q4_K | RED: 5 MmqQ8Imma Q4_K cases (`tests/test_mmq_q8_imma.cu:268`, nrmse 0.109 > 2e-2); MmqQ4kImma* green | DELETE holds |
| M-Q4K2 | same | `src/compute/mmq_q8_imma_q4k.cu:28` `& 63u` -> `& 31u` | MmqQ8Imma Q4_K | RED: 4 MmqQ8Imma Q4_K cases (`tests/test_mmq_q8_imma_q4k_moe.cu:116`, 0.0224 > 2e-2); MmqQ4kImma* green | DELETE holds |
| M-Q4K3 | same (dead-code proof) | `src/compute/mmq_q4k_imma_tile.cu:405` -> `return false;` first | only MmqQ4kImmaGemm | RED: only `MmqQ4kImmaGemm` E2EFFNLike, E2EMultiTile, E2ESmallest (`tests/test_mmq_q4k_imma_gemm.cu:134`); rest of test-quant 354 ok / 9 skip; test-compute binary byte-identical to base (TU not linked), 340 ok | DELETE holds: the 7-arg overload has no production caller |
| M-MV1 | MMVQ SHRINK | `src/compute/ggml_mmvq.cu:390` Q8_0 `d * 0.5f` | GgufRef.Q8_0_GemvMmvq | RED: `GgufRef.Q8_0_GemvMmvq` (`tests/test_gguf_dequant_ref.cu:455`, rms_rel 0.50), `MMVQ.Q8_0_LargerDims`, `MMVQ.Q8_0_MatchesDp4a` | SHRINK holds (kept LargerDims also red) |
| M-MV2 | same | `src/compute/ggml_mmvq.cu:215` Q4_K `d * 1.0625f` | GgufRef.Q4_K_GemvMmvq | RED: `GgufRef.Q4_K_GemvMmvq` (rms_rel 0.062), `MMVQ.Q4_K_LargerDims`, `MMVQ.Q4_K_MatchesDp4a` | SHRINK holds |
| M-MV3 | same | `src/compute/ggml_mmvq.cu:362` Q5_K `d * 1.0625f` | GgufRef.Q5_K_GemvMmvq | RED: `GgufRef.Q5_K_GemvMmvq` (rms_rel 0.061), `MMVQ.Q5_K_LargerDims`, `MMVQ.Q5_K_MatchesDp4a` | SHRINK holds |
| M-DP1 | GemmDP4ATest SHRINK | `src/compute/gemv_dp4a_traits.cuh:263` `(float)sumi` -> `(float)(sumi + 1)` | GgufRef.Q8_0_GemvDp4a | SURVIVED: GgufRef 23 of 23, GemmDP4ATest 3 of 3 green (off-by-one inside the 1.5e-2 envelope) | too weak, replaced by M-DP1b |
| M-DP1b | same | same line, `* 1.0625f` | GgufRef.Q8_0_GemvDp4a | RED: `GgufRef.Q8_0_GemvDp4a` (`tests/test_gguf_dequant_ref.cu:361`, rms_rel 0.062); GemmDP4ATest 3 of 3 green | SHRINK holds: the deleted cases are blind to it |
| M-DP2 | same | `src/compute/gemv_dp4a_traits.cuh:230` Q6_K `bp + 192` -> `bp + 193` | GgufRef.Q6_K_GemvDp4a | RED: `GgufRef.Q6_K_GemvDp4a`, `GgufRef.DenseRowsDp4aAllTypes` (rms_rel 1.36); GemmDP4ATest 3 of 3 green | SHRINK holds |
| M-E2E1 | EndToEndModelTest SHRINK | `src/api/imp_api_token.cpp:57` `ids(tokens + 1, ...)` | PrimaryModelTest.TokenizeRoundtrip | RED: only `PrimaryModelTest.TokenizeRoundtrip` (`tests/test_e2e_models.cpp:140`, "Hello" missing); EndToEndModelTest.Tokenize green | SHRINK holds |
| M-E2E2 | same | `src/api/imp_api.cpp:786` `n_tokens - 1` | PrimaryModelTest / GreedyLockTest | RED (rc 139): PrimaryModelTest x4, GreedyLockTest x3, EndToEndModelTest LongContext8k, MultiDecodeOutputIsolation, PrefillDecodeStep (`tests/test_e2e.cpp:230`) | SHRINK holds |
| M-PA1 | PagedAttentionTest MERGE | `src/compute/attention_paged_common.cuh:107` `first_block + 1` | PagedOracle window rows | RED: only `PagedOracle/0.SinksAndWindowAcrossEvictedBlocks` F16 (`tests/test_attention_paged_oracle.cu:698`, 0.750 > 0.01) | MERGE holds |
| M-PA2 | same | `src/compute/attention_paged_common.cuh:84` `ctx_len / block_size` | PagedOracle sweeps | RED: `PagedOracle/0` HD128_Sweep, HD256_Sweep; PagedAttentionTest x5 (`tests/test_paged_attention.cu:184`) | MERGE holds |
| M-PA3a | same | `src/compute/attention_paged.cu:111` `bt[first_block + 1]` | PagedOracle F16 | RED: only `PagedOracle/0.SinksAndWindowAcrossEvictedBlocks` (`tests/test_attention_paged_oracle.cu:691`, 0.315 > 0.01) | MERGE holds |
| M-PA3b | same | `src/compute/attention_paged.cu:305` (MHA kernel) `bt[blk - 1]` | PagedOracle F16, else keep MHA cases | RED: `PagedOracle/0` HD128_Sweep, HD256_Sweep, SinksAndWindow...; `PagedAttentionTest.MultiBlock` (`tests/test_paged_attention.cu:253`) | MERGE holds: PagedOracle reaches the MHA kernel |
| M-GR | Grammar+RegexConstrainMaskTest MERGE | `src/compute/grammar_constrain.cu:146` `vocab_size` (the #1493 bug) | the merged TEST_P | RED: only `GrammarConstrainMaskTest.ModelVocabLargerThanTokenizerMasksPadding` (`tests/test_grammar_regex_mask.cu:95`) | MERGE holds, both cases stay |
| M-GR2 | same | `src/compute/regex_constrain.cu:235` same edit | the merged TEST_P | RED: only `RegexConstrainMaskTest.ModelVocabLargerThanTokenizerMasksPadding` (`tests/test_grammar_regex_mask.cu:61`) | each case is the sole catcher of its copy: merge, never drop one |
| M-TK | TensorKindCoverage DELETE | `src/model/tensor_kind_matcher.cpp:109` `".attn_qx."` | TensorKindMatcher (CI) | RED: `TensorKindMatcher.AttentionProjections` (`tests/test_tensor_kind_matcher.cpp:8`) | DELETE holds |
| M-QP | QuantPipelineTest DELETE | `src/exec/executor_pre_dequant.cpp:234` Phase-1 FP16 cache call -> `(void)0;` | QuantIntegrationTest | SURVIVED: QuantIntegrationTest 24 of 24 green | FLIP: DELETE -> MOVE(re-point) |
| M-Q8 | Q8DropSourceE2E MOVE | `src/exec/pre_dequant_phase4c_q8_planes.cpp:39` `continue;` removed | none expected | SURVIVED: Q8PlanesOnly + QuantIntegrationTest 27 of 27 green | MOVE(re-point) confirmed over delete |

Survivors:

| id | why it survived | verdict effect |
|---|---|---|
| M-NV1 | tensor-scale divisor change cancels in quantize + GEMV; the candidate NVFP4QuantTest is blind too | none; open gap (no test pins `quantize_fp16_to_nvfp4`'s tensor scale) |
| M-DP1 | `sumi + 1` stays inside the 1.5e-2 envelope | none; M-DP1b (`* 1.0625f`) on the same line is red by GgufRef only |
| M-QP | no remaining suite checks the Phase-1 FP16 decode cache of a Q8_0 GGUF | QuantPipelineTest DELETE -> MOVE(re-point) |
| M-Q8 | expected: no suite checks the Q8_0 source release rule | Q8DropSourceE2E stays MOVE(re-point) |

### Per-suite verdicts

Rows with "Mutant note" or "FLIPPED by a mutant" carry the Part B mutant result.

| Suite | module | cases | verdict | failure class | origin | redundancy | impl/behaviour; dead/flaky | reason |
|---|---|---:|---|---|---|---|---|---|
| ApaPrefillTest | test-attention | 8 | KEEP | APA two-pass prefill (opt-in) drifts from the FP32 oracle or paged KV differs from flat | `d6d803fe` #2635 f1 | Ra2PrefillTest (same oracle, other kernel) | - | only gate of the APA kernel |
| AttentionChunkedTest | test-attention | 5 | KEEP | cuBLAS prefill attention (executor_attention_prefill.cpp:234, encoder_forward.cu:236) wrong causal mask at q_offset, GQA or window | `d50cc86c` #130 (body) f0 | AttentionCrossPathTest (coarser) | - | live route, body of #130 names a chunked-prefill fix |
| AttentionCrossPathTest | test-attention | 11 | KEEP | FMHA, FA2 and cuBLAS routes disagree on golden MHA/GQA/softcap/window configs and 1-2 token edges | `5e31f21f` #527 f3 | per-kernel suites (finer) | - | only cross-route agreement oracle |
| Fa2Hd512Test | test-attention | 4 | KEEP | Gemma-4 hd=512 split-D FA2 prefill wrong vs CPU reference | `e2aa723e` #2900 f0 | FmhaHd512Test (WMMA kernel) | - | own kernel |
| FmhaFA2Dense2CtaTest | test-attention | 5 | SHRINK | 2-CTA dense FA2 (hd128, Bq128) wrong on causal, odd, window, chunked shapes | `92d98766` #1843 f5 | FmhaFA2Test (same oracle and harness) | - | fold the 5 FA2 variant suites of test_fmha_fp8.cu into one TEST_P over variant x shape, every shape kept |
| FmhaFA2Hd256Test | test-attention | 14 | SHRINK | hd=256 FA2 PV-f16 prefill wrong on causal, GQA, window, softcap | `b1e1b376` #930 f5 | FmhaFA2PvF16Test | contains a WMMA micro-bench case | micro-bench case to the bench target, rest folds into the FA2 TEST_P |
| FmhaFA2HeavyFirstTest | test-attention | 3 | SHRINK | heavy-first CTA order changes output bits | `6caa966d` #1865 f5 | FmhaFA2Test | - | fold into the FA2 TEST_P |
| FmhaFA2PvF16Test | test-attention | 13 | SHRINK | PV f16 accumulate (default on) loses precision past tolerance | `75d86673` #674 f5 | FmhaFA2Test | - | fold into the FA2 TEST_P |
| FmhaFA2Test | test-attention | 21 | SHRINK | default FA2 prefill kernel wrong mask, GQA, window, chunk offset, occupancy band | `6fc367e3` #477 (body) f5 | - | - | fold the 5 suites, keep every shape |
| FmhaHd512Test | test-attention | 9 | SHRINK | WMMA FMHA hd=512 wrong at Gemma-4 geometry; sliced cuBLAS parity | `7360fbb7` #1042 f1 | Fa2Hd512Test | 2 DISABLED_ benches | the 2 DISABLED_ benches to the bench target |
| FmhaSm120Test | test-attention | 24 | KEEP | native sm_120 FMHA hd 64/96/128/256 wrong (live in attention_dispatch.cpp) | `54ce135f` f2 | AttentionCrossPathTest (coarser) | 2 GTEST_SKIP lines | - |
| GptOssSinkRef | test-attention | 9 | KEEP | attention-sink logit dropped or misweighted in FMHA, FA2 hd64, paged decode, streaming vs fp64 | `f6419bf5` #587 (body) f0 | PagedOracle sinks (paged only) | - | only fp64 sink oracle for prefill kernels |
| INT4QuantTest | test-attention | 1 | DELETE | none: packs and unpacks with lambdas defined in the test body, no production code runs | `b03137a9` f1 | PagedAttentionINT4Test runs the kernel unpack (attention_paged_int4.cu:20) | dead: tests itself | obviously dead, mutant M-INT4 shows the kernel path stays covered Mutant note: M-INT4 red by PagedAttentionINT4Test.DecodeSingleHead and 3 PagedOracle INT4 rows, INT4QuantTest green |
| PagedAttentionINT4Test | test-attention | 1 | KEEP | INT4 KV decode wrong nibble sign extension or per-head scale vs FP32 reference | `b03137a9` f1 | PagedOracle INT4 rows | - | - |
| PagedAttentionReduce | test-attention | 3 | KEEP | paged split-K LSE merge wrong vs fp64; staged vs unstaged bits differ | `0dfa7d7c` #1420 f0 | - | - | - |
| PagedAttentionTest | test-attention | 24 | MERGE(PagedOracle) | FP16 paged decode wrong block-table walk, GQA mapping, split-K, sinks, HD512 | `8d28c662` f1 | PagedOracle (fp64, every KV dtype, HD128/256) overlaps block walk, GQA, split-K | - | merge the overlapping FP16 cases into PagedOracle rows, keep batch-length, sinks, HD512 cases; paged KV = MERGE only Mutant note: M-PA1, M-PA3a red by PagedOracle only; M-PA2, M-PA3b (MHA kernel) red by PagedOracle F16 rows and PagedAttentionTest |
| PagedF16Decode | test-attention | 1 | MOVE(bench) | none: times kernels, prints numbers | `aff8b17d` #1877 f3 | - | BENCH | benchmark, own target |
| PagedF16Multitok | test-attention | 1 | KEEP | F16 multitok paged attention wrong on split-K vs direct route | `f70e072a` #1880 f3 | - | - | - |
| PagedFp8Decode | test-attention | 1 | MOVE(bench) | none: times kernels, prints numbers | `fd13ada8` #1872 f3 | - | BENCH | benchmark, own target |
| PagedFp8Multitok | test-attention | 2 | KEEP | FP8 multitok paged decode wrong at single and grouped split; split arm writes no scratch | `fd13ada8` #1872 f3 | - | split arm scratch check added by fix #3333 | - |
| PagedNvfp4Decode | test-attention | 1 | MOVE(bench) | none: times kernels, prints numbers | `905a7db4` #3114 (body) f3 | - | BENCH | benchmark, own target |
| PagedNvfp4Mma | test-attention | 1 | KEEP | NVFP4 KV MMA split-K decode wrong vs fp64 over GQA ratios, HD128/256 | `8c7edef7` #2440 f3 | - | - | NVFP4 MMA class |
| PagedNvfp4Multitok | test-attention | 1 | KEEP | NVFP4 four-token paged kernels wrong on both split-K routes | `8035adaa` #1876 f3 | - | - | - |
| PagedOracle | test-attention | 18 | KEEP | paged decode wrong vs fp64 on every KV dtype; window and sinks over evicted blocks | `f6419bf5` #587 (body) f3 | - | - | catcher for the PagedAttentionTest merge |
| PagedSplitKFallback | test-attention | 1 | KEEP | split-K phase-1 launch failure does not fall back, long context diverges | REG `ff22a520` #817 f3 | - | - | REGRESSION-ORIGIN |
| QsaIndexer | test-attention | 5 | KEEP | QSA indexer top-k, query prep, split-K or GQA path diverges from dense paged | `dc05d8b8` #2057 (body) f1 | - | - | - |
| Ra2PrefillTest | test-attention | 6 | KEEP | RA2 FP4 prefill (opt-in) wrong vs FP32 oracle at hd 64/128/256, chunk continuation, declines | `c56c3462` #2628 f1 | ApaPrefillTest | - | only gate of the RA2 kernel |
| SparseAttnE2E | test-attention | 3 | KEEP | sparse block select drops the needle; identity table not bit-identical | `cefd5e81` #1805 f1 | - | - | - |
| SparseMinMaxTest | test-attention | 8 | KEEP | per-block key min/max and mean/std metadata wrong after prefill, decode, rollback (FP8, NVFP4) | `cefd5e81` #1805 f1 | - | - | - |
| SparsePrefillGeometry | test-attention | 1 | MOVE(test-core) | sparse prefill tail-geometry helper drops tail tokens | `bc1c69c0` #2379 f0 | - | host-only helper in a GPU binary | no CUDA call in the body: runs in the CI lane after the move |
| SparsePrefillTest | test-attention | 2 | KEEP | compacted sparse block table not bit-identical at full budget; sink/recent/needle order wrong | `bc1c69c0` #2379 f0 | - | - | - |
| SparseScoreMeanStd | test-attention | 1 | KEEP | outlier page keeps its budget slot | `3564feb3` #1994 f1 | - | - | - |
| SparseScoreSplit | test-attention | 1 | KEEP | split score path diverges at Qwen3.8-27B shape | `23ae2ddb` #3107 f1 | - | - | - |
| SparseSelectTest | test-attention | 5 | KEEP | sparse top-k order, negative scores, spec rows wrong | `cefd5e81` #1805 f1 | - | - | - |
| ActivationTest | test-compute | 14 | KEEP | SwiGLU/GELU/GeGLU wrong on FP16 vector/tail path, Inf input, NVFP4 producer bytes | `7265d56f` f1 | - | - | - |
| CudaFaultSignalTest | test-compute | 2 | KEEP | sticky CUDA fault not exiting, non-sticky fault not cleared | REG `c76d9066` #1913 f2 | - | death test | REGRESSION-ORIGIN |
| DrafterNorms | test-compute | 2 | KEEP | drafter head-norm + RoPE per-row positions or weight types wrong | `86e5cf30` #3347 f0 | - | - | new with #3347 |
| EmbeddingTest | test-compute | 9 | KEEP | embedding lookup wrong at boundary ids, large vocab, device-side ids | `7265d56f` f0 | - | - | - |
| ExecutorKernelsTest | test-compute | 13 | KEEP | elementwise add, bias, scale, residual-RMSNorm wrong | `36507c8c` f0 | - | - | - |
| FfnGraphCacheGpu | test-compute | 2 | KEEP | FFN graph capture refusal not falling back to eager; evicted exec not updated | REG `c4436b40` #2549 f1 | - | - | REGRESSION-ORIGIN |
| Fp32ResidualNorm | test-compute | 5 | KEEP | fused fp32 residual+RMSNorm differs from norm-then-quant; row bits depend on M | `44578a7f` #3012 f0 | - | - | - |
| FP8GemmTest | test-compute | 13 | SHRINK | FP8 GEMM/GEMV rowscale heads wrong vs fp64; rows depend on M | `ff4d81d8` f0 | - | 2 env-gated probes (IMP_FP8_PROBE, IMP_FP8_HEAD_SWEEP) skip in every lane | the 2 env-gated probe/sweep cases to the bench target |
| FP8ProjStreamTest | test-compute | 2 | SHRINK | streamed FP8 projection not bit-equal to split-K reference | `9046541d` #3043 f0 | - | Sweep skips unless IMP_FP8_PROJ_SWEEP=1 | Sweep case to the bench target |
| GatedResidualDecode | test-compute | 5 | KEEP | gated residual decode differs from unfused reference | `2934f575` #2647 f0 | - | - | - |
| GemmCaptureProbe | test-compute | 2 | KEEP | GEMM workspace lost after arena reset; cold shape inside capture invalid | REG `b5ec2d04` #2396 f2 | - | - | REGRESSION-ORIGIN |
| GemmCaptureSm120 | test-compute | 5 | SHRINK | sm_120 FP16 GEMM wrong at narrow N | `34b12a87` #181 f2 | GemmTest (coarser) | 1 DISABLED_ bench | DISABLED_ bench to the bench target |
| GemmDP4ATest | test-compute | 3 | SHRINK | DP4A GEMV Q8_0/Q6_K wrong; fused SwiGLU Q8_1 quantize wrong | `b03137a9` f0 | GgufRef.Q8_0_GemvDp4a and Q6_K_GemvDp4a call the same gemv_q8_0_q8_1 / gemv_q6k_q8_1 vs fp64 at N256 K1024 | - | delete Q8_0_Q8_1_Basic and Q6K_Q8_1_Basic (M64/16, K256), keep FusedSwiGLUQuantize; mutants M-DP1, M-DP2 Mutant note: M-DP1b and M-DP2 red by GgufRef only, all 3 GemmDP4ATest cases green on both |
| GemmF16NarrowPrefill | test-compute | 7 | SHRINK | narrow-N FP16 prefill GEMM wrong or launch-nondeterministic | `c85aa9fe` #2039 f1 | - | BenchVsCublas 4.80 s in the default run | BenchVsCublas to the bench target |
| GemmF16NarrowSmallM | test-compute | 5 | SHRINK | narrow-N small-M FP16 GEMM wrong or nondeterministic | `0b095c4d` #1977 f0 | - | BenchQwen38Shape 0.44 s | bench case to the bench target |
| GemmF16SplitK | test-compute | 1 | KEEP | cluster split-K FP16 GEMM diverges from cuBLASLt | `03096565` #2946 f0 | - | - | - |
| GemmF16TileBench | test-compute | 1 | MOVE(bench) | none: wall-time bench | `7c929607` #2896 f0 | - | BENCH 1.82 s | benchmark, own target |
| GemmF16TileTest | test-compute | 6 | KEEP | FP16 tile GEMM diverges from cuBLASLt over shape/variant | `7c929607` #2896 f0 | - | - | - |
| GemmGroupedNvfp4Tile | test-compute | 1 | KEEP | grouped NVFP4 tile GEMM rows differ from single-expert GEMM | `6c6b7048` #3163 f0 | - | - | NVFP4 MMA class |
| GemmKernelRegistryTest | test-compute | 17 | KEEP | GEMM registry returns wrong kernel or refuses wrong scratch | `340fb795` #199 (body) f4 | - | impl-coupled keys | dispatch guard |
| GemmMxfp4RepeatTest | test-compute | 2 | MERGE(GemmMxfp4TileTest) | MXFP4 tile GEMM not bit-stable across repeats | `00e143a4` #2914 (body) f0 | - | - | same file and fixture, rename only |
| GemmMxfp4TileBench | test-compute | 1 | MOVE(bench) | none: wall-time bench | `00e143a4` #2914 (body) f0 | - | BENCH 0.43 s | benchmark, own target |
| GemmMxfp4TileTest | test-compute | 8 | KEEP | MXFP4 tile GEMM differs from dequant + FP16 tile | `00e143a4` #2914 (body) f0 | - | - | - |
| GemmNvfp4TileBench | test-compute | 1 | MOVE(bench) | none: wall-time bench | `76463605` #3162 f0 | - | BENCH 0.36 s | benchmark, own target |
| GemmNvfp4TileQuant | test-compute | 1 | KEEP | NVFP4 tile GEMM differs from dequantize_nvfp4 after quantize | `76463605` #3162 f0 | - | - | NVFP4 MMA class |
| GemmNvfp4TileTest | test-compute | 7 | KEEP | NVFP4 tile GEMM not bit-equal to exact F16 weight; E4M3 subnormal scales | `76463605` #3162 f0 | - | - | NVFP4 MMA class |
| GemmQ45kRepeat | test-compute | 1 | MERGE(GemmQ45kTileTest) | Q4_K tile GEMM not bit-stable across repeats | `78642e64` #2915 (body) f0 | - | - | same file, rename only |
| GemmQ45kTileBench | test-compute | 1 | MOVE(bench) | none: wall-time bench vs IMMA | `78642e64` #2915 (body) f0 | - | BENCH 6.26 s | benchmark, own target |
| GemmQ45kTileTest | test-compute | 10 | KEEP | Q4_K-to-F16 tile GEMM differs from dequant + F16 tile | `78642e64` #2915 (body) f0 | - | - | GGUF legacy, still loads |
| GemmQ6kRepeatTest | test-compute | 2 | MERGE(GemmQ6kTileTest) | Q6_K tile GEMM not bit-stable across repeats | `821683fe` #2913 f0 | - | - | same file, rename only |
| GemmQ6kTileBench | test-compute | 1 | MOVE(bench) | none: wall-time bench | `821683fe` #2913 f0 | - | BENCH 2.19 s | benchmark, own target |
| GemmQ6kTileTest | test-compute | 8 | KEEP | Q6_K-to-F16 tile GEMM differs from dequant + F16 tile | `821683fe` #2913 f0 | - | - | GGUF legacy, still loads |
| GemmStrategy | test-compute | 1 | MOVE(test-core) | GemmStrategy operator== compares by identity, registry lookup misses | `340fb795` #199 (body) f4 | - | host-only | no CUDA call, CI lane after the move |
| GemmTest | test-compute | 11 | KEEP | FP32/FP16 GEMM, GEMV dispatch wrong vs CPU reference, alpha/beta | `ff4d81d8` f0 | - | - | - |
| GemvDp4aMultiTest | test-compute | 3 | KEEP | multi-weight DP4A GEMV differs from single-weight GEMVs | `4c438959` #2918 f0 | - | - | - |
| GemvFp16HotCold | test-compute | 1 | MOVE(bench) | none: DISABLED_ hot/cold timing | `e6b9f3ab` #3225 f0 | - | BENCH, DISABLED_ | benchmark, own target |
| GptOssYarnRef | test-compute | 2 | KEEP | gpt-oss YaRN RoPE drifts from fp64 golden at long positions | `f6419bf5` #587 (body) f0 | RoPETest YaRN (coarser) | - | - |
| HadamardTest | test-compute | 12 | KEEP | Hadamard transform (live in executor_ffn/attention_qkv) wrong or not an involution | `0b37bf6e` f0 | - | - | - |
| InklingAttention | test-compute | 1 | KEEP | Inkling few-row attention differs from row form | `6da70aff` #3222 f0 | - | - | - |
| InklingAttnPaged | test-compute | 2 | SHRINK | Inkling paged attention not bit-equal to contiguous on shuffled blocks | `4d0b7a24` #3308 f0 | - | 1 DISABLED_ hot/cold bench | DISABLED_ bench to the bench target |
| InklingCrossSlot | test-compute | 2 | KEEP | Inkling prefix restore across slots diverges from a fresh run | `fd7300ce` #3340 (body) f0 | - | skips in every default lane: INKLING_ORACLE dumps not in the model store | wire the oracle dir (anomaly B3) |
| InklingDecode | test-compute | 2 | KEEP | Inkling small-tile expert GEMM / I8B32 projection differ from references | `5600f36f` #3228 f0 | - | - | - |
| InklingDraftModel | test-compute | 4 | KEEP | Inkling draft continuation or ring layout diverges from reference | `e4e00861` #3214 f0 | - | oracle-gated, B3 | - |
| InklingLookahead | test-compute | 2 | KEEP | rejected draft rows leave state behind | `e4e00861` #3214 f1 | - | oracle-gated, B3 | - |
| InklingLora | test-compute | 3 | KEEP | Inkling LoRA delta differs from merged weights | `c041da46` #3342 f0 | - | checkpoint only | - |
| InklingPagedKv | test-compute | 1 | KEEP | Inkling full-attention K/V in the pool not bit-equal to contiguous | `b5f11699` #3310 f1 | - | oracle-gated, B3 | - |
| InklingParity | test-compute | 4 | KEEP | Inkling residual stream, logits, greedy continuation diverge from transformers | `b22e5f10` #3209 f1 | - | oracle-gated, B3 | only port-parity gate |
| InklingPhraseRewind | test-compute | 2 | KEEP | phrase-ban seek does not replay first-run rows | `5c062af7` #3346 f0 | - | oracle-gated, B3 | - |
| InklingPrefill | test-compute | 4 | SHRINK | Inkling grouped MXFP8/NVFP4 expert GEMM differs from slot GEMV | `ad049efb` #3227 f1 | - | 2 DISABLED_ timing benches | the 2 DISABLED_ benches to the bench target |
| InklingQuality | test-compute | 1 | KEEP | Inkling KLD vs reference logits regresses | `b763f1d1` #3223 f0 | - | skips unless INKLING_KLD set, B3 | - |
| InklingRewind | test-compute | 6 | KEEP | resumed prefix differs from fresh run; request binds to wrong slot | `f72be7f3` #3211 f1 | - | oracle-gated, B3 | - |
| InklingServePath | test-compute | 1 | KEEP | pool runtime differs from contiguous forward on the serve path | `b5f11699` #3310 f1 | - | oracle-gated, B3 | - |
| InklingVisionParity | test-compute | 1 | KEEP | Inkling vision tower features diverge from reference | `569e549a` #3213 f1 | - | oracle-gated, B3 | - |
| LayerNormTest | test-compute | 9 | KEEP | RMSNorm FP32/FP16, residual, row-block, NVFP4 producer wrong | `ff4d81d8` f0 | - | - | - |
| LtGemmEx | test-compute | 5 | KEEP | cuBLASLt strided-batch, grouped MoE, FP8 per-expert scale, KV batched wrong | `f26bad0b` #2950 f0 | - | - | - |
| MLAAbsorb | test-compute | 1 | KEEP | MLA absorbed decode differs from materialized-KV reference | `1bf99888` #803 f1 | - | - | - |
| MLAAttnOutput | test-compute | 6 | KEEP | MLA output wrong for v_head_dim != head_dim, compaction, head gate, padded V | `fd5c7c13` #802 (body) f1 | - | - | - |
| MLAProjection | test-compute | 2 | KEEP | MLA two-step KV projection or latent RMSNorm wrong | `fd5c7c13` #802 (body) f1 | - | - | - |
| MmqQ4kImmaLayout | test-compute | 3 | DELETE | mmq_q4k_imma_reorder (Phase 2C) round trip wrong | `b900f2bf` #255 f0 | - | code reachable only from tests/bench: no caller of mmq_q4k_imma_reorder outside mmq_q4k_imma_tile.cu | production Q4_K IMMA repacks in mmq_q8_imma_q4k.cu (MmqQ8Imma); delete with src/compute/mmq_q4k_imma_{layout,tile}.cu |
| MMVQ | test-compute | 6 | SHRINK | MMVQ Q4_K/Q5_K/Q8_0 GEMV wrong | `11cab48a` f0 | GgufRef.Q4_K/Q5_K/Q8_0_GemvMmvq: same ggml_mmvq kernels vs fp64 at N256 K1024 | - | delete the 3 *_MatchesDp4a (N64 K256, relative to DP4A), keep the 3 *_LargerDims (N2048 K2816); mutants M-MV1..3 Mutant note: M-MV1..3 red by GgufRef.<type>_GemvMmvq and by both kept *_LargerDims cases |
| MoeRouterDecode | test-compute | 5 | KEEP | MoE router decode differs at Flint, bias, wide, odd-row shapes | `1ed60a03` #3135 f0 | - | - | - |
| MoeRouterFusedBench | test-compute | 1 | MOVE(bench) | none: DISABLED_ tile/split sweep | `e66bf1d9` #3001 f1 | - | BENCH, NO-ASSERT, DISABLED_ | benchmark, own target |
| MoeRouterFusedRefusal | test-compute | 1 | KEEP | fused router accepts experts > 128 | `e66bf1d9` #3001 f1 | - | - | - |
| MoeRouterFusedTest | test-compute | 8 | KEEP | fused router ids/weights wrong; rows depend on M or split | `e66bf1d9` #3001 f1 | - | - | - |
| MRope | test-compute | 5 | KEEP | multi-axis RoPE text positions differ from single axis; interleaved/sectioned layout wrong | `813afe1e` #1172 (body) f0 | - | - | - |
| Nvfp4PromptStreamK | test-compute | 2 | KEEP | NVFP4 stream-K rows depend on M; narrow tile differs from wide | REG `21cb829b` #3006 f2 | - | - | REGRESSION-ORIGIN |
| Nvfp4TileSwigluQuant | test-compute | 7 | KEEP | fused NVFP4 SwiGLU+quant bytes differ from unfused | `ff59504a` #3024 f0 | - | - | - |
| PenaltyHistAppendTest | test-compute | 1 | KEEP | penalty history append skips/bounds wrong between sample slots | `dc6e5114` #1755 f0 | - | impl-coupled bytes | sampling class |
| PromptLogprobsRows | test-compute | 4 | KEEP | prompt logprob rows wrong (row/slab-major), entropy buffer | `3679fe4e` #2264 (body) f0 | - | - | - |
| QkNormRopeFusedTest | test-compute | 3 | KEEP | fused QK-norm+RoPE differs from norm then RoPE | `92e7c381` #1964 f0 | - | - | - |
| QkNormRopeWarpTest | test-compute | 7 | KEEP | warp QK-norm+RoPE not bit-identical to CTA kernel | `bbd7eef8` #2969 f0 | QkNormRopeFusedTest | - | other kernel |
| RoPETest | test-compute | 10 | KEEP | RoPE FP32/FP16, YaRN, partial, MTP M-RoPE, query temperature wrong | `ff4d81d8` f4 | - | - | - |
| RowwiseTopM | test-compute | 4 | KEEP | row-wise top-m tie break, M=1 argmax, arena exhaustion wrong | `e8c27c17` #1054 f1 | - | - | - |
| SamplerCubPath | test-compute | 1 | KEEP | CUB sampler path samples from previous call's logits | REG `acc40ffc` #1144 f11 | - | - | REGRESSION-ORIGIN |
| SamplerSeeding | test-compute | 2 | KEEP | consecutive seeds draw the same; device salt differs from host seed | REG `e880c647` #1143 f11 | - | - | REGRESSION-ORIGIN |
| SampleVerifyRows | test-compute | 9 | KEEP | speculative verify draws differ from eager sampler; min-p; coupled draft | `def6fcd0` #2710 f0 | - | - | sampling class |
| SamplingAdvancedTest | test-compute | 7 | KEEP | logit bias, DRY, Mirostat v2 move wrong token or lose seed stability | `42f999d4` #1945 f1 | - | 20 ms cudaLaunchHostFunc hold widens a race window, no wall-clock assert | sampling class |
| SamplingTest | test-compute | 30 | KEEP | greedy, top-k, top-p (multiblock, CUB), NaN/inf logits, penalty rows wrong | `7265d56f` f11 | - | - | sampling class |
| ShortConv | test-compute | 1 | KEEP | short conv chunked prefill + batched decode differs from reference | `a6000336` #2426 f0 | - | - | - |
| SoftmaxTest | test-compute | 9 | KEEP | softmax unstable or wrong on large rows, NVFP4 producer | `7265d56f` f1 | - | - | - |
| SwigluQuantizeCutlass | test-compute | 1 | KEEP | CUTLASS SwiGLU+quantize bytes differ from two-step | `dc3abb30` #2585 (body) f0 | - | - | - |
| TopkRadixSampler | test-compute | 3 | KEEP | radix top-k bytes/tokens differ from legacy path | `5e10de3c` #2779 f0 | SamplingTest top-k | 2.45 s | sampling class |
| AnswerLostToReasoning | test-core | 4 | KEEP | breaks: answer_lost_to_reasoning fires only when thinking ate the whole reply, bool cases | f90148c7 server #1442 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AnswerReserveKey | test-core | 4 | KEEP | breaks: answer reserve tokens: default constant, floor, zero and negative clamp, EXPECT_EQ | REG 09e25bb7 fix #1963 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicCacheControl | test-core | 11 | KEEP | breaks: Anthropic cache_control markers map to cache-prompt flags across system/message/tool | REF? 8fc18e2b feat #541 (refs 522,536,538) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicCacheUsage | test-core | 6 | MERGE | breaks: Anthropic usage: spec counters and cache read/creation fields pass through, EXPECT_EQ | REF? 8fc18e2b feat #541 (refs 522,536,538) | same suite in test_anthropic_transform.cpp + test_anthropic_usage.cpp | behaviour | one suite name split over 2 files; merge into one file, 0 cases removed. |
| AnthropicContentBlocks | test-core | 9 | KEEP | breaks: Anthropic content-block validator accepts convertible blocks, refuses unknown/malformed | REG d3982ad8 fix #1929 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicEffort | test-core | 2 | KEEP | breaks: Anthropic effort maps to reasoning_effort, nearest template level, EXPECT_EQ | 6877280d feat #3066 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AnthropicEnvelope | test-core | 5 | KEEP | breaks: Anthropic envelope: path detection, error wrapper per dialect, request-id header/body | REG 7e9f03d6 fix #1702 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicGuidedPassthrough | test-core | 3 | KEEP | breaks: Anthropic shim passes guided-decoding and response_format fields through unchanged | REG 00e3582f fix #1263 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicMessages | test-core | 12 | KEEP | breaks: Anthropic messages to OpenAI shape: system, images, tool_use, tool_result, reasoning | 342e0cf0 test #909 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AnthropicPrefill | test-core | 2 | KEEP | breaks: Anthropic assistant prefill sets continue_final_message only with trailing text | 43d94343 feat #2764 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AnthropicReasoningReport | test-core | 2 | KEEP | breaks: Anthropic usage reports reasoning tokens as output details, finish-detail copy rule | REG 09e25bb7 fix #1963 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicResponse | test-core | 12 | KEEP | breaks: Anthropic response shape: error envelope, text/thinking/tool_use blocks, bad args | 75ab6ba8 test #910 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AnthropicSampling | test-core | 1 | KEEP | breaks: Anthropic sampling fields seed, top_k, temperature reach the chat body, EXPECT_EQ | REG 7f3d3adb fix #3094 | none found | behaviour | protected class (sampling); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicStopReason | test-core | 4 | KEEP | breaks: Anthropic stop_reason: end_turn vs stop_sequence, engine reasons never leak verbatim | REG 7e9f03d6 fix #1702 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicSystemRole | test-core | 6 | KEEP | breaks: Anthropic system role: leading system messages fold into the system prompt | REG 84970818 fix #1261 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicThinking | test-core | 15 | MERGE | breaks: Anthropic thinking config maps to enable_thinking and budget, clamp/signature rules | REG 13278485 fix #719 | same suite in test_anthropic_thinking_blocks.cpp + test_anthropic_transform.cpp | behaviour | one suite name split over 2 files; merge into one file, 0 cases removed (maintenance only). |
| AnthropicToolChoice | test-core | 6 | KEEP | breaks: Anthropic tool_choice maps to OpenAI tool_choice and parallel_tool_calls flags | REF? cf97f674 test #905 (refs 892,901) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AnthropicToolResult | test-core | 2 | KEEP | breaks: Anthropic tool_result is_error labels the tool output for the model | REG 7e9f03d6 fix #1702 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ApiKeyAuth | test-core | 5 | KEEP | breaks: API key auth: Bearer and x-api-key accepted, mismatch and empty key refused | REG 1c1bcb81 Soundness  #772 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ArchIdentity | test-core | 4 | KEEP | breaks: Arch identity: every arch string keeps its enum, one table, HF/GGUF loaders agree | cdf37707 refactor #2492 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Arena | test-core | 8 | KEEP | breaks: Arena allocator: aligned takes, usage, growth, exhaustion, reset, close, typed errors | ff4d81d8 feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ArenaLazy | test-core | 3 | KEEP | breaks: Lazy arena commits nothing until a take reaches it, fixed-region fallback, exhaustion | d8c47857 perf #1981 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AttentionDispatchRules | test-core | 6 | KEEP | breaks: Attention dispatch: FMHA/FA2 head-dim coverage, MLA refusal, MXFP4 decode refusal | REG 11eb5673 fix #1443 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AttnDispatchTable | test-core | 9 | KEEP | breaks: Attention dispatch table: default FA2, decline fallbacks, sinks routing, all-decline throw | REG 1a8e1353 test #585 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AttnRoutingModelCoupling | test-core | 3 | KEEP | breaks: Attention path replay: every tier replays to itself, sinks/no-accept map to none | REG c8532a27 fix #1211 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AttnTierPrecedence | test-core | 2 | KEEP | breaks: Attention tier precedence: FA2 beats FMHA sm120, FMHA beats Blackwell when both accept | REG c8532a27 fix #1211 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AudioConfigTest | test-core | 3 | KEEP | breaks: Audio config: modality flag set by object only, not null or absent, HF config load | REG d3982ad8 fix #1929 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AudioWav | test-core | 3 | KEEP | breaks: WAV reader: PCM16 mono, stereo averaging, extensible float, malformed files refused | 9529acc0 feat #2980 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AwqArch | test-core | 4 | KEEP | breaks: AWQ arch table: offset convention per arch, layer prefix resolution, plain-norm archs | 25fd0e60 feat #1966 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AwqCalibration | test-core | 8 | KEEP | breaks: AWQ calibration file round-trip, foreign/truncated refusal, scale and fold algebra | 41c17ea8 feat #1156 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AwqNormFold | test-core | 6 | KEEP | breaks: AWQ norm fold: unit-offset gain division, BF16 near-zero loss, clamp bounds, vector fold | 25fd0e60 feat #1966 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| AwqSites | test-core | 19 | KEEP | breaks: AWQ site selection: attention/GDN fold groups, group selector, tie map, default groups | 25fd0e60 feat #1966 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| BailingHybridConfig | test-core | 1 | KEEP | breaks: Bailing hybrid config: Ling3Tiny maps KDA, MLA and router fields, EXPECT_EQ | REF? 1bd027ac feat #2979 (refs 2978) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BailingHybridTensors | test-core | 3 | KEEP | breaks: Bailing hybrid tensors: KDA layer fused and MLA renamed, incomplete layer refused | REF? 1bd027ac feat #2979 (refs 2978) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BannedStringsParams | test-core | 1 | KEEP | breaks: Banned-strings param parse: list rules, null/absent handling, status codes | ba0080da feat #2767 | none found | behaviour | protected class (sampling); no overlapping suite found, runtime 0.00C |
| Base64 | test-core | 5 | KEEP | breaks: Base64: RFC 4648 vectors, padding at each residue, decode inverse, all byte values | REF? cf97f674 test #905 (refs 892,901) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BatchVerifyPredicates | test-core | 18 | KEEP | breaks: Batched verify predicates: head slots, economics guard, carry order, history bindings | REG c78aaa67 fix #2002 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BearerAuth | test-core | 8 | KEEP | breaks: Bearer auth: token match accepted, wrong, missing prefix, empty, prefix and long tokens | REG a6a9b0a7 test #717 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BlockPool | test-core | 7 | KEEP | breaks: BlockPool: acquire/release balance, distinct stable blocks, exhaustion ref, churn | REG cbe3d2b3 feat #1106 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BlockPoolSlots | test-core | 2 | KEEP | breaks: BlockPool slots: lowest free id first, shrink drops only the free tail | 8db4e704 perf #3270 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| BlockRef | test-core | 1 | KEEP | breaks: BlockRef move transfers the reference without changing the referent count | REG cbe3d2b3 feat #1106 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BudgetRecount | test-core | 4 | KEEP | breaks: Budget recount: thinking opener counted between markers, prompt-seeded and no-think | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| BufferStamp | test-core | 3 | KEEP | breaks: BufferStamp: live until a view retires it, matches buffer/weight/rows, clear drops it | c36e0bbd refactor #3345 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| CalibrationOutPath | test-core | 4 | KEEP | breaks: Calibration output path: flag wins over config key, empty key does not clobber | REG 66a1f633 fix #1508 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CapabilityRefusal | test-core | 6 | KEEP | breaks: Capability refusal: Inkling refuses loop features, OpenAI/Anthropic envelopes carry param | REG fa48ece9 fix #3301 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CApiContract | test-core | 4 | KEEP | breaks: C API contract: null args invalid, lora unload, kv dtype validation, six dtypes pass | REG eb00defe fix #1917 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CApiEnumBinding | test-core | 6 | KEEP | breaks: C API enum binding: arch ids match public header, unique, round-trip, dtype stable | REG b79a5f67 fix #1206 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CfgParams | test-core | 1 | KEEP | breaks: cfg_scale and cfg_negative_prompt parse rules, status codes, defaults | 50dd22ba feat #2770 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| CgroupHeadroomTest | test-core | 1 | KEEP | breaks: cgroup headroom: limit minus non-reclaimable memory, parse cases, EXPECT_EQ | REG df11bc8a fix #2070 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ChannelSplit | test-core | 4 | KEEP | breaks: Channel split: thought, content, unknown and analysis routing of reasoning output | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CheckpointLimits | test-core | 20 | KEEP | breaks: Checkpoint limits: huge/overflow layer and expert indices, dims, shard names, traversal | REG a3905f77 fix #1694 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.25C |
| CliConfigPins | test-core | 1 | KEEP | breaks: CLI config pins: --perplexity commits KV pool up front, other paths stay growable | REG 332cd637 fix #3185 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ClientErrorLogLine | test-core | 2 | KEEP | breaks: Client error log line: status, route, reason on one line, control bytes stripped | REG f7626195 fix #2281 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CombineImageHash | test-core | 1 | KEEP | breaks: combine_image_hash: order-sensitive, never zero, single image hashes to itself | 5408eb5c feat #1187 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ConfigBadValue | test-core | 5 | KEEP | breaks: Config bad value: bool, int, float and tri-state keys reject unreadable values | REG 0b1f841b fix #1690 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ConfigBinding | test-core | 1 | KEEP | breaks: Config binding: prefill_chunk_size default -1, override reaches the field | REG 0b1f841b fix #1690 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ConfigBindings | test-core | 1 | KEEP | breaks: Config bindings: every example key moves only its own field, count floor | 29b3241f test #2515 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ConfigExplicitKeys | test-core | 1 | KEEP | breaks: Config explicit keys: was_set true only for keys the operator set | a1148fbd feat #1809 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ConstraintValidation | test-core | 17 | KEEP | breaks: Constraint validation: regex/grammar accepted or refused with parser error, all spellings | REG fb524b3c fix #1258 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ContentParts | test-core | 8 | KEEP | breaks: Content parts: text, image, video, input_audio accepted, unknown and malformed refused | REG 68b0d273 fix #1264 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ControlVector | test-core | 4 | KEEP | breaks: Control-vector GGUF load: direction per layer, layer/width/type/truncated refusals | 8d4c4148 feat #2728 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| CorsOrigins | test-core | 3 | KEEP | breaks: CORS allow-origin: no origin by default, exact echo, wildcard allows any origin | REG 60c6468f fix #2431 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CudaErrorsTest | test-core | 6 | KEEP | breaks: CUDA error classes: sticky faults unrecoverable, sync throws naming site, EXPECT_THROW | REG c76d9066 fix #1913 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| CudaStaticReset | test-core | 1 | KEEP | breaks: CUDA static-reset hook registry is populated, EXPECT_GT | ce9552f2 refactor #1207 | none found | impl (registry count) | #1207: guards registrars dropped by the archive linker; nothing else sees an empty hook registry. |
| DecodeGraphCtx | test-core | 6 | KEEP | breaks: Decode graph capture: pow2 buckets, deterministic capture contexts, batch key order | REG 617d2753 fix #950 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DequantAwq | test-core | 3 | KEEP | breaks: AWQ dequant bit-exact vs reference per group/column, GEMM layout shape checks | REF? 7254cb9e Dequantize #2250 (refs 2196,2249) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DequantCap | test-core | 3 | KEEP | breaks: Dequant graph-capture cap: lm_head exempt, eligible plane over cap disables capture | REF? 92e7c381 feat #1964 (refs 1003,1630,1924,1957,1963) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DequantGptq | test-core | 7 | KEEP | breaks: GPTQ dequant bit-exact (asym zeros v1/v2), g_idx group select, zero-format parse | REG b3141829 fix #2252 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DeviceInfoTest | test-core | 4 | KEEP | breaks: Device info: SM count fallback 170, sm120 and PDL cc gates, cached values stable | REG 01f76b7b fix #2748 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DeviceOwnedTest | test-core | 17 | KEEP | breaks: DeviceOwned handles: empty/zero-byte, adopt allocator releases once, async stream release | d95001df refactor #3296 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| DispatchPaths | test-core | 2 | KEEP | breaks: Dispatch path enum: every value named, names distinct within each family | REG c35fad75 fix #1205 | none found | behaviour; 1 case(s) assert via file helper | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DispatchPolicyFwd | test-core | 1 | KEEP | breaks: Dispatch policy: null reads the default, wired policy returned as itself | 3819930e refactor #3325 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| DispatchRecord | test-core | 2 | KEEP | breaks: Dispatch record starts unset, tier flags track their setters, dump gate | REG c35fad75 fix #1205 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DraftRowDraws | test-core | 2 | KEEP | breaks: Draft row draws: level rows take verify seed, greedy top_k 1, empty without sampled row | ffb965c2 perf #3095 | none found | behaviour | protected class (sampling); no overlapping suite found, runtime 0.00C |
| DropNullFields | test-core | 1 | KEEP | breaks: Null JSON fields dropped at top level only, nested nulls kept | REG fdbc49be fix #2143 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| EffectiveCapabilities | test-core | 7 | KEEP | breaks: Effective capabilities: NVFP4 stripped for sub-5-bit or F16 sources, floor supported | REG 83fd9fdf refactor #398 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| EndThinking | test-core | 7 | KEEP | breaks: End-thinking forcing: ChatML newline then closer, Harmony final opener, no-op outside | REF? a90d75a8 feat #2542 (refs 2420) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| EndThinkingRegistry | test-core | 5 | KEEP | breaks: End-thinking registry: unknown id NotFound, pending flagged once, phase sets status | REF? a90d75a8 feat #2542 (refs 2420) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| EngineCapabilities | test-core | 8 | KEEP | breaks: Engine capability gates: dense serves all, Inkling/gated forwards refuse unsupported | REG fa48ece9 fix #3301 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| EngineConfigBridge | test-core | 4 | KEEP | breaks: Engine config bridge: explicit default beats imp.conf, unset falls through, EXPECT_EQ | c3cab661 refactor #3321 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| EngineMetricsCell | test-core | 4 | KEEP | breaks: Metrics cell: zero before publish, copies, no torn concurrent reads, bounded | REG 91c90b96 fix #3306 | none found | behaviour; TIMING: PublishCostIsBounded EXPECT_LT(ns, 5000) over 1e6 publishes (~1000x slack), reader/writer thread race; 0 flakes in 3 runs | REGRESSION-ORIGIN #3306 (11 racy accessors). |
| EngineStreamOpen | test-core | 2 | KEEP | breaks: Engine stream open: result matches the stream, no device fails the open | REG ec3dbcb3 fix #3327 | none found | behaviour; cond. GTEST_SKIP x1 (0 skips in 3 CI runs) | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ErrorEnvelope | test-core | 1 | KEEP | breaks: Error envelope: param and code fields present only when supplied | 75ab6ba8 test #910 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ExecApaWorkspaceBytes | test-core | 1 | KEEP | breaks: APA workspace size follows the carve layout, exact byte value | d6d803fe feat #2635 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ExecGdnDtBytes | test-core | 2 | KEEP | breaks: GDN dt buffer bound holds for every row count up to max tokens, ASSERT_LE | REG abd587b0 fix #2327 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExecKdaKimiScratch | test-core | 1 | KEEP | breaks: KDA scratch: split-QKV and LoRA buffers fit, offsets exact, EXPECT_EQ | 63a9b933 feat #2984 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ExecMaxTokens | test-core | 3 | KEEP | breaks: exec max_tokens clamps to 4096, config fallback, replicates as-built GDN cap | REG 6faa7d54 feat #1110 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExecMaxWeightK | test-core | 1 | KEEP | breaks: Max logical weight K over tensors equals 12288, EXPECT_EQ | REG 6faa7d54 feat #1110 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExecRa2WorkspaceBytes | test-core | 1 | KEEP | breaks: RA2 workspace size follows the carve layout, exact byte value | 567737ed feat #2633 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ExecSsmConvF32 | test-core | 1 | KEEP | breaks: SSM conv f32 region ends inside proj buffer for GDN and KDA, EXPECT_LE | REF? 1bd027ac feat #2979 (refs 2978) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExecSsmZCols | test-core | 4 | KEEP | breaks: SSM z-columns cover the attention gate or the recurrent inner, whichever is larger | REG bf2ed383 fix #1278 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExecT2Demand | test-core | 25 | KEEP | breaks: T2 workspace demand: GEMM scratch formula, cap handling, largest fitting target | REG cbe3d2b3 feat #1106 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExitCodes | test-core | 4 | KEEP | breaks: Exit codes: success 0, taxonomy maps one-to-one, distinct bytes, unknown error to 1 | REG 056c8bc7 fix #1698 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExpertBatchPolicy | test-core | 9 | KEEP | breaks: Expert batch policy: auto cap at device rows, host-expert tier resolves to one | REF? 702a9cd3 feat #2170 (refs 2150) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExpertCacheReinitCpu | test-core | 2 | KEEP | breaks: Expert cache reinit: refused budgets leave cache disabled, empty layer mask refused | REG 64214afa fix #2165 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExpertDestack | test-core | 9 | KEEP | breaks: Expert destack: layout by model type, GPT-OSS/Gemma4 plans, gate-up and down reads | cf96f768 feat #2040 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ExpertPack | test-core | 5 | KEEP | breaks: Expert pack: uniform F16 shape, quantized/mixed refused, failed alloc leaves experts | 4c60482b perf #2983 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ExpertPlacement | test-core | 6 | KEEP | breaks: Expert placement: NVFP4 host path needed only with host-resident expert layers | REG 1e4fad60 fix #1403 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ExtractReasoning | test-core | 6 | KEEP | breaks: Reasoning extraction: think block split, unclosed think, last close tag wins | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| FallbackSite | test-core | 2 | KEEP | breaks: Fallback counters: every hit counted, logged once per load, same-name sites sum | REG 58b7728d fix #3322 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| FfnGraphCache | test-core | 2 | KEEP | breaks: FFN graph cache: row bucket steps and bounds, empty cache runs eager | 5806ee32 perf #2548 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Fim | test-core | 5 | KEEP | breaks: FIM token discovery: synthetic markers, PSM order, GGUF metadata ids win | REF? f60e4267 tools #594 (refs 525) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| FinishReason | test-core | 2 | KEEP | breaks: finish_reason: engine-only values map to the OpenAI enum, enum members pass through | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| FlintConfigTest | test-core | 3 | KEEP | breaks: Flint config load: layer map, router and shared expert, unsupported builds refused | 553e4e7a feat #3129 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| FlintNames | test-core | 1 | KEEP | breaks: Flint tensor names: sink and expert-bias canonical names, EXPECT_EQ | 553e4e7a feat #3129 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ForcedToolEnvelope | test-core | 3 | KEEP | breaks: Forced tool-call envelope literals per family, Harmony and Gemma4 parse as a call | REG f7626195 fix #2281 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ForceThinkEnd | test-core | 9 | KEEP | breaks: Forced think-end: fires at budget while thinking, reserve cap grows answer room | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ForwardProfile | test-core | 2 | KEEP | breaks: Forward profile never runs on a capturing stream, eager only when flagged | REG 3ef8c3fc fix #2297 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Fp8Keep | test-core | 9 | KEEP | breaks: FP8 keep-list: fused group selection, rank parse, byte budget by score, ModelOpt output | 36ea374c feat #3000 | none found | behaviour | no overlapping suite found, runtime 0.01C |
| Fp8LmHead | test-core | 5 | KEEP | breaks: FP8 lm_head: decode matches bit-field reference for all codes, RNE encode, saturation | REF? 48cba09b Per-row FP #2176 (refs 2152,2156,2166) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.09C |
| Fp8Source | test-core | 13 | KEEP | breaks: FP8 E4M3 source: grid decode, block-edge derivation, per-block scale application | 643ca24a imp-quanti #1424 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Fp8SourceAction | test-core | 4 | KEEP | breaks: FP8 source action: refused weights widened, gated flag from caller, lm_head flag | REG 2d02f6cc fix #1450 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| FpBits | test-core | 9 | KEEP | breaks: Half/bf16 bit conversions: known values, round-trip, rescale, subnormals, RNE | 23123977 refactor #1531 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| FpBitsLegacyParity | test-core | 6 | SHRINK | breaks: Exhaustive FP bit-parity vs legacy: all 65536 fp16/bf16 codes, E4M3, MXFP8, ngram | 732b7cb9 cleanup #2757 | FpBits (known values, round trips, subnormals), WeightUploadTraits | impl (frozen pre-merge bodies as oracle); 2.46 s = 56 % of the CI-lane gtest sum | FP-1 (normal-range tie -> always round up): FpBits SURVIVED, red by Fp16EncodeFloatSweep, NgramGatherElementAll16M, WeightUploadTraits; so the sweep stays but its 1.44e9-float window shrinks to every upper mantissa x the tie neighbourhood (low 13 bits 0x0FFF/0x1000/0x1001/0/0x1FFF) x 43 exponents x 2 signs plus the 251-stride (~18M floats, ~0.03 s). Phase 4 must re-run FP-1 against the shrunk sweep. |
| FusedGateQProj | test-core | 4 | KEEP | breaks: Fused gated q_proj finder: gated found, dense left alone, per layer, nested prefix | REF? edf33b7c feat #1280 (refs 1273) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| FuzzCorpus | test-core | 9 | KEEP | breaks: Fuzz corpus runs each parser target over seeded corpora via run_target helper | REF? cf2afce1 feat #1686 (refs 1564,1567,1603) | SafeTensorsHostileHeader, GgufFaultInjection, TokenizerJsonHostileIds (finer, asserted) | behaviour; NO-ASSERT by design: a crash/ASan report is the failure; 0.81 s (Regex corpus 0.57 s); 9 case(s) assert via file helper | drives the 6 fuzz/ targets in the CPU lane (S-28). |
| GbnfDepthCap | test-core | 2 | KEEP | breaks: GBNF nesting depth cap: deep groups refused, ordinary nesting parses | REG c2621e33 fix #1661 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GbnfGrammar | test-core | 23 | KEEP | breaks: GBNF grammar parser refusals: no root, undefined rule, left recursion, bad bounds | a398dd87 feat #1095 | none found | behaviour | protected class (grammar/JSON schema); no overlapping suite found, runtime 0.00C |
| GbnfGrammarTest | test-core | 1 | KEEP | breaks: GBNF matcher stays bounded under deep self-recursion, would_accept battery | REF? d584c530 test #1319 (refs 1096) | none found | behaviour | protected class (grammar/JSON schema); introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GdnChunkparArgs | test-core | 1 | DELETE | breaks: GDN chunk-parallel args struct: fields carried in declaration order, EXPECT_EQ | 50169093 refactor #3343 | compiler: designated initializers at all call sites (0 positional inits in src/) | impl | only case is CarriesEveryFieldInDeclarationOrder; a field swap fails the build at the call sites (LA-1 g++, LA-2/LA-4 nvcc) and is behaviour-neutral otherwise. |
| GdnFusedSplit | test-core | 3 | KEEP | breaks: GDN fused qkvz/ba split regroups per key-head group, refuses bad shape or quantized | REF? e3a54f79 feat #2575 (refs 2410) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GdnProjection | test-core | 2 | KEEP | breaks: GDN projection name matcher: Qwen3.6 recipe names only, role flag vocabulary | 0ef120b0 perf #2050 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| GdnScanArgs | test-core | 2 | SHRINK | breaks: GDN scan args struct: fields in declaration order, omitted fields take defaults | 50169093 refactor #3343 | compiler for field order | impl+defaults | drop CarriesEveryFieldInDeclarationOrder (LA-4 class); keep OmittedFieldsTakeTheOldDefaults. |
| GdnScanBatchedArgs | test-core | 2 | SHRINK | breaks: Batched GDN scan args struct: field order and strides, omitted fields default | REF? 1bbcdfde refactor #3334 (refs 3324) | compiler for field order | impl+defaults | drop CarriesEveryFieldInDeclarationOrder (LA-1/LA-2 class); keep OmittedFieldsTakeTheOldDefaults. |
| Gemma4AudioFeatures | test-core | 5 | KEEP | breaks: Gemma4 audio mel features match HF extractor, clip length refusals, HTK filters | 9529acc0 feat #2980 | none found | behaviour | no overlapping suite found, runtime 0.04C |
| Gemma4AudioLoad | test-core | 5 | KEEP | breaks: Gemma4 audio config and tensor load: slots, tables, missing or wrong-shape refused | 9529acc0 feat #2980 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Gemma4AudioPlaceholders | test-core | 2 | KEEP | breaks: Gemma4 audio placeholder expansion per clip, count mismatch refused and untouched | 9529acc0 feat #2980 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Gemma4EConfig | test-core | 4 | KEEP | breaks: Gemma4 E4B config: per-layer inputs, KV sources, PLI pad, double-wide MLP refused | 8a97b792 feat #2978 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Gemma4EWeights | test-core | 1 | KEEP | breaks: Gemma4 E weights: per-layer input table and shared-KV mapping non-null, EXPECT_NE | 8a97b792 feat #2978 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Gemma4KvSources | test-core | 1 | KEEP | breaks: Gemma4 KV source layers edge cases: empty and nothing-to-share, EXPECT | 8a97b792 feat #2978 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Gemma4RopeConfigTest | test-core | 2 | KEEP | breaks: Gemma4 RoPE config: proportional full-attention table, absent without partial factor | REG 2c807d62 fix #2527 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GemmAlgoCapture | test-core | 3 | KEEP | breaks: cuBLASLt algo probe runs only when query ok and not capturing, first fitting algo | REG b5ec2d04 fix #2398 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GemmFallbackFailsLoud | test-core | 6 | KEEP | breaks: GEMM fallback paths throw loudly on missing scratch, residual, cache or source | REG 1d432cdb fix #3288 | none found | behaviour; cond. GTEST_SKIP x1 (0 skips in 3 CI runs) | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GemmKernelRegistryLink | test-core | 1 | KEEP | breaks: GEMM kernel registry holds exactly the produced tier/shape keys, EXPECT_EQ | REG 811026e0 fix #2318 | none found | impl (registry keys) | #2318: guards static registrars dropped by the archive linker. |
| GemmMode | test-core | 4 | KEEP | breaks: GEMM mode flags default off, decode-row and verify-small-M gating, EXPECT_FALSE | ee184158 refactor #3338 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| GenerationConfigTest | test-core | 4 | KEEP | breaks: generation_config.json: partial fields leave sentinels, do_sample false forces greedy | 7d8a84ca feat #74 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| GgufFaultInjection | test-core | 27 | KEEP | breaks: GGUF fault injection: truncated, bad magic/version, huge counts refused, valid | REG db6c47f0 test #532 | FuzzCorpus.GgufLoader (crash only) | behaviour; TIMING: 2 cases bound load time < 5 s (loose) | hostile-GGUF refusals; GGUF legacy but must keep loading safely (CLAUDE.md). |
| GgufLlama3Bos | test-core | 1 | KEEP | breaks: Llama-3 BPE GGUF defaults add_bos true, EXPECT on tokenizer flag | REG 04020f07 fix #2528 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GgufLoaderTest | test-core | 9 | KEEP | breaks: GGUF loader: missing file, bad magic, truncation, version, zero tensors, arch refusal | ff4d81d8 feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| GgufRopeFreqs | test-core | 3 | KEEP | breaks: GGUF rope_freqs tensor: absent leaves tables empty, Llama3 closed form, bad length ignored | REG 04020f07 fix #2528 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GgufSilentDrops | test-core | 8 | KEEP | breaks: GGUF tensor-name mapping: bias and NextN blocks not silently dropped or clobbered | REG 5c6ffc68 fix #1932 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GptqHessianFile | test-core | 1 | KEEP | breaks: GPTQ Hessian file round-trip of entries, truncation rejected, ASSERT_EQ | REF? e8ddff42 docs #2988 (refs 2986) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GracePeriod | test-core | 6 | KEEP | breaks: Grace-period stop: blocks exit until content seen, hard cap releases, whitespace not | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GraphEligibility | test-core | 16 | KEEP | breaks: Graph eligibility reasons: names distinct, KV pressure only mid-run, demote records first | REG 34573033 fix #1210 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GraphEpoch | test-core | 7 | KEEP | breaks: Graph epoch bump monotonic, stale graph refused and recaptured once, regrow bump rules | REF? 74af550d refactor #3289 (refs 3283) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GraphSlotPool | test-core | 12 | KEEP | breaks: Graph slot pool: open shape, bad capacities, double open, non-overlapping slot buffers | REG 05c6943e feat #1107 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| GrowRows | test-core | 3 | KEEP | breaks: Growable rows commit in steps, refused commit keeps row count, fixed without backend | 5aa85156 perf #3235 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| HarmonySplit | test-core | 4 | KEEP | breaks: Harmony channel split: analysis to reasoning, final to content, commentary as reasoning | REG a7998c91 fix #768 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HealthUnservable | test-core | 4 | KEEP | breaks: Health verdict: floored pool unservable and named, healthy and unknown cases | REF? 70662858 server #1451 (refs 874) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HFConfigGolden | test-core | 1 | KEEP | breaks: HF config golden: all 98 fixtures match stored goldens, mismatch count zero | REF? db9162d5 refactor #2541 (refs 2537) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.03C |
| HfFetchInfo | test-core | 1 | KEEP | breaks: HF repo info parse: siblings, commit sha, gated flag, EXPECT_EQ | REF? 713841d3 Fetch hf #2236 (refs 2200,2210) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HfFetchMock | test-core | 10 | KEEP | breaks: HF fetch on mock server: verify sha, resume with Range, retry, sha mismatch fails | REF? 713841d3 Fetch hf #2236 (refs 2200,2210) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.04C |
| HfFetchParse | test-core | 2 | KEEP | breaks: hf:// URI parse: repo and file accepted, malformed rejected | REF? 713841d3 Fetch hf #2236 (refs 2200,2210) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HfFetchSelect | test-core | 3 | KEEP | breaks: HF file selection: single GGUF auto-picked, several need selector, safetensors set | REF? 713841d3 Fetch hf #2236 (refs 2200,2210) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HfFetchSha | test-core | 1 | KEEP | breaks: SHA-256 of files vs known vectors, EXPECT_EQ | REF? 713841d3 Fetch hf #2236 (refs 2200,2210) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HfFetchUrl | test-core | 1 | KEEP | breaks: HF API and resolve URL builders for repo and revision refs, EXPECT_EQ | REF? 713841d3 Fetch hf #2236 (refs 2200,2210) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Holdback | test-core | 10 | KEEP | breaks: Stop-sequence holdback: flush without overrun, earliest match wins, empty stop skipped | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HoldbackDecision | test-core | 2 | KEEP | breaks: Holdback decision: flush cut on UTF-8 boundary, stop match cuts exactly at the match | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HostEmbedding | test-core | 3 | KEEP | breaks: Host embedding lookup bits match upload staging for bf16, f32, f16 | REF? e7b39b18 perf #2573 (refs 2484) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HostExpertTier | test-core | 6 | KEEP | breaks: Host expert tier: LRU eviction, transient scratch, exchange, aligned and unaligned loads | REF? 41963185 perf #2625 (refs 2621) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.01C |
| HostFileMap | test-core | 1 | KEEP | breaks: Host file map resolves pointers to fd-buffered offsets from maps text, EXPECT_LT | REF? 41963185 perf #2625 (refs 2621) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HostPinnedSites | test-core | 1 | KEEP | breaks: Host pinned-memory sites: acquire, move, reset update byte and count totals | d2ef3783 feat #3249 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| HostRamHeadroomTest | test-core | 1 | KEEP | breaks: Host RAM headroom: 6 GiB floor or 10 percent, unknown total, EXPECT_EQ | b4ca8081 perf #2629 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| HostRegistration | test-core | 8 | KEEP | breaks: Host registration RAII: register once, unregister once, move and reset semantics | REF? 3e003d0e feat #1128 (refs 1124,1125,1127) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HostSpillAuto | test-core | 2 | KEEP | breaks: Host spill auto size: two-thirds of spare RAM in whole GiB, only unset key resolved | cb8bd50e feat #3208 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| HostTaskPool | test-core | 3 | KEEP | breaks: Host task pool: wait sees every task, destructor joins, zero threads still runs | 9702f019 perf #2337 | none found | behaviour | no overlapping suite found, runtime 0.01C |
| ImageDecode | test-core | 9 | KEEP | breaks: JPEG decode bit-matches Pillow, file and memory agree, GIF/BMP refused, corrupt fails | REG 29cdf197 fix #2388 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ImagePlaceholders | test-core | 8 | KEEP | breaks: Image placeholder expansion: per-image count, mismatch refused, sequence unchanged | f12379d0 feat #1177 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ImageTokensBefore | test-core | 3 | KEEP | breaks: Image token offset before a chunk: counts prefix only, clamps, text-only starts at zero | REG 383952d5 fix #1184 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| InflightGate | test-core | 2 | KEEP | breaks: In-flight gate: refuses limit+1, readmits after leave, concurrent entries capped | e89b1c90 server #1944 | none found | behaviour; threads, no wall-clock assertion; threads x1 | no overlapping suite found, runtime 0.01C |
| InklingDense8 | test-core | 1 | KEEP | breaks: Inkling dense8 fit check: VRAM budget arithmetic, KV pool not charged, EXPECT | 5600f36f perf #3228 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingGuardsFailLoud | test-core | 5 | KEEP | breaks: Inkling kernel guards throw loudly: head dim, missing state, router expert range | REF? 86e5cf30 refactor #3347 (refs 3330) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| InklingImageParts | test-core | 4 | KEEP | breaks: Inkling image parts: picture inside text split out, header role kept, expansion count | 569e549a feat #3213 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingKvLayout | test-core | 2 | KEEP | breaks: Inkling KV layout: global apps own one layer each, prefix-cache snapshot key | b5f11699 refactor #3310 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingLoraTargets | test-core | 3 | KEEP | breaks: Inkling LoRA targets: module names map to qkvo, other tensors refused, dims | c041da46 feat #3342 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingPrefixCap | test-core | 2 | KEEP | breaks: Inkling prefix cap: never past slot match, pool holds max-seq plus lookahead rows | b5f11699 refactor #3310 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingRegistry | test-core | 1 | KEEP | breaks: Inkling weight registry: five borrowed fp16 handles per layer, 24-layer count | c041da46 feat #3342 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingSlabLayout | test-core | 6 | KEEP | breaks: Inkling slab layout: slot conv state plus sliding rings, aligned disjoint regions | 99b31105 refactor #3326 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingSlotTable | test-core | 4 | KEEP | breaks: Inkling slot table: rows find slot by request id, unknown id throws, release unbinds | REG fda73be8 fix #3304 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| InklingSnapLayout | test-core | 3 | KEEP | breaks: Inkling snapshot layout: entry bytes and sliding windows, restore onto another slot | REF? fd7300ce feat #3340 (refs 2174) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| InternVLLmNames | test-core | 3 | KEEP | breaks: InternVL LM tensor names: every LM tensor assigned, vision counted apart, Qwen3-VL | e21ab416 feat #2382 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InternVLPreprocess | test-core | 2 | KEEP | breaks: InternVL tile preprocess matches HF within one step, prompt ids match HF processor | 78bf33a6 feat #2387 | none found | behaviour | no overlapping suite found, runtime 0.01C |
| InternVLVision | test-core | 4 | KEEP | breaks: InternVL vision config parse, refusals, loader reads every tensor, HF per-stage match | e341ee4e feat #2384 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| JsonConstrainFsm | test-core | 14 | KEEP | breaks: JSON-constrain FSM rejects a second decimal point in numbers, sim_token_valid battery | REG 3b87ba92 fix #1068 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| JsonConstrainPropertyTest | test-core | 9 | KEEP | breaks: JSON constrainer property battery: valid docs and prefixes accepted, bad syntax rejected | REF? dd4be83a test #1070 (refs 1002,1014,1067,1068,517) | none found | behaviour; 2 case(s) assert via file helper | protected class (grammar/JSON schema); introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.01C |
| JsonEscape | test-core | 1 | KEEP | breaks: JSON escaping: control chars, quotes, UTF-8 round-trip, EXPECT_EQ | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| JsonNestingDepth | test-core | 5 | KEEP | breaks: JSON nesting depth counter: plain nesting, braces inside strings ignored | REG aa8818c0 fix #1666 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| JsonNumberBounds | test-core | 6 | KEEP | breaks: JSON number bounds: exact compare, exclusivity, tighten, prefix reachability, brute force | a653dac7 feat #2708 | none found | behaviour | protected class (grammar/JSON schema); no overlapping suite found, runtime 0.01C |
| JsonOutTest | test-core | 9 | KEEP | breaks: JSON writer: empty object, key order, precision, escapes, UTF-8 pass-through | f851627c feat #1715 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| KdaReference | test-core | 4 | KEEP | breaks: KDA reference recurrence matches FLA naive, gate bound and softplus closed form | REF? 1bd027ac feat #2979 (refs 2978) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| KimiLinearConfig | test-core | 2 | KEEP | breaks: Kimi-Linear config: KDA, MLA nope and router mapping, unsupported refused | 63a9b933 feat #2984 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| KimiLinearTensors | test-core | 4 | KEEP | breaks: Kimi-Linear tensors: dense QKV fuse, LoRA rename, NVFP4 scales kept, broken layout refused | 63a9b933 feat #2984 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| KVAccounting | test-core | 5 | KEEP | breaks: KV accounting: for-accounting geometry, salt seeds lookup, probe count, reuse stops at | 5da361cc test #1860 | none found | behaviour | protected class (paged KV / prefix cache); no overlapping suite found, runtime 0.00C |
| KvAdmission | test-core | 5 | KEEP | breaks: KV admission estimate: P90 of finished outputs with floor, expected length admits more | REF? 13336809 perf #2547 (refs 2486) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| KvBlockSize | test-core | 2 | KEEP | breaks: KV block size: multiples of 16 up to 256 served, all others refused with a reason | b018362f mem #1941 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| KVCacheManagerTest | test-core | 47 | KEEP | breaks: KV cache manager: allocate/rollback/append/free, LRU eviction, live seq never evicted | ff4d81d8 feat | none found | behaviour | protected class (paged KV / prefix cache); no overlapping suite found, runtime 0.00C |
| KVCacheTest | test-core | 8 | KEEP | breaks: KV cache: construction, block alloc/free, refcount, NVFP4 head-dim reject, bytes per block | ff4d81d8 feat | none found | behaviour | protected class (paged KV / prefix cache); no overlapping suite found, runtime 0.00C |
| KvDtypeGolden | test-core | 5 | KEEP | breaks: KV dtype golden: block bytes and planner byte math per layer, FP4 head-dim multiple | 1edcb2a7 refactor #2518 | none found | behaviour | protected class (paged KV / prefix cache); no overlapping suite found, runtime 0.00C |
| KvDtypeInfo | test-core | 2 | KEEP | breaks: KV dtype info flags and block bytes match golden table, EXPECT_EQ | 1edcb2a7 refactor #2518 | none found | behaviour | protected class (paged KV / prefix cache); no overlapping suite found, runtime 0.00C |
| KvFp8Scale | test-core | 10 | KEEP | breaks: KV FP8 scale: mode parse, pow2 ceil, amax/checkpoint sources, histogram bins, EXPECT | REF? 5f7291f9 feat #3016 (refs 89,91) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.01C |
| KVHostSpillTest | test-core | 8 | KEEP | breaks: KV host spill tier: budget off, committed round-trip, LRU replace, chunk pin, bindings | 5c96d8cd feat #2262 | none found | behaviour | protected class (paged KV / prefix cache); no overlapping suite found, runtime 0.00C |
| KvPoolVerdict | test-core | 6 | KEEP | breaks: KV pool verdict: full sequence sufficient, clamped or floored messages, partial block | REG dd0cef06 fix #1286 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| KvReplan | test-core | 10 | KEEP | breaks: KV replan from cached measurement: ceiling gains freed reserve, live reserve floor | REG 5ea8e547 fix #2491 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| KvResidualSizing | test-core | 8 | KEEP | breaks: KV residual sizing: plan fit, residual clamp vs floor, headroom boundary exact | REG 384200c7 fix #1253 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| KvScaleTensor | test-core | 3 | KEEP | breaks: KV scale tensor names parse to K/V slot, host scalar read, matched-step routing | REF? 5f7291f9 feat #3016 (refs 89,91) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| KVSessionPins | test-core | 9 | KEEP | breaks: KV session pins: prefix hit under pressure, LRU unpinned first, TTL and budget unpin | f123208b feat #2496 | none found | behaviour | protected class (paged KV / prefix cache); no overlapping suite found, runtime 0.00C |
| KvWriteRopeArgs | test-core | 2 | SHRINK | breaks: KV write RoPE args struct: fields in declaration order, omitted fields default | REF? 1bbcdfde refactor #3334 (refs 3324) | compiler for field order | impl+defaults | drop CarriesEveryFieldInDeclarationOrder (LA-1/LA-2 class); keep OmittedFieldsTakeTheOldDefaults. |
| LaneNeeds | test-core | 4 | KEEP | breaks: Lane needs: field combinations map to old lanes, logprobs not forced, mid-run bits | d512b911 refactor #3320 | none found | behaviour | no overlapping suite found, runtime 0.01C |
| LatencyLadder | test-core | 5 | KEEP | breaks: Latency histogram ladders: seconds vs ITL buckets, cumulative sums, negatives clamp | REG 70cf2778 fix #1705 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LayerHostKeep | test-core | 8 | KEEP | breaks: Layer host-keep: Qwen3-8B GPU-layers 20 keeps 16 layers on host, dense-only gate | REF? 6ae8527a feat #2320 (refs 2298) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LayerOffloadPlan | test-core | 5 | KEEP | breaks: Layer offload plan: Q8_0 block-sized bytes, 16 host layers planned, init refusals | REG 7cc460fd fix #2299 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LeakedStaticTest | test-core | 6 | KEEP | breaks: Leaked-static holder: destructor never runs, owners release once, reset idempotent | ff1dfae0 refactor #3297 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Lfm2Names | test-core | 1 | KEEP | breaks: LFM2 canonical tensor names: modules and suffixes map, EXPECT_EQ | a6000336 feat #2602 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LibraryForwardWindow | test-core | 2 | KEEP | breaks: Library forward window: tracked growth is no library claim, bytes never wrap | REG a7300795 fix #3268 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LibraryReserveCache | test-core | 9 | KEEP | breaks: Library reserve cache: key stability, file round-trip, missing key, remeasure replaces | REF? 5b094b7d feat #1119 (refs 1112) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LibraryReserveMeasurement | test-core | 3 | KEEP | breaks: Library reserve measurement: no window means none, larger of window and residual | REG 1ff84c60 fix #2352 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LlmCompressorFormatDetect | test-core | 8 | KEEP | breaks: llm-compressor format detect: ModelOpt preferred, recipe YAML and compressed-tensors | ff6fb633 feat #63 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LlmCompressorIgnoreList | test-core | 3 | KEEP | breaks: llm-compressor ignore list: config.json names replace recipe patterns, router-proj rule | REG 6d8e3f1c fix #1970 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LlmCompressorKeepVision | test-core | 6 | KEEP | breaks: llm-compressor keep-vision: visual prefix verbatim, vision suffixes kept, MTP skipped | REF? 4c5ac7e0 feat #1384 (refs 1379) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LlmCompressorRecipe | test-core | 6 | KEEP | breaks: llm-compressor recipe parse: Gemma4 and Qwen3.6 recipes ok, non-NVFP4 scheme rejected | ff6fb633 feat #63 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LlmCompressorTranslate | test-core | 17 | KEEP | breaks: llm-compressor tensor-name translate: packed and global-scale renames, vision skipped | ff6fb633 feat #63 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LlmCompressorUnused | test-core | 4 | KEEP | breaks: llm-compressor unused-tensor rule: MTP skipped not unused, drop only when no consumer | 16e60fab Restore th #1434 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LmHeadRoute | test-core | 3 | KEEP | breaks: LM-head route decision table, NVFP4 not FP8-cached, draft head route, EXPECT_EQ | f1bceba1 refactor #3314 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LoaderRopeParity | test-core | 7 | KEEP | breaks: Loader RoPE parity: Gemma4, Llama3, Granite, Qwen3, Gemma3 params match reference | REF? 966559cd test #2533 (refs 2519,2527,2530,2531) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.01C |
| LogitBiasVocabError | test-core | 1 | KEEP | breaks: Logit-bias vocab check: id at or above vocab size named in the error | REG 2adebaf2 fix #2104 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LogitLensParams | test-core | 1 | KEEP | breaks: Logit-lens params parse: lens count, logprobs flag, EXPECT_EQ | 2915d2da feat #2766 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LogLevel | test-core | 2 | KEEP | breaks: Log level parse: every word case-insensitive, anything else rejected | ff4d81d8 feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| LogOnce | test-core | 5 | KEEP | breaks: LogOnce latch: fires once then after rearm, exactly one thread wins a race | REG 3164523c fix #3311 | none found | behaviour; threads x1 | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LogprobShapes | test-core | 6 | KEEP | breaks: Logprob response shapes: chat array, completions parallel arrays, offsets, streaming | REG d979e94a fix #1700 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LogprobTokens | test-core | 3 | KEEP | breaks: Logprob token JSON: valid UTF-8 unchanged, half codepoint serialises, raw bytes | REG d979e94a fix #1700 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LoraDimsTest | test-core | 5 | KEEP | breaks: LoRA dims: projection widths follow base, GQA kv narrower, FFN targets refused without | REG d3414fa4 fix #1914 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| LtGemmArgs | test-core | 3 | SHRINK | breaks: cuBLASLt GEMM args struct: fields in declaration order, GemmEx unstrided batch of one | f2bcd7b5 refactor #3337 | compiler for field order | impl+defaults | drop CarriesEveryFieldInDeclarationOrder (LA-1: swap m/n = BUILD_RED at 9 call sites); keep OmittedFieldsTakeTheOldDefaults and GemmExForcesOneUnstridedGemm (default values are behaviour). |
| LtGemmBatchedArgs | test-core | 2 | SHRINK | breaks: Batched cuBLASLt GEMM args struct: field order and defaults, EXPECT_EQ | f2bcd7b5 refactor #3337 | compiler for field order | impl+defaults | drop CarriesEveryFieldInDeclarationOrder (LA-1 class); keep OmittedFieldsTakeTheOldDefaults. |
| LtRouteTrace | test-core | 3 | KEEP | breaks: cuBLASLt route trace labels: gemm precedence, FP8 label, note and dump follow flag | 2280df5b docs #2999 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MemAccountKeyed | test-core | 2 | KEEP | breaks: Keyed memory accounting: free uncharges exactly charged bytes, unknown pointer no-op | REG 41b137ce fix #2355 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MemAccountRanges | test-core | 1 | KEEP | breaks: Memory account ranges: listed until their pool drops them, counts and bytes, EXPECT | 70f02782 feat #3246 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MeminfoParseTest | test-core | 2 | KEEP | breaks: meminfo parse: MemAvailable, MemTotal and cgroup limit, EXPECT_EQ on parsed values | ce35ae0d feat #954 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MemoryPlan | test-core | 25 | KEEP | breaks: Memory plan battery: dense reference fits, library reserve line item, KV residual, tight | REG cbe3d2b3 feat #1106 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MLAConfig | test-core | 6 | KEEP | breaks: MLA config: YaRN mscale, DeepSeek-V2-Lite fields parse, MLA profile selection | REG fd5c7c13 feat #802 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MLATensorKindMatcher | test-core | 1 | KEEP | breaks: MLA tensor kind matcher: kv_a and kv_a_layernorm projection kinds, EXPECT_EQ | REG fd5c7c13 feat #802 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MLAWeightMap | test-core | 5 | KEEP | breaks: MLA weight map: DeepSeek dense, router, routed and shared experts, attention names | REG fd5c7c13 feat #802 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ModelContext | test-core | 5 | KEEP | breaks: Model context length: configured wins, declared capped at auto ceiling, config | 8ce2cab9 feat #3153 | none found | behaviour; not flaky: steady_clock only names a temp dir | no overlapping suite found, runtime 0.00C |
| ModelIdForPath | test-core | 1 | KEEP | breaks: Model id from path: HF snapshot dir maps to repo id, EXPECT_EQ | REG 22789f23 fix #3110 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ModelLifetimeGate | test-core | 8 | KEEP | breaks: Model lifetime gate: exclusive waits for readers, timeout stays usable, stale | REG 4d7eebf8 fix #3291 | none found | behaviour; TIMING: sleep_for(50ms) window + 5 s eventually() deadline; the sleep can only miss a bug, not false-fail; 0.08 s | REGRESSION-ORIGIN (swap waits for in-flight handlers). |
| ModelNamePolicy | test-core | 5 | KEEP | breaks: Model name policy: basename looked up in models dir, repo-id shape, absolute/traversal | REG c852ef40 fix #1912 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ModeloptMixedPrecision | test-core | 5 | KEEP | breaks: ModelOpt mixed-precision table: NVFP4 table accepted, missing or non-NVFP4 refused | d27b2591 feat #1385 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ModelProfileTable | test-core | 4 | KEEP | breaks: Model profile table: one row per arch, SWA windows, loader trait lists, no-SWA mask | cdf37707 refactor #2492 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ModelScan | test-core | 4 | KEEP | breaks: Model scan lists GGUF and safetensors dirs, symlink loops and unreadable dirs tolerated | REG 5d550496 fix #2888 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ModelSwapState | test-core | 4 | KEEP | breaks: Model swap state: swap or suspend under a lease answers 503 and frees nothing | REG fc1bb711 test #3309 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.10C |
| MoeGroupLimit | test-core | 4 | KEEP | breaks: MoE group-limited routing matches Ling gate golden, stable under fp16 bias | REF? 1bd027ac feat #2979 (refs 2978) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MoeLaunchRows | test-core | 4 | KEEP | breaks: MoE launch rows: one-hot takes all rows, skew ratio, empty inputs zero | 0dc07438 feat #1741 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MoeNvfp4CacheRule | test-core | 4 | KEEP | breaks: MoE NVFP4 whole-layer cache rule: lone eligible down not cached, dense not cached | REG 76ae2669 fix #2637 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MoePrefillSelect | test-core | 2 | KEEP | breaks: MoE prefill: fused dp4a and IMMA qkind refuse unsupported qtypes, EXPECT_THROW | REG afb4a139 fix #2543 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MoePrefillTable | test-core | 7 | KEEP | breaks: MoE prefill path table: device args, GPT-OSS staged/unstaged, grouped and legacy fallback | REG 1a8e1353 test #585 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MoeRoutingModelCoupling | test-core | 2 | KEEP | breaks: MoE prefill path replay: every tier replays to itself, GPT-OSS to grouped | REG 553a2d79 fix #1217 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MoeTierPrecedence | test-core | 3 | KEEP | breaks: MoE prefill tier precedence: device args over grouped over legacy, entry gate first | REG 553a2d79 fix #1217 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MRopeConfig | test-core | 4 | KEEP | breaks: M-RoPE config: split read from rope params or scaling, text-only off, malformed ignored | REG 813afe1e feat #1172 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MRopePositions | test-core | 11 | KEEP | breaks: M-RoPE positions: text ascending on 3 axes, image row/col, continuation start | c83a2b9f feat #1173 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MtpAttnGuard | test-core | 1 | KEEP | breaks: MTP attention guard: zero KV heads with weights returns false with a log message | REF? 125c2a84 build #2293 (refs 2210,2288) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MtpAuto | test-core | 28 | KEEP | breaks: MTP auto sentinel: default auto, single-stream and concurrent pair/pool choices, 28 cases | a1148fbd feat #1809 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MtpDraftRows | test-core | 4 | KEEP | breaks: MTP draft rows: copy holds prefix or every row, FP8 head and cutlass head vocab | 7b5c26bf perf #3097 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MtpExternalHead | test-core | 1 | KEEP | breaks: MTP external head attaches only a matching Qwen head file | 4d183d5b feat #2852 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MtpHeadGraph | test-core | 7 | KEEP | breaks: MTP head graph: eager first use, capture then replay, ring window and resume rules | 173d66be perf #3045 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MtpKvSlots | test-core | 2 | KEEP | breaks: MTP KV slots: slot committed at first use, refused commit reported, ASSERT_NE | 7b40f35a perf #3247 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MtpLayoutDispatch | test-core | 5 | KEEP | breaks: MTP layout dispatch: Qwen4-Exp 310 names mapped, missing expert scale incomplete | REF? 7687c3ce feat #2173 (refs 2150,55513) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MtpNameRule | test-core | 4 | KEEP | breaks: MTP name rule: both checkpoint spellings accepted, main-model tensors rejected | REF? f39441d0 mtp #1475 (refs 1384) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MtpProbe | test-core | 5 | KEEP | breaks: MTP probe: sidecar, sharded and single-file layouts found, non-head and no-head quiet | REF? f39441d0 mtp #1475 (refs 1384) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| MtpSiblingHead | test-core | 1 | KEEP | breaks: MTP sibling head completes only from a byte-equal neighbour file | cd446ac2 feat #3197 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| MtpWorkspaceBytes | test-core | 2 | KEEP | breaks: MTP workspace bytes: Qwen3.8 head at server defaults, NVFP4 body copy priced for dense | 11088fbb perf #3040 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Mxfp8Widen | test-core | 4 | KEEP | breaks: MXFP8 widen: element decode exact, pair to bf16, NaN and NVFP4 scales skipped, E4M3 | REF? af7e035b feat #2586 (refs 2475) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NGramContext | test-core | 4 | KEEP | breaks: N-gram context: EOS at sequence start, spans prompt and output, sequences isolated | REF? 702a9cd3 feat #2170 (refs 2150) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NgramDraft | test-core | 8 | KEEP | breaks: N-gram draft: empty without repeat, simple repeat, truncates at K, prefers longer and | REF? 5e878b69 feat #668 (refs 553,667) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NGramHash | test-core | 4 | KEEP | breaks: N-gram hash: EOS-filled start, shift blocked by EOS, carried equals one-shot, Python | dc05d8b8 feat #2057 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| NonStreamReasoningTokens | test-core | 5 | KEEP | breaks: Non-stream reasoning token count: between markers, prompt opener from token zero, fallback | REG 09e25bb7 fix #1963 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NpyWriterTest | test-core | 4 | KEEP | breaks: NPY writer: parseable header, spec-padded header, row-major payload, shape for one row | REG b5e37182 Qwen3.8-27 #1750 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Nvfp4DequantWs | test-core | 4 | KEEP | breaks: NVFP4 dequant workspace: none when absent or too small, deferred take runs once | f2268df8 refactor #3329 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| NvFP4ExpertKey | test-core | 1 | KEEP | breaks: NVFP4 expert key matcher: per-expert MoE weights only, EXPECT_TRUE | REF? 8a7bd8ca feat #1407 (refs 1370,1403) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NvFP4HostExpertsServable | test-core | 2 | KEEP | breaks: NVFP4 host experts servable: promoted host projection accepted, staging-unusable refused | REF? 8a7bd8ca feat #1407 (refs 1370,1403) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NvFP4IgnoreList | test-core | 5 | KEEP | breaks: NVFP4 ignore list enforced, missing global scale refused, inventory counts match | REG ba88b6ba fix #1962 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NvFP4MergedScaleProvenance | test-core | 7 | KEEP | breaks: NVFP4 merged-scale provenance: fused split recorded, separate tensors no split, slice | REG ba88b6ba fix #1962 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Nvfp4OutputBytes | test-core | 2 | KEEP | breaks: NVFP4 output bytes: packed nibbles, micro scales, tensor scale, empty matrix refused | 43ecc0b4 imp-quanti #1423 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Nvfp4ScratchKey | test-core | 1 | KEEP | breaks: NVFP4 scratch key: malformed index resolves to nothing, EXPECT_EQ | REG ec741f7d fix #2747 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NvFP4SlotLayout | test-core | 5 | KEEP | breaks: NVFP4 slot layout: expert ranges, uint2 load alignment, scale inside slot, no overlap | REF? 8a7bd8ca feat #1407 (refs 1370,1403) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Nvfp4Undeclared | test-core | 1 | KEEP | breaks: NVFP4 undeclared module: packed codes without declaration refused and named | 553e4e7a feat #3129 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| OffloadSpecDefaults | test-core | 5 | KEEP | breaks: Offload spec defaults: offload turns recycling on at width 8, operator keys win, MC rows | 09797f27 perf #2780 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| OutOfRangeMessage | test-core | 2 | KEEP | breaks: Out-of-range message: penalty kernel bounds refused, bounds and defaults accepted | REG fdbc49be fix #2143 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| OutputMarkersInkling | test-core | 8 | KEEP | breaks: Inkling output markers: think tags kept only, tool-call header dropped, args renamed | b22e5f10 feat #3209 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| PagedDecodeArgs | test-core | 2 | SHRINK | breaks: Paged decode args struct: fields in declaration order, omitted fields default, EXPECT_EQ | f06e26c5 refactor #3324 | compiler for field order | impl+defaults | drop CarriesEveryFieldInDeclarationOrder (LA-2: swap n_heads/n_kv_heads = BUILD_RED in attention_paged_fp8.cu); keep OmittedFieldsTakeTheOldDefaults. |
| PagedDecodeTensorArgs | test-core | 1 | KEEP | breaks: Paged decode tensor args: omitted fields take defaults, kv_scale 1.0, EXPECT_EQ | f0a6af1a refactor #3332 | none found | impl+defaults | only the defaults case: kv_scale 1.0 and omitted-field defaults are read by 21 call sites. |
| PagedFp8Decode | test-core | 2 | KEEP | breaks: Paged FP8 decode fast kernel only at head dim 128, serving vs fast question | fd13ada8 perf #1872 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| PagedQ8Epilogue | test-core | 3 | KEEP | breaks: Paged Q8 epilogue taken once per call, not outside its shape, nothing armed never taken | 9784bf3d refactor #3316 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| PagedSplitKScratch | test-core | 3 | KEEP | breaks: Paged split-K scratch bytes per split, null scratch never splits, small scratch caps | f0a6af1a refactor #3332 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ParseCompletionPrompt | test-core | 2 | KEEP | breaks: Completion prompt parse: string, ids and one-element lists accepted, batches refused by | REG 0837d71a fix #2129 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ParseGemmaToolCallBody | test-core | 1 | KEEP | breaks: Gemma tool-call body parser rejects malformed bodies, EXPECT_FALSE | REF? c93fb092 feat #820 (refs 818) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ParseLogitBias | test-core | 4 | KEEP | breaks: Logit-bias parse: absent empty, valid kept, malformed entries error not drop, cap refuses | REG 2adebaf2 fix #2104 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ParseMaxTokensField | test-core | 5 | KEEP | breaks: max_tokens parse: default, max_tokens and max_completion_tokens precedence, nulls ignored | REF? c93fb092 feat #820 (refs 818) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ParseStopField | test-core | 5 | KEEP | breaks: Stop field parse: string or array up to 16, truncation reported, non-strings skipped | REF? c93fb092 feat #820 (refs 818) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ParseStreamToolBody | test-core | 2 | KEEP | breaks: Stream tool-call body parse: Gemma body, JSON then XML fallback, failure | REF? c93fb092 feat #820 (refs 818) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| PathWithin | test-core | 4 | KEEP | breaks: path_within: inside vs outside, sibling prefix outside, traversal resolved before compare | REG c852ef40 fix #1912 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| PhaseFixture | test-core | 26 | KEEP | breaks: Phase fixture: region release on destruction, move ownership once, reset idempotent | REG cbe3d2b3 feat #1106 | none found | behaviour; cond. GTEST_SKIP x4 (0 skips in 3 CI runs) | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.09C |
| PhraseBan | test-core | 6 | KEEP | breaks: Phrase ban: match ends in new text only, token-at-byte, snapshot pick, regex battery | ba0080da feat #2767 | none found | behaviour | protected class (sampling); no overlapping suite found, runtime 0.00C |
| PinnedBuffer | test-core | 13 | KEEP | breaks: Pinned buffer RAII: frees exactly once, failed acquire is empty, zero bytes refused, move | REF? 3e003d0e feat #1128 (refs 1124,1125,1127) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| PreDequantBudget | test-core | 4 | KEEP | breaks: Pre-dequant budget: NVFP4 reservation charged to early phases only, KV growth once, clamp | REG 87d3c6ce fix #1102 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| PrefillGraphCache | test-core | 3 | KEEP | breaks: Prefill graph cache: short keys keep a runner each, long keys share one on consecutive | 3026932c perf #3017 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| PrefillPacing | test-core | 7 | KEEP | breaks: Prefill pacing: uncapped with no decoder, cap off, weight arithmetic, protection scenario | 340efbf8 serving #1950 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ProcessDiag | test-core | 6 | KEEP | breaks: Process diagnostics install: mirrored fields transferred, defaults, log level applied | REG 26a80695 refactor #347 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| PromptLogprobs | test-core | 1 | KEEP | breaks: Prompt logprobs chunk rows formula, EXPECT_EQ against max chunk rows | e898811b feat #2246 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| PromptLogprobsAssembly | test-core | 7 | KEEP | breaks: Prompt logprobs assembly: vLLM and echo shapes, limit cuts alternatives not the target | e898811b feat #2246 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| PromptLogprobsParse | test-core | 5 | KEEP | breaks: Prompt logprobs parse: off when absent, range 0-20, stream refused, engine top-N | e898811b feat #2246 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| PromptTail | test-core | 7 | KEEP | breaks: Prompt tail split: issue split keeps 33 rows, reuse cap leaves minimum rows, short prompt | REG 09e25bb7 fix #1963 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| QknormRopeArgs | test-core | 2 | SHRINK | breaks: QK-norm RoPE args struct: fields in declaration order, omitted fields default | REF? 1bbcdfde refactor #3334 (refs 3324) | compiler for field order | impl+defaults | drop CarriesEveryFieldInDeclarationOrder (LA-1/LA-2 class); keep OmittedFieldsTakeTheOldDefaults. |
| QuantizeCheckpointOut | test-core | 28 | KEEP | breaks: Checkpoint writer: format names, per-format tensor names, global scale, fused group match | 52955176 imp-quanti #1433 | none found | behaviour; 1 case(s) assert via file helper | no overlapping suite found, runtime 0.00C |
| QuantizeFp8Head | test-core | 6 | KEEP | breaks: FP8 head writer: modes and defaults, writes codes and row scales under loader names | 339d2891 feat #2489 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QuantizeKvHint | test-core | 2 | KEEP | breaks: KV FP8 hint allowlist covers measured families only, loader reads the writer hint | 339d2891 feat #2489 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QuantizePolicy | test-core | 7 | KEEP | breaks: Quantize policy: linear weights taken, MLA latent and router refused, embeddings kept | REG 64fdc832 fix #1188 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| QuantizeRecipe | test-core | 5 | KEEP | breaks: Quantize recipe JSON: fixed field order, uncalibrated null, same recipe same bytes | REF? 7d192c61 feat #2522 (refs 2481) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| QuantReport | test-core | 6 | KEEP | breaks: Quant error report: exact values zero error, off-grid distance, worst tensors ordered | 25fd0e60 feat #1966 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QueuedTokenGate | test-core | 7 | KEEP | breaks: Queued-token gate: admits up to cap, refuses over, lease releases once, concurrent | REF? e70abc65 feat #2495 (refs 2408) | none found | behaviour; threads, no wall-clock assertion; threads x1 | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.01C |
| Qwen3VLVisionConfig | test-core | 7 | KEEP | breaks: Qwen3-VL vision config: real checkpoint parse, inconsistent geometry and deepstack refused | 4799db1e feat #1169 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Qwen3VLVisionGrid | test-core | 10 | KEEP | breaks: Qwen3-VL vision grid: merge-block token order, patch coverage, interpolation weights | REF? ef74c024 feat #1171 (refs 1170) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Qwen3VLVisionLoad | test-core | 9 | KEEP | breaks: Qwen3-VL vision load: real checkpoint loads, null slot or shape mismatch refused | f647fbed feat #1170 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Qwen3VLVisionMap | test-core | 7 | KEEP | breaks: Qwen3-VL vision tensor map: every real tensor recognised once, malformed index unknown | 6f49739d feat #1168 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QwenPatchify | test-core | 4 | KEEP | breaks: Qwen patchify: grid and shape, merge-block grouping, channel-major temporal repeat | 3b05a019 feat #1167 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QwenSmartResize | test-core | 7 | KEEP | breaks: Qwen smart-resize: aligned unchanged, min/max pixel scaling, ties round to even | fff9a3d2 feat #1166 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QwenVideoFold | test-core | 1 | KEEP | breaks: Qwen video fold: pads rewritten and token types kept, EXPECT_EQ | 53b68e51 feat #2370 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QwenVideoLayout | test-core | 7 | KEEP | breaks: Qwen video layout: grids, timestamps, placeholder ids and rope positions match HF | d7205cd2 feat #2363 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QwenVideoPatchify | test-core | 4 | KEEP | breaks: Qwen video patchify matches HF video processor within one fp16 ulp, bad input refused | d7205cd2 feat #2363 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| QwenVideoSmartResize | test-core | 1 | KEEP | breaks: Qwen video smart-resize matches HF for stored vectors, EXPECT_EQ | d7205cd2 feat #2363 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| RateLimiter | test-core | 3 | KEEP | breaks: Rate limiter: quiet buckets evicted, window slides, refuses past limit, admits | REG b5b59f7b fix #1696 | none found | behaviour; not flaky: time points injected (t0 + 59 s / 61 s / 120 s); wall clock x2 | #1614 bounded tracker. |
| RateLimitKey | test-core | 3 | KEEP | breaks: Rate-limit key: forwarded-for ignored from untrusted peer, trusted proxy honoured | REG b5b59f7b fix #1696 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ReasoningFinishDetail | test-core | 4 | KEEP | breaks: Reasoning finish detail: fires only when answer lost to reasoning, field written only then | REG 09e25bb7 fix #1963 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| RecurrentSlotForReset | test-core | 3 | KEEP | breaks: Recurrent slot for reset: known id returns its slot, unknown id resets nothing | REG d786a45b fix #3284 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| RegexConstrain | test-core | 15 | KEEP | breaks: Regex constrainer: accepts only matching continuation, multi-char tokens, alternation | REF? 8302ea3e feat #1091 (refs 1090) | none found | behaviour | protected class (grammar/JSON schema); introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| RegexNfaCostBounds | test-core | 4 | KEEP | breaks: Regex NFA cost bounds: huge repeats, nested repeats, deep groups refused, ordinary | REG c2621e33 fix #1661 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.01C |
| RequestTables | test-core | 7 | KEEP | breaks: Request tables: eight tables in finish order, cancel leaves no entry, release idempotent | REG b840f988 fix #3295 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ResponsesContinue | test-core | 2 | KEEP | breaks: Responses continue: merged input equals stateless resend, chaining, string input | REF? 45a12909 feat #2237 (refs 2206) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ResponsesPrefill | test-core | 1 | KEEP | breaks: Responses prefill: continue_final_message forwarded, EXPECT_TRUE/EXPECT_FALSE | 43d94343 feat #2764 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ResponseStore | test-core | 8 | KEEP | breaks: Response store: put/get round-trip, TTL, purge, entry and byte caps evict LRU | REF? 45a12909 feat #2237 (refs 2206) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ResponsesTransform | test-core | 34 | KEEP | breaks: Responses-to-chat transform: input items, tool round-trip, tool nesting, format knobs | REF? 484c2802 feat #841 (refs 820,840) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| RopeOverrideTest | test-core | 10 | KEEP | breaks: RoPE override: off by default, YaRN and linear injection, explicit origin ctx, refusals | 655edb8a feat #923 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| RopeScalingConfigTest | test-core | 18 | KEEP | breaks: RoPE scaling config: Llama3 factor table, tie-embeddings tri-state, per-arch layer maps | 2de77fb0 feat #116 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| RoutingMtpOwnersTest | test-core | 2 | KEEP | breaks: Routing and MTP owners: buffers released once, move leaves no view, free clears views | 157b9718 refactor #3344 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| RuntimeConfigTest | test-core | 24 | KEEP | breaks: Runtime config defaults and bindings: KV block size, lm-dequant bisect, SWA tri-state | 4daf3803 Type-syste #72 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Sae | test-core | 1 | KEEP | breaks: SAE token info: top by activation then id, relative error, EXPECT_FLOAT_EQ | REF? 778587a6 feat #2389 (refs 2381,2388) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SaeDir | test-core | 2 | KEEP | breaks: SAE directory load: layer and tensors load, mismatches refused | b7d7f596 feat #2774 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SafeTensorsAwq | test-core | 3 | KEEP | breaks: SafeTensors AWQ: unsupported variants refused with detected config, 4-bit zero-point load | REG 31fd3a33 fix #2223 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsDtypeTable | test-core | 2 | KEEP | breaks: SafeTensors dtype table: servable rows have matching widths, mismatched wire types refused | REG e86420b1 fix #1660 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsGptq | test-core | 5 | KEEP | breaks: SafeTensors GPTQ: unknown format refused, pre-fix qzeros refused, auto-GPTQ v1/v2 offsets | REG b3141829 fix #2252 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsHighDimTensors | test-core | 1 | KEEP | breaks: SafeTensors high-dim tensors flattened not dropped, conv5d in the captured log | REG f4d284d1 fix #1164 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsHostileHeader | test-core | 6 | KEEP | breaks: SafeTensors hostile header: bad dtype, huge offset, negative/overflow shapes dropped | REG e86420b1 fix #1660 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsMalformedEntryWarnings | test-core | 2 | KEEP | breaks: SafeTensors malformed entries: missing dtype/shape and byte-count mismatch warn | REG eb126946 audit #126 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsRaw | test-core | 2 | KEEP | breaks: SafeTensors raw writer/reader round-trip, truncated and bogus files rejected | 224738ea feat #1081 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SafeTensorsValidateHeaderSize | test-core | 6 | KEEP | breaks: SafeTensors header size validation: below 8 bytes, exceeds file, uint64 overflow, soft cap | REG eb126946 audit #126 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsValidateTensorOffsets | test-core | 7 | KEEP | breaks: SafeTensors tensor offset validation: start/end past file, byte mismatch, zero-size valid | REG eb126946 audit #126 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SafeTensorsWriter | test-core | 5 | KEEP | breaks: SafeTensors writer: round-trip of bytes and header, contiguous offsets, escaped names | 224738ea feat #1081 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SamplingDefaultsParity | test-core | 4 | KEEP | breaks: Sampling defaults parity: CLI equals server per arch, Qwen preset, generation_config | REG 82c14c32 fix #2544 | none found | behaviour | protected class (sampling); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SamplingParamsGolden | test-core | 1 | KEEP | breaks: Sampling params golden: every dialect matches the recorded parse, golden file loads | REF? 3f29858c refactor #2493 (refs 1803,2461) | none found | behaviour; cond. GTEST_SKIP x1 (0 skips in 3 CI runs) | protected class (sampling); introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.02C |
| SanitizeForEcho | test-core | 3 | KEEP | breaks: sanitize_for_echo: non-printable replaced, truncation marker, result serialises | REG b5b59f7b fix #1696 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaAdditionalProperties | test-core | 10 | KEEP | breaks: JSON-schema additionalProperties: explicit true, free-form objects, malformed JSON | REG cbe13bdc fix #1735 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaConstrainPropertyTest | test-core | 8 | KEEP | breaks: JSON-schema constrainer property battery: conforming docs and prefixes accepted, missing | REF? dd4be83a test #1070 (refs 1002,1014,1067,1068,517) | none found | behaviour | protected class (grammar/JSON schema); introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.02C |
| SchemaDepthCap | test-core | 2 | KEEP | breaks: JSON-schema depth cap: deep nesting refused, moderate nesting parses | REG c2621e33 fix #1661 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaEnumLiteral | test-core | 2 | KEEP | breaks: JSON-schema enum literals: numbers, booleans, null and mixed legal-token checks | REF? 620bfdf5 feat #2119 (refs 1564) | none found | behaviour | protected class (grammar/JSON schema); introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaEnvelope | test-core | 1 | KEEP | breaks: Schema envelope: forced open bounds leading whitespace, token_legal battery | REG f1e4fd72 fix #2274 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaIntegerBound | test-core | 3 | KEEP | breaks: Schema integer bound: digits stop at int64 width, short ints unaffected, number unbounded | REG 469f8844 fix #1707 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaKeywords | test-core | 12 | KEEP | breaks: Schema allOf/anyOf/optional keywords: value parsed not a string, EXPECT_EQ | a653dac7 feat #2708 | none found | behaviour; 10 case(s) assert via file helper | protected class (grammar/JSON schema); no overlapping suite found, runtime 0.00C |
| SchemaKeywordsProperty | test-core | 2 | KEEP | breaks: Schema keywords property: conforming docs complete, every prefix legal, no dead end | a653dac7 feat #2708 | none found | behaviour; 1 case(s) assert via file helper | protected class (grammar/JSON schema); no overlapping suite found, runtime 0.01C |
| SchemaParserDesync | test-core | 6 | KEEP | breaks: Schema parser desync: additionalProperties variants, enum literals, trailing input error | REG c2621e33 fix #1661 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaUnenforceableKeywords | test-core | 3 | KEEP | breaks: Schema unenforceable assertion keywords refused, annotations ignored, const as enum | REG c2621e33 fix #1661 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SchemaUnicodeEscape | test-core | 4 | KEEP | breaks: Schema JSON unicode escapes: BMP, surrogate pairs, 3-byte escapes decode to UTF-8 | REG b4600d5d fix #1711 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SentencePieceLoader | test-core | 10 | KEEP | breaks: SentencePiece model parse: pieces, scores, trainer spec, truncated protobuf refused | REG 394593a0 feat #128 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ServableContext | test-core | 5 | KEEP | breaks: Servable context: pool wins when smaller, plan stands when pool larger, growable ceiling | REG 9061fde4 fix #1703 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ServerArgs | test-core | 7 | KEEP | breaks: Server args defaults and parses: idle unload, responses store, VRAM budget, limits | c182b76c feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ServerArgsDeathTest | test-core | 3 | KEEP | breaks: Server args death test: garbage values exit via EXPECT_EXIT | REF? 91b48001 feat #2216 (refs 2199) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.08C |
| ServerJsonKeyOrder | test-core | 3 | KEEP | breaks: Server JSON key order kept through parse/dump for tool schemas and tool_use input | REG a448ced8 fix #3150 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ServerVisionInline | test-core | 5 | KEEP | breaks: Vision inline placement: blocks land where parts sat, count mismatch strips and reports | REG 9e85bbe4 fix #3207 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ServingMetadataLayout | test-core | 4 | KEEP | breaks: Serving metadata layout: 256-byte aligned disjoint regions, size bounds, SWA off adds | 3ab8c83c mem #1939 | none found | behaviour; 1 case(s) assert via file helper | no overlapping suite found, runtime 0.00C |
| ShadowPlan | test-core | 16 | KEEP | breaks: Shadow plan: reports both pool ceilings, capped admission sizes KV, zero-KV names context | REG cbe3d2b3 feat #1106 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ShouldQuantize | test-core | 1 | KEEP | breaks: Quantize filter leaves the MTP draft head alone, EXPECT_FALSE | REF? 1d28e315 imp-quanti #1428 (refs 1422) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Sm120FmhaTileSelect | test-core | 3 | KEEP | breaks: SM120 FMHA tile selection: every selected tile has an instance, old holes gone | REG 76457800 fix #2260 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SmallMXqTag | test-core | 3 | KEEP | breaks: Small-M MXFP quantize tag: producer stamp live until buffer touched, consumer adopts own | 6c902d25 refactor #3335 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SnapshotHostDefault | test-core | 4 | KEEP | breaks: Snapshot host default: spill off keeps default, spill raises host tier to half budget | 673c31fb perf #3077 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Span | test-core | 1 | KEEP | breaks: Span: stable widens to device span, subspans inherit the guarantee, arena-backed | REF? 2f6ba4bb docs #592 (refs 540,547,572,574) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SparseGeometry | test-core | 7 | KEEP | breaks: Sparse attention geometry: budget, engage threshold, sink and recent windows per block | REG a383223c fix #1819 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SpecAcceptRows | test-core | 4 | KEEP | breaks: Speculative accept rows: unfinished chunk keeps bonus outside rows, think edge, stop ends | REG 2b6a1d37 fix #3090 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SpecBatchRr | test-core | 3 | KEEP | breaks: Spec batch round-robin verify active only with a batch and a non-recurrent model | REF? 92e7c381 feat #1964 (refs 1003,1630,1924,1957,1963) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SpecField | test-core | 7 | KEEP | breaks: Speculative request field: absent leaves unset, bool meaning kept, depth object, range 400 | REF? 92e7c381 feat #1964 (refs 1003,1630,1924,1957,1963) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SpecGates | test-core | 12 | KEEP | breaks: Speculative gates: MTP, n-gram, token recycling, DSpark enable independently, nothing | 37cd1543 spec #1464 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SpecialTokensMapTest | test-core | 3 | KEEP | breaks: special_tokens_map.json: Mistral object form and Qwen string form parse, missing file | 52d87981 feat #77 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SpecRequest | test-core | 10 | KEEP | breaks: Speculative request resolve: unset takes server default, request lowers depth, false wins | REF? 92e7c381 feat #1964 (refs 1003,1630,1924,1957,1963) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SpecUsage | test-core | 4 | KEEP | breaks: Spec usage block: absent without verify step, counters beside details, decline reported | REG 960b9319 fix #1915 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SpecUsageKeys | test-core | 2 | KEEP | breaks: Spec usage keys: table matches chat shape, copy moves present keys only, EXPECT_EQ | REF? 92e7c381 feat #1964 (refs 1003,1630,1924,1957,1963) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SseChunk | test-core | 5 | KEEP | breaks: SSE chunk JSON: envelope, finish reason and logprobs, completion shape, finish detail | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SseWriter | test-core | 4 | KEEP | breaks: SSE writer frames: content and reasoning keys, id/model escaped, no NUL byte in frame | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SsmScanPrefillArgs | test-core | 1 | DELETE | breaks: SSM scan prefill args struct: fields in declaration order, dtype fields, EXPECT_EQ | 50169093 refactor #3343 | compiler: designated initializers at all call sites (0 positional inits in src/) | impl | only case is CarriesEveryFieldInDeclarationOrder; a field swap fails the build at the call sites (LA-1, LA-2, LA-3). |
| SSMStateLazy | test-core | 3 | KEEP | breaks: SSM state lazy: reserves every slot commits none, ensure commits only that slot | d8c47857 perf #1981 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SsmStatePool | test-core | 4 | KEEP | breaks: SSM state pool: Qwen3.8 geometry sized for allocator, tail alignment, failure fatal only | REG b954e390 fix #1960 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SsrfClassifier | test-core | 9 | KEEP | breaks: SSRF classifier: loopback, cloud metadata, private ranges, v4-mapped and unspecified | REG cf961159 fix #1665 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SsrfFetch | test-core | 5 | KEEP | breaks: SSRF fetch: remote fetch off by default, private hosts refused even when enabled, scheme | REG cf961159 fix #1665 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StackedExperts | test-core | 3 | KEEP | breaks: Stacked experts found under each naming convention, 3-D and non-float tensors ignored | REG 64fdc832 fix #1188 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StepSeed | test-core | 2 | KEEP | breaks: Step seed: seeded request adds output count, unseeded keeps one base, EXPECT_EQ 10 | REG 52167c0c fix #2013 | none found | behaviour | protected class (sampling); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StopMask | test-core | 7 | KEEP | breaks: Stop mask: masks in think with budget and during grace, released on content or hard cap | REG e4bef19e fix #1990 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StreamEquality | test-core | 2 | KEEP | breaks: Streaming equals non-stream: concat equals full decode, stop truncation matches non-stream | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StreamMultibyteSim | test-core | 1 | KEEP | breaks: Stream multibyte: split codepoint never emitted as half, UTF-8 buffer empty at end | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StreamReasoningSplit | test-core | 21 | MERGE | breaks: Stream reasoning split: second think pass stays out of content, reentry across pieces | REG 5de32550 fix #793 | same suite in test_sse_stream_utils.cpp + test_stream_reasoning_split.cpp | behaviour | one suite name split over 2 files; merge into one file, 0 cases removed. |
| StreamSim | test-core | 1 | KEEP | breaks: Stream sim: stop match split across chunks is caught, emitted text and stopped flag | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StrictCliNumbers | test-core | 2 | KEEP | breaks: Strict CLI numbers: whole value or refused, valid values still parse | ca32f031 cleanup #2754 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| StrictCliNumbersDeathTest | test-core | 1 | KEEP | breaks: Strict CLI numbers death test: malformed value exits 1 naming the flag | ca32f031 cleanup #2754 | none found | behaviour | no overlapping suite found, runtime 0.05C |
| StrictToolConstraint | test-core | 5 | KEEP | breaks: Strict tool constraint: loose tool does not disable strict one, unnamed tool refused | REG cbe13bdc fix #1735 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StripChannelHeaders | test-core | 1 | KEEP | breaks: Strip channel headers: header tokens removed, body kept, EXPECT_EQ | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| StripThinkBlock | test-core | 4 | MERGE | breaks: Strip think block: unclosed trailing or leading think cleared, block removed, answer kept | REF? 546454b6 test #564 (refs 510,557) | same suite in test_reasoning_reconcile.cpp + test_sse_stream_utils.cpp | behaviour | one suite name split over 2 files; merge into one file, 0 cases removed. |
| StructuredOutputThinking | test-core | 5 | KEEP | breaks: Structured output thinking: whole-reply constraint excludes thinking, unconstrained may | REF? e6af8f08 Fix empty  #1431 (refs 934) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SuffixDraft | test-core | 17 | KEEP | breaks: Suffix draft: empty without repeat, repeat found, majority vote beats recency, truncation | REF? c719f85d feat #848 (refs 824) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| SuffixShadow | test-core | 5 | KEEP | breaks: Suffix shadow: line waits for outcome, full agreement no draft, declined step replaced | 21fc346c diag #3093 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| SystemFingerprint | test-core | 2 | KEEP | breaks: system_fingerprint stable per model, differs across models, empty name still yields one | REG d979e94a fix #1700 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TensorKindMatcher | test-core | 8 | KEEP | breaks: Tensor kind matcher: attention, FFN, MoE, GDN, norms, embeddings, unknown names | c8763ade weight-sto #27 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| TensorKindTable | test-core | 7 | KEEP | breaks: Tensor kind table: entry per kind, GDN and norms restricted to FP16/FP32 tiers | c8763ade weight-sto #27 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| TensorTest | test-core | 10 | KEEP | breaks: Tensor basics: default, numel, nbytes, strides, reshape, dtype names, scalar | ff4d81d8 feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| TextThink | test-core | 5 | KEEP | breaks: Text think-state machine: enter/exit across split tokens, marker survives window eviction | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ThinkCloser | test-core | 1 | KEEP | breaks: Think closer: message end after answer does not mask stop, grace holds stop back | REG 587fc674 fix #3221 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ThinkingReconcile | test-core | 7 | KEEP | breaks: Thinking reconcile: open prefix turns thinking on, closed block defaults off, explicit | REG 38698655 fix #937 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ThinkLimit | test-core | 5 | KEEP | breaks: Think limit: graph and host paths agree, reserve follows config, cap bounds default | REG 09e25bb7 fix #1963 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ThinkSeedFromPromptTail | test-core | 4 | KEEP | breaks: Think seed from prompt tail: trailing opener seeds both flags, closed block seeds neither | REG 09e25bb7 fix #1963 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ThinkTokenAccept | test-core | 6 | KEEP | breaks: Think token accept: control and user-defined types accepted, normal and absent rejected | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenCategory | test-core | 2 | KEEP | breaks: Token category: non-ASCII counts as string content, string-ending tokens carry quote | REG 232319d4 fix #1200 | none found | behaviour | protected class (grammar/JSON schema); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenExtras | test-core | 2 | KEEP | breaks: Token extras: handler keeps worker copy of logprobs as request grows, no copy without them | REG 9a566c18 fix #2768 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerFlagsTest | test-core | 5 | KEEP | breaks: Tokenizer flags from tokenizer_config: add_bos, Mistral BOS/EOS, default system prompt | 52d87981 feat #77 | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| TokenRecycle | test-core | 17 | KEEP | breaks: Token recycling draft: pair chain, most recent pair wins, eviction, top-k rank order, 17 | e8c27c17 Token-Recy #1054 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| TokenSpansTest | test-core | 5 | KEEP | breaks: Token spans: flush attributes pieces on token boundary, remainder rebased, untracked bytes | REG d979e94a fix #1700 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolArgCoerce | test-core | 1 | KEEP | breaks: Tool-argument coercion matches vLLM coerce-to-schema-type table, golden rows | REG 5ecafe37 fix #3137 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolCallChatML | test-core | 6 | KEEP | breaks: ChatML tool-call parse: single and sequential calls, arguments key, malformed body, drift | REG a6a9b0a7 test #717 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolCallCohere | test-core | 1 | KEEP | breaks: Cohere action array parses through the ChatML parser into two named calls | f712136b feat #2604 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ToolCallDialectGolden | test-core | 1 | KEEP | breaks: Tool-call dialect golden: all families match the recorded fixture | REF? d45b4f29 refactor #2494 (refs 2464) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.01C |
| ToolCallDialectRegistry | test-core | 2 | KEEP | breaks: Tool-call dialect registry: every family matches pre-registry switches, ChatML default | REF? d45b4f29 refactor #2494 (refs 2464) | ToolCallDialectGolden (registry fields the golden cannot see) | impl (expected = pre-registry switches) | gate pair, JSON envelope, XML body, native token, tool-response placement are read nowhere else. |
| ToolCallDispatch | test-core | 1 | DELETE | breaks: Tool-call dispatch routes each family to its parser, EXPECT_EQ on name | REG a6a9b0a7 test #717 | ToolCallDialectGolden, ForcedToolEnvelope, ToolChoiceEnforcement, ToolCallReconstruct, ToolTagScan | behaviour | 3 hand cases are a subset of the golden; TD-1 (LLAMA3 -> ChatML dialect) red by 10 other cases incl. ToolCallDialectGolden, TD-2 (GEMMA -> ChatML) red by 12 cases incl. ForcedToolEnvelope, ToolChoiceEnforcement, ToolCallReconstruct.GemmaWrapsCall, ToolResponseFormat.Gemma* (ToolCallDialectGolden stayed GREEN on TD-2: anomaly); heuristic origin flag is the word "regression" in #717 body, mutants cover it. |
| ToolCallFault | test-core | 4 | KEEP | breaks: Tool-call fault line: clean call has none, raw text on request, raw capped at 2000 bytes | 8623a6aa feat #3063 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ToolCallGemma | test-core | 3 | KEEP | breaks: Gemma tool-call parse: single call, no call returns content, digit-leading token kept | REG a6a9b0a7 test #717 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolCallGlm | test-core | 5 | KEEP | breaks: GLM tool-call parse: arg-key body, fault line raw body, pairs match vLLM finditer | REG 75454b9d feat #2603 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolCallHarmony | test-core | 8 | KEEP | breaks: Harmony tool-call parse: recipient channel becomes call, truncated call parses, final not | REG b01c6b6a fix #1719 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolCallLlama3 | test-core | 8 | KEEP | breaks: Llama-3 tool-call parse: single, bare JSON object, plain JSON answer not a call | REG a6a9b0a7 test #717 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolCallQwen36Xml | test-core | 21 | MERGE | breaks: Qwen3.6 XML tool-call parse: parameter coercion, string params, no-arg call, 21 cases | 59b664eb test #912 | same suite in test_tool_call.cpp + test_tool_call_xml_params.cpp | behaviour | one suite name split over 2 files; merge into one file, 0 cases removed. |
| ToolCallReconstruct | test-core | 7 | KEEP | breaks: Tool-call reconstruct: content kept, each dialect re-wraps parsed call, no-function entry | c864aa82 test #914 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ToolCallValidate | test-core | 6 | KEEP | breaks: Tool-call args validate against schema: required, type, non-object, unknown tool no-op | REG a6a9b0a7 test #717 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolChoice | test-core | 5 | KEEP | breaks: tool_choice validation: named tool must exist, required without tools refused | REF? cf97f674 test #905 (refs 892,901) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolChoiceEnforcement | test-core | 10 | KEEP | breaks: tool_choice enforcement per family: named and required enforced only where supported | REG 6a03db4f fix #1717 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolNamesFromRequest | test-core | 1 | KEEP | breaks: Tool names from request: OpenAI and bare shapes extracted, EXPECT_EQ | REG de0ea79c fix #1088 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolResponseFormat | test-core | 6 | KEEP | breaks: Tool response format: content passthrough, non-string serialised, text parts joined | c864aa82 test #914 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ToolResultImages | test-core | 4 | KEEP | breaks: Tool-result images: picture moves to a new user turn behind the tool run, ordering rules | REG 590bc7fe fix #3215 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolStreamFilterChatML | test-core | 18 | KEEP | breaks: Tool stream filter ChatML: chunked JSON body, args streamed before close, escapes | REF? c93fb092 feat #820 (refs 818) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolStreamFilterCohere | test-core | 1 | KEEP | breaks: Tool stream filter Cohere: action array yields one call per element across chunk sizes | f712136b feat #2604 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ToolStreamFilterGemma | test-core | 5 | KEEP | breaks: Tool stream filter Gemma: char-by-char, whole string, random chunks, two calls in one feed | REF? c93fb092 feat #820 (refs 818) | none found | behaviour; 3 case(s) assert via file helper | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolStreamFilterLlama3 | test-core | 2 | KEEP | breaks: Tool stream filter Llama-3: args stream incrementally, chunked input yields same call | REF? c93fb092 feat #820 (refs 818) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolStreamFilterUtf8 | test-core | 4 | KEEP | breaks: Tool stream filter UTF-8: argument deltas never end mid-codepoint, 4-byte chars survive | REG 6a1d363c fix #1667 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ToolTagScan | test-core | 2 | KEEP | breaks: Tool tag scan: Gemma marker prefixes held back, ChatML and Llama-3 unchanged | REF? c93fb092 feat #820 (refs 818) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TracingTest | test-core | 7 | KEEP | breaks: Tracing: traceparent accepted only in well-formed version 00, random ids width and | 994c5a71 feat #1855 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| TypeTablesGolden | test-core | 2 | KEEP | breaks: Wire and QType type tables match golden for every id, element bytes and names | REF? f6f54609 refactor #2241 (refs 1917,2210,2212) | none found | behaviour; 1 case(s) assert via file helper | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| UploadExceedsCheckpoint | test-core | 4 | KEEP | breaks: Upload exceeds checkpoint: measured pair, quarter boundary inclusive, unknown size not a | 23429032 memory #1476 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| UploadReserve | test-core | 7 | KEEP | breaks: Upload reserve reproduces logged 64-slot value, Qwen3.8 batch kept by trimming KV, KV | REG 4999b75b fix #2397 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Utf8 | test-core | 3 | KEEP | breaks: utf8_complete_len: complete ASCII and multibyte, truncated trailing sequence held back | REG a8eeeee5 test #529 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Utf8ChunkLen | test-core | 4 | KEEP | breaks: UTF-8 chunk length never splits a multibyte character, chunks reassemble, always | REG aa8818c0 fix #1666 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Utf8Stitch | test-core | 5 | KEEP | breaks: UTF-8 stitch rejoins split characters across pieces, invalid lead not held back forever | REF? 674f2b8d feat #1078 (refs 760) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| VideoEvs | test-core | 12 | KEEP | breaks: Video EVS pruning: retained count floors at one pair, ties keep lower index, pad layout | 274168cc feat #2974 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ViewOrOwnedTest | test-core | 10 | KEEP | breaks: View-or-owned: view set/reset never releases, owned replace releases once, idempotent | ff1dfae0 refactor #3297 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| VisionCacheKeys | test-core | 5 | KEEP | breaks: Vision cache keys: text before first picture keys as text, new picture leaves old keys | REG 0f3ef903 fix #3152 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| VisionFamily | test-core | 4 | KEEP | breaks: Vision family registry: exact names map, loadable only with encoder, probe follows | ec75c491 feat #2375 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| VisionLoaderCheck | test-core | 12 | KEEP | breaks: Vision loader check: Qwen3-VL merger refused, Gemma3/4v tower complete, fused QKV caught | REG de85c377 fix #1245 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| VisionPreprocess | test-core | 6 | KEEP | breaks: Vision preprocess: CHW layout, channel order, mean/std, resize to square, garbage fails | REF? 546454b6 test #564 (refs 510,557) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| VisionPromptLayout | test-core | 1 | KEEP | breaks: Vision prompt layout matches golden recorded on main, EXPECT_EQ over model types and cases | REF? d246218f refactor #2507 (refs 2463) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| VramReserveFloor | test-core | 2 | KEEP | breaks: VRAM reserve floor: default equals allocator headroom, policy percent only raises it | REG 3bb3864d fix #2783 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| WeightCacheEntriesTest | test-core | 12 | KEEP | breaks: Weight cache entries: FP16 move leaves source empty, map erase releases once, NVFP4 owner | REF? d34185f0 refactor #3298 (refs 3297) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| WeightKeyTest | test-core | 1 | KEEP | breaks: Weight key canonical formats: L<layer>.<name> strings, EXPECT_STREQ | ce35ae0d feat #954 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| WeightMapExpertGuard | test-core | 5 | KEEP | breaks: Weight map expert guard: up/down-only MoE accepted, refused without any expert, dense | d27b2591 feat #1385 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| WeightMapModalitySkips | test-core | 7 | KEEP | breaks: Weight map modality skips: audio and vision counted apart, text-only load skips nothing | REG d3982ad8 fix #1929 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| WeightMapOutNormWarn | test-core | 3 | KEEP | breaks: Weight map out-norm warning: mixer without final norm not warned, missing norm warns | REG 1ff84c60 fix #2352 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| WeightOwnedTest | test-core | 10 | KEEP | breaks: Weight owner: scope end frees once on adoption stream, reset idempotent, move transfers | REF? d34185f0 refactor #3298 (refs 3297) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| WeightPageRelease | test-core | 4 | KEEP | breaks: Weight page release: data still reads back after release, split shards released | 8b3330b9 perf #1934 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| WeightRecordIndicesTest | test-core | 1 | KEEP | breaks: Weight record indices: file-supplied indices bounds-checked, EXPECT_TRUE | 2a968200 harden #1942 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| WeightSnapshotArmSlotTest | test-core | 1 | KEEP | breaks: Weight snapshot arm slot: arm, take, disarm sequence on the global slot, EXPECT_EQ | ce35ae0d feat #954 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| WeightTierEntries | test-core | 2 | KEEP | breaks: Weight tier entries: bind takes source map nodes, every read tier needs its entry | c8bf273d refactor #3339 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| WeightUploadLogTest | test-core | 6 | KEEP | breaks: Weight upload log: record resolves sizes and offsets, evict marks dead and drops live | ce35ae0d feat #954 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| WeightUploadTraits | test-core | 2 | KEEP | breaks: Weight upload traits: per-format uploaders byte-for-byte equal, one-byte change detected | REF? 4e8daf0d refactor #2300 (refs 2212) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.08C |
| WrongFieldTypeMessage | test-core | 3 | KEEP | breaks: Wrong-field-type message names the first wrong field, valid bodies and nulls not reported | REG fdbc49be fix #2143 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| AdmissionCapacityTest | test-e2e | 1 | KEEP | prompt larger than the KV pool is cancelled instead of a typed error | `6faa7d54` #1110 (body) f0 | - | runs in test-gpu via default path /models/Qwen3-8B-NVFP4-cortecs | - |
| ApiGenerateParityTest | test-e2e | 3 | KEEP | C API generate returns empty, stream drops tokens, manual prefill loop fails | `4b1ed78a` #320 f0 | PrimaryModelTest.GenerateCoherentOutput (finer: asserts "Paris") | - | - |
| BatchBuilderTest | test-e2e | 7 | KEEP | continuous-batching builder wrong for decode, prefill, mixed, 16 sequences | `ff4d81d8` f2 | - | CI lane | - |
| BatchInvarianceTest | test-e2e | 4 | KEEP | NVFP4/FP8 decode row depends on batch neighbours | REG `11571cff` #1924 f4 | - | runs twice: test-gpu (default paths) and test-e2e batch-invariance group, 46.3 s each | REGRESSION-ORIGIN; lane dedup lever L4 |
| CancelLifecycleHybridTest | test-e2e | 1 | KEEP | cancelling a queued request corrupts the running hybrid sequence state | `fc1bb711` #3309 (body) f0 | CancelLifecycleTest | - | - |
| CancelLifecycleTest | test-e2e | 2 | KEEP | cancel during a graph burst leaks tables or emits a token after cancel | `fc1bb711` #3309 (body) f0 | - | - | - |
| CandidateTokenGuardTest | test-e2e | 8 | MOVE(wire env) | candidate-token guard accepts merged, duplicate or multi-token letters | `b6e638b9` #2198 f0 | - | dead: IMP_TEST_TOKENIZER_QWEN3 set by no target (not IMP_TEST_MODEL_* so check_test_model_env.py misses it), 8 of 8 skip | tokenizer class: wire into test-e2e (needs tokenizer files only, no GPU) |
| ChunkedPrefillTest | test-e2e | 7 | MOVE(re-point) | chunked prefill logits differ from unchunked at model level | `d50cc86c` #130 (body) f3 | RaggedPrefillTest chunked case (synthetic), AttentionChunkedTest (kernel) | dead: 7 of 7 skip, IMP_TEST_MODEL_QWEN4B / _LLAMA GGUF not in the store (allowlist) | REGRESSION-ORIGIN (#130 body): re-point to Qwen3-8B-NVFP4-cortecs, do not delete |
| DecodeGraphCtxGrowthTest | test-e2e | 1 | MOVE(wire) | decode graph dies on a long prompt after a short request | REG `617d2753` #948 f2 | - | dead: in no Makefile/script filter, skips in test-gpu (needs IMP_TEST_MODEL) | REGRESSION-ORIGIN #948 runs nowhere: add to the model-battery filter |
| DeepStackInject | test-e2e | 5 | KEEP | DeepStack injection adds outside image tokens or loses surplus | `7c0db91c` #1176 f1 | - | - | - |
| DegenerationTest | test-e2e | 6 | KEEP | repetition loop, greedy nondeterminism, graph replay != eager, leaked special tokens | `3a36d036` (body) f3 | GreedyLockTest (finer) | runs in test-gpu and test-e2e locks group, 13.1 s | lane dedup L4 |
| DetEvalE2ETest | test-e2e | 10 | KEEP | deterministic mode not bit-identical greedy/PPL in one context | REG `1468568e` #522 f1 | - | 4 DISABLED_ cross-context cases = documented known limit (docs/DETERMINISM.md 4) | REGRESSION-ORIGIN |
| DeviceFaultSignalTest | test-e2e | 1 | KEEP | device fault inside a batching step does not cancel and refuse later requests | REG `c76d9066` #1913 f3 | - | forked child with 10-120 s deadlines, measured 2.14 s | REGRESSION-ORIGIN |
| DFlashDrafterTest | test-e2e | 1 | KEEP | DFlash drafter block output diverges from reference | `8406a2b4` #2991 f1 | - | runs in test-gpu (default paths) and test-e2e drafters group | lane dedup L4 |
| DSparkDrafterTest | test-e2e | 4 | KEEP | DSpark block, rollback, FP8 weights, graph replay diverge | `92bb937d` #2806 f1 | - | runs twice like DFlash, 11.66 s | lane dedup L4 |
| EncoderEmbedTest | test-e2e | 1 | MOVE(checkpoint) | NomicBert encoder embedding structure wrong | `2d35ac29` #836 (body) f0 | - | dead: /models/nomic-embed-text-v1.5.Q8_0.gguf not in the store, wired in no filter | docs/MODELS.md lists nomic-embed as supported: fetch the 0.2 GB GGUF or drop the MODELS.md row (owner) |
| EndToEndModelTest | test-e2e | 8 | SHRINK | C API load/tokenize/generate/prefill-decode on a real model | `051d9437` f8 | PrimaryModelTest (same IMP_TEST_MODEL, same 4 API paths, stronger asserts) | PrefillDecodeStep went red once on a test defect (#3333, token id 0) | delete LoadModel, Tokenize, CreateContextAndGenerate, PrefillDecodeStep (4 model loads); keep LongContext8k/16k, MultiDecodeOutputIsolation, InThinkStop; mutants M-E2E1..3 Mutant note: M-E2E1 red by PrimaryModelTest.TokenizeRoundtrip only (EndToEndModelTest.Tokenize green); M-E2E2 red by PrimaryModelTest x4, GreedyLockTest x3, EndToEndModelTest x3 |
| EndToEndTest | test-e2e | 7 | KEEP | C API version/config defaults, error strings, null args | `ff4d81d8` f8 | - | CI lane | - |
| EngineIntegrationTest | test-e2e | 12 | KEEP | engine init, step prefill/decode, FP8/NVFP4 paths, pipelined batched decode wrong | `1876d86e` f2 | - | synthetic, 0.69 s | - |
| EngineRelaunchTest | test-e2e | 2 | KEEP | model reload leaks VRAM; second engine on same handle IMAs | REG `9542fd5d` #507 f5 | - | - | REGRESSION-ORIGIN |
| FimRealTokenizer | test-e2e | 2 | MOVE(wire env) | FIM token ids / PSM prompt wrong on Qwen-Coder; non-FIM tokenizer accepted | `da006aa1` #2240 f0 | - | dead: IMP_TEST_FIM_TOKENIZER / _NOFIM_ set by no target | tokenizer class: wire (Qwen3-Coder-30B-A3B-Instruct-FP4 and Gemma-4 dirs exist) |
| ForwardPassTest | test-e2e | 8 | KEEP | synthetic-model forward logits wrong for prefill, decode, GQA, batch composition | `ce4f29aa` f2 | - | synthetic, 0.85 s | - |
| GdnGraphBucketTest | test-e2e | 2 | KEEP | graph batch bucket reuses another bucket's recurrent slot | REG `b954e390` #1960 f1 | - | - | REGRESSION-ORIGIN |
| GDNModelTest | test-e2e | 2 | KEEP | GDN model incoherent or loses recurrent state across turns | `9144a758` f2 | - | - | GDN class |
| Gemma4AudioEncoderTest | test-e2e | 2 | MOVE(generate ref) | Gemma-4 audio front end/tower drift from HF features | `9529acc0` #2980 f0 | - | dead: IMP_TEST_GEMMA4_AUDIO_REF dir not in the store (allowlist) | generate the ref with tools/analysis/gemma4_audio_ref.py and wire |
| Gemma4E4BRef | test-e2e | 2 | MOVE(generate ref) | Gemma-4 E4B greedy text drifts from HF | `8a97b792` #2978 f0 | - | dead: IMP_TEST_GEMMA4_E4B_REF not in the store | generate the ref and wire |
| Gemma4GraphsTest | test-e2e | 2 | KEEP | Gemma-4 with CUDA graphs wrong answer or incoherent 256-token decode | `f5d622e2` (body) f2 | Gemma4ModelTest | - | graph capture class |
| Gemma4ModelTest | test-e2e | 4 | KEEP | Gemma-4 chat answer, raw completion, SWA prefill route, repetition | `2e43f994` (body) f2 | - | - | - |
| Gemma4SafeTensorsTest | test-e2e | 1 | KEEP | needle beyond the sliding window not retrieved | `966559cd` #2530 (body) f2 | - | - | - |
| GgufDeadWeightForms | test-e2e | 2 | MOVE(re-point) | GGUF load keeps the CUTLASS SfAtom / LM-head source no path reads (VRAM) | REG `cd41e4bf` #2711 f1 | - | dead: /models/Qwen3-14B-Q6_K.gguf and Qwen3.8-27B-Q4_K_M.gguf not in the store, wired nowhere | REGRESSION-ORIGIN #2711, no remaining catcher found: re-point (flint-nvfp4.gguf, Qwen3.8-27B-Q8_0.gguf exist) |
| GPUBatchTest | test-e2e | 6 | KEEP | GPU batch upload/free double free, empty batch, varying contexts | `ff4d81d8` f2 | - | - | - |
| GreedyLockTest | test-e2e | 3 | KEEP | greedy tokens drift from locked table; teacher-forced decode feeds/scores wrong token | `b4e08e59` #531 (body) f1 | - | - | - |
| HybridBatchedDecodeTest | test-e2e | 3 | KEEP | hybrid batched decode rows depend on order; think-close repeats | `103f0566` #2020 f4 | Mamba2BatchedDecodeTest | - | - |
| HybridRestoreChainTest | test-e2e | 8 | KEEP | 30 chained hybrid restores drift from a cold prefill | REG `b954e390` #1960 f1 | - | - | REGRESSION-ORIGIN |
| HybridSharedScoreTest | test-e2e | 1 | MOVE(wire) | rows restoring one hybrid snapshot interfere across waves | `5c7113eb` #2198 (body) f0 | - | dead: in no filter, skips in test-gpu | add to the hybrid-restore group |
| InternVLEncoder | test-e2e | 2 | KEEP | InternVL encoder stage outputs drift from HF FP32 | `e341ee4e` #2384 f0 | - | test-vision only | - |
| KimiTiktokenParityTest | test-e2e | 4 | KEEP | Kimi tiktoken encode differs from HF | `d8550086` #2981 f0 | - | - | tokenizer class |
| Ling3Ref | test-e2e | 2 | MOVE(generate ref) | Ling-3 greedy and chunked prefill drift from HF | `1bd027ac` #2979 f0 | - | dead: IMP_TEST_LING3_REF not in the store | generate the ref and wire |
| LlmCompressorE2E | test-e2e | 4 | SHRINK | LLM-compressor / ModelOpt checkpoints fail to load or generate | `ff6fb633` #63 f0 | - | 2 of 4 skip: Mistral dir absent, coder default /models/Qwen3-Coder-30B-A3B-FP4 absent (Qwen3-Coder-30B-A3B-Instruct-FP4 exists); wired in no filter | fix the coder default path, one shared Gemma-4 llmc load for the 2 Gemma cases (4.97 + 3.38 s) |
| LoraHotSwap | test-e2e | 3 | MOVE(re-point) | zero LoRA adapter not identity; shape check; prefix-cache key ignores adapter | `daa16376` #522 (body) f2 | - | dead: Llama-3.2-3B Q8_0 GGUF not in the store, 3 of 3 skip | re-point to a SafeTensors base |
| Mamba2BatchedDecodeTest | test-e2e | 1 | KEEP | Mamba2 batched decode rows depend on order | REG `b6e4e533` #3009 f4 | - | - | REGRESSION-ORIGIN |
| MoePrefixHitE2ETest | test-e2e | 2 | KEEP | MoE prefix-cache hit not bit-equal to fresh | REG `21cb829b` #3006 f6 | - | - | REGRESSION-ORIGIN |
| MtpGreedyIdentityTest | test-e2e | 6 | KEEP | MTP, n-gram, DSpark, DFlash change constrained greedy tokens | `92e7c381` #1964 f1 | - | runs in test-gpu (default paths) and 2 test-e2e groups, 113.1 s | lane dedup L4 |
| MtpProbe | test-e2e | 1 | KEEP | MTP head not found in a real checkpoint | `f39441d0` #1475 f0 | - | - | - |
| MtpQwen4ExpReference | test-e2e | 1 | KEEP | MTP two draft steps drift from vLLM bf16 math | `20266928` #2177 f1 | - | runs in test-gpu and drafters group, 35.3 s | lane dedup L4 |
| Nvfp4PrefillCacheKnobTest | test-e2e | 1 | KEEP | decode knob drops the CUTLASS NVFP4 prefill cache | REG `697bc9ad` #1503 f1 | - | - | REGRESSION-ORIGIN |
| PrefixCacheE2ETest | test-e2e | 8 | KEEP | prefix hit differs from fresh; hybrid snapshot and FP8-KV restores | `1a1a6dcb` #537 f6 | MoePrefixHitE2ETest | - | prefix-cache class |
| PrimaryModelTest | test-e2e | 4 | KEEP | primary model incoherent, multi-turn broken, tokenize roundtrip, prefill/decode | `9144a758` f2 | EndToEndModelTest (weaker copy) | - | catcher for the EndToEndModelTest shrink |
| PromptLogprobsE2ETest | test-e2e | 1 | MOVE(wire) | prompt-logprob chunk logits not released after the pass | `3679fe4e` #2264 (body) f0 | - | dead: in no filter, skips in test-gpu | add to the model battery |
| Q8DropSourceE2E | test-e2e | 1 | MOVE(re-point) | Q8_0 source freed while a fused n==1 dp4a route still reads it | REG `c09bfbe2` #2712 f1 | Q8PlanesOnly (planes compute, not the drop rule) | dead: /models/Qwen3-4B-Instruct-2507-Q8_0.gguf not in the store | REGRESSION-ORIGIN #2712: re-point to Qwen3.8-27B-Q8_0.gguf (fit on 32 GB unverified) Mutant note: M-Q8 survived Q8PlanesOnly + QuantIntegrationTest (27 of 27 green): no other catcher, re-point confirmed over delete |
| QuantPipelineTest | test-e2e | 1 | MOVE(re-point) | Q8_0 GGUF decode caches unpopulated (empty decode) | `ebcaee92` #629 (body) f0 | QuantIntegrationTest.Q8_0ForwardPass / FP16ForwardPass run pre_dequant_weights on synthetic models | dead: default /models/Qwen3-8B-Q8_0.gguf absent, in no filter; asserts only len > 0 and >= 10 words | FLIPPED by a mutant: M-QP (executor_pre_dequant.cpp:234 Phase-1 FP16 cache call removed) SURVIVED QuantIntegrationTest (24 of 24 green), so the synthetic suite is no proven catcher: keep the #629 test, re-point to a Q8_0 GGUF in the store (Qwen3.8-27B-Q8_0.gguf, fit on 32 GB unverified) or owner decides |
| Qwen38TokenizerParityTest | test-e2e | 4 | KEEP | Qwen3.8 tokenizer encode differs from HF | `b5e37182` #1750 (body) f0 | - | - | tokenizer class |
| Qwen3VLEncoder | test-e2e | 2 | KEEP | Qwen3-VL encoder differs from CPU reference incl. attention chunks | `ef74c024` #1171 (body) f0 | - | - | - |
| Qwen3VLPipelineTest | test-e2e | 6 | KEEP | Qwen3-VL image pipeline bytes, determinism, HF video frames | `54b977c1` #1175 (body) f0 | - | test-vision only | - |
| Qwen3VLPipelineVideoStamps | test-e2e | 1 | KEEP | Qwen3-VL video timestamp ids differ from HF | `53b68e51` #2370 f0 | - | test-vision only | - |
| RaggedPrefillTest | test-e2e | 6 | KEEP | ragged prefill differs from serial dense; mixed decode | `49044bf8` #1780 f2 | - | synthetic | - |
| RequestTest | test-e2e | 5 | KEEP | request status transitions, prefix/ragged admission wrong | `ff4d81d8` f2 | - | CI lane | - |
| SamplerRowBansTest | test-e2e | 1 | KEEP | stashed sampler row loses its ban list | REG `93da4a1b` #2019 f1 | - | - | REGRESSION-ORIGIN |
| SchedulerGpuTest | test-e2e | 1 | KEEP | pool does not grow under aggregate memory pressure | `b89217f7` #1864 f9 | SchedulerTest | - | - |
| SchedulerTest | test-e2e | 43 | KEEP | admission gate, memory-aware admission, priority, cancel, aging wrong | `ff4d81d8` f11 | - | CI lane | - |
| SentencePieceLoader | test-e2e | 1 | MOVE(fixture) | real spiece.model fails to parse | `394593a0` #128 (body) f0 | test-text SentencePiece unit tests (synthetic) | dead: /hf_cache not mounted by any lane, container HOME has no HF cache | tokenizer class: commit a small spiece fixture or mount the HF cache |
| ServingSignalsTest | test-e2e | 8 | SHRINK | KV exhaustion, valve arming, host swap, queue time, prefill prewarm signals wrong | REG `960b9319` #1915 f4 | - | flaky: EXPECT_LE(first_med, 1.5*warm_med + 2) on wall-clock TTFT; 20 ms polling for a transient 1/1 queue split; runs in test-gpu and model battery | REGRESSION-ORIGIN #1915: FirstShortPrompts... sets prefill_graph_prewarm=64 and measured 154.33 s, it uses lengths 20-24: prewarm 25 or lengths 2-6 with prewarm 8 Mutant note: L5 prewarm 64 -> 25 REFUTED (case 157.09 -> 158.00 s); the cost is the 10 s post-TTFT drain per request (tests/test_serving_signals.cpp:323): skipping it when the first event is last gives 7.15 s, all green (Phase 3 P4) |
| SpecCaptureFidelityTest | test-e2e | 2 | KEEP | cached-graph speculative forward differs from eager; hybrid spec collapses | REG `697bc9ad` #1503 f1 | - | runs in test-gpu, model battery and test-spec-fidelity, 49.6 s | REGRESSION-ORIGIN; lane dedup L4 |
| StoragePlanner | test-e2e | 14 | KEEP | weight tier plans exceed budget, NVFP4 tiers, VRAM sums wrong | `c8763ade` #27 f2 | - | CI lane | - |
| StubModelTest | test-e2e | 5 | KEEP | stub GGUF load/tokenize/infer fails; VRAM leak | `1958e29b` f8 | - | 2 cases in CI | - |
| SuspendResumeTest | test-e2e | 3 | KEEP | suspend/resume snapshot not token-identical; armed mismatch honoured | `ce35ae0d` #954 f0 | - | 17.7 s, 3 cold loads by design | - |
| TensorKindCoverage | test-e2e | 1 | DELETE | GGUF tensor name maps to an unknown kind | `c8763ade` #27 f0 | TensorKindMatcher / TensorKindTable (test-core, CI) | dead: IMP_TEST_GGUF set by no target (filter names it, var unset) | Mutant note: M-TK red by TensorKindMatcher.AttentionProjections (test-core, CI lane) |
| TokenizerCompatTest | test-e2e | 1 | KEEP | tokenizer differs from HF on the loaded model | `9870fa3b` f1 | - | - | tokenizer class |
| VideoEvsKernels | test-e2e | 2 | KEEP | EVS dissimilarity or gather order wrong | `274168cc` #2974 f0 | - | - | - |
| VisionChunkOffset | test-e2e | 5 | KEEP | vision embeddings at chunk offset / DeepStack offset wrong | REG `383952d5` #1184 f1 | - | - | REGRESSION-ORIGIN |
| VisionGolden | test-e2e | 2 | MOVE(re-golden) | Gemma3 SigLIP / Gemma4v encoder + projector drift from golden | `54587c38` #583 f0 | Qwen3VLEncoder, InternVLEncoder (other towers) | dead: IMP_TEST_MMPROJ GGUFs deleted 2026-10-30 (Makefile:367) | re-golden on the SafeTensors Gemma-4 vision tower, or delete with the mmproj loader (owner) |
| VramBudgetReserve | test-e2e | 21 | MOVE(test-core, partial) | VRAM budget planner: KV pool, weight-cache room, reserve floor, IMMA planes wrong | REG `b30160e9` #874 f14 | - | 16 of 21 cases SKIP_IF_NO_CUDA, 5 are host arithmetic, none in the CI subset | move the 5 host cases to the CI lane |
| VramLedgerE2E | test-e2e | 1 | MOVE(re-point) | MXFP4 GDN native load holds an FP16 cache (ledger double charge) | REG `db61ed7d` #2713 f1 | - | dead: /models/Qwen3.5-4B-mxfp4.gguf not in the store, wired nowhere | REGRESSION-ORIGIN #2713, no MXFP4 GDN checkpoint in the store: owner decides re-point or delete |
| WarmCacheTest | test-e2e | 3 | KEEP | warm weight cache boot not token-identical; corrupt cache used | `5455df97` #956 f0 | - | 17.6 s, cold/warm loads by design | - |
| WeightRegistryPreservation | test-e2e | 1 | KEEP | NVFP4 mode drops SSM in/out tensors | `c8763ade` #27 f2 | - | CI lane | - |
| HfFetchNetwork | test-hf-network | 3 | DELETE | breaks: HF fetch (network): GGUF fetch verifies SHA then hits cache; gated repo fails cleanly | REF? 713841d3 Fetch hf #2236 (refs 2200,2210) | HfFetchMock (same 3 behaviours on a local mock: SHA verify + cache hit, several-GGUF list, gated repo) | behaviour; DEAD: GTEST_SKIP unless IMP_TEST_NETWORK=1, no lane sets it (A6); FLAKY: live huggingface.co | HF-1/HF-2/HF-3 all red by HfFetchMock; only unique value is live HF API drift, which no lane ever checks; drops the test-hf-network target (TU 2.58 s + link 0.2 s). Alternative if owner wants live drift: MOVE to a nightly network job. |
| CaptureAbort | test-kv | 3 | KEEP | graph capture abort leaves stream unusable; first replay failure skips the step | REG `b30160e9` #874 f4 | - | - | REGRESSION-ORIGIN |
| DeviceAllocAllGpu | test-kv | 3 | KEEP | alloc-all leaks or half-allocates on failure | `20a13cdc` #3258 f0 | - | - | - |
| DeviceOwnedGpu | test-kv | 3 | KEEP | DeviceOwned ledger bytes per allocator family wrong | `d95001df` #3296 (body) f0 | - | impl-coupled ledger | - |
| FP8KVCache | test-kv | 9 | KEEP | FP8 KV scale calibration, block bytes, split-K, HD256 differ from FP16 | `164cf383` f0 | - | - | paged KV class |
| Fp8KvStaticScale | test-kv | 6 | KEEP | FP8 KV static scale off grid or round trip wrong | `5f7291f9` #3016 f0 | - | - | paged KV class |
| INT8KVCache | test-kv | 2 | KEEP | INT8 KV construction or split-K inconsistent | `164cf383` f0 | - | - | paged KV class |
| KVBlockCopy | test-kv | 4 | KEEP | KV block copy loses key min/max or touches other blocks | `e8c27c17` #1054 f0 | - | - | paged KV class |
| KVCacheGrowAdmissionTest | test-kv | 2 | KEEP | capped reclaim reads VRAM per block; grow-ahead ignores admission | REG `d1a2da52` #3018 f1 | - | flaky: EXPECT_LT(ms, 5.0) wall clock next to the counter assert reads <= 1; fixed 50 ms sleep on the negative branch | REGRESSION-ORIGIN; the counter carries the signal |
| KVCacheGrowTest | test-kv | 7 | KEEP | growable pool overshoots headroom, ceiling raise, scale plane wrong | `fcd115bd` #1938 f5 | - | 5 ms polling bounded at 2 s | paged KV class |
| KVCacheIdleReleaseTest | test-kv | 2 | KEEP | idle release returns cached prefix or referenced tail | `8db4e704` #3270 f1 | - | - | - |
| KVCacheManagerGrowTest | test-kv | 1 | MERGE(KVCacheManagerTest) | KV manager reclaims cached blocks before growing | `fcd115bd` #1938 f5 | - | - | same file, 1 case, rename only |
| KVCacheManagerTest | test-kv | 7 | KEEP | fingerprint gate, reclaim count, SWA rollback, snapshot pack/restore wrong | `ff4d81d8` f5 | - | - | paged KV / prefix class |
| KVCacheTest | test-kv | 10 | KEEP | KV block pointers, NVFP4 layout, SWA pointers, residency probe wrong | `ff4d81d8` f5 | - | - | paged KV class |
| KVCacheWriteTest | test-kv | 4 | KEEP | paged/fused K/V write, batched block table, INT8 per-head scale wrong | `33ba646b` f0 | - | - | paged KV class |
| KVGatherTest | test-kv | 5 | KEEP | paged-to-flat gather wrong for FP16/FP8/INT8, sentinel fill | `d50cc86c` #130 (body) f1 | - | - | paged KV class |
| KVHostSpillGpuTest | test-kv | 27 | KEEP | host spill restore not bit-exact incl key min/max; snapshot chain | `5c96d8cd` #2203 f1 | - | - | - |
| MtpUploadBudgetTest | test-kv | 3 | MOVE(test-core) | MTP upload peak bytes miscounted (packed layout, per-expert slabs) | REG `fb74d8b1` #1494 f1 | - | host-only (model/mtp_head.h, no CUDA call) | REGRESSION-ORIGIN #1494, gains CI coverage |
| PrefixEquivTest | test-kv | 13 | KEEP | prefix reuse counts, shared blocks, eviction, chain holes, salt wrong | `1a1a6dcb` #537 f3 | - | - | prefix-cache class |
| PrefixPersistTest | test-kv | 1 | KEEP | quantized KV restored without its scales | REG `d10e2497` #1189 f3 | - | - | REGRESSION-ORIGIN |
| RecurrentSnapshotStoreTest | test-kv | 13 | KEEP | host-tier eviction/restore, LRU, protected entries wrong | `e80a26a4` #831 (body) f6 | - | cv waits bounded 300 ms / 5 s via a test hook (sync, not sleep), one 5 ms poll | defect fix #3333 added a case |
| SSMStateLazyGpu | test-kv | 2 | KEEP | lazy SSM slot commit backs wrong stride; decommit leaks | `d8c47857` #1981 f5 | - | - | - |
| TouchAuditGpu | test-kv | 1 | KEEP | touch audit miscounts written pages | `b1a4dabf` #3234 f0 | - | impl-coupled | measurement tool gate |
| VmmBackendTest | test-kv | 8 | KEEP | VMM reservation costs memory; growth past ceiling; blocks not zeroed | `4e4997b6` #1452 (body) f0 | - | - | - |
| VramFits | test-kv | 1 | KEEP | fit check ignores margin or credit | `c67015f9` #3278 f0 | VramQueryTest | - | 4 suites in test_vram_query.cpp could share one name |
| VramLedger | test-kv | 2 | KEEP | plain allocation not counted; unknown free crashes | `c68f8ebe` #3280 f0 | - | - | - |
| VramQuerySnapshot | test-kv | 2 | KEEP | free at install not capped; pending reservations counted | `41225f0b` #3279 f0 | - | - | - |
| VramQueryTest | test-kv | 4 | KEEP | budget query passthrough, cap, exhaustion clamp wrong | `dc3d21a8` #838 f0 | - | - | - |
| ActivationCalibratorExperts | test-moe-gdn | 1 | KEEP | calibrator merges routed expert segments | `3eaec5a0` #2476 f0 | - | - | - |
| ExpertCacheLayerMask | test-moe-gdn | 1 | KEEP | uncached layers get expert rows | `290ccf0d` #2910 f1 | - | - | - |
| ExpertCachePhase3Test | test-moe-gdn | 11 | KEEP | expert-cache LRU eviction order, per-layer isolation, hit promotion wrong | `71e22477` #235 (body) f1 | - | - | - |
| ExpertCacheReinitTest | test-moe-gdn | 3 | KEEP | failed reinit leaves a half cache instead of fallback/disable | REG `64214afa` #2165 f1 | - | - | REGRESSION-ORIGIN |
| GdnBatchedScanTest | test-moe-gdn | 12 | KEEP | batched GDN scan not bit-identical at batch 1/8/32, non-contiguous slots | `b5e37182` #1750 (body) f0 | - | - | GDN class |
| GdnRaggedRidersTest | test-moe-gdn | 1 | KEEP | ragged decode riders differ from solo runs | `6ffa941e` #2987 f0 | - | - | GDN class |
| GDNScanTest | test-moe-gdn | 14 | KEEP | GDN scan GPU differs from CPU; fused vs legacy; chunk handoff | `b03137a9` f0 | GdnBatchedScanTest | 2 GTEST_SKIP lines | GDN class |
| GdnVerifySnapshotTest | test-moe-gdn | 7 | KEEP | verify snapshot slab written one row off; F32/BF16 | REG `d0d26d0f` #2132 f2 | - | - | REGRESSION-ORIGIN |
| GdnVerifyTwoDraftsTest | test-moe-gdn | 12 | KEEP | two-draft verify scan folds draft states wrong | `15d35035` #3037 f0 | - | - | GDN class |
| GptOssMoeFusedBias | test-moe-gdn | 4 | KEEP | gpt-oss fused GLU/scatter bias bytes differ from separate launches | `9f5213dd` #2466 f0 | - | - | - |
| GrammarConstrainMaskTest | test-moe-gdn | 1 | MERGE(RegexConstrainMaskTest) | grammar mask leaves model-vocab padding unmasked | REG `544a6d50` #1493 f2 | RegexConstrainMaskTest (same rule, regex constraint) | - | REGRESSION-ORIGIN #1493 for both: one TEST_P over constraint kind keeps both cases Mutant note: M-GR red by this case only, M-GR2 (regex_constrain.cu:235) by RegexConstrainMaskTest only: the TEST_P keeps both |
| JsonConstrainFsmTest | test-moe-gdn | 5 | MOVE(test-core) | JSON FSM accepts mismatched closers or trailing content | REG `3b87ba92` #1067 f10 | - | host-only (no CUDA call) | REGRESSION-ORIGIN #1067, grammar class gains CI coverage |
| JsonConstrainTest | test-moe-gdn | 12 | KEEP | JSON token classes, valid-token mask, control chars wrong | `60f3eaf8` (body) f10 | - | GPU mask kernel | grammar class |
| KdaScan | test-moe-gdn | 7 | KEEP | KDA recurrent/chunk scan, softplus gate, ragged, bf16 wrong | `1bd027ac` #2979 f0 | - | - | - |
| MoEDeterministicPermute | test-moe-gdn | 1 | KEEP | MoE permute order differs from the serial rule | `2ad413a4` #1546 f2 | - | - | - |
| MoEExecutorTest | test-moe-gdn | 12 | KEEP | MoE executor forward invalid, nondeterministic, logit drift | `ff4d81d8` f2 | - | - | - |
| MoeGroupTopk | test-moe-gdn | 1 | KEEP | group top-k differs from Ling gate | `1bd027ac` #2979 f0 | - | - | - |
| MoeImbalanceKernel | test-moe-gdn | 3 | KEEP | imbalance kernel miscounts skewed routing | `0dc07438` #1548 f2 | - | - | - |
| MoEMultiCtaPermute | test-moe-gdn | 1 | KEEP | multi-CTA permute differs from single-CTA | `49236dfb` #2465 f2 | - | - | - |
| MoERoutingEdgeTest | test-moe-gdn | 2 | MERGE(MoERoutingTest) | empty expert / all tokens to one expert mis-routed | `33988579` f2 | - | - | same file, 2 cases, rename only |
| MoERoutingTest | test-moe-gdn | 7 | KEEP | MoE top-k, weight normalization, offsets, gather/scatter wrong | `ff4d81d8` f2 | - | - | - |
| MoERoutingWideTest | test-moe-gdn | 2 | KEEP | routing with > 256 experts loses candidates | REG `3dbcde56` #2056 f2 | - | - | REGRESSION-ORIGIN |
| MtpBodyNvfp4Test | test-moe-gdn | 2 | KEEP | MTP body NVFP4 copy not read by one/multi-row feeds | `3f85ff2a` #3127 f0 | - | - | - |
| MtpFeedBatchTest | test-moe-gdn | 11 | KEEP | batched MTP feed differs from per-pair; packed tables; captured head | `b5e37182` #1750 (body) f0 | - | 4.60 s, AttnScanMatchesSoftmaxReference 4.56 s | shape lever L7 |
| PLEBatched | test-moe-gdn | 3 | KEEP | PLE batched decode rows leak between sequences | `702a9cd3` #2170 f0 | - | - | - |
| PreambleGateTest | test-moe-gdn | 29 | MOVE(test-core) | JSON preamble gate absorbs wrong tokens, budget mode, thinking absorb, tool transitions | `56704f56` #103 f10 | - | host-only, 29 cases (PreambleGate object, no CUDA call) | grammar class gains CI coverage |
| RegexConstrainMaskTest | test-moe-gdn | 1 | MERGE(GrammarConstrainMaskTest) | regex mask leaves model-vocab padding unmasked | REG `544a6d50` #1493 f2 | GrammarConstrainMaskTest | - | REGRESSION-ORIGIN #1493, merged not deleted |
| RegexNfaTest | test-moe-gdn | 4 | MOVE(test-core) | regex NFA literals, class ranges, counted repeats, anchors wrong | REG `73005c63` #497 f3 | - | host-only | REGRESSION-ORIGIN #497, gains CI coverage |
| SchemaConstrainTest | test-moe-gdn | 32 | KEEP | JSON schema patterns, enums, item bounds, refs, recursion, tool calls wrong | REG `73005c63` #497 f4 | - | - | REGRESSION-ORIGIN, grammar class |
| SSMConv1dTest | test-moe-gdn | 10 | KEEP | SSM conv1d decode shift / prefill causal wrong | `60f3eaf8` (body) f2 | - | - | - |
| SSMConvSplitTest | test-moe-gdn | 2 | KEEP | conv split differs from separate kernels | `85de86c4` #2630 f0 | - | - | - |
| SsmDecodeBatchedKernelTest | test-moe-gdn | 3 | KEEP | batched SSM decode rows differ from single-sequence | REG `b6e4e533` #3009 f1 | - | - | REGRESSION-ORIGIN |
| SSMScanSsdTest | test-moe-gdn | 2 | KEEP | SSD scan differs from double reference | `d6cc5345` #2627 f0 | - | - | - |
| SSMScanTest | test-moe-gdn | 5 | KEEP | SSM scan padding, prefill snapshot, register scan wrong | `169b2165` #847 (body) f2 | - | - | - |
| BatchedSmallM | test-quant | 3 | SHRINK | NVFP4 batched small-M rows differ from single-row call | `ea547a53` #1467 (body) f1 | SmallMV2Pair | 1 DISABLED_ bench | DISABLED_ bench to the bench target |
| DequantAwqGpu | test-quant | 1 | KEEP | AWQ GPU dequant differs from host | `7254cb9e` #2250 f0 | - | - | - |
| DequantGptqGpu | test-quant | 1 | KEEP | GPTQ GPU dequant differs from host | REG `b3141829` #2252 f1 | - | - | REGRESSION-ORIGIN |
| DualPathQuant | test-quant | 2 | KEEP | attention weights quantized to NVFP4 in dual-path mode | `d6a17a26` f4 | - | - | - |
| FmhaVLoadBench | test-quant | 4 | MOVE(bench) | none: cp.async vs TMA timing (src/bench kernel) | `adbf4e46` #169 f0 | - | BENCH | benchmark, own target |
| Fp4PvBenchTest | test-quant | 4 | MOVE(bench) | none: FP4 PV bench kernel accuracy/throughput (src/bench) | `28c93df5` #240 f0 | - | BENCH | benchmark, own target |
| Fp8ActRowsTest | test-quant | 3 | KEEP | FP8 activation row quant differs from host rule or depends on neighbours | REG `2c10bbe2` #3005 f1 | - | - | REGRESSION-ORIGIN |
| Fp8GemmSmallMBench | test-quant | 1 | MOVE(bench) | none: DISABLED_ timing | `25fbd77c` #3004 f0 | - | BENCH | benchmark, own target |
| Fp8GemmSmallMTest | test-quant | 6 | KEEP | FP8 small-M GEMM differs from fp64 per M, beta, row bits | `25fbd77c` #3004 f0 | - | 4.56 s | FP8 MMA class |
| Fp8GemmTileBench | test-quant | 1 | MOVE(bench) | none: DISABLED_ timing | `998b87fd` #2927 f0 | - | BENCH | benchmark, own target |
| Fp8GemmTileRefusal | test-quant | 1 | KEEP | illegal shape accepted or output touched | `998b87fd` #2927 f0 | - | - | - |
| Fp8GemmTileTest | test-quant | 14 | KEEP | FP8 tile GEMM differs from cuBLASLt; beta=1 | `998b87fd` #2927 f0 | - | - | FP8 MMA class |
| Fp8GemvFold | test-quant | 2 | KEEP | folded FP8 GEMV differs from gemv_fp8 + host norm | `b44df45e` #3182 f0 | - | - | - |
| GemmNvfp4Batched | test-quant | 5 | KEEP | NVFP4 batched GEMM wrong at attn/FFN/down/narrow dims | `85a06deb` #998 (body) f0 | - | - | NVFP4 MMA class |
| GemvKparLoopRepro | test-quant | 2 | KEEP | NVFP4 KPAR per-row loop differs from single call at Gemma-4 dims | `e7f2a1f7` #69 (body) f0 | - | - | - |
| GgufDequant | test-quant | 14 | KEEP | GGUF dequant scale modes differ from reference | `f6419bf5` #587 (body) f6 | - | - | - |
| GgufDequantCoverage | test-quant | 1 | KEEP | a supported GGUF format has no dequant reference | REG `11571cff` #1924 f6 | - | - | REGRESSION-ORIGIN |
| GgufDequantKQuants | test-quant | 1 | KEEP | K-quant dequant tail, CTA, unaligned dst wrong | `1210d0d7` #2873 f6 | - | - | - |
| GgufRef | test-quant | 23 | KEEP | GGUF GEMV dp4a/MMVQ/MoE dispatch differ from fp64 for every type | `cbccb908` #535 f6 | - | - | catcher for the MMVQ and GemmDP4ATest shrinks |
| GptOssMxfp4ConvertRef | test-quant | 6 | KEEP | gpt-oss MXFP4 convert slices, extra scale, clamp wrong | `f6419bf5` #587 (body) f0 | - | - | - |
| GptqNvfp4 | test-quant | 4 | KEEP | GPTQ-to-NVFP4 worse than round-to-nearest; Cholesky wrong | `a6492284` #2775 f0 | - | - | - |
| GroupedNvfp4ArgsK | test-quant | 2 | KEEP | grouped NVFP4 device args differ from host args | `af0e1605` #2929 f0 | - | impl-coupled | - |
| GroupedNvfp4ArgsTest | test-quant | 1 | KEEP | grouped host args differ from per-expert dense GEMM | `af0e1605` #2929 f0 | - | - | - |
| GroupedNvfp4DecodeTest | test-quant | 3 | SHRINK | grouped NVFP4 decode F16 differs from tile/fp64; act epilogue | `9e167ce6` #3008 f2 | - | Bench case 0.41 s | Bench case to the bench target |
| GroupedNvfp4RefManyExperts | test-quant | 2 | KEEP | grouped NVFP4 reference wrong with many experts | `af0e1605` #2929 f0 | - | - | - |
| GroupedNvfp4RefSmallK | test-quant | 4 | KEEP | grouped NVFP4 reference wrong at small K | `af0e1605` #2929 f0 | - | - | - |
| GroupedNvfp4RefTest | test-quant | 3 | KEEP | empty-expert boundary, single active expert wrong | `af0e1605` #2929 f0 | - | - | - |
| GroupedNvfp4TileK | test-quant | 6 | KEEP | grouped NVFP4 tile GEMM differs from fp64 | `87d330f7` #2898 f0 | - | - | NVFP4 MMA class |
| GroupedNvfp4TileTest | test-quant | 1 | MOVE(bench) | none: wall-time bench | `42a8b5be` #2897 f0 | - | BENCH 0.14 s | benchmark, own target |
| LmHeadDraftRows | test-quant | 1 | KEEP | multi-row LM-head draft not row-count invariant | `d72a4aa1` #3039 f0 | - | timed cost part | - |
| MmaPeakSaturatedTest | test-quant | 1 | MOVE(bench) | calibration invariants of a src/bench kernel | `368139e0` #606 f0 | - | times kernels | bench support, own target |
| MmaVariantsBench | test-quant | 1 | MOVE(bench) | none: MMA variant timing | `208f25bb` #56 f0 | - | BENCH 0.49 s | benchmark, own target |
| MmqQ4kImmaBench | test-quant | 1 | MOVE(bench) | none: times Phase 2C Q4_K IMMA variants | `3efba4c7` #254 (body) f0 | - | BENCH of code reachable only from tests | benchmark rule; goes with the code if the owner deletes it |
| MmqQ4kImmaGemm | test-quant | 3 | DELETE | Phase 2C mmq_q4k_imma_gemm (7-arg overload, mmq_q4k_imma_tile.cu:403) wrong | `791b6d40` #268 (body) f1 | MmqQ8Imma covers the production 9-arg mmq_q4k_imma_gemm (mmq_q8_imma.cu:648) | code reachable only from tests | mutants M-Q4K1, M-Q4K2 on the production path Mutant note: M-Q4K3 red by MmqQ4kImmaGemm only (3 cases), rest of test-quant 354 ok, test-compute binary byte-identical; M-Q4K1/2 red by MmqQ8Imma (5/4 cases) |
| MmqQ4kImmaTile | test-quant | 5 | DELETE | Phase 2C mmq_q4k_imma_tile wrong | `c6500386` #267 f0 | MmqQ8Imma (production) | code reachable only from tests; BenchSweep case is a bench | BenchSweep goes to the bench target or with the code |
| MmqQ8Imma | test-quant | 51 | SHRINK | Q8 IMMA MMQ square, tails, small-M split-K, MoE grouped wrong vs reference | `330e2524` #612 f1 | - | LongPromptBench 0.82 s | LongPromptBench to the bench target |
| Mxf4nvf4MmaBenchTest | test-quant | 1 | MOVE(bench) | none: block-scale vs legacy MMA timing | `596dda58` #54 f0 | - | BENCH | benchmark, own target |
| Mxf4nvf4ProbeTest | test-quant | 2 | MOVE(bench) | src/bench probe kernel launches; zero A gives zero | `596dda58` #54 f0 | - | tests a src/bench kernel | moves with its kernel |
| Mxf4nvf4QkTest | test-quant | 4 | KEEP | mxf4nvf4 Q*K^T (used by nvfp4_gemm_smallm_v2.cu) wrong on indicator/random E2M1 | `105ce2ca` #55 f0 | - | - | NVFP4 MMA class |
| Mxfp4GemmTile | test-quant | 1 | KEEP | MXFP4 tile GEMM accepts a partial atom K | `6e03c7dd` #2911 f0 | - | - | - |
| Mxfp4GemmTileTest | test-quant | 6 | KEEP | MXFP4 tile GEMM differs from decoded reference | `6e03c7dd` #2911 f0 | - | - | - |
| Mxfp4Unpacked | test-quant | 2 | KEEP | MXFP4 unpacked dequant/embedding differs from host decode | `c70d32e1` #2912 (body) f0 | - | - | - |
| Mxfp8Gemm | test-quant | 6 | KEEP | MXFP8 GEMM outside E4M3 budget; partial scale tile | `34479468` #2928 f0 | - | - | - |
| Mxfp8GemmTileBench | test-quant | 1 | MOVE(bench) | none: variant timing | `008ea234` #2902 f0 | - | BENCH 0.44 s | benchmark, own target |
| Mxfp8GemmTileTest | test-quant | 8 | KEEP | MXFP8 tile GEMM differs from host reference | `008ea234` #2902 f0 | - | - | - |
| NvFP4CompressedTensorsRef | test-quant | 4 | KEEP | compressed-tensors two-level scale, zero tensor scale, sign wrong | `eb126946` #126 (body) f1 | - | - | NVFP4 class |
| NvFP4ExportRoundTrip | test-quant | 2 | KEEP | NVFP4 export out of grid bound; offset-folded norm loses gain | `25fd0e60` #1966 f0 | - | - | - |
| Nvfp4GemmAlphaTest | test-quant | 4 | KEEP | NVFP4 alpha not applied / FP8 clamp boundary; Mistral L0 reproducer | `ee35f4b5` #2938 f0 | - | - | NVFP4 MMA class |
| NvFP4GemmGraphCapture | test-quant | 5 | KEEP | NVFP4 GEMM lazy alloc inside capture instead of workspace | REG `454ca581` #121 f2 | - | - | REGRESSION-ORIGIN |
| Nvfp4GemmTile | test-quant | 1 | KEEP | NVFP4 GEMM tile rows depend on M | `ee35f4b5` #2938 f0 | - | - | - |
| Nvfp4GemmTileBench | test-quant | 1 | MOVE(bench) | none: VsCublasLt wall-time bench | `d05fe6d0` #2893 f0 | - | BENCH, enabled: 113.54 s in every unfiltered test-quant run | largest single lever (L1) |
| Nvfp4GemmTileF32 | test-quant | 1 | KEEP | FP32-output NVFP4 tile wrong | `56d391ef` #2937 f0 | - | - | - |
| Nvfp4GemmTileTest | test-quant | 8 | KEEP | NVFP4 tile GEMM differs from reference over shape/variant | `d05fe6d0` #2893 f0 | - | - | NVFP4 MMA class |
| NvFP4GemvGdnInput | test-quant | 2 | KEEP | 4-launch GDN input differs from fused at Qwen3.8 dims | `957d1aa8` #1978 f0 | - | - | - |
| NvFP4GemvGemma3Dims | test-quant | 3 | KEEP | KPAR GEMV, fused QKV, gate/up, GeGLU down wrong at Gemma-3 dims | REG `6946a6cd` #959 f1 | - | - | REGRESSION-ORIGIN |
| NvFP4MergedScaleSpread | test-quant | 1 | KEEP | merged sibling projections share one global scale | REG `ba88b6ba` #1962 f1 | - | - | REGRESSION-ORIGIN |
| NvFP4NormFold | test-quant | 2 | KEEP | folded RMSNorm GEMV differs from RMSNorm then GEMV | `99574388` #2121 f0 | - | - | - |
| NVFP4OutlierTest | test-quant | 2 | KEEP | NVFP4 outlier quant-dequant GEMV differs from independent reference | `d5c9d423` #530 (body) f0 | - | - | - |
| NvFP4PromoteWeightScale2 | test-quant | 9 | MOVE(test-core) | weight_scale_2 NaN/inf/zero promotion wrong for ModelOpt / LLM-compressor | `eb126946` #126 (body) f1 | - | host-only (nvfp4_promote_weight_scale_2, no CUDA call) | loader validation gains CI coverage |
| Nvfp4QuantHwTest | test-quant | 4 | KEEP | NVFP4 KV quant round trip wrong at head_dim 64/128 | `596dda58` #54 f1 | - | - | - |
| Nvfp4QuantRefTest | test-quant | 2 | KEEP | NVFP4 quant round trip error too high; representable values not bit-exact | `596dda58` #54 f0 | - | - | catcher for NVFP4QuantTest |
| NVFP4QuantTest | test-quant | 2 | DELETE | none: no production call, only cudaMalloc/cudaMemcpy and EXPECT_NO_THROW(cudaStreamSynchronize) | `ff4d81d8` f0 | Nvfp4QuantRefTest (quant round trip), NvFP4GemvGemma3Dims (GEMV) | dead: assertion-free on production code | not a kernel-correctness test despite its name; M-NV1 shows the round trip stays covered Mutant note: M-NV1 (nvfp4_quant.cu:344 tensor scale /4) survived every NVFP4 suite incl. this one (gap or equivalent mutant, no reason to keep); M-NV2 red by Nvfp4QuantRefTest only |
| NvFP4SmallMTest | test-quant | 4 | SHRINK | NVFP4 small-M GEMM differs from host reference; A4 variant | `aebf572d` #1756 (body) f1 | - | flaky: 2 bandwidth-floor cases (EXPECT_LT(us, 70.0), 1.13 + 1.05 s) | bandwidth-floor cases to the perf label |
| NvFP4SmallMV2Test | test-quant | 12 | SHRINK | small-M v2 GEMM reference, striped tiles, accumulate wrong | `41b9d7e1` #1766 f2 | - | flaky: BandwidthAboveStarvationFloor EXPECT_GT(best_pct, 40.0) 1.79 s; 4 env-gated sweeps skip everywhere | floor + 4 sweeps to perf/bench; RepeatedLaunchesBitwiseStable 7.33 s: fewer repeats |
| Nvfp4SmallMWsBench | test-quant | 1 | MOVE(bench) | none: DISABLED_ timing | `ab2c3f12` #3011 f0 | - | BENCH | benchmark, own target |
| Nvfp4SmallMWsTest | test-quant | 6 | KEEP | small-M workspace GEMM differs from double reference at every M | `ab2c3f12` #3011 f0 | - | - | - |
| Nvfp4StaticActScaleTest | test-quant | 3 | KEEP | static activation scale wrong in small-M/prompt tile | REG `65f984d0` #3015 f2 | - | - | REGRESSION-ORIGIN |
| Nvfp4TinyBlockScaleTest | test-quant | 2 | KEEP | tiny block-scale SF atom bytes differ from plain quantize | REG `006238cc` #3019 f2 | - | - | REGRESSION-ORIGIN |
| NvFP4ValidatePackedScaleShapes | test-quant | 7 | MOVE(test-core) | packed-scale shape validator accepts bad groups | `eb126946` #126 (body) f1 | - | host-only | gains CI coverage |
| NvFP4ValidateWeightScaleDtype | test-quant | 5 | MOVE(test-core) | weight-scale dtype validator accepts uint8/E5M2/F16 | `eb126946` #126 (body) f1 | - | host-only | gains CI coverage |
| PrefillRowInvariance | test-quant | 3 | KEEP | router logits / MoE act quant / HD512 attention depend on batch and chunk | REG `a11b544f` #2167 f2 | - | - | REGRESSION-ORIGIN |
| Q8PlanesOnly | test-quant | 3 | KEEP | Q8 planes-to-F16 differs from dequant; decline fallback wrong | REG `c09bfbe2` #2712 f1 | - | - | REGRESSION-ORIGIN |
| QuantIntegrationTest | test-quant | 24 | KEEP | GGUF Q4_0/Q8_0/Q4_K/Q5_K upload, forward, determinism wrong on synthetic models | `ff4d81d8` f4 | - | - | catcher for QuantPipelineTest Mutant note: not a catcher for QuantPipelineTest: M-QP survived it |
| QuantizeMoeNative | test-quant | 10 | KEEP | MoE native NVFP4 quantize offsets, alpha, SFA offsets wrong | `dc236560` #161 (body) f3 | - | - | - |
| QuantTest | test-quant | 2 | KEEP | QuantConfig defaults; INT4 GEMM basic wrong | `ff4d81d8` f0 | QuantIntegrationTest.QuantGemmInt4LargerMatrix | - | QuantConfigDefaults could move to test-core |
| SmallMV2Pair | test-quant | 3 | SHRINK | small-M v2 pair not bit-exact to two single calls | `86c6a686` #1788 f1 | BatchedSmallM | 1 DISABLED_ M1 bench | DISABLED_ bench to the bench target |
| TmaBlockScaleBench | test-quant | 1 | MOVE(bench) | none: descriptor launch timing | `dc236560` #161 (body) f1 | - | BENCH | benchmark, own target |
| WeightDispatchTest | test-quant | 13 | KEEP | weight dispatch FP16/FP8/NVFP4/CUTLASS/MXFP4 differs from direct kernel | `c8763ade` #27 f4 | - | - | - |
| ChatTemplateApplyTest | test-text | 20 | KEEP | breaks: Chat template apply: ChatML, Llama 2/3, Nemotron, Gemma, DeepSeek-R1, Phi structure | 78a3c057 test | ChatTemplateGolden pins bytes but only on the Jinja path | behaviour | apply() without template source runs the hardcoded family path; the golden goes through render_jinja, so no overlap. |
| ChatTemplateArchDefaultTest | test-text | 1 | KEEP | breaks: Chat template default family for each architecture | 78a3c057 test | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ChatTemplateContinue | test-text | 6 | KEEP | breaks: Chat template continue: Qwen3 trimmed and verbatim content match HF; cut rule | 43d94343 feat #2764 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ChatTemplateDefaultSysTest | test-text | 5 | KEEP | breaks: Chat template default system prompt: Jinja extraction; injected only without user system | 78a3c057 test | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ChatTemplateDetectTest | test-text | 14 | KEEP | breaks: Chat template family detection from template text (ChatML, Llama, Gemma, Mistral, Phi) | 78a3c057 test | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ChatTemplateFamilyNameTest | test-text | 1 | KEEP | breaks: Chat template family names for every family | 78a3c057 test | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ChatTemplateGolden | test-text | 13 | KEEP | breaks: Chat template goldens: per-family rendered prompts vs committed golden; generation prompt | 871a6492 test #1721 | none found | behaviour; 10 case(s) assert via file helper | no overlapping suite found, runtime 0.00C |
| ChatTemplateInitTest | test-text | 10 | KEEP | breaks: Chat template init: per-family success, XML tool dialect detection, missing-token fallback | 78a3c057 test | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ChatTemplateParseFamilyTest | test-text | 1 | KEEP | breaks: Chat template parse_family: every name maps to its family | 78a3c057 test | none found | behaviour | no overlapping suite found, runtime 0.00C |
| ChatTemplateTokenize | test-text | 1 | KEEP | breaks: Chat template tokenize: control-token strip flags swallow template whitespace | REG 26171d68 fix #2159 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ChatTemplateTools | test-text | 1 | KEEP | breaks: Chat template tools: parameter past int64 max renders instead of throwing | REG 92c01c5c fix #1973 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| ChatTemplateVisionTest | test-text | 6 | KEEP | breaks: Chat template vision: Gemma image tokens per turn; text-only fallback for other families | 78a3c057 test | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Cl100kPreTokenizeTest | test-text | 1 | KEEP | breaks: cl100k pre-tokenizer: digit triples and case-blind standalone contractions | REG 9298b1c7 fix #664 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| DigitTriplesRightSplitTest | test-text | 1 | KEEP | breaks: Llama-BPE digit triples: right-aligned split only before a word boundary | f712136b feat #2604 | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| HarmonyGoldenTest | test-text | 4 | KEEP | breaks: gpt-oss Harmony golden: user-only, system+user, multi-turn final channel, channel markers | REF? f6419bf5 test #587 (refs 534,547,560,572,574) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| HFChatTemplateTest | test-text | 5 | KEEP | breaks: HF tokenizer_config chat_template: string, array default/fallback/empty, missing field | 6fef186f wip | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingChatTemplate | test-text | 4 | KEEP | breaks: Inkling chat template: reasoning levels advertised, named and numeric levels, refusals | 6da70aff feat #3222 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingDraftPolicy | test-text | 3 | KEEP | breaks: Inkling draft policy: rows follow measured chance; out-of-vocab token ends the draft | fe6fd810 perf #3229 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| InklingEffortSpec | test-text | 1 | KEEP | breaks: Inkling effort spec: other templates accept any value | 6da70aff feat #3222 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| JinjaChatTest | test-text | 7 | KEEP | breaks: Jinja chat templates: ChatML, Gemma, Qwen3 and multi-turn render vs expected text | b25f3b19 feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| JinjaFilterTest | test-text | 1 | KEEP | breaks: Jinja selectattr filter | a6892787 feat #2599 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| JinjaParity | test-text | 8 | KEEP | breaks: Jinja parity: real chat templates (Qwen3, Nemotron, Qwen3.8, GLM, Flint, Kimi, gpt-oss) | REG 657dacff fix #3148 | none found | behaviour; 7 case(s) assert via file helper | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| JinjaTest | test-text | 65 | KEEP | breaks: Jinja engine: expressions, control flow, filters, slices, macros, block set, whitespace | b25f3b19 feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| JinjaUndefinedTest | test-text | 6 | KEEP | breaks: Jinja undefined: unset vs false/true variables, is defined, Qwen thinking guard | REG b5e37182 Qwen3.8-27 #1750 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| JinjaValueTest | test-text | 6 | KEEP | breaks: Jinja value types: none, bool, int, string, array, object conversions | b25f3b19 feat | none found | behaviour | no overlapping suite found, runtime 0.00C |
| KimiChatTemplate | test-text | 3 | KEEP | breaks: Kimi chat template render: one turn, multi-turn with system, tool-call history | 63a9b933 feat #2984 | none found | behaviour | no overlapping suite found, runtime 0.00C |
| Ling3ChatTemplate | test-text | 3 | KEEP | breaks: Ling-3 chat template: thinking default, thinking off, thinking forced | REF? 1bd027ac feat #2979 (refs 2978) | none found | behaviour | introduced with issue refs, no proven second catcher; no overlapping suite found, runtime 0.00C |
| NfcNormalize | test-text | 2 | KEEP | breaks: NFC normalization: Latin combining and Hangul jamo compose; decompose, reorder, recompose | REG c1703da3 fix #2122 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| O200kPreTokenizeTest | test-text | 8 | KEEP | breaks: o200k pre-tokenizer: code lines, digit triples, URLs, contractions, case runs, symbols | REG 6cc97c07 fix #662 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| PreTokenizeSplitSequenceTest | test-text | 4 | KEEP | breaks: HF-json pre-tokenizer split sequences match HF on DeepSeek-shaped steps; regex refusals | REG 26171d68 fix #2159 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| PreTokenizeUnicodeTest | test-text | 4 | KEEP | breaks: Unicode pre-tokenize: emoji symbols, whitespace/digits/CJK punctuation, marks, cased runs | REG 26171d68 fix #2159 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Qwen2PreTokenizeTest | test-text | 10 | KEEP | breaks: Qwen2 pre-tokenizer: symbol runs, digits, contractions, newline and trailing whitespace | REG 032c7396 fix #660 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| Qwen38ReasoningEffort | test-text | 9 | KEEP | breaks: Qwen3.8 chat template reasoning effort: advertised values, default xhigh, preamble | REG b5e37182 Qwen3.8-27 #1750 | none found | behaviour | REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TiktokenLoader | test-text | 5 | KEEP | breaks: tiktoken loader: lowest-rank merges, atomic specials, Kimi tool names, malformed ranks | d8550086 feat #2981 | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| TokenizerAddedTokens | test-text | 3 | KEEP | breaks: Added tokens: non-special atomic, normalized flags, lstrip/rstrip whitespace swallowing | REG 13278485 fix #719 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerControlTest | test-text | 4 | KEEP | breaks: SPM control tokens: mark-as-control allocates, idempotent, ignores bad ids, keeps types | 52d87981 feat #77 | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| TokenizerDispatchTest | test-text | 3 | KEEP | breaks: Tokenizer dispatch: SPM default, set_type, max-length input | 78a3c057 test | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| TokenizerEos | test-text | 2 | KEEP | breaks: EOS: first added EOS replaces placeholder; a named EOS keeps first place | REG 7d0b045b fix #2378 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerGemma4Test | test-text | 4 | KEEP | breaks: Gemma-4 tokenizer: merge skipped when result not in vocab; truncated UTF-8; byte fallback | 95d3e779 tokenizer #37 | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| TokenizerGPT2Test | test-text | 11 | KEEP | breaks: GPT-2 byte-BPE: encode, merges, decode roundtrip, digits, spaces, UTF-8, stale ranks | 78a3c057 test | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| TokenizerJsonHostileIds | test-text | 4 | KEEP | breaks: tokenizer.json robustness: negative or huge vocab ids dropped; well-formed vocab loads | REG e86420b1 fix #1660 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerJsonPretok | test-text | 5 | KEEP | breaks: tokenizer.json pre-tokenizer rewrite: digit runs, splits, char and Onig classes vs HF | REG 5592ddc1 fix #2749 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerLlamaBpeTest | test-text | 2 | KEEP | breaks: Llama-BPE pre-tokenizer: cl100k chunks for Llama, Nemotron chunks for Tekken | REG 04020f07 fix #2528 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerNormalizer | test-text | 2 | KEEP | breaks: Normalizer: NFC only when tokenizer JSON declares it; Gemma-4 JSON routes to Gemma-4 | REG c1703da3 fix #2122 | none found | behaviour | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerRobustness | test-text | 16 | KEEP | breaks: Byte tokenizer roundtrip for ASCII, UTF-8, NUL and surrogate bytes; decode ids never crash | REG db6c47f0 test #532 | none found | behaviour; 7 case(s) assert via file helper | protected class (tokenizer); REGRESSION-ORIGIN, no proven second catcher; no overlapping suite found, runtime 0.00C |
| TokenizerSPMTest | test-text | 16 | KEEP | breaks: SPM tokenizer: vocab load, encode words and subwords, byte fallback, decode, UTF-8 and CJK | 78a3c057 test | none found | behaviour | protected class (tokenizer); no overlapping suite found, runtime 0.00C |
| TranscriptIdStoreTest | test-text | 2 | KEEP | breaks: Transcript id store: longest prefix wins with FIFO cap; resent reply keeps model split | a3a94092 feat #2038 | none found | behaviour | no overlapping suite found, runtime 0.00C |

## Phase 3: plan

Stop here: no test, build or CI file changes before the owner's Go. Phase 4 executes these rows; Phase 5 re-runs every mutant of Phase 2 against the new suite.

Anti-cheat rules (owner assignment, binding for every row):

| # | rule |
|---:|---|
| R1 | No assertion, atol/rtol or reference is loosened to make a test faster or green |
| R2 | No test leaves through `GTEST_SKIP`/`DISABLED_`: deleted with the reason in this report, or kept |
| R3 | A test from a regression fix stays unless another test catches the same regression by mutant |
| R4 | Kernel-correctness suites (NVFP4/FP8 MMA, Gated DeltaNet, paged KV, prefix cache, sampling, tokenizer, grammar/JSON schema) only SHRINK or MERGE, never DELETE |
| R5 | Benchmarks move to their own target; none is deleted |

Savings: GPU s = sum of per-case medians of the quiet baseline (`p3/levers_sum.md`, `scripts/tsa3_levers_sum.py`) unless the row names an A/B; CI s from the Phase 1 CI-lane medians; build s from the Phase 1 TU medians. `n/m` = not measured, no number claimed.

| # | measure | files | verdict ids (suites) | CI lane s | local GPU s | build s | risk | mutants that must stay red |
|---:|---|---|---|---:|---:|---:|---|---|
| P1 | Bench target: an opt-in `test-bench` module (built in CI, run by `make test-perf`, not by `imp-tests`/`make test-gpu`) takes the 24 MOVE(bench) suites and the bench cases of 14 SHRINK suites, incl. `Nvfp4GemmTileBench.VsCublasLt` | `CMakeLists.txt` (`imp_add_test_module` lists), `tests/test_nvfp4_gemm_tile.cu:273`, `tests/test_gemm_q45k_f16_tile.cu` and the other bench sources of `p3/levers_sum.md` L2, `Makefile` `test-perf` | L1 Nvfp4GemmTileBench; L2 GemmQ45kTileBench, GemmQ6kTileBench, GemmF16TileBench, GemmMxfp4TileBench, GemmNvfp4TileBench, MmaVariantsBench, Mxfp8GemmTileBench, GroupedNvfp4TileTest, Mxf4nvf4MmaBenchTest, Mxf4nvf4ProbeTest, Fp4PvBenchTest, TmaBlockScaleBench, FmhaVLoadBench, MmaPeakSaturatedTest, PagedF16Decode, PagedFp8Decode, PagedNvfp4Decode, GemvFp16HotCold, MoeRouterFusedBench, Fp8GemmSmallMBench, Fp8GemmTileBench, Nvfp4SmallMWsBench, MmqQ4kImmaBench; bench cases of GemmF16NarrowPrefill, GemmF16NarrowSmallM, MmqQ8Imma, GroupedNvfp4DecodeTest, FmhaFA2Hd256Test, FmhaHd512Test, GemmCaptureSm120, InklingAttnPaged, InklingPrefill, BatchedSmallM, SmallMV2Pair, FP8GemmTest, FP8ProjStreamTest, NvFP4SmallMV2Test (sweeps) | 0 | L1 A/B 112.78 (test-quant 168.06 -> 55.28, 3 runs each, load <= 1.20); L2 23.44 | 0 (TUs move, not removed) | a bench nobody runs rots: build it in CI, run in `make test-perf`; R5 | all 23 part-B mutants (none depends on a bench case) |
| P2 | Wall-clock floors to the perf lane: the 3 bandwidth-floor cases leave the default GPU run, assertions unchanged | `tests/test_nvfp4_smallm.cu:315`, `tests/test_nvfp4_smallm_v2.cu`, `Makefile` | L3 NvFP4SmallMTest (2 cases), NvFP4SmallMV2Test.BandwidthAboveStarvationFloor | 0 | 3.77; flake source: red 2 of 2 at load 16 | 0 | floor regressions show only in `make test-perf` | none (perf floors, no mutant) |
| P3 | Lane dedup: `make test-gpu` stops running the 9 model suites that the `make test-e2e` groups run with model env (negative filter, or env-only defaults) | `Makefile:256-350` (`test-gpu`, `test-e2e`), `tests/test_serving_signals.cpp:28` and the other `/models/...` defaults | L4 ServingSignalsTest, MtpGreedyIdentityTest, SpecCaptureFidelityTest, BatchInvarianceTest, MtpQwen4ExpReference, DegenerationTest, DSparkDrafterTest, DFlashDrafterTest, DeviceFaultSignalTest | 0 | 448.75 per `make test-gpu` (33 cases; test-e2e binary 503.67) | 0 | `make test-gpu` alone no longer runs them: `verify`, `check-release` and the `IMP_HOOK_GPU` hook must run `test-e2e` | M-E2E1, M-E2E2 (model battery) |
| P4 | ServingSignals TTFT case: drop the dead post-TTFT wait. Lever L5 as planned (prewarm 64 -> 25) is REFUTED: case 157.09 -> 158.00 s (A/B, 3 runs each, load <= 1.27, rc 0). The cost is `(void)drain(s, 10000)`: with `max_tokens = 1` the first popped event is already `is_last`, so each of the 15 requests waits the full 10 s; mutant `if (!ev.is_last)` before the drain: case 157.09 -> 7.15 s (3 runs, rc 0, every assertion green; load 8.3-9.9, contended) | `tests/test_serving_signals.cpp:323` | ServingSignalsTest (SHRINK, REGRESSION-ORIGIN #1915) | 0 | 149.94 per run of the case (test-e2e model battery; also test-gpu unless P3 lands) | 0 | R1: no assertion touched, a dead wait goes; requests now arrive back to back instead of 10 s apart; prewarm stays 64 | add in Phase 4: prewarm off (`prefill_graph_prewarm` ignored) must turn the case red |
| P5 | test-attention host references: the 14 cases > 5 s are host-bound (test-attention at 90 % of one core; `FmhaSm120Test` reference is a scalar 4-deep host loop, `tests/test_attention_fmha_sm120.cu:32`): thread the reference over heads, reuse one reference per shape across window variants; same values, same order per output | `tests/test_attention_fmha_sm120.cu`, `tests/test_fmha_fp8.cu`, `tests/test_gpt_oss_sinks_ref.cu` | L11 FmhaSm120Test, GptOssSinkRef, FmhaFA2Dense2CtaTest, FmhaFA2Test, FmhaFA2Hd256Test (KEEP/SHRINK) | 0 | n/m (the 14 cases sum to 216.06 of 240.83) | 0 | a threaded reference with a different summation order changes reference bits: keep per-output order | M-PA1..3b, M-INT4 (attention binary) |
| P6 | Host-only suites to test-core (CI lane gains them) | `CMakeLists.txt` module lists, the 10 suites' files | L9 PreambleGateTest, NvFP4PromoteWeightScale2, NvFP4ValidatePackedScaleShapes, JsonConstrainFsmTest, NvFP4ValidateWeightScaleDtype, RegexNfaTest, MtpUploadBudgetTest, GemmStrategy, SparsePrefillGeometry, VramBudgetReserve (5 host cases) | +n/m (69 cases; the 64 outside VramBudgetReserve sum to 0.00 s on the GPU run) | 0.00 | n/m | test-core TU count grows; CI lane must stay skip-free (`guard_unit_skips`) | M-GR, M-GR2 lane unaffected; CI coverage for 4 REGRESSION-ORIGIN suites (#1067, #497, #1494, #874) |
| P7 | DELETE 10 suites, 21 cases (part A 4, part B 6), plus the dead Phase 2C sources `src/compute/mmq_q4k_imma_{layout,tile}.cu` | part A: `tests/test_launcher_args.cpp`, `tests/test_tool_call.cpp`, `tests/test_hf_fetch_network.cpp` + target `test-hf-network`; part B: `tests/test_nvfp4_quant.cu`, `tests/test_paged_attention.cu:1461`, `tests/test_mmq_q4k_imma_{gemm,layout}.cu`, `tests/test_mmq_q4k_imma_tile_bench.cu`, `tests/test_tensor_kind_coverage.cpp` | GdnChunkparArgs, SsmScanPrefillArgs, ToolCallDispatch, HfFetchNetwork, INT4QuantTest, NVFP4QuantTest, MmqQ4kImmaTile, MmqQ4kImmaGemm, MmqQ4kImmaLayout, TensorKindCoverage | ~0 | 0.60 (with the P8 MMVQ/DP4A cases) | 42.04 CPU-s of whole test TUs (6.77 + 10.70 + 10.60 + 8.73 + 2.66 + 2.58) + the 2 src TUs (n/m) | MmqQ4kImmaBench benches only the deleted code: R5 forbids deleting a bench, owner decides whether it goes with the code | LA-1..4 (BUILD_RED), TD-1, TD-2, HF-1..3, M-INT4, M-NV2, M-Q4K1, M-Q4K2, M-TK |
| P8 | SHRINK cases that a stronger suite covers: MMVQ 3 `*_MatchesDp4a`, GemmDP4ATest 2 basics, EndToEndModelTest 4 (LoadModel, Tokenize, CreateContextAndGenerate, PrefillDecodeStep), 7 `CarriesEveryFieldInDeclarationOrder`, `FpBitsLegacyParity.Fp16EncodeFloatSweep` window | `tests/test_mmvq.cu`, `tests/test_gemm_dp4a.cu`, `tests/test_e2e.cpp:130-257`, `tests/test_launcher_args.cpp`, `tests/test_gemm_lt_args.cpp`, `tests/test_fp_bits_legacy_parity.cpp` | MMVQ, GemmDP4ATest, EndToEndModelTest, GdnScanArgs, GdnScanBatchedArgs, KvWriteRopeArgs, LtGemmArgs, LtGemmBatchedArgs, PagedDecodeArgs, QknormRopeArgs, FpBitsLegacyParity | up to 2.38 (sweep case median, gtest CPU; lane wall unchanged while `guard_makefile_filters` 10.76 s is the critical path) | model battery 8.81 (4 cases, 1 run, dev build, load ~14: contended) | < 1 | the sweep keeps the full tie neighbourhood (FP-1 red only by the sweep) | M-MV1..3, M-DP1b, M-DP2, M-E2E1, M-E2E2, FP-1, FP-2, LA-1..4 |
| P9 | MERGE: PagedAttentionTest overlap into PagedOracle rows (keep batch-length, sinks, HD512 cases), Grammar+Regex mask into one TEST_P (both cases kept), 5 FA2 suites into one TEST_P over variant x shape (every shape kept), 6 rename-only (3 Repeat suites, KVCacheManagerGrowTest, MoERoutingEdgeTest), 5 part-A split suites into one file each | `tests/test_paged_attention.cu`, `tests/test_attention_paged_oracle.cu`, `tests/test_grammar_regex_mask.cu`, `tests/test_fmha_fp8.cu`, the rename files | PagedAttentionTest, GrammarConstrainMaskTest, RegexConstrainMaskTest, FmhaFA2{,Dense2Cta,Hd256,HeavyFirst,PvF16}Test, GemmMxfp4RepeatTest, GemmQ45kRepeat, GemmQ6kRepeatTest, KVCacheManagerGrowTest, MoERoutingEdgeTest, AnthropicCacheUsage, AnthropicThinking, StreamReasoningSplit, StripThinkBlock, ToolCallQwen36Xml | 0 | n/m (maintenance; no case removed except PagedAttentionTest overlap) | n/m | a merge that drops a shape loses coverage: diff the case list before/after | M-PA1, M-PA2, M-PA3a, M-PA3b, M-GR, M-GR2 |
| P10 | LlmCompressorE2E: fix the coder default path (`/models/Qwen3-Coder-30B-A3B-Instruct-FP4`), one shared Gemma-4 llmc load for its 2 cases | `tests/test_e2e_llm_compressor.cpp:31` | LlmCompressorE2E (SHRINK) | 0 | n/m (the 4 cases sum to 8.85; one load of the pair) | 0 | the fixed path makes a skipped case run: runtime goes up | none (no mutant) |
| P11 | Wire the dead suites (anomalies B1-B3): env for CandidateTokenGuardTest, FimRealTokenizer; filters for DecodeGraphCtxGrowthTest, HybridSharedScoreTest, PromptLogprobsE2ETest; re-point ChunkedPrefillTest, GgufDeadWeightForms, LoraHotSwap, Q8DropSourceE2E, QuantPipelineTest, VramLedgerE2E; refs for Gemma4AudioEncoderTest, Gemma4E4BRef, Ling3Ref; fixtures for EncoderEmbedTest, SentencePieceLoader; VisionGolden re-golden; Inkling oracle dir (22 cases) | `Makefile` test-e2e groups, `tools/test_model_env_allowlist.txt`, the suites' default paths | the 16 MOVE(wire/re-point) suites of part B + Inkling oracle suites | 0 | negative, n/m (adds runtime) | 0 | 5 REGRESSION-ORIGIN suites (#948, #130, #2711, #2712, #2713) run nowhere today; owner decides VramLedgerE2E, VisionGolden, EncoderEmbedTest | M-QP, M-Q8 must turn red once QuantPipelineTest and Q8DropSourceE2E are re-pointed |
| P12 | CI lane critical path: `guard_makefile_filters` (10.76 s) vs `unit_core` 4.74 s | `scripts/check_makefile_filters.sh` (`CMakeLists.txt:1816`) | (part A lever) | up to ~6 wall (A, not measured) | 0 | 0 | guard keeps its checks | none |
| P13 | `httplib.h` out of the test TUs (PCH or a header without it) | `tests/` server-handler TUs, `tools/imp-server/utils.h`, `handlers_internal.h` | (part A lever) | 0 | 0 | n/m (A: `httplib.h` 8.8-9.9 s per TU single, 23 test-core TUs >= 8 s; an estimate, measure in Phase 4) | none | none |
| P14 | Shape/repeat shrink candidates, each to be read before a claim | `tests/test_nvfp4_smallm_v2.cu`, `tests/test_mtp_feed_batch.cu`, `tests/test_fp8_gemm.cu`, `tests/test_fp8_gemm_smallm.cu`, `tests/test_sampling_topk_radix.cu` | L7 NvFP4SmallMV2Test, MtpFeedBatchTest, FP8GemmTest, Fp8GemmSmallMTest, TopkRadixSampler | 0 | n/m (the 5 cases sum to 18.98) | 0 | R1: fewer repeats or smaller shapes must keep every tail/tile path | the suite's own mutants, to be written in Phase 4 |

Expected savings (no overlap counted twice: P4 lands in the model battery because P3 takes ServingSignalsTest out of `make test-gpu`):

| lane | baseline | rows | saving s | after s |
|---|---:|---|---:|---:|
| `make test-gpu` (8 binaries + 3 extra lines, sum of medians) | 1022.51 | P1 112.78 + 23.44, P2 3.77, P3 448.75, P7/P8 0.60 | 589.34 | 433.17 |
| `make test-e2e` model battery | n/m (not timed whole) | P4 149.94, P8 8.81 (contended) | 158.75 | n/m |
| CI lane `ctest -L unit` | 10.77 wall | P8 up to 2.38 gtest CPU-s; P12 up to ~6 wall (estimate) | 2.38 CPU, 0 wall until P12 | 10.77 |
| clean build of the test targets | 227.4 wall | P7 42.04 CPU-s of test TUs; P13 n/m | 42.04 CPU-s | n/m |

Coverage moves the other way on purpose: P6 puts 69 host cases (4 REGRESSION-ORIGIN suites) into the CI lane, P11 makes 5 REGRESSION-ORIGIN suites run at all. Verdicts after Phase 2: 753 KEEP, 32 SHRINK, 13 MERGE, 51 MOVE, 10 DELETE over 859 suites; DELETE = 21 cases, each with a red mutant on its remaining catcher or a dead-code / build proof.

Owner decisions needed before Phase 4: MmqQ4kImmaBench with the deleted Phase 2C code (R5); VramLedgerE2E, VisionGolden, EncoderEmbedTest re-point or delete; QuantPipelineTest / Q8DropSourceE2E re-point target (Qwen3.8-27B-Q8_0.gguf fit on 32 GB unverified).
