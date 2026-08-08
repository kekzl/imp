# Test inventory — baseline for the test-hardening audit

Date: 2026-08-08 · Commit: `ad067d76` (v0.23.0, `main`) · Box: RTX 5090 / WSL2 /
CUDA 13.3.1 · Build: `build-dev` (incremental, identical codegen to `make build`)

Every number here was produced by a command whose output is in
`loop/evidence/`. Nothing is quoted from documentation. Where a number
contradicts prose elsewhere in the repo, the prose is listed in
`loop/STALE_CLAIMS.md`.

**Coverage percentage is deliberately absent.** It is a diagnostic for finding
unexecuted code, not a quality claim. The quality metric for this audit is the
mutation score (`docs/audit/MUTATION_BASELINE.md`).

---

## 1. What exists

| Target | Test cases | Runs in CI? | Runtime (GPU box) | Gate condition |
|---|---:|---|---:|---|
| `test-core` | 955 | **yes** (`ctest -L unit`) | 0.18 s (CPU) / 1.4 s (GPU) | — |
| `test-text` | 197 | **yes** | 0.08 s | — |
| `test-e2e` (CPU slice) | 39 | **yes** (gtest filter) | 0.06 s | — |
| `guard_e2e_lane_split` | 1 | **yes** | 0.07 s | — |
| `test-compute` | 204 | no | 3.3 s | GPU |
| `test-attention` | 200 | no | 279 s | GPU |
| `test-quant` | 199 | no | 15.3 s | GPU |
| `test-kv` | 58 | no | 0.4 s | GPU |
| `test-moe-gdn` | 144 | no | 0.8 s | GPU |
| `test-e2e` (GPU slice) | 129 | no | 122 s | GPU + models |
| `tests/api` (pytest) | 91 | **yes, against a mock** | ~5 s | `IMP_USE_MOCK=1` |

C++ total on `main`: **2 125 gtest cases** across the eight binaries, from
2 110 `TEST*` bodies in `tests/` (91 `.cu` + 89 `.cpp`); the difference is
`DISABLED_` plus TYPED/parameterised expansion.

> **Provenance correction (2026-08-08).** The first pass of this table was
> measured against a `build-dev` tree that had been compiled on the branch
> `fix/answer-reserve-scales`, so `test-core` was reported as 958 — three
> `ForceThinkEnd.*` cases that exist only on that branch. Every count in this
> file has been re-measured against `main` (`loop/evidence/P0b-gpu-per-binary/`).
> The lesson generalises: **an incremental build directory carries whatever
> branch last compiled into it**, and `git checkout` does not rebuild it.

CI jobs (`.github/workflows/ci.yml`): `Build` (**the only required check**),
`clang-tidy` (advisory), `Mock API contract`, `Lint` (advisory), `File size`,
`Release hygiene`. The `test` job — full ctest + compute-sanitizer + perf gate —
is gated on `vars.HAS_GPU_RUNNER`, which is unset by owner decision (2026-08-03),
so it has never run.

## 2. The three gates, and what each can actually see

| Gate | What it runs | Kernels executed? |
|---|---|---|
| **GitHub CI** (`Build`, required) | `ctest -L unit` on a GPU-less container | **none** |
| **`make verify-fast`** (pre-push hook, installed) | gtest filter on `imp-tests` | a subset |
| **`make test-gpu`** (pre-commit hook) | full GPU suite | all — **but the hook is not installed** |

### 2.1 CI executes 1 130 test cases in 0.39 seconds and touches no CUDA kernel

```
$ docker run --rm -v $PWD:/src -w /src/build-dev imp:toolchain \
    ctest -L unit --output-on-failure --timeout 300
100% tests passed, 0 tests failed out of 4
Total Test time (real) = 0.39 sec        # loop/evidence/P0-cpu-lane.log
```

That is the complete correctness signal behind a merge. It is not a defect by
itself — the repo has no GPU runner on purpose — but it fixes the ceiling on
what a green PR can mean.

### 2.2 61 test cases silently skip in CI, 54 of them the KV-cache suite

Same binaries, GPU present vs absent:

| Binary | no GPU | with GPU |
|---|---|---|
| `test-core` | 896 pass, **60 skip** | 955 pass, 1 skip |
| `test-text` | 196 pass, 1 skip | 196 pass, 1 skip |

The 60 skipped `test-core` cases (`loop/evidence/P0-core-nogpu.log`):

| Suite | Cases | What goes dark |
|---|---:|---|
| `KVCacheManagerTest` | 44 | block allocation, LRU eviction, prefix-hash chaining, pin budget, SWA snapshot/rollback |
| `KVCacheTest` | 10 | pool sizing, block copy |
| `MLAConfig` | 4 | DeepSeek MLA YaRN/mscale parsing |
| `TensorKindCoverage` | 1 | |
| `SentencePieceLoader` | 1 | |

`KVCacheManagerTest` is pure host-side bookkeeping — hash chaining, an LRU list,
a pin FIFO — yet it is CUDA-gated because the fixture allocates a real pool. It
is therefore invisible to the only gate that runs on every PR, in the subsystem
where a paged-attention engine is most likely to corrupt state silently.
`make verify-fast`'s filter does include `KVCache*`, so these are *conditional*,
not dead — conditional on a local pre-push run on a 5090.

### 2.3 The Stage-1 GPU hook is not installed on this box

`.git/hooks/pre-push` is present and byte-identical to `scripts/pre-push.hook`.
`.git/hooks/pre-commit` **does not exist**, so `scripts/pre-commit.hook`
(`exec make -s test-gpu`, the full GPU suite) never runs. CI's own comment —
"GPU correctness lives in the local pre-commit hook (Stage 1)" — describes a
gate that is not armed here.

### 2.4 `verify-fast` runs a filter, not the suite

`scripts/verify.sh:233`:

```
FILTER="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:KVCache*:GemmTest.*:
        FP8GemmTest.*:SamplingTest.*:SoftmaxTest.*:AttentionTest.*:VramBudget*"
```

Not matched by that filter, and therefore not covered by any gate that runs
automatically: every `Fmha*`, `Paged*`, `Rope*`, `MRope*`, `Moe*`, `Gdn*`,
`Ssm*`, `Quantize*`, `Dequant*`, `PrefixCache*`, `ChunkedPrefill*`, `Spec*`,
`Mtp*`, `Vision*` and `Lora*` suite.

### 2.5 The API surface is tested against a reimplementation of itself

`Mock API contract` is the only CI job that touches the HTTP API. It runs
`pytest tests/api -m "not perf and not tools"` (82 of 91 collected) with
`IMP_USE_MOCK=1`, which starts `tests/api/mock_server.py` — 555 lines of Python
that reimplement the endpoints. `tools/imp-server/` is **never executed by CI**.
The suite's own docstring is candid about its scope
(`tests/api/test_contract.py:8`): *"What it does NOT test: Model correctness,
token quality, numerical precision."*

## 3. Skips, disables and gates

| Marker | Count | Notes |
|---|---:|---|
| `GTEST_SKIP` call sites | 174 | across 30 files |
| `DISABLED_` tests | 4 real (+2 in comments) | 2 benchmarks, 2 determinism known-limits |
| `#if 0` | 0 | in `tests/`, `src/`, `include/` |
| pytest `skip`/`xfail` | 11 | |
| env-gated model tests | via `tests/test_models.h` | 13 `IMP_TEST_*` variables |

The dominant gate conditions are `!can_run()` (41), `!gpu_available()` (17),
`!has_sm120()` (10) and model-file presence (~25).

**Dead vs conditional.** No test was found that runs *nowhere*. The four
`DISABLED_` tests are the only permanently-off cases:

| Test | Reason recorded in the source |
|---|---|
| `DetEvalE2ETest.DISABLED_GreedyReproducibleAcrossFreshContexts` | known limit: layout-sensitive across fresh contexts |
| `DetEvalE2ETest.DISABLED_PerplexityBitIdenticalAcrossFreshContexts` | same |
| `FhmaMxFP4Test.DISABLED_BasicHD256` | HD=256 needs more shared memory |
| `FmhaHd512Test.DISABLED_BenchVsCublas` | benchmark, not an assertion |

`FmhaHd512Test.DISABLED_BenchLongCtxFallback` also exists but is a benchmark.

**The real gap is not dead tests — it is that 934 of 2 125 executable cases
(44 %) require a GPU that no automated gate has.** CI registers 1 191 of them
(`test-core` + `test-text` + the 39-case e2e slice) and 61 of those skip for
want of a device, so 1 130 actually execute.

### 3.1 A wrong model path fails instead of skipping

`tests/test_determinism_e2e.cpp:54` skips on `!path` — i.e. only when the env
var is *unset*. Point `IMP_TEST_MOE_MODEL` at a path that does not exist and the
suite reports hard failures instead of skips. This audit hit it: pointing at
`/models/Qwen3-30B-A3B-Q4_K_M.gguf` (the file is one directory deeper) produced
20 red tests that were purely a harness typo. Evidence:
`loop/evidence/P0-gpu-lane.log`.

## 4. Assertion strength

Mechanical screen over all 2 110 `TEST*` bodies
(`tools/mutation/classify_assertions.py`, raw data
`loop/evidence/P0-assertion-classes.json`). One-line tests that delegate to a
helper are resolved through the helper (245 of them do), so a golden-checked
kernel test is not miscounted as a smoke test.

| Class | Count | Share |
|---|---:|---:|
| A0 — smoke / boolean only | 402 | 19.1 % |
| A1 — shape/type only | 37 | 1.8 % |
| A2 — weak value (range, monotone) | 123 | 5.8 % |
| A3 — fixed expected value | 1 274 | 60.4 % |
| A4 — independent oracle / golden | 274 | 13.0 % |

**This does not support "the suite asserts nothing".** 73 % of tests compare
against a fixed expected value or an independent reference. 246 `EXPECT_NEAR`
and 83 `EXPECT_FLOAT_EQ` sites carry explicit tolerances.

Two caveats, stated because they cut against the headline:

* The A0 bucket is dominated by constrained-decoding suites
  (`test_json_constrain.cu` 40/44, `test_gbnf_grammar.cpp` 22/22,
  `test_constraint_validation.cpp` 24/26). For an *acceptor*, `EXPECT_TRUE
  (accepts(s))` is the value under test, not a smoke check — several of these
  are in fact A4 (`test_json_constrain_property.cpp` cross-checks every case
  against `nlohmann::json::accept` as an independent oracle). The mechanical A0
  share overstates weakness here.
* Only **5** tests contain no assertion at all, and 4 are benchmarks. The fifth,
  `TmaBlockScaleBench.BothDescriptorsLaunch` (`tests/test_tma_block_scale_bench.cu:28`),
  is named as a check but asserts nothing.

Hand-read sample (40 tests, hot paths). The weaknesses found are specific, not
systemic:

| Test | file:line | Class | Finding |
|---|---|---|---|
| `SamplingTest.NaNLogits` | `tests/test_sampling.cu:211` | A2 | Asserts only `0 <= token < V` under a NaN logit — "no crash". Cannot detect NaN propagating into a plausible token. |
| `SamplingTest.TopPFiltering` | `tests/test_sampling.cu:164` | A2 | Comment says `top_p=0.5`; the code passes `0.99`. Competing logits are `5.0` vs `-100.0`, so ignoring `top_p` entirely still satisfies it. |
| `SamplingTest.TopKRespectsK` | `tests/test_sampling.cu:132` | A2 | One logit at `10.0`, the rest at `-100.0`: ignoring `top_k` still returns token 0. |
| `SamplingTest.TemperatureZeroIsGreedy` | `tests/test_sampling.cu:150` | A3 | Passes `temperature=0.01`, never `0.0` — the actual `temperature == 0` branch is untested here. |
| `AttentionCrossPathTest.*` | `tests/test_attention_crosspath.cu:382+` | A4 | Golden spot values per config, strict `1e-2` on the f32 chain. Strong. |
| `FhmaMxFP4Test.*` | `tests/test_attention_fmha_mxfp4.cu:149+` | A4 | Mean/max error against an FP16 reference with stated limits. Strong. |
| `JsonConstrainPropertyTest.*` | `tests/test_json_constrain_property.cpp:198+` | A4 | 1 000 generated documents per case, cross-checked against `nlohmann::json::accept`. Strong. |
| `tests/api/test_contract.py` (all 43) | | A1 | Status codes and field presence only, against the mock. |

`M20`/`M21` in the mutant catalogue exist to settle the `top_p` claims above by
experiment rather than by reading.

## 5. Reproducibility of the suite itself

Running `test-e2e` as one process with all model env vars set produces **38
failures**; the same tests pass in isolation:

```
$ ./test-e2e                                  # 168 ran, 124 pass, 6 skip, 38 FAIL
imp_api.cpp:383: imp_context_create: KVCache: cudaMalloc failed for 28.00 MiB (out of memory)
mem_account: reserved 17856->14592 MiB used 14948->14948 MiB

$ ./test-e2e --gtest_filter='ChunkedPrefillTest.*'    # 7 ran, 7 PASSED
```

Cause is the documented WSL2/WDDM behaviour — a process never gets its peak
commitment back — compounded by the suite loading Qwen3-8B-Q8_0 (8.7 GB),
gpt-oss-20b-mxfp4 (12 GB) and gemma-4-26B-Q4_K_M (16.8 GB) in one process. It
is **not** a code defect, and none of the 38 is reported as a bug. It is a
property of the harness worth fixing, because a lane that is red for
environmental reasons trains readers to ignore red.

Evidence: `loop/evidence/P0-gpu-per-binary/test-e2e.log`.

## 6. Reproducing this inventory

```
loop/run_gpu_suite.sh                 # ctest -L gpu with every IMP_TEST_MODEL* set
loop/run_gpu_binaries.sh              # per-binary pass/skip/fail summary
docker run --rm -v $PWD:/src -w /src/build-dev imp:toolchain ctest -L unit
docker run --rm -v $PWD:/src -w /src imp:toolchain \
  python3 tools/mutation/classify_assertions.py /src
```

`make test-gpu` is **not** the right command: it sets none of the
`IMP_TEST_MODEL*` variables and mounts the repo's `models/` symlink directory,
which silently skips ~63 tests while exiting 0.
