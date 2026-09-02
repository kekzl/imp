---
name: building-and-testing
description: Use when building imp, running its test suite, checking CI status, or debugging build/test failures - "make build", "run the tests", "test-gpu", "verify-fast", GTEST_FILTER, Docker/CUDA toolchain, dependency bumps, "CI is red/blocked", "which gate failed", stale objects / segfault after a header edit, hook edits, docker-entrypoint env vars, determinism or perplexity checks. Do NOT use for benchmarking/profiling (benchmark-cuda) or output-quality batteries (check-degeneration).
---

# Build & Test - imp

## Hard rules

| # | Rule | Detail |
|---|---|---|
| 1 | No CUDA toolkit on the host | Everything runs in Docker. Toolchain `nvidia/cuda:13.3.1-devel-ubuntu26.04`, GCC 15.2, C++23 for host AND device (`CMAKE_CUDA23_STANDARD_COMPILE_OPTION` shim at the top of `CMakeLists.txt`). |
| 2 | `build/` and `build-dev/` are root-owned | `make dev-clean`, or `docker run --rm -v $PWD:/src -w /src ubuntu rm -rf build`. Never `sudo`. |
| 3 | No `--mount=type=cache` in the Dockerfile | Silently invalidates test results. |
| 4 | `models/` in the repo is a symlink farm | Custom `docker run` mounts `$HOME/models:/models`; Makefile targets already do. |
| 5 | Dependency pins live ONLY in `cmake/imp-deps.cmake` | CUTLASS, GTest, httplib, nlohmann/json. `make build` injects them via `scripts/dep_build_args.sh`. `Lint` runs `scripts/check_dep_pins.sh --online` (blocking inside Lint, Lint itself advisory). |
| 6 | CI has no GPU runner | `Test` job skipped unless repo var `HAS_GPU_RUNNER=true`. GPU correctness and perf are gated LOCALLY: pre-commit (`make test-gpu`), pre-push (`make verify-fast`). |
| 7 | Required check = `Build`, ruleset 14716423 | Static gates run as its FIRST step (`scripts/ci_static_gates.sh`, unfiltered) and block the merge since #1527. Rename the job without the ruleset and every PR sits at `mergeStateStatus=BLOCKED`. |
| 8 | `main`'s CI status is stale by default | Auto-merge squashes with `GITHUB_TOKEN`, which starts no workflow run. Judge a fix by the PR run or the file; refresh with `gh workflow run CI --ref main`. |

## Command table

| Task | Command | Time |
|---|---|---|
| Incremental build (inner loop) | `make dev` | 2-14 s |
| Incremental build + CI unit lane | `make dev-test` (= `ctest -L unit`) | ~3 s |
| Full image (the gate; anything you measure or push) | `make build` -> `imp:test` | ~3.5 min regardless of diff |
| CPU unit binary | `make test-unit` (`imp-tests-unit`) | <5 s |
| Full GPU suite | `make test-gpu` | 4-10 min |
| E2E on real models | `make test-e2e` | Qwen3-4B-Instruct-2507-Q8_0, Qwen3.5-4B-mxfp4, gemma-4-26B-A4B Q4_K_M, `MOE_MODEL` (gpt-oss-20b-mxfp4), Qwen3-Coder-30B FP4, Nemotron-3-Nano NVFP4 (paths in `Makefile`) |
| Server e2e (the ONLY place handlers/batching run end to end) | `make test-server` | includes `degen_suite.py` and `tests/test_server_tracing.py` |
| Rerank / agents / external harnesses | `make test-rerank`, `make test-agents`, `make test-agents-external` | GPU + local model |
| NIAH retrieval gate, spec fidelity | `make test-niah`, `make test-spec-fidelity` | spec-fidelity needs ~26 GB free |
| Vision goldens | `make test-vision` (`IMP_VISION_GOLDEN_DUMP=1` regenerates) | |
| Chat-template goldens | `make chat-goldens` (network, writes `tests/refs/chat_template_goldens.h`) | |
| Pre-push gate | `make verify-fast` | 18 s without perf gate, 36 s with |
| Full / chunked / north-star gates | `make verify`, `make verify-chunked`, `make verify-north-star` | ~5 min full |
| Kernel register/spill ratchet (CI `kernels` gate) | `make kernel-resources`, re-pin `make kernel-resources-update` | needs `libimp.a` |
| clang-format | `make format-check`; never bare `make format` | repo is not format-clean |
| Host ASan/UBSan over test-core/test-text | `make asan` | works on WSL2 |
| compute-sanitizer | `make sanitize` | does NOT work on WSL2 (WDDM) |
| Alloc-interpose census | `make check-alloc-interpose` (`build-interpose/`) | never benchmark it (reads ~3% low) |
| Server gcov | `make coverage` | GPU + model |
| API mock contract | `pytest tests/api/` or `tests/api/run_mock_tests.sh` | no GPU; the mock lane also runs the nomodel tests |
| GPU free? | `make check-gpu` (`scripts/require_free_gpu.sh`: utilisation AND memory) | |

Filtered GPU run (GTEST_FILTER is not threaded through `make test-gpu`):

```bash
docker run --rm --gpus all -v $HOME/models:/models \
  -e IMP_TEST_MODEL=/models/Qwen3-8B-Q8_0.gguf \
  imp:test imp-tests --gtest_filter="DegenerationTest.*"
```

`DetEvalE2ETest.*` matches 0 tests and prints PASSED; the working filter is `*DetEvalE2ETest*`.

## Binaries

| Where | What |
|---|---|
| `imp:test` image | `imp-server`, `imp-cli`, `imp-bench`, `imp-quantize`, `imp-tests` (full GPU), `imp-tests-unit` (generated wrapper = `ctest -L unit`: test-core + test-text + e2e unit filter), modules `test-core test-text test-compute test-attention test-quant test-kv test-moe-gdn test-e2e` |
| `build-dev/` (host, from `make dev`) | same binaries; run them inside `imp:toolchain` with `-v $PWD:/src -w /src` (that mount hides the image's own `/src`) |

`tests/test_ssm.cu` -> `test-moe-gdn`. `SamplingTest` -> `test-compute`. A new CPU test belongs in `test-core`. Lane census: `python3 tools/check_test_lanes.py --report` (the `Test lanes` check pins the no-lane count; a new GPU test in no lane fails it, raise `PINNED` with a reason). The unit lane has no skips (`guard_unit_skips` in `ctest -L unit` fails on any): a test that needs a GPU or a model file goes to `test-kv` or to `test-e2e` outside `_unit_e2e_filter`; fixture files live in `tests/fixtures/` (`IMP_TEST_FIXTURES_DIR`). A new tool binary needs a `cp` in the builder stage AND a `COPY --from=builder` line in the Dockerfile.

## Gates and hooks

| Gate group | Runs where | Content |
|---|---|---|
| `filesize` | Build, hooks | `check_filesize.py` (two-way `[allow]` ceiling), `check_determinism_sites.py`, `check_dead_inline_accessors.py`, `check_log_fatal.py` |
| `lanes` | Build, hooks, own check `Test lanes` (#1770) | `check_test_lanes.py --report` (macros per lane); `guard_unit_skips` inside `ctest -L unit` fails on any runtime skip |
| `entrypoint` | Build, hooks | `tests/test_entrypoint.sh` drives `docker-entrypoint.sh` against a stub (25 assertions) |
| `alloc` | Build, hooks | `check_alloc_sites.py` + `check_alloc_pairs.py` (allowlist is two-way: removing a site needs the allowlist edit) |
| `kernels` | Build only (needs artifact) | `make kernel-resources` vs `tools/kernel_resource_baseline.txt` (REG >= 240 or non-zero local frame) |
| `launchguards` | Build, hooks | `check_launch_guards.py` |
| `docs` | Build, hooks, `Docs` job | `sync_docs.py --check`, `docs_lint.py` |
| `citations` | Build, hooks | `check_doc_citations.py` over all living docs (#1783) |
| `hygiene` | Build, `Release hygiene` job only | `check-release.sh` with `SKIP_VERIFY=1` (also resolves relative links in docs) |

Advisory jobs: `Lint`, `clang-tidy`, `Mock API contract`, `Real API contract (model-less)`, `Sanitizers`, `PTX fallback`.

- Hooks: `make install-hooks` copies `scripts/pre-commit.hook` and `scripts/pre-push.hook`. Never edit `.git/hooks/*`: `guard_precommit_filter` in the CPU lane diffs them against the tracked files.
- Pre-commit = static gates (~2 s) then `make test-gpu` (full suite, ~10 min) when staged `src/ include/ tools/ tests/ CMakeLists cmake/` files change; `.md .py .hook` skip. A 2-min timeout around `git commit` kills the suite and leaves files staged only. Alternative: `make test-gpu` by hand, read exit 0, then `git commit --no-verify`.
- Pre-push = static gates, `require_free_gpu.sh`, then `make verify-fast`; the perf gate runs only when the diff matches `PERF_RE` (`src/{compute,exec,quant,runtime,model}/`, any `.cu/.cuh`, `cmake/`, `CMakeLists.txt`, `tests/perf_baseline*`). Skips: `IMP_VERIFY_SKIP_PERF/_VRAM/_GRAPHS=1`.
- Hooks run `filesize lanes entrypoint alloc launchguards docs citations`; `hygiene` only in CI. Local hygiene check: `docker run --rm -v $PWD:/src -w /src -e HOME=/tmp imp:toolchain bash -c 'git config --global --add safe.directory /src; bash scripts/ci_static_gates.sh hygiene'`.
- `docs_lint.py` regenerates `docs/audit/docs-rewrite/STALE.md` on every run; it blocks `git pull` until `git checkout -- docs/audit/docs-rewrite/STALE.md` or committed as an `.md`-only follow-up.

## Traps

| Symptom | Cause | Fix |
|---|---|---|
| Segfault / `cudaFree` on garbage / `MoEExecutorTest` 5/5 red after a header or class-layout edit | stale `.o` despite ninja | `grep -rl exec/executor.h src tools tests \| xargs touch && make dev` (healed 3/3) |
| Link error `__cudaRegisterLinkedBinary` after `make dev` | same class | touch includers, rebuild |
| Gate result contradicts the tree | `make build` COPYs the tree at start (#1531); `verify-fast` never builds (`SKIP build` line) | rebuild after every edit or amend, then gate |
| Bench numbers move while a chain runs | `make dev` rebuilds `build-dev/` under the running binary | bench from copies (`-v dir:/bin_arm`) |
| `NvFP4SmallMTest.BandwidthAboveStarvationFloor` red in the full suite | contention in the run (401 vs 537.6 GB/s threshold; isolated 590-622) | rerun isolated |
| `test-core` fails to link in `Sanitizers` / `make asan` | sources inside the `IMP_BUILD_SERVER` block (#1821) | keep test-core's deps outside that block |
| `--perplexity` OOMs or pool floors on a head checkpoint | `mtp_k=auto` loads the MTP head (+0.79 GiB) | `--set speculative.mtp_k=0` in PPL harnesses |
| PPL differs 0.35% between runs of the SAME binary | non-deterministic forward | `--set runtime.deterministic=true` on both arms (implies `deterministic_gemm`) |
| `std::sort` etc. "not declared" | libstdc++ 15 dropped transitive `<algorithm>` (#906) | add the include |
| nvcc arch error | target is `compute_120a/sm_120a` + optional `compute_120f` PTX (`IMP_SM120_FLAGS`) | never generic `sm_120` (sm120-cuda-expert) |
| Compose/container ignores a config key | only `IMP_CONFIG` / `IMP_SET` reach every key (#1823); the 19 legacy `IMP_*` names in `docker-entrypoint.sh` are frozen | `-e IMP_SET="key=v key2=v"`; a bogus key fails at start (`no such key`) |
| Value looks like an unread env var | engine reads only `IMP_CONFIG IMP_DETERMINISTIC IMP_FMHA_FA2 IMP_WORKER_TIMING IMP_SPEC_TRACE IMP_JUMP_TRACE IMP_PPL_DUMP`; the entrypoint translates the rest | grep `docker-entrypoint.sh` too |
| `for x in $VAR` in a script produces filenames | unquoted expansion globs | `set -f` / `set +f` around the loop |
| `git worktree remove` fails | root-owned `build-dev/` inside it | `rm -rf` via container, `git worktree prune`; a worktree also needs the `models/` symlink dir or verify-fast fails in the hook |
| A gitignored `./imp.conf` in the repo root | dev builds with `-w /src` load it, images do not (#1784 fixed the build context) | read the `imp.conf loaded from` line in every log |

## Determinism and quality

- `--set runtime.deterministic=true`: temp=0 reproducibility incl. MoE routing atomics and top-k races; implies `runtime.deterministic_gemm`; promoted process-wide since #542; default OFF. Qwen3.6-35B stays non-deterministic at temp=0 (routing flips); never assert exact output there.
- 5 `DISABLED_` tests with reasons: 2 determinism boundaries (cross-context GDN, FMHA smem), the rest benches. Do not re-enable.
- `imp-cli --perplexity <file>` is teacher-forced prefill (chunk-aware since #553); use `tools/analysis/ppl_corpus_45k.txt`, not the 199-token `ppl_corpus.txt`.
- `imp-cli --json`: one JSON document on stdout, logs on stderr (#1715).

## PR / merge

Branch off `origin/main`, `gh pr create --base main`, never stack. Perf-moving change: refresh `tests/perf_baseline.json` in the same PR (benchmark-cuda). Ship flow, auto-merge race, release cut: skill **shipping-prs**.
