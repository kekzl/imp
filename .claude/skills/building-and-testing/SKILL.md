---
name: building-and-testing
description: Use when building imp, running its test suite, checking CI status, or debugging build/test failures — "make build", "run the tests", "test-gpu", "verify-fast", GTEST_FILTER, Docker/CUDA toolchain, dependency bumps, "CI is red/blocked", determinism or perplexity checks. Do NOT use for benchmarking/profiling (benchmark-cuda) or output-quality batteries (check-degeneration).
---

# Build & Test — imp

## Hard rules (violations cost hours)

1. **The host has NO CUDA toolkit by design.** All build/run/test happens inside Docker (`make build` → image `imp:test`, CUDA 13.3 on Ubuntu 26.04 / GCC 15.2, C++23 for host AND device — `CMAKE_CUDA_STANDARD 23` needs the shim at `CMakeLists.txt:10`). Never apt-install toolchains on the host.
2. **`build/` is root-owned** (created by the container). Remove via throwaway container, never `sudo` on the host: `docker run --rm -v $PWD:/src -w /src ubuntu rm -rf build`.
3. **Never use `--mount=type=cache`** in the Dockerfile — it silently invalidates test results.
4. **`models/` in the repo is a symlink farm** to `$HOME/models`. Most Makefile targets mount `$(PWD)/models`, which works because Docker resolves on access — but for custom `docker run`, mount `$HOME/models:/models` directly so symlink targets resolve.
5. **Dependency pins are single-sourced in `cmake/imp-deps.cmake`** (current: CUTLASS v4.5.2, GTest v1.17.0, nlohmann/json v3.12.0, httplib v0.48.0). `make build` extracts them and injects Docker `--build-arg`s via `scripts/dep_build_args.sh` — bump ONLY that one file; never re-pin in the Dockerfile or CMakeLists.
6. **CI has no GPU runner.** The `Test` job is auto-skipped until repo var `HAS_GPU_RUNNER=true` — GPU correctness/perf validation is LOCAL-ONLY (`make verify-fast` before push; `make install-hooks` installs the pre-push hook). CI jobs: **`Build`** (compile + `ctest -L unit`, the only REQUIRED check — renaming it without updating ruleset "Require CI" id 14716423 leaves PRs stuck at `mergeState=BLOCKED`), `clang-tidy` (advisory), `Mock API contract`, `Lint`, `File size` (`tools/check_filesize.py` — the hard-threshold step BLOCKS; see `codebase-audit`).

## Command table

| Task | Command | Time |
|---|---|---|
| Build Docker image (tests on) | `make build` | varies |
| CPU/host unit tests | `make test-unit` | <5 s |
| Full GPU suite | `make test-gpu` | ~4–5 min (`test-attention` alone ~241 s) |
| E2E model tests (real models) | `make test-e2e` | needs Qwen3-4B + Qwen3.5-4B + Gemma-4 GGUFs |
| Vision goldens | `make test-vision` | `IMP_VISION_GOLDEN_DUMP=1` to regenerate |
| Pre-push gate | `make verify-fast` | ~90 s; build + filtered tests + perf gate + smoke |
| Full pre-merge gate | `make verify` | ~5 min |
| Chunked-prefill gate | `make verify-chunked` | vs `perf_baseline_chunked.json`, 5%/8% |
| North-star gate | `make verify-north-star` | Qwen3-14B Q6_K vs its own baseline |
| clang-format (in container) | `make format` / `make format-check` | repo is NOT fully format-clean — never format whole files you didn't touch |
| API mock contract suite | `pytest tests/api/` (see `tests/api/run_mock_tests.sh`) | no GPU needed |

Filtered run with explicit model:

```bash
docker run --rm --gpus all -v $HOME/models:/models \
  -e IMP_TEST_MODEL=/models/Qwen3-8B-Q8_0.gguf \
  imp:test imp-tests --gtest_filter="DegenerationTest.*"
```

Test binaries in the image: `imp-tests` (full GPU), `imp-tests-unit` (CPU), plus split binaries `test-core test-text test-compute test-attention test-quant test-kv test-moe-gdn test-e2e test-gdn`. Gotcha: `test_ssm.cpp` tests live in **`test-moe-gdn`**, not a binary of their own.

**`make test-unit` is NOT the CI lane.** It runs `imp-tests-unit` (~37 tests); CI runs `ctest -L unit` → **`test-core`** (550+) + test-text + an e2e subset. A new CPU test belongs in `test-core`, and the honest no-GPU check is `docker run --rm imp:test test-core` (no `--gpus`). Green in `imp-tests-unit` says nothing about CI.

Tool binaries in the image: `imp-server`, `imp-cli`, `imp-bench`, and `imp-quantize` (offline BF16/FP16 → NVFP4 conversion, experimental — see `quant-formats`). A new tool needs BOTH a `cp` in the builder stage and a `COPY --from=builder` line in the Dockerfile, or it silently isn't in the image.

## Determinism & quality caveats

- `--set runtime.deterministic=true` gives full temp=0 reproducibility (covers MoE routing atomics + top-k sampling races + implies `deterministic_gemm`). Default OFF — costs throughput. The engine promotes it process-wide since PR #542.
- 3 `DISABLED_` tests mark known determinism boundaries (cross-context GDN, FMHA smem) — do not "fix" them by re-enabling.
- Qwen3.6-35B is non-deterministic at temp=0 even with full methodology — never assert exact output for it.
- Perplexity: `imp-cli --perplexity <textfile>` (teacher-forced, chunk-aware since PR #553 — PPL absolutes from before 2026-06-06 on corpora >2k tokens were wrong; same-corpus A/B deltas remain valid).
- `imp-cli` logs to **stdout** — strip log lines before hashing output.
- `make sanitize` (compute-sanitizer) does NOT work on WSL2 (WDDM, no debugger interface) — native-Linux hosts only.

## When the build fails

- nvcc/arch errors: target is raw gencode `compute_120a/sm_120a` + optional `compute_120f` PTX fallback (`CMakeLists.txt` ~line 31). Don't "fix" by switching to generic `sm_120`/`compute_120` — see `sm120-cuda-expert`.
- `std::sort/find/max_element` etc. "not declared": libstdc++ 15 no longer includes `<algorithm>` transitively (#906) — add the missing include, don't downgrade the toolchain.
- FetchContent mismatch vs Docker deps-clone → see hard rule 5 (bump `cmake/imp-deps.cmake` only).
- Out-of-space: Docker images are large; prune old `imp:*` images first.

## PR / merge conventions

Branch off `main`, `gh pr create --base main`, never stack PRs, batch related fixes. CI green (`Build`, no GPU runner) + auto-merge; GPU validation stays your job locally (`make verify-fast`). If a change intentionally moves perf, refresh `tests/perf_baseline.json` (see `benchmark-cuda` → Publishing numbers) and say so in the PR. For the full ship/merge/release flow — the **auto-merge race** (pushing after arming auto-merge can drop your last commit), the `Build`-ruleset `BLOCKED` gotcha, and version-bump + CHANGELOG + tag steps — use **`shipping-prs`**.
