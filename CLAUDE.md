# imp — Project Instructions

From-scratch C++23/CUDA LLM inference engine targeting **exactly one chip: NVIDIA Blackwell `sm_120a`** (RTX 5090 / GB202, 32 GB GDDR7, 1792 GB/s, native FP4 tensor cores). No portability layer, no FP16 dequant fallback in the hot path. ~100k LOC (src/ + include/). See [`docs/architecture.md`](docs/architecture.md) (canonical narrative) and [`docs/sm120.md`](docs/sm120.md).

## Where to start (task → entry point)

Match the task, invoke that skill first; the skills below are imp-specific and carry the detailed playbooks.

| Task | Start here |
|------|-----------|
| Build / run tests / CI red / dep bump | skill **building-and-testing** |
| Write/optimize a CUDA kernel (sm_120a) | skill **sm120-cuda-expert** |
| Benchmark, profile, refresh perf baseline | skill **benchmark-cuda** |
| Verify output coherence after hot-path change | skill **check-degeneration** |
| Quant formats / loaders / dequant (GGUF, NVFP4, FP8) | skill **quant-formats** |
| imp-server / OpenAI+Anthropic HTTP API | skill **server-api** |
| Add a new model architecture | skill **add-model-arch** |
| Open/merge a PR, cut a release | skill **shipping-prs** |
| Structure audit / dead code / god-files | skill **codebase-audit** |
| Keep docs in sync after a change | skill **docs-sync** |

Canonical references: `docs/architecture.md` (narrative), `docs/sm120.md` (hardware), `AGENTS.md` (subagent roles + guardrails), `docs/BENCHMARKING.md` (measurement contract).

## Build & test

**Before using the GPU, ALWAYS check that it is free.** Run any GPU job (`make test-gpu`, benchmarks, profiling, inference) only after confirming no other process is using the card: `docker ps -q | wc -l` MUST be `0` and `nvidia-smi` must show no active compute processes (idle ~30°C, low power draw). A busy GPU corrupts benchmark numbers and can OOM; never start GPU work on a card that is already in use.

The host has **no CUDA toolkit** by design — build inside Docker. `build/` is created root-owned by the container; remove it via a throwaway container, never `sudo` on the host.

```
make build         # Docker build (CUDA 13.3) (or: cmake --preset ci && cmake --build --preset ci on a CUDA 13.3+ host)
make test-unit     # CPU/host unit tests
make test-gpu      # GPU tests (requires RTX 5090)
make verify-fast   # ~90s pre-push gate
make verify        # ~5min full
make install-hooks # pre-push hook: runs verify-fast on src/include/tools/tests changes
make format        # clang-format
```

- Targets exist for `bench`, `test-perf`, `test-golden`, `test-vision`, `gen-perf-baseline`, `verify-north-star`, `verify-chunked`, `tidy` (clang-tidy, advisory), `sanitize`, and the roofline pipeline (`roofline-measure`, `roofline-pin`, `roofline-regress` — see `tools/roofline/README.md`).
- Configure presets live in `CMakePresets.json` (`default`/`ci`/`debug`/`relwithdebinfo`); dependency pins are single-sourced in `cmake/imp-deps.cmake` (the Dockerfile takes them as build-args). sm_120a is selected via raw gencode (CMake <3.31 workaround); 120f PTX fallback covers non-5090 SKUs.
- See also: [`AGENTS.md`](AGENTS.md) (subagent roles + guardrails) and [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) (measurement methodology contract).

## Conventions

- **English only in the repo.** All PRs (title + body), commit messages, code comments, docs, and `.md` files are written entirely in English. (Chat replies to the user stay German per global instructions — this rule covers artifacts that land in the repository or on GitHub.)
- **Always branch off `main` and `gh pr create --base main`.** Never stack PRs (squash-merge + stacking caused recovery-PR cascades). Prefer fewer, batched PRs over one-per-fix.
- **Performance is gated.** `tests/perf_baseline.json` is the canonical baseline (CI: 3% decode / 5% prefill thresholds). Refresh via `scripts/gen_perf_baseline.sh` when a change intentionally moves perf, and say so in the PR.
- Runtime config lives in `src/runtime/config.h` as `RuntimeConfig` (loaded from `imp.conf` + `--config` + `--set`; the only env vars still seeded are `IMP_DETERMINISTIC` and `IMP_FMHA_FA2` — the rest of the legacy `IMP_*` surface was retired 2026-07-07). Don't reintroduce ad-hoc env-var reads.
- Internal errors throw and are translated to `ImpError` at the `src/api/imp_api.cpp` boundary — this is intentional, don't convert those throws to status returns.
- Match surrounding code style; keep it simple and direct, no speculative abstraction.

## File Layout & Size

The real cost of an oversized file here is **recompile blast radius**, not line count.
Each `.cu` is one translation unit: touching a kernel in a 1.5k-LOC `.cu` re-`ptxas`es
the whole TU (the slowest sm_120a build stage) with no intra-file parallelism, and a
fat header re-triggers every includer. Optimize for compile-time isolation:

- **One logical unit per file** — one kernel concept, one module. If a `.cu` bundles
  several unrelated kernels, that's a split candidate.
- **Separate kernel definition / host launch-wrapper / explicit template
  instantiations.** When recompiles bite, move explicit instantiations into their own
  `.cu` so a kernel edit doesn't re-build every instantiation.
- **Line-count thresholds are a proxy/smell, not the metric.** The gate
  `tools/check_filesize.py` (config `tools/filesize_thresholds.toml`) measures *code*
  LOC (comments + blanks stripped) per category — kernel `.cu` warn>500/hard>600, normal
  TU warn>600/hard>800, header warn>500/hard>700. It runs in CI as the `File size` job
  (advisory warn step + blocking hard step); a hard-review violation fails CI.
- **A legitimately monolithic file is fine** — add it to `[allow]` in the toml **with a
  reason** (the gate rejects an empty reason). Don't split for splitting's sake. Current
  baseline + per-file rationale: `docs/audit/AUDIT_FILESIZE.md`.

## Benchmarking gotchas

- **The GPU is water-cooled and never warm** — do NOT insert temperature cooldown waits between benchmarks (it idles ~30°C and does not thermal-throttle). A decode drop across a sweep is a real regression or stale baseline, never heat.
- **The GPU downclocks at idle and takes ~1s for clocks to ramp up under load.** The first ~1s of any benchmark runs at reduced clocks and reads artificially LOW — this (not heat) is the dominant cold-start bench artifact (it produced a spurious −42% in one A/B that re-measured +20% clean). **Always warm the clocks before timed reps**: precede the measured run with a discarded warmup run (or busy-spin >1s); imp's built-in `Warmup...` is too few iterations to cover the 1s ramp. Never trust a cold single-shot number.
- cuBLAS prefill varies up to **2.6× across container restarts** and drifts under sustained load — use **decode** for A/B and follow the bench methodology (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, 10 reps, 3+ trials, one model per process / isolated).
- **Decode is stable within a session but can read 8–15% low for a whole day** (host/driver state on this WSL2 box; issue #526: identical builds measured tg 238 on one day, 278 the next — full methodology both times). Before trusting a cross-day decode delta or refreshing the baseline, sample `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw` DURING the bench: healthy load = ~2850 MHz SM / 13801 MHz mem / ~500 W. Lower mem clock or power = depressed host state, not a regression. GPU history: `gpu-stats` skill (7-day retention).
- nsys with CUDA Graphs ON hides captured kernels — profile with `--no-cuda-graphs` to see the true decode kernel mix.

## Hardware reality (sm_120 ≠ datacenter Blackwell)

- **No `tcgen05` / TMEM / wgmma / TMA-WS grouped GEMM.** FP4 path is `mma.sync` `mxf4nvf4` with FA2-style block-scaling. Ignore B200/`sm_100` (FlashAttention-4-style) kernel designs unless porting.
- **No FP4 cuBLASLt kernels on sm_120** → CUTLASS is the primary GEMM path. Dependency pins are single-sourced in `cmake/imp-deps.cmake` (the Dockerfile takes them as build-args via `scripts/dep_build_args.sh`; bump only that one file). FP8 prefill is unavailable on sm_120.
- For GGUF, weights are converted to an NVFP4 decode cache at init (bandwidth win on the decode hot path); prefill stays full-precision via source dequant. `nvfp4_beneficial` weights skip the FP16 cache.
- Docker build cache must **not** use `--mount=type=cache` (silently invalidates test results).

## After hot-path changes

Verify the model still produces coherent output (no repetition loops / token-stuck / state corruption) after touching the forward pass, MoE routing, KV cache, GDN state, or CUDA-graph capture.
