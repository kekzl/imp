<!--
layer: L2
audience: kernel-devs
verified: 2026-08-28
commit: be825e4a
-->

# Benchmarking methodology

The one-page contract for how imp performance is measured and gated. Numbers that do not follow this are not comparable. (Narrative gotchas: `CLAUDE.md`; this is the auditable summary.)

## Headline metric

- **Decode `tg128` (tokens/s)** is the headline and the A/B signal. Stable within a session.
- **Prefill `pp512`** is reported but **not** used for A/B on its own: it varies far more than decode across process starts, and by model. Measured on one quiet host, three fresh processes per arm: **0.6-1.2 %** on the cuBLAS-FP16 prefill model (Qwen3-8B Q8_0), **37.6 %** on a fully resident NVFP4 MoE model (Qwen3-Coder-30B-A3B). cuBLAS algo re-timing, long blamed for this, measured **3.50 %** on its own over nine process starts. Treat prefill regressions as warnings; resolve them with a paired, alternating A/B rather than more reps.

## How to measure (every run)

1. **GPU must be free**: `docker ps -q | wc -l` == 0 and `nvidia-smi` shows no compute process. A busy GPU corrupts numbers and can OOM. (`make check-gpu`.)
2. **Warm the clocks**: the GPU downclocks at idle and the first ~1 s reads low. One discarded warmup run before the measured run (imp's built-in `Warmup…` is too few iterations to cover the 1 s ramp).
3. **Single session only**: compiler/cuBLAS autotuning makes cross-session and cross-day numbers unreliable. Only compare results captured **within one run**. Decode can read 8-15 % low for a whole day (host/driver state on the WSL2 box).
4. **Reps & isolation**: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, ≥3 reps (A/B claims: 3+ trials), one model per process.
5. **No cooldown waits**: the GPU is water-cooled, idles ~30 °C, never throttles. A decode drop across a sweep is a real regression or stale baseline, never heat.
6. **Sample clocks during the run** to rule out a depressed host state: `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw`. Healthy load ≈ 2850 MHz SM / 13801 MHz mem / ~500 W. Lower mem clock or power = depressed host state, not a regression.
7. **Check the process is VRAM-resident** before reading a large decode collapse as a code regression. At ~0 MiB free, WSL2/WDDM oversubscribes into host memory and keeps returning `cudaSuccess`; bandwidth falls off a cliff (~1530 GB/s resident vs ~237 GB/s spilled). That is #1103: 55 tok/s at server defaults on a model that benches far higher. `--mem-report` prints the free-VRAM figure; a successful allocation is not evidence.

## Context-dependent changes need their own A/B

The gate measures `tg128` at **`pp512`**; a change whose effect depends on context length is invisible there **by construction**. #1270 shipped a split-count heuristic that gained +10.0% at 32k on Qwen3-8B-Q8_0, cost **−7.30% at 32k on Qwen3-30B-A3B-NVFP4**, and passed `verify-fast` at +0.33% because the boost is inactive at pp512. Reverted in #1271.

```bash
scripts/bench_longctx_ab.sh <A> <B> [ctx-list] [model-list]
```

A and B are docker images or build directories. Two rules it encodes:

- **Two models minimum.** Six context lengths on one checkpoint gave a clean monotone curve with sub-0.5% spreads and was still wrong. The default pair has deliberately different GQA shapes (`n_kv_heads` 8/`g` 4 vs 4/8); that is what caught #1270. Precision is not coverage.
- **Spec-OFF.** n-gram speculation puts **14-17%** spread on short-context points, enough to hide a 1% effect and to make a −11% median look real. The script passes `speculative.ngram=false`, matching the gate.

The script refuses to run on a busy GPU, alternates the arms so host drift hits both equally, and marks any delta smaller than the larger spread as `(within spread)` rather than reporting it as a result.

## The gate

- Canonical baseline: **`tests/perf_baseline.json`**, thresholds **8 % decode / 8 % prefill** and **10 % peak VRAM**. One file, **two scripts that are not interchangeable** (#1625):

| | `scripts/verify.sh` | `scripts/bench_gate.sh` |
|---|---|---|
| driven by | `make verify`, `make verify-fast`, the pre-push hook | the `Test` CI job (self-hosted GPU runner) |
| trials | 3 independent processes, median | 1 process, 3 internal reps |
| VRAM gate | yes | no |
| speculation | off | off since #1625; **was on**, so it measured a quantity the pin does not describe |

  Both pass `--set speculative.ngram=false` now, which `tests/perf_baseline.json` states in its own `methodology` field. Until #1625 only `verify.sh` did, while the two were documented as one gate.
- A decode delta worse than −8 % **fails**; a prefill delta worse than −8 % warns (cuBLAS variance).
- **When it runs.** The pre-push hook runs the perf gate only when the diff can move it: `src/{compute,exec,quant,runtime,model}/`, any `.cu`/`.cuh`, the build definition, or a baseline file. A push outside those paths keeps the correctness half (tests, peak VRAM, graphs ON/OFF, degeneration smoke) and skips the three benched processes, half the wall clock (36 s → 18 s). `scripts/check-release.sh` always runs everything, so a release is gated on the full set. Of the 40 commits between v0.24.0 and v0.25.0, 13 touch a measured path and 27 do not.
- **Why 8 % and not 3 %.** The threshold has to sit above this host's own movement, or it reports the box instead of the diff. Within one session the gate is tight (three independent processes agree to 0.16 %), but *between* sessions the same tree reads 287.63 one day and 276.92 the next (−3.58 %) at healthy clocks, and six quiet runs spanned 278.59…289.77. Ordinary desktop use (a stream, a browser) costs a few percent more; a depressed-host day costs 8-15 %. The old 3 % sat below all of that and failed on docs-only changes. What the gate still catches: the split-K mutation M29 measured **−36 %**, a 4.5x margin. A red gate has never been a regression on its own; the proof is a paired A/B against `main`, alternating the arms.
- **Peak VRAM is gated too** (`scripts/verify.sh`, both `verify` and `verify-fast`): a `--mem-report` run vs the pinned `metrics.memory_mb.own_peak_mb` against `thresholds.vram_increase_pct`. It gates `own_peak` (this process's allocations since engine init), **not** device `peak_used`, which also carries the CUDA primary context and any neighbour process. `own_peak` measures byte-identical across repeat runs, a stricter signal than any throughput number. (Skip with `IMP_VERIFY_SKIP_VRAM=1`.)
- **Intentional perf moves:** refresh the baseline with `scripts/gen_perf_baseline.sh` (cold-median: 5 trials, median per metric; re-pins `own_peak_mb` in the same run) and **say so in the PR**. A change that intentionally moves VRAM needs the same refresh.

## Profiling builds

- Use the `relwithdebinfo` preset (Release optimizer + `-lineinfo`) for Nsight.
- nsys with CUDA Graphs ON hides captured kernels; profile with `--no-cuda-graphs` to see the true decode kernel mix.
