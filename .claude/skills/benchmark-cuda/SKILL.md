---
name: benchmark-cuda
description: Use when benchmarking, profiling, or A/B-testing CUDA kernels or end-to-end perf in the imp inference engine on RTX 5090 (sm_120), including refreshing tests/perf_baseline.json or publishing numbers to docs/BENCHMARKS.md and the README. Triggers on "benchmark kernel", "profile cuda", "ncu", "nsys", "kernel timing", "occupancy", "bandwidth bound", "compute bound", "roofline", "perf baseline", "is this regression real", "decode dropped". Do NOT use for writing/optimizing kernel code (sm120-cuda-expert) or output-quality checks (check-degeneration).
---

# CUDA Kernel Benchmarking — imp / sm_120 / RTX 5090

Pair with `sm120-cuda-expert` for optimization decisions.

## STOP — what is a real signal on this box

1. **Decode is the only reliable A/B signal.** Prefill (`pp512`) spread across process starts is a property of the MODEL, not of cuBLAS: measured on one quiet host, 3 fresh processes per arm, **0.6-1.2 % on Qwen3-8B Q8_0** (the cuBLAS-FP16 prefill model) against **37.6 % on a fully resident NVFP4 MoE model**. cuBLAS algo re-selection on its own measures 3.50 % over nine starts. (The "2.6×" this line used to quote was a carried-forward citation, retracted 2026-08-03.) Never gate on prefill alone; resolve it with a paired, alternating A/B, not with more reps. For prefill-*kernel* A/Bs ≤5%, end-to-end pp cannot resolve the delta at all — compare **nsys per-kernel time sums** instead. Metric keys: the gate uses **`tg128`**, ad-hoc runs usually print `tg256` — compare like with like.
2. **The GPU is water-cooled and never throttles** (idles ~30 °C). Do NOT add temperature cooldowns. The 15 s cooldown in `gen_perf_baseline.sh` resets **cuBLAS algo state**, not temperature.
3. **Idle downclock is the dominant cold-start artifact.** Clocks need ~1 s to ramp, so the first second reads LOW — once producing a spurious −42% that re-measured +20% clean. Precede timed reps with a discarded warmup >1 s; imp's built-in `Warmup...` is too short.
4. **Decode can read 8–15% low for a whole day** (host/driver state on this WSL2 box, issue #526). Sample clocks DURING the bench: healthy load ≈ **2850 MHz SM / 13801 MHz mem / ~500 W**. Lower mem clock or power = depressed host → don't trust cross-day deltas or refresh baselines that day.
5. **Back-to-back sweeps read 6–10% low** vs isolated runs. One model per process.
6. **n-gram speculation is default-ON for dense models, and `--bench` prompts are self-repetitive** (~99.9% draft accept), so a dense `--bench` mostly measures the batched *verify* path. For decode-kernel/GEMV A/Bs pass `--set speculative.ngram=false` **to both arms** — a real +2–3% GEMV win is otherwise invisible.
7. **A number is only comparable to a run with the same flags.** Before calling a delta a regression, check what the measurement changed: prefix caching (`--bench` disables it since #1061; older pins measured cache hits — `--set server.prefix_cache=true` restores the old behaviour), context (`tg128_at_ctx_2048` needs `--bench-pp 2048`; pp16/pp512 read ~+13%/+10% on the same build and look like north-star gains), and trial count (`verify-fast` runs fewer than `gen-perf-baseline`, so it reads lower).
8. **No-graphs profiles OVERSTATE tiny-kernel classes — validate any launch/latency-class lever with a graphs-ON e2e A/B BEFORE building it.** The roofline pipeline profiles with `--no-cuda-graphs`, but under the shipped graphs+PDL loop those launches overlap away (the no-graphs kernel-time sum is ~1.8× the real step). Two refutations: a fused gate-GEMV+top-k kernel with bit-identical output and 2 fewer launches/layer moved e2e **0%**; capping decode split-K REGRESSED −21…−35%. A decode lever must hold real bytes or critical-path math — grid-(1,1,1) classes (moe_routing, rmsnorm, rope, kv_write, elementwise) are not levers.

## Methodology (every A/B)

`CUBLAS_WORKSPACE_CONFIG=:4096:8` · 10 reps · 3+ trials · one model per process · `make check-gpu` first (no concurrent GPU consumers) · warm clocks >1 s before timing.

**Before any GPU job: `docker ps -q | wc -l` must be 0.** Detached/`nohup` containers survive session ends and silently depress every number (cost a v0.10.0 re-bench). `make check-gpu` helps but doesn't see a container that isn't currently on the GPU.

The host has **no CUDA toolkit** — all binaries run inside Docker (`imp:test`, models mounted from `$HOME/models`, NOT the repo's `models/` symlinks). `imp-cli` has no `--ctx` flag — the context ceiling is `--max-seq-len`.

## Pick the right tool

| Goal | Tool | Notes |
|------|------|-------|
| End-to-end engine perf | `make bench` | `imp-cli --bench --bench-pp 512 --bench-reps 5` sweep across baseline models |
| Single model quick check | `make test-perf` | Qwen3-8B Q8_0 only |
| Per-config sweep MBU/MFU/TTFT/TBT | `bench/bench.py` | CSV output, optional llama.cpp compare |
| Refresh perf baseline | `make gen-perf-baseline [MODEL=/models/…]` | cold-median: 5 trials × 15 s cooldown; writes `tests/perf_baseline.json`, including the `own_peak_mb` VRAM pin (extra `--mem-report` run) |
| Regression gate | `make verify-fast` | 8% decode / 8% prefill / 10% peak VRAM (`own_peak`) |
| VRAM attribution for one run | `imp-cli --mem-report` | lifecycle checkpoints, per-pool notes, named charges (context / library reserve / engine arena), `own_peak` vs any `--vram-budget`, residual; the gate parses `own_peak=` from it |
| North-star gate (Qwen3-14B Q6_K) | `make verify-north-star` | vs `tests/perf_baseline_north_star.json` |
| Single kernel — wall-clock A/B | `cudaEvent` in launcher | see Step 1 |
| Single kernel — metrics, stalls | `ncu` | see Step 2 |
| Timeline / launch overhead / graphs | `nsys` | see Step 3 |
| Full roofline sweep (ncu+nsys pipeline) | `make roofline-measure` | `tools/roofline/` (see its README); classifies kernels, attributes nsys time shares |
| Pin roofline run as regression baseline | `make roofline-pin` / `roofline-regress` | baseline ref in `tools/roofline/history/BASELINE` |
| Compare imp vs llama.cpp | `bench/profile.sh` | same models, apples-to-apples |

Known phantom: **gemma-3-12b `--bench` prints bogus tok/s** (issue #514 reopened) — for that model trust perplexity only, never its bench numbers.

## Step 1: cudaEvent in-code (quick A/B)

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);

// Warmup — >=3 iterations AND >1s total busy time (clock ramp, see STOP #3)
for (int i = 0; i < 3; i++) kernel<<<...>>>(...);
cudaDeviceSynchronize();

cudaEventRecord(start);
for (int i = 0; i < N_ITER; i++) kernel<<<...>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
float avg_us = (ms / N_ITER) * 1000.0f;
```

Rules: N_ITER ≥100 for kernels <100 µs, check stddev not just mean, kill concurrent GPU consumers, sample clocks during the run (STOP #4).

## Step 2: Nsight Compute (ncu) — per-kernel metrics

ncu is NOT in the runtime image. Use the **host** install mounted into the container, and call the real binary (not the cuda symlink wrapper):

```bash
docker run --rm --gpus all -v $HOME/models:/models \
  -v /opt/nvidia/nsight-compute/2026.2.0:/ncu -v /tmp/out:/out --user root \
  imp:test /ncu/ncu --kernel-name "regex:my_kernel.*" --launch-skip 3 --launch-count 10 \
  -o /out/profile ./build/imp-bench …   # chmod 777 /tmp/out first
```

Canonical metric set: `./.claude/skills/benchmark-cuda/ncu-basic.sh "<kernel-regex>" <binary> [args]`. Key metrics:

| Metric | Meaning | Target |
|--------|---------|--------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization | >70% compute-bound |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth | >70% memory-bound |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy | context-dependent |
| `smsp__inst_executed_pipe_tensor_op_*` | TC activity | non-zero if TC kernel |
| `l1tex__t_sector_hit_rate` | L1 hit rate | >90% for cached |
| `stall_*` | Where warps stall | lowest = bottleneck |

Always `--launch-skip 3 --launch-count N`. Compile with `-lineinfo` for source-correlated stalls (`--set detailed --import-source yes`).

## Step 3: Nsight Systems (nsys) — timeline

When you suspect: launch overhead, H2D/D2H stalls, stream serialization, CUDA Graph behavior.

**WSL2 needs sampling disabled** or nsys hangs/errors:

```bash
nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda,nvtx \
    --stats=true --cuda-memory-usage=true -o timeline --force-overwrite=true \
    ./build/imp-bench …
nsys stats timeline.nsys-rep
```

**CUDA Graphs hide captured kernels** — profile with imp's `--no-cuda-graphs` flag to see the true decode kernel mix. If you must profile WITH graphs ON (e.g. capture-only paths like graph-captured spec verify), add `--cuda-graph-trace=node` so nsys attributes per-kernel times inside the replayed graph (PR #856 lesson).

Red flags: gaps between launches >10 µs (CPU-bound) · H2D/D2H during compute without overlap · graph not collapsing launches (silent fallback — see `check-degeneration`).

**compute-sanitizer does NOT work on WSL2** (WDDM exposes no debugger interface). `make sanitize` is documented for native-Linux hosts only.

## Step 4: Roofline (one-liner)

`AI = total_flops / total_bytes_moved` (matmul FLOPs = `2·M·N·K`; bytes from `dram__bytes.sum` in ncu). Peaks: HBM 1,792 GB/s · FP16 838 TFLOPS · FP8 1,677 · FP4 3,354 TOPS (datasheet) · L2 96 MB. **Calibrated reality (2026-06-07): FP4 `mma.sync` reaches ≈2,019 TOPS (~½ datasheet), f32-accumulate ¼ rate** — use the measured peak for "% of roofline" claims or every FP4 kernel looks falsely bad. Ridge points (datasheet): FP16=468, FP8=936, FP4=1873 FLOP/byte. AI < ridge → memory-bound. For full sweeps use `make roofline-measure` instead of hand math.

## Report template

```
Kernel: <name>, config: <block=X, grid=Y, smem=Z>
  Wall:        <us> µs (N=<iters>, warmup >1s)
  DRAM:        <pct>% of 1792 GB/s
  SM:          <pct>% of peak
  Occup:       <pct>%
  TC util:     <pct>%
  Clocks live: <MHz SM>/<MHz mem>/<W>   (healthy: 2850/13801/~500)
  Bound by:    <memory|compute|latency|stalls>  reason: <top stall>
  vs baseline: <±X%> on tg (decode)
```

## Publishing numbers (keep docs from going stale)

- **`tests/perf_baseline.json` is the canonical gate — read the current values there, never from this skill** (a number copied into a doc is a number that will be wrong). It pins two gates, not one: throughput (`metrics.prefill_tps` / `decode_tps`, 3%/5%) and **peak VRAM** (`metrics.memory_mb.own_peak_mb` against `thresholds.vram_increase_pct`, evaluated by `scripts/verify.sh`). The file carries its own `_note` explaining any pin that is not comparable to older ones. Refresh ONLY when a change *intentionally* moves perf **or peak VRAM**: `make gen-perf-baseline`, on a healthy-host day (STOP #4), and say so in the PR. **The gate measures spec-OFF decode** (`--set speculative.ngram=false`): with speculation ON the self-repetitive bench prompt (~99.9% accept) measures the batched spec-verify GEMMs, which are restart-volatile — ungateable at 3%.
- **Refreshing on the wrong day bakes in the wrong bar, in both directions.** A baseline sampled on a peak day put its 3% threshold inside the normal range, so ordinary days failed spuriously. A baseline sampled while another process holds the GPU pins a floor that hides real regressions. Before refreshing: no other compute process, healthy clocks, and a second cold-median run that agrees.
- **When a gate fails, rule out the cheap causes before bisecting**, in this order: (0) is this process still VRAM-resident — at ~0 MiB free, WSL2/WDDM oversubscribes into host memory and every allocation keeps succeeding while bandwidth falls off a cliff (~1530 GB/s resident vs ~237 GB/s spilled). That is #1103: 55 tok/s at server defaults on a model that benches far higher. `--mem-report` prints free VRAM at init; **a successful `cudaMalloc` is not evidence of room** (28 GiB succeeds with 22.6 GiB reported free) — measure bandwidth or read the free figure. (1) is anything else on the GPU — **read `nvidia-smi --query-gpu=memory.used`, not the process list**: `--query-compute-apps` returns an EMPTY table on WSL2 even while VRAM is held, and `docker ps` misses it too once the holder is gone. Measured 2026-08-11: 16.4 GiB held against a ~1.3-1.6 GiB WSLg baseline, no container running, `--query-compute-apps` blank — decode read −5.5% at healthy clocks (2895 MHz / 13801 MHz / 490 W) and a paired A/B found no code effect, so it looked exactly like a #526 depressed-host day. It was not: when the driver reclaimed the memory the same build measured −1.24% and passed. A `docker run` killed mid-flight (a `timeout` around a bench) can leave the commitment behind for tens of minutes. A forgotten server container reads ~−12%; (2) can the diff even reach the measured code (`git diff --stat main -- src/ include/ tools/imp-cli/` empty ⇒ a decode regression is impossible); (3) does a cold-median run reproduce the verify-fast number.
- **`docs/BENCHMARKS.md`** is SHA-anchored (method, date, commit, command, tok/s). Update it — and the README numbers — in the same commit as the perf change. `scripts/check-release.sh` gates release-touching PRs.
- `bash scripts/scoreboard.sh` tallies hero-model status vs llama.cpp.

## A published verdict expires when its path is fixed

**Three times in two days a documented verdict priced a build that no longer
existed**, and each time the fix was younger than the verdict by weeks, not
months:

| verdict, as written | re-measured | what sat between |
|---|---|---|
| "MTP loses: 84.7-85.8 vs ~88 tok/s" | **+21.3 %** at k=1 (#1481) | `ea547a53`, 3 weeks |
| "`token_recycling` still net-negative, −7 %" | **−0.27 %**, neutral (#1483) | the same commit |
| "the marginal row cost is unattributed" | register pressure, both fixes refuted (#1482) | — |

The pattern is not "old numbers drift". It is that **a fix to the measured path
retires every verdict that ran through it**, immediately and completely, while
the document keeps reading like a current finding. `token_recycling` and MTP
share one line of code — `greedy_argmax_all` on the verify chunk — so one commit
invalidated both, and nothing in either entry pointed at the other.

So, when you read a verdict before acting on it:

1. `git log --oneline <PROV commit>..HEAD -- <the files the measured path lives in>`.
   Not the whole tree: on this repo *every* provenance block has perf-path
   commits behind it (all 20, checked 2026-08-19), so "commits happened" ranks
   nothing. The question is whether one of them touched **this** path.
2. Prefer re-running the harness over reasoning about the delta. The re-runs
   above cost minutes each because the harnesses are in `tools/analysis/`
   (`mtp_k_sweep.sh`, `token_recycling_ab.sh`) — ship one with every verdict.
3. Check the **level**, not only the delta: `token_recycling` re-measured at
   156 tok/s where the original read 99.37, because an unrelated fix (#1102)
   sits between. An absolute number weeks old is not a baseline.

An automated staleness gate was tried and is not worth building: keyed on
perf-path commits it fires on 100 % of blocks, which is the failure mode skill
`find-stubs` exists to warn about — a check without a baseline reads normal as a
finding.

## Red flags — STOP and re-run

- Reporting `pp512` delta without a decode delta → on a MoE model that spread is ~38 % across process starts, you're seeing noise
- Reading `nvidia-smi` as "the GPU is free" on WSL2 → load from the **Windows side** is invisible here: `--query-compute-apps` stays blank and `docker ps` shows nothing, while `memory.used` and `utilization.gpu` do report it. Guard on those two, not on the process list (observed 2026-08-14: 12.9 GiB held at 96 % util, no container, no visible process)
- Trusting a cold single-shot number → first ~1 s runs at idle clocks (NOT heat — this box never throttles)
- Cross-day decode delta without sampling clocks during the bench → host drift is 8–15%
- Refreshing the baseline on a depressed-host day → bakes a low bar in
- Including `cudaMalloc/Free` in timing → allocate once outside the loop
- Measuring after a build with non-default CMake options → `verify-fast` does NOT rebuild, so the last image stands in as the baseline. An `IMP_ALLOC_INTERPOSE=ON` image reads ~3% low and reproduced to 0.3% across four re-measurements before anyone doubted the binary (`AUDIT.md` G16). Reproducibility says nothing about which binary you hold.
- Trusting `ncu` wall-clock → ncu serializes/replays; use `nsys` or `cudaEvent` for real time
- Comparing against wrong peak → FP16 ≠ FP8 ≠ FP4; pick the kernel's dtype
- A/B without graphs both ON and OFF → graph replay can hide silent fallback (see `check-degeneration`)
- Back-to-back multi-model sweep deltas → isolate per process, one model each
- Decode-kernel A/B on a dense model without `--set speculative.ngram=false` → you measured the spec-verify path (STOP #6)
- Claiming a compiler/source tweak is "perf-neutral" without a SASS diff → byte-identical SASS is *provably* inert; diff `cuobjdump -sass` before wasting bench trials (`[[assume]]` lesson, 2026-07-08)
