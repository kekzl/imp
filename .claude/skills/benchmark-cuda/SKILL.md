---
name: benchmark-cuda
description: Use when benchmarking, profiling, or A/B-testing CUDA kernels or end-to-end perf in the imp inference engine on RTX 5090 (sm_120), including refreshing tests/perf_baseline.json or publishing numbers to docs/BENCHMARKS.md and the README. Triggers on "benchmark kernel", "profile cuda", "ncu", "nsys", "kernel timing", "occupancy", "bandwidth bound", "compute bound", "roofline", "perf baseline", "is this regression real", "decode dropped". Do NOT use for writing/optimizing kernel code (sm120-cuda-expert) or output-quality checks (check-degeneration).
---

# CUDA Kernel Benchmarking — imp / sm_120 / RTX 5090

Pair with `sm120-cuda-expert` for optimization decisions.

## STOP — what is a real signal on this box

1. **Decode is the only reliable A/B signal.** Prefill (`pp512`) varies up to **2.6× across container restarts** (cuBLAS algo re-selection each cold start). Never gate on prefill-only numbers. For prefill-**kernel** A/Bs ≤5%, end-to-end pp cannot resolve the delta even with full methodology — compare **nsys per-kernel time sums** instead (PR #648 lesson). Metric keys: `tests/perf_baseline.json` gates on **`tg128`**; ad-hoc bench runs usually report `tg256` — compare like with like.
2. **The GPU is water-cooled and never throttles** (idles ~30 °C). Do NOT insert temperature cooldowns. The 15 s cooldown in `gen_perf_baseline.sh` resets **cuBLAS algo-selection state** under sustained load, not temperature.
3. **Idle downclock is the dominant cold-start artifact.** Clocks take ~1 s to ramp under load — the first second reads artificially LOW (produced a spurious −42% that re-measured +20% clean). Always precede timed reps with a **discarded warmup run >1 s**; imp's built-in `Warmup...` is too short.
4. **Decode can read 8–15% low for a whole day** (host/driver state, WSL2 — issue #526). Sample clocks DURING the bench: `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv -l 1`. Healthy load ≈ **2850 MHz SM / 13801 MHz mem / ~500 W**. Lower mem clock or power = depressed host day → do NOT trust cross-day deltas or refresh baselines that day.
5. **Back-to-back sweeps read 6–10% low** vs isolated runs. One model per process, isolated, before trusting cross-model deltas.
6. **n-gram speculation is default-ON for dense models (PR #781) and `--bench` prompts are self-repetitive → ~99.9% draft accept.** A dense `--bench` run mostly measures the batched *verify* path, not the decode GEMV you changed. For decode-kernel/GEMV A/Bs add `--set speculative.ngram=false` (both arms!) — a real +2–3% GEMV win is invisible under speculation (KPAR lesson, 2026-07-07).
7. **No-graphs nsys/ncu time shares OVERSTATE tiny-kernel classes — validate any launch/latency-class lever with a graphs-ON e2e A/B BEFORE building it.** The roofline pipeline profiles with `--no-cuda-graphs`; under the shipped graphs+PDL decode loop the tiny launches largely overlap away. Measured 2026-07-13 (Qwen3-30B-A3B): the sum of no-graphs per-kernel times is ~4.6 ms/step vs the real graphs-ON step of ~2.55 ms — 44% of the no-graphs window is already hidden. Concretely: fusing gemv_gate_fp32+topk_gating (6.9% no-graphs window share, 2 launches/layer eliminated, bit-identical outputs) moved e2e by **0%**; capping decode split-K (reduce pass "5.8 µs at 40 GB/s") REGRESSED −21 to −35% because the splits' SM-fill matters more than the reduce. The Amdahl lever list in the roofline report is only trustworthy for classes that hold real bytes/critical-path time (big GEMVs, data-dependent chains) — not for grid-(1,1,1) launch-latency classes (moe_routing, rmsnorm, rope, kv_write, elementwise).

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
| Refresh perf baseline | `make gen-perf-baseline [MODEL=/models/…]` | cold-median: 5 trials × 15 s cooldown; writes `tests/perf_baseline.json` |
| Regression gate | `make verify-fast` | 3% decode / 5% prefill thresholds |
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

- **`tests/perf_baseline.json`** is the canonical CI gate. Refresh ONLY when a change *intentionally* moves perf: `make gen-perf-baseline`, on a healthy-host day (STOP #4), and say so in the PR. Current Q8 `tg128 = 269.50` (cold-median, 2026-06-12; normal healthy range 266–278). The previous 286.4 baseline (2026-06-07) was sampled on a documented **peak day** — its 3% gate threshold (277.8) sat INSIDE the normal range, so it failed on ordinary days; it was corrected down in #697. A small gate failure can be host drift, not a regression; sample clocks before concluding.
- **`docs/BENCHMARKS.md`** is SHA-anchored (method, date, commit, command, tok/s). Update it — and the README numbers — in the same commit as the perf change. `scripts/check-release.sh` gates release-touching PRs.
- `bash scripts/scoreboard.sh` tallies hero-model status vs llama.cpp.

## Red flags — STOP and re-run

- Reporting `pp512` delta without a decode delta → 2.6× restart variance, you're seeing noise
- Trusting a cold single-shot number → first ~1 s runs at idle clocks (NOT heat — this box never throttles)
- Cross-day decode delta without sampling clocks during the bench → host drift is 8–15%
- Refreshing the baseline on a depressed-host day → bakes a low bar in
- Including `cudaMalloc/Free` in timing → allocate once outside the loop
- Trusting `ncu` wall-clock → ncu serializes/replays; use `nsys` or `cudaEvent` for real time
- Comparing against wrong peak → FP16 ≠ FP8 ≠ FP4; pick the kernel's dtype
- A/B without graphs both ON and OFF → graph replay can hide silent fallback (see `check-degeneration`)
- Back-to-back multi-model sweep deltas → isolate per process, one model each
- Decode-kernel A/B on a dense model without `--set speculative.ngram=false` → you measured the spec-verify path (STOP #6)
- Claiming a compiler/source tweak is "perf-neutral" without a SASS diff → byte-identical SASS is *provably* inert; diff `cuobjdump -sass` before wasting bench trials (`[[assume]]` lesson, 2026-07-08)
