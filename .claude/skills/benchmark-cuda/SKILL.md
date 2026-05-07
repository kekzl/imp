---
name: benchmark-cuda
description: Use when benchmarking, profiling, or A/B-testing CUDA kernels in the imp inference engine on RTX 5090 (sm_120). Triggers on "benchmark kernel", "profile cuda", "ncu", "nsys", "kernel timing", "occupancy", "bandwidth bound", "compute bound", "roofline", "perf baseline", "kernel feels slow", "is this regression real".
---

# CUDA Kernel Benchmarking — imp / sm_120 / RTX 5090

Pair with `sm120-cuda-expert` for optimization decisions.

## STOP — read first

**Decode (`tg256`) is the only reliable A/B signal.** `pp512` varies up to **2.6× across container restarts** because cuBLAS picks different algorithms each cold start. Never gate on prefill-only numbers; always show decode delta too. The CI gate (`tests/perf_baseline.json`) reflects this: 3% decode threshold, 5% prefill threshold.

## Theoretical peaks (RTX 5090)

| Metric | Peak |
|--------|------|
| HBM bandwidth | 1,792 GB/s |
| FP16 TC | 838 TFLOPS |
| FP8 TC | 1,677 TFLOPS |
| FP4 TC | 3,354 TOPS |
| L2 cache | 96 MB |

Below 60% bandwidth (memory-bound) or 50% compute (compute-bound) → investigate.

## Pick the right tool

| Goal | Tool | Notes |
|------|------|-------|
| End-to-end engine perf | `make bench` → `tools/imp-bench/` | runs across baseline models, reports tg/pp |
| Per-config sweep with MBU/MFU/TTFT/TBT | `bench/bench.py` | CSV output, optional llama.cpp compare |
| Refresh perf baseline | `scripts/gen_perf_baseline.sh` | run after intentional perf changes; writes `tests/perf_baseline.json` |
| Regression gate | `make verify-fast` | reads `tests/perf_baseline.json`, 3%/5% thresholds |
| Single kernel — wall-clock A/B | `cudaEvent` in launcher | see Step 1 |
| Single kernel — metrics, occupancy, stalls | `ncu` | see Step 2 |
| Multi-kernel timeline / launch overhead / graph behavior | `nsys` | see Step 3 |
| Compare imp vs llama.cpp | `bench/profile.sh` | apples-to-apples on same models |

## Step 1: cudaEvent in-code (quick A/B)

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);

// Warmup — always >=3 iterations (first launch has JIT/cache penalty)
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

Rules: warmup ≥3, N_ITER ≥100 for kernels <100µs, lock clocks (`sudo nvidia-smi -lgc 2400,2400`; reset `-rgc`), kill concurrent GPU consumers.

## Step 2: Nsight Compute (ncu) — per-kernel metrics

Use the helper to get a consistent metric set:

```bash
./.claude/skills/benchmark-cuda/ncu-basic.sh "my_kernel.*" ./build/imp-bench
```

Or invoke directly with the canonical metric list — see `ncu-basic.sh` for the full set. Key metrics to read:

| Metric | Meaning | Target |
|--------|---------|--------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization | >70% compute-bound |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth | >70% memory-bound |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy | context-dependent |
| `smsp__inst_executed_pipe_tensor_op_*` | TC activity | non-zero if TC kernel |
| `l1tex__t_sector_hit_rate` | L1 hit rate | >90% for cached |
| `stall_*` | Where warps stall | lowest = bottleneck |

Always `--launch-skip 3 --launch-count N` to skip warmup. Compile with `-lineinfo` for source-correlated stalls (`--set detailed --import-source yes`).

## Step 3: Nsight Systems (nsys) — timeline

When you suspect: launch overhead, H2D/D2H stalls, stream serialization, CUDA Graph behavior.

```bash
nsys profile --stats=true -t cuda,nvtx,osrt \
    --cuda-memory-usage=true -o timeline \
    --force-overwrite=true ./build/imp-bench
nsys stats timeline.nsys-rep
```

**`nsys` needs `--no-cuda-graphs`** when measuring per-kernel times — graph replay hides individual kernel timings.

Red flags: gaps between launches >10µs (CPU-bound) · H2D/D2H during compute without overlap · CUDA Graph not collapsing launches (graph disabled or stream-captured wrong).

## Step 4: Roofline (one-liner)

`AI = total_flops / total_bytes_moved` (matmul FLOPs = `2·M·N·K`; bytes from `dram__bytes.sum` in ncu). Ridge points: FP16=468, FP8=936, FP4=1873 FLOP/byte. AI < ridge → memory-bound; AI > ridge → compute-bound.

## Report template

```
Kernel: <name>, config: <block=X, grid=Y, smem=Z>
  Wall:        <us> µs (N=<iters>, warmup=3)
  DRAM:        <pct>% of 1792 GB/s
  SM:          <pct>% of peak
  Occup:       <pct>%
  TC util:     <pct>%
  Bound by:    <memory|compute|latency|stalls>  reason: <top stall>
  vs baseline: <±X%>
```

## Red flags — STOP and re-run

- Reporting `pp512` delta without `tg256` delta → variance is up to 2.6×, you're seeing noise
- Skipping warmup → first launch 2–10× worse (JIT + cold caches)
- Profiling with clocks unlocked → thermal throttling skews A/B
- Including `cudaMalloc/Free` in timing → allocate once outside the loop
- Trusting `ncu` wall-clock → ncu serializes/replays kernels; use `nsys` or `cudaEvent` for real time
- `ncu` without `-lineinfo` → no source correlation
- Comparing against wrong peak → FP16 ≠ FP8 ≠ FP4 TOPS; pick the kernel's dtype
- Small `N_ITER` on noisy kernels → run ≥100, check stddev not just mean
- A/B without graphs both ON and OFF → graph replay can hide silent fallback (see `check-degeneration` skill)
