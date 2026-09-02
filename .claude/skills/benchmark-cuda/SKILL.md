---
name: benchmark-cuda
description: Use when benchmarking, profiling, or A/B-testing CUDA kernels or end-to-end perf in the imp inference engine on RTX 5090 (sm_120), including refreshing tests/perf_baseline.json or publishing numbers to docs/BENCHMARKS.md and the README. Triggers on "benchmark kernel", "profile cuda", "ncu", "nsys", "kernel timing", "kernel sum", "occupancy", "bandwidth bound", "compute bound", "roofline", "perf baseline", "is this regression real", "decode dropped", "aggregate throughput", "two-image A/B", "prefill kernel A/B". Do NOT use for writing/optimizing kernel code (sm120-cuda-expert) or output-quality checks (check-degeneration).
---

# CUDA Benchmarking - imp / sm_120 / RTX 5090

Pair with `sm120-cuda-expert` (levers) and `docs/internals/BENCHMARKING.md` (measurement contract).

## STOP: what is a real signal on this box

| # | Fact | Consequence |
|---|---|---|
| 1 | pp512 spread across process starts is a property of the MODEL: 0.6-1.2% on Qwen3-8B Q8_0, 37.6% on a resident NVFP4 MoE; cuBLAS algo re-selection alone 3.50% over nine starts | Decode is the A/B signal. Prefill-kernel deltas <=5% resolve ONLY in nsys per-kernel sums (`tools/analysis/prefill_kernel_ab.sh`), never in e2e pp. Gate key is `tg128`; ad-hoc runs print `tg256` |
| 2 | The GPU is water-cooled, idles ~30 C, never throttles | No temperature cooldowns. The 15 s cooldown in `gen_perf_baseline.sh` resets cuBLAS algo state |
| 3 | Idle downclock: clocks ramp ~1 s; between bench shapes they fall to 210 MHz | Discard >1 s warmup before timing; warm per shape in isolated benches. A cold shot once read -42% that re-measured +20% |
| 4 | Decode reads 8-15% low for a whole day (host/driver state, #526) | Sample clocks DURING the bench: healthy 2850 MHz SM / 13801 MHz mem / ~500 W. Never refresh a baseline or trust a cross-day delta without them |
| 5 | Back-to-back sweeps read 6-10% low | One model per process |
| 6 | `--bench` prompt is `tokens[i] = i % vocab` (512 distinct ids, `tools/imp-cli/main.cpp`; no 6-gram repeats at `speculative.min_match=6`); n-gram drafts come from the GENERATION looping on it, quantisation-dependent: Qwen3-14B NVFP4 accepts 504/504, Q6_K 6/96 and bistable | Decode-kernel A/Bs on dense models: `--set speculative.ngram=false` in BOTH arms, or a real +2-3% GEMV win is invisible |
| 7 | `imp-cli --bench` pins `speculative.moe=false`, `speculative.suffix=false`, `speculative.hybrid=false`, `speculative.mtp_k=0`, `server.recurrent_snapshot_mb=0` (`tools/imp-cli/args.cpp`, `apply_config_pins`); an explicit `--set` wins | Every new auto-default needs a bench pin or `gen_perf_baseline.sh` bakes speculation into the gate (mtp auto engaged the head before its pin, 71.7 vs 86.8 tok/s) |
| 8 | A number is comparable only to a run with the same flags | `--bench` disables prefix caching since #1061 (`--set server.prefix_cache=true` restores the old band); `tg128_at_ctx_2048` needs `--bench-pp 2048`; pp16/pp512 read ~+13%/+10% on the same build; `verify-fast` runs fewer trials than `gen-perf-baseline`. Log the `imp.conf loaded from`, `dtype=`, `attn_decode=` lines per arm (a stray repo-root `imp.conf` made two profile arms identical before #1784) |
| 9 | No-graphs profiles overstate tiny-kernel classes ~1.8x at batch=1 on MoE; on Qwen3.8 M=1 the decode graph is strictly serial (union == sum, #1797) | A batch=1 lever must hold bytes or critical-path math. At 32 streams the class pays (RMSNorm +6.8% #1769, act-quantize +4.6% #1771, producer fusion +2.6% #1773) but residual-accumulate lost -0.9% (#1793). Measure the regime you claim |
| 10 | Isolated kernel benches measure L2: every single Qwen3.8 weight fits the 96 MB L2 (>1792 GB/s = self-disqualified; the GQA-tile decode kernel built on that misread measured -9% e2e, #1785); balanced per-expert inputs flip the sign vs real routing (grouped GEMM mt32 -8.7% isolated, +4% in situ) | Rotate >=4-8 x 100 MB slabs; sub-wave shapes are bistable 1.6x between runs; verdicts from in-process nsys kernel sums; when the in-situ sweep ranks like the isolated one, record that too (#1768) |
| 11 | Capture costs under nsys are CUPTI-inflated (27.8 ms gaps; >1 ms gaps at the wave ramp = graph captures) | Never price graph capture from a profile |
| 12 | "Neutral" on a gate-based feature is usually a dead path | Prove activity by kernel launch counts (stream-K workspace refused every launch: 1960 -> 1400 CUTLASS launches, 22k -> 4.2k tok/s; sparse needs the `sparse decode attention ACTIVE` line in one arm only) |
| 13 | Batch=1 roofline: 1628 GB/s resident (8 GiB sweep, #1797), spilled ~237 GB/s; Qwen3.8 spec-off ceiling ~112 tok/s, measured 87.4 | Re-measure the sweep before quoting bandwidth; the old 1530 pin is stale |

## Methodology (every A/B)

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 10 reps, 3+ trials, one model per process, warm >1 s.
- GPU free first: `make check-gpu` (`scripts/require_free_gpu.sh`) or `~/.claude/skills/gpu-stats/gpu-busy-check.sh` (utilisation + memory; a Windows tenant is invisible to `docker ps` and `--query-compute-apps`). `gpu-busy-check && (nohup ...) ; echo launched` prints "launched" on BUSY too; test the exit code.
- Server A/Bs: nothing besides the harness's own server (`docker ps -q | wc -l` = 0 before single-process benches).
- Everything runs in Docker (`imp:test`, models from `$HOME/models`). `imp-cli` has no `--ctx`; the ceiling is `--max-seq-len`. 32k prompts via `--prompt-file` (argv caps at ~128 KiB).
- Bench from binary COPIES (`-v dir:/bin_arm`): `make dev` rebuilds `build-dev/` under a running chain.
- dtype/KV A/Bs: force emitted tokens equal (`max_tokens` below need); library-reserve state equal in both arms (a `--rm` container plans with the 3900 MiB constant, a mounted `/home/imp/.cache/imp` with its measurement: 716 blocks apart on Qwen3.8).
- PPL verdicts: `--set runtime.deterministic=true` both arms (0.35% run-to-run otherwise), `tools/analysis/ppl_corpus_45k.txt`, `--set speculative.mtp_k=0` (auto loads the head, +0.79 GiB). Qwen3.6-35B PPL moves +-0.2..0.5% between fp32-equivalent kernels (routing flips): numerics judge is Qwen3.8-27B-NVFP4-vllm (fused GDN 4.6283) plus the unit-test state diff and `tools/analysis/layer_ab_diff.py` added divergence.
- Every verdict ships its harness in `tools/analysis/` or its md5 in the PROV block (`scripts/bench_competitive.sh` re-execs from a `mktemp` copy and prints `harness: md5=`).
- Check ad-hoc `grep`/`awk` against a known case first (`grep -c 'hero$'` also matched `nonhero`).

## Pick the right tool

| Goal | Tool |
|---|---|
| End-to-end engine perf | `make bench` (`imp-cli --bench --bench-pp 512 --bench-reps 5`) |
| Single model quick check | `make test-perf` (Qwen3-8B Q8_0) |
| Refresh perf baseline | `make gen-perf-baseline [MODEL=/models/...]`: 5 trials x 5 reps, 15 s cooldown, median; writes `tests/perf_baseline.json` incl. `own_peak_mb` |
| Guarded re-pin (refuses top-range / volatile days) | `scripts/repin_baselines_if_median.sh` |
| Regression gate | `make verify-fast` (`scripts/verify.sh`): 8% decode / 8% prefill / 10% peak VRAM |
| VRAM attribution | `imp-cli --mem-report` (`own_peak=` is what the gate parses) |
| North-star gate | `make verify-north-star` (`tests/perf_baseline_north_star.json`) |
| Long-context decode A/B | `scripts/bench_longctx_ab.sh`; capacity vs decode `tools/analysis/ctx_capacity_decode_sweep.sh` |
| Prefill-kernel A/B for one config key (nsys kernel sum per arm) | `tools/analysis/prefill_kernel_ab.sh`; FA2 hd=256 variant `fa2_hd256_bkv_ab.sh` |
| 32-stream aggregate A/B, config key | `tools/analysis/smallm_v2_conc_ab.sh` + `tools/analysis/conc_client.py` |
| 32-stream aggregate A/B, CODE change | `tools/analysis/two_image_conc_ab.sh` (two prebuilt images, alternating) |
| Long-context serving A/B (tg8/tg520 differential) | `tools/analysis/serving_sparse_ab.sh` + `longctx_conc_client.py` |
| Serving idle attribution | `tools/analysis/serving_idle_profile.sh` + `nsys_gap_attribution.py` |
| MTP / speculation | `tools/analysis/mtp_adaptive_ab.sh`, `mtp_k_sweep.sh`, `token_recycling_ab.sh`, `scripts/mtp_accuracy_bench.sh` |
| Decoder ITL under concurrent ingest | `scripts/bench_prefill_latency.py` |
| Per-config sweep MBU/MFU/TTFT/TBT | `bench/bench.py` |
| Single kernel wall-clock | `cudaEvent` in the launcher (below) |
| Per-kernel metrics, stalls | `ncu` (below); wrapper `bench/profile.sh` |
| Timeline, launch gaps, graphs | `nsys` (below) |
| Full roofline sweep | `make roofline-measure` (`tools/roofline/`, README there); pin `make roofline-pin`, `roofline-regress`; history `tools/roofline/history/` |
| imp vs llama.cpp | `make bench-competitive` (`scripts/bench_competitive.sh`, writes `/tmp/bench_competitive.tsv`) |
| Hero scoreboard | `bash scripts/scoreboard.sh` |

Phantom: gemma-3-12b `--bench` prints bogus tok/s (#514); trust its PPL only.

## Aggregate throughput (the serving regime)

- N-stream burst vs a fresh `imp-server` per arm; aggregate = sum `completion_tokens` / wall per wave; median over waves, 3+ alternating trials, all pairs same sign.
- `conc_client.py PORT CONC WAVES [TAG]`: unique prompts per stream, 300-token greedy. Harness baseline ~1714-1780 tok/s on 40-token prompts vs ~1030-1050 published on 130-token prompts; never mix.
- Pin `runtime.max_batch_size`, `runtime.max_seq_len`, `kv_cache.max_blocks` in BOTH arms (free-VRAM swing ~1.6 GB re-resolves auto batch: 5 -> 28 once moved 224.68 -> 630.19 tok/s with no code change).
- `MEDIAN 0.0` = broken client (exceptions become `(0, elapsed)`, model name hardcoded), not a regression. Read stderr and per-wave token counts.
- Discard wave 1 (graph captures + ramp: 629 vs 954-991 steady).
- Default-ON knobs that define "same flags": `runtime.prefill_batch`, `gdn.state_bf16`, `gemm.nvfp4_smallm` (+`_impl=2`, `_pair`), `speculative.mtp_adaptive_k`, `runtime.prefill_chunk_decode_cap` (1024; 4096 = +~10% burst lever), `runtime.graph_prewarm`.
- Profile the right binary: `make build` tags only `imp:test`; `imp:builder` can be days old.
- Method behind #1750-#1793: two-image A/B, all pairs same sign. Prefill-bound bursts cannot see KV capacity (32x8k with 60-token completions: identical walls while the pool grew 3.2x); capacity levers need `max_tokens >= 512`.
- Per-kernel time inside replayed decode graphs: `nsys --cuda-graph-trace=node` (#856). On a server: no `--delay`/`--duration` (flush hangs on `cudaProfilerStop`, SIGKILL loses all); capture fully, `docker stop` gracefully, filter the window by timestamp.

## Step 1: cudaEvent (quick A/B)

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
for (int i = 0; i < 3; i++) kernel<<<...>>>(...);   // plus >1 s busy warmup (STOP #3)
cudaDeviceSynchronize();
cudaEventRecord(start);
for (int i = 0; i < N_ITER; i++) kernel<<<...>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);
float avg_us = (ms / N_ITER) * 1000.0f;
```

N_ITER >= 100 for kernels <100 us; report stddev; allocate outside the loop; sample clocks during the run.

## Step 2: ncu

ncu is not in the runtime image; mount the host install (`/opt/nvidia/nsight-compute/2026.2.1`, also 2026.2.0, 2025.4.0) or build `tools/Dockerfile.ncu`:

```bash
docker run --rm --gpus all -v $HOME/models:/models \
  -v /opt/nvidia/nsight-compute/2026.2.1:/ncu -v /tmp/out:/out --user root \
  imp:test /ncu/ncu --kernel-name "regex:my_kernel.*" --launch-skip 3 --launch-count 10 \
  -o /out/profile imp-bench ...      # chmod 777 /tmp/out first
```

Canonical metric set: `.claude/skills/benchmark-cuda/ncu-basic.sh "<kernel-regex>" <binary> [args]`.

| Metric | Meaning | Target |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization | >70% compute-bound |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth | >70% memory-bound |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy | context |
| `smsp__inst_executed_pipe_tensor_op_*` | TC activity | non-zero for TC kernels |
| `smsp__average_warps_issue_stalled_*` | stall reasons | lowest = bottleneck |

ncu traps: `--launch-skip` above the launch count waits forever; ncu + CUDA graphs hangs in the async graph loop (always `--no-cuda-graphs`, as the roofline harness does); the "Available Kernels" list shows base names, template regexes do not match; `--clock-control base` does NOT stop idle downclock between replays (a launch at 0.31 GHz read 99.7% of roofline; `tools/roofline/config.json` `ncu.clock_floor_ghz=1.2` now drops such launches as `n_launches_dropped_clock`); multi-pass replay dies on TMA kernels on this WSL2 driver (the roofline runs single-pass metric groups). A low memory-stall ratio next to low bandwidth means "not bandwidth-bound", not "too few warps".

## Step 3: nsys

```bash
nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda,nvtx \
    --stats=true --cuda-memory-usage=true -o timeline --force-overwrite=true imp-bench ...
nsys stats timeline.nsys-rep
```

- WSL2 needs sampling off or nsys hangs.
- Graphs hide captured kernels: `--no-cuda-graphs` for the mix, `--cuda-graph-trace=node` when graphs must stay on.
- In `imp:toolchain` the qdstrm -> nsys-rep conversion fails silently (`libcap.so.2`, `libdw.so.1` missing): `apt-get install libcap2 libdw1t64`, then `/opt/nvidia/nsight-compute/*/host/*/QdstrmImporter -i x.qdstrm -o x.nsys-rep`.
- nsys prints template args as `<(int)64, (int)256, ...>`; grep accordingly.
- Class maps come from the nsys steady-state sum (n >= 100 launches, decode vs prefill split by per-launch duration from the sqlite), never from the roofline's 120-launch ncu window: that window showed no GDN scan class while the scan was 42% of the hybrid pp512 wall.
- On Qwen3.6-35B `total_kernel_ms` scatters 1203-1255 (MoE routing); judge by class sums plus pp.
- `.nsys-rep`/`.sqlite` and `diagnostics.dump_hidden_dir` dumps (3.6 GB per arm at 300 tokens) fill the 40 GB tmpfs; delete after reading.
- compute-sanitizer (`make sanitize`) does not work on WSL2 (WDDM).

## Step 4: roofline

`AI = FLOPs / bytes` (matmul `2*M*N*K`; bytes from `dram__bytes.sum`). Peaks: 1792 GB/s DRAM, FP16 838 TFLOPS, FP8 1677, FP4 3354 (datasheet), L2 96 MB. Measured FP4 `mma.sync` = 2019 TOPS (~1/2 datasheet), f32-accumulate 1/4 rate; GeForce TF32 = 1/2 the FP16-fp32acc rate. Use measured peaks or every FP4 kernel reads falsely bad. Roofline cells (`tools/roofline/config.json`): `q8-dense` Qwen3-8B-Q8_0, `nvfp4-dense` Qwen3-14B-NVFP4, `nvfp4-moe` Qwen3-30B-A3B-NVFP4, `q4k-moe`, `q4k-dense-hd256` gemma-3-12b, `nvfp4-hybrid` = Qwen3.6-35B (NOT Qwen3.8-27B). Pinned baseline `tools/roofline/history/BASELINE` (run 1d5b9230 pinned #1835).

## Report template

```
Kernel: <name>, config: <block=X, grid=Y, smem=Z>
  Wall:        <us> us (N=<iters>, warmup >1s)
  DRAM:        <pct>% of 1792 GB/s
  SM:          <pct>% of peak
  Occup:       <pct>%
  TC util:     <pct>%
  Clocks live: <MHz SM>/<MHz mem>/<W>   (healthy: 2850/13801/~500)
  Bound by:    <memory|compute|latency|stalls>  reason: <top stall>
  vs baseline: <+-X%> on tg (decode), pairs <n/n same sign>
```

## Publishing numbers

- `tests/perf_baseline.json` is the gate: read values THERE, never from a skill. Two gates: throughput (8%/8% since #1400) and peak VRAM (`metrics.memory_mb.own_peak_mb`, 10%). Refresh only for an intentional perf or VRAM move: `make gen-perf-baseline` on a healthy day (STOP #4), second cold-median run agreeing, and say so in the PR. The gate measures spec-OFF decode.
- Before bisecting a red gate: (0) VRAM-resident? At ~0 MiB free WDDM spills into host memory and every `cudaMalloc` still succeeds (28 GiB succeeds at 22.6 GiB reported free); bandwidth is the discriminator (#1103: 55 vs 391 tok/s). (1) Anything else on the GPU? Read `nvidia-smi --query-gpu=memory.used,utilization.gpu`, not the process list (16.4 GiB held with a blank `--query-compute-apps` read -5.5%; a forgotten server ~-12%; a killed `docker run` keeps its commitment for tens of minutes). (2) Can the diff reach the measured code (`git diff --stat main -- src/ include/ tools/imp-cli/`)? (3) Does a cold-median run reproduce it?
- `docs/BENCHMARKS.md` rows are SHA-anchored; update them and the README block in the same commit; `scripts/check-release.sh` gates release PRs. `docs/PERF.md` owns every number (docs-sync).

## A published verdict expires when its path is fixed

| Verdict as written | Re-measured | Between |
|---|---|---|
| "MTP loses: 84.7-85.8 vs ~88 tok/s" | +21.3% at k=1 (#1481) | `ea547a53`, 3 weeks |
| "`token_recycling` net-negative, -7%" | -0.27%, neutral (#1483) | same commit |
| "+21.3% at k=1" (LIMITATIONS) and "+15% k=2" (/health) | measured without template/think; on think traffic MTP was dead (#1796) | spec verdicts need a think arm |
| "GDN scan: `must be FP32`" | layout bug; BF16 +12.5% (#1776) | - |

1. `git log --oneline <PROV commit>..HEAD -- <files of the measured path>` (all provenance blocks have perf-path commits behind them; only THIS path matters). Renamed TUs: `engine_scheduler.cpp` split into `engine_prefill.cpp`, `engine_prefill_ragged.cpp`, `engine_decode_pipeline.cpp` (#1782).
2. Re-run the harness instead of reasoning about the delta.
3. Check the level, not only the delta (`token_recycling` re-read at 156 tok/s where the original said 99.37; #1102 sat between). A verdict can also be right for the wrong mechanism: the MR row cost was register pressure, both obvious fixes refuted (#1482).

## Red flags

- pp512 delta without a decode delta (MoE spread ~38% across starts).
- `nvidia-smi` process list as "GPU free" (Windows-side load invisible; 12.9 GiB at 96% util with no container, 2026-08-14).
- Cold single shot; cross-day delta without live clocks; baseline refresh on a depressed day.
- `cudaMalloc/Free` inside the timed loop.
- Measuring after a non-default CMake build (`verify-fast` does not rebuild; an `IMP_ALLOC_INTERPOSE=ON` image reproduced -3% four times).
- ncu wall-clock as real time (ncu serializes; use nsys or cudaEvent).
- Wrong peak dtype; A/B with graphs ON only; multi-model back-to-back sweeps.
- Dense decode-kernel A/B without `speculative.ngram=false` (STOP #6).
- "perf-neutral" without a SASS diff (`cuobjdump -sass`; byte-identical SASS is proof, a bench is not).
- Gate figures typed into a PR body from memory: capture the output first, paste second.
