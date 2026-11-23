---
name: kernel-tuning
description: Make a kernel or a whole decode/prefill path fast on the RTX 5090 from its first version, and decide which lever to pull on a slow one - "kernel is slow", "GPU at 80 %", "tok/s target", new hot path for a model port, memory-bound vs compute-bound vs launch-bound, hot/cold timing, launch ablation, deletion mutant, "is this fusion worth it", "at the bandwidth, what now". Not PTX/MMA/smem mechanics (sm120-cuda-expert), not the A/B protocol or profiler recipes (benchmark-cuda), not output quality (check-degeneration).
---

# Kernel tuning - pressure from the first version

The card: 170 SMs, 1628 GB/s resident, 96 MB L2. A correct but narrow kernel leaves it idle and no later pass finds that for free.

| Evidence (Inkling-2-Mini, same math, same weights) | tg tok/s |
|---|---|
| v1, reference-grade kernels, eager (#3209) | 26.36-27.94 |
| decode graph + wide kernels (#3210) | 109.98-110.69 |
| #3216-#3222, 10 prompts x 700 tokens, no lookahead | 152.3 |
| B-91: 5 small mix / gate kernels with compile-time counts (no stack arrays, loads issued before the first use), bit-equal, no lookahead | 177.4 -> 211.3 |

Every step from 27 to 152 is a row of section 2; each one is cheaper at first write than as a rework.

## 1. Budget before the first line

| Quantity | Rule |
|---|---|
| Step budget | target tok/s -> ms per step; list every kernel class x applications per token |
| Floor per kernel | bytes it must read / 1628 GB/s; cold weights above the L2 cannot beat it |
| Grid | one CUDA block reads ~4 GB/s: blocks = what the bytes need, not what the math suggests |
| Nodes | count graph nodes per token: ~2450 nodes spread ~11 ms beside ~6.7 ms of kernel time; at 918 nodes the wall was 16.4 ms for 10.6 ms of kernel time |
| Host | 0 round trips per token; position and lengths live in device memory |

## 2. First-version checklist

| Property | Without it (measured) | With it |
|---|---|---|
| Multi-block grid for every read above a few KB | 1 block over 1.2 MB of gate weights: 180 us, decode fell to 31 tok/s | hundreds of blocks; 24-27 blocks still read 14-19 us |
| Loads overlap across lanes, never a serial loop in one thread | one key after the other ~1 us each; a thread loading global sequentially ~0.3 us per load | 32 keys per lane group 73.64 -> 87.46 tok/s; 256 threads + smem tiles |
| MMA fragments staged as full 128 B lines (cp.async into smem) | fragment loads from global cap a stream at ~1390 GB/s | ~1650 GB/s, FP8 LM head 926.6 -> 771.5 us |
| Word loads on packed bytes | 20 `LDG.E.U8` per iteration | +15.7 % (#1817) |
| A loop over the context is split-K across the grid | one CTA per row: `qsa_select` 128.4 us/layer at 13.9k; Inkling attention 7.6 of 16 ms per token at 600 tokens | 9.1 us (#2074); `attn_part_k` 60.76 -> 73.64 tok/s |
| Top-k / routing: one warp per candidate, then a pick kernel | `route_topk_k` (replaced) 153 us x 48 = 37.8 % of kernel time | 39.9 -> 55.05 tok/s |
| Reductions in two phases (partial slices, then out; Gram matrix instead of a third pass) | single-block mix 73 x 18.4 us per token | `mix_partial_k` + `mix_out_k` 95.53 -> 108.11 tok/s |
| One-row step captured as a CUDA graph on day one | eager 27 tok/s; a per-token host round trip costs -27..-45 % on a graph path | 39.9 tok/s before any kernel work |
| No cuBLAS call per token at M=1 | ~800 cuBLAS calls + ~2000 launches per token | stacked projections, one block per layer: 55.05 -> 62.51 tok/s |
| One host copy per forward call | 3 blocking `cudaMemcpy` = 0.18 ms per token | #3217 |
| Independent chains on side streams inside the graph | serial k / v / q chains | 2 side streams, part of 108 -> 143 tok/s (#3216) |
| Multi-row form shares blocks across rows | 1 row 8.37 ms, 8 rows 17.82 ms; per-row attention blocks 19.4 ms per 8-row call | one block per KV head x chunk for all rows: 15.2 ms |
| Trip counts are template parameters; no runtime-bounded register arrays, no local arrays; a thread issues its strided loads into registers before the first use (9 loads in a runtime loop: `gate_rms_k` 0.26 ms per token, unrolled 0.06) | `half2 w[64]` reloaded after the PDL wait; cleared register slots in local memory: STACK 128, 170 STL/LDL, 4.4 us; frame 24 -> 544 B | `make kernel-resources` clean |

Write the wide form first and prove it against the oracle; a "reference-grade" kernel that ships becomes the baseline of four more PRs.

## 3. Classify the binding before any rework (5 minutes)

Time the kernel alone twice: on the weights it just read (hot, L2) and rotating over more than the L2 holds (cold). Template: `GemvFp16HotCold.DISABLED_Bench` (`tests/test_gemv_hot_cold_bench.cu`): kernel variants as template instances, memcmp against the shipped kernel, 400 launches per arm, ring above 400 MB, 7 rounds, min / median / max. Run: `make dev`, then `docker run --rm --gpus all -v $PWD:/src -w /src imp:toolchain build-dev/test-compute --gtest_also_run_disabled_tests --gtest_filter='*GemvFp16HotCold*'` (10 s). Read the MINIMUM: the hot side is bistable (8.37 to 22.36 us on one kernel), and back-to-back launches from the host have a floor of 6.4-6.9 us (0.44 MB shape).

| Reading | Binding | Levers | Dead on arrival |
|---|---|---|---|
| hot much faster than cold, cold near 1628 GB/s (MoE experts 64 slots: hot = cold 119.6 us at 1600 GB/s; 32 slots 31.6 / 62.2; 8 slots 11.9 / 19.2) | memory | fewer bytes: narrower dtype, fewer distinct weights per step, higher draft acceptance, prefetch while DRAM idles | grouping, decode-once, input reuse, more loader threads |
| hot minimum = cold minimum, well under bandwidth, both above the launch floor (no case measured yet: the dense fp16 GEMV read "15.4 us both" once and re-measured hot 8.37 / cold 15.08) | compute / instructions | MMA instead of scalar, load width, smem staging, fewer instructions per byte | bytes levers alone |
| tens of us on a few blocks, bytes tiny | grid or latency chain | more blocks, lane overlap, two-phase reduce | occupancy pins |
| kernel sum well under the step wall | launch gaps, host | graph, side streams, host copies; fusion only after section 4 | PDL edges on a serial graph |

The L2 holds ~85-96 MB and dedups repeated experts by itself: 64 slots with 8 distinct experts read 67.1 us (the compute floor), which is why grouping by expert gains nothing.

## 4. Attribution: which number to believe

| Method | Use | Trap |
|---|---|---|
| `if (false)` around one launch in the real run, read tok/s | true share of a kernel in situ (outputs garbage, timing valid) | noise ~+-0.4 ms per token |
| Empty-kernel deletion mutant | upper bound of any fusion or speedup of that kernel | fused add+norm +0.7 %, qknorm_rope +1.6 %: both inside pair spread = dead lever |
| nsys in situ, `tools/analysis/nsys_exclusive_time.py` | per-kernel sums on the real workload | a short test hides context-dependent cost (27-token run hid the attention); a duration includes the PDL wait (6.3 us in model vs 2.4 isolated) |
| ncu | bandwidth %, stalls, occupancy | base clock, cold: inflates small kernels; never a time share |
| A calculation | choosing what to measure | wrong 3 times in a row on the MoE GEMV; fusion priced at +4 % measured +1.2..1.6 % |

Throughput verdict: 10 prompts x 700 tokens, temp 0, stream, tok/s = (n-1) / (total - starttransfer). One prompt with lookahead reads 141-194 tok/s on identical code.

## 5. Levers that did not pay

| Lever | Measured | Why |
|---|---|---|
| Fuse a kernel into the launch before a PDL-prefetching GEMM (gate\|up + SwiGLU) | 9/9 pairs -0.6..-3.2 % | the next GEMM already prefetches under the small kernel |
| Fuse 96 graph nodes per token (#3218) | +1.2..1.6 % | gaps are not a fixed cost per node |
| PDL edges on the Inkling graph (950 edges) | 6.217 vs 6.204 ms | chain is kernel time, not launch latency |
| Group expert slots by expert | 6.19 -> 6.42 ms | L2 dedups already |
| int8 activations + dp4a on experts | 15.22 -> 15.45 ms per 8-row call | memory-bound: same bytes |
| FP8 dense projections | 6.635 -> 6.267 ms, logits error 1.4 -> 3.2 %, parity red | precision spent on a path that is not memory-bound |
| Load order, hoisted x, PDL weight prefetch, block size on the M=1 FP16 GEMV (B-87) | 8 variants cold 14.86-15.34 us vs 15.08 | memory-bound plus 3-4 us per launch |
| 512 load threads in `attn_part_k` | 149.63 vs 149.72 tok/s | overlap was already sufficient |
| More CTAs / K-stripes on small-M GEMMs at the floor | neutral | SM coverage was not the limiter |
| Min-blocks `__launch_bounds__` | 14.5 -> 26.8 us | spills (sm120-cuda-expert law 2) |

## 6. A rework keeps the bits

- Summation order in a router or mix kernel flips near-tie experts: logits error 1.401 % -> 2.47 %. Keep the order or prove bit equality.
- Fusing two kernels changes FMA contraction: 87436 of 172800 values differed until `__fmaf_rn` / `__fmul_rn` + `__fadd_rn` were explicit (test `FusedConvSitesAreBitEqual`).
- cuBLAS GEMMs (T > 1) never run in parallel on side streams: shared workspace, text falls apart, the row-equality test does not see it.
- A variant picked by row count produces identical per-row bits or is gated to decode rows (sm120-cuda-expert, numerics rules).
- Pin one check number per path before the first rework (Inkling: parity 4/4 and logits rel_err 0.01401 / 0.00831) and read it after every step.

## 7. Stop rule

- Cold reading at 85-89 % of 1628 GB/s: the kernel is done; the step is bytes. Remaining levers are quantisation, fewer distinct weights, speculation.
- Price every idea against the step in ms first (Qwen3.8 MTP lane: 15.0 ms = fwd 12.7 + head 2.27).
- A step whose GEMMs sit at the floor leaves 1-2 % per fusion; say so instead of building the fifth one.
- Record the verdict, refuted ones included: `docs/audit/PERF_LOG.md` + a `docs/ROADMAP.md` ledger row; a refutation also goes into `sm120-cuda-expert/references/known-issues.md`.

## Order of work on a new hot path

1. Budget (section 1). 2. Wide kernels + graph (section 2) against the oracle test. 3. Hot/cold per kernel class (section 3). 4. Ablation in situ (section 4). 5. Two-arm verdict per benchmark-cuda. 6. check-degeneration, `make kernel-resources`. GPU lock before every run.
