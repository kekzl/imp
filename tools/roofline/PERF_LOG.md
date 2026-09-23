<!--
layer: L2
audience: kernel-devs
verified: 2026-09-23
commit: 679866b6
-->

# Kernel-limits perf log

Append-only, newest first: one entry per kernel iteration of the kernel-limits dispatch (hypothesis, before/after counters, e2e A/B, verdict).
Peaks: [`peaks/PEAKS.md`](peaks/PEAKS.md). Inventory: [`inventory/KERNELS.md`](inventory/KERNELS.md).

## 2026-09-23 · Q5_K MoE experts on the grouped IMMA kernel

[PROV: commit=55f0ee39+branch perf/moe-q5k-imma date=2026-09-23 hw=RTX5090 clocks=locked 2842/13801 MHz
kernel=nsys kernel means (inventory.sh) e2e=inventory/ab_bench.sh imp:ab-c2b8f047 vs imp:test, 5 pairs]

| Qwen3.6-35B-A3B UD-Q4_K_M | Before | After |
|---|---|---|
| Q5_K expert tensor per layer, pp512 | `dequant_q5k_kernel` of all 256 experts (791 us, 36.6 %) + FP16 grouped GEMM `Kernel2` (226 us, 11.3 %) | `mmq_imma_q5k_raw_kernel` 239 us, 17.6 % |
| `bench:pp` wall per rep, pp512 / pp4096 | 88 / 297 ms | 60 / 239 ms |
| e2e pp512 / pp4096 (medians) | 5957.09 / 13982.07 tok/s | 8660.70 (+45.4 %) / 17235.98 (+23.3 %) |
| e2e tg128 at 512 / 4096 context | 304.27 / 297.21 | 304.57 / 296.58 |
| PPL, 45k corpus, deterministic, 2 runs per arm | 6.5617, 6.5505 | 6.5437, 6.5502 |
| `degen_suite.py --skip-deterministic` (`runtime.max_batch_size=4`) | - | 50 checks, 0 FAIL |

| Check | Result |
|---|---|
| `MmqQ8Imma.MoeGroupedQ5K`, BM 32 and 128, N=192 partial tile, K=512 | NRMSE < 2e-2 vs the ggml Q5_K dequant formula |
| Mutation: fifth bit taken from bit `kb` instead of `2g + kb` | red |
| Raw-kernel dispatch | Q4_K / Q5_1 / Q5_K launch through one function-pointer path (the Q4_K and Q5_1 copies removed) |

Also measured, from #2091 (cuBLASLt inside the prefill capture): gemma-3-12b Q4_K_M pp4096 ran its captured offset-0 chunk on the WMMA fallback `gemm_fp16_kernel` (47.3 % of kernel time, 709 us/call); pp4096 7343.74 -> 8057.04 tok/s (3 pairs).

Found on the way, pre-existing on main: `imp-server` with Qwen3.6-35B-A3B UD-Q4_K_M fails to start at defaults (`recurrent state slot 8 could not be committed: the card cannot spare it above the allocator headroom`), card free (1774 MiB used); starts with `runtime.max_batch_size=4`.

## 2026-09-23 · Prefill graph: last chunk, FP8 KV, kept across context resets

[PROV: commit=c2b8f047+branch perf/prefill-graph-buckets date=2026-09-23 hw=RTX5090 clocks=locked 2842/13801 MHz
gaps=tools/roofline/inventory/host_gaps.py on nsys bench:pp e2e=inventory/ab_bench.sh imp:ab-c2b8f047 vs imp:test]

| Qwen3-30B-A3B-NVFP4 pp512 (FP8 KV), nsys `bench:pp` | GPU busy | idle | launches per forward |
|---|---:|---:|---:|
| eager (main) | 54.4 % | 45.6 %, 10-100 us gaps 29.29 ms of 91.99 | 1829 (`cudaLaunchKernel` 9.8 us each under nsys) |
| graph, re-captured per request (first cut) | 45.4 % | 54.6 %: 3 x `cudaGraphInstantiate` 43.74 ms | same host cost + instantiate |
| graph kept across `imp_context_reset` | 70.8 % | 29.2 %, one 26.51 ms instantiate in rep 1 | replay |

| e2e, 5 pairs, medians | main | this change |
|---|---:|---:|
| Qwen3-30B-A3B-NVFP4 pp512 | 20024.43 tok/s (17802..22973) | 24708.99 (24281..24976), 5/5 pairs up |
| same, rerun with cuBLASLt inside the capture (shipped) | 19687.98 | 23288.74 |
| Qwen3-14B-NVFP4 pp512 | 19509.69 (15674..24093) | 23695.60 (18603..24511) |
| Qwen3-8B-Q8_0 pp512, 7 pairs (shipped) | 12279..12569 | 12316..12468; pair deltas -1.73..+0.96 %, mean -0.58 %, 5 of 7 negative |
| tg128 (all three) | 382.08 / 175.42 / 299.37 | 382.30 / 175.49 / 298.96 |

| Change | Why |
|---|---|
| last (and only) offset-0 chunk captured, greedy event-sync path | a single-chunk prompt never took the graph before |
| FP8 KV capturable once every layer is calibrated; generation key on `reset_kv_calibration()` | calibration is the only D2H absmax sync, first prefill after warmup only |
| graph kept across `imp_context_reset`; dropped on an open capture (#874) or a LoRA switch | it bakes pool buffers only; dropping it re-instantiated every prefill |
| capture on the second consecutive sighting of (chunk_len, block count, calibration generation) | capture + instantiate cost more than one eager forward: one-off lengths stay eager, the cached graph stays |
| vision requests excluded | `n_vision_tokens` is a host arg and the graph outlives the request |
| cuBLASLt allowed inside the capture (as in the verify graphs) | the WMMA capture fallback changed greedy output on Qwen3-30B-A3B at token 11 (`DegenerationTest.PrefillGraphReplayMatchesEager` red); the hysteresis ran the shape eagerly first |

Scope: serial prefill (`imp-cli`, `imp_prefill*`). The server's default path (ragged prefill, prefix cache) never reaches it: repeat probe with `server.prefix_cache=false runtime.prefill_batch=false` captured (18 vs 16 captures) and returned byte-identical content in all 4 requests, graph on vs off. Next lever: length-bucketed graphs for the ragged path.

## 2026-09-23 · FA2 for gpt-oss: head_dim 64 + learned sinks

[PROV: commit=2889d8b2+branch perf/fa2-hd64-sinks date=2026-09-23 hw=RTX5090 clocks=locked 2842/13801 MHz
kernel=nsys kernel means (inventory.sh) ncu=ncu_cell.sh full e2e=inventory/ab_bench.sh imp:ab-b3625fd0 vs imp:test, 5 pairs]

| | Before | After |
|---|---|---|
| gpt-oss-20b pp4096 attention | `fmha_sm120_kernel` (tiled WMMA, only sink-capable tier) 1571 us/call, 46.6 % | `fmha_sm120_fa2_kernel<128,64>` 288 us/call, 13.8 % |
| gpt-oss-20b pp512 attention | cuBLAS + `causal_softmax_inplace_kernel` (78 us/call, 7.7 %) | FA2 31 us/call, 4.1 % |
| e2e pp4096 / pp512 (medians) | 19766.31 / 16070.52 tok/s | 36412.67 (+84.2 %) / 20849.72 (+29.7 %) |
| e2e tg128 at 512 / 4096 context | 388.11 / 333.55 tok/s | 386.47 / 331.37 (4 of 5 pairs -0.2..-1.0 % at 4096) |
| FA2 kernel (ncu, pp4096) | - | 48.6 % of the measured FP16 f32-acc tensor peak (119.5 of 246 TFLOPS), math_pipe_throttle 3.82 cycles/issue, occupancy 16.3 % (151 regs, 1 CTA/SM) |

| Accuracy vs fp64 (max rel err, RMS-normalised, sinks on) | cuBLAS | WMMA FMHA | FA2 f16-acc | FA2 f32-acc (shipped) |
|---|---:|---:|---:|---:|
| 48 x 48, 16/4 heads | 5.30e-03 | 5.59e-03 | 5.16e-03 | 3.49e-03 |
| 777 x 777, 64/8 heads | - | 2.00e-02 | 1.81e-02 | 1.49e-02 |
| chunk 2: 300 queries at q_offset 1024, full | - | 6.18e-03 | 3.64e-02 | 5.96e-03 |
| chunk 2, SWA 128 | - | 4.21e-03 | 1.50e-02 | 3.87e-03 |
| 300 x 300, input amplitude 8 (no QK norm) | 3.41e-02 | 2.87e-03 | - | 1.76e-03 |

| Check | Result |
|---|---|
| f16 QK/PV accumulators (the hd=128 defaults) | refuted for hd=64: chunk-2 error 3.6e-2 vs 6e-3; hd=64 uses f32 accumulators |
| gpt-oss PPL, 45k corpus, deterministic, chunk 1024 | cuBLAS 308.48, WMMA FMHA 297.61, FA2 267.28 (f16-acc) / 267.29 (f32-acc); default chunking FA2 256.22 vs main 312.50. No verdict: layer dump diff WMMA vs FA2 grows 0.0003 -> 0.25 relative by layer 15 while each attention block adds at most +0.0033 (MoE routing amplifies); chunk size alone moves PPL 1.3-4 % |
| `degen_suite.py` gpt-oss | 41 PASS, 6 FAIL, identical on main (constrained JSON empty content x4, anthropic default thinking block x2: pre-existing) |
| Mutation: sink seed without `/ s_eff` | `GptOssSinkRef.Fa2Hd64SinkMatchesReference` red |

## 2026-09-23 · RoPE prefill: angle once per (token, pair), bit-identical

[PROV: commit=b3625fd0+branch perf/prefill-bitexact-rewrites date=2026-09-23 hw=RTX5090 clocks=locked 2842/13801 MHz
kernel=nsys kernel means (inventory.sh) e2e=inventory/ab_bench.sh imp:ab-144ff6aa vs imp:test, 5 pairs]

| | Before | After |
|---|---|---|
| Launch | grid (tokens, max_heads), one block per (token, head), FP64 `pow` + double `rope_sincos` per head | grid (tokens, head groups), angle computed by row 0 into smem once per block; head groups only when tokens < 340 |
| gpt-oss-20b MXFP4 pp4096, `rope_forward_kernel<half>` | 669.1 us/call, 16.6 % | 21.0 us/call, 0.6 % |
| Qwen3-14B-NVFP4 pp4096 | 76.0 us/call, 3.8 % | 13.9 us/call, 0.7 % |
| gpt-oss-20b e2e pp4096 / tg128 (medians) | 19975.57 / 335.84 tok/s | 23520.90 (+17.7 %) / 336.14 tok/s |
| Qwen3-14B-NVFP4 e2e pp4096 / tg128 | 25611.10 / 163.94 tok/s | 26380.04 (+3.0 %) / 164.13 tok/s |
| Deterministic PPL, 45k corpus | gpt-oss 312.4986, Qwen3-14B 10.0649 | identical |

| Refuted on the way | Measured |
|---|---|
| One block per token at every M | gpt-oss tg128 335.41 -> 332.41 (-0.9 %, 5/5 pairs): decode runs `rope_forward` at M = 1 (no QK norm), 64 heads serial in one block |

Qwen3.6-35B (MRoPE path) is no judge: deterministic PPL on `ppl_4k.txt` varies inside one image (8.1367 / 8.1957 on main, 8.1670 / 8.1122 here); MRoPE unit tests pass.

## 2026-09-23 · Mamba2 SSM scan: register-resident, bit-identical

[PROV: commit=679866b6+branch perf/ssm-scan-register date=2026-09-23 hw=RTX5090 clocks=locked 2842/13801 MHz
kernel=nsys kernel means (inventory.sh) ncu=ncu_cell.sh full e2e=inventory/ab_bench.sh imp:ab-144ff6aa vs imp:test]

| Iteration | Change | us per 2048-token call (Nemotron pp4096) | ncu |
|---|---|---:|---|
| 0 | legacy `ssm_scan_kernel`: 64 CTAs (one per head), h_state read+written in global memory per token, 2 barriers per token | 4991 | - |
| 1 | state in registers, grid (heads, head_dim / 8) = 512 CTAs, smem tree -> `shfl_down` in the same order | 760 | issue 37.5 %, long_scoreboard 4.20 cycles/issue |
| 2 | 4-token register prefetch ring | 790 | refuted: no gain |
| 3 | one uniform snapshot branch per token, gate computed on all lanes (predicated store) | 696 | issue 50.1 %, selected 1.00 / not_selected 0.98 |
| 4 | dt/a_bar once per warp for 32 tokens + shuffle, ring without copies, 32-bit offsets, hot path without real_n/snapshot checks, paired FP16 rounding | 496 | issue 47.7 %, 3 warps/scheduler, stalls wait 0.88 / long_scoreboard 0.81 |

[PROV: e2e=tools/roofline/inventory/ab_bench.sh, alternating pairs, one model per process, clocks locked 2842/13801 MHz]

| Check | Result |
|---|---|
| Bit identity vs legacy (y, h_state, snapshot; FP16/FP32 state, gate on/off, padded real_n, snapshot row, hd 64/128, M = 1/12/40/257/300/777) | equal (`SSMScanTest.RegisterScanBitIdenticalToLegacy`); dropping the FP16 state rounding turns it red |
| FMA contraction | pinned with `__fmaf_rn`/`__fmul_rn` to the legacy SASS form `fma(x*dt, b, a_bar*h)`; the compiler's own choice broke bit identity in iteration 4 |
| vs double reference (FP32 state, 64 tokens) | max rel err < 4e-3 (`RegisterScanMatchesDoubleReference`) |
| Nemotron-3-Nano PPL, 45k corpus, deterministic | 9.3898 both arms |
| e2e A/B, 5 pairs, medians (Nemotron-3-Nano) | pp4096 13149 -> 39605 tok/s (+201 %), tg128@4k 391.24 -> 396.73 (+1.4 %), pp512 9711 -> 16097 (+65.8 %), tg128 399.50 -> 404.78 (+1.3 %) |
| e2e A/B, 3 pairs (Nemotron-3.5-Lightning) | pp4096 12692 -> 36524 (+188 %), tg128@4k 366.86 -> 371.50 (+1.3 %) |
| Control, 5 pairs (Qwen3-8B Q8_0, no SSM) | pp512 12260 -> 12267, tg128 299.21 -> 299.38 |

Verdict: kept. Remaining limit: 2048 warps of parallelism in the bit-identical formulation; the next step is the chunked SSD form on tensor cores, which gives up bit identity with the per-token FP16 state rounding.
