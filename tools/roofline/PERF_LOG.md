<!--
layer: L2
audience: kernel-devs
verified: 2026-09-23
commit: 679866b6
-->

# Kernel-limits perf log

Append-only, newest first: one entry per kernel iteration of the kernel-limits dispatch (hypothesis, before/after counters, e2e A/B, verdict).
Peaks: [`peaks/PEAKS.md`](peaks/PEAKS.md). Inventory: [`inventory/KERNELS.md`](inventory/KERNELS.md).

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
