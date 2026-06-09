# `gemv_nvfp4` MoE-decode roofline — structural-ceiling analysis (issue #600)

**Date:** 2026-06-09 · **Resolves:** [#600](https://github.com/kekzl/imp/issues/600)
(`gemv_nvfp4`: 30 % roofline on `nvfp4-moe/tg256`) · **Verdict:** documented
structural ceiling (the acceptance-criteria alternative to >70 %). No code change.

## The finding

From the 2026-06-07 roofline run (`e66cfa45-dirty_20260607_042900`), the
`gemv_nvfp4` kernel class on the MoE decode cell `nvfp4-moe/tg256`
(Qwen3-30B-A3B-NVFP4-Modelopt) reaches only **30.2 % of the DRAM roofline** at
49.9 % of the decode window — versus **61.4 %** dense (`nvfp4-dense/tg256`,
Qwen3-14B) and **58.8 %** Q8-via-NVFP4-cache. This document explains why that
gap is structural, not a tuning deficit.

## Model shape (why MoE decode is different)

Qwen3-30B-A3B: `hidden=2048`, `num_experts=128`, `top_k=8`,
`moe_intermediate=768`, 48 layers. Per decode token, per layer, the NVFP4 MoE
path issues exactly **two** fused GEMV launches
(`src/exec/executor_forward_moe_batch.cu:831`):

1. `gemv_nvfp4_moe_gate_up_fused` — gate+up, **K=2048** (`n_mb=128`), N=768/expert
2. `gemv_nvfp4_moe_swiglu_decode` — SwiGLU+down fused, **K=768** (`n_mb=48`), N=2048/expert

The dense FFN of Qwen3-14B, by contrast, has `K=5120` (`n_mb=320`) and
N=17408 — an order of magnitude more work per GEMV.

## Measured per-kernel breakdown (ncu, committed raw of the audit run)

| Kernel (NR=8) | K | n_mb | DRAM moved | dur | achieved BW | %-roofline | **waves/SM** | occ |
|---|---|---|---|---|---|---|---|---|
| MoE gate_up `mr<8>` | 2048 | 128 | 14.2 MB | 19.2 µs | 740 GB/s | **41.3 %** | **1.51** | 83 % |
| MoE swiglu/down `mr<8>` | 768 | 48 | 7.1 MB | 18.8 µs | 378 GB/s | **21.1 %** | **2.01** | 85 % |
| *dense* gate_up `mr<8>` (14B) | 5120 | 320 | 100.7 MB | 86.9 µs | 1159 GB/s | **64.7 %** | **5.12** | 74 % |
| *dense* down `residual` (14B) | 17408 | — | 32.5 MB | 30.9 µs | 1052 GB/s | **58.7 %** | **4.30** | 48 % |

Source: `tools/roofline/history/raw/e66cfa45-dirty_20260607_042900/{nvfp4-moe,nvfp4-dense}_tg256_r0_full.ncu_raw.csv.gz`.

## Root cause — two compounding small-tensor effects

**Occupancy is not the limiter.** The MoE kernels run at *higher* achieved
occupancy (83–85 %) than the dense ones that reach 59–65 % roofline (48–74 %).
The DRAM gap is explained by two things, both downstream of "MoE experts at
batch=1 are tiny":

1. **Too few waves → fill/drain-dominated, never steady-state.** A decode token
   routes to only `top_k=8` of 128 experts, so the grid is small: gate_up =
   1536 blocks (**1.51 waves**), down = 2048 blocks (**2.01 waves**). The dense
   FFN launches 4.3–5.1 waves and *reaches* steady-state DRAM saturation. At
   1.5–2 waves a large fraction of the kernel's life is grid fill + tail drain,
   so the *average* BW sits far below the roofline even though the active SMs
   are busy. Per-launch byte volume is correspondingly tiny — 7–14 MB vs the
   dense 33–100 MB — so there is nothing to amortize the ramp against.

2. **Tiny K → near-zero memory-level parallelism per warp.** Each warp owns one
   output row and strides its 32 lanes over `n_mb` micro-blocks:
   `loads/lane = n_mb/32`. Dense gets **10** (n_mb=320); MoE gate_up gets **4**;
   MoE down gets **1.5** (n_mb=48). With ~1–2 dependent 8-byte loads then a warp
   reduction + write, the down kernel is pure latency, not bandwidth — which is
   exactly why it bottoms out at 21 %.

## The one available lever is saturated (fresh measurement, 2026-06-09)

`moe.mr_nr` (rows-per-block, env `IMP_MOE_MR_NR`, values 4/8/16/32) is the knob
that trades wave depth against per-block work. Decode tg256 sweep, isolated
process per config, discarded warmup, 3×10-rep trials, Qwen3-30B-A3B-NVFP4,
healthy host (mem 13801 MHz, up to 375 W during load):

| `moe.mr_nr` | tg256 median tok/s | Δ vs default |
|---|---|---|
| 4  | 341.7 | **+0.9 %** |
| **8 (default)** | 338.7 | — |
| 16 | 333.0 | −1.7 % |
| 32 | 306.5 | −9.5 % |

Tool: `tools/analysis/moe_mr_nr_ab.sh`.

NR=4 *doubles* the wave count (gate_up 1.5→3, down 2→4) yet buys only **+0.9 %**
end-to-end — itself the decisive evidence that wave depth is **not** the
dominant lever: the work is simply too small per warp for more waves to help.
Larger NR (fewer, fatter blocks) regresses monotonically. The knob is at its
optimum; NR=4 is consistently but marginally best (within the 3 % decode gate),
so the global default stays **8** (validated on only one of five NVFP4-MoE
models; the knob remains for per-deployment tuning).

## Why >70 % is not reachable on the existing decode path

The per-token MoE-GEMV byte budget is **~1022 MB** (48 layers × (14.2+7.1) MB).
Floors: 571 µs @ 100 % roofline, 815 µs @ 70 %, ~1889 µs at the measured 30.2 %.
Even hypothetically matching the dense 61 % would need each warp to stream many
back-to-back cache lines — but K is fixed by the architecture (768/2048), so the
loads-per-lane ceiling (1.5/4) cannot be raised without assigning multiple rows
per warp, which removes the parallelism that produces the waves. The two
objectives (more MLP per warp vs more waves) are in direct tension, and the NR
sweep shows the optimum of that trade is already ~where we sit.

This is the canonical batch-1 MoE wall: MoE shines under batching (many tokens
amortize each expert's weight read), but single-stream decode touches only
`top_k` small experts, producing small, shallow, latency-bound grids. It is the
same structural story already recorded for the decode frontier
(`docs/MISSION_JOURNAL.md`; memory `decode_gemv_roofline_sweep`,
`qwen36_35b_decode_gap_structural`).

## Caveats on the 30.2 % number itself

- **L1 hit is 96–97 %** for these kernels (artifact (b) in
  `docs/audit/roofline_2026_06_07.md`): the shared activation vector `x` is
  re-read across all warps and served from L1, so a large share of *requests*
  never reach DRAM. The DRAM roofline therefore understates effective
  utilization; the weights (which dominate *bytes*) are the genuine DRAM
  traffic and set the 30 % figure.
- `attn_decode_paged` on the same cell reads only ctx≈320 KV → its 1.4 %
  roofline is a workload artifact, already excluded as a lever in the audit.

## Conclusion

`gemv_nvfp4` at 30 % roofline on `nvfp4-moe/tg256` is a **structural ceiling**
of batch-1 MoE decode (shallow grids + tiny K), not a kernel or tuning defect.
The only available knob (`moe.mr_nr`) is saturated. No code change is warranted;
this analysis is the documented resolution per the issue's acceptance criteria.
Decode throughput for this model already reflects the +54–80 % win from the
shipped per-expert `gemv_nvfp4_moe` path (`runtime/config.h: moe.nvfp4_moe_decode`,
default on) over CUTLASS grouped-GEMM at M=1.
