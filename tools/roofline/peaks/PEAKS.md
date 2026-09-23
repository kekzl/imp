<!--
layer: L2
audience: kernel-devs
verified: 2026-09-23
commit: 679866b6
-->

# Measured sm_120a limits (RTX 5090)

Achievable peaks for the kernel dispatch: every "% of peak" in `KERNELS_INVENTORY.md` divides by a row here, never by the datasheet.
Harness: `tools/roofline/peaks/run_peaks.sh [mem|tc|simt|launch]` (builds `imp-peaks` in `imp:toolchain`, refuses a busy card, writes `out/peaks_<utc>.{json,log,clocks.csv}`).

[PROV: commit=679866b6 date=2026-09-23 hw=RTX5090 driver=616.92 cuda=13.4.1 (nvcc V13.4.59)
clocks=locked from Windows admin shell (nvidia-smi -lgc/-lmc), idle 2842 MHz SM / 13801 MHz mem;
1372 of 1457 100-ms samples at 2842 MHz, the rest 2797-2835 (FFMA at the 575 W power limit)
power_limit=575 W runs=peaks_20260923T044423Z (all suites), peaks_20260923T044701Z (mem, smem fix)]

## Method

| Rule | Value |
|---|---|
| Warmup | >= 1.5 s busy per kernel before timing (0.5 s per extra launch shape) |
| Statistic | median of 5-7 timed windows; `spread` = (max - min) / median |
| Launch shape | best of 5 grid shapes (memory) or 6 warp/CTA shapes (tensor cores), shape in the row |
| DRAM working set | 2 GiB (21x L2) |
| Clock per row | NVML sample after each timed window, in the JSON |
| Per-clock column | value / (170 SMs x 2.842 GHz) |

## Memory

| Resource | Measured | Per clock | vs theory | Shape |
|---|---:|---:|---:|---|
| DRAM read | 1692-1698 GB/s | - | 94.4-94.8 % of 1792 | 2x512 / 16x128 |
| DRAM write | 1670-1674 GB/s | - | 93.2-93.4 % | 4x256 / 1x1024 |
| DRAM copy (read + write bytes) | 1488-1490 GB/s | - | 83.0-83.1 % | 4x256 / 1x1024 |
| TMA `cp.async.bulk` 16 KiB, DRAM | 1686-1697 GB/s | - | = LDG read | 1-2 CTA/SM, 3 stages |
| L2 read, working set <= 96 MB | 6840-7163 GB/s | 2.4-2.5 KB/clk | - | ld.global.cg |
| TMA 16 KiB, 8 MB working set | 6981-7192 GB/s | - | = LDG L2 | 1-2 CTA/SM |
| SMEM `ld.shared.v4` per SM | 349.6 GB/s | 123 B/clk/SM | 96 % of 128 | 1x1024/SM |

L2 size effect (read, one launch = 2 GiB of traffic):

| Working set | 8-96 MB | 112 MB | 128 MB | 192 MB | 256 MB | 512 MB |
|---|---:|---:|---:|---:|---:|---:|
| GB/s | 6840-7163 | 5636 | 5329-5364 | 1902-1907 | 1872 | 1775 |

Rule: a working set <= 96 MB reads at ~7.1 TB/s (4.2x DRAM); an isolated kernel bench without a > 192 MB rotating ring measures L2.

## Tensor cores (dense `mma.sync`, 4 independent accumulators per warp)

| Format (PTX kind, accumulate) | SASS | TOPS | ops/SM/clk | % of the per-clock rate |
|---|---|---:|---:|---:|
| NVFP4 `mxf4nvf4` ue4m3 4X, f32 | `OMMA.SF.16864` | 1898.7 | 3930 | 96 % of 4096 |
| MXFP4 `mxf4` ue8m0 2X, f32 | `OMMA.SF.16864` | 1904.4 | 3942 | 96 % of 4096 |
| MXFP8 `mxf8f6f4` ue8m0 1X e4m3, f32 | `QMMA.SF.16832` | 939.6 | 1945 | 95 % of 2048 |
| FP8 e4m3 `f8f6f4`, f32 | `QMMA.16832.F32` | 490.5 | 1015 | 99 % of 1024 |
| FP8 e4m3, f16 | `QMMA.16832.F16` | 985.0 | 2039 | 99.6 % of 2048 |
| INT8, s32 | `IMMA.16832` | 969.3 | 2006 | 98 % of 2048 |
| FP16, f16 | `HMMA.16816.F16` | 490.7 | 1016 | 99 % of 1024 |
| FP16, f32 | `HMMA.16816.F32` | 246.1 | 509 | 99 % of 512 |
| BF16, f32 | `HMMA.16816.F32.BF16` | 245.4 | 508 | 99 % of 512 |
| TF32, f32 | `HMMA.1688.F32.TF32` | 122.7 | 254 | 99 % of 256 |

| Consequence | Rule |
|---|---|
| FP8 with f32 accumulate | block-scaled `kind::mxf8f6f4` (unit ue8m0 scales = 0x7f) runs 1.92x plain `kind::f8f6f4` f32: every compute-bound FP8 `mma.sync` path is a candidate |
| NVFP4 peak | 1899 TOPS at 2842 MHz = 57 % of the 3354 datasheet; the per-clock rate is 4096 ops/SM |
| f32 accumulate on FP16/BF16 | 1/2 of f16 accumulate |

## SIMT and SFU

| Op | Measured | per SM per clk | Note |
|---|---:|---:|---|
| FFMA fp32 | 104.3 TFLOPS | 220 flops | 86 % of 256; SM clock 2790 MHz, 563 W: power-limited |
| HFMA2 fp16x2 | 120.3 TFLOPS | 249 flops | same flop rate as FFMA, not 2x |
| HFMA2 bf16x2 | 120.2 TFLOPS | 249 flops | |
| MUFU ex2 | 7.60 Tops | 15.7 | 16 per SM per clk |
| MUFU rsqrt | 7.61 Tops | 15.8 | |
| MUFU tanh | 7.41 Tops | 15.3 | |

## Launch overhead (empty kernel, 256-kernel chain, per kernel)

| Mode | grid=1 | grid=170 |
|---|---:|---:|
| Plain stream launch | 5.80 us | 5.63 us |
| PDL chain (stream launch) | 6.09 us | 6.72 us |
| CUDA graph node | 0.68 us | 0.68 us |
| CUDA graph node + PDL edge | 0.47 us | 0.47 us |

Rule: outside a graph, WSL2/WDDM stream launches cost ~5.7 us each (8.4x a graph node); a hot path outside a graph is launch-bound below ~6 us of kernel time.

## Reproduce

```bash
~/.claude/skills/gpu-stats/gpu-busy-check.sh      # exit 0 = free
# Windows admin PowerShell: nvidia-smi -lgc 2850,2850; nvidia-smi -lmc 13801,13801   (reset: -rgc / -rmc)
tools/roofline/peaks/run_peaks.sh                 # all suites, ~4 min
```
