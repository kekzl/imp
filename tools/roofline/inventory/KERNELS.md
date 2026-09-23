<!--
layer: L2
audience: kernel-devs
verified: 2026-09-23
commit: 679866b6
-->

# Kernel inventory (sm_120a, full benchmark matrix)

Hot kernels per workload, ranked by max time share over all cells; input to the kernel-limits dispatch.
Peaks it divides by: [`../peaks/PEAKS.md`](../peaks/PEAKS.md).

[PROV: commit=679866b6+branch perf/kernel-limits-dispatch date=2026-09-23 hw=RTX5090 clocks=locked 2842/13801 MHz
harness=tools/roofline/inventory/inventory.sh (nsys 2026.1.3, graphs ON, --cuda-graph-trace=node)
matrix=matrix.tsv (13 models x pp512/pp4096/tg128/tg128_ctx8k), Nemotron cells re-run after the SSM scan change]

## Method

| Rule | Value |
|---|---|
| Phase | NVTX `bench:pp` / `bench:tg` range of `imp-cli --bench` (`tools/imp-cli/mode_bench.cpp`), kernel joined to its launching call by correlationId |
| Share | kernel time / kernel-time sum of that phase in that cell |
| Decode arms | `speculative.ngram=false` (the gate regime) |
| Rank | max share over all 51 cells; `cells >= 1 %` = breadth |
| Bound resource | only from ncu counters (`ncu_cell.sh` + `ncu_summary.py`); empty = not profiled yet |
| Regenerate | `inventory.sh [model_key]`, then `python inventory_report.py out out/kernels.json 0.05` in `python:3.13-slim` |

## Top 30 kernels

| # | Kernel | max share | cell | us/call | calls/rep | cells >= 1 % | grid/block | regs | smem |
|---:|---|---:|---|---:|---:|---:|---|---:|---:|
| 1 | `imp::<unnamed>::mmq_imma_kernel<(int)128, (bool)0, (bool)0, (bool)0>` | 60.5 % | q8-8b__pp512 | 135.7 | 180 | 7 | 8x4x1/256x1x1 | 199 | 44032 |
| 2 | `cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblem...` | 56.7 % | nvfp4-q3-30b__pp512 | 68.1 | 144 | 10 | 170x1x1/384x1x1 | 167 | 89088 |
| 3 | `cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, ...` | 56.1 % | nvfp4-14b__pp4096 | 221.7 | 400 | 10 | 16x40x1/384x1x1 | 149 | 88064 |
| 4 | `imp::<unnamed>::gemm_fp16_kernel<(int)128, (int)2>` | 47.3 % | q4k-gemma3-12b__pp4096 | 709.8 | 336 | 2 | 16x16x1/128x1x1 | 243 | 36864 |
| 5 | `imp::mmq_imma_q4k_raw_kernel<(int)128, (bool)0>` | 47.0 % | q4k-q3-30b__pp512 | 189.7 | 120 | 6 | 6x4x128/256x1x1 | 229 | 41984 |
| 6 | `imp::gemv_nvfp4_gate_up_fused_mr_kernel<(int)8>` | 42.4 % | nvfp4-14b__tg128 | 62.1 | 5120 | 8 | 4352x1x1/256x1x1 | 46 | 0 |
| 7 | `nvjet_sm120_hsh_mma_96x256x64_2_48x64x64_tmaAB_alignCD4_bz_TNNN` | 41.7 % | q4k-gemma3-12b__pp512 | 280.2 | 96 | 2 | 1280x1x1/128x2x1 | 255 | 91136 |
| 8 | `cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, ...` | 39.1 % | nvfp4-14b__pp512 | 91.6 | 80 | 6 | 170x1x3/384x1x1 | 156 | 88064 |
| 9 | `imp::fmha_sm120_kernel<(int)64, (int)64>` | 39.1 % | mxfp4-gptoss-20b__pp4096 | 1565.9 | 48 | 1 | 32x64x1/32x8x1 | 91 | 49664 |
| 10 | `nvjet_sm120_hhh_mma_256x128x64_2_64x64x64_tmaAB_alignCD4_bz_TNNN` | 38.6 % | q6k-14b__pp4096 | 1064.3 | 160 | 1 | 16x68x1/64x4x1 | 255 | 99328 |
| 11 | `imp::dequant_q5k_kernel` | 36.7 % | q4k-q36-35b__pp512 | 791.4 | 37 | 2 | 1048576x1x1/256x1x1 | 28 | 0 |
| 12 | `imp::dequant_q6k_v2_kernel` | 36.3 % | q6k-14b__pp512 | 94.0 | 280 | 8 | 348160x1x1/128x1x1 | 24 | 0 |
| 13 | `imp::fmha_sm120_fa2_kernel_2cta<(int)128, (int)128, (bool)1, (bool)1, (int)64, (bool)1,...` | 33.4 % | nvfp4-q3-30b__pp4096 | 348.5 | 96 | 6 | 16x32x1/32x8x1 | 128 | 34816 |
| 14 | `nvjet_sm120_hhh_mma_256x112x64_2_64x56x64_tmaAB_alignCD4_bz_TNNN` | 31.0 % | q6k-14b__pp512 | 280.7 | 80 | 3 | 5x68x1/64x4x1 | 255 | 95232 |
| 15 | `imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)3>>` | 30.8 % | q4k-gemma3-12b__tg128 | 45.9 | 6144 | 2 | 15360x2x1/128x1x1 | 38 | 16 |
| 16 | `imp::<unnamed>::mmq_imma_kernel<(int)128, (bool)1, (bool)0, (bool)0>` | 30.5 % | q8-8b__pp4096 | 605.6 | 144 | 5 | 32x16x1/256x1x1 | 199 | 44032 |
| 17 | `imp::gemv_nvfp4_residual_kernel` | 30.0 % | nvfp4-14b__tg128 | 22.0 | 10240 | 16 | 5120x1x1/128x1x1 | 42 | 16 |
| 18 | `imp::paged_attention_splitk_pipeline_kernel<(int)64>` | 29.0 % | mxfp4-gptoss-20b__tg128_ctx8k | 39.9 | 3072 | 2 | 1x64x6/256x1x1 | 56 | 3072 |
| 19 | `imp::<unnamed>::ssm_scan_reg_kernel<(bool)1, (bool)1, (int)16, (int)8>` | 25.8 % | nvfp4-nemotron-30b__pp4096 | 495.7 | 46 | 4 | 64x8x1/128x1x1 | 96 | 0 |
| 20 | `imp::gemv_nvfp4_moe_decode_mr_kernel<(int)8>` | 24.5 % | nvfp4-nemotron-30b__tg128 | 12.6 | 5888 | 10 | 1392x1x1/256x1x1 | 40 | 0 |
| 21 | `imp::gemv_fp8_e4m3_kernel<(bool)1>` | 24.0 % | nvfp4-q36-35b__tg128 | 11.4 | 7680 | 6 | 1544x1x1/256x1x1 | 38 | 0 |
| 22 | `imp::gemv_nvfp4_moe_gate_up_mr_kernel<(int)8>` | 23.4 % | mxfp4-gptoss-20b__tg128 | 24.3 | 3072 | 8 | 1440x2x1/256x1x1 | 40 | 0 |
| 23 | `nvjet_sm120_hhh_mma_96x256x64_2_48x64x64_tmaAB_alignCD4_bx_TNNN` | 21.5 % | q6k-14b__pp4096 | 1185.0 | 80 | 1 | 432x1x1/128x2x1 | 255 | 91136 |
| 24 | `nvjet_sm120_hsh_mma_192x128x64_2_96x32x64_tmaAB_alignCD4_splitK_TNNN` | 21.1 % | q4k-gemma3-12b__pp512 | 284.5 | 48 | 1 | 32x3x2/128x2x1 | 255 | 82944 |
| 25 | `imp::paged_attention_splitk_fp8_tile_gqa_kernel<(int)128>` | 21.1 % | nvfp4-q3-30b__tg128_ctx8k | 14.4 | 6144 | 10 | 1x4x85/256x1x1 | 80 | 9216 |
| 26 | `nvjet_sm120_hsh_mma_256x128x64_2_64x64x64_tmaAB_alignCD4_bz_TNNN` | 20.6 % | q4k-gemma3-12b__pp4096 | 1082.1 | 96 | 1 | 16x60x1/64x4x1 | 255 | 99328 |
| 27 | `imp::mmq_imma_q51_raw_kernel<(int)128, (bool)0>` | 19.9 % | q4k-gemma4-26b__pp512 | 278.6 | 30 | 2 | 22x4x128/256x1x1 | 194 | 41984 |
| 28 | `imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)3>, (bool)0>` | 18.8 % | q4k-gemma3-12b__tg128 | 11.2 | 15360 | 4 | 3840x1x1/128x1x1 | 39 | 16 |
| 29 | `imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2>` | 17.1 % | mxfp4-gptoss-20b__tg128 | 426.3 | 128 | 5 | 12568x1x1/256x1x1 | 48 | 3600 |
| 30 | `imp::gemv_nvfp4_kpar_kernel` | 17.0 % | nvfp4-gemma4-26b__tg128 | 4.8 | 16000 | 13 | 2816x1x1/128x1x1 | 42 | 16 |

## Classified (ncu, locked clocks)

| Kernel | Cell | Before | After | Bound by (ncu) | Status |
|---|---|---:|---:|---|---|
| `ssm_scan_kernel` -> `ssm_scan_reg_kernel<1,1,16,8>` | nemotron pp4096 | 4991 us/call, 77.9 % | 495.7 us/call, 25.8 % | issue + latency: 47.7 % issue slots, 3 warps/scheduler (2048 warps total), stalls wait 0.88 / long_scoreboard 0.81 / short_scoreboard 0.64 | bit-identical to legacy; next step needs the chunked SSD form (tensor cores), which gives up bit-identity |

## Launch/host gaps (GPU idle inside the phase >= 20 %)

| Cell | phase wall ms/rep | kernel sum ms/rep | GPU idle |
|---|---:|---:|---:|
| nvfp4-q3-30b pp512 | 40.6 | 17.3 | 57.4 % |
| nvfp4-q36-35b pp512 | 38.5 | 22.4 | 41.8 % |
| q4k-gemma3-12b pp512 | 110.9 | 64.6 | 41.7 % |
| mxfp4-gptoss-20b pp512 | 39.8 | 24.3 | 39.0 % |
| nvfp4-nemotron-30b pp512 | 27.9 | 18.4 | 34.1 % |
| nvfp4-q36-35b pp4096 | 185.5 | 125.5 | 32.4 % |
| nvfp4-gemma4-26b pp4096 | 145.8 | 105.8 | 27.7 % |
| nvfp4-gemma4-26b pp512 | 22.1 | 16.9 | 23.6 % |

## Open cells

| Cell | Result |
|---|---|
| q4k-gemma4-26b tg128_ctx8k | no decode: KV pool 7808 tokens < 9360 requested, StreamingLLM auto-enabled, `imp_decode_step: engine produced no token in 8 steps` |
