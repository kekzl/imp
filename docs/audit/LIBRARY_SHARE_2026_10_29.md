# cuBLAS/cuBLASLt and CUTLASS kernel share per phase (ROADMAP row 123)

Source: `ghcr.io/kekzl/imp:latest` v0.59.2 (revision 9aac23ed), RTX 5090, 2026-10-29.
Harness: `IMAGE=ghcr.io/kekzl/imp:latest tools/roofline/inventory/inventory.sh` (nsys 2026.1.3, graphs ON, `--cuda-graph-trace=node`, decode arms `speculative.ngram=false`), 13 models x pp512/pp4096/tg128/tg128_ctx8k = 52 cells.
Table: `python library_share.py <out> 2` (`tools/roofline/inventory/library_share.py`) in `python:3.13-slim`. Phase = NVTX `bench:pp` / `bench:tg`; share = class ns / phase kernel sum.
Class: cuBLAS/Lt = `nvjet_*`, `cutlass3x_*` (cuBLASLt-internal), `*xmma*`, `gemv2*`, `splitKreduce`; CUTLASS = `cutlass::` templates from `src/compute/gemm_cutlass_*.cu`.

## Result

| path | models | pp512 share | pp4096 share | library |
|---|---|---:|---:|---|
| decode, every model | all 13, tg128 + tg128_ctx8k | 0.0 % | 0.0 % | none |
| NVFP4 dense | Qwen3-14B, Qwen3.8-27B | 72.1 / 54.6 % | 58.2 / 52.6 % | CUTLASS below M 2048, cuBLASLt block-scaled from M 2048 (`gemm.nvfp4_cublaslt_min_m`) |
| NVFP4 MoE | Qwen3-30B-A3B, Qwen3.6-35B, Gemma-4-26B, Nemotron-3-Nano | 57.6-67.2 % CUTLASS + 1.4-9.0 % Lt | 27.4-38.8 % CUTLASS + 5.0-26.8 % Lt | CUTLASS grouped, cuBLASLt for dense layers |
| MXFP4 | gpt-oss-20b | 51.6 % | 36.4 % | CUTLASS |
| GGUF Q6_K | Qwen3-14B | 58.8 % | 74.9 % | cuBLASLt FP16 (`nvjet_*_hhh`) |
| GGUF Q4_K | Gemma-3-12B, Qwen3-30B-A3B, Qwen3.6-35B, Gemma-4-26B | 2.1-13.4 % | 5.9-13.8 % | cuBLASLt FP16 |
| GGUF Q8_0 | Qwen3-8B | 0.0 % | 0.0 % | none |

- Row 123 cannot move decode tok/s: the Qwen3.8-27B tg128 step (94.24 tok/s here) runs 0 library kernels; `gemv_nvfp4_*` own kernels are 84.7 % of it, the FP8 LM head 8.8 %.
- Prefill order by library ms: Q6_K FP16 GEMM, NVFP4 dense cuBLASLt (M >= 2048), NVFP4 dense CUTLASS (M < 2048), MoE CUTLASS grouped, MXFP4 CUTLASS.
- Raw per-cell JSONs (3.2 MB) are not committed: rerun the harness.

## Step 1: own dense NVFP4 GEMM (`src/quant/nvfp4_gemm_tile.cu`)

Isolated: `test-quant --gtest_filter='*Nvfp4GemmTile*'` (4 weight copies rotate, 40 timed calls). In situ: `MODEL=Qwen3.8-27B-NVFP4-vllm tools/analysis/prefill_kernel_ab.sh gemm.nvfp4_tile_min_m 0 1 'gemm_nvfp4_tile|device_kernel|cutlass3x|nvjet' <pp> <pairs>` (measured with the earlier bool key `gemm.nvfp4_tile` false/true, the same dispatch as min_m 0/1), build-dev at a23d0bc3 + branch.

| Design step | 4096x17408x5120 us (CUTLASS 591.3, cuBLASLt 517.3) |
|---|---:|
| sync cp.async ring, LDS fragments | 1029.6 |
| + ldmatrix | 949.4 |
| BK 256 x 2 stages | 911.6 |
| producer warp, 16 B cp.async / per-row 64 B bulk copies | 2446.4 / 11587.5 |
| 2D TMA tiles + 1D bulk SF, producer warp, 128x128 | 659.4 |
| 128x256, thread 0 refills the previous slot (8 warps: 234 regs, 9 warps capped ptxas at 168 and spilled) | 604.8 |

| In situ, Qwen3.8-27B | GEMM kernel sum, library arm | own arm | e2e pp tok/s |
|---|---:|---:|---|
| pp512 (M 512, CUTLASS) | 216.8-220.6 ms | 235.1-236.1 ms | 10656-10839 vs 10382-10642 |
| pp4096 (M 4096, cuBLASLt) | 1548.8 / 1551.4 ms | 1536.3 / 1535.3 ms | 12086 / 12103 vs 12151 / 12130 |

- Output bit-equal to CUTLASS in 5/5 test shapes per variant; Qwen3.8 PPL 4.6079 both arms (`runtime.deterministic`, 45k corpus, 13811 tokens).
- Default `gemm.nvfp4_tile_min_m = 2048`: replaces cuBLASLt for dense NVFP4; M < 2048 stays CUTLASS.
- pp512 gap per grid: N=17408 94.3 vs 87.6 us (CUTLASS stream-K), small-N grids 17.4 vs 12.0 us (CUTLASS 128x64 tile). Next: stream-K tail and a 64-wide tile.
- Refuted on the isolated bench: 5 stages, grouped-M raster, persistent CTAs, direct epilogue stores (0..+8 %).

## Per cell

| cell | phase | kernel ms | cuBLAS/Lt ms | % | CUTLASS ms | % | top library kernels (us/call x calls) |
|---|---|---|---|---|---|---|---|
| mxfp4-gptoss-20b__pp4096 | bench:pp | 152.5 | 0.7 | 0.4 | 55.5 | 36.4 | [CT] device_kernel 191.4x288, [Lt] nvjet_sm120_hsh_mma_32x128x64_4_16x64x64_tmaAB_a 5.6x96 |
| mxfp4-gptoss-20b__pp4096 | bench:tg | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| mxfp4-gptoss-20b__pp512 | bench:pp | 43.9 | 0.3 | 0.7 | 22.7 | 51.6 | [CT] device_kernel 103.7x216, [Lt] nvjet_sm120_hsh_mma_16x64x128_4_16x16x128_tmaAB_ 3.5x72 |
| mxfp4-gptoss-20b__pp512 | bench:tg | 7.9 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| mxfp4-gptoss-20b__tg128 | bench:pp | 28.7 | 0.2 | 0.8 | 14.5 | 50.6 | [CT] device_kernel 99.8x144, [Lt] nvjet_sm120_hsh_mma_16x64x128_4_16x16x128_tmaAB_ 3.6x48 |
| mxfp4-gptoss-20b__tg128 | bench:tg | 693.2 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| mxfp4-gptoss-20b__tg128_ctx8k | bench:pp | 172.4 | 0.7 | 0.4 | 55.7 | 32.3 | [CT] device_kernel 192.1x288, [Lt] nvjet_sm120_hsh_mma_32x128x64_4_16x64x64_tmaAB_a 5.6x96 |
| mxfp4-gptoss-20b__tg128_ctx8k | bench:tg | 384.5 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-14b__pp4096 | bench:pp | 316.0 | 184.0 | 58.2 | 0.0 | 0.0 | [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 165.7x640, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 307.3x160 |
| nvfp4-14b__pp4096 | bench:tg | 13.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-14b__pp512 | bench:pp | 140.1 | 0.0 | 0.0 | 101.0 | 72.1 | [CT] device_kernel 234.6x240, [CT] device_kernel 107.5x360 |
| nvfp4-14b__pp512 | bench:tg | 18.1 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-14b__tg128 | bench:pp | 37.4 | 0.0 | 0.0 | 26.1 | 69.8 | [CT] device_kernel 85.6x160, [CT] device_kernel 43.4x240 |
| nvfp4-14b__tg128 | bench:tg | 1644.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-14b__tg128_ctx8k | bench:pp | 364.4 | 187.8 | 51.5 | 0.0 | 0.0 | [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 169.3x640, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 318.8x160 |
| nvfp4-14b__tg128_ctx8k | bench:tg | 899.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-gemma4-26b__pp4096 | bench:pp | 193.7 | 52.0 | 26.8 | 53.0 | 27.4 | [CT] device_kernel 136.4x360, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 44.1x360 |
| nvfp4-gemma4-26b__pp4096 | bench:tg | 8.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-gemma4-26b__pp512 | bench:pp | 55.6 | 1.4 | 2.5 | 34.8 | 62.5 | [CT] device_kernel 98.5x270, [CT] device_kernel 13.5x450 |
| nvfp4-gemma4-26b__pp512 | bench:tg | 11.4 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-gemma4-26b__tg128 | bench:pp | 30.0 | 0.9 | 3.1 | 17.3 | 57.6 | [CT] device_kernel 68.2x180, [CT] device_kernel 12.5x300 |
| nvfp4-gemma4-26b__tg128 | bench:tg | 1020.4 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-gemma4-26b__tg128_ctx8k | bench:pp | 224.5 | 74.0 | 32.9 | 51.0 | 22.7 | [CT] device_kernel 133.3x360, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 46.0x360 |
| nvfp4-gemma4-26b__tg128_ctx8k | bench:tg | 565.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-nemotron-30b__pp4096 | bench:pp | 143.5 | 34.8 | 24.2 | 55.5 | 38.7 | [CT] device_kernel 300.5x184, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 67.6x160 |
| nvfp4-nemotron-30b__pp4096 | bench:tg | 4.9 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-nemotron-30b__pp512 | bench:pp | 46.4 | 4.2 | 9.0 | 31.2 | 67.2 | [CT] device_kernel 197.5x138, [CT] device_kernel 15.7x240 |
| nvfp4-nemotron-30b__pp512 | bench:tg | 6.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-nemotron-30b__tg128 | bench:pp | 30.1 | 2.8 | 9.2 | 20.1 | 66.9 | [CT] device_kernel 190.4x92, [CT] device_kernel 15.7x160 |
| nvfp4-nemotron-30b__tg128 | bench:tg | 595.6 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-nemotron-30b__tg128_ctx8k | bench:pp | 144.7 | 34.1 | 23.6 | 51.9 | 35.9 | [CT] device_kernel 280.8x184, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 66.9x160 |
| nvfp4-nemotron-30b__tg128_ctx8k | bench:tg | 311.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q3-30b__pp4096 | bench:pp | 191.5 | 17.0 | 8.9 | 57.1 | 29.8 | [CT] device_kernel 97.9x576, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 33.9x192 |
| nvfp4-q3-30b__pp4096 | bench:tg | 6.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q3-30b__pp512 | bench:pp | 48.6 | 0.7 | 1.4 | 32.4 | 66.7 | [CT] device_kernel 62.0x432, [CT] device_kernel 8.5x432 |
| nvfp4-q3-30b__pp512 | bench:tg | 7.9 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q3-30b__tg128 | bench:pp | 32.2 | 0.4 | 1.4 | 21.4 | 66.5 | [CT] device_kernel 61.3x288, [CT] device_kernel 8.5x288 |
| nvfp4-q3-30b__tg128 | bench:tg | 727.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q3-30b__tg128_ctx8k | bench:pp | 239.9 | 17.0 | 7.1 | 58.7 | 24.5 | [CT] device_kernel 100.3x576, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 32.6x192 |
| nvfp4-q3-30b__tg128_ctx8k | bench:tg | 475.9 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q36-35b__pp4096 | bench:pp | 225.1 | 11.3 | 5.0 | 87.3 | 38.8 | [CT] device_kernel 105.0x480, [CT] device_kernel 100.4x360 |
| nvfp4-q36-35b__pp4096 | bench:tg | 6.4 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q36-35b__pp512 | bench:pp | 64.5 | 1.3 | 2.1 | 37.1 | 57.6 | [CT] device_kernel 72.8x360, [CT] device_kernel 26.1x270 |
| nvfp4-q36-35b__pp512 | bench:tg | 9.6 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q36-35b__tg128 | bench:pp | 43.8 | 0.9 | 2.0 | 25.2 | 57.5 | [CT] device_kernel 74.6x240, [CT] device_kernel 26.1x180 |
| nvfp4-q36-35b__tg128 | bench:tg | 771.2 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q36-35b__tg128_ctx8k | bench:pp | 238.4 | 11.3 | 4.7 | 86.2 | 36.1 | [CT] device_kernel 105.8x480, [CT] device_kernel 95.8x360 |
| nvfp4-q36-35b__tg128_ctx8k | bench:tg | 400.2 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q38-27b__pp4096 | bench:pp | 651.1 | 342.2 | 52.6 | 0.0 | 0.0 | [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 214.4x1088, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 306.9x256 |
| nvfp4-q38-27b__pp4096 | bench:tg | 21.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q38-27b__pp512 | bench:pp | 131.3 | 1.7 | 1.3 | 71.7 | 54.6 | [CT] device_kernel 70.4x576, [CT] device_kernel 56.7x528 |
| nvfp4-q38-27b__pp512 | bench:tg | 31.6 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q38-27b__tg128 | bench:pp | 86.8 | 1.2 | 1.4 | 47.2 | 54.4 | [CT] device_kernel 70.6x384, [CT] device_kernel 54.9x352 |
| nvfp4-q38-27b__tg128 | bench:tg | 2744.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| nvfp4-q38-27b__tg128_ctx8k | bench:pp | 686.9 | 344.4 | 50.1 | 0.0 | 0.0 | [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 216.2x1088, [Lt] cutlass3x_sm120_bstensorop_s16864gemm_block_scal 307.2x256 |
| nvfp4-q38-27b__tg128_ctx8k | bench:tg | 1403.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma3-12b__pp4096 | bench:pp | 893.5 | 123.5 | 13.8 | 0.0 | 0.0 | [Lt] nvjet_sm120_hsh_mma_128x128x64_3_64x32x64_tmaAB_ 1135.8x96, [Lt] nvjet_sm120_hsh_mma_128x208x64_2_32x104x64_tmaAB 150.5x96 |
| q4k-gemma3-12b__pp4096 | bench:tg | 13.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma3-12b__pp512 | bench:pp | 178.4 | 23.9 | 13.4 | 0.0 | 0.0 | [Lt] nvjet_sm120_hsh_mma_192x128x64_2_96x32x64_tmaAB_ 273.7x72, [Lt] nvjet_sm120_hsh_mma_128x112x64_3_32x56x64_tmaAB_ 42.7x72 |
| q4k-gemma3-12b__pp512 | bench:tg | 19.4 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma3-12b__tg128 | bench:pp | 119.0 | 16.3 | 13.7 | 0.0 | 0.0 | [Lt] nvjet_sm120_hsh_mma_192x128x64_2_96x32x64_tmaAB_ 281.4x48, [Lt] nvjet_sm120_hsh_mma_128x112x64_3_32x56x64_tmaAB_ 42.6x48 |
| q4k-gemma3-12b__tg128 | bench:tg | 1647.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma3-12b__tg128_ctx8k | bench:pp | 916.8 | 122.5 | 13.4 | 0.0 | 0.0 | [Lt] nvjet_sm120_hsh_mma_128x128x64_3_64x32x64_tmaAB_ 1124.7x96, [Lt] nvjet_sm120_hsh_mma_128x208x64_2_32x104x64_tmaAB 151.0x96 |
| q4k-gemma3-12b__tg128_ctx8k | bench:tg | 890.4 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma4-26b__pp4096 | bench:pp | 386.8 | 22.8 | 5.9 | 3.5 | 0.9 | [Lt] nvjet_sm120_hss_mma_192x176x64_2_48x88x64_tmaAB_ 146.5x60, [Lt] nvjet_sm120_hsh_mma_256x96x64_2_64x48x64_tmaAB_a 143.2x60 |
| q4k-gemma4-26b__pp4096 | bench:tg | 8.1 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma4-26b__pp512 | bench:pp | 96.0 | 2.0 | 2.1 | 0.0 | 0.0 | [Lt] nvjet_sm120_hsh_mma_80x192x64_2_80x24x64_tmaAB_a 71.9x15, [Lt] nvjet_sm120_hsh_mma_80x192x64_2_80x24x64_tmaAB_a 29.4x15 |
| q4k-gemma4-26b__pp512 | bench:tg | 11.6 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma4-26b__tg128 | bench:pp | 61.6 | 0.9 | 1.5 | 0.0 | 0.0 | [Lt] nvjet_sm120_hsh_mma_80x192x64_2_80x24x64_tmaAB_a 30.3x10, [Lt] nvjet_sm120_hss_mma_80x192x64_2_80x24x64_tmaAB_a 29.1x10 |
| q4k-gemma4-26b__tg128 | bench:tg | 1022.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-gemma4-26b__tg128_ctx8k | bench:pp | 419.7 | 45.8 | 10.9 | 2.3 | 0.5 | [Lt] nvjet_sm120_hsh_mma_256x96x64_2_64x48x64_tmaAB_a 186.6x75, [Lt] nvjet_sm120_hsh_mma_128x80x64_3_64x40x64_tmaAB_a 140.3x60 |
| q4k-gemma4-26b__tg128_ctx8k | bench:tg | 548.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q3-30b__pp4096 | bench:pp | 374.1 | 51.2 | 13.7 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_128x208x64_2_32x104x64_tmaAB 111.1x192, [Lt] nvjet_sm120_hhh_mma_128x208x64_2_32x104x64_tmaAB 103.6x192 |
| q4k-q3-30b__pp4096 | bench:tg | 6.2 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q3-30b__pp512 | bench:pp | 97.5 | 12.7 | 13.1 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_112x128x64_3_112x16x64_tmaAB 29.1x144, [Lt] nvjet_sm120_hhh_mma_128x176x64_2_32x88x64_tmaAB_ 27.4x144 |
| q4k-q3-30b__pp512 | bench:tg | 7.9 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q3-30b__tg128 | bench:pp | 65.3 | 8.9 | 13.7 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_128x176x64_2_32x88x64_tmaAB_ 29.7x96, [Lt] nvjet_sm120_hhh_mma_112x128x64_3_112x16x64_tmaAB 29.5x96 |
| q4k-q3-30b__tg128 | bench:tg | 707.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q3-30b__tg128_ctx8k | bench:pp | 417.2 | 50.9 | 12.2 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_128x208x64_2_32x104x64_tmaAB 111.4x192, [Lt] nvjet_sm120_hhh_mma_128x208x64_2_32x104x64_tmaAB 101.5x192 |
| q4k-q3-30b__tg128_ctx8k | bench:tg | 445.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q36-35b__pp4096 | bench:pp | 368.9 | 39.9 | 10.8 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_128x160x64_2_64x40x64_tmaAB_ 206.7x120, [Lt] nvjet_sm120_hhh_mma_128x208x64_2_32x104x64_tmaAB 97.0x120 |
| q4k-q36-35b__pp4096 | bench:tg | 6.6 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q36-35b__pp512 | bench:pp | 104.1 | 11.0 | 10.5 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_256x112x64_2_64x56x64_tmaAB_ 70.5x90, [Lt] nvjet_sm120_hhh_mma_128x176x64_2_32x88x64_tmaAB_ 30.1x90 |
| q4k-q36-35b__pp512 | bench:tg | 10.1 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q36-35b__tg128 | bench:pp | 69.1 | 7.4 | 10.7 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_256x112x64_2_64x56x64_tmaAB_ 69.7x60, [Lt] nvjet_sm120_hhh_mma_128x176x64_2_32x88x64_tmaAB_ 26.6x60 |
| q4k-q36-35b__tg128 | bench:tg | 842.5 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q4k-q36-35b__tg128_ctx8k | bench:pp | 382.7 | 41.1 | 10.7 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_128x160x64_2_64x40x64_tmaAB_ 214.2x120, [Lt] nvjet_sm120_hhh_mma_128x208x64_2_32x104x64_tmaAB 99.7x120 |
| q4k-q36-35b__tg128_ctx8k | bench:tg | 443.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q6k-14b__pp4096 | bench:pp | 903.3 | 676.4 | 74.9 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_256x128x64_2_64x64x64_tmaAB_ 1078.6x320, [Lt] nvjet_sm120_hhh_mma_96x256x64_2_48x64x64_tmaAB_a 1219.5x160 |
| q6k-14b__pp4096 | bench:tg | 14.0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q6k-14b__pp512 | bench:pp | 246.0 | 144.7 | 58.8 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_256x112x64_2_64x56x64_tmaAB_ 309.9x240, [Lt] nvjet_sm120_hhh_mma_128x128x64_3_32x64x64_tmaAB_ 353.9x120 |
| q6k-14b__pp512 | bench:tg | 18.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q6k-14b__tg128 | bench:pp | 152.8 | 88.5 | 57.9 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_256x112x64_2_64x56x64_tmaAB_ 290.2x160, [Lt] nvjet_sm120_hhh_mma_128x128x64_3_32x64x64_tmaAB_ 300.8x80 |
| q6k-14b__tg128 | bench:tg | 1641.1 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q6k-14b__tg128_ctx8k | bench:pp | 957.2 | 682.3 | 71.3 | 0.0 | 0.0 | [Lt] nvjet_sm120_hhh_mma_256x128x64_2_64x64x64_tmaAB_ 1081.3x320, [Lt] nvjet_sm120_hhh_mma_96x256x64_2_48x64x64_tmaAB_a 1227.2x160 |
| q6k-14b__tg128_ctx8k | bench:tg | 1008.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__pp4096 | bench:pp | 478.6 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__pp4096 | bench:tg | 8.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__pp512 | bench:pp | 94.5 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__pp512 | bench:tg | 11.9 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__tg128 | bench:pp | 62.7 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__tg128 | bench:tg | 1024.3 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__tg128_ctx8k | bench:pp | 515.8 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| q8-8b__tg128_ctx8k | bench:tg | 616.3 | 0.0 | 0.0 | 0.0 | 0.0 | - |
