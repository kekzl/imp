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

## Step 2: stream-K and the tile pick

Stream-K 128x128 (grid = SMs, forward unit ranges: a CTA's first segment writes its partial at once, the tile's k-step-0 owner adds it at the end of its own range), used when the data-parallel tile idles >= 15 % of its waves; off under `runtime.deterministic` (other summation order). 128x64 tile (8 warps of 64x16) refuted: M=512 N=1024 17.8 vs CUTLASS 12.5 us.

| In situ, Qwen3.8-27B (`gemm.nvfp4_tile_min_m` A/B) | library arm | own arm |
|---|---:|---:|
| pp512 GEMM kernel sum (2048 vs 1, 3 pairs) | 219.2-221.0 ms | 221.7-223.2 ms |
| pp4096 GEMM kernel sum (0 vs 2048, 2 pairs) | 1541.2 / 1544.7 ms | 1533.9 / 1534.8 ms |
| pp4096 e2e | 12109 / 12101 tok/s | 12147 / 12136 tok/s |

- pp512 per shape, own vs CUTLASS us/call (before the idle rule): N=17408 89.3 vs 86.0, N=6144 36.6 / 77.5 vs 31.7 / 68.5, N=1024 14.0 vs 11.9; N=5120/10240 group 83.2 vs 87.9 ms total.
- Stream-K epilogue staged in its own buffer (3 stages, 168 regs): isolated M=512 N=1024 19.0 -> 15.8, N=6144 32.3 -> 30.8 us; in situ pp512 GEMM sum 221.7-222.7 vs CUTLASS 217.1-219.5 ms, unchanged: in situ the stream-K tiles run slower than isolated (N=17408 90.6 vs 87.6 us), CUTLASS does not.
- CUTLASS stays the default below M 2048 until the own arm leads there.
- Isolated vs in situ, explained: `Nvfp4GemmTileBench` rotated 4 weight copies, which fit the 96 MB L2 up to 6144 x 5120 (4 x 15.7 MB). With the L2 flushed (256 MiB memset) before each timed call, median of 31, two runs agree within 2-5 % and rank like in situ. M=512 own auto vs CUTLASS us: N 17408 K 5120 103.4 / 104.5 vs 98.3 / 96.3, N 10240 51.2 / 52.0 vs 56.6 / 56.6, N 1024 19.2 / 19.2 vs 17.7 / 19.7, the rest within 5 %. Rotating > 512 MiB of copies instead swung 12.6 -> 28.3 us between runs. 5 stages (rotating bench, noisy): no clear gain.
- Stream-K flags: a `__device__` array reset by each tile's finisher replaces the per-launch `cudaMemsetAsync` (one launch fewer per stream-K GEMM); flush bench M=512 N=1024 17.7 / 19.1 -> 17.2 / 17.2 us, other shapes unchanged. So the memset was not the stream-K gap.
- Refuted, hybrid stream-K (whole waves data-parallel, only the remainder tiles split, as CUTLASS stream-K): M=512 N=17408 K=5120 104.5 -> 113.7 us, N=6144 40 -> 54 us; with the last full wave also split, N=17408 108.6-109.9 us. Short split segments (8 k-steps) cost a wave; the full split stays.
- The own kernels run at the 575 W power cap, CUTLASS below it (M=512 N=17408 K=5120, 3 s back to back, nvidia-smi): CUTLASS 89.6 us at 2350 MHz / 531 W, own 128x128 98.8 us at 2358 MHz / 571 W, own 128x256 89.0 us at 2492 MHz / 575 W, own stream-K 88.5 us at 2203 MHz / 575 W. ncu (base clocks): tensor pipe 73.6 % CUTLASS vs 71.6 % own stream-K.
- Probed, not committed: 4 MMA warps of 64 x 64 per 128x128 tile (half the A fragment loads per MMA, 210-254 regs, bit-exact). Back to back: stream-K 84.7 us at 2405 MHz vs CUTLASS 88.9. In situ Qwen3.8-27B pp512 (3 pairs, `gemm.nvfp4_tile_min_m` 2048 vs 1): GEMM kernel sum 219.9 / 220.8 / 218.7 vs 222.2 / 222.3 / 218.8 ms; per shape gate/up 89.0 vs 91.4 us, N 5120 / 10240 56.6 vs 53.7, N 6144 32.1 vs 36.0, N 12288 70.7 vs 75.1, N 1024 12.0 vs 15.1 (CUTLASS 128x64-wide grid). Parity, no lead: CUTLASS stays below M 2048.

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

## Step 3: own FP16 GEMM (`src/compute/gemm_f16_tile.cu`)

The NVFP4 skeleton with `mma.sync m16n8k16` (fp16 or fp32 accumulator), BK 64 halves, 128x256 x 2 stages, grouped-M raster 16. FP16 moves 4x the bytes per flop of NVFP4: the 178 MB weight of 17408x5120 exceeds the 96 MB L2, so N-fastest order re-read it from DRAM.

| Isolated (`test-compute --gtest_filter='GemmF16TileBench*'`) | cuBLASLt us | own us |
|---|---:|---:|
| 4096x17408x5120 (128x128 / no group / group 8 / group 16) | 1990.9 | 3034.7 / 2382.1 / 2183.8 / 2102.6 |
| 4096x5120x17408 | 2330.3 | 2133.1 |
| 512x1024x5120 | 30.7 | 120 (no split-K) |

| In situ, Qwen3-14B-Q6_K (`gemm.f16_tile`, 2 pairs) | cuBLASLt | own |
|---|---:|---:|
| pp2048, own when grid >= 1 wave | 1505.4 / 1510.1 ms | 1438.6 / 1441.6 ms |
| pp2048, own when grid >= 2 waves | 1501.1 / 1505.8 ms | 1522.3 / 1522.4 ms |
| pp2048, own when K >= N and grid >= 1 wave (default) | 1502.9 / 1508.7 ms | 1419.5 / 1426.1 ms |
| pp2048 e2e, default rule | 9304 / 9330 tok/s | 9692 / 9695 tok/s |

- In situ the deep-K shapes win (down proj 2048x5120x17408 lost isolated, 1072.2 vs 1007.1 us) and the N=17408 gate/up lose ~1 %: the rule follows the in-situ data.
- fp16 accumulate is bit-equal to cuBLASLt COMPUTE_16F (max diff 0); PPL 9.2022 both arms, ACTIVE line in the own arm (M=2048 N=5120 K=5120).

## Step 4: own grouped NVFP4 MoE GEMM (`src/compute/gemm_grouped_nvfp4_tile.cu`)

Persistent CTAs over a per-expert tile prefix (warp scan in shared memory), A through one TMA map over the expert-sorted rows, B per expert through a per-CTA tensormap re-pointed with `tensormap.replace` (two slots). Same `GroupedNvfp4DeviceArgs` operands as CUTLASS 3.x grouped, bit-exact in both B modes (`test-quant --gtest_filter='GroupedNvfp4TileTest.*'`).

| Qwen3-30B-A3B-NVFP4 in situ (`gemm.moe_tile`, 3 pairs) | CUTLASS grouped | own |
|---|---:|---:|
| pp512 GEMM kernel sum | 98.8-99.1 ms | 94.5-95.1 ms |
| pp512 e2e | 30066-30770 tok/s | 31663-32112 tok/s |
| pp4096 GEMM kernel sum | 260.0-260.7 ms | 243.2-245.6 ms |
| pp4096 e2e | 40935-41025 tok/s | 41811-41831 tok/s |

- First version with a separate prefix kernel per call: kernel sums lower, pp512 e2e mixed (30202 / 30787 vs 30747 / 30799 tok/s); the in-kernel scan removed the extra launch.
- Qwen3.6-35B-A3B-NVFP4 pp512 GEMM sum 115.8 / 115.9 -> 112.2 / 112.3 ms (2 pairs; e2e spread too wide for a verdict).
- PPL 11.4338 both arms (`runtime.deterministic`, 45k corpus), ACTIVE line (experts=128 N=768 K=2048).
- Measured before the port: pp512 CUTLASS grouped sits near the weight floor (~208 vs ~197 us per layer), pp4096 at ~40 % of the dense own kernel's rate.
- K % 128 == 64 (half last stage, second SF atom zeroed in shared memory): bit-exact at K 704 / 2880. gpt-oss-20b (MXFP4 converted to NVFP4 at load, `moe_prefill=cutlass3x → device_args`) GEMM sum pp512 70.1 / 81.5 -> 67.1 / 67.3 ms, pp4096 256.0 / 296.4 -> 238.0 / 238.5 ms, e2e pp4096 51964 / 51906 -> 53165 / 53342 tok/s; PPL 262.6413 both arms (raw corpus, no harmony template). Correction: Gemma-4-26B (N 704, K 2816) and Nemotron-3-Nano (N 1856 / 2688, K 2688 / 1856) do run the own kernel since #2897 (ACTIVE line); the "no ACTIVE line" read came from a mis-named log glob. pp4096, 2 pairs: grouped time 211.7 / 211.0 -> 200.4 / 202.1 ms and 236.2 / 236.2 -> 232.2 / 233.2 ms, e2e 40785 / 40756 -> 41166 / 40982 and 55080 / 55098 -> 55385 / 55667 tok/s.

## Step 5: own FA2 for head_dim 512 (`src/compute/attention_fa2_hd512.cu`)

8 warps = 4 row groups x 2 D halves; each warp keeps Q (16 x 256, 64 regs) and O (16 x 256 fp32, 128 regs) in registers, the two warps of a row group swap their partial scores per 16-row KV tile through shared memory (double-buffered, one named barrier) and run the identical online softmax. 253 registers, no spill. CPU reference test (`test-attention --gtest_filter='*Fa2Hd512*'`): max error 3.5e-4 with q_offset, GQA, softcap.

| Gemma-4-26B-NVFP4 (`attention.hd512_prefill`, 2 pairs) | cuBLAS materialized | own FA2 | own WMMA (`fmha`) |
|---|---:|---:|---:|
| pp512 e2e | 33450 / 33851 tok/s | 34013 / 33461 tok/s | 32689 / 32652 tok/s |
| pp4096 attention kernels | 211.4 / 211.2 ms | 194.2 / 194.4 ms | 528.8 / 525.3 ms (other regex) |
| pp4096 e2e | 41057 / 41213 tok/s | 42789 / 42620 tok/s | 31419 / 31596 tok/s |

- PPL 409.5548 (cuBLAS) -> 407.3936 (FA2), `runtime.deterministic`, 45k raw corpus (instruct model, no template).
- The kernel runs at ~246 TFLOPS at pp4096 (1115.6 us per call): the fp32-accumulate FP16 rate. The hd<=256 FA2 accumulates in fp16; that is the next lever.
- A first A/B showed equal launch counts in both arms; the rerun after a fresh build differed (776 vs 666 at pp512), only the rerun is recorded.
- Refuted 2026-10-29: fp16 accumulators in the hd=512 FA2. Gemma-4-26B-NVFP4, `runtime.deterministic` PPL (45k corpus): fp32 407.3936, fp16 output accumulator 418.6447 (+2.8 %, pp4096 +1.1 % single run), fp16 scores and output 446.9929 (+9.7 %, fa2_hd512 kernel 99.0 / 98.8 -> 71.4 / 71.6 ms, pp4096 e2e +3.3..3.5 %). The hd<=256 FA2 tolerates fp16 accumulation; 512-wide dot products and 256 KV tiles of rescaled fp16 sums do not.

## Step 6: own MXFP8 GEMM (`src/quant/nvfp4_gemm_tile.cu`, MX8 configs)

The GDN projection prefill copies (`gemm.mxfp8_gdn_proj_prefill`, FP8 row-scale view by default) ran CUTLASS MXFP8 plus a separate column-scale pass. An MXFP8 stage of BK elements moves the bytes of an NVFP4 stage of 2 x BK, and the `m16n8k32` E4M3 fragments are the `m16n8k64` E2M1 ones byte for byte, so the NVFP4 tile kernel takes MXFP8 as a format parameter: `mma.sync kind::mxf8f6f4 scale_vec::1X`, one UE8M0 byte per row per K group (a BK=64 stage reads half an SfAtom), column scale fused into the epilogue. `test-quant --gtest_filter='*Mxfp8GemmTile*'`: max diff 0 vs CUTLASS on plain weights in the data-parallel variants, one fp16 ulp (0.0039) on the row-scale view.

| Qwen3.6-35B-A3B-NVFP4 pp4096 in situ (`gemm.mxfp8_tile`, pair 2, M = 2048 per chunk) | CUTLASS + scale pass | own |
|---|---:|---:|
| qkv N 8192 K 2048, per call | 141.2 + 14.4 us | 143.8 us (128x128) |
| z N 4096 K 2048, per call | 77.1 + 8.3 us | 75.9 us (stream-K) |
| out N 2048 K 4096, per call | 79.1 + 4.5 us | 75.9 us (stream-K) |
| class sum, 540 calls each | 175.3 ms | 159.6 ms |
| regex sum incl. the unchanged NVFP4 dense, 2 pairs | 221.1 / 222.7 ms | 204.9 / 208.1 ms |
| e2e pp4096, 2 pairs | 34067 / 33595 tok/s | 34365 / 33774 tok/s |

- pp512, 2 pairs: regex sum 34.8 / 34.6 -> 31.8 / 31.0 ms, launches 3060 -> 2250; e2e inside the MoE spread (21046 / 17745 vs 20741 / 17964 tok/s).
- The qkv GEMM alone is 1.8 % slower than CUTLASS in situ (143.8 vs 141.2 us), 12.5 % faster isolated (121.8 vs 139.2 us incl. the scale pass, `Mxfp8GemmTileBench`): the in-situ/isolated gap of the NVFP4 tiles, unexplained.
- PPL (`runtime.deterministic`, 45k corpus): 6.7308 -> 6.7158 (ACTIVE line in the own arm), inside the +-0.2..0.5 % routing band of this model; Qwen3.8-27B-NVFP4-vllm takes no MXFP8 route (0 route lines), 4.6079 both arms.
- Refuted: stream-K for qkv (idle 14 %, threshold 0.15 -> 0.10): 145.3 vs 143.8 us data-parallel, regex sum 207.8 ms (1 run).
- CUTLASS MXFP8 stays as the refusal fallback (K % 64 != 0, unaligned operands).
