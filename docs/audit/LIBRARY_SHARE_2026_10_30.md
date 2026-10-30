# cuBLAS/cuBLASLt and CUTLASS share on the SafeTensors heroes (ROADMAP row 123, second inventory)

Source: `imp:test` built from `main` 901bd711 (imp.tree ae0cc04c), RTX 5090, 2026-10-30. First inventory (v0.59.2, before steps 3-11, GGUF rows): [`LIBRARY_SHARE_2026_10_29.md`](LIBRARY_SHARE_2026_10_29.md).
Harness: `IMAGE=imp:test WORKLOADS="pp512 pp4096 tg128" tools/roofline/inventory/inventory.sh`, matrix `tools/roofline/inventory/matrix.tsv` (SafeTensors only), 9 models x 3 workloads = 27 cells. Same nsys flags, classes and share definition as the first inventory.
Table: `library_share.py <out> 4`; CUTLASS family (dense NVFP4 / grouped / MXFP8) from the full template name.

## Result

| model | checkpoint | pp512 Lt / CUTLASS | pp4096 Lt / CUTLASS | library kernels |
|---|---|---:|---:|---|
| Qwen3-8B | FP8 | 77.0 / 0.0 % | 69.2 / 0.0 % | cuBLASLt FP8 x FP8 (`nvjet_sm120_qqhsh_*`), 252 native FP8 weights |
| Qwen3-8B | NVFP4 (cortecs) | 0.0 / 63.1 % | 0.0 / 0.0 % | CUTLASS dense NVFP4 below M 2048 |
| Qwen3-14B | NVFP4 | 0.0 / 68.9 % | 0.0 / 0.0 % | CUTLASS dense NVFP4 below M 2048 |
| Qwen3.8-27B | NVFP4 (vllm) | 1.3 / 55.5 % | 0.8 / 0.0 % | CUTLASS dense NVFP4, small FP16 split-K (`hhh 48x64`) |
| Qwen3-30B-A3B | NVFP4 (Modelopt) | 1.4 / 11.3 % | 1.5 / 0.0 % | CUTLASS dense NVFP4, small FP16 split-K |
| Qwen3-Coder-30B-A3B | NVFP4 | 1.3 / 10.8 % | 1.1 / 0.0 % | same as Qwen3-30B-A3B |
| Qwen3.6-35B-A3B | NVFP4 | 2.1 / 5.8 % | 1.7 / 0.0 % | CUTLASS dense NVFP4, FP16 split-K (62 fp16-tier tensors) |
| Nemotron-3-Nano-30B-A3B | NVFP4 | 10.0 / 8.2 % | 8.8 / 0.0 % | cuBLASLt FP16 (37 fp16-tier tensors, `hhh 256x112` 320.3 us x 24 at pp4096), CUTLASS dense NVFP4 |
| Gemma-4-26B-A4B | NVFP4 | 1.7 / 17.5 % | 2.9 / 0.0 % | CUTLASS dense NVFP4 (`4x22` grids, N 2816), cuBLASLt block-scaled NVFP4 at M 2048 (29.0 us x 120) |

- Decode: 0.0 % library in all 9 tg128 cells (`bench:tg`), as on 2026-10-29.
- pp4096: 0.0 % CUTLASS in every cell; dense NVFP4 >= M 2048, grouped MoE and MXFP8 run the own tile kernels.
- CUTLASS grouped 3.x and CUTLASS MXFP8 GEMM: 0 launches in 27 cells; both are refusal fallbacks only. Qwen3.6-35B runs 90 GDN MXFP8 projections on `gemm.mxfp8_tile` (`quantize_fp16_mxfp8_cutlass_kernel` is own code in the CUTLASS TU).
- Gemma-4 pp4096 cuBLASLt sum 5.8 ms (29.10.: ~52 ms cuBLAS hd512 attention): the hd512 attention left cuBLAS.
- Gemma-4 at M 2048 (grid 16x22, N 2816): the own NVFP4 tile declines the shared-MLP down_proj (K 2112, K % 128 = 64) and cuBLASLt block-scaled serves; see "Gemma-4 down_proj K 2112" below.
- Raw per-cell JSONs are not committed: rerun the harness.

## Remaining call sites, ordered by share

| # | call site | share | state | next step |
|---|---|---:|---|---|
| 1 | dense FP8 prefill, cuBLASLt `qqhsh` | 69.2-77.0 % | step 12 below: own from M 2048, cuBLASLt below | M < 2048: per-SM efficiency of the tile at < 1 wave (same gap as row 2) |
| 2 | dense NVFP4 M < 2048, CUTLASS | 55.5-68.9 % dense, 5.8-17.5 % MoE | own tile +1-2 % slower in situ; 128x64, hybrid stream-K, 4-MMA warps refuted | per-SM MMA efficiency only |
| 3 | FP16 GEMMs of fp16-tier tensors, cuBLASLt | 8.8-10.0 % Nemotron, <= 2.1 % else | own FP16 tile covers K >= N only | N > K and split-K shapes |
| 4 | Gemma-4 NVFP4 at M 2048, cuBLASLt block-scaled | 1.8 % pp4096 | own tile with a half last stage loses in situ (+2.6..3.5 %) | per-SM efficiency, as row 2 |
| 5 | CUTLASS grouped 3.x, CUTLASS MXFP8 | 0 launches | removed (steps 13, 14) | - |

## Gemma-4 down_proj K 2112: own tile refuted, cuBLASLt stays (2026-10-30)

Decline: `launch_tile` needs `K % kBK == 0` (BK 128 NVFP4); Gemma-4 `intermediate_size` 2112 = 16.5 x 128. Probe (not committed): the grouped kernel's half last stage (#2898: TMA zero-fills past K, the second SfAtom per slab zeroed) in the TMA configs, bit-equal to CUTLASS at 2048x2816x2112 and 200x384x192 (max diff 0; NaN scale mutant red).

| Gemma-4-26B-A4B-NVFP4 pp4096, `main` af0e1605 vs probe, `build-dev` imp-cli under nsys, 3 pairs, order flipped per pair | main (cuBLASLt) | probe |
|---|---:|---:|
| down_proj set (542 Lt + 90 tile vs 630 tile launches), ms | 23.45 / 23.53 / 23.45 | 24.05 / 24.35 / 24.11 |
| with stream-K over the half step: dense NVFP4 sum, ms | 129.60-130.45 | 131.25-131.80 |
| e2e tok/s (data-parallel run) | 43774-43945 | 43790-43837 |

Flushed bench 2048x2816x2112 (us): cuBLASLt 32.0, CUTLASS 35.7, own 128x128 31.8-33.6, 128x256 39.6. 352 tiles = 2.07 waves on 170 SMs for both kernels; the gap is per-SM efficiency (row 2), not the wave tail.

## Step 12: own per-tensor FP8 GEMM (`src/quant/nvfp4_gemm_tile.cu`, F8 configs)

`Fmt::kF8`: MXFP8 byte geometry without SF slabs, `*a_scale * *w_scale` (device scalars) and `beta` in the epilogue; beta != 0 stores per fragment in fp32 (one rounding). Call sites: `gemm_fp8_prefill_`, attention q/k/v and o (beta 1), FFN gate/up and down (beta 1); LM head stays cuBLASLt. `gemm.fp8_tile_min_m` (default 2048, 0 off).

| flushed bench `Fp8GemmTileBench.DISABLED_VsCublasLt`, us | cuBLASLt | plain `kind::f8f6f4` | block-scaled, UE8M0 1.0 |
|---|---:|---:|---:|
| 2048 x 4096 x 4096 | 136.0 | 162.6 | 130.4 |
| 2048 x 1024 x 4096 | 50.2 | 48.0 | 36.1 |
| 2048 x 12288 x 4096 | 323.7 | 486.7 | 318.8 |
| 2048 x 4096 x 12288 | 359.3 | 461.6 | 337.0 |
| 2048 x 24576 x 4096 | 621.9 | 980.3 | 722.0 |
| 512 x 4096 x 4096 | 32.1 | 50.4 | 41.4 |
| 512 x 12288 x 4096 | 101.6 | 132.4 | 114.0 |

- The plain `kind::f8f6f4` f32-accumulate MMA caps at 424-447 TFLOPS; the block-scaled `kind::mxf8f6f4` with constant UE8M0 1.0 reaches 647 (same numerics, no SF loads).
- Correctness vs cuBLASLt (`Fp8GemmTileTest`, 7 shapes x 5 variants, beta 0 and 1): max|diff| <= 0.0078 at max|ref| 15.4 (1 fp16 ulp); a mutant that drops beta on one column fails all 7 beta-1 cases.

| Qwen3-8B-FP8 in situ (`prefill_kernel_ab.sh`, 3 pairs) | cuBLASLt | own |
|---|---:|---:|
| pp512 e2e, own at every M | 30481-31167 tok/s | 25544-25709 tok/s |
| pp4096 e2e, own from M 2048 | 26548-26651 tok/s | 26759-26802 tok/s |
| pp4096 GEMM kernel sum | 936.2-940.3 ms | 923.2-925.9 ms |
| PPL 45k, `runtime.deterministic` | 10.7654 | 10.7691 |

- pp4096 per shape (M 2048 chunks): k/v 62.2 -> 38.4 ms, gate/up 448.1 -> 430.4 ms, q/o/down 416.1 -> 445.5 ms.
- Stream-K workspace (10.6 MiB) REFUTED in situ: pp512 GEMM sum 163 -> 155 ms (still behind 131), pp4096 932 -> 963 ms; not wired.

## Step 13: CUTLASS MXFP8 GEMM removed

`src/compute/gemm_cutlass_mxfp8_sm120.cu` deleted; quantizer, FP8 row-scale view and FP16 rebuild moved verbatim to `src/quant/mxfp8_sfatom.cu`; `gemm.mxfp8_tile` removed. The own tile refuses K % 64 != 0 (CUTLASS took K % 32), the projection then keeps its previous route; every local GDN model has K % 128 == 0.

| Qwen3.6-35B-A3B-NVFP4 pp4096, `imp-cli --bench`, 3 pairs, B first | main | step 13 |
|---|---:|---:|
| e2e tok/s | 35385-35564 | 35444-35529 |
| MXFP8 activation scratch | 8.2 MiB | 8.2 MiB |

- A first in each pair read B 35227-35268 vs A 35434-35516: order effect, gone with B first.
- Tests vs a host double reference from the operand bytes; a mutant that drops `col_scale` on odd columns fails 4/4 row-view cases.

## Step 14: CUTLASS 3.x grouped GEMM removed

`src/compute/gemm_cutlass_grouped_3x.{cu,h}` deleted, `gemm.moe_tile` removed. The legacy host-args MoE prefill (`moe.nvfp4_device_args=false`, gpt-oss with staged host experts) ran CUTLASS; it now uploads the per-expert arrays into a 64 KiB T2 slice and runs the own kernel (`gemm_grouped_nvfp4_host_args`). Four B tensormap slots per CTA instead of two lift the K >= 384 limit (K % 64 == 0 now); shapes the kernel refuses skip the grouped path (`gemm_grouped_nvfp4_moe_supports`) and take the NVFP4 dequant batch path. T2 charge 2 MiB -> 64 KiB.

| `imp-cli --bench`, 4 pairs, both orders, tok/s | main | step 14 |
|---|---:|---:|
| Qwen3-30B-A3B-NVFP4 device-args pp512 | 32717-32982 | 32810-32975 |
| Qwen3-30B-A3B-NVFP4 device-args pp4096 | 42996-43172 | 43194-43249 |
| Qwen3-30B-A3B-NVFP4 host-args pp512 | 12379-16748 | 15915-18840 |
| Qwen3-30B-A3B-NVFP4 host-args pp4096 | 31580-33456 | 32475-34752 |
| Gemma-4-26B-A4B-NVFP4 pp512 | 35108-35323 | 35016-35163 |
| Gemma-4-26B-A4B-NVFP4 pp4096 | 44346-44653 | 44432-44692 |
| Qwen3-30B-A3B PPL 45k, deterministic, device-args / host-args | 11.4338 / 11.3110 | 11.4338 / 11.3110 |

- Tests: fp64 reference from the exact NVFP4 bits (`tests/nvfp4_fp64_ref.h`) at K 64..2880 and 512 experts; host-args bit-equal to the per-expert dense GEMM and to device-args. A mutant that drops alpha on odd experts fails 16/18.
