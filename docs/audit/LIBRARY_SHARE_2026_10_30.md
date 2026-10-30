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
- Gemma-4 at M 2048 (grid 16x22, N 2816): the own NVFP4 tile declines and cuBLASLt block-scaled serves; the decline reason is not checked yet.
- Raw per-cell JSONs are not committed: rerun the harness.

## Remaining call sites, ordered by share

| # | call site | share | state | next step |
|---|---|---:|---|---|
| 1 | dense FP8 prefill, cuBLASLt `qqhsh` | 69.2-77.0 % | never inventoried before | own FP8 tile (`kind::f8f6f4`, per-tensor scales in the epilogue) on the TMA tile skeleton, A/B vs cuBLASLt in situ |
| 2 | dense NVFP4 M < 2048, CUTLASS | 55.5-68.9 % dense, 5.8-17.5 % MoE | own tile +1-2 % slower in situ; 128x64, hybrid stream-K, 4-MMA warps refuted | per-SM MMA efficiency only |
| 3 | FP16 GEMMs of fp16-tier tensors, cuBLASLt | 8.8-10.0 % Nemotron, <= 2.1 % else | own FP16 tile covers K >= N only | N > K and split-K shapes |
| 4 | Gemma-4 NVFP4 at M 2048, cuBLASLt block-scaled | 1.8 % pp4096 | own tile declines | find the decline, then the CUTLASS / Lt fallback goes |
| 5 | CUTLASS grouped 3.x, CUTLASS MXFP8 | 0 launches | refusal fallbacks | cover the refusal shapes in the own kernels, then delete both TUs |
