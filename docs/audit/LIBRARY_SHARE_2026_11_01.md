# cuBLASLt routes after CUTLASS and cuBLAS v2 left (ROADMAP row 123, inventory 3)

Source: `build-dev/imp-cli` of `perf/lt-inventory-3` (base `main` 5a94512f + the route counter below), RTX 5090, 2026-11-01. Previous inventories: [`LIBRARY_SHARE_2026_10_29.md`](LIBRARY_SHARE_2026_10_29.md), [`LIBRARY_SHARE_2026_10_30.md`](LIBRARY_SHARE_2026_10_30.md).
Harness: `tools/roofline/inventory/inventory.sh` flags (nsys `--cuda-graph-trace=node`, `bench:*` NVTX phases, `inventory_extract.py`), 8 SafeTensors heroes of `matrix.tsv` (Qwen3.6-35B-A3B no longer on disk) x pp512 / pp4096 / tg128 / tgmtp (`speculative.mtp_k=-1`), plus Qwen3-8B-NVFP4 tgdspark (`speculative.dspark_model=dflash_qwen3_8b_block7`) = 33 cells.
Share = cuBLASLt kernel ns (`nvjet_*`, `splitKreduce`, `gemvx`) / phase kernel ns. Library-share regex of `library_share.py`.

## Route counter

`diagnostics.log_gemm_algo` now also counts every cuBLASLt entry (`src/compute/lt_route_trace.cpp`): route, caller file:line (`std::source_location` default argument on `gemm()`, `gemm_cublaslt()`, `lt_gemm_ex()`, `lt_gemm_strided_batched()`, `lt_gemm_batched()`), m n k batch; `gemm_cleanup()` logs `[lt-route]` lines. Host calls: a graph-replayed prefill counts once at capture. Off: one flag read per Lt call.

## Share per cell

| model | pp512 | pp4096 | tg128 decode | tgmtp prompt / decode | Lt calls (route counter) |
|---|---:|---:|---:|---:|---|
| Qwen3-8B-FP8 | 81.17 % (38.41 of 47.32 ms) | 0.00 % | 0.00 % | 81.02 % / 0.00 % (no MTP head) | `gemm_fp8x8_` FP8 x FP8 at M 512 (and the 16 / 32-token warmup prompts) |
| Qwen3-8B-NVFP4 | 0.00 % | 0.00 % | 0.00 % | 0.00 % / 0.00 % (no MTP head) | none |
| Qwen3-8B-NVFP4 + DSpark | - | - | 0.29 % (3.10 of 1082.48 ms) | prompt 5.41 % (1.20 of 22.22 ms) | drafter k/v projections (M 2-32, 7), drafter fc FP16 -> FP32 (K 20480) |
| Qwen3-14B-NVFP4 | 0.00 % | 0.00 % | 0.00 % | 0.00 % / 0.00 % | none |
| Qwen3.8-27B-NVFP4 | 0.00 % | 0.00 % | 0.00 % | 0.59 % (0.56 of 94.04 ms) / 0.00 % | MTP head (BF16 dense) prompt feed: fc 256 x 5120 x 10240, k/v 256 x 1024 x 5120 |
| Qwen3-30B-A3B-NVFP4 | 1.60 % (0.70 ms) | 1.21 % (2.13 ms) | 0.00 % | 1.56 % / 0.00 % | MoE router 128 x 2048, fp16 accumulate |
| Qwen3-Coder-30B-A3B | 1.43 % (0.67 ms) | 1.11 % (2.05 ms) | 0.00 % | 1.50 % / 0.00 % | same router |
| Nemotron-3-Nano-30B-A3B | 9.69 % (4.27 ms) | 14.16 % (19.78 of 139.70 ms) | 0.00 % | 9.33 % / 0.00 % | fp16-tier tensors via `gemm_dispatch_uncached_fallback` (N x K: 10304 x 2688, 4096 x 2688, 2688 x 4096, 256 x 2688), router 128 x 2688 |
| Gemma-4-26B-A4B | 1.25 % (0.53 ms) | 0.00 % | 0.00 % | 1.10 % / 0.00 % | router 128 x 2816 at n 512 (own split-K from n 1024) |

- Decode: 0.00 % in all 16 tg128 / tgmtp decode phases; the only decode Lt is the opt-in FP16 DSpark drafter path (0.29 %).
- pp4096: 0.00 % except the routers (1.1-1.2 %) and Nemotron (14.16 %).
- NVFP4 dense heroes (8B, 14B, 27B): 0 cuBLASLt calls in the whole process, warmup included.

Per-kernel detail (pp phase, nsys):

| cell | kernel | calls | ms | us / call |
|---|---|---:|---:|---:|
| Qwen3-8B-FP8 pp512 | `qqhsh 160x128x128` (gate/up) | 216 | 18.114 | 83.9 |
| | `qqhsh 128x112x128` (down) | 108 | 10.097 | 93.5 |
| | `qqhsh 112x128x128` x 2 (q, o) | 216 | 6.453 | 29.6-30.2 |
| | `qqhsh 128x112x128` split-K (k/v) + `splitKreduce` | 216 + 216 | 2.660 + 1.089 | 12.3 + 5.0 |
| Nemotron pp4096 | `hhh 256x112x64` | 24 | 8.614 | 358.9 |
| | `hhh 128x256x64` | 48 | 6.079 | 126.6 |
| | `hhh 128x208x64` | 24 | 3.340 | 139.1 |
| | `hhh 64x128x64` split-K (router) | 92 | 0.822 | 8.9 |
| | `hhh 128x112x64` split-K + `splitKreduce` | 48 + 140 | 0.616 + 0.308 | 12.8 + 2.2 |
| Qwen3-30B pp4096 | `hhh 64x128x64` split-K (router) + `splitKreduce` | 192 + 192 | 1.682 + 0.451 | 8.8 + 2.3 |
| Qwen3-30B pp512 | `hhh 64x64x64` split-K (router) + `splitKreduce` | 144 + 144 | 0.559 + 0.136 | 3.9 + 0.9 |
| Gemma-4 pp512 | `hsh 64x64x64` split-K (router) + `splitKreduce` | 90 + 90 | 0.424 + 0.101 | 4.7 + 1.1 |
| Qwen3.8-27B tgmtp prompt | `hhh 128x128x64` + `hhh 112x128x64` split-K + reduce (MTP head) | 4 + 8 + 12 | 0.420 + 0.098 + 0.037 | 105.1 / 12.3 / 3.1 |
| Qwen3-8B DSpark prompt | `hss 128x112x64` (fc, FP32 out) | 2 | 0.841 | 420.3 |
| Qwen3-8B DSpark decode | `hhh 128x16x64` split-K (k/v) + reduce | 200 + 200 | 1.215 + 0.184 | 6.1 + 0.9 |
| | `gemvx` + `hss 64x8x128` (fc at 1-4 rows) | 12 + 4 | 1.255 + 0.425 | 104.6 / 106.3 |

## Routes by class

(a) hot on a hero (measured above):

| # | route (call site) | models / workload | share | own kernel covering the shape | state |
|---|---|---|---:|---|---|
| a1 | FP8 x FP8 below M 2048 (`gemm_fp8x8_`) | Qwen3-8B-FP8 prompts < 2048 rows | 81 % of pp512 | F8 tile (`gemm_fp8_tile`) | refuted: pp512 GEMM sum Lt 127.9-130.6 vs own 136.5-138.5 ms, 2-CTA 167.7-179.5 ([10-30 doc](LIBRARY_SHARE_2026_10_30.md#2026-11-01-two-ctas-per-sm-below-m-2048-refuted)); own from M 2048 |
| a2 | fp16-tier tensors (`gemm_dispatch_uncached_fallback` -> `gemm()`) | Nemotron-3-Nano | 9.3-14.2 % (incl. a3) | `gemm_f16_tile` (K >= N, >= 0.8 wave) | refuted: 128x256 for every shape 50.04 vs 50.48 ms Lt, e2e 56278-56406 vs 56447-56584 tok/s ([10-30 doc](LIBRARY_SHARE_2026_10_30.md#remaining-call-sites-ordered-by-share), row 3) |
| a3 | MoE router (`router_logits_gemm` -> `gemm()`) | Qwen3-30B, Coder, Nemotron (fp16 accumulate), Gemma-4 n < 1024 | 1.1-1.6 % | `gemm_f16_splitk` | own for Gemma-4 n >= 1024; refuted for K 2048 (11.3 vs 10.5 us), n <= 1000 (5-7 vs 1.2-4.4 us), Nemotron fp16-acc (4.48 vs 5.83 ms) ([10-30 doc](LIBRARY_SHARE_2026_10_30.md#moe-router-gemm-f16-n--n_experts-own-split-k-for-the-gemma-4-shape-cublaslt-for-the-rest-2026-10-31)); Qwen3-30B n >= 1024 in situ refuted (close section) |
| a4 | MTP head BF16 dense (`mtp_feed_dense_rows`) | Qwen3.8-27B with MTP | 0.59 % of the prompt phase, 0 in decode | `gemm_f16_tile` declines (40 / 8 tiles < 1 wave) | accepted, below 1 % |
| a5 | DSpark FP16 k/v + fc (`DSparkDrafter` context projections) | Qwen3-8B-NVFP4 + drafter (opt-in) | 0.29 % decode, 5.4 % prompt | none for M 2-32 x N 1024 x K 4096 FP16 | accepted, opt-in, below 1 % of decode |

Batched decode (M 2-16, not in the single-stream bench): measured under conc 16 in the close section below; NVFP4 dense takes `gemm_nvfp4_smallm` (own).

(b) GGUF only (legacy, no new GGUF kernel work: stays on cuBLASLt):

| route | reached by |
|---|---|
| `generic_dequant_kernel` -> `gemm()` | Q6_K above M 1536 and k/v grids, Q3_K, other K-quants outside the own decode tiles |
| `gemm_dispatch_uncached_fallback` dequant arm, FP16 cache in `gemm_via_handle_` | GGUF weights without an own prompt kernel |
| `gemm_moe_batched` (`try_run_moe_fp16_batch_prefill`, legacy MoE fallback) | GGUF MoE experts dequantized to FP16 |

(c) tools / offline:

| route | reached by |
|---|---|
| `tools/imp-quantize/gptq.cu` (7 `lt_gemm_ex`, FP32 TF32) | `imp-quantize --gptq` / `--gptaq` |
| `activation_calibrator_hessian.cu`, `activation_calibrator_pair.cpp` (X^T X, FP16 -> FP32) | AWQ / GPTQ calibration |

(d) cold fallbacks (0 calls in 33 cells):

| route | when |
|---|---|
| `gemm_nvfp4_cublaslt` | own NVFP4 tile refused at M >= `gemm.nvfp4_cublaslt_min_m` |
| FP8 copy of a CUTLASS_NVFP4 weight, native FP8 without activation scratch (`gemm_via_handle_`, `gemm_fp8_prefill_`) | tile declined / scratch too small |
| `gemm_nvfp4` dequant + `gemm()` | NVFP4 tier without the tile payload |
| `attention_cublas_prefill` (materialized) | FA2 / hd512 kernels decline and the S matrix fits |
| exact-alignment `lt_gemm_ex` retries in `gemm()`, `gemm_cublaslt_fp8_probe` | cached descriptor failed; startup probe |
| 16 / 32-token warmup prompts (`Engine::warmup`) | init only, same routes as (a) |

(e) SafeTensors features outside the hero set (not measured here):

| route | reached by |
|---|---|
| vision encoders (`vision_encoder.cu`, `internvl_encoder.cpp`, `qwen3vl_encoder.cpp`) | image input (~1.5 ms per 192x320 image, M 240) |
| `gemma4_audio_encoder.cpp` | audio input |
| `encoder_forward.cu` + materialized attention | embedding / rerank models |
| `executor_lora.cu`, `executor_gemma_pli.cpp`, `executor_qsa.cpp`, `executor_prompt_logprobs.cu` | LoRA adapters, Gemma PLI, Flash-Next QSA index, prompt logprobs |
| `gemm_kv_batched`, `gemm_pair_batched`, FP16 tier of `gemm_dispatch` | BF16 / FP16 dense checkpoints |

## Close (2026-11-02)

Owner decision 2026-11-01: close after the router A/B and the batched-decode inventory.

| Qwen3-30B-A3B-NVFP4 router, `main` vs own `gemm_f16_splitk` from n 1024 (fp16-acc and d >= 2560 gates dropped, 8 MiB workspace for every MoE), pp4096, 3 pairs, order flipped | Lt (main) | own |
|---|---:|---:|
| e2e tok/s | 44362.75-44460.71 | 42443.90-44536.62 |
| router set in the whole run (nvjet + splitKreduce / `gemm_f16_splitk_kernel`), ms | 9.521-9.887 (1932 launches) | 12.826-13.108 (1064 launches) |

Verdict: refuted, the router stays on cuBLASLt outside the Gemma-4 shape.

| conc 16, build-dev imp-server under nsys (whole run), `conc_client.py` 4 waves x 16 x 400 tokens | kernels ms | Lt ms | Lt % | Lt routes (`[lt-route]`, distinct shapes) | median tok/s |
|---|---:|---:|---:|---|---:|
| Qwen3-8B-FP8 | 14269.17 | 11411.981 | 79.98 | `gemm_fp8x8_` FP8 x FP8 at M 2-16 (150) | 2279.3 |
| Qwen3-30B-A3B-NVFP4 | 16683.94 | 265.318 | 1.59 | router at M 2-16 (36) | 1672.9 |
| Nemotron-3-Nano-30B-A3B | 4711.86 | 301.460 | 6.40 | fp16-tier tensors (72), router (19) | 581.1 |
| Gemma-4-26B-A4B | 38144.89 | 440.539 | 1.15 | router (38) | 692.7 |

- Qwen3-8B-FP8 decode rows run the M 512 algo pin (`kFp8RowPinRefM`, #2152): `qqhsh 160x128x128` 4840.04 ms over 104832 calls (46 us per gate/up at M <= 16).
- Nemotron replies stop early (638 / 474 tokens per wave), its run is prompt-heavy.

| Qwen3-8B-FP8 conc 16 without nsys, own F8 tile at every M (`gemm.fp8_tile_min_m=1`) vs default, 3 pairs, order flipped | default (Lt below M 2048) | own from M 1 |
|---|---:|---:|
| median aggregate tok/s | 1680.1-1702.6 | 1479.0-1483.8 |

Verdict: refuted (-11.7 to -13.1 %); FP8 batched decode stays on cuBLASLt with the FP8 prefill rows below M 2048. A dedicated FP8 small-M kernel (the NVFP4 `gemm_nvfp4_smallm` analogue) is the remaining lever, outside this row.

Accepted cuBLASLt uses (row 123 closed):

| use | evidence |
|---|---|
| (b) GGUF legacy | no new GGUF kernel work (`DESIGN_DECISIONS.md`) |
| a1 FP8 x FP8 below M 2048, prefill and batched decode | own tile pp512 e2e 25544-25709 vs 30481-31167 tok/s, conc 16 decode -11.7 to -13.1 % |
| a2 FP16 N > K and sub-wave grids (Nemotron) | own 128x256 at parity, e2e -0.2..-0.3 % |
| a3 MoE routers outside the Gemma-4 shape | own split-K +3.3 ms per pp4096 run (above), n <= 1000 and decode rows 1.2-4.4 vs 5-7 us |
| a4 MTP head, a5 DSpark FP16 | < 1 % of decode |
| (c) offline tools, (d) cold fallbacks, (e) feature encoders | off the hot SafeTensors path |
