# Prefill Gap Analysis — 2026-06-07

Complete prefill gap analysis: where imp loses prefill against llama.cpp (GGUF)
and vLLM (NVFP4), and why — fresh same-day cross-engine measurements combined
with the calibrated kernel-level roofline attribution (run
`e66cfa45-dirty_20260607_042900`, re-parsed at `config_version: 3` after the
PR #606 TC-rate calibration) and nsys kernel-name evidence on the competitor.

**TL;DR.**
- **GGUF is a ceiling-class gap, not a tuning gap.** imp's prefill GEMMs run at
  83–88 % of their real ceiling — but that ceiling is the GeForce-quartered
  FP16-f32-accumulate rate (253 TFLOPS measured). llama.cpp's MMQ computes on
  INT8 tensor cores with int32 accumulate (not quartered) and fuses dequant into
  the GEMM (activation quant costs it 2.6 % where imp's weight-dequant
  materialization costs 30–65 %). Verified by nsys: llama.cpp runs `mul_mat_q`
  (fused INT8 MMQ) for 73 % (dense Q8) / 69 % (MoE Q4_K) of its prefill GPU
  time — zero cuBLAS.
- **NVFP4 is an attention-scaling gap.** imp *wins* vs vLLM 0.22.1 below ~2k
  context (vLLM has a pathological flat-cost small-M regime) but loses 1.8–2.6×
  at pp4096, where imp's FA2 share grows to 31–40 % of the window at only ~30 %
  of its (also-quartered) FP16-f32acc ceiling while vLLM's FlashInfer attention
  scales. The CUTLASS NVFP4 GEMM itself is at 60 % of the real FP4 ceiling
  (structural TMA-WS pipeline cost) and chunking costs only ~6 %.

## 1. Setup

| | |
|---|---|
| imp | `main` @ `22f8a338` (post #606 fp16-acc opt-in + small-N pingpong, post #607 dp4a LDG.128), Docker `imp:test` built from a clean worktree |
| llama.cpp | build `19e92c3`, CUDA 12.8, sm_120 native (`llamacpp:sm120`), `-fa 1 -ngl 999 -r 5` |
| vLLM | 0.22.1 (`vllm/vllm-openai:latest`, digest `953d3a06`), quant auto-detected (modelopt / compressed-tensors), `max_num_seqs=1`, prefix caching OFF, exact-token-count prompts, `max_tokens=1`, median of 5; FlashInfer-CUTLASS NVFP4 GEMM; defaults include `kv_cache_dtype=fp8_e4m3` (imp runs FP16 KV — slight method asymmetry in vLLM's favor) |
| Host | RTX 5090, driver 610.47, WSL2; clocks sampled at 2 s across the whole session: avg under load **2813 MHz SM / 13801 MHz mem / 505 W** (healthy-host profile) |
| imp method | fresh container per cell (= cuBLAS algo re-selection per restart), `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 10 reps, 2 restarts per (model, pp), decode tg256 recorded as sanity |

Known caveat: cuBLAS prefill varies up to 2.6× across restarts. imp numbers
below are per-restart pairs (both shown); today's pairs agree within ~2 %
except NVFP4-MoE pp512 (1.39× spread — shown).

## 2. Engine-level gap (measured today)

### 2.1 GGUF — imp vs llama.cpp (same files)

| Model | imp pp512 | llama pp512 | gap | imp pp2048 | llama pp2048 | gap | imp tg | llama tg32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B Q8_0 (dense) | 8 401 / 8 422 | 13 724 ± 1455 | **1.63×** | 8 019 / 7 912 | 12 899 ± 61 | **1.62×** | 274 | 157 |
| + `gemm.cublas_fp16_acc` | **10 544 / 10 460** | — | **1.31×** | **9 878 / 9 812** | — | **1.31×** | 275 | — |
| Qwen3-14B Q6_K (dense) | 5 278 / 5 247 | 6 522 ± 222 | **1.24×** | 5 068 / 5 080 | 6 240 ± 17 | **1.23×** | 130 | 115 |
| Qwen3-30B-A3B Q4_K_M (MoE) | 3 903 / 3 884 | 9 288 ± 191 | **2.39×** | 3 897 / 3 871 | 9 313 ± 143 | **2.40×** | 283 | 314 |
| gemma-4-26B-A4B Q4_K_M (MoE) | 4 285 / 4 262 | 10 749 ± 632 | **2.51×** | 4 179 / 4 177 | 10 678 ± 197 | **2.56×** | 263 | 214 |
| Qwen3.6-35B-A3B Q4_K_M (hybrid) | 3 675 / 3 721 | 8 027 ± 118 | **2.17×** | 3 724 / 3 742 | 7 833 ± 105 | **2.10×** | 165 | 192 |

Decode sanity confirms the known picture (imp wins dense decode +74 %, wins
gemma-4 MoE decode +23 %, loses Qwen3 MoE/hybrid decode −10/−14 %) and a
healthy host day (Q8 tg ≈ 274).

**The `gemm.cublas_fp16_acc` arm (opt-in from #606) measured +24.9 % / +23.9 %
today (paired same-day restarts), better than the +17 % recorded at #606 time.**
Decode neutral (275 vs 274). It alone shrinks the dense-Q8 gap from 1.63× to
1.31×.

### 2.2 NVFP4 — imp vs vLLM 0.22.1

| Model | pp | imp (r1/r2) | vLLM (median) | gap |
|---|---|---:|---:|---|
| Qwen3-8B-cortecs (compressed-tensors) | 512 | 33 486 / 29 705 | 2 443 | imp **+12.9×** |
| | 1024 | — | 3 970 | |
| | 2048 | 24 020 / 26 999 | 8 899 | imp **+2.9×** |
| | 4096 | 20 997 / 22 086 | 44 937 | vLLM **2.09×** |
| Qwen3-14B (modelopt) | 512 | 19 112 / 21 149 | 1 496 | imp **+13.4×** |
| | 2048 | 18 440 / 17 974 | 20 178 | vLLM **1.11×** |
| | 4096 | 15 789 / 16 067 | 28 412 | vLLM **1.78×** |
| Qwen3-30B-A3B-Modelopt (MoE) | 512 | 13 852 / 19 279 | 8 386 | imp **+1.98×** |
| | 2048 | 17 725 / 16 682 | 35 379 | vLLM **2.06×** |
| | 4096 | 15 242 / 15 289 | 40 487 | vLLM **2.65×** |

Two regimes:

- **vLLM's small-M pathology**: below a shape threshold its NVFP4 prefill cost
  is *flat* (~200–340 ms regardless of 512 vs 2048 tokens on the
  compressed-tensors model; pp512-only on modelopt) — every rep identical, so
  it is a kernel-dispatch/autotune-coverage artifact, not warmup. In the
  interactive/agentic regime (short prompts, prefix-cache hits) **imp wins
  NVFP4 prefill outright**.
- **Long context**: vLLM throughput *rises* with context (28–45 k at pp4096)
  while imp *falls* (8B: 31.6k→21.5k; 14B: 20.1k→15.9k). The divergence is
  attention-driven (see §4.4).

### 2.3 The effective-TFLOPS lens (why GGUF is a ceiling-class gap)

Effective model FLOP rate = `2 · params · pp_rate` (dense models, attention
FLOPs excluded — a lower bound):

| | eff. TFLOPS @ pp512 | interpretation |
|---|---:|---|
| llama.cpp Qwen3-8B Q8_0 | **225** | nsys: MMQ+fixup = 79.9 % of its window → GEMM-window rate ≈ 281 TFLOP-equivalent — **above the 253 TFLOPS f32acc ceiling**, only possible in the un-quartered INT8 (int32-acc) ceiling class (~696 TOPS nominal) |
| imp Qwen3-8B Q8_0 | 138 | GEMM-window rate ≈ 138 / 0.615 ≈ 224 ≈ **88 % of the f32acc ceiling** — imp's cuBLAS is nearly maxed *within its ceiling class* |
| imp + fp16_acc | 172 | f16-accumulate lifts the ceiling ~4× (full-rate HMMA ~1956 TFLOPS measured); the cuBLAS f16-acc kernels deliver ~1.5× at these shapes |
| llama.cpp Qwen3-14B Q6_K | 193 | Q6_K MMQ unpack is costlier → llama's own efficiency drops — which is why this gap (1.24×) is the smallest |
| imp Qwen3-14B Q6_K | 155 | |

Measured `mma.sync` ceilings on this silicon (PR #606,
`tests/bench/mma_peak_saturated.cu`): FP16-f32acc **253 TFLOPS** (¼ datasheet),
FP16-f16acc ~1956 TFLOPS (full rate), FP4 mxf4nvf4 **2019 TOPS** (½ datasheet),
FP8-f32acc 496 TOPS (¼). INT8/int32acc is *not* quartered on GeForce (~696
TOPS nominal in `tools/roofline/config.json`; not yet mma-bench-measured).

### 2.4 Competitor kernel evidence (nsys, this box, today)

`llama-bench -p 512 -n 0` profiled with host nsys mounted into the container
(profiles: `/tmp/prefill_gap/nsys/llama_{q8,moe}.nsys-rep`):

- **Qwen3-8B Q8_0**: `mul_mat_q<Q8_0>` **73.2 %** + `mul_mat_q_stream_k_fixup`
  6.7 % + `quantize_mmq_q8_1` **2.6 %** + `flash_attn_ext_f16` 3.8 % + norms/
  rope/glue ~9 %. **Zero cuBLAS/nvjet kernels.**
- **Qwen3-30B-A3B Q4_K_M**: `mul_mat_q<Q4_K>` 55.6 % + `mul_mat_q<Q6_K>` 13.5 %
  + `mm_ids_helper` (expert routing) 4.0 % + fixup 3.4 % + `quantize_mmq_q8_1`
  2.9 % + `flash_attn_ext_f16` 3.7 %. Fully fused per-expert MMQ, no
  materialization.

(Note: `GGML_CUDA_FORCE_CUBLAS=1` is a silent no-op in build 19e92c3 — bench
numbers under that env are unchanged and no `FORCE_*` init log appears. Don't
use it for decomposition experiments.)

## 3. Where imp's prefill time goes (calibrated attribution)

From the 2026-06-07 roofline run (3 container restarts, nsys time shares, ncu
counters, clocks locked), re-parsed at config v3 — **%-of-ceiling values below
are against the real, measured ceilings**, not datasheet:

| Cell | #1 class | #2 class | #3 class |
|---|---|---|---|
| q8-dense pp512 | gemm_cublas **61.5 %** @ 83 % ceiling | dequant_q8 **29.8 %** @ 38 % BW | attn_fa2 2.6 % @ 19 % |
| q8-dense pp4096 | gemm_cublas 53.3 % @ 85 % | dequant_q8 29.8 % @ 38 % BW | attn_fa2 12.0 % @ 32 % |
| q4k-moe pp512 | dequant_q4k **64.3 %** @ 39 % BW | gemm_cublas 29.5 % @ 51 % | attn_fa2 1.8 % |
| q4k-moe pp4096 | dequant_q4k 63.1 % @ 39 % BW | gemm_cublas 24.8 % @ 55 % | attn_fa2 8.4 % @ 33 % |
| nvfp4-dense pp512 | gemm_cutlass_nvfp4 **61.1 %** @ 60 % ceiling | attn_fa2 9.5 % @ 19 % | rmsnorm 8.3 % |
| nvfp4-dense pp4096 | gemm_cutlass_nvfp4 50.1 % @ 60 % | attn_fa2 **31.4 %** @ 30 % | rmsnorm 6.7 % |
| nvfp4-moe pp512 | gemm_grouped_nvfp4 **48.5 %** @ 52 % BW | attn_fa2 10.9 % @ 18 % | gemm_cutlass 10.0 % @ 20 % |
| nvfp4-moe pp4096 | attn_fa2 **40.0 %** @ 33 % | gemm_grouped_nvfp4 30.4 % @ 44 % BW | gemm_cutlass 8.1 % @ 20 % |
| gemma-3-12b (hd=256) pp2048 | gemm_cublas **77.1 %** @ 88 % | dequant 14.3 % @ 33 % BW | attn_legacy 1.9 % (99 % of its attn) |

Legacy materialized attention is **0.0 % on every hd=128 model** (the
2026-05-31 "~18 % materialized attention" figure is dead — fixed by #525/#478).
Only hd≠128 (gemma-3/4) still runs it, at 3.6–6.9 % of window.

## 4. Gap decomposition by family

### 4.1 GGUF dense (Q8_0: 1.63× — two stacked structural taxes)

imp pipeline: `dequant_q8 → HBM round-trip → cuBLAS FP16 (f32 acc)`.
llama.cpp pipeline: `quantize activations to q8_1 (2.6 %) → fused MMQ on INT8 TC (int32 acc)`.

1. **Materialized source dequant**: ~30 % of imp's window at 38 % BW (write
   FP16 weights to HBM, cuBLAS reads them back). llama.cpp never materializes.
2. **The f32acc ceiling**: imp's GEMM runs at 83–86 % of a ceiling that is ¼ of
   the silicon's HMMA rate; llama.cpp's INT8 path is un-quartered.

The taxes compound: perfect dequant fusion alone (same ceiling) ≈ 1/(1−0.298)
= **+42 %** (8.4k → ~12k, still below llama's 13.7k); the ceiling fix alone
(`cublas_fp16_acc`) measured **+25 %** (→ 10.5k). llama.cpp is the existence
proof that fixing both closes the gap entirely on this silicon.

Q6_K (1.24×) has the same structure, but llama's own Q6_K MMQ efficiency drops
(193 vs 225 eff TFLOPS) — the gap narrows without imp doing anything better.

### 4.2 GGUF MoE (2.4–2.6× — the dequant tax dominates)

`dequant_q4k` is **63–65 % of the window at 39 % BW** (Qwen3-30B-A3B): every
expert's weights are dequantized to FP16 *per chunk* before the expert GEMM,
and the expert GEMMs are small-M, reaching only 51–56 % of the quartered
ceiling. llama.cpp runs fused per-expert MMQ with zero materialization (§2.4).
Eliminating materialization entirely would be worth up to ~2.8× here — this is
issue #599's territory and the largest single prefill liability in the stack.
gemma-4-26B (2.5×) stacks the hd≠128 legacy-attention share (#603) on top.

### 4.3 GGUF hybrid (Qwen3.6-35B, 2.1×)

Same expert dequant tax plus GDN in/out projections that are quality-locked to
FP16 GEMM (NVFP4 there regresses −9 to −20 %, refuted 2026-05-30).

### 4.4 NVFP4 vs vLLM (attention-scaling gap at long context)

At pp4096 dense, imp spends **50.1 %** in the CUTLASS NVFP4 GEMM @ 60 % of the
real FP4 ceiling and **31.4 %** in FA2 @ 30 % of the quartered FP16-f32acc
ceiling. Decomposition of the 1.78× (14B):

- **Chunking is NOT the gap**: prefill-chunk A/B today (pp4096, 2 restarts
  each, spread < 0.3 %): chunk=512 (default) 15.9k → chunk=2048 **16.9k
  (+5.5 %)** → chunk=0 (single chunk) 16.9k. Even unchunked, imp is 1.68×
  behind — the gap is kernel-level, not scheduling.
- **Attention is the largest identified term**: FA2's ceiling class is also
  quartered (f32 accumulate); at 30 % of 253-class TFLOPS it delivers ~52
  TFLOPS of attention compute. vLLM's FlashInfer prefill attention (with fp8
  KV) scales with context instead of eating the window. Lifting attention to
  vLLM-class speed ≈ −20 % window at pp4096 (≈ 16k → 20k); the rest of the
  distance to 28k is spread across the GEMM's structural 60 %-of-ceiling
  (TMA-WS pipeline cost, #606), `quant_cvt`/rmsnorm overhead passes (~10 %
  combined, vLLM fuses these into epilogues), and vLLM's fp8 KV traffic.
- **MoE (2.65× @ pp4096)**: same attention term (FA2 hits 40 % of window) plus
  the grouped GEMM at 44 % BW (#601) vs vLLM's FlashInfer-CUTLASS grouped path.

imp's outright win below ~2k context (and 2× at MoE pp512) means the gap only
matters for long-context bulk-ingest workloads.

## 5. Levers, ranked

| # | Lever | Evidence | Est. gain | Effort/risk |
|---|---|---|---|---|
| 1 | **INT8-MMQ prefill GEMM (fused dequant + int8-mma, int32 acc) for Q8_0/Q*_K, incl. grouped/MoE variant** | llama.cpp existence proof (nsys-verified); INT8 TC un-quartered; imp already owns q8_1 activation quant + dp4a int8 infra | dense Q8 to parity (+60 %); MoE Q4_K toward parity (+130–150 %) | multi-week; block_q8_1 36-B alignment needs SoA repack (#598 finding); small-M-per-expert regime must be covered |
| 2 | **`gemm.cublas_fp16_acc` default-ON for non-Gemma archs** | today: +24.9 % q8 pp512, +23.9 % pp2048, decode neutral, PPL flat on Qwen3-8B (+0.02 %); Gemma-3 +0.7 % PPL → per-arch deny | **+25 % dense GGUF prefill, now** | config flip + per-arch deny-list + PPL ship-gate |
| 3 | **Q4_K fused/faster dequant (#599)** — short of full MMQ, lift dequant from 39 % to ~70 % BW | 63–65 % of q4k-moe window | +25–30 % MoE GGUF prefill | kernel work, no quality risk |
| 4 | **FA2 kernel family (#597)** — the NVFP4 long-ctx lever | 18–33 % of (quartered) ceiling at 31–40 % window; chunk A/B proves it's not scheduling | up to ~+25 % NVFP4 pp4096 | structural; consider f16-acc QK^T (doubles ceiling class; needs quality gate) and fp8-KV-aware loads |
| 5 | **Prefill chunk 512 → 2048** | +5.5 % pp4096 (both restarts, < 0.3 % spread); chunk=0 equal to 2048 | +5–6 % NVFP4 long-ctx prefill, free | VRAM workspace + TTFT/streaming granularity tradeoff; per-arch scope check |
| 6 | **hd≠128 prefill coverage (#603)** | gemma legacy attn 3.6–6.9 % of window (92–99 % of its attention) | small, gemma-only | medium |
| 7 | **rmsnorm (#602) / quant_cvt epilogue fusion** | 7–8 % + 3–5 % of NVFP4 windows at low BW% | ~5 % NVFP4 prefill | small kernel fixes |

### Dead-ends — measured, do not re-pursue

- **Forge HMMA-MMQ (`feat/q4k-mmq-hmma`)**: ties cuBLAS, can't beat it — **with
  a reframe**: that experiment lived entirely in the f32acc ceiling class, so
  its "beat cuBLAS = refuted" verdict is *no evidence against an INT8 MMQ*
  (llama.cpp proves the int8 class works). Forge's small-M and split-K
  refutations stand.
- **`GGML_CUDA_FORCE_CUBLAS` decomposition experiments** — silent no-op in
  current llama.cpp builds (§2.4).
- **NVFP4 GEMV occupancy / KPAR→MR rerouting** (decode, 2026-05-30) — refuted.
- **GDN in/out projections in NVFP4** — quality-locked, regresses (2026-05-30).
- **`32F_FAST_16F` cuBLAS compute type** — does nothing on this cuBLAS (#606).
- **`convert_scales_sfatom` in NVFP4 profiles** — load-time artifact (2026-05-31).

## 6. Methodology appendix

- Raw logs: `/tmp/prefill_gap/results/` (session-local), clock trace
  `/tmp/prefill_gap/clocks.csv`, llama.cpp kernel profiles
  `/tmp/prefill_gap/nsys/llama_{q8,moe}.nsys-rep`.
- Roofline raw data: `tools/roofline/history/raw/e66cfa45-dirty_20260607_042900/`
  (committed; re-parse with `tools/roofline/roofline report --run …` — config
  v3 ceilings apply automatically).
- Ceiling calibration: `tests/bench/mma_peak_saturated.cu` (PR #606).
- vLLM prefill rate = prompt_tokens / median wall-time of `generate()` with
  `max_tokens=1`, prefix cache disabled — includes ~5–15 ms engine overhead per
  request, i.e. slightly *under*-estimates vLLM at small pp (conservative
  against imp; irrelevant next to the observed small-M pathology).
- vLLM defaults to fp8 KV cache; imp ran FP16 KV — a small structural
  asymmetry in vLLM's favor at long context, noted in §4.4.
- llama-bench σ shown in §2.1; imp restart pairs shown raw.
