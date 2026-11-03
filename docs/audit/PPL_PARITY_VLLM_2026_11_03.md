# imp vs vLLM perplexity parity (2026-11-03)

Goal: first measurement of release bar 1 (`docs/GOAL.md`: PPL within 0.5 % of vLLM on the same SafeTensors checkpoint, documented opt-out trades excluded) against vLLM instead of llama.cpp ([archive](../archive/PPL_PARITY_2026_07_12.md)).

[PROV: commit=14e99cd6 (branch ppl-vs-vllm-parity = main 87a61491 + harness) date=2026-11-03 hw=RTX5090 cuda=13.4.3 path=imp-cli --perplexity
       imp=imp:pplvllm2 (make build, image 07679c8c46f8, imp.tree f362e6bc) pre-#3019 arms imp:pplvllm (669c3396a49a, tree 2ffbc54f, main 5f7291f9)
       vllm=vllm/vllm-openai:v0.31.0 c1c9f6fd5c10 harness=tools/analysis/ppl_vs_vllm.sh + ppl_vs_vllm.py (md5 b18c0262 / 23289cce, band 83af0175)
       plans=/mnt/data/scratch/pplvllm_tools/gpu_plan.sh (8cd19082), gpu_extra.sh (b9f3c616), gpu_main.sh (5320396e), where.py (c8394673)
       logs=/mnt/data/scratch/pplvllm_runs/{run.log,summary.tsv,bands.txt,where_*.txt,<model>/<corpus>/{imp-*,vllm-*}.{log,nll}}]

## Method

| Item | Setting |
|---|---|
| Tokens | imp's own `--perplexity` stream (`diagnostics.dump_tokens`) is fed to vLLM as `TokensPrompt`; both engines score identical ids |
| Window | whole corpus as one sequence, rows 0..n-2 (row i = NLL of token i+1), no stride; imp chunked prefill accumulates per chunk |
| imp | `IMP_DETERMINISTIC=1`, `speculative.mtp_k=0`, `diagnostics.ppl_dump=<file>` (6 decimals per row) |
| vLLM | offline `LLM`, `SamplingParams(max_tokens=1, prompt_logprobs=0)`, prefix caching off, `max_num_batched_tokens` 2048 (Gemma-4 1024, `language_model_only`), `logprobs_mode` raw, `gpu_memory_utilization` 0.85 |
| vLLM kernels (sm_120) | NVFP4 dense `FlashInferCutlassNvFp4LinearKernel` (W4A4, checkpoint input scale), FP8 `FlashInferFP8ScaledMMLinearKernel`; attention FLASH_ATTN with BF16 KV, FLASHINFER with FP8 KV |
| Reference | vLLM PPL moves per process start, so the reference is the pool of N starts (per-row NLL averaged, `ppl_vs_vllm.py band`); Δ = imp / pooled - 1 |
| Corpora | Qwen3: `ppl_corpus_45k.txt` (13 537 ids). Gemma-4-it: `ppl_corpus_gemma4_turn.txt` (704) and 45k in turn framing (14 676, 3000-char turns as in `awq_moe_gemma4_ab.sh`); raw 45k is out of Gemma-4's regime (archive addendum) |

Token parity, HF tokenizer (transformers 5.17, vLLM's text path) vs imp: Qwen3 checkpoints IDENTICAL (13 537); Gemma-4 identical except imp's leading BOS (id 2), which HF does not add.

## Results (main 87a61491)

Opt-out arm = `gemm.nvfp4_lm_head=off` (the GOAL bar-1 trade). KV: imp F16 / vLLM BF16 unless noted.

| Model | Corpus | imp default | imp opt-out | vLLM starts | vLLM pooled | Δ opt-out | Δ default | vLLM spread | per-row NLL sd, imp-vLLM / vLLM-vLLM | Bar |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B NVFP4 (cortecs) | 45k | 11.2352 | 11.2178 | 11.1627, 11.1962, 11.2264 | 11.1951 | +0.203 % | +0.359 % | 0.571 % | 0.447 / 0.444 | PASS |
| Qwen3-8B FP8 | 45k | 10.7754 | 10.7604 | 10.7587, 10.7587 | 10.7587 | +0.016 % | +0.155 % | 0.000 % | 0.190 / 0.000 | PASS |
| Qwen3-8B FP8, vLLM FP8 KV | 45k | 10.7754 | 10.7604 | 10.7290, 10.7833 | 10.7561 | +0.040 % | +0.179 % | 0.506 % | 0.255 / 0.225 | PASS |
| Qwen3-30B-A3B NVFP4 (Modelopt), FP8 KV both | 45k | 10.9566 | 10.9265 | 10.9399, 10.9429, 10.9757 | 10.9528 | -0.240 % | +0.035 % | 0.328 % | 0.487 / 0.492 | PASS |
| Qwen3-30B-A3B, KV F16 / BF16 | 45k | - | 10.8477 | 10.9024, 10.8140 | 10.8581 | -0.096 % | - | 0.818 % | 0.440 / 0.446 | PASS |
| Gemma-4-26B-A4B NVFP4 | 45k_turn | 17.5295 | 17.4793 | 17.6705, 18.0092, 17.6947 | 17.7908 | -1.751 % | -1.469 % | 1.917 % | 0.957 / 0.951 | imp lower, see below |
| Gemma-4-26B-A4B NVFP4 | turn (703 rows) | 15.4533 | 15.4281 | 15.7680, 16.3227, 15.4452 | 15.8411 | -2.607 % | -2.448 % | 5.681 % | 1.253 / 1.343 | no signal |

- vLLM `--enforce-eager` single starts: Qwen3-8B NVFP4 11.2188, Gemma-4 45k_turn 18.1573, turn 15.0096 (pooled with them: Δ opt-out +0.150 %, -2.250 %, -1.286 %).
- imp is bit-reproducible: repeat arms give byte-identical NLL files (Qwen3-8B NVFP4, Qwen3-30B-A3B, Gemma-4 both corpora).
- vLLM is not: NVFP4 starts spread 0.33-1.92 % on the long corpora; Qwen3-8B FP8 with BF16 KV is the one stable vLLM cell (two starts equal to 4 decimals).
- imp-vs-vLLM per-row sd equals vLLM's own start-to-start sd in every NVFP4 cell: imp's per-token disagreement with vLLM is no larger than vLLM's with itself.
- KV dtype: `kv_cache.dtype=auto` resolves FP16 for Qwen3-8B FP8 in imp (dense qwen3 not on the FP8-hint list) and FP8 E4M3 for Qwen3-30B-A3B; `kv_cache.fp8_scale=checkpoint` (48 of 48 layers) is byte-identical to `legacy` on Qwen3-30B-A3B.

### Gemma-4: imp below vLLM

Breakdown of ln(PPL_imp / PPL_vllm_pooled) = -0.01766 on 45k_turn (`where_gemma_45k_turn.txt`):

| Slice | Rows | Contribution | imp wins |
|---|---|---|---|
| turn framing targets | 270 | -0.00352 | 55.9 % |
| first 32 text tokens of a turn | 480 | -0.00057 | 51.2 % |
| body | 13 925 | -0.01357 | 53.4 % |
| pooled-vLLM NLL deciles 0-5 (easy) | 8 805 | +0.01040 | - |
| deciles 8-9 (hard) | 2 935 | -0.02561 | - |

- The gap is spread over the whole body, not a few framing rows; imp is slightly less confident on easy tokens and much less wrong on hard ones.
- Single vLLM starts disagree by 19.5 nats on a body token (row 6411: 8.746 / 0.005 / 19.513); per-start logit noise raises the expected NLL (Jensen), so a noisier engine reads a higher PPL.
- The gap appeared with #3019 (act block scales clamp at 2^-9): pre-#3019 imp read 17.8012 on 45k_turn, Δ +0.059 % to the same pool.
- Verdict: not an imp defect (imp lower, deterministic, inside vLLM's per-row noise); a deterministic reference (HF BF16 dequant forward) is the next step to rank the two.

## Pre-#3019 imp (main 5f7291f9, same vLLM pools)

| Model | Corpus | imp default | imp opt-out | Δ opt-out | Δ default |
|---|---|---|---|---|---|
| Qwen3-8B NVFP4 | 45k | 11.2587 | 11.2434 | +0.432 % | +0.568 % |
| Qwen3-8B FP8 (vs BF16-KV pool) | 45k | 10.7754 | 10.7604 | +0.016 % | +0.155 % |
| Qwen3-30B-A3B (FP8 KV) | 45k | 10.9678 | 10.9375 | -0.140 % | +0.137 % |
| Gemma-4 | 45k_turn | 17.8489 | 17.8012 | +0.059 % | +0.327 % |

Reproduce: `MODEL=<dir> CORPORA="45k=<abs path>" IMP_ARMS="default: lmhead_off:gemm.nvfp4_lm_head=off" VLLM_ARMS="default: default2: default3:" bash tools/analysis/ppl_vs_vllm.sh`, then `ppl_vs_vllm.py band <imp_tokens> <imp nll> <vllm nll>...`.
