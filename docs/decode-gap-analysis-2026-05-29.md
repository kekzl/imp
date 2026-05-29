# Decode competitive-gap analysis (2026-05-29, post-FA2 / CUDA 13.3)

Fresh nsys profiling of the decode hot path on the post-#478/#479 build, focused on
**Qwen3.6-35B-A3B-NVFP4** — the one decode benchmark imp loses to llama.cpp (−31%).
nsys WSL2 recipe: `nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda`
(bare `nsys profile` silently produces no report on WSL2) + `--set runtime.cuda_graphs=never`
to expose per-kernel times.

## Dense decode (Qwen3-14B-NVFP4) — at the frontier

~70% CUTLASS NVFP4 `GemmUniversal` (M=1 weight matmuls, HBM-roofline-bound) + ~6% inherent
per-step activation quantization. No exploitable lever (matches the long-standing 87%-in-GEMVs
profile; GEMV tuning and ptxas tuning are both refuted). Note: `convert_scales_sfatom` looks
like ~10-15% in a short capture but is **one-time init** (561 instances regardless of decode
length) — don't chase it; isolate steady-state when profiling decode.

## Qwen3.6-35B-A3B hybrid decode — the −31% gap is mostly structural

| share | kernel | precision | status |
|------:|--------|-----------|--------|
| 29.4% | `gemv_fp16_kernel` | FP16 | GDN `ssm_in`/`ssm_out` + attention projections — **NVFP4-excluded for quality** |
| 14.3% | cuBLAS `gemvx` | FP16 | **lm_head** (248k-vocab) — **closable** (see below) |
| 8.8%  | `paged_attention_gqa` | — | memory-bound |
| ~20%  | NVFP4 MoE gate_up/swiglu/kpar | NVFP4 | already fast |
| 3.4%  | `gdn_scan_fused` | FP16 | recurrent scan |

**~44% of decode is FP16 GEMV.** The GDN recurrent projections are deliberately kept FP16:
NVFP4 accumulates quantization error in the recurrent state H across tokens and degrades quality
on 9B+ models (`pre_dequant_phase3_nvfp4_decode.cu`). That part of the gap is a **quality
tradeoff, not a missed optimization** — closing it needs quality-preserving low-precision GDN
(research; prior model-level NVFP4-SSM attempts are refuted).

## Shipped: NVFP4 lm_head opt-in for hybrids (#479) — +11.4% decode

The 14% lm_head slice **is** closable. NVFP4 lm_head was hard-excluded for GDN models on an older
refutation; re-measured with the current quantize-FP16→NVFP4 path (#465 method): quality holds
(matches FP16 on math/primes/explanations; output lands within FP16's own distribution) and decode
is **+11.4%** (tg128 219.6→244.7). Shipped as opt-in `gemm.nvfp4_lm_head_gdn` (default false).

**Methodology gotcha:** Qwen3.6-35B (GDN+MoE) is **non-deterministic at temp=0** — two FP16 greedy
runs diverge (MoE routing / GDN atomics). So **greedy-token A/B is invalid** on this model; quality
must be judged by perplexity or many-sample statistics, not a single greedy diff.

## Next step: a perplexity harness (spec)

To rigorously decide quality questions on non-deterministic models (e.g. flipping the lm_head
default-on for hybrids, or evaluating NVFP4 GDN projections), imp needs a teacher-forced perplexity
tool — it has none today (server logprobs are completion-only; `forward_logits` slices to the last
token in prefill). Proposed:

- **Engine:** an additive `compute_perplexity(tokens)` that runs the existing transformer stack,
  applies the lm_head to **all** positions' final-normed hidden states → `[n, V]` logits, then a
  reduction kernel sums `-log_softmax(logits[i])[tokens[i+1]]`; return `exp(mean)`. Additive and
  bench-only — does not change the production `forward_logits` (last-token slice) path.
- **CLI:** `imp-cli --perplexity <textfile>` → prints PPL.
- **Use:** PPL(FP16 lm_head) vs PPL(NVFP4 lm_head) on a fixed corpus → a clean, determinism-proof
  quality delta; unblocks the lm_head default-flip and the GDN-projection research.

Reproducibility harnesses for this analysis live in `tools/analysis/` (`lmhead_nvfp4_qwen36_ab.sh`,
`lmhead_greedy_agree.sh`, the cuTile/ptxas/sawtooth probes).
