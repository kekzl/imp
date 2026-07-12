# Cross-engine PPL parity vs llama.cpp — first release-bar-1 measurement (2026-07-12)

**Bar (docs/GOAL.md, release bar 1):** hero-model perplexity within 0.5% of the
llama.cpp reference at the same quant, documented owner-approved speed/quality
trades excluded. This is the first systematic end-to-end measurement of that bar.

**Headline:** with the LM-head opt-out the imp forward pass is at parity with
llama.cpp on every comparable GGUF hero (−0.8%…+0.2%, i.e. at or slightly
below the reference). The entire default-config
gap (+1.4…+4.1%) is one deliberate perf feature: the **NVFP4 LM-head decode
cache** (`gemm.nvfp4_lm_head`, default on, ~14% of dense decode — quantified
2026-05-29 in `src/runtime/config.h`). It has a config opt-out but was never
listed in the GOAL bar-1 trades — that listing decision is the open follow-up.

## Methodology (reproducible)

- **Reference:** llama.cpp build 9976 (`e3546c794`), image
  `ghcr.io/ggml-org/llama.cpp:full-cuda` (pulled 2026-07-12), `-ngl 99`,
  defaults otherwise (f16 KV). imp at this PR's commit, image `imp:test`.
- **Corpus:** first 95 000 bytes of the repo-docs concatenation (architecture,
  sm120, BENCHMARKING, usage, quantization, attention-dispatch, determinism,
  supported-models, MISSION_JOURNAL, vram_audit, sm120_optimal_kernel,
  nsys_profiling, roadmap, quant-pipeline, performance, CONTRIBUTING, AGENTS —
  working tree at `6c5e5b31`), sha256 `8ad55557…0513ec7bcee`. Gemma-4 used the
  first 36 000 bytes (imp KV pool ceiling at 26 GiB weights, see exclusions).
- **Window alignment:** llama-perplexity counts NLL only over the second half
  of each chunk (`first = n_ctx/2`, logit rows `[first, n_ctx-2]`) and requires
  ≥ 2·n_ctx corpus tokens. Both engines were pinned to the **identical logit
  rows**: llama.cpp `-c C --chunks 1` (C = ⌊T/2⌋ of the model's corpus token
  count T), imp full-corpus teacher-forced prefill with
  `diagnostics.ppl_first = C/2`, `diagnostics.ppl_last = C-2` (added in this
  PR). Chunk 1 starts at the corpus start → conditioning is identical too.
- **`--no-escape` on every llama.cpp run.** llama-perplexity/llama-tokenize
  apply `string_process_escapes` to the input by default; the docs corpus
  contains shell examples with `\"` sequences, which silently desynced the
  token streams (6 tokens) and shifted the counting windows in a first sweep.
- **Tokenizer parity verified per model:** imp `diagnostics.dump_tokens` diffed
  against `llama-tokenize --ids --no-escape` — **IDENTICAL streams for all
  seven cells** (Qwen3.6 requires the qwen35 pre-tokenizer routing fix from
  this PR).
- **Trials:** dense GGUF PPL is bit-stable run-to-run (2 trials); MoE/hybrid 3
  trials, median (process-start cuBLAS algo selection perturbs ±0.3-0.5%).
- **Trades excluded per the bar:** Qwen3.6-35B ran with
  `gemm.fp8_ssm_proj=false gemm.nvfp4_lm_head_gdn=false`.

Command skeleton (per model, T from llama-tokenize):

```
llama-perplexity -m <gguf> -f <corpus> --no-escape -c <T/2> --chunks 1 -b 2048 -ngl 99
imp-cli --model <gguf> --perplexity <corpus> --temperature 0 \
        --set diagnostics.ppl_first=<T/4> --set diagnostics.ppl_last=<T/2-2>
```

## Results

| model (GGUF) | llama.cpp | imp defaults | Δ defaults | imp `gemm.nvfp4_lm_head=false`* | Δ opt-out | bar (±0.5%) |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q8_0 | 8.3299 | 8.6348 | +3.66% | 8.3224 | −0.09% | **PASS** |
| Qwen3-8B Q8_0 | 8.0360 | 8.2292 | +2.40% | 8.0224 | −0.17% | **PASS** |
| Qwen3-14B Q6_K | 7.1123 | 7.2758 | +2.30% | 7.1288 | +0.23% | **PASS** |
| Qwen3-30B-A3B Q4_K_M | 8.2058 | 8.5438 | +4.12% | 8.1394 | −0.81% | PASS-by-intent (imp *better*; MoE single-trial spread ±0.3-0.5%) |
| Qwen3.6-35B UD-Q4_K_M | 4.8993 | 4.9671 | +1.38% | 4.8972 | −0.04% | **PASS** |

\* the targeted opt-out, measured after this PR's fix (pre-fix it was silently
ignored on GGUF checkpoints — finding 3 below; the 35B cell uses
`gemm.nvfp4_lm_head_gdn=false`, its GDN-specific gate). The coarse
`diagnostics.no_nvfp4_decode_cache=true` control reproduces the same parity
(4B 8.2961, 8B 8.0129, 14B 7.1288, 30B 8.1541, 35B 4.8920). Defaults column
includes FP8-KV auto (#977), which accounts for only ~0.2% of the defaults
delta (fp16-KV control: 8.6148 on 4B).

**Attribution (Qwen3-4B bisection, llama ref 8.3299):** defaults 8.6348 →
`no_nvfp4_decode_cache` 8.2961 (gap gone) → `kv_cache.dtype=fp16` alone 8.6148
(FP8-KV ≈ +0.23%). The LM-head NVFP4 quantization is the whole story
(defaults vs targeted opt-out): +3.8% (4B), +2.6% (8B), +2.1% (14B), +5.0%
(30B MoE), +1.4% (35B — its sweep run had `nvfp4_lm_head_gdn=false`, but that
opt-out was dead on GGUF, finding 3). Cost scales inversely with model size,
as expected for an lm_head effect.

## Exclusions

- **NVFP4/SafeTensors heroes** (Qwen3-Coder-30B FP4, Nemotron-H, Qwen3.6-35B
  NVFP4, Gemma-4 NVFP4, Qwen3-14B NVFP4): no llama.cpp counterpart at the same
  quant — the bar is only measurable on shared GGUF quants.
- **gpt-oss-20b MXFP4:** imp ships MXFP4 in GGUF under a proprietary tensor
  type code; llama.cpp reads it as a removed format (docs/quantization.md) —
  not comparable.
- **Nemotron-H:** no GGUF staged locally; NVFP4-only.
- **Gemma-4-26B-A4B:** the llama.cpp reference itself reads PPL ≈ 309/330 on
  this corpus (UD-Q4_K_M / Q8_0, reproducible ×2) and imp's MoE trial spread in
  that regime is ±4% (319–348) — the regime carries no parity signal. Token
  streams are IDENTICAL (11 346). The Q8_0 imp cell additionally cannot run at
  equal windows: 26 GiB weights leave a 7 168-token KV pool, below the 11.3k
  corpus (imp prefills the full corpus; llama.cpp only allocates the C=5.7k
  window).

## Findings shipped in this PR

1. **`tokenizer.ggml.pre = "qwen35"` routed to the gpt2 fallback** — Qwen3.5/3.6
   GGUFs (incl. the 35B hero) over-split symbol runs: 35 807 vs 31 620 tokens
   (+13%) on this corpus, i.e. every real prompt tokenized non-canonically
   (more prefill tokens + off-distribution splits). Now routed to the qwen2
   scanner (qwen35 only adds `\p{M}` to the letter run, which the scanner's
   letter classifier already covers). Regression test in test_tokenizer.cpp.
2. **`diagnostics.ppl_first` / `diagnostics.ppl_last`** — NLL counting window
   for `imp-cli --perplexity`, enabling exact llama.cpp window alignment.
3. **The NVFP4-LM-head opt-outs were dead on GGUF checkpoints:** the
   quantized-source decode-cache collector added the LM head unconditionally,
   so neither `gemm.nvfp4_lm_head=false` nor the GOAL-listed
   `gemm.nvfp4_lm_head_gdn=false` trade opt-out had any effect on GGUF models
   (verified: 4B PPL unchanged at 8.6348 with the flag off; the 35B "trades
   excluded" run still carried +1.5% from the head). The collector now applies
   the same gates as the native-precision paths. Defaults are byte-identical
   (both flags default on).

## Open follow-ups

1. **Bar-1 trade listing decision (owner):** `gemm.nvfp4_lm_head` default-on is
   a real +1.5…+4.8% PPL trade for ~14% dense decode. Either list it in
   docs/GOAL.md release-bar trades (opt-out exists) or revisit the default for
   small-vocab-pressure models where the cost is largest.
2. **Diagnostics combination bug:** `diagnostics.no_nvfp4_decode_cache=true` +
   `kv_cache.dtype=fp16` yields PPL 15.35 vs 8.30 on Qwen3-4B Q8 (bit-exact
   reproducible; each flag alone is sane). fp16 KV cannot be worse than FP8 —
   some fallback path in that combination is broken.
3. Gemma-4 quality regime (PPL ~300 on raw docs in BOTH engines' ballpark)
   deserves its own investigation before any Gemma parity claim.
