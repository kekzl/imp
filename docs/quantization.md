<!--
layer: L1
audience: operators
verified: 2026-08-28
commit: be825e4a
-->

# Quantization

imp reads GGUF quantization (llama.cpp-compatible files, loaded directly) and SafeTensors NVFP4
prequant (external calibration tools). Per-model picks: [`supported-models.md`](MODELS.md).
Benchmark numbers: [`performance.md`](performance.md).

## Formats and where they show up

| Format | Bits / weight | Source | Used for |
|---|---:|---|---|
| Q8_0 | 8.0 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q6_K | 6.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q5_K_M | 5.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q4_K_M | 4.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q4_0 | 4.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| IQ4_NL / IQ4_XS | 4.5 / 4.25 | GGUF | dequant->FP16 cache decode + dequant->cuBLAS prefill (no dp4a/MMVQ kernels) |
| FP8 E4M3 | 8.0 | runtime | KV cache (opt-in), prefill weight cache |
| INT8 | 8.0 | runtime | KV cache (opt-in) |
| INT4 | 4.0 | runtime | KV cache (long-ctx, opt-in) |
| NVFP4 | 4.0 | SafeTensors | weights (decode + prefill), KV cache |
| MXFP4 | 4.5 | GGUF | weights (decode + prefill attention) |

GGUF is mmap'd and uploaded as-is; `*.K` quants store block scales in the format the dp4a kernels
expect. NVFP4 prequant arrives packed with FP8 E4M3 micro-scales (per-16) and an FP32 tensor scale;
imp registers it directly into the NVFP4 decode cache and CUTLASS NVFP4 GEMM path, no
re-quantization.

## NVFP4 prequant (SafeTensors)

Calibrated per-tensor scales via AWQ or SmoothQuant. Compatible producers:

| Tool | Status |
|---|---|
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (Modelopt) | Primary path. Coherent on Qwen3-Coder-30B, Mistral-3.2, Qwen3.6, Gemma-4 (after PR #88 lit up the CUTLASS NVFP4xNVFP4 prefill cache). |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | Loads; several models degenerate past ~30 tokens. See [roadmap](roadmap.md). Prefer Modelopt. |
| `imp-quantize` (in-tree) | **Experimental.** AWQ-calibrated with `--calib`, round-to-nearest without. Below a published export either way. |

### imp-quantize: converting a checkpoint yourself (EXPERIMENTAL)

> **Experimental.** Pipeline verified end to end; `--calib` recovers a measurable part of the
> quantization loss; the result still sits below a published Modelopt export. For evaluation and
> perf work, not shipping.

Turns a dense BF16/FP16 or block-scaled FP8 SafeTensors checkpoint into NVFP4:

```bash
# 1. one calibration pass over a corpus - writes per-channel activation stats
imp-cli --model ./Qwen3-1.7B --perplexity ./calib_corpus.txt --calibrate ./calib.bin

# 2. quantize using them
imp-quantize --model ./Qwen3-1.7B --out ./Qwen3-1.7B-nvfp4 --calib ./calib.bin
imp-cli --model ./Qwen3-1.7B-nvfp4 --prompt "Hello"
```

- Drop `--calib` (and step 1) for round-to-nearest; `--dry-run` previews without touching the GPU.
- Copies tokenizer/config, rebuilds the shard index for sharded sources, leaves embeddings, norms
  and (unless `--lm-head`) the LM head at full precision.
- Block-scaled FP8 sources work (reaches the FP8-only release lines: DeepSeek-V3, Qwen3.8's FP8
  line): each E4M3 weight is widened with its `weight_scale_inv` grid before quantizing. The pair
  is one unit, so weights kept at full precision are widened too, not copied (raw E4M3 without its
  scales is valid E4M3 that means something else). On Qwen3.8-27B-FP8 that is the MTP draft head,
  whose corruption would cost draft acceptance and nothing louder; honesty there costs 350 MiB of
  output.

#### Which layout it writes (`--format`)

| `--format` | tensors | declared in | read by |
|---|---|---|---|
| `modelopt` (default) | `.weight` (U8 packed) · `.weight_scale` (F8_E4M3) · `.weight_scale_2` (F32) | `hf_quant_config.json` | imp |
| `vllm` (= `compressed-tensors`) | `.weight_packed` · `.weight_scale` · `.weight_global_scale` | `quantization_config` in `config.json`, repeated in `recipe.yaml` | imp **and** vLLM |

```bash
imp-quantize --model ./Qwen3.8-27B --out ./Qwen3.8-27B-nvfp4 --format vllm
vllm serve ./Qwen3.8-27B-nvfp4      # loads as compressed-tensors NVFP4A16
```

Three silent-when-wrong differences beyond names:

- **Tensor scale stored inverted.** compressed-tensors stores a divisor (readers compute
  `1 / weight_global_scale`); Modelopt stores the multiplier. One convention's number under the
  other's name leaves every weight scaled by `absmax^2 / 36`: loads, generates, wrong.
- **`input_activations` stays null.** Weights-only tool; vLLM reads that as NVFP4A16. Declaring
  unmeasured activation quantization would make vLLM quantize activations against absent scales.
- **Everything left at source precision is listed in `ignore`.** vLLM builds an unquantized layer
  per entry; a missing module is one it hunts scales for that were never written.

#### Making the checkpoint smaller: what each exclusion costs

`--dry-run` breaks the full-precision third down by reason. Qwen3.8-27B (BF16 source 51.75 GiB):

| what | size in the output | quantizing it costs | quantizing it is |
|---|---:|---|---|
| `lm_head` | 2 425 MiB | nothing on top of what imp already pays | `--lm-head`, but see below |
| `embed_tokens` | 2 425 MiB | +0.94 % perplexity | not possible: no NVFP4 lookup |
| vision tower | 875 MiB | tower loads at source precision only | not possible |
| MTP draft head | 810 MiB | draft acceptance 81 % -> 0 (#1428) | refused |

**`--lm-head` costs nothing extra because imp already pays it**: a native LM head becomes an NVFP4
decode cache at load anyway (`gemm.nvfp4_lm_head`, auto -> on for native sources). Qwen3.8-27B:
perplexity 4.6158 either way, greedy output byte-identical over four prompts, weights
17 920 -> 16 192 MiB, checkpoint 19.15 -> 17.44 GiB. That is NVFP4 vs NVFP4; the default's cost
against a real BF16 head (`gemm.nvfp4_lm_head=off`, how every other engine runs it), Qwen3.8-27B,
248 320-token vocabulary:

| `gemm.nvfp4_lm_head` | perplexity | decode, 128 tokens greedy | ITL @4 | ITL @16 |
|---|---:|---:|---:|---:|
| `off` (BF16 head) | **4.5707** | 78.56 tok/s | 53.1 ms | 206.2 ms |
| `on` (default here) | 4.6158 (**+0.99 %**) | **86.70 tok/s (+10.4 %)** | **39.7 ms** | **190.1 ms** |

```
[PROV: commit=bca9e9e date=2026-08-16 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4
       cuda=13.3 path=nvfp4-decode-cache n=3-alternating-pairs
       cmd=`imp-cli --prompt … --max-tokens 128 --temperature 0 --set gemm.nvfp4_lm_head=off|on`;
       ITL from `tools/agent_bench.py --concurrency 1,4,16`, one run per arm;
       perplexity over ppl_corpus_45k.txt with runtime.deterministic_gemm=true]
```

- Decode: median of three alternating pairs (fixed arm order overstates); every pair favoured
  `on`. The #982 trade holds, cheaper than the +2.2 % recorded there.
- Not amortised away by concurrency: the head is read whole once per token (~11 % of the batch-1
  step, 2.43 GiB of 17.9 GiB weights); the advantage shrinks +25 % -> +8 % between 4 and 16
  concurrent decodes, never inverts. Other engines lack the feature, not the verdict: vLLM's
  `ParallelLMHead` accepts no scales (stops at `no module or parameter named
  lm_head.weight_global_scale`); Modelopt / llm-compressor put `lm_head` in `ignore` for W4A4.
- What `--lm-head` costs is the option: with a BF16 head in the checkpoint,
  `--set gemm.nvfp4_lm_head=off` buys the 0.99 % back; quantized in, it cannot. Use it when the
  model would not otherwise fit. Belongs with `--format modelopt`; the tool warns on `vllm`.
- **Embeddings stay at source precision; the price is known.** imp's embedding lookup handles
  F32/F16/BF16/Q8_0/Q6_K, not NVFP4 (vLLM leaves embeddings unquantized too). Measured via the
  exact quantize round trip written back at source precision: Qwen3-0.6B perplexity
  29.4204 -> 29.6982, +0.94 %, ~10 % off a 27B checkpoint.

#### Fused layers share one tensor scale

Engines merge `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` into single linears (vLLM's
`packed_modules_mapping`; imp's GDN path merges `in_proj_qkv`/`in_proj_z` and
`in_proj_b`/`in_proj_a`). A merged layer carries one tensor scale; three independently calibrated
scales leave two matrices dequantized against the third's. vLLM warns and continues; the amax
spread inside those groups reaches 3.7x on Qwen3-0.6B. `imp-quantize` decides the scale per fused
group in a pre-pass over the source (members are not guaranteed to share a shard). Also the better
quantization: Qwen3-0.6B, `ppl_corpus_45k.txt`, `deterministic_gemm`, round-to-nearest both arms:
**29.42** vs **30.40** for per-tensor scales. Looks-better-but-is-not: scaling by
`absmax / (6 x 448)` so the FP8 micro-scales fill their range (what published exports do) measured
**31.05**; imp writes `absmax / 6`. Readers multiply either convention back out; do not "fix"
without re-measuring.

#### Roles that stay full precision, and why

Three roles are 2-D and K-aligned (every shape check waves them through) and must not be quantized;
each found by measurement:

| role | why | found by |
|---|---|---|
| MLA latent projections (`kv_a_proj`, `kv_b_proj`) | the runtime slices and reshapes both | bisection on DeepSeek-V2-Lite: quantized, the checkpoint loaded and emitted cross-script garbage |
| MoE router (`.gate.weight`) | FP4 across 16 shared-scale values changes the top-k pick | measured separately, MLA pair already excluded |
| **fused Q+gate `q_proj`** (Qwen3.5 / Qwen3-Next `attn_output_gate`) | reported, **not excluded**: see below | #1273 |

The last row is a correction. The gate half feeds a sigmoid; E2M1 is coarsest near zero, where a
sigmoid is most sensitive: rounding only that half on a healthy GGUF twin reproduces the #1273
divergence (+0.0169 injected per attention block vs +0.0156 for the actual NVFP4 checkpoint; the Q
half below the noise floor; the same half in Q4_K healthy, so E2M1-specific). That divergence is
NOT a quality win for excluding it, though this doc briefly shipped asserting one. Qwen3.5-4B
(8 gated layers of 32), `ppl_corpus_45k.txt`, three runs each:

| arm | runs | spread |
|---|---|---|
| gate quantized (default) | **14.6665 / 14.6476 / 14.6716** | 0.16% |
| gate excluded | 14.8672 / 14.9339 / 14.8672 | 0.45% |
| BF16 reference | 12.6735 | |

Excluding is ~1.5 % worse, non-overlapping spreads, plus 1-4 % of checkpoint size; quality against
a corpus decides, not divergence against a twin. `--keep-attn-gate` stays, but the reason first
given ("lower gate share than the worst #1273 offender, 8/32 against 16/64") was wrong: every
gated checkpoint staged here has share 0.250 (Qwen3.5-4B 8/32, Qwen3.6-27B-Text 16/64,
Qwen3.6-35B-A3B 10/40, Ornith-1.0-35B 10/40); the flag stays for a model with a genuinely higher
share. imp cannot exclude half a tensor, so the whole `q_proj` stays. Detection is by shape (a
gated `q_proj` emits twice what its layer's `o_proj` consumes), not a config flag. Published
exports (llm-compressor, Modelopt) exclude `linear_attn.*` and quantize this tensor whole: same
gap.

That gap was once offered here as why every hybrid NVFP4 checkpoint degrades. **Not the reason**
(#1287): the final RMSNorm was the single norm without Qwen3.5/3.6's `gamma = 1 + W` offset, so
SafeTensors checkpoints scaled the last hidden state by `W` instead of `1 + W`. Every layer
correct, only the LM-head input wrong: coherent but much worse.

| checkpoint | before | after | its GGUF twin |
|---|---|---|---|
| Qwen3.6-27B-Text-NVFP4-MTP | 65.1275 | **7.5302** | none staged |
| Ornith-1.0-35B-NVFP4 | 16.1630 | **7.0702** | 6.4974 (1.09x) |
| Qwen3.6-35B-A3B-NVFP4 | 13.6486 | **6.8184** | 6.5465 (1.04x) |

2.1-2.5x their twins before, 1.04-1.09x after: ordinary NVFP4 cost. Dense and GGUF checkpoints
byte-identical either way (Qwen3-14B-NVFP4 10.0301, Qwen3-8B-NVFP4 11.6677, ornith Q4_K_M 6.4974).
Found because the degradation persisted at BF16 while per-layer hidden states matched an HF
`transformers` reference within 0.4 % across all 32 layers at 41 % perplexity off: states right,
output wrong, after the last layer. Method note: every degraded #1273 checkpoint was SafeTensors,
every healthy twin GGUF; format and load path confounded, and the conclusion followed the format.

#### What `--calib` does

NVFP4 error scales with the magnitude quantized; scaling an input channel's weights up buys it
precision at the others' expense, provided something divides the activation back down. Which
channels deserve it takes a forward pass; hence calibration. The transform is exact before
quantization, `y = x W^T = (x/s)(W diag(s))^T`; imp picks `s` by measurement (per candidate
exponent it quantizes with the real kernel and keeps the winner; `alpha = 0`, plain
round-to-nearest, is always in the grid). The compensating `1/s` folds into the producer (plain
NVFP4 checkpoint, no runtime support): four groups per layer, q/k/v and gate/up into the preceding
RMSNorm weight, `o_proj` into `v_proj`'s output rows (GQA-tied), `down_proj` into `up_proj`'s.

- The norm fold assumes plain multiplicative RMSNorm, so `--calib` **refuses** `(1 + g)`
  architectures (Gemma-class) rather than silently producing a different model.
- A norm can only be folded when every consumer is scaled: the two excluded roles (MLA latent
  projections, MoE router) never receive compensation, and the router reads exactly the norm the
  gate/up group folds into. `--calib` checks each norm's consumers and refuses the fold when an
  unscaled one exists, naming it. On DeepSeek-V2-Lite that leaves 2 of 108 groups scaled (layer
  0's dense MLP, the only routerless layer), one line per refusal.
- `--calib` does not calibrate MoE experts yet: the planner groups the dense FFN by name
  (`mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj`), not `mlp.experts.<e>.*`. Attention groups
  still calibrate; experts stay at round-to-nearest, stated per layer in the output.
- Calibrate on a different corpus than you score on: `tools/analysis/fetch_calib_corpus.sh`
  assembles general public-domain prose; scoring happens on `ppl_corpus_45k.txt`. One text for
  both reports a gain that exists only on it.
- `--calibrate` forces `runtime.deterministic_gemm`, not a formality: without it, two runs of the
  identical command differed on 94 % of recorded floats (up to 0.5 % each), and three checkpoints
  built from three such calibration files scored PPL 28.84, 28.94 and 28.48, a 1.6 % spread from
  which run produced the file. Forced, calibration file and checkpoint are bit-identical run to
  run.

#### Quality, measured

`imp-cli --perplexity` over `tools/analysis/ppl_corpus_45k.txt` (13 537 tokens), calibration over
36 058 tokens of general prose; chain reproducible with `tools/analysis/awq_ppl_ab.sh`:

| Model | BF16 | NVFP4 RTN | NVFP4 `--calib` | AWQ gain | gap to BF16 |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | 24.08 | 29.42 | **27.60** | -6.2% | +22.2% -> **+14.6%** |
| Qwen3-1.7B (2 shards) | 17.22 | 20.39 | **18.71** | -8.2% | +18.4% -> **+8.7%** |

- Re-measured 2026-08-17 after fused layers started sharing a tensor scale; both arms moved
  (previously 30.10 / 28.48 and 20.43 / 19.21), so sharing helps calibrated and RTN alike.
  `--calib` recovers about a quarter of the gap on the 0.6B and nearly two fifths on the 1.7B; it
  does not close it against BF16.
- `degen_suite.py` reads 45/45 on every checkpoint in the table (AWQ ones re-run three and two
  times). Checkpoints from the earlier non-deterministic calibration files each flipped exactly
  one of the 45 probes, a different one each time (stream-vs-non-stream whitespace, think-leak,
  adherence returning empty content); calibration determinism removed that.
- Refuted hypothesis: folding `o_proj`'s scale into `v_proj` (the tensor the KV cache stores;
  default KV dtype here resolves to FP8_E4M3) looked like it should cost more in the cache than
  the scale wins. The FP8-vs-FP16-KV penalty is 0.300 PPL calibrated vs 0.595 round-to-nearest
  (28.478/28.178 vs 30.098/29.503); the scaled `v_proj` is if anything friendlier to FP8 KV.
- Not established: per-group contribution. Norm-folds-only and no-`o_proj` variants measured 29.40
  and 29.25, but each from a different pre-determinism calibration file, and the 1.6 % spread is
  the size of the gaps; attribution needs one fixed calibration file.

> **Measure this on the 45k corpus, not `ppl_corpus.txt`.** The 199-token corpus reads wildly
> different numbers and inverts the model-size trend: too few tokens, not a quantizer property.

#### Head-to-head against a Modelopt export

"Prefer a published Modelopt checkpoint" does not hold on the one locally comparable model
(measured 2026-07-31). `Qwen3-14B-NVFP4` is a genuine Modelopt export (`producer: modelopt`);
its untouched tensors (`model.norm.weight`, every `input_layernorm`, the 1.5 GB embedding table)
hash identical to the `Qwen/Qwen3-14B` BF16 source; both quantizers quantize the same 280 tensors
and exclude `lm_head`. Same corpus, engine, `deterministic_gemm`; each number reproduced to four
decimals:

| NVFP4 checkpoint | PPL (`ppl_corpus_45k.txt`) |
|---|---:|
| Modelopt export | 10.0301 |
| `imp-quantize`, no `--calib` | **9.9252** |

The uncalibrated in-tree quantizer is 1.05 % ahead: one model on one corpus, not a better-quantizer
claim, but it retires "a published export will beat this". Confirmed mechanism half: the export
ships 280 `input_scale` and 40 `k_scale`/`v_scale` tensors (a recipe quantizing activations and KV
too); imp does not apply them (`input_scale` loaded for diagnostics, read by no GEMM kernel,
`weight_upload.cu`, uploaded only under audit), so imp runs W4A16 against weights rounded for
W4A4-with-quantized-KV. Inferred half: that this is why the export loses (the alternative, a
calibration-corpus mismatch, needs a second corpus to separate). Applying `input_scale` would not
fix it: those scales quantize activations down, which imp keeps at higher precision anyway. The
BF16 baseline cannot be measured: 27.5 GiB weights plus the allocator's 5 % headroom does not fit
32 GiB.

#### Calibrating a model that will not fit, and what it exposed

A calibration file is keyed by (layer index, tensor kind), not tensor name or dtype, and the
recording hook sits before the tier switch in `gemm_via_handle_`, so nothing ties the file to the
checkpoint it came from: stats for a model too large to run can be collected from any quantization
of the same model, and the BF16 source quantized with them. Measured 2026-08-01,
`ppl_corpus_45k.txt`, 13 537 tokens, deterministic:

| Model | round-to-nearest | AWQ, stats from the BF16 source | AWQ, stats from a quantized twin |
|---|---:|---:|---:|
| Qwen3-0.6B | 30.0979 | **28.4782** | **28.8868** |
| Qwen3-14B | 9.9252 | *(impossible: will not fit)* | **12.6016** / **12.2853** |

The detour is sound: on the 0.6B, stats from imp's own RTN checkpoint recover three quarters of the
BF16-source gain (1.21 of 1.62 PPL), and that twin is 25 % worse than its source, so twin fidelity
is not the sensitive part. Exposed: **`--calib` hurts at 14B.** The two 14B figures come from two
independent twins (imp's RTN 12.6016, NVIDIA's Modelopt export 12.2853) that agree with each other
and disagree with RTN by 24-27 % in the wrong direction; the quantizers share no code, so the
calibration source is not the variable. (Re-scored same day: RTN 9.9225, calibrated 12.5371.)
Ruled out: incomplete plan (both runs scaled 160 groups, 4 per layer across all 40), degenerate
statistics (280 entries, no zero or non-finite channel), a magnitude effect (the search normalises
by the group mean), the FP8 KV path (`fp8_e4m3` and `fp16` identical to four decimals). What
remains is the scale search's objective, a local proxy: it minimises per-group
weight-reconstruction error and improved on every group of the 14B run; better-reconstructed
weights can still be a worse model, and at 40 layers are.

**Why it flips between 1.7B and 14B (measured 2026-08-05).** Four groups per layer
(`awq_plan.cpp`); `--calib-groups` runs any subset, so the result is attributed, not guessed.
Deltas against each model's own RTN baseline:

| subset | Qwen3-14B (`n_rep=5`) | Qwen3-0.6B (`n_rep=2`) |
|---|---|---|
| **B+D, the two FFN groups** | **−0.1330** *(best)* | |
| B+C+D | −0.0825 | |
| C: o_proj | +0.0159 | **−0.6115** |
| A: q,k,v | +0.6522 | +0.2751 |
| A+B+D (C off) | +0.7641 | −0.6475 |
| A+C | +2.0326 | −0.1276 |
| ABCD | **+2.6764** | **−1.2111** *(best)* |

Interactions, same baselines:

| | Qwen3-14B | Qwen3-0.6B |
|---|---|---|
| A x C | **+1.3645** | +0.2088 |
| A x BD | +0.2449 | |
| **BD x C** | **+0.0346** | |
| C x ABD | **+1.8964** | **+0.0479** |

The split is attention vs FFN; groups stop being independent only on the attention side. FFN is
clean at both sizes: on the 14B, `BD` is the best measured configuration of all at −0.1330,
beating round-to-nearest, and barely interacts with C (+0.03). Everything harmful involves A
(A x C +1.36; C x ABD +1.90, i.e. 71 % of ABCD's damage is interaction, not sum of parts); at
`n_rep=2` the same C x ABD interaction is +0.05, forty times smaller, so effects simply add and
the full set wins. No single group is broken (C alone +0.016 on the 14B, neutral); the attention
pair fails once GQA gets wide. Mechanism: C and D run first (their folds rewrite `v_proj` and
`up_proj`, members of groups A and B), so A searches its scale on a `v_proj` C already divided,
`search_group_scale` summing one objective over q, k, v. The `n_rep` dependence is C's statistic:
tied across query heads sharing a KV head (`awq_plan.cpp:302-313`) via `max`, inflating a
channel's weight in the error term by a median factor of 1.346 at `n_rep=5` vs 1.000 at `n_rep=2`
(20.5 % of channels inflated >=2x vs 8.3 %); `a_j` is the weight in the objective
(`err += (a_j/s_j)^2 * (...)^2`), so a distorted `a_j` makes the search optimise the wrong thing.

**The obvious fix was built and is REFUTED; do not re-try.** The tie serves two roles: it shapes
`s` (genuine constraint: C's fold writes `s` into `v_proj`'s shared rows) and weights the error (a
measurement). Splitting them (tied statistic for the scale, recorded statistic for the weight) is
a 15-line change to `search_group_scale`. Measured 2026-08-05:

| | before | with the split |
|---|---|---|
| 14B `BD` *(control: C not involved)* | 9.7922 | **9.7922**, bit-identical |
| 14B `C` | 9.9411 | **10.0098** *(worse)* |
| 14B `ABCD` | 12.6016 | 12.4794 *(still +2.55 over RTN)* |
| **0.6B `ABCD`** | **28.8868** | **29.5937** *(worse by 0.71)* |

It does not rescue the 14B and damages the working configuration, giving back more than half of
the 0.6B's −1.21 gain. The split is not more correct: with `s` forced constant across a KV group,
weighting the error by channels the search cannot steer separately is inconsistent with the
constraint. The `max` tie is a real coupling, not a bug; a fix must change the constraint (how the
fold works), not the weighting.

**The second variant is measured and REFUTED for the same reason (2026-08-10):** keep the single
role, change the aggregation to `mean` over the `n_rep` query-head channels. Same harness, RTN
re-measured in-pipeline:

| | RTN | `ABCD` with `max` | `ABCD` with `mean` | mean − max |
|---|---|---|---|---|
| 14B (`n_rep=5`) | 9.9766 | 18.0223 | 17.7464 | **−0.276** |
| 0.6B (`n_rep=2`) | 30.3977 | 27.4846 | 27.5326 | +0.048 |

The tie behaves as the mechanism predicts (~6x larger on wide GQA, sign reverses on narrow), and
is worth 0.276 of an 8.05 problem, 3 %: a minor consequence of the coupling, not its cause; only
the constraint can fix this. Flag removed rather than shipped (a knob buying 3 % invites the
re-try this section warns against). Caveat: `ABCD` costs +8.05 over RTN here vs +2.68 above, on a
run whose RTN reproduces (9.9766 vs 9.9252); the setup difference is the calibration corpus (this
run calibrated on `ppl_corpus_45k.txt`, the scoring text; the earlier numbers on general prose).
Same sign, 3x magnitude, unexplained: a free lead for the attention half.

Two standalone findings: **group A hurts both models** (+0.28 / +0.65), independent of `n_rep`,
previously unknown. And **`--calib` is not what fails at 14B; its attention half is**:
`--calib-groups BD` scores 9.7922 against round-to-nearest's 9.9252, so calibration pays at this
size with attention left out. `--calib-groups` is therefore a production switch: **`BD` on
wide-GQA models, default `ABCD` on narrow-GQA ones** (0.6B: ABCD −1.21, clearly best there). The
−0.133 is well outside reproduction noise (RTN re-scores 9.9225-9.9252, 0.03 % spread, vs a
1.34 % gain). It also explains why the single-cause eliminations found nothing: an effect that is
71 % interaction between two individually harmless steps is invisible to all of them.

Verdict: `imp-quantize --calib` is validated on Qwen3-0.6B and Qwen3-1.7B, measured harmful on
Qwen3-14B; the tool says so; score the calibrated checkpoint against the uncalibrated one before
use. Larger models: `--calib-groups BD` or round-to-nearest (`n_rep` is 8 on most 70B-class
checkpoints, further along the axis that breaks attention; FFN showed no such dependence; `BD` at
70B untested, score before trusting; RTN is a solid floor, it beat the Modelopt export on the 14B,
9.9252 vs 10.0301). No VRAM ceiling on quantizing: the quantizer never resides the model
(`search_group_scale` uploads one group, `main.cpp` quantizes one tensor at a time; ~0.7 GiB for a
14B, 1.8 GiB for a 70B). Only calibration and scoring run the model (the twin recipe above),
bounding the calibrated route at roughly 40-50B on a 32 GiB card.

#### MoE, and two roles that must stay full precision

"MoE is not supported" was too broad, and wrong in the dangerous direction: HF-standard per-expert
2-D tensors were never skipped, they were quantized and silently produced a broken checkpoint.
DeepSeek-V2-Lite (MLA + 64 routed experts, 2026-07-31): quantizing everything gave cross-script
repetition garbage; the BF16 source answered normally. Bisection:

| Quantized | Result |
|---|---|
| everything | garbage |
| everything except MLA `kv_a_proj`/`kv_b_proj` | garbage (router still in) |
| everything except the router | garbage (MLA pair still in) |
| everything except **both** | **coherent** |
| MLP + all 4992 expert tensors, attention left BF16 | coherent |

Expert quantization works; the culprits are the MLA latent projections (the runtime slices
`kv_a_proj_with_mqa` into latent+RoPE and reshapes `kv_b_proj` into per-head nope/v halves) and
the MoE router (FP4 across 16 shared-scale values changes the top-k pick). Both refused, costing a
handful of small matrices per layer. With them excluded: 29.26 GiB -> 8.91 GiB (3.28x) in ~70 s;
`degen_suite.py` 3 FAIL / 32 vs the BF16 source's 5 FAIL / 32, a strict subset, so quantization
introduces none. Still unsupported: expert weights as one 3-D `[n_experts, N, K]` stack
(gpt-oss-style); reported and left unquantized. Open: a head-to-head against a Modelopt export of
the same model (needs one staged locally in both precisions).

Workflow with Modelopt:

```bash
pip install nvidia-modelopt

python -m modelopt.llm.ptq \
  --model Qwen/Qwen3-8B \
  --quant nvfp4 \
  --output ./Qwen3-8B-nvfp4/

imp-cli --model ./Qwen3-8B-nvfp4/ --prompt "Hello"
```

Modelopt quantization modes:

| Mode | What's quantized |
|---|---|
| `nvfp4` | all linear layers |
| `nvfp4_mlp_only` | MLP / FFN layers only |
| `nvfp4_experts_only` | MoE expert layers only |
| `nvfp4_omlp_only` | MLP + output projection |

### NVFP4 internal pipeline

Dense layers:

```
SafeTensors NVFP4 packed weights + scales
  → loader (BF16 norms / router → FP16, packed FP4 stays packed)
  → Phase 0: register in NVFP4 decode cache (no re-quant)
  → Phase 3b: CUTLASS scale-factor layout (SfAtom) for prefill
  → prefill: CUTLASS NVFP4 GEMM via gemm_dispatch() (sm_120 tensor cores)
  → decode:  NVFP4 GEMV (prmt register LUT, K-parallel)
```

MoE layers (Modelopt SafeTensors, per-expert):

```
SafeTensors per-expert weights
  → cache_moe_native_nvfp4 builds one contiguous [ne, N, K_packed]
    buffer per layer per projection (D2D-memcpy from per-expert tensors)
  → per-expert tensors freed inline (32 GiB VRAM ceiling on 35B-A3B)
  → CUDA Graphs capture cleanly via the decode fast-path
```

Without `cache_moe_native_nvfp4` the legacy FP16 dequant + cuBLAS sm_80 WMMA fallback fires per
layer per token, killing CUDA Graphs and dropping decode 5-17x. NVFP4 KV cache (`--kv-nvfp4`)
supports chunked prefill since PR #149: past chunks' K/V are gathered from the paged cache via
`paged_kv_gather_nvfp4_to_fp16` (PTX `cvt.rn.f16x2.e2m1x2` inner loop + UE4M3 scale fold) and
concatenated with the current chunk before rectangular cuBLAS attention. Hybrid GDN+MoE /
Mamba2+MoE archs (Qwen3.5/3.6, Nemotron-H) in scope since PR #156.

## MXFP4 (GGUF)

- Same FP4 E2M1 nibble layout as NVFP4, UE8M0 micro-scales (per 32 elements), no separate tensor
  scale; the format Blackwell tensor cores expect natively, so MXFP4 prefill goes through CUTLASS
  at full FP4 throughput.
- Shipped inside GGUF under a proprietary tensor-type code (31); llama.cpp reads that as the
  removed `Q4_0_4_4`, so cross-tool perplexity comparison needs a standard MXFP4 export.
- Round-to-nearest MXFP4 is +5-15% perplexity vs Q8_0, worse than Q4_K_M (+2.2% on Qwen3-4B
  wikitext-2). MR-GPTQ calibration would close the gap; on the [roadmap](roadmap.md).

## KV cache element type

Set via `--kv-fp8` / `--kv-int8` / `--kv-int4` / `--kv-nvfp4` / `--kv-mxfp4`, or in `imp.conf`:

```toml
[kv_cache]
dtype = "auto"  # auto (default) | fp16 | fp8 | int8 | int4 | nvfp4 | mxfp4
```

- `auto` keeps FP16 but upgrades to FP8 E4M3 for models declaring `kv_cache_quant_algo=FP8`
  (Modelopt NVFP4 checkpoints) whose arch family passed the long-context FP8-KV quality gate
  (`kv_fp8_hint_default_safe` in `src/model/model.cpp`). Allowlisted: Qwen3 dense + Qwen3 MoE;
  measured on a 3.9k-token context, FP8 vs FP16 KV: Qwen3-14B PPL 13.95 -> 14.10 (+1.07%),
  Qwen3-30B-A3B ~16.20 -> ~15.99 (neutral), both coherent, ~768 MiB KV VRAM saved. Other
  hint-declaring families (Phi-4, Nemotron-H, Qwen3.5/3.6, Gemma-4) stay FP16 until measured;
  `--kv-fp8` forces, `dtype = "fp16"` opts out.
- The default flipped to FP16 in PR #51 (FP8 silently broke Llama, Mistral, DeepSeek at first
  decode). Beyond the `auto` allowlist FP8 is opt-in; verified coherent on Qwen3 dense,
  Qwen3.5 / 3.6 GDN, Llama-3.2, Gemma-4 (FP8 KV warmup-calibration bug fixed in PR #89; Gemma-4
  dual-head_dim carve-out removed in PR #91).
- INT4 KV is for VRAM pressure only: coherent, ~22% decode regression at 20K context.
- NVFP4-KV (`--kv-nvfp4`) and MXFP4-KV (`--kv-mxfp4`, PR #249) store FP4 at 25% of FP16 (E4M3 vs
  UE8M0 micro-scales), both with chunked prefill via `paged_kv_gather_*_to_fp16`. (TurboQuant
  retired in PR #251; its deprecated `--kv-turboquant{,-lite}` alias flags removed 2026-07-07.)

## Choosing a quant

Quick guidance, not a benchmark:

- **Q8_0**: cleanest baseline; use when quality matters and VRAM allows.
- **Q4_K_M**: most VRAM-efficient GGUF; sufficient for most chat; can degenerate on long code-gen
  on Gemma-4 (use Q5_K_M or Q8_0 there).
- **Q6_K**: in between; good MoE pick on Qwen3-Coder-30B.
- **IQ4_NL / IQ4_XS**: load and run since #556 via the dequant path (FP16-cache decode like Q4_K,
  dequant->cuBLAS prefill); for community-quant compatibility. At equal VRAM prefer Q4_K_M
  (dedicated dp4a/MMVQ kernels). IQ1/IQ2/IQ3 families unsupported.
- **NVFP4** (SafeTensors prequant): highest decode throughput on prequant-aware models (per-model
  numbers: [`BENCHMARKS.md`](BENCHMARKS.md)). Requires AWQ/SmoothQuant calibration; only Modelopt
  fully tested.
- **MXFP4**: GGUF-native FP4, smallest footprint (Qwen3-4B at 2.8 GB); quality lags Q4_K_M without
  MR-GPTQ calibration.

#### Refuted: micro-scale search (2026-07-26)

Choosing each micro-scale by minimizing block reconstruction error (searching FP8 candidates
around `absmax`) moved Qwen3-0.6B from PPL 30.10 to 29.88: 0.7%, for ~6x the quantization cost.
Reverted; do not re-attempt. The micro-block is 16 values, where `absmax` is already near-optimal
(clipping pays when one outlier spoils a 64-128 group), and the dominant error is the FP4 grid
itself, eight magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6), which no scale choice improves. That is why
the follow-up work was AWQ/GPTQ class (calibration moves the error rather than shrinking it): see
`--calib` above.
