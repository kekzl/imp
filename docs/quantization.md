# Quantization

imp supports both GGUF quantization (loaded directly from llama.cpp-compatible files) and SafeTensors NVFP4 prequant (produced by external calibration tools). This page explains what each format is, where it is used inside the engine, and what the trade-offs are.

For per-model picks see [`supported-models.md`](supported-models.md). For benchmark numbers see [`performance.md`](performance.md).

## Formats and where they show up

| Format | Bits / weight | Source | Used for |
|---|---:|---|---|
| Q8_0 | 8.0 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q6_K | 6.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q5_K_M | 5.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q4_K_M | 4.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q4_0 | 4.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| IQ4_NL / IQ4_XS | 4.5 / 4.25 | GGUF | dequant→FP16 cache decode + dequant→cuBLAS prefill (no dp4a/MMVQ kernels) |
| FP8 E4M3 | 8.0 | runtime | KV cache (opt-in), prefill weight cache |
| INT8 | 8.0 | runtime | KV cache (opt-in) |
| INT4 | 4.0 | runtime | KV cache (long-ctx, opt-in) |
| NVFP4 | 4.0 | SafeTensors | weights (decode + prefill), KV cache |
| MXFP4 | 4.5 | GGUF | weights (decode + prefill attention) |

GGUF formats are mmap'd from disk and uploaded as-is to GPU; the `*.K` quants store block scales in the format the dp4a kernels expect. NVFP4 prequant arrives in two-byte-per-element packed form with FP8 E4M3 micro-scales (per-16) and an FP32 tensor scale; imp registers these directly into the NVFP4 decode cache and CUTLASS NVFP4 GEMM path with no re-quantization.

## NVFP4 prequant (SafeTensors)

Calibrated per-tensor scales using AWQ or SmoothQuant. Two upstream tools produce compatible files:

| Tool | Status |
|---|---|
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (Modelopt) | Primary path. Coherent on Qwen3-Coder-30B, Mistral-3.2, Qwen3.6, Gemma-4 (after PR #88 lit up the CUTLASS NVFP4×NVFP4 prefill cache). |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | Loads, but several models degenerate past ~30 tokens. See [roadmap](roadmap.md). Prefer Modelopt where available. |
| `imp-quantize` (in-tree) | **Experimental.** AWQ-calibrated with `--calib`, plain round-to-nearest without. Below a published export either way. See below. |

### imp-quantize — converting a checkpoint yourself (EXPERIMENTAL)

> **Experimental.** The pipeline is verified end to end and `--calib` recovers a
> measurable part of the quantization loss, but the result still sits below a
> published Modelopt export. Use it to get a model onto the NVFP4 path for
> evaluation or performance work — not to produce a checkpoint you ship.

`imp-quantize` turns a dense BF16/FP16 SafeTensors checkpoint into an NVFP4 one
without leaving the repo, for models nobody has published an export for:

```bash
# 1. one calibration pass over a corpus — writes per-channel activation stats
imp-cli --model ./Qwen3-1.7B --perplexity ./calib_corpus.txt --calibrate ./calib.bin

# 2. quantize using them
imp-quantize --model ./Qwen3-1.7B --out ./Qwen3-1.7B-nvfp4 --calib ./calib.bin
imp-cli --model ./Qwen3-1.7B-nvfp4 --prompt "Hello"
```

Drop `--calib` (and step 1) for plain round-to-nearest; `--dry-run` previews
what would be quantized without touching the GPU.

It writes the layout the loader already recognises — `<prefix>.weight` (U8,
packed nibbles), `.weight_scale` (F8_E4M3 micro-scales), `.weight_scale_2`
(F32 tensor scale), plus `hf_quant_config.json` — copies the tokenizer and
config files, and rebuilds the shard index when the source is sharded.
Embeddings, norms and (unless `--lm-head`) the LM head stay full precision.

#### Roles that stay full precision, and why

Three weight roles are 2-D and K-aligned — every shape check waves them through
— and must not be quantized anyway. Each was found by measurement, not by
reasoning about shapes:

| role | why | found by |
|---|---|---|
| MLA latent projections (`kv_a_proj`, `kv_b_proj`) | the runtime slices and reshapes both | bisection on DeepSeek-V2-Lite: quantizing them gave a checkpoint that loaded and emitted cross-script garbage |
| MoE router (`.gate.weight`) | FP4 across 16 shared-scale values changes the top-k pick | measured separately, with the MLA pair already excluded |
| **fused Q+gate `q_proj`** (Qwen3.5 / Qwen3-Next `attn_output_gate`) | the gate half feeds a **sigmoid**, and E2M1 is coarsest near zero — exactly where a sigmoid is most sensitive | #1273: rounding *only* that half on a healthy GGUF twin reproduces the real defect (+0.0169 injected divergence per attention block vs +0.0156 for the actual NVFP4 checkpoint); the Q half sits below the noise floor |

The last one is **not** a 4-bit problem: the same half in Q4_K is healthy at
6.55 perplexity. It is specific to NVFP4/E2M1, which offers 8 magnitudes per
micro-block.

imp cannot exclude half a tensor, so the whole `q_proj` is kept — 230 MiB on
Qwen3.6-35B-A3B (10 gated layers of 40) and 552 MiB on the 27B (16 of 64), about
1% and 4% of the checkpoint, against a 2.08x–6x perplexity penalty.
`--quantize-attn-gate` opts back in and warns.

A gated `q_proj` is detected from shapes rather than a config flag: it emits
twice what its own layer's `o_proj` consumes. Note that **published exports have
the same gap** — both llm-compressor and Modelopt exclude `linear_attn.*` but
quantize this tensor whole, which is why every hybrid NVFP4 checkpoint tested
degrades.

#### What `--calib` does

NVFP4's error scales with the magnitude of what it quantizes, so multiplying an
input channel's weights *up* before quantizing buys that channel precision at
the others' expense — provided something divides the activation back *down*.
Which channels deserve it is not visible in the weights; it takes a forward
pass. That is the whole reason a calibration step exists.

The transform is exact before quantization, `y = x Wᵀ = (x/s)(W·diag(s))ᵀ`, so
the only open question is which `s` minimises the error *after* quantizing
`W·diag(s)`. imp answers it by measurement rather than a closed form: for each
candidate exponent it quantizes with the real kernel and keeps the winner.
`alpha = 0` (plain round-to-nearest) is always in the grid, so a group where
scaling does not pay keeps its untransformed weights.

The compensating `1/s` is folded into the producer, which keeps the output a
plain NVFP4 checkpoint needing no runtime support — four groups per layer:
q/k/v and gate/up fold into their preceding RMSNorm weight, `o_proj` into
`v_proj`'s output rows (GQA-tied), `down_proj` into `up_proj`'s. Because the
norm fold assumes a plain multiplicative RMSNorm, `--calib` **refuses**
architectures whose norm applies `(1 + g)` (Gemma-class) rather than silently
producing a different model.

**A norm can only be folded when every consumer of it is scaled.** Dividing a
norm's weight by `s` divides its output by `s` for *every* reader, and each
reader only stays correct if its own columns were multiplied by the same `s`.
Two roles are deliberately excluded from quantization (MLA latent projections,
the MoE router) and so never receive that compensation — and the router reads
exactly the norm the gate/up group folds into. `--calib` therefore checks the
consumers of each norm and **refuses the fold** when an unscaled one exists,
naming it. On DeepSeek-V2-Lite that leaves 2 of 108 groups scaled (layer 0's
dense MLP, the only layer with no router) and prints a line per refusal.

**`--calib` does not calibrate MoE experts yet.** The planner groups the dense
FFN by name (`mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj`); a MoE
checkpoint's weight lives in `mlp.experts.<e>.*`, which it does not model. On a
MoE model the attention groups still calibrate and the experts — the bulk of the
model — stay at round-to-nearest. That is now stated per layer in the output
rather than folded into a `skipped` count.

**Calibrate on a different corpus than you score on.** `tools/analysis/`
`fetch_calib_corpus.sh` assembles general public-domain prose for exactly this
reason; scoring happens on `ppl_corpus_45k.txt` (imp's own architecture doc).
Calibrating and scoring on one text reports a gain that exists only on it.

**`--calibrate` forces `runtime.deterministic_gemm`,** and that is not a
formality. Without it, two runs of the identical command differed on **94% of
the recorded floats** (up to 0.5% each) — imp's forward has run-to-run variance
on this config — and that carried straight through: three checkpoints built
from three such calibration files scored PPL 28.84, 28.94 and 28.48, a 1.6%
spread from nothing but which run happened to produce the file. With
determinism forced the calibration file is bit-identical run to run, and so is
the checkpoint.

#### Quality, measured

`imp-cli --perplexity` over `tools/analysis/ppl_corpus_45k.txt` (13 537 tokens),
calibration over 36 058 tokens of general prose. Full chain reproducible with
`tools/analysis/awq_ppl_ab.sh`:

| Model | BF16 | NVFP4 RTN | NVFP4 `--calib` | AWQ gain | gap to BF16 |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | 24.06 | 30.10 | **28.48** | −5.4% | +25.1% → **+18.3%** |
| Qwen3-1.7B (2 shards) | 17.22 | 20.43 | **19.21** | −6.0% | +18.6% → **+11.5%** |

`degen_suite.py` reads 45/45 on every checkpoint in that table (the AWQ ones
re-run three and two times respectively). `--calib` closes about a quarter of
the quantization gap on the 0.6B and nearly two fifths on the 1.7B — it does
not close it against BF16 — but see the head-to-head below before assuming a
published export is automatically better.

**The battery is worth a note, because it did not always read 45/45.**
Checkpoints built from the earlier, non-deterministic calibration files each
flipped exactly one of the 45 probes — and a *different* one each time (a
stream-vs-non-stream whitespace check, a think-leak check, an adherence probe
returning empty content). Forcing calibration determinism removed that too.
Treat it as one more reason the calibration pass has to be reproducible, not as
a coherence property that happens to hold.

**One hypothesis measured and refuted.** Folding `o_proj`'s scale into `v_proj`
writes into the tensor the KV cache stores, and imp's default KV dtype here
resolves to FP8_E4M3 — so widening v's per-channel range looked like it should
cost more in the cache than the scale wins in the weight. It does not. The
FP8-vs-FP16-KV penalty is **0.300 PPL on the calibrated checkpoint and 0.595
on the round-to-nearest one** (28.478/28.178 vs 30.098/29.503). The scaled
`v_proj` is, if anything, friendlier to FP8 KV than the unscaled one.

**Not established: the per-group contribution.** Norm-folds-only and
no-`o_proj` variants were measured at 29.40 and 29.25, which would suggest an
ordering — but each was built from a different (pre-determinism) calibration
file, and the 1.6% spread above is the same size as the gaps. Attributing the
gain to individual groups needs a re-run against one fixed calibration file;
what the table above measures is the shipped configuration against
round-to-nearest, where the gap is far larger than that spread.

> **Measure this on the 45k corpus, not `ppl_corpus.txt`.** The 199-token corpus
> reads wildly different numbers and inverts the model-size trend — an artifact
> of too few tokens, not a property of the quantizer.

#### Head-to-head against a Modelopt export

The standing advice here was "prefer a published Modelopt checkpoint when one
exists". Measured 2026-07-31, it does not hold on the one model that can be
compared locally.

`Qwen3-14B-NVFP4` is a genuine Modelopt export (`producer: modelopt`), and its
untouched tensors are **bit-identical** to the `Qwen/Qwen3-14B` BF16 source —
`model.norm.weight`, every `input_layernorm`, and the 1.5 GB embedding table all
hash the same, so both quantizers started from exactly the same weights. Both
quantize the same **280 tensors** and both exclude `lm_head`. Same corpus, same
engine, same `deterministic_gemm`; each number reproduced to four decimals:

| NVFP4 checkpoint | PPL (`ppl_corpus_45k.txt`) |
|---|---:|
| Modelopt export | 10.0301 |
| `imp-quantize`, no `--calib` | **9.9252** |

The *uncalibrated* in-tree quantizer comes out 1.05% ahead. That is not a claim
that imp-quantize is the better quantizer — it is one model on one corpus — but
it does retire the blanket "a published export will beat this".

**The mechanism, one half confirmed and one half inferred.** The Modelopt export
ships 280 `input_scale` and 40 `k_scale`/`v_scale` tensors alongside the weights
— it was produced for a recipe that quantizes activations and the KV cache too.
imp **does not apply them**: `input_scale` is loaded for diagnostics and read by
no GEMM kernel (`weight_upload.cu`, and it is only uploaded at all under audit).
So imp runs W4A16 against weights rounded for W4A4-with-quantized-KV. That is
confirmed. What stays inferred is that this is *why* the export loses here —
the alternative explanation, that Modelopt's calibration corpus (general text)
simply sits further from this one (technical English) than round-to-nearest's
absence of any calibration does, would need a second corpus to separate.

Note the direction: applying `input_scale` would not fix it. Those scales exist
to quantize activations *down*; imp already keeps them at higher precision. The
export is simply not rounded for the runtime it is being run on.

**The BF16 baseline for this model still cannot be measured** — 27.5 GiB of
weights plus the allocator's 5% headroom does not fit in 32 GiB (the upload dies
partway through the layer stack), and there is nowhere to spill to. But
`--calib` on it turned out not to need that, and what it showed is the section
below.

#### Calibrating a model that will not fit, and what it exposed

A calibration file is keyed by **(layer index, tensor kind)** — not by tensor
name, not by dtype — and the recording hook sits *before* the tier switch in
`gemm_via_handle_`, so the statistic is the activation a weight consumes rather
than whatever a particular tier's kernel materialises. Nothing in that ties a
calibration file to the checkpoint it was collected from. So the statistics for
a model too large to run can be collected from **any quantization of the same
model**, and the BF16 source quantized with them.

Measured 2026-08-01, `ppl_corpus_45k.txt`, 13 537 tokens, deterministic:

| Model | round-to-nearest | AWQ, stats from the BF16 source | AWQ, stats from a quantized twin |
|---|---:|---:|---:|
| Qwen3-0.6B | 30.0979 | **28.4782** | **28.8868** |
| Qwen3-14B | 9.9252 | *(impossible — will not fit)* | **12.6016** / **12.2853** |

**The detour itself is sound.** On Qwen3-0.6B, where both routes are possible,
stats collected from imp's own round-to-nearest checkpoint recover three
quarters of the gain the BF16-source stats give (1.21 of 1.62 PPL). Note what
that twin is: a checkpoint 25% worse than the BF16 source it was made from. Its
statistics still work, so twin *fidelity* is not the sensitive part.

**What it exposed is that `--calib` hurts at 14B.** The two 14B figures come
from two independently produced twins — imp's own round-to-nearest checkpoint
(12.6016) and NVIDIA's Modelopt export (12.2853) — which agree with each other
and disagree with round-to-nearest by 24-27% in the wrong direction. Since the
two quantizers share no code, the calibration *source* is not the variable:
AWQ calibration makes this model worse. (Re-scored later the same day, the
round-to-nearest checkpoint read 9.9225 and the calibrated one 12.5371 — the
residual spread is far below the gap.) Ruled out along the way: an incomplete
plan (both runs scaled 160 groups, which is 4 per layer across all 40, so none
were skipped), degenerate
statistics (280 entries over all 40 layers, no zero or non-finite channel), a
magnitude effect (the search normalises by the group mean, so it is
scale-invariant, and the floor is relative), and the FP8 KV path (`fp8_e4m3`
and `fp16` score identically to four decimals).

That leaves the scale search's objective, which is a **local proxy**: it
minimises per-group weight-reconstruction error, and it improved on every group
of the 14B run. A checkpoint whose weights are each reconstructed better can
still be a worse model, and at 40 layers apparently is.

**Why it flips between 1.7B and 14B — measured 2026-08-05.** The planner has four
groups per layer (`awq_plan.cpp`), and `--calib-groups` runs any subset of them, so
the result can be attributed instead of guessed. Scored the same way as everything
above, against each model's own round-to-nearest baseline:

| subset | Qwen3-14B (`n_rep=5`) | Qwen3-0.6B (`n_rep=2`) |
|---|---|---|
| **B+D — the two FFN groups** | **−0.1330** *(best)* | — |
| B+C+D | −0.0825 | — |
| C — o_proj | +0.0159 | **−0.6115** |
| A — q,k,v | +0.6522 | +0.2751 |
| A+B+D (C off) | +0.7641 | −0.6475 |
| A+C | +2.0326 | −0.1276 |
| ABCD | **+2.6764** | **−1.2111** *(best)* |

Interactions, same baselines:

| | Qwen3-14B | Qwen3-0.6B |
|---|---|---|
| A × C | **+1.3645** | +0.2088 |
| A × BD | +0.2449 | — |
| **BD × C** | **+0.0346** | — |
| C × ABD | **+1.8964** | **+0.0479** |

**The split is attention versus FFN, and the groups stop being independent only on
the attention side.** The two FFN groups are clean at both sizes: on the 14B `BD`
is the *best measured configuration of all* at **−0.1330, beating round-to-nearest**,
and it barely interacts with C (+0.03). Everything harmful involves **A**, whose
interaction with C is +1.36 — and C × ABD reaches +1.90, i.e. **71 % of ABCD's total
damage is interaction, not the sum of parts**. At `n_rep=2` the same C × ABD
interaction is +0.05, forty times smaller, so there the effects simply add and the
full set wins.

So no single group is "broken": C alone is +0.016 on the 14B, essentially neutral,
and blaming it — the obvious reading of the GQA tie — would have been wrong. What
fails is the *attention* pair once GQA gets wide.

The mechanism the numbers point at is in the ordering. C and D run first because
their folds rewrite `v_proj` and `up_proj` — and those two tensors are *members* of
groups A and B. A therefore searches its scale on a `v_proj` that C has already
divided, with `search_group_scale` summing one objective over q, k and v together.
What makes it `n_rep`-dependent is C's own statistic: it must be tied across the
query heads sharing a KV head (`awq_plan.cpp:302-313`), and that tie is a `max`, so
it inflates a channel's weight in the error term by a median factor of 1.346 at
`n_rep=5` against 1.000 at `n_rep=2` — 20.5 % of channels inflated ≥2x versus 8.3 %.
Since `a_j` is the *weight* in the objective (`err += (a_j/s_j)^2 * (...)^2`), a
distorted `a_j` makes the search optimise the wrong thing, faithfully.

**The obvious fix for that follows directly, was built, and is REFUTED — do not
re-try it.** The tie serves two roles at once: it shapes `s` (a genuine constraint,
since C's fold writes `s` into `v_proj`'s rows and those rows are shared per KV
head) *and* it weights the error (a measurement, which nothing constrains). Splitting
them — tied statistic for the scale, recorded statistic for the weight — is a
15-line change to `search_group_scale`. Measured 2026-08-05:

| | before | with the split |
|---|---|---|
| 14B `BD` *(control: C not involved)* | 9.7922 | **9.7922** — bit-identical |
| 14B `C` | 9.9411 | **10.0098** *(worse)* |
| 14B `ABCD` | 12.6016 | 12.4794 *(still +2.55 over RTN)* |
| **0.6B `ABCD`** | **28.8868** | **29.5937** *(worse by 0.71)* |

It does not rescue the 14B and it **damages the configuration that worked**, giving
back more than half of the 0.6B's −1.21 gain. The reason is that the split is not
actually more correct: when `s` is forced constant across a KV group, the search can
only pick *one* value for that whole group, so weighting the error by individual
channels it cannot steer separately makes the objective inconsistent with its own
constraint. The `max` tie is a conservative aggregation that matches what the search
can control — a real coupling, not a bug. Whatever fixes the attention half will have
to change the *constraint* (how the fold works), not the weighting.

Two findings worth keeping separately. **Group A hurts both models** (+0.28 / +0.65),
which has nothing to do with `n_rep` and was not previously known. And **`--calib`
is not the thing that fails at 14B — the attention half of it is.** `--calib-groups
BD` scores 9.7922 against round-to-nearest's 9.9252, so calibration *does* pay at
this size once the attention groups are left out. That makes `--calib-groups` a
production switch and not only a diagnostic: **use `BD` on wide-GQA models, and the
default `ABCD` on narrow-GQA ones** (0.6B: ABCD −1.21, clearly the best there).
The −0.133 is well outside reproduction noise — the round-to-nearest checkpoint
re-scores to 9.9225-9.9252, a spread of 0.03 %, against a 1.34 % gain.

This also explains why the earlier eliminations found nothing: an incomplete plan,
degenerate statistics, a magnitude effect, the FP8 KV path and the calibration
source are all tests for a **single** cause. An effect that is 71 % interaction
between two individually harmless steps is invisible to every one of them.

So: `imp-quantize --calib` is validated on Qwen3-0.6B and Qwen3-1.7B and
measured harmful on Qwen3-14B. The tool now says so, and the rule is to score
the calibrated checkpoint against the uncalibrated one before using it.

**For anything larger, calibrate the FFN only (`--calib-groups BD`), or use
round-to-nearest.** `n_rep` is 8 on most 70B-class checkpoints, i.e. further along
the axis that breaks the attention groups, while the FFN groups showed no such
dependence. Both routes are safe against the failure above; `BD` is the one that
also gains something, and `ABCD` at that size is the one to avoid. Round-to-nearest
remains a solid floor — on the 14B it beat a genuine Modelopt export (9.9252 vs
10.0301). Whether `BD` still pays at 70B is untested; score it before trusting it.

Neither route has a VRAM ceiling, because **the quantizer never resides the model.**
`search_group_scale` uploads one group and `main.cpp` quantizes one tensor at a
time, so demand scales with the largest single weight matrix — roughly 0.7 GiB for a
14B and 1.8 GiB for a 70B — not with the checkpoint. Only *calibration* and
*scoring* have to run the model, which is what the twin recipe above is for, and
what bounds the calibrated route at roughly 40-50B on a 32 GiB card.

#### MoE, and two roles that must stay full precision

"MoE is not supported" was too broad, and it was wrong in the dangerous
direction. Checkpoints that store experts the HF-standard way — one 2-D tensor
per expert — were never skipped; they were quantized and **silently produced a
broken checkpoint**. Measured on DeepSeek-V2-Lite (MLA + 64 routed experts,
2026-07-31): quantizing everything gave a model that loaded and then emitted
cross-script repetition garbage, while the BF16 source answered normally.

Bisection named the two culprits, and neither is the experts:

| Quantized | Result |
|---|---|
| everything | garbage |
| everything except MLA `kv_a_proj`/`kv_b_proj` | garbage (router still in) |
| everything except the router | garbage (MLA pair still in) |
| everything except **both** | **coherent** |
| MLP + all 4992 expert tensors, attention left BF16 | coherent |

So **expert quantization works**; what breaks is the MLA latent projections
(the runtime slices `kv_a_proj_with_mqa` into latent+RoPE and reshapes
`kv_b_proj` into per-head nope/v halves) and the MoE router (FP4 across 16
shared-scale values changes the top-k expert pick). Both are now refused, at a
cost of a handful of small matrices per layer.

With them excluded, DeepSeek-V2-Lite quantizes 29.26 GiB → 8.91 GiB (3.28×) in
~70 s and `degen_suite.py` reads **3 FAIL / 32** against the BF16 source's
**5 FAIL / 32** — the quantized model's failures are a strict *subset*, so the
quantization introduces none of them. (This model is weak at instruction
following either way; the residual failures are the model's, not the
quantizer's.)

Still unsupported: expert weights stored as one 3-D `[n_experts, N, K]` **stack**
(gpt-oss-style). Those are reported and left unquantized.

Remaining open work: a head-to-head against a Modelopt export of the same model,
which needs one staged locally in both precisions.

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

Without `cache_moe_native_nvfp4` the legacy FP16 dequant + cuBLAS sm_80 WMMA fallback fires per layer per token, killing CUDA Graphs and dropping decode 5–17×.

NVFP4 KV cache (`--kv-nvfp4`) supports chunked prefill since PR #149 — past chunks' K/V are gathered from the paged cache via `paged_kv_gather_nvfp4_to_fp16` (PTX `cvt.rn.f16x2.e2m1x2` inner loop + UE4M3 scale fold) and concatenated with the current chunk before rectangular cuBLAS attention. Hybrid GDN+MoE / Mamba2+MoE archs (Qwen3.5/3.6, Nemotron-H) are in scope since PR #156.

## MXFP4 (GGUF)

MXFP4 uses the same FP4 E2M1 nibble layout as NVFP4 but with UE8M0 micro-scales (per 32 elements) and no separate tensor scale. This matches the format the Blackwell tensor cores expect natively, so MXFP4 prefill goes through CUTLASS at full FP4 throughput.

imp ships MXFP4 inside GGUF using a proprietary tensor-type code (31). llama.cpp reads this as the removed `Q4_0_4_4` format, so cross-tool perplexity comparison is not possible without a standard MXFP4 export.

Round-to-nearest MXFP4 is +5–15% perplexity vs Q8_0, worse than Q4_K_M (+2.2% on Qwen3-4B wikitext-2). MR-GPTQ calibration would close this gap; it is on the [roadmap](roadmap.md).

## KV cache element type

Set via `--kv-fp8` / `--kv-int8` / `--kv-int4` / `--kv-nvfp4` / `--kv-mxfp4`, or in `imp.conf`:

```toml
[kv_cache]
dtype = "auto"  # auto (default) | fp16 | fp8 | int8 | int4 | nvfp4 | mxfp4
```

`dtype = "auto"` (the default) keeps FP16 but upgrades to FP8 E4M3 for models whose author declares `kv_cache_quant_algo=FP8` (Modelopt NVFP4 checkpoints) **and** whose arch family has passed the long-context FP8-KV quality gate (`kv_fp8_hint_default_safe` in `src/model/model.cpp`). Currently allowlisted: **Qwen3 dense + Qwen3 MoE** — measured on a 3.9k-token context, FP8 vs FP16 KV: Qwen3-14B PPL 13.95→14.10 (+1.07%), Qwen3-30B-A3B ~16.20→~15.99 (neutral), both coherent, ~768 MiB KV VRAM saved. Other hint-declaring families (Phi-4, Nemotron-H, Qwen3.5/3.6, Gemma-4) stay FP16 until measured; pass `--kv-fp8` to force, or `dtype = "fp16"` to opt out.

The default flipped to FP16 in PR #51 — FP8 had been silently breaking Llama, Mistral, and DeepSeek at first decode. Beyond the `auto` allowlist, FP8 stays opt-in; it is verified coherent on Qwen3 dense, Qwen3.5 / 3.6 GDN, Llama-3.2, and Gemma-4 (FP8 KV warmup-calibration bug fixed in PR #89; Gemma-4 dual-head_dim carve-out removed in PR #91).

INT4 KV is for VRAM-pressure cases only — coherent but ~22% decode regression at 20K context. NVFP4-KV (`--kv-nvfp4`) and MXFP4-KV (`--kv-mxfp4`, PR #249) both store FP4 at 25% of FP16 — NVFP4 uses E4M3 micro-scales, MXFP4 uses UE8M0; both ship chunked prefill via `paged_kv_gather_*_to_fp16`. (TurboQuant was retired in PR #251; its deprecated `--kv-turboquant{,-lite}` alias flags were removed 2026-07-07.)

## Choosing a quant

Quick guidance, not a benchmark:

- **Q8_0** is the cleanest baseline. Use it when output quality matters and VRAM allows.
- **Q4_K_M** is the most VRAM-efficient GGUF. Sufficient for most chat; can degenerate on long code-gen on Gemma-4 — use Q5_K_M or Q8_0 there.
- **Q6_K** sits in between. Good MoE pick on Qwen3-Coder-30B.
- **IQ4_NL / IQ4_XS** (i-quants) load and run since #556 via the dequant path (FP16-cache decode like Q4_K, dequant→cuBLAS prefill). Supported for community-quant compatibility — at equal VRAM prefer Q4_K_M, which has dedicated dp4a/MMVQ kernels. The IQ1/IQ2/IQ3 families remain unsupported.
- **NVFP4** (SafeTensors prequant) gives the highest decode throughput on prequant-aware models (current per-model numbers in [`BENCHMARKS.md`](BENCHMARKS.md)). Requires AWQ/SmoothQuant calibration; only Modelopt is fully tested.
- **MXFP4** is GGUF-native FP4. Smallest footprint (Qwen3-4B at 2.8 GB), but quality lags Q4_K_M without MR-GPTQ calibration.

#### Refuted: micro-scale search (2026-07-26)

Before reaching for calibration, the cheap hypothesis was tested and **does not
pay**: choosing each micro-scale by minimizing the block's reconstruction error
(searching FP8 candidates around `absmax` instead of taking it) moved
Qwen3-0.6B from PPL 30.10 to **29.88** — 0.7%, for ~6x the quantization cost.
Reverted; do not re-attempt.

Why it cannot help much: the micro-block is only **16** values. Clipping the
scale pays when one outlier spoils a large group (64-128), but across 16 values
`absmax` is already near-optimal. The dominant error is the FP4 grid itself —
eight magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6) — which no choice of scale
improves.

That is why the open work is **AWQ / GPTQ class**, not better scales: they do
not shrink the error, they move it. AWQ uses calibration activations to protect
the channels that carry the most signal; GPTQ compensates each column's error in
the columns still to be quantized. Both need infrastructure imp does not have
yet (activation statistics hooks / a Hessian pass).
