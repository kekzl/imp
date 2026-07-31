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

**A likely reason, stated as a hypothesis.** The Modelopt export also ships 280
`input_scale` and 40 `k_scale`/`v_scale` tensors: its weights were calibrated
*jointly with* activation and KV-cache quantization. imp runs the weight half of
that recipe. A weight rounding tuned for W4A4-plus-quantized-KV is not
necessarily the best weight rounding for W4A16, and a calibration set of general
web text is not this corpus of technical English. Confirming that would need an
A/B with the activation scales actually applied.

**What could not be measured:** the BF16 baseline for this model, and therefore
`--calib` on it. 27.5 GiB of weights plus the allocator's 5% headroom does not
fit in 32 GiB (upload fails at layer 39), and calibration needs a full forward
pass on the BF16 source. So the 14B row above is round-to-nearest only; the AWQ
numbers in the table further up are from the models that do fit.

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
